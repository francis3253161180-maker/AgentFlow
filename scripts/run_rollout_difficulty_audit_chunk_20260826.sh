#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 CHUNK (0, 1, 2, or 3)" >&2
  exit 2
fi
CHUNK="$1"
REPO=/root/autodl-tmp/AgentFlow
BASE_CONFIG="$REPO/train/config_5090_lora_mini20.yaml"
MANIFEST="$REPO/log/2026-08-26_rollout_difficulty_audit_sample_manifest.json"
SOURCE_PARQUET=/root/autodl-tmp/tmp/rollout_difficulty_audit_20260826/prompts_100.parquet
AUDIT_TMP=/root/autodl-tmp/tmp/rollout_difficulty_audit_complete_20260826
CHUNK_PARQUET="$AUDIT_TMP/prompts_chunk_$CHUNK.parquet"
CONFIG="$AUDIT_TMP/config_chunk_$CHUNK.yaml"
META="$AUDIT_TMP/chunk_$CHUNK.meta.json"
EXP_NAME="rollout-difficulty-100-20260826-chunk$CHUNK"
TRAIN_LOG="$REPO/log/20260826_rollout_difficulty_audit_complete_chunk$CHUNK""_train.log"
ROLLOUT_LOG="$REPO/log/20260826_rollout_difficulty_audit_complete_chunk$CHUNK""_rollout.log"
GPU_LOG="$AUDIT_TMP/chunk_$CHUNK""_gpu.tsv"
cd "$REPO"
source /root/.env
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export WANDB_MODE=disabled
export AGENTFLOW_REWARD_JUDGE_ENABLED=1
export AGENTFLOW_REWARD_SCORER_LOG=1
export AGENTFLOW_REWARD_JUDGE_CACHE_DIR=/root/autodl-tmp/tmp/reward_judge_20260826_difficulty_complete
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
mkdir -p "$AUDIT_TMP" "$REPO/log"

/root/autodl-tmp/conda/envs/agentflow/bin/python - "$CHUNK" "$MANIFEST" "$SOURCE_PARQUET" "$CHUNK_PARQUET" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

chunk = int(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
source = Path(sys.argv[3])
target = Path(sys.argv[4])
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != manifest["selected_parquet_sha256"]:
    raise SystemExit("fixed parquet sha mismatch")
if manifest["selected_count"] != 100 or len(manifest["rows"]) != 100:
    raise SystemExit("manifest is not the expected 100-row fixed sample")
frame = pd.read_parquet(source)
if len(frame) != 100:
    raise SystemExit("fixed parquet row count is not 100")

for order, expected in enumerate(manifest["rows"]):
    row = frame.iloc[order]
    value = row["extra_info"]
    info = value if isinstance(value, dict) else json.loads(value)
    observed = {"source": str(row["source"]), "idx": int(info["idx"]), "id": int(row["id"])}
    wanted = {key: expected[key] for key in ("source", "idx", "id")}
    if observed != wanted:
        raise SystemExit(f"manifest mismatch at order {order}")

start = chunk * 25
selected = frame.iloc[start:start + 25].reset_index(drop=True)
target.parent.mkdir(parents=True, exist_ok=True)
selected.to_parquet(target, index=False)
print(json.dumps({"chunk": chunk, "start": start, "end": start + 25,
                  "rows": len(selected), "sources": selected["source"].value_counts().to_dict(),
                  "source_sha256": source_sha,
                  "chunk_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, sort_keys=True))
PY

/root/autodl-tmp/conda/envs/agentflow/bin/python - "$BASE_CONFIG" "$CONFIG" "$CHUNK_PARQUET" "$EXP_NAME" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
chunk_parquet = sys.argv[3]
exp_name = sys.argv[4]
text = base.replace("EXPERIMENT_NAME: 'qwen25-3b-lora-mini20-seed20260825'",
                    f"EXPERIMENT_NAME: '{exp_name}'")
text = text.replace("data.val_files: '" + "$" + "{BASE_DATA_DIR}" + "/val/aime24.parquet'",
                    f"data.val_files: '{chunk_parquet}'")
text = text.replace("actor_rollout_ref.rollout.n: 2", "actor_rollout_ref.rollout.n: 4")
needle = "  actor_rollout_ref.rollout.n: 4\n"
if needle not in text:
    raise SystemExit("failed to set rollout.n=4")
text = text.replace(needle, needle + "  actor_rollout_ref.rollout.temperature: 0.7\n", 1)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
PY

rm -f "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG"
export AGENTFLOW_TRAIN_CONFIG="$CONFIG"
TRAIN_PID=""
ROLLOUT_PID=""
MONITOR_PID=""

cleanup() {
  status=$?
  for pid in "$ROLLOUT_PID" "$TRAIN_PID" "$MONITOR_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

check_abort_conditions() {
  local path
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    [ -f "$path" ] || continue
    if grep -Eqi "CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained.*False|RayTaskError|deadlock|No valid rollout|worker died|Training data keys|Training Progress|optimizer\\.step|backward\\(" "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}

echo "AUDIT_CHUNK=$CHUNK ORDERS=$((CHUNK * 25))..$((CHUNK * 25 + 24))"
echo "AUDIT_CONFIG=Qwen2.5-3B-Instruct LoRA rank8 alpha16 temp=0.7 rollout_n=4"
echo "AUDIT_MANIFEST=$MANIFEST"
echo "AUDIT_PROMPT_DATA=$CHUNK_PARQUET"
echo "TRAIN_COMMAND=train_agent.py val_only=true rollout_n=4 temperature=0.7"

PYTHONUNBUFFERED=1 /root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py \
  --config "$CONFIG" \
  trainer.val_only=true trainer.val_before_train=true \
  trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP_NAME" \
  actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 \
  data.val_files="$CHUNK_PARQUET" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

(while kill -0 "$TRAIN_PID" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
  sleep 5
done) &
MONITOR_PID=$!

ready=0
for _ in $(seq 1 240); do
  check_abort_conditions || exit 2
  if grep -qE "Total tasks queued:|Task queued:" "$TRAIN_LOG" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID"
    exit $?
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo "ABORT_CONDITION timed_out_waiting_for_rollout_tasks" >&2
  exit 2
fi

PYTHONUNBUFFERED=1 /root/autodl-tmp/conda/envs/agentflow/bin/python train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  check_abort_conditions || exit 2
  sleep 5
done
wait "$TRAIN_PID"
check_abort_conditions

grep -q "Validation summary: 100/100 total rollouts (100.0%), 100 valid rollouts" "$TRAIN_LOG" || {
  echo "ABORT_CONDITION incomplete_chunk_summary" >&2
  exit 2
}
if grep -Eqi "Training data keys|optimizer\\.step|backward\\(|global_step: [1-9]" "$TRAIN_LOG"; then
  echo "ABORT_CONDITION unexpected_training_execution_marker" >&2
  exit 2
fi

for _ in $(seq 1 90); do
  kill -0 "$ROLLOUT_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then
  echo "rollout launcher did not exit; terminating launcher only" >&2
  kill -TERM "$ROLLOUT_PID" 2>/dev/null || true
  wait "$ROLLOUT_PID" 2>/dev/null || true
fi
ROLLOUT_PID=""

TRAIN_ROLLOUT_DIR=$(find "$REPO/rollout_data" -type d -path "*/$EXP_NAME""_*/Qwen2.5-3B-Instruct_*/train" -print | sort | tail -1)
if [ -z "$TRAIN_ROLLOUT_DIR" ]; then
  echo "ABORT_CONDITION missing_rollout_data_directory" >&2
  exit 2
fi

/root/autodl-tmp/conda/envs/agentflow/bin/python - "$META" "$CHUNK" "$EXP_NAME" "$CHUNK_PARQUET" "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$TRAIN_ROLLOUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
meta = {"chunk": int(sys.argv[2]), "experiment_name": sys.argv[3],
        "prompt_parquet": sys.argv[4], "train_log": sys.argv[5],
        "rollout_log": sys.argv[6], "gpu_log": sys.argv[7],
        "train_rollout_dir": sys.argv[8]}
Path(sys.argv[1]).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY
echo "ROLLOUT_DIFFICULTY_CHUNK_COMPLETED=1 chunk=$CHUNK"
