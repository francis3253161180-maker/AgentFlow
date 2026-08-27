#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^(2wiki|hotpotqa|musique|gaia|aime24|gameof24|gpqa|medqa)$ ]]; then
  echo "usage: $0 DATASET (2wiki, hotpotqa, musique, gaia, aime24, gameof24, gpqa, or medqa)" >&2
  exit 2
fi
DATASET="$1"
REPO=/root/autodl-tmp/AgentFlow
ENV_PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE_CONFIG="$REPO/train/config_5090_lora_mini20.yaml"
PROBE_TMP=/root/autodl-tmp/tmp/remaining_benchmark_screen_20260827
PARQUET="$PROBE_TMP/${DATASET}.parquet"
CONFIG="$PROBE_TMP/${DATASET}.yaml"
RUN_TAG="${AGENTFLOW_PROBE_RUN_TAG:-base}"
if ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "AGENTFLOW_PROBE_RUN_TAG must contain only letters, digits, underscore, or hyphen" >&2
  exit 2
fi
EXP_NAME="benchmark-difficulty-${DATASET}-20260827-${RUN_TAG}"
TRAIN_LOG="$REPO/log/20260827_benchmark_difficulty_${DATASET}_${RUN_TAG}_train.log"
ROLLOUT_LOG="$REPO/log/20260827_benchmark_difficulty_${DATASET}_${RUN_TAG}_rollout.log"
GPU_LOG="$PROBE_TMP/${DATASET}_gpu.tsv"
EXPECTED_PROMPTS="${AGENTFLOW_PROBE_EXPECTED_PROMPTS:-10}"
EXPECTED_ROLLOUTS=$((EXPECTED_PROMPTS * 4))

if ! [[ "$EXPECTED_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "AGENTFLOW_PROBE_EXPECTED_PROMPTS must be a positive integer" >&2
  exit 2
fi

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
export AGENTFLOW_REWARD_JUDGE_CACHE_DIR=/root/autodl-tmp/tmp/reward_judge_20260827_benchmark_probe
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
mkdir -p "$PROBE_TMP" "$REPO/log"

"$ENV_PY" - "$BASE_CONFIG" "$CONFIG" "$PARQUET" "$EXP_NAME" <<'PY'
from pathlib import Path
import sys
base = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
parquet = sys.argv[3]
experiment = sys.argv[4]
text = base.replace("EXPERIMENT_NAME: 'qwen25-3b-lora-mini20-seed20260825'", f"EXPERIMENT_NAME: '{experiment}'")
text = text.replace("data.val_files: '" + "$" + "{BASE_DATA_DIR}/val/aime24.parquet'", f"data.val_files: '{parquet}'")
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
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
  "$ENV_PY" -m ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM
check_abort_conditions() {
  local path
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    [ -f "$path" ] || continue
    if grep -Eqi "CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained.*False|RayTaskError|deadlock|No valid rollout|worker died|Training data keys|optimizer\.step|backward\(" "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}

echo "BENCHMARK_PROBE_DATASET=$DATASET"
echo "BENCHMARK_PROBE_CONFIG=Qwen2.5-3B-Instruct LoRA rank8 alpha16 temp=0.7 rollout_n=4 rollout_only=1"
echo "BENCHMARK_PROBE_PARQUET=$PARQUET"
echo "BENCHMARK_PROBE_EXPECTED_PROMPTS=$EXPECTED_PROMPTS EXPECTED_ROLLOUTS=$EXPECTED_ROLLOUTS"
echo "BENCHMARK_PROBE_COMMAND=train_agent.py val_only=true save_freq=0 optimizer_steps=0"
PYTHONUNBUFFERED=1 "$ENV_PY" train/train_agent.py --config "$CONFIG" trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP_NAME" actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.val_files="$PARQUET" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true; sleep 5; done
) &
MONITOR_PID=$!
ready=0
for _ in $(seq 1 240); do
  check_abort_conditions || exit 2
  if grep -qE "Total tasks queued:|Task queued:" "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then echo "ABORT_CONDITION timed_out_waiting_for_rollout_tasks" >&2; exit 2; fi
PYTHONUNBUFFERED=1 "$ENV_PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort_conditions || exit 2; sleep 5; done
wait "$TRAIN_PID"
check_abort_conditions
if ! grep -q "Validation summary:" "$TRAIN_LOG"; then
  echo "ABORT_CONDITION missing_probe_summary" >&2
  exit 2
fi
if ! grep -q "Validation summary: ${EXPECTED_ROLLOUTS}/${EXPECTED_ROLLOUTS} total rollouts (100.0%), ${EXPECTED_ROLLOUTS} valid rollouts" "$TRAIN_LOG"; then
  echo "BENCHMARK_PROBE_PARTIAL_VALIDITY summary_differs_from_expected=1" >&2
fi
if grep -Eqi "Training data keys|optimizer\.step|backward\(|global_step: [1-9]" "$TRAIN_LOG"; then echo "ABORT_CONDITION unexpected_training_execution_marker" >&2; exit 2; fi
for _ in $(seq 1 90); do if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then break; fi; sleep 1; done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; wait "$ROLLOUT_PID" 2>/dev/null || true; fi
ROLLOUT_PID=""
TRAIN_ROLLOUT_DIR=$(find "$REPO/rollout_data" -type d -path "*/$EXP_NAME""_*/Qwen2.5-3B-Instruct_*/train" -print | sort | tail -1)
if [ -z "$TRAIN_ROLLOUT_DIR" ]; then echo "ABORT_CONDITION missing_rollout_data_directory" >&2; exit 2; fi
"$ENV_PY" - "$PROBE_TMP/${DATASET}.meta.json" "$DATASET" "$EXP_NAME" "$PARQUET" "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$TRAIN_ROLLOUT_DIR" "$EXPECTED_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path
meta = {"dataset": sys.argv[2], "experiment_name": sys.argv[3], "prompt_parquet": sys.argv[4], "train_log": sys.argv[5], "rollout_log": sys.argv[6], "gpu_log": sys.argv[7], "train_rollout_dir": sys.argv[8], "expected_prompts": int(sys.argv[9]), "expected_rollouts": int(sys.argv[9]) * 4}
Path(sys.argv[1]).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY
echo "BENCHMARK_PROBE_COMPLETED=1 dataset=$DATASET"
