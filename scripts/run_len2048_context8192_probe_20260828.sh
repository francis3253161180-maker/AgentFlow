#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE_CONFIG="$REPO/train/config_5090_lora_smoke.yaml"
DATA=/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/len2048_probe.parquet
SELECTION=/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/len2048_probe_selection_manifest.json
SOURCE_SNAPSHOT=/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828/random30-fresh-rollout-replay-20260828_20260828_115632_behavior_snapshot.pt
TMP=/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP="${AGENTFLOW_LEN2048_EXPERIMENT:-reward-audit-len2048-context8192-20260828}"
MAX_RESPONSE="${AGENTFLOW_LEN2048_MAX_RESPONSE:-2048}"
CONFIG="${TMP}/${EXP}_${STAMP}.yaml"
TRAIN_LOG="${REPO}/log/${EXP}_${STAMP}_train.log"
ROLLOUT_LOG="${REPO}/log/${EXP}_${STAMP}_rollout.log"
GPU_LOG="${TMP}/${EXP}_${STAMP}_gpu.tsv"
ROUTE_STATE="${TMP}/${EXP}_${STAMP}_role_route.json"
SNAPSHOT="${TMP}/${EXP}_${STAMP}_behavior_snapshot.pt"
SNAPSHOT_META="${TMP}/${EXP}_${STAMP}_behavior_snapshot.json"
EVIDENCE_DIR="${TMP}/${EXP}_${STAMP}_trajectories"
LENGTH_AUDIT="${TMP}/${EXP}_${STAMP}_length_audit.json"
RUN_META="${TMP}/${EXP}_${STAMP}_run_meta.json"
VLLM_UTIL="${AGENTFLOW_LEN2048_VLLM_UTIL:-0.50}"
cd "$REPO"
source /root/.env
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export WANDB_MODE=disabled
export AGENTFLOW_TRAIN_CONFIG="$CONFIG"
export AGENTFLOW_DISABLE_EXTERNAL_LLM=1
export AGENTFLOW_UNIFIED_LOCAL_ROLES=1
export AGENTFLOW_REWARD_JUDGE_ENABLED=0
export AGENTFLOW_REWARD_SCORER_LOG=1
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1
export AGENTFLOW_ROLE_ROUTING_STATE="$ROUTE_STATE"
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base
export AGENTFLOW_UNIFIED_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export AGENTFLOW_UNIFIED_SMOKE_RUN_ID="$EXP"_"$STAMP"
export AGENTFLOW_UNIFIED_TEMPERATURE=0.7
export AGENTFLOW_UNIFIED_ROLLOUT_N=4
export AGENTFLOW_UNIFIED_SEED=20260829
export AGENTFLOW_UNIFIED_SCORER="local deterministic; external judge disabled"
export AGENTFLOW_UNIFIED_MAX_PROMPT_LENGTH=3072
export AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH="$MAX_RESPONSE"
export AGENTFLOW_UNIFIED_MAX_MODEL_LEN=8192
export AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1
export AGENTFLOW_RESPONSE_LENGTH_AUDIT_PATH="$LENGTH_AUDIT"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH="$SOURCE_SNAPSHOT"
export AGENTFLOW_STRUCTURED_OUTPUT_HARNESS="${AGENTFLOW_STRUCTURED_OUTPUT_HARNESS:-0}"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_PATH="$SNAPSHOT"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_METADATA_PATH="$SNAPSHOT_META"
export AGENTFLOW_REPLAY_CAPTURE_ENABLED=0
export AGENTFLOW_ROLLOUT_EVIDENCE_DIR="$EVIDENCE_DIR"
export AGENTFLOW_UNIFIED_CODE_COMMIT="$(git rev-parse HEAD)"
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS=1800
mkdir -p "$TMP" "$REPO/log" "$EVIDENCE_DIR"

"$PY" - "$SOURCE_SNAPSHOT" "$SELECTION" <<'PY'
import json, sys
from pathlib import Path
import torch
source, selection = map(Path, sys.argv[1:])
payload = torch.load(source, map_location="cpu", weights_only=False)
expected = "2f46d9002978cbbf623f28d5113a3d03634246a9332d308d768fc13b86ddf8c9"
if payload.get("lora_hash") != expected or payload.get("kind") != "agentflow_behavior_policy_snapshot":
    raise SystemExit("source behavior snapshot hash/kind verification failed")
manifest = json.loads(selection.read_text(encoding="utf-8"))
if manifest.get("status") != "selected_before_generation" or not (1 <= manifest.get("selected_group_count", 0) <= 8):
    raise SystemExit("selection manifest was not prepared before generation")
print("LEN2048_PREFLIGHT snapshot_hash_verified=1 selected_groups=%d" % manifest["selected_group_count"])
PY

"$PY" - "$BASE_CONFIG" "$CONFIG" "$DATA" "$EXP" "$VLLM_UTIL" "$MAX_RESPONSE" "$SELECTION" "$SOURCE_SNAPSHOT" "$RUN_META" "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$ROUTE_STATE" <<'PY'
from pathlib import Path
import hashlib, json, os, sys
base, out, data, exp, util, max_response, selection, source_snapshot, run_meta, train_log, rollout_log, gpu_log, route = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
base_data = "$" + "{BASE_DATA_DIR}"
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{exp}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'reward-audit-len2048-context8192'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '" + base_data + "/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '" + base_data + "/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.max_prompt_length: 1280": "data.max_prompt_length: 3072",
    "data.max_response_length: 384": f"data.max_response_length: {max_response}",
    "actor_rollout_ref.rollout.n: 2": "actor_rollout_ref.rollout.n: 4",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": f"actor_rollout_ref.rollout.gpu_memory_utilization: {util}",
    "actor_rollout_ref.rollout.max_model_len: 2048": "actor_rollout_ref.rollout.max_model_len: 8192",
    "actor_rollout_ref.rollout.max_num_batched_tokens: 2048": "actor_rollout_ref.rollout.max_num_batched_tokens: 1024",
    "actor_rollout_ref.rollout.max_num_seqs: 2": "actor_rollout_ref.rollout.max_num_seqs: 1",
    "agentflow.port: 9999": "agentflow.port: 9996",
    "trainer.val_before_train: False": "trainer.val_before_train: True\n  trainer.val_only: True",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing config anchor: {old}")
    text = text.replace(old, new, 1)
needle = "  actor_rollout_ref.rollout.n: 4\n"
if needle not in text:
    raise SystemExit("rollout.n replacement failed")
text = text.replace(needle, needle + "  actor_rollout_ref.rollout.temperature: 0.7\n", 1)
Path(out).write_text(text, encoding="utf-8")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
payload = {
    "schema_version": 1, "selection": selection, "selection_sha256": sha(selection),
    "source_behavior_snapshot": source_snapshot, "source_snapshot_sha256": sha(source_snapshot),
    "prepared_parquet": data, "prepared_parquet_sha256": sha(data), "config": out, "config_sha256": sha(out),
    "model_path": os.environ["AGENTFLOW_UNIFIED_MODEL_PATH"], "seed": 20260829,
    "sampling": {"temperature": 0.7, "rollout_n": 4, "max_prompt_length": 3072, "max_response_length": int(max_response), "max_model_len": 8192, "dynamic_response_padding": True},
    "vllm_gpu_memory_utilization": float(util), "rollout_only": True, "optimizer_steps": 0,
    "checkpoint_disabled": True, "external_calls_disabled": True, "code_commit": os.environ["AGENTFLOW_UNIFIED_CODE_COMMIT"],
    "artifacts": {"run_meta": run_meta, "train_log": train_log, "rollout_log": rollout_log, "gpu_log": gpu_log, "route_state": route},
}
Path(run_meta).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

TRAIN_PID=""; ROLLOUT_PID=""; MONITOR_PID=""
cleanup() {
  status=$?
  for pid in "$ROLLOUT_PID" "$TRAIN_PID" "$MONITOR_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "$ROLLOUT_PID" "$TRAIN_PID" "$MONITOR_PID"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  "$PY" -m ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM
check_abort() {
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    [[ -f "$path" ]] || continue
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|worker died|No valid (training|validation) rollout|BadRequestError|HTTP/1.1" 5[0-9][0-9]|structured output backend' "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}
echo "LEN2048_PROTOCOL model=Qwen2.5-7B-Instruct temp=0.7 n=4 max_prompt=3072 max_response=$MAX_RESPONSE max_model_len=8192 seed=20260829 structured_harness=$AGENTFLOW_STRUCTURED_OUTPUT_HARNESS"
echo "LEN2048_MEMORY vllm_gpu_memory_utilization=$VLLM_UTIL max_num_seqs=1 max_num_batched_tokens=1024"
echo "LEN2048_MODE rollout_only=1 val_only=1 optimizer_steps=0 checkpoint=disabled external_calls=disabled"
echo "LEN2048_SNAPSHOT source=$SOURCE_SNAPSHOT output=$SNAPSHOT"
echo "LEN2048_SELECTION=$SELECTION"
PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.max_prompt_length=3072 data.max_response_length="$MAX_RESPONSE" +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.offload_policy=true >"$TRAIN_LOG" 2>&1 & TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
    sleep 5
  done
) & MONITOR_PID=$!
ready=0
for _ in $(seq 1 360); do
  check_abort
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "ABORT_CONDITION timed_out_waiting_for_tasks" >&2; exit 2; fi
if ! grep -q 'status=restored' "$TRAIN_LOG"; then echo "ABORT_CONDITION behavior_snapshot_not_restored" >&2; exit 2; fi
PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 & ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort; sleep 5; done
wait "$TRAIN_PID"
check_abort
if ! grep -q 'Validation summary:' "$TRAIN_LOG"; then echo "ABORT_CONDITION missing_validation_summary" >&2; exit 2; fi
if grep -Eqi 'Training data keys|optimizer\.step|backward\(|global_step: [1-9]|Training Progress' "$TRAIN_LOG"; then echo "ABORT_CONDITION unexpected_training_marker" >&2; exit 2; fi
for _ in $(seq 1 180); do [[ ! -e "/proc/$ROLLOUT_PID" ]] && break; sleep 1; done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; fi
wait "$ROLLOUT_PID" 2>/dev/null || true
ROLLOUT_PID=""
"$PY" scripts/aggregate_len2048_probe_20260828.py --evidence-dir "$EVIDENCE_DIR" --selection "$SELECTION" --output "$TMP/"$EXP"_"$STAMP"_aggregate.json" --max-response "$MAX_RESPONSE" --max-model-len 8192
echo "LEN2048_STATUS=passed"
