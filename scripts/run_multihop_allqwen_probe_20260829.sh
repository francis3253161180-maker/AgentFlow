#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && "$1" =~ ^(musique|2wiki)$ ]] || { echo "usage: $0 musique|2wiki" >&2; exit 2; }
DATASET="$1"
REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE="$REPO/train/config_5090_lora_smoke.yaml"
TMP=/root/autodl-tmp/tmp/multihop_allqwen_probe_20260829
DATA="$TMP/${DATASET}.parquet"
CONFIG="$TMP/${DATASET}.yaml"
EXP="multihop-allqwen7b-${DATASET}-20260829"
TRAIN_LOG="$REPO/log/20260829_multihop_allqwen7b_${DATASET}_train.log"
ROLLOUT_LOG="$REPO/log/20260829_multihop_allqwen7b_${DATASET}_rollout.log"
GPU_LOG="$TMP/${DATASET}_gpu.tsv"
ROUTE="$TMP/${DATASET}.role_route.json"
VLLM_UTIL="${AGENTFLOW_MULTIHOP_VLLM_UTIL:-0.60}"
cd "$REPO"
source /root/.env
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache TMPDIR=/root/autodl-tmp/tmp RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export WANDB_MODE=disabled AGENTFLOW_TRAIN_CONFIG="$CONFIG"
export AGENTFLOW_DISABLE_EXTERNAL_LLM=1 AGENTFLOW_UNIFIED_LOCAL_ROLES=1 AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE=
export AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE=0.0 AGENTFLOW_REWARD_JUDGE_ENABLED=0 AGENTFLOW_REWARD_SCORER_LOG=1
export AGENTFLOW_UNIFIED_MEMORY_LOG=1
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1 AGENTFLOW_ROLE_ROUTING_STATE="$ROUTE"
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base AGENTFLOW_UNIFIED_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export AGENTFLOW_UNIFIED_SMOKE_RUN_ID="${EXP}_$(date +%Y%m%d_%H%M%S)" AGENTFLOW_UNIFIED_TEMPERATURE=0.7
export AGENTFLOW_UNIFIED_ROLLOUT_N=4 AGENTFLOW_UNIFIED_SEED=20260829
export AGENTFLOW_UNIFIED_SCORER="current deterministic scorer; external judge disabled"
export AGENTFLOW_UNIFIED_MAX_PROMPT_LENGTH=1536 AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH=1024 AGENTFLOW_UNIFIED_MAX_MODEL_LEN=4096
export AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1 AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30 AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS=3600
mkdir -p "$TMP" "$REPO/log"

"$PY" - "$BASE" "$CONFIG" "$DATA" "$EXP" "$VLLM_UTIL" <<'PY'
from pathlib import Path
import sys
base, out, data, exp, util = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{exp}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'multihop-allqwen7b-probe'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.max_prompt_length: 1280": "data.max_prompt_length: 1536",
    "data.max_response_length: 384": "data.max_response_length: 1024",
    "actor_rollout_ref.rollout.n: 2": "actor_rollout_ref.rollout.n: 4",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": f"actor_rollout_ref.rollout.gpu_memory_utilization: {util}",
    "actor_rollout_ref.rollout.max_model_len: 2048": "actor_rollout_ref.rollout.max_model_len: 4096",
    "actor_rollout_ref.rollout.max_num_batched_tokens: 2048": "actor_rollout_ref.rollout.max_num_batched_tokens: 1024",
    "actor_rollout_ref.rollout.max_num_seqs: 2": "actor_rollout_ref.rollout.max_num_seqs: 1",
    "agentflow.port: 9999": "agentflow.port: 9994",
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
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|worker died|No valid (training|validation) rollout|HTTP/[^ ]+ 5[0-9][0-9]|status[_ ]?code[=: ]+5[0-9][0-9]' "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}
: > "$TRAIN_LOG"
: > "$ROLLOUT_LOG"
: > "$GPU_LOG"
echo "MULTIHOP_ALLQWEN_PROTOCOL dataset=$DATASET model=Qwen2.5-7B-Instruct planner=qwen-actor-lora fixed=qwen-base-frozen temp=0.7 fixed_temp=0.0 n=4 prompts=10 rollout_only=1 optimizer_steps=0 checkpoint=disabled external_calls=0"
PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.max_prompt_length=1536 data.max_response_length=1024 +actor_rollout_ref.ref.model.path=/root/autodl-tmp/models/Qwen2.5-7B-Instruct critic.model.path=/root/autodl-tmp/models/Qwen2.5-7B-Instruct +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.offload_policy=true >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
    sleep 5
  done
) & MONITOR_PID=$!
ready=0
for _ in $(seq 1 360); do
  check_abort || exit 2
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "ABORT_CONDITION timed_out_waiting_for_tasks" >&2; exit 2; fi
PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 & ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort; sleep 5; done
wait "$TRAIN_PID"
check_abort
grep -q 'Validation summary:' "$TRAIN_LOG" || { echo "ABORT_CONDITION missing_validation_summary" >&2; exit 2; }
if grep -Eqi 'Training data keys|optimizer\.step|backward\(|global_step: [1-9]|Training Progress' "$TRAIN_LOG"; then
  echo "ABORT_CONDITION unexpected_training_marker" >&2
  exit 2
fi
for _ in $(seq 1 180); do [[ ! -e "/proc/$ROLLOUT_PID" ]] && break; sleep 1; done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; fi
wait "$ROLLOUT_PID" 2>/dev/null || true
ROLLOUT_PID=""
ROLLOUT_DIR=$(find "$REPO/rollout_data" -type d -path "*/${EXP}_*/Qwen2.5-7B-Instruct_*/train" -print | sort | tail -1)
[[ -n "$ROLLOUT_DIR" ]] || { echo "ABORT_CONDITION missing_rollout_data_directory" >&2; exit 2; }
echo "MULTIHOP_ALLQWEN_COMPLETED dataset=$DATASET rollout_dir=$ROLLOUT_DIR"
