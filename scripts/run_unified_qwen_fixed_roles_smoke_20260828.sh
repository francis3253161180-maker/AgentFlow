#!/usr/bin/env bash
set -euo pipefail

# Architecture smoke only: one local Qwen vLLM engine, no paid provider and
# no validation/checkpoint. The size is explicit so a 7B failure is preserved.
SIZE="${1:-7b}"
case "$SIZE" in
  7b) MODEL="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"; TAG="qwen7b" ;;
  3b) MODEL="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"; TAG="qwen3b" ;;
  *) echo "usage: $0 [7b|3b]" >&2; exit 2 ;;
esac

REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
TMP=/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828
DATA="${AGENTFLOW_SMOKE_DATA:-/root/autodl-tmp/tmp/gameof24_xot_baseline_20260827/gameof24_xot_smoke_4.parquet}"
VLLM_UTIL="${AGENTFLOW_SMOKE_VLLM_GPU_UTIL:-0.60}"
FSDP2_OFFLOAD_POLICY="${AGENTFLOW_SMOKE_FSDP2_OFFLOAD_POLICY:-true}"
SMOKE_N="${AGENTFLOW_SMOKE_N:-2}"
SMOKE_MAX_RESPONSE="${AGENTFLOW_SMOKE_MAX_RESPONSE:-64}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP="unified-${TAG}-fixed-roles-smoke-20260828"
CONFIG="$TMP/${EXP}_${STAMP}.yaml"
TRAIN_LOG="$REPO/log/${EXP}_${STAMP}_train.log"
ROLLOUT_LOG="$REPO/log/${EXP}_${STAMP}_rollout.log"
GPU_LOG="$TMP/${EXP}_${STAMP}_gpu.tsv"
ROLE_ROUTE_STATE="$TMP/${EXP}_${STAMP}_role_routes.json"

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
export AGENTFLOW_UNIFIED_MEMORY_LOG=1
export AGENTFLOW_ROLE_ROUTING_STATE="$ROLE_ROUTE_STATE"
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME="qwen-base"
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS=900
mkdir -p "$TMP" "$REPO/log"

"$PY" - "$REPO/train/config_5090_lora_smoke.yaml" "$CONFIG" "$MODEL" "$DATA" "$EXP" "$VLLM_UTIL" "$SMOKE_N" "$SMOKE_MAX_RESPONSE" <<'PY'
from pathlib import Path
import sys

base_path, out_path, model, data, experiment, vllm_util, smoke_n, max_response = sys.argv[1:]
text = Path(base_path).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": f"BASE_MODEL: '{model}'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{experiment}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'unified-qwen-fixed-roles-smoke'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.max_response_length: 384": f"data.max_response_length: {max_response}",
    "actor_rollout_ref.rollout.n: 2": f"actor_rollout_ref.rollout.n: {smoke_n}",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": f"actor_rollout_ref.rollout.gpu_memory_utilization: {vllm_util}",
    "actor_rollout_ref.rollout.max_num_batched_tokens: 2048": "actor_rollout_ref.rollout.max_num_batched_tokens: 1024",
    "actor_rollout_ref.rollout.max_num_seqs: 2": "actor_rollout_ref.rollout.max_num_seqs: 1",
    "trainer.test_freq: 1000": "trainer.test_freq: 0",
    "agentflow.port: 9999": "agentflow.port: 9998",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing config anchor: {old}")
    text = text.replace(old, new, 1)
Path(out_path).write_text(text, encoding="utf-8")
PY

TRAIN_PID=""
ROLLOUT_PID=""
MONITOR_PID=""
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
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|worker died|No valid (training|validation) rollout' "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
}

echo "UNIFIED_QWEN_SMOKE size=$SIZE model=$MODEL"
echo "UNIFIED_QWEN_PROTOCOL lora_rank=8 lora_alpha=16 target_modules=all-linear temp=0.7 rollout_n=$SMOKE_N max_response_length=$SMOKE_MAX_RESPONSE train_prompts=4 train_batch=2 ppo_mini_batch=2 micro_batch=1 ppo_epochs=1 fsdp_model_dtype=bf16 fsdp2_offload_policy=$FSDP2_OFFLOAD_POLICY save_freq=0 external_llm=disabled"
echo "UNIFIED_QWEN_ROLES planner_main=trainable_actor_lora planner_fixed=frozen_base_no_lora verifier=frozen_base_no_lora executor=frozen_base_no_lora tool=frozen_base_no_lora"
echo "UNIFIED_QWEN_MEMORY vllm_gpu_memory_utilization=$VLLM_UTIL max_num_seqs=1 max_num_batched_tokens=1024"
echo "UNIFIED_QWEN_CONFIG=$CONFIG"
echo "UNIFIED_QWEN_ROLE_ROUTE_STATE=$ROLE_ROUTE_STATE"
echo "UNIFIED_QWEN_TRAIN_LOG=$TRAIN_LOG"
echo "UNIFIED_QWEN_ROLLOUT_LOG=$ROLLOUT_LOG"

PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" \
  trainer.val_before_train=false trainer.val_only=false trainer.test_freq=0 trainer.save_freq=0 \
  trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" \
  actor_rollout_ref.rollout.n="$SMOKE_N" actor_rollout_ref.rollout.temperature=0.7 data.max_response_length="$SMOKE_MAX_RESPONSE" \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$FSDP2_OFFLOAD_POLICY" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
  actor_rollout_ref.rollout.max_num_seqs=1 actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
    sleep 3
  done
) &
MONITOR_PID=$!

ready=0
for _ in $(seq 1 300); do
  check_abort
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "ABORT_CONDITION timed_out_waiting_for_tasks" >&2; exit 2; fi

PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  check_abort
  sleep 3
done
wait "$TRAIN_PID"
check_abort
if ! grep -q 'Training finished at step' "$TRAIN_LOG"; then
  echo "ABORT_CONDITION missing_training_finished" >&2
  exit 2
fi
echo "UNIFIED_QWEN_STATUS=passed"
