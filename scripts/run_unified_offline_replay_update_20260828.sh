#!/usr/bin/env bash
set -euo pipefail

# Fresh-process diagnostic only. It loads an authentic runtime DataProto and
# calls the real actor update; it never queues AgentFlow tasks or calls rollout.
MODEL="${AGENTFLOW_UNIFIED_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
PACK="${AGENTFLOW_OFFLINE_REPLAY_PACK_PATH:?set AGENTFLOW_OFFLINE_REPLAY_PACK_PATH}"
DATA="${AGENTFLOW_OFFLINE_REPLAY_DATA:-/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/mixed_signal_smoke_4.parquet}"
VLLM_UTIL="${AGENTFLOW_OFFLINE_VLLM_GPU_UTIL:-0.10}"
FSDP2_OFFLOAD_POLICY="${AGENTFLOW_OFFLINE_FSDP2_OFFLOAD_POLICY:-true}"
REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
TMP=/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP="unified-qwen7b-offline-replay-${STAMP}"
CONFIG="$TMP/${EXP}.yaml"
LOG="$REPO/log/${EXP}.log"
CHECKSUM="$TMP/${EXP}_lora_checksum.json"

cd "$REPO"
source /root/.env
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export WANDB_MODE=disabled
export AGENTFLOW_DISABLE_EXTERNAL_LLM=1
export AGENTFLOW_UNIFIED_LOCAL_ROLES=1
export AGENTFLOW_REWARD_JUDGE_ENABLED=0
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base
export AGENTFLOW_UNIFIED_MODEL_PATH="$MODEL"
export AGENTFLOW_UNIFIED_SMOKE_RUN_ID="$EXP"
export AGENTFLOW_UNIFIED_TEMPERATURE=0.7
export AGENTFLOW_UNIFIED_ROLLOUT_N=4
export AGENTFLOW_UNIFIED_PPO_EPOCHS=2
export AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH=512
export AGENTFLOW_UNIFIED_SCORER="local deterministic only; external disabled"
export AGENTFLOW_OFFLINE_REPLAY_PACK_PATH="$PACK"
export AGENTFLOW_LORA_CHECKSUM_ENABLED=1
export AGENTFLOW_LORA_CHECKSUM_PATH="$CHECKSUM"
export AGENTFLOW_REPLAY_CAPTURE_ENABLED=0
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
mkdir -p "$TMP" "$REPO/log"

"$PY" - "$REPO/train/config_5090_lora_smoke.yaml" "$CONFIG" "$MODEL" "$DATA" "$EXP" "$VLLM_UTIL" <<'PY'
from pathlib import Path
import sys

base_path, out_path, model, data, experiment, vllm_util = sys.argv[1:]
text = Path(base_path).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": f"BASE_MODEL: '{model}'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{experiment}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'unified-qwen-offline-replay'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.train_batch_size: 2": "data.train_batch_size: 4",
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

cleanup() {
  status=$?
  "$PY" -m ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "AGENTFLOW_OFFLINE_REPLAY_CONFIG=$CONFIG"
echo "AGENTFLOW_OFFLINE_REPLAY_PACK=$PACK"
echo "AGENTFLOW_OFFLINE_REPLAY_LOG=$LOG"
echo "AGENTFLOW_OFFLINE_REPLAY_CHECKSUM=$CHECKSUM"
echo "AGENTFLOW_OFFLINE_REPLAY_PROTOCOL model=$MODEL ppo_epochs=2 trainer_total_epochs=1 rollout_requests=0 external_calls=0"

PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" \
  trainer.val_before_train=false trainer.val_only=false trainer.test_freq=0 trainer.save_freq=0 \
  trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_epochs=2 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 \
  data.max_response_length=512 actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
  actor_rollout_ref.rollout.max_num_seqs=1 actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$FSDP2_OFFLOAD_POLICY" \
  >"$LOG" 2>&1

grep -Eq 'AGENTFLOW_OFFLINE_REPLAY_UPDATE|AGENTFLOW_OFFLINE_REPLAY_METRICS' "$LOG"
grep -q 'Training finished at step' "$LOG"
grep -q 'UNIFIED_LORA_CHECKSUM stage=post' "$LOG"
echo "AGENTFLOW_OFFLINE_REPLAY_STATUS=passed"
