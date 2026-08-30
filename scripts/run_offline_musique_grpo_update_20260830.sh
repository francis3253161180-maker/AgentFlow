#!/usr/bin/env bash
set -euo pipefail

# One bounded, terminal-only GRPO pass over the persisted 128x8 MuSiQue pack.
# This path performs no rollout, retrieval, scoring, or external model call.
REPO=/root/autodl-tmp/AgentFlow-offline-musique
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
RAY=/root/autodl-tmp/conda/envs/agentflow/bin/ray
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
ADAPTER=/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/direct_vllm/qwen-actor-lora
ROOT=/root/autodl-tmp/offline_musique_grpo_20260830
RUN_TAG="${AGENTFLOW_OFFLINE_REPLAY_RUN_TAG:-terminal_a}"
PACK="${AGENTFLOW_OFFLINE_REPLAY_PACK_PATH:-$ROOT/train_replay_pack.pt}"
SNAPSHOT=/root/autodl-tmp/tmp/gameof24_planner_temp0_causal_sanity_20260829/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_behavior_snapshot.pt
DATA=/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/mixed_signal_smoke_4.parquet
LOG="$ROOT/grpo_${RUN_TAG}.log"
GPU_LOG="$ROOT/grpo_${RUN_TAG}_gpu.csv"
CHECKSUM="$ROOT/grpo_${RUN_TAG}_lora_checksum.json"
POST_SNAPSHOT="$ROOT/grpo_${RUN_TAG}_post_lora_snapshot.pt"
KL_AUDIT="$ROOT/grpo_${RUN_TAG}_kl_audit.json"
EXP="offline-musique-${RUN_TAG}-n8-20260830"

for required in "$MODEL" "$ADAPTER" "$PACK" "$SNAPSHOT" "$DATA"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

cd "$REPO"
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
export AGENTFLOW_UNIFIED_ROLLOUT_N=8
export AGENTFLOW_UNIFIED_PPO_EPOCHS=1
export AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH=256
export AGENTFLOW_UNIFIED_SEED=20260830
export AGENTFLOW_UNIFIED_SCORER='outcome_v2 terminal only; F1/F2/delta weight=0; external disabled'
export AGENTFLOW_OFFLINE_REPLAY_PACK_PATH="$PACK"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH="$SNAPSHOT"
export AGENTFLOW_LORA_CHECKSUM_ENABLED=1
export AGENTFLOW_LORA_CHECKSUM_PATH="$CHECKSUM"
export AGENTFLOW_LORA_POST_SNAPSHOT_PATH="$POST_SNAPSHOT"
export AGENTFLOW_REPLAY_CAPTURE_ENABLED=0
export AGENTFLOW_OFFLINE_REPLAY_KL_AUDIT_PATH="$KL_AUDIT"
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05

mkdir -p "$ROOT"
: >"$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >>"$GPU_LOG" || true
    sleep 10
  done
) &
monitor_pid=$!

cleanup() {
  status=$?
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
  "$RAY" stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "AGENTFLOW_MUSIQUE_GRPO_PROTOCOL groups=128 rollouts_per_group=8 trajectories=1024 transitions=6007 ppo_epochs=1 ppo_minibatch=6007 dynamic_token_batch=16384 reward=outcome_v2 diagnostic_weight=0 rollout_requests=0 external_calls=0" | tee "$LOG"

PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config train/config_5090_lora_smoke.yaml \
  trainer.val_before_train=false trainer.val_only=false trainer.test_freq=0 trainer.save_freq=0 \
  trainer.total_epochs=1 trainer.experiment_name="$EXP" \
  data.train_files="$DATA" data.val_files="$DATA" data.train_batch_size=4 \
  data.max_prompt_length=1536 data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.ppo_mini_batch_size=6007 \
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true \
  >>"$LOG" 2>&1

grep -q 'Training finished at step' "$LOG"
if [[ "${AGENTFLOW_OFFLINE_REPLAY_AUDIT_ONLY:-0}" == "1" ]]; then
  test -s "$KL_AUDIT"
  grep -q 'AGENTFLOW_OFFLINE_REPLAY_KL_AUDIT' "$LOG"
  echo 'AGENTFLOW_MUSIQUE_GRPO_STATUS=kl_audit_completed' | tee -a "$LOG"
else
  grep -q 'AGENTFLOW_OFFLINE_REPLAY_UPDATE' "$LOG"
  grep -q 'AGENTFLOW_OFFLINE_REPLAY_METRICS' "$LOG"
  grep -q 'UNIFIED_LORA_CHECKSUM stage=post' "$LOG"
  test -s "$CHECKSUM"
  test -s "$POST_SNAPSHOT"
  echo 'AGENTFLOW_MUSIQUE_GRPO_STATUS=completed' | tee -a "$LOG"
fi
