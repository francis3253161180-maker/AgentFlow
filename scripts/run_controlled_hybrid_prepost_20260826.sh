#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AgentFlow
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
export AGENTFLOW_REWARD_JUDGE_CACHE_DIR=/root/autodl-tmp/tmp/reward_judge_20260826_hybrid_prepost
export AGENTFLOW_TRAIN_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_lora_mini20.yaml

RUN_ID="20260826_hybrid_prepost_$(date +%H%M%S)"
TRAIN_LOG="log/${RUN_ID}_train.log"
ROLLOUT_LOG="log/${RUN_ID}_rollout.log"
EXPERIMENT_TAG="qwen25-3b-lora-mini20-seed20260825-hybrid-prepost-20260826"
CHECKPOINT_DIR="checkpoints/agentflow-mini-baseline/${EXPERIMENT_TAG}"
TRAIN_PID=""
ROLLOUT_PID=""

mkdir -p log /root/autodl-tmp/tmp/reward_judge_20260826_hybrid_prepost

cleanup() {
  status=$?
  if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
    kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
  fi
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

check_abort_conditions() {
  local bad_events=0
  if [[ -f "$TRAIN_LOG" ]]; then
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|No valid rollout|No valid rollouts|(^|[^0-9])0 valid rollouts' "$TRAIN_LOG"; then
      echo "ABORT_CONDITION train_log_failure" >&2
      return 1
    fi
  fi
  if [[ -f "$ROLLOUT_LOG" ]]; then
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|No valid rollout|No valid rollouts|(^|[^0-9])0 valid rollouts' "$ROLLOUT_LOG"; then
      echo "ABORT_CONDITION rollout_log_failure" >&2
      return 1
    fi
    bad_events=$(grep 'HYBRID_REWARD_EVENT ' "$ROLLOUT_LOG" | grep -v 'error=none' | wc -l || true)
    if [[ "$bad_events" -ge 3 ]]; then
      echo "ABORT_CONDITION repeated_reward_judge_failure count=$bad_events" >&2
      return 1
    fi
  fi
  if [[ -d "$CHECKPOINT_DIR" ]] && find "$CHECKPOINT_DIR" -type f -print -quit | grep -q .; then
    echo "ABORT_CONDITION unexpected_checkpoint_file path=$CHECKPOINT_DIR" >&2
    return 1
  fi
  return 0
}

CMD=(
  /root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py
  --config "$AGENTFLOW_TRAIN_CONFIG"
  trainer.val_before_train=true
  trainer.save_freq=0
  trainer.experiment_name="$EXPERIMENT_TAG"
  data.val_files=/root/autodl-tmp/AgentFlow/data/train/flowgrpo_mini_20_seed20260825.parquet
)

echo "CONTROLLED_RUN_ID=$RUN_ID"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "ROLLOUT_LOG=$ROLLOUT_LOG"
echo "EXPERIMENT_TAG=$EXPERIMENT_TAG"
echo "VALIDATION_DATA=/root/autodl-tmp/AgentFlow/data/train/flowgrpo_mini_20_seed20260825.parquet"
echo "REWARD_JUDGE=hybrid_deepseek"
echo "TRAIN_COMMAND=${CMD[*]}"

PYTHONUNBUFFERED=1 "${CMD[@]}" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

READY=0
for _ in $(seq 1 180); do
  check_abort_conditions || exit 2
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID"
    exit $?
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
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
check_abort_conditions || exit 2
echo "CONTROLLED_RUN_COMPLETED=1"
