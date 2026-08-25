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
export AGENTFLOW_TRAIN_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_lora_mini20.yaml

RUN_ID=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="log/${RUN_ID}_lora_mini20_train.log"
ROLLOUT_LOG="log/${RUN_ID}_lora_mini20_rollout.log"
ROLLOUT_PID=""

cleanup() {
  if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
    kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
  fi
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
pkill -f 'python train/rollout.py' 2>/dev/null || true
mkdir -p log

/root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py --config "$AGENTFLOW_TRAIN_CONFIG" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

READY=0
for _ in $(seq 1 180); do
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
  echo 'Timed out waiting for rollout tasks.' >&2
  kill "$TRAIN_PID" 2>/dev/null || true
  exit 1
fi

/root/autodl-tmp/conda/envs/agentflow/bin/python train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
wait "$TRAIN_PID"
