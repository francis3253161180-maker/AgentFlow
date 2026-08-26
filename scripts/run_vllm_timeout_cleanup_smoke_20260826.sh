#!/usr/bin/env bash
set -euo pipefail

# Lifecycle smoke only: val_only rollout-only mode, no optimizer/checkpoint.
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
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1

MODE=${1:-forced}
BASE_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_lora_mini20.yaml
SOURCE_PROMPTS=/root/autodl-tmp/tmp/rollout_difficulty_audit_20260826/prompts_100.parquet
SMOKE_TMP=/root/autodl-tmp/tmp/vllm_timeout_cleanup_smoke_20260826
mkdir -p "$SMOKE_TMP" log

case "$MODE" in
  forced)
    PROMPT_COUNT=8; WAIT_TIMEOUT=180; HEALTH_CHECK=1; REQUEST_HOLD=180; REQUEST_HOLD_COUNT=1; REQUEST_HOLD_AFTER=4
    EXPERIMENT=timeout-cleanup-forced-20260826
    TRAIN_LOG=log/20260826_vllm_timeout_cleanup_forced_train.log
    ROLLOUT_LOG=log/20260826_vllm_timeout_cleanup_forced_rollout.log
    GPU_LOG="$SMOKE_TMP/forced_gpu.tsv"
    ;;
  mini)
    PROMPT_COUNT=4; WAIT_TIMEOUT=; HEALTH_CHECK=0; REQUEST_HOLD=; REQUEST_HOLD_COUNT=; REQUEST_HOLD_AFTER=
    EXPERIMENT=timeout-cleanup-mini-20260826
    TRAIN_LOG=log/20260826_vllm_timeout_cleanup_mini_train.log
    ROLLOUT_LOG=log/20260826_vllm_timeout_cleanup_mini_rollout.log
    GPU_LOG="$SMOKE_TMP/mini_gpu.tsv"
    ;;
  *) echo "usage: $0 {forced|mini}" >&2; exit 2 ;;
esac

PROMPT_DATA="$SMOKE_TMP/prompts_${MODE}_${PROMPT_COUNT}.parquet"
CONFIG="$SMOKE_TMP/config_${MODE}.yaml"
/root/autodl-tmp/conda/envs/agentflow/bin/python - "$SOURCE_PROMPTS" "$PROMPT_DATA" "$PROMPT_COUNT" <<'PY'
from pathlib import Path
import sys
import pandas as pd
source, target, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
frame = pd.read_parquet(source).head(count).copy()
if len(frame) != count:
    raise SystemExit(f"expected {count} prompts, got {len(frame)}")
target.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(target, index=False)
print(f"SMOKE_PROMPTS={target} count={len(frame)}")
PY

/root/autodl-tmp/conda/envs/agentflow/bin/python - "$BASE_CONFIG" "$CONFIG" "$EXPERIMENT" <<'PY'
from pathlib import Path
import sys
base = Path(sys.argv[1]).read_text(encoding="utf-8")
experiment = sys.argv[3]
text = base.replace("EXPERIMENT_NAME: 'qwen25-3b-lora-mini20-seed20260825'", f"EXPERIMENT_NAME: '{experiment}'")
text = text.replace("  actor_rollout_ref.rollout.n: 2", "  actor_rollout_ref.rollout.n: 4")
Path(sys.argv[2]).write_text(text, encoding="utf-8")
PY

export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_CLEANUP_HEALTH_CHECK="$HEALTH_CHECK"
export AGENTFLOW_TRAIN_CONFIG="$CONFIG"
if [[ "$MODE" == "forced" ]]; then export AGENTFLOW_CLEANUP_SMOKE_MIN_COMPLETION_RATE=0.01; else unset AGENTFLOW_CLEANUP_SMOKE_MIN_COMPLETION_RATE || true; fi
if [[ -n "$REQUEST_HOLD" ]]; then export AGENTFLOW_VLLM_TEST_REQUEST_HOLD_SECONDS="$REQUEST_HOLD"; else unset AGENTFLOW_VLLM_TEST_REQUEST_HOLD_SECONDS || true; fi
if [[ -n "$REQUEST_HOLD_COUNT" ]]; then export AGENTFLOW_VLLM_TEST_REQUEST_HOLD_COUNT="$REQUEST_HOLD_COUNT"; else unset AGENTFLOW_VLLM_TEST_REQUEST_HOLD_COUNT || true; fi
if [[ -n "$REQUEST_HOLD_AFTER" ]]; then export AGENTFLOW_VLLM_TEST_REQUEST_HOLD_AFTER="$REQUEST_HOLD_AFTER"; else unset AGENTFLOW_VLLM_TEST_REQUEST_HOLD_AFTER || true; fi
if [[ -n "$WAIT_TIMEOUT" ]]; then export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT"; else unset AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS || true; fi
export AGENTFLOW_REWARD_JUDGE_CACHE_DIR="$SMOKE_TMP/reward_cache_$MODE"

TRAIN_PID=""; ROLLOUT_PID=""; MONITOR_PID=""
cleanup() {
  status=$?
  for pid in "$MONITOR_PID" "$ROLLOUT_PID" "$TRAIN_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

check_failure_logs() {
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    if [[ -f "$path" ]] && grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|device-side assert|deadlock|No valid rollout|No valid rollouts' "$path"; then
      echo "SMOKE_ABORT failure_marker=$path" >&2
      return 1
    fi
  done
  return 0
}

rm -f "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG"
cmd=(/root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py --config "$CONFIG"
  trainer.val_before_train=true trainer.val_only=true trainer.save_freq=0 trainer.test_freq=0
  trainer.experiment_name="$EXPERIMENT" actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.temperature=0.7 data.val_files="$PROMPT_DATA")
echo "SMOKE_MODE=$MODE"
echo "SMOKE_CONFIG=Qwen2.5-3B-Instruct LoRA rank8 alpha16 temp=0.7 rollout.n=4 val_only=true save_freq=0"
echo "SMOKE_COMMAND=${cmd[*]}"
PYTHONUNBUFFERED=1 "${cmd[@]}" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
( while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true; fi
    sleep 5
  done ) &
MONITOR_PID=$!

ready=0
for _ in $(seq 1 300); do
  check_failure_logs || exit 2
  if grep -q 'Total tasks queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "SMOKE_ABORT timed_out_waiting_for_tasks" >&2; exit 2; fi

PYTHONUNBUFFERED=1 /root/autodl-tmp/conda/envs/agentflow/bin/python train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_failure_logs || exit 2; sleep 5; done
wait "$TRAIN_PID"
check_failure_logs || exit 2

if grep -Eqi 'Training data keys|optimizer\.step|actor/pg_loss|backward\(|global_step: [1-9]' "$TRAIN_LOG"; then echo "SMOKE_ABORT unexpected_training_marker" >&2; exit 2; fi
if find checkpoints/agentflow-mini-baseline -path "*${EXPERIMENT}*" -type f -print -quit 2>/dev/null | grep -q .; then echo "SMOKE_ABORT unexpected_checkpoint_marker" >&2; exit 2; fi
if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill "$ROLLOUT_PID" 2>/dev/null || true; wait "$ROLLOUT_PID" 2>/dev/null || true; fi
ROLLOUT_PID=""
echo "VLLM_TIMEOUT_CLEANUP_SMOKE_COMPLETED=1 mode=$MODE"
