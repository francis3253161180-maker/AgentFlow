#!/usr/bin/env bash
set -euo pipefail

# Fixed rollout-only audit: 100 prompts, temperature 0.7, n=4.
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
export AGENTFLOW_REWARD_JUDGE_CACHE_DIR=/root/autodl-tmp/tmp/reward_judge_20260826_difficulty
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1

BASE_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_lora_mini20.yaml
PROMPT_DATA=/root/autodl-tmp/tmp/rollout_difficulty_audit_20260826/prompts_100.parquet
AUDIT_TMP=/root/autodl-tmp/tmp/rollout_difficulty_audit_20260826
TRAIN_LOG=log/20260826_rollout_difficulty_audit_train.log
ROLLOUT_LOG=log/20260826_rollout_difficulty_audit_rollout.log
GPU_LOG="$AUDIT_TMP/gpu_monitor.tsv"
CONFIG="$AUDIT_TMP/config_rollout_difficulty.yaml"
mkdir -p "$AUDIT_TMP" log

/root/autodl-tmp/conda/envs/agentflow/bin/python - "$BASE_CONFIG" "$CONFIG" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
text = base.replace("EXPERIMENT_NAME: 'qwen25-3b-lora-mini20-seed20260825'", "EXPERIMENT_NAME: 'rollout-difficulty-100-20260826'")
text = text.replace("actor_rollout_ref.rollout.n: 2", "actor_rollout_ref.rollout.n: 4")
needle = "  actor_rollout_ref.rollout.n: 4\n"
if needle not in text:
    raise SystemExit("failed to locate rollout.n replacement")
text = text.replace(needle, needle + "  actor_rollout_ref.rollout.temperature: 0.7\n", 1)
text = text.replace("data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'", "data.val_files: '/root/autodl-tmp/tmp/rollout_difficulty_audit_20260826/prompts_100.parquet'")
out.write_text(text, encoding="utf-8")
PY

TRAIN_PID=""
ROLLOUT_PID=""
MONITOR_PID=""
cleanup() {
  status=$?
  for pid in "$MONITOR_PID" "$ROLLOUT_PID" "$TRAIN_PID"; do if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi; done
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

check_abort_conditions() {
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    if [[ -f "$path" ]] && grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|No valid rollout|No valid rollouts|Traceback|illegal memory access|device-side assert|(^|[^0-9])0 valid rollouts' "$path"; then echo "ABORT_CONDITION log_failure=$path" >&2; return 1; fi
  done
  return 0
}

rm -f "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG"
export AGENTFLOW_TRAIN_CONFIG="$CONFIG"
cmd=(/root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py --config "$CONFIG" trainer.val_before_train=true trainer.val_only=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name=rollout-difficulty-100-20260826 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.val_files="$PROMPT_DATA")
echo "AUDIT_CONFIG=Qwen2.5-3B-Instruct LoRA rank8 alpha16 temp=0.7 rollout_n=4"
echo "AUDIT_PROMPT_DATA=$PROMPT_DATA"
echo "TRAIN_COMMAND=${cmd[*]}"

PYTHONUNBUFFERED=1 "${cmd[@]}" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
( while kill -0 "$TRAIN_PID" 2>/dev/null; do if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true; fi; sleep 5; done ) &
MONITOR_PID=$!

ready=0
for _ in $(seq 1 240); do
  check_abort_conditions || exit 2
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "ABORT_CONDITION timed_out_waiting_for_rollout_tasks" >&2; exit 2; fi

PYTHONUNBUFFERED=1 /root/autodl-tmp/conda/envs/agentflow/bin/python train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort_conditions || exit 2; sleep 5; done
wait "$TRAIN_PID"
check_abort_conditions || exit 2
if grep -Eqi 'Training data keys|optimizer\.step|(^| )actor/pg_loss:|backward\(|global_step: [1-9]' "$TRAIN_LOG"; then echo "ABORT_CONDITION unexpected_training_execution_marker" >&2; exit 2; fi
checkpoint_dir="checkpoints/agentflow-mini-baseline/rollout-difficulty-100-20260826"
if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -type f -print -quit | grep -q .; then echo "ABORT_CONDITION unexpected_checkpoint_marker" >&2; exit 2; fi
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill "$ROLLOUT_PID" 2>/dev/null || true; wait "$ROLLOUT_PID" 2>/dev/null || true; fi
ROLLOUT_PID=""
echo "ROLLOUT_DIFFICULTY_AUDIT_COMPLETED=1"
