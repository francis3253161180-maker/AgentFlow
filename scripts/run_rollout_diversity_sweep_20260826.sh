#!/usr/bin/env bash
set -euo pipefail

# Rollout-only 2x2 diversity sweep. This script is val_only and uses an
# explicit trainer-side opt-in so rollout.n is honored without entering
# _train_step, backward, optimizer, or checkpoint code.

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
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1

BASE_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_lora_mini20.yaml
PROMPT_DATA=/root/autodl-tmp/tmp/rollout_diversity_sweep_20260826/prompts_10.parquet
SWEEP_TMP=/root/autodl-tmp/tmp/rollout_diversity_sweep_20260826
mkdir -p "$SWEEP_TMP" log

python - "$BASE_CONFIG" "$SWEEP_TMP" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).read_text(encoding="utf-8")
out_dir = Path(sys.argv[2])
conditions = {"A0": (0.7, 2), "B0": (1.0, 2), "C0": (0.7, 4), "D0": (1.0, 4)}
for condition, (temperature, n) in conditions.items():
    text = base.replace(
        "EXPERIMENT_NAME: 'qwen25-3b-lora-mini20-seed20260825'",
        f"EXPERIMENT_NAME: 'rollout-diversity-{condition.lower()}-20260826'",
    )
    text = text.replace("TRAIN_TEMPERATURE: 0.7", f"TRAIN_TEMPERATURE: {temperature}")
    text = text.replace("actor_rollout_ref.rollout.n: 2", f"actor_rollout_ref.rollout.n: {n}")
    (out_dir / f"config_{condition}.yaml").write_text(text, encoding="utf-8")
PY

TRAIN_PID=""
ROLLOUT_PID=""
CURRENT_TRAIN_LOG=""
CURRENT_ROLLOUT_LOG=""

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
  if [[ -f "$CURRENT_TRAIN_LOG" ]] && grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|No valid rollout|No valid rollouts|(^|[^0-9])0 valid rollouts' "$CURRENT_TRAIN_LOG"; then
    echo "ABORT_CONDITION train_log_failure" >&2
    return 1
  fi
  if [[ -f "$CURRENT_ROLLOUT_LOG" ]] && grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|No valid rollout|No valid rollouts|(^|[^0-9])0 valid rollouts' "$CURRENT_ROLLOUT_LOG"; then
    echo "ABORT_CONDITION rollout_log_failure" >&2
    return 1
  fi
  return 0
}

start_condition="${START_CONDITION:-A0}"
started=0
for condition in A0 B0 C0 D0; do
  if [[ "$started" -eq 0 && "$condition" != "$start_condition" ]]; then
    continue
  fi
  started=1
  case "$condition" in
    A0) temperature=0.7; n=2 ;;
    B0) temperature=1.0; n=2 ;;
    C0) temperature=0.7; n=4 ;;
    D0) temperature=1.0; n=4 ;;
  esac
  experiment="rollout-diversity-${condition,,}-20260826"
  CURRENT_TRAIN_LOG="log/20260826_rollout_diversity_${condition}_train.log"
  CURRENT_ROLLOUT_LOG="log/20260826_rollout_diversity_${condition}_rollout.log"
  config="$SWEEP_TMP/config_${condition}.yaml"
  export AGENTFLOW_TRAIN_CONFIG="$config"
  TRAIN_PID=""
  ROLLOUT_PID=""
  rm -f "$CURRENT_TRAIN_LOG" "$CURRENT_ROLLOUT_LOG"

  cmd=(
    /root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py
    --config "$config"
    trainer.val_before_train=true
    trainer.val_only=true
    trainer.save_freq=0
    trainer.test_freq=0
    trainer.experiment_name="$experiment"
    actor_rollout_ref.rollout.n="$n"
    data.val_files="$PROMPT_DATA"
  )
  echo "SWEEP_CONDITION=$condition temperature=$temperature rollout_n=$n"
  echo "TRAIN_LOG=$CURRENT_TRAIN_LOG"
  echo "ROLLOUT_LOG=$CURRENT_ROLLOUT_LOG"
  echo "TRAIN_COMMAND=${cmd[*]}"

  PYTHONUNBUFFERED=1 "${cmd[@]}" >"$CURRENT_TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    check_abort_conditions || exit 2
    if grep -qE 'Total tasks queued:|Task queued:' "$CURRENT_TRAIN_LOG" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      wait "$TRAIN_PID"
      exit $?
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "ABORT_CONDITION timed_out_waiting_for_rollout_tasks" >&2
    exit 2
  fi

  PYTHONUNBUFFERED=1 /root/autodl-tmp/conda/envs/agentflow/bin/python train/rollout.py >"$CURRENT_ROLLOUT_LOG" 2>&1 &
  ROLLOUT_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    check_abort_conditions || exit 2
    sleep 5
  done
  wait "$TRAIN_PID"
  check_abort_conditions || exit 2
  if grep -Eqi 'Training data keys|optimizer\.step|(^| )actor/pg_loss:' "$CURRENT_TRAIN_LOG"; then
    echo "ABORT_CONDITION unexpected_training_update_marker" >&2
    exit 2
  fi
  checkpoint_dir="checkpoints/agentflow-mini-baseline/$experiment"
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -type f -print -quit | grep -q .; then
    echo "ABORT_CONDITION unexpected_checkpoint_marker" >&2
    exit 2
  fi
  echo "SWEEP_CONDITION_COMPLETED=$condition"

  if kill -0 "$ROLLOUT_PID" 2>/dev/null; then
    kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
  fi
  TRAIN_PID=""
  ROLLOUT_PID=""
  /root/autodl-tmp/conda/envs/agentflow/bin/ray stop --force >/dev/null 2>&1 || true
done

echo "ROLLOUT_DIVERSITY_SWEEP_COMPLETED=1"
