#!/usr/bin/env bash
set -euo pipefail

# Bounded causal sanity check: original AgentFlow flow, no optimizer step.
REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE_CONFIG="$REPO/train/config_5090_lora_smoke.yaml"
DATA=/root/autodl-tmp/tmp/gameof24_low_calibration_20260829/gameof24_low_calibration_3.parquet
MANIFEST=/root/autodl-tmp/tmp/gameof24_low_calibration_20260829/gameof24_low_calibration_3_manifest.json
TMP=/root/autodl-tmp/tmp/gameof24_planner_temp0_causal_sanity_20260829
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP="gameof24-planner-temp0-causal-sanity-20260829"
CONFIG="$TMP/${EXP}_${STAMP}.yaml"
TRAIN_LOG="$REPO/log/${EXP}_${STAMP}_train.log"
ROLLOUT_LOG="$REPO/log/${EXP}_${STAMP}_rollout.log"
GPU_LOG="$TMP/${EXP}_${STAMP}_gpu.tsv"
ROUTE_STATE="$TMP/${EXP}_${STAMP}_role_route.json"
SNAPSHOT="$TMP/${EXP}_${STAMP}_behavior_snapshot.pt"
SNAPSHOT_META="$TMP/${EXP}_${STAMP}_behavior_snapshot.json"
EVIDENCE_DIR="$TMP/${EXP}_${STAMP}_trajectories"
RUN_META="$TMP/${EXP}_${STAMP}_run_meta.json"

[[ -x "$PY" && -f "$BASE_CONFIG" && -f "$DATA" && -f "$MANIFEST" ]] || exit 2
cd "$REPO"
source /root/.env
[[ -n "${ARK_API_KEY:-}" ]] || { echo "ARK_API_KEY=missing" >&2; exit 2; }
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export WANDB_MODE=disabled
export AGENTFLOW_TRAIN_CONFIG="$CONFIG"
export AGENTFLOW_DISABLE_EXTERNAL_LLM=0
export AGENTFLOW_UNIFIED_LOCAL_ROLES=1
export AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE=doubao-seed-2-0-lite-260428
export AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE=0
export ARK_REASONING_EFFORT=minimal
export AGENTFLOW_REWARD_JUDGE_ENABLED=0
export AGENTFLOW_REWARD_SCORER_LOG=1
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1
export AGENTFLOW_ROLE_ROUTING_STATE="$ROUTE_STATE"
export AGENTFLOW_UNIFIED_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base
export AGENTFLOW_UNIFIED_SMOKE_RUN_ID="${EXP}_${STAMP}"
export AGENTFLOW_UNIFIED_TEMPERATURE=0.0
export AGENTFLOW_UNIFIED_ROLLOUT_N=4
export AGENTFLOW_UNIFIED_SEED=20260828
export AGENTFLOW_UNIFIED_SCORER="local deterministic; external reward judge disabled"
export AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1
export AGENTFLOW_BEHAVIOR_SNAPSHOT_PATH="$SNAPSHOT"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_METADATA_PATH="$SNAPSHOT_META"
export AGENTFLOW_REPLAY_CAPTURE_ENABLED=0
export AGENTFLOW_ROLLOUT_EVIDENCE_DIR="$EVIDENCE_DIR"
export AGENTFLOW_UNIFIED_CODE_COMMIT="$(git rev-parse HEAD)"
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS=3600
mkdir -p "$TMP" "$REPO/log" "$EVIDENCE_DIR"

"$PY" - "$BASE_CONFIG" "$CONFIG" "$DATA" "$EXP" <<'PY'
from pathlib import Path
import sys
base, out, data, exp = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{exp}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'game24-causal-sanity'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "TRAIN_TEMPERATURE: 0.7": "TRAIN_TEMPERATURE: 0.0",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.max_prompt_length: 1280": "data.max_prompt_length: 1536",
    "data.max_response_length: 384": "data.max_response_length: 1024",
    "actor_rollout_ref.rollout.n: 2": "actor_rollout_ref.rollout.n: 4",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": "actor_rollout_ref.rollout.gpu_memory_utilization: 0.50",
    "actor_rollout_ref.rollout.max_model_len: 2048": "actor_rollout_ref.rollout.max_model_len: 4096",
    "actor_rollout_ref.rollout.max_num_batched_tokens: 2048": "actor_rollout_ref.rollout.max_num_batched_tokens: 1024",
    "actor_rollout_ref.rollout.max_num_seqs: 2": "actor_rollout_ref.rollout.max_num_seqs: 1",
    "agentflow.port: 9999": "agentflow.port: 9997",
    "trainer.val_before_train: False": "trainer.val_before_train: True\n  trainer.val_only: True",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing config anchor: {old}")
    text = text.replace(old, new, 1)
needle = "  actor_rollout_ref.rollout.n: 4\n"
if needle not in text:
    raise SystemExit("rollout.n replacement failed")
text = text.replace(needle, needle + "  actor_rollout_ref.rollout.temperature: 0.0\n", 1)
Path(out).write_text(text, encoding="utf-8")
PY

"$PY" - "$RUN_META" "$MANIFEST" "$DATA" "$CONFIG" "$SNAPSHOT" "$SNAPSHOT_META" "$EVIDENCE_DIR" "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$ROUTE_STATE" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
out, manifest, data, config, snapshot, snapshot_meta, evidence, train_log, rollout_log, gpu_log, route = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
payload = {
    "schema_version": 1, "purpose": "3x4 planner temperature zero causal sanity check; rollout-only",
    "manifest": manifest, "manifest_sha256": sha(manifest), "prepared_data": data,
    "prepared_data_sha256": sha(data), "config": config, "config_sha256": sha(config),
    "model": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    "planner_main": {"model": "Qwen2.5-7B-Instruct", "lora": "current actor snapshot", "temperature": 0.0},
    "fixed_roles": {"engine": "doubao-seed-2-0-lite-260428", "temperature": 0.0, "reasoning_effort": "minimal"},
    "rollout_n": 4, "prompt_count": 3, "expected_rollouts": 12,
    "max_prompt_length": 1536, "max_response_length": 1024, "max_model_len": 4096,
    "vllm_gpu_memory_utilization": 0.50, "rollout_only": True, "optimizer_steps": 0,
    "checkpoint_disabled": True, "reward_scorer": "strict deterministic Game24; external judge disabled",
    "code_commit": os.environ["AGENTFLOW_UNIFIED_CODE_COMMIT"],
    "artifacts": {"behavior_snapshot": snapshot, "behavior_snapshot_metadata": snapshot_meta,
                   "evidence_dir": evidence, "train_log": train_log, "rollout_log": rollout_log,
                   "gpu_log": gpu_log, "role_route": route},
}
Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
  exit "$status"
}
trap cleanup EXIT INT TERM

check_abort() {
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    [[ -f "$path" ]] || continue
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|worker died|No valid (training|validation) rollout|HTTP/[0-9.]+ 5[0-9][0-9]' "$path"; then
      echo "CAUSAL_CHECK_ABORT log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}

rm -f "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$SNAPSHOT" "$SNAPSHOT_META" "$ROUTE_STATE"
echo "GAME24_CAUSAL_PROTOCOL model=Qwen2.5-7B-Instruct planner_temperature=0.0 fixed_engine=doubao-seed-2-0-lite-260428 fixed_temperature=0.0 reasoning_effort=minimal n=4 prompts=3 rollout_only=1 optimizer_steps=0 checkpoint=disabled reward_judge=disabled"
echo "GAME24_CAUSAL_MEMORY vllm_gpu_memory_utilization=0.50 max_num_seqs=1 max_num_batched_tokens=1024 max_model_len=4096"
echo "GAME24_CAUSAL_SNAPSHOT path=$SNAPSHOT metadata=$SNAPSHOT_META"
echo "GAME24_CAUSAL_MANIFEST path=$MANIFEST"

PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" \
  trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 \
  trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" \
  actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.0 \
  data.max_prompt_length=1536 data.max_response_length=1024 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
    sleep 5
  done
) &
MONITOR_PID=$!

ready=0
for _ in $(seq 1 360); do
  check_abort
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "CAUSAL_CHECK_ABORT timed_out_waiting_for_tasks" >&2; exit 2; fi
if [[ ! -s "$SNAPSHOT" || ! -s "$SNAPSHOT_META" ]]; then echo "CAUSAL_CHECK_ABORT missing_behavior_snapshot" >&2; exit 2; fi

PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort; sleep 5; done
wait "$TRAIN_PID"
check_abort
if ! grep -q 'Validation summary:' "$TRAIN_LOG"; then echo "CAUSAL_CHECK_ABORT missing_validation_summary" >&2; exit 2; fi
if grep -Eqi 'optimizer\.step|backward\(|Training Progress|actor update' "$TRAIN_LOG" "$ROLLOUT_LOG"; then echo "CAUSAL_CHECK_ABORT unexpected_training_marker" >&2; exit 2; fi

for _ in $(seq 1 180); do
  [[ ! -e "/proc/$ROLLOUT_PID" ]] && break
  sleep 1
done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; fi
wait "$ROLLOUT_PID" 2>/dev/null || true
ROLLOUT_PID=""
echo "GAME24_CAUSAL_STATUS=rollout_process_finished"
echo "GAME24_CAUSAL_STATUS=passed"
