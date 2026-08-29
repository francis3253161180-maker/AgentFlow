#!/usr/bin/env bash
set -euo pipefail

# Rollout-only random30 diagnostic. This uses val_only plus AgentFlow's
# training-rollout group mode; it never enters _train_step.
REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE_CONFIG="$REPO/train/config_5090_lora_smoke.yaml"
DATA=/root/autodl-tmp/tmp/random30_len1024_context4096_20260828/random30_train.parquet
MANIFEST="$REPO/log/2026-08-28_random30_len1024_context4096_probe_sample_manifest.json"
TMP=/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP="random30-fresh-rollout-replay-20260828"
CONFIG="$TMP/${EXP}_${STAMP}.yaml"
TRAIN_LOG="$REPO/log/${EXP}_${STAMP}_train.log"
ROLLOUT_LOG="$REPO/log/${EXP}_${STAMP}_rollout.log"
GPU_LOG="$TMP/${EXP}_${STAMP}_gpu.tsv"
ROUTE_STATE="$TMP/${EXP}_${STAMP}_role_route.json"
SNAPSHOT="$TMP/${EXP}_${STAMP}_behavior_snapshot.pt"
SNAPSHOT_META="$TMP/${EXP}_${STAMP}_behavior_snapshot.json"
REPLAY_PACK="$TMP/${EXP}_${STAMP}_pre_update_replay.pt"
EVIDENCE_DIR="$TMP/${EXP}_${STAMP}_trajectories"
LENGTH_AUDIT="$TMP/${EXP}_${STAMP}_length_audit.json"
RUN_META="$TMP/${EXP}_${STAMP}_run_meta.json"
VLLM_UTIL="${AGENTFLOW_FRESH_VLLM_UTIL:-0.30}"

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
export AGENTFLOW_DISABLE_EXTERNAL_LLM="${AGENTFLOW_DISABLE_EXTERNAL_LLM:-1}"
export AGENTFLOW_UNIFIED_LOCAL_ROLES=1
export AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE="${AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE:-}"
export AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE="${AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE:-}"
export ARK_REASONING_EFFORT="${ARK_REASONING_EFFORT:-}"
export AGENTFLOW_REWARD_JUDGE_ENABLED=0
export AGENTFLOW_REWARD_SCORER_LOG=1
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1
export AGENTFLOW_ROLE_ROUTING_STATE="$ROUTE_STATE"
export AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base
export AGENTFLOW_UNIFIED_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export AGENTFLOW_UNIFIED_SMOKE_RUN_ID="${EXP}_${STAMP}"
export AGENTFLOW_UNIFIED_TEMPERATURE=0.7
export AGENTFLOW_UNIFIED_ROLLOUT_N=4
export AGENTFLOW_UNIFIED_SEED=20260828
export AGENTFLOW_UNIFIED_SCORER="local deterministic; external judge disabled"
export AGENTFLOW_UNIFIED_MAX_PROMPT_LENGTH=1536
export AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH=1024
export AGENTFLOW_UNIFIED_MAX_MODEL_LEN=4096
export AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1
export AGENTFLOW_RESPONSE_LENGTH_AUDIT_PATH="$LENGTH_AUDIT"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_PATH="$SNAPSHOT"
export AGENTFLOW_BEHAVIOR_SNAPSHOT_METADATA_PATH="$SNAPSHOT_META"
export AGENTFLOW_REPLAY_PACK_PATH="$REPLAY_PACK"
export AGENTFLOW_REPLAY_CAPTURE_ENABLED=0
export AGENTFLOW_ROLLOUT_EVIDENCE_DIR="$EVIDENCE_DIR"
export AGENTFLOW_UNIFIED_CODE_COMMIT="$(git rev-parse HEAD)"
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30
export AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05
export AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS="${AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS:-3600}"
mkdir -p "$TMP" "$REPO/log" "$EVIDENCE_DIR"

"$PY" - "$BASE_CONFIG" "$CONFIG" "$DATA" "$EXP" "$VLLM_UTIL" <<'PY'
from pathlib import Path
import sys

base, out, data, exp, vllm_util = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{exp}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'random30-fresh-rollout-replay'",
    "TOOL_ENGINE: ['deepseek-v4-flash']": "TOOL_ENGINE: ['frozen']",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.max_prompt_length: 1280": "data.max_prompt_length: 1536",
    "data.max_response_length: 384": "data.max_response_length: 1024",
    "actor_rollout_ref.rollout.n: 2": "actor_rollout_ref.rollout.n: 4",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": f"actor_rollout_ref.rollout.gpu_memory_utilization: {vllm_util}",
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
text = text.replace(needle, needle + "  actor_rollout_ref.rollout.temperature: 0.7\n", 1)
Path(out).write_text(text, encoding="utf-8")
PY

"$PY" - "$RUN_META" "$MANIFEST" "$DATA" "$CONFIG" "$SNAPSHOT" "$SNAPSHOT_META" "$REPLAY_PACK" "$EVIDENCE_DIR" "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$ROUTE_STATE" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

out, manifest, data, config, snapshot, snapshot_meta, replay, evidence, train_log, rollout_log, gpu_log, route = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
payload = {
    "schema_version": 1,
    "manifest": str(Path(manifest)), "manifest_sha256": sha(manifest),
    "prepared_parquet": str(Path(data)), "prepared_parquet_sha256": sha(data),
    "config": str(Path(config)), "config_sha256": sha(config),
    "model_path": os.environ["AGENTFLOW_UNIFIED_MODEL_PATH"], "model_role": "Qwen2.5-7B-Instruct",
    "sampling": {"temperature": 0.7, "rollout_n": 4, "max_prompt_length": 1536, "max_response_length": 1024, "max_model_len": 4096, "dynamic_response_padding": True},
    "rollout_only": True, "optimizer_steps": 0, "checkpoint_disabled": True,
    "external_calls_disabled": os.environ.get("AGENTFLOW_DISABLE_EXTERNAL_LLM", "1").lower() in {"1", "true", "yes", "on"},
    "fixed_role_engine": os.environ.get("AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE", ""),
    "fixed_role_temperature": os.environ.get("AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE", ""),
    "ark_reasoning_effort": os.environ.get("ARK_REASONING_EFFORT", ""),
    "seed": 20260828, "code_commit": os.environ["AGENTFLOW_UNIFIED_CODE_COMMIT"],
    "artifacts": {"snapshot": snapshot, "snapshot_metadata": snapshot_meta, "replay_pack": replay, "evidence_dir": evidence, "route_state": route, "length_audit": os.environ["AGENTFLOW_RESPONSE_LENGTH_AUDIT_PATH"], "train_log": train_log, "rollout_log": rollout_log, "gpu_log": gpu_log},
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
  ray stop --force >/dev/null 2>&1 || true
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
  return 0
}

rm -f "$TRAIN_LOG" "$ROLLOUT_LOG" "$GPU_LOG" "$SNAPSHOT" "$SNAPSHOT_META" "$REPLAY_PACK" "$ROUTE_STATE" "$LENGTH_AUDIT"
echo "RANDOM30_FRESH_PROTOCOL model=Qwen2.5-7B-Instruct temp=0.7 n=4 max_prompt=1536 max_response=1024 max_model_len=4096 seed=20260828"
echo "RANDOM30_FRESH_MEMORY vllm_gpu_memory_utilization=$VLLM_UTIL max_num_seqs=1 max_num_batched_tokens=1024"
echo "RANDOM30_FRESH_MODE rollout_only=1 val_only=1 optimizer_steps=0 checkpoint=disabled external_calls_disabled=$AGENTFLOW_DISABLE_EXTERNAL_LLM fixed_role_engine=${AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE:-none} ark_reasoning_effort=${ARK_REASONING_EFFORT:-none}"
echo "RANDOM30_FRESH_SNAPSHOT path=$SNAPSHOT metadata=$SNAPSHOT_META"
echo "RANDOM30_FRESH_EVIDENCE_DIR=$EVIDENCE_DIR"
PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.max_prompt_length=1536 data.max_response_length=1024 +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.offload_policy=true >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
    sleep 5
  done
) & MONITOR_PID=$!

ready=0
for _ in $(seq 1 360); do
  check_abort
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then echo "ABORT_CONDITION timed_out_waiting_for_tasks" >&2; exit 2; fi
if [[ ! -s "$SNAPSHOT" || ! -s "$SNAPSHOT_META" ]]; then echo "ABORT_CONDITION missing_pre_rollout_behavior_snapshot" >&2; exit 2; fi
PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 & ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort; sleep 5; done
wait "$TRAIN_PID"
check_abort
if ! grep -q 'Validation summary:' "$TRAIN_LOG"; then echo "ABORT_CONDITION missing_validation_summary" >&2; exit 2; fi
if grep -Eqi 'Training data keys|optimizer\.step|backward\(|global_step: [1-9]|Training Progress' "$TRAIN_LOG"; then echo "ABORT_CONDITION unexpected_training_marker" >&2; exit 2; fi
for _ in $(seq 1 180); do [[ ! -e "/proc/$ROLLOUT_PID" ]] && break; sleep 1; done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; fi
wait "$ROLLOUT_PID" 2>/dev/null || true; ROLLOUT_PID=""
echo "RANDOM30_FRESH_STATUS=rollout_process_finished"
"$PY" scripts/aggregate_random30_fresh_rollout_replay_20260828.py --evidence-dir "$EVIDENCE_DIR" --manifest "$MANIFEST" --output "$TMP/${EXP}_${STAMP}_aggregate.json" --max-response 1024 --max-model-len 4096
"$PY" scripts/validate_random30_replay_pack_20260828.py --pack "$REPLAY_PACK" --snapshot "$SNAPSHOT" --evidence-dir "$EVIDENCE_DIR" --expected 120 --output "$TMP/${EXP}_${STAMP}_validation.json"
echo "RANDOM30_FRESH_STATUS=passed"
