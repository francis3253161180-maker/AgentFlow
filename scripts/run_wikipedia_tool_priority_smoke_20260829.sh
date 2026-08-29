#!/usr/bin/env bash
# One frozen MuSiQue prompt x four rollouts; never performs a training update.
set -euo pipefail

REPO=/root/autodl-tmp/AgentFlow
PY=/root/autodl-tmp/conda/envs/agentflow/bin/python
BASE="$REPO/train/config_5090_lora_smoke.yaml"
RUN_TAG="${AGENTFLOW_TOOL_SMOKE_TAG:-wikipedia_tool_priority_smoke_20260829}"
TMP="/root/autodl-tmp/tmp/${RUN_TAG}"
SOURCE=/root/autodl-tmp/tmp/multihop_allqwen_probe_20260829/musique.parquet
DATA="$TMP/musique_group4.parquet"
CONFIG="$TMP/config.yaml"
EXP="${AGENTFLOW_TOOL_SMOKE_EXPERIMENT:-wikipedia-tool-priority-musique-group4-20260829}"
TRAIN_LOG="$REPO/log/20260829_${RUN_TAG}_train.log"
ROLLOUT_LOG="$REPO/log/20260829_${RUN_TAG}_rollout.log"
GPU_LOG="$TMP/gpu.tsv"
ROUTE="$TMP/role_route.json"
VLLM_UTIL="${AGENTFLOW_WIKIPEDIA_SMOKE_VLLM_UTIL:-0.60}"
ENABLE_TOOLS="${AGENTFLOW_TOOL_SMOKE_ENABLE_TOOLS:-['Base_Generator_Tool', 'Python_Coder_Tool', 'Wikipedia_Search_Tool']}"
TOOL_ENGINES="${AGENTFLOW_TOOL_SMOKE_TOOL_ENGINES:-['frozen', 'frozen', 'frozen']}"
TOOL_STEPS="${AGENTFLOW_TOOL_SMOKE_TOOL_STEPS:-3}"
AGENT_MAX_TIMEOUT="${AGENTFLOW_TOOL_SMOKE_AGENT_MAX_TIMEOUT:-180}"
MAX_MODEL_LEN="${AGENTFLOW_TOOL_SMOKE_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${AGENTFLOW_TOOL_SMOKE_MAX_NUM_SEQS:-1}"
N_WORKERS="${AGENTFLOW_TOOL_SMOKE_N_WORKERS:-1}"

cd "$REPO"
source /root/.env
export PATH=/root/autodl-tmp/conda/envs/agentflow/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache TMPDIR=/root/autodl-tmp/tmp RAY_TMPDIR=/root/autodl-tmp/tmp/ray WANDB_MODE=disabled
export AGENTFLOW_TRAIN_CONFIG="$CONFIG" AGENTFLOW_DISABLE_EXTERNAL_LLM=1 AGENTFLOW_UNIFIED_LOCAL_ROLES=1 AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE=
export AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE=0.0 AGENTFLOW_REWARD_JUDGE_ENABLED=0 AGENTFLOW_REWARD_SCORER_LOG=1 AGENTFLOW_UNIFIED_MEMORY_LOG=1
export AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1 AGENTFLOW_ROLE_ROUTING_STATE="$ROUTE" AGENTFLOW_UNIFIED_BASE_MODEL_NAME=qwen-base
export AGENTFLOW_HIERARCHICAL_PLANNING="${AGENTFLOW_HIERARCHICAL_PLANNING:-0}"
export AGENTFLOW_UNIFIED_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct AGENTFLOW_UNIFIED_SMOKE_RUN_ID="${EXP}_$(date +%Y%m%d_%H%M%S)"
export AGENTFLOW_UNIFIED_TEMPERATURE=0.7 AGENTFLOW_UNIFIED_ROLLOUT_N=4 AGENTFLOW_UNIFIED_SEED=20260829
export AGENTFLOW_UNIFIED_SCORER="current deterministic scorer; external judge disabled" AGENTFLOW_UNIFIED_MAX_PROMPT_LENGTH=1536
export AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH=512 AGENTFLOW_UNIFIED_MAX_MODEL_LEN="$MAX_MODEL_LEN" AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1
export AGENTFLOW_VLLM_CLEANUP_DRAIN_TIMEOUT_SECONDS=30 AGENTFLOW_VLLM_CLEANUP_DRAIN_POLL_SECONDS=0.05 AGENTFLOW_ROLLOUT_WAIT_TIMEOUT_SECONDS=3600
mkdir -p "$TMP" "$REPO/log"

"$PY" - "$SOURCE" "$DATA" "$TMP/input_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
import pandas as pd
source, output, manifest = map(Path, sys.argv[1:])
rows = pd.read_parquet(source)
selected = rows.loc[rows["id"] == 4].copy()
if len(selected) != 1:
    raise SystemExit(f"expected exactly one frozen group-4 row, got {len(selected)}")
row = selected.iloc[0].to_dict()
extra = row.get("extra_info", {})
if extra.get("idx") != 259 or extra.get("benchmark_id") != "2hop__13592_49388":
    raise SystemExit("frozen group-4 provenance mismatch")
selected.to_parquet(output, index=False)
manifest.write_text(json.dumps({
    "schema_version": 1,
    "purpose": "bounded structural smoke only; no optimizer/checkpoint/training",
    "selection": "fixed existing MuSiQue probe group 4; no resampling",
    "source_parquet": str(source),
    "source_parquet_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "output_parquet": str(output),
    "output_parquet_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "id": int(row["id"]), "source_idx": int(extra["idx"]), "benchmark_id": extra["benchmark_id"],
    "question_sha256": hashlib.sha256(str(row["question"]).encode()).hexdigest(),
    "ground_truth_sha256": hashlib.sha256(str(row["result"]).encode()).hexdigest(),
    "rollout_n": 4, "seed": 20260829,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PY" - "$BASE" "$CONFIG" "$DATA" "$EXP" "$VLLM_UTIL" "$ENABLE_TOOLS" "$TOOL_ENGINES" "$TOOL_STEPS" "$AGENT_MAX_TIMEOUT" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$N_WORKERS" <<'PY'
from pathlib import Path
import sys
base, out, data, exp, util, enable_tools, tool_engines, tool_steps, agent_max_timeout, max_model_len, max_num_seqs, n_workers = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
replacements = {
    "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'": "BASE_MODEL: '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'",
    "EXPERIMENT_NAME: 'qwen25-3b-lora-flowgrpo-smoke'": f"EXPERIMENT_NAME: '{exp}'",
    "PROJECT_NAME: 'agentflow-smoke'": "PROJECT_NAME: 'wikipedia-tool-priority-smoke'",
    "ENABLE_TOOLS: ['Base_Generator_Tool']": f"ENABLE_TOOLS: {enable_tools}",
    "TOOL_ENGINE: ['deepseek-v4-flash']": f"TOOL_ENGINE: {tool_engines}",
    "MODEL_ENGINE: ['trainable', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']": "MODEL_ENGINE: ['trainable', 'frozen', 'frozen', 'frozen']",
    "data.train_files: '${BASE_DATA_DIR}/train/flowgrpo_smoke_2.parquet'": f"data.train_files: '{data}'",
    "data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'": f"data.val_files: '{data}'",
    "data.train_batch_size: 2": "data.train_batch_size: 1",
    "data.max_prompt_length: 1280": "data.max_prompt_length: 1536",
    "data.max_response_length: 384": "data.max_response_length: 512",
    "actor_rollout_ref.actor.ppo_mini_batch_size: 2": "actor_rollout_ref.actor.ppo_mini_batch_size: 1",
    "actor_rollout_ref.rollout.n: 2": "actor_rollout_ref.rollout.n: 4",
    "actor_rollout_ref.rollout.gpu_memory_utilization: 0.24": f"actor_rollout_ref.rollout.gpu_memory_utilization: {util}",
    "actor_rollout_ref.rollout.max_model_len: 2048": f"actor_rollout_ref.rollout.max_model_len: {max_model_len}",
    "actor_rollout_ref.rollout.max_num_batched_tokens: 2048": "actor_rollout_ref.rollout.max_num_batched_tokens: 1024",
    "actor_rollout_ref.rollout.max_num_seqs: 2": f"actor_rollout_ref.rollout.max_num_seqs: {max_num_seqs}",
    "agentflow.port: 9999": "agentflow.port: 9994",
    "N_WORKERS: 1": f"N_WORKERS: {n_workers}",
    "TOOL_STEPS: 2": f"TOOL_STEPS: {tool_steps}",
    "AGENT_MAX_TIMEOUT: 180": f"AGENT_MAX_TIMEOUT: {agent_max_timeout}",
    "trainer.val_before_train: False": "trainer.val_before_train: True\n  trainer.val_only: True",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing config anchor: {old}")
    text = text.replace(old, new, 1)
text = text.replace("  actor_rollout_ref.rollout.n: 4\n", "  actor_rollout_ref.rollout.n: 4\n  actor_rollout_ref.rollout.temperature: 0.7\n", 1)
Path(out).write_text(text, encoding="utf-8")
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
  "$PY" -m ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM
check_abort() {
  for path in "$TRAIN_LOG" "$ROLLOUT_LOG"; do
    [[ -f "$path" ]] || continue
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|illegal memory access|blocks are not freed yet|Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|worker died|No valid (training|validation) rollout|HTTP/[^ ]+ 5[0-9][0-9]|status[_ ]?code[=: ]+5[0-9][0-9]' "$path"; then
      echo "ABORT_CONDITION log_failure=$path" >&2
      return 1
    fi
  done
  return 0
}
: > "$TRAIN_LOG"; : > "$ROLLOUT_LOG"; : > "$GPU_LOG"
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then GOOGLE_AVAILABILITY=present; else GOOGLE_AVAILABILITY=missing; fi
echo "TOOL_BOUNDARY_SMOKE tag=$RUN_TAG model=Qwen2.5-7B-Instruct planner=qwen-actor-lora fixed_roles=qwen-base-adapter-off enabled_tools=$ENABLE_TOOLS tool_engines=$TOOL_ENGINES tool_steps=$TOOL_STEPS agent_max_timeout=$AGENT_MAX_TIMEOUT max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS n_workers=$N_WORKERS hierarchical_planning=$AGENTFLOW_HIERARCHICAL_PLANNING external_llm_calls=0 google_api_key=$GOOGLE_AVAILABILITY temp=0.7 n=4 prompts=1 rollout_only=1 optimizer_steps=0 checkpoint=disabled"
PYTHONUNBUFFERED=1 "$PY" train/train_agent.py --config "$CONFIG" trainer.val_only=true trainer.val_before_train=true trainer.save_freq=0 trainer.test_freq=0 trainer.experiment_name="$EXP" data.train_files="$DATA" data.val_files="$DATA" actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 data.max_prompt_length=1536 data.max_response_length=512 +actor_rollout_ref.ref.model.path=/root/autodl-tmp/models/Qwen2.5-7B-Instruct critic.model.path=/root/autodl-tmp/models/Qwen2.5-7B-Instruct +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.offload_policy=true >"$TRAIN_LOG" 2>&1 &
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
  check_abort || exit 2
  if grep -qE 'Total tasks queued:|Task queued:' "$TRAIN_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; exit $?; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { echo "ABORT_CONDITION timed_out_waiting_for_tasks" >&2; exit 2; }
PYTHONUNBUFFERED=1 "$PY" train/rollout.py >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do check_abort; sleep 5; done
wait "$TRAIN_PID"
check_abort
grep -q 'Validation summary:' "$TRAIN_LOG" || { echo "ABORT_CONDITION missing_validation_summary" >&2; exit 2; }
if grep -Eqi 'Training data keys|optimizer\.step|backward\(|global_step: [1-9]|Training Progress' "$TRAIN_LOG"; then
  echo "ABORT_CONDITION unexpected_training_marker" >&2
  exit 2
fi
for _ in $(seq 1 "${AGENTFLOW_TOOL_SMOKE_POST_VALIDATION_WAIT_SECONDS:-10}"); do [[ ! -e "/proc/$ROLLOUT_PID" ]] && break; sleep 1; done
if kill -0 "$ROLLOUT_PID" 2>/dev/null; then kill -TERM "$ROLLOUT_PID" 2>/dev/null || true; fi
wait "$ROLLOUT_PID" 2>/dev/null || true
ROLLOUT_PID=""
ROLLOUT_DIR=$(find "$REPO/rollout_data" -type d -path "*/${EXP}_*/Qwen2.5-7B-Instruct_*/train" -print | sort | tail -1)
[[ -n "$ROLLOUT_DIR" ]] || { echo "ABORT_CONDITION missing_rollout_data_directory" >&2; exit 2; }
COUNT=$(find "$ROLLOUT_DIR" -name '*.json' -type f | wc -l | tr -d ' ')
[[ "$COUNT" -eq 4 ]] || { echo "ABORT_CONDITION expected_four_rollouts_got=$COUNT" >&2; exit 2; }
echo "TOOL_BOUNDARY_SMOKE_COMPLETED rollout_dir=$ROLLOUT_DIR count=$COUNT input_manifest=$TMP/input_manifest.json gpu_log=$GPU_LOG"
