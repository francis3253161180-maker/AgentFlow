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
export AGENTFLOW_TRAIN_CONFIG=/root/autodl-tmp/AgentFlow/train/config_5090_no_lora_min_step.yaml

exec /root/autodl-tmp/conda/envs/agentflow/bin/python train/train_agent.py --config "$AGENTFLOW_TRAIN_CONFIG"
