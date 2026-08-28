# Unified Qwen fixed-role architecture smoke handoff

## Observed facts

- Repository: `/root/autodl-tmp/AgentFlow`, branch `experiment/flow-grpo-3b-lora`; source baseline before this work was `4d7736a`.
- Environment: `verl==0.5.0`, `vllm==0.9.2`, `torch==2.7.1+cu128`, one RTX 5090 (32,607 MiB reported), Qwen2.5-7B-Instruct already present locally at `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`.
- No formal baseline, benchmark sweep, checkpoint, DeepSeek/Doubao/GPT request, or optimizer configuration change was started. The smoke script sets `AGENTFLOW_DISABLE_EXTERNAL_LLM=1` and `AGENTFLOW_REWARD_JUDGE_ENABLED=0`; reward telemetry in the successful run shows only local deterministic/conservative-fallback routes.
- Initial 7B actor initialization with default FP32 failed near the GPU limit. BF16 initial load plus FSDP2 `offload_policy=true` then initialized and ran successfully with vLLM co-location at `gpu_memory_utilization=0.60`.
- A first signal attempt was intentionally stopped after detecting configuration drift: the generated AgentFlow server still had `rollout.n=2` while the trainer override was `n=4`, and progress stopped at 8/8. Its logs are preserved as interrupted evidence and are not counted as a successful smoke.

## Hypotheses

- The default actor FP32 load was the main 7B initialization failure mechanism; BF16 plus the verified FSDP2 CPU offload path removes that memory pressure sufficiently for this 32 GB card.
- The stable logical aliases prevent fixed roles from inheriting the trainable adapter, while the runtime route registry bridges VERL's ephemeral in-memory LoRA id to `qwen-actor`.

## Conclusions

- 7B unified-base is feasible for this tiny single-GPU smoke with the documented conservative settings. It is not yet evidence that a 60-prompt formal run will fit at the same margin.
- In the final consistent signal smoke, all 4 groups produced 16/16 valid raw records and the reward vectors were: `17845=[0,0,0,0]`, `90185=[0,0,0,0]`, `50365=[0,1,1,0]`, `51933=[0,0,0,0]`. Reward mean was `2/16=0.125`; one group was mixed.
- Training step 1 was all-zero. Step 2 logged `critic/rewards/mean=0.25`, `critic/advantages` min/max `-0.9765625/0.9765625`, `actor/pg_loss=0.0015756587187449138`, `actor/grad_norm=0.5930989583333334`, and `actor/ppo_kl=0.0016729555351076564`. This is a real nonzero policy-gradient signal and update path, not a zero-gradient-only smoke.
- Runtime OpenTelemetry request attributes in the successful logs contain both `qwen-base` and `qwen-actor` (train log counts 150/46; rollout log counts 75/23). The route state was atomically published with a final adapter id/version, and two adapter-registration warnings mark the two post-update synchronizations. Fixed-role requests used the base alias; actor requests used the registered actor alias. A direct LoRA tensor checksum was not persisted, so the parameter-change claim is supported by nonzero optimizer metrics plus synchronized adapter registration, not by a saved full parameter diff.
- Cleanup was clean in the successful run: both normal-completion markers report `drained=True`, `outstanding_before=0`, `abort_errors=0`, reset/sleep completed, and final GPU usage was 2 MiB with no matching AgentFlow/Ray/vLLM process. No CUDA illegal access, OOM, prefix-cache reset failure, or deadlock appeared in the successful run.

## Recommended minimal fix

- Use `+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` and `actor_rollout_ref.actor.fsdp_config.offload_policy=true` only in the unified 7B smoke/formal config after separate approval; do not broaden the global defaults.
- Keep the one-base route contract: `qwen-base` has no adapter, and `qwen-actor` is refreshed from the latest VERL `TensorLoRARequest` publication. Keep KL disabled, so no separate RefPolicy model is required.
- The route-serving code now points the vLLM registry tokenizer lookup at the already-loaded base model path rather than an ephemeral numeric adapter path. The successful run was executed immediately before that last warning cleanup and therefore still contains the harmless historical `simon_lora_path` tokenizer fallback warning; the new behavior is covered by the focused unit test.
- Do not start formal training from this handoff. Before any formal run, add optional pre/post LoRA checksum instrumentation if a direct parameter-delta proof is required.

## Exact configuration and implementation deltas

- Smoke-only: model `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`; BF16 actor; FSDP2; `fsdp2.offload_policy=true`; vLLM TP=1; vLLM GPU utilization `0.60`; `max_num_seqs=1`; `max_num_batched_tokens=1024`; gradient checkpointing on.
- Signal smoke: 4 existing non-evaluation training prompts; `temperature=0.7`; `rollout.n=4`; `max_response_length=128`; train batch 2; PPO mini-batch 2; micro-batch 1; LR `1e-5`; `ppo_epochs=1`; total epochs 1; `save_freq=0`; no validation/checkpoint.
- Added stable role routing in `agentflow/agentflow/engine/role_routing.py`, unified aliases in `agentflow/agentflow/solver.py`, and request registry refresh/alias isolation in `agentflow/verl/async_server.py`.
- Added the reproducible VERL 0.5.0 site-package backport at `patches/verl_vllm_unified_route_backport.patch`. The installed site-package change is not itself tracked; the patch applies with `patch --dry-run` to a pristine VERL 0.5.0 wheel.
- Added `scripts/export_unified_replay_pack_20260828.py` and a dry-run test. The exporter was corrected to match the actual writer layout `train/step_N/idx_<id>/*.json`.

## Replay artifact

- Successful run raw root (local, untracked): `rollout_data/46.38.243.197/unified-qwen7b-fixed-roles-smoke-20260828_20260828-084052`.
- Immutable pre-update pack (local, untracked): `/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/replay_pack_pre_update_signal_step1.json`.
- Pack dry-run result: `status=ok`, `records=5`, `token_ids_available=false`.
- Pack SHA256: `d3f4317ef91346086225803d0cf3f46b4ef592e633e8c1dbf933c09b388beb93`.
- Limitation: the current AgentFlow JSON writer does not persist response token ids or old log-probs. The pack preserves the full available trajectory/reward metadata and records a deterministic pre-update recomputation contract. Step directories are per writer task, so `step_1` is the pinned pre-update slice, not a claim that all 16 records share one optimizer batch.

## Evidence paths and checks

- Successful train log: `log/unified-qwen7b-fixed-roles-smoke-20260828_20260828_083919_train.log`.
- Successful rollout log: `log/unified-qwen7b-fixed-roles-smoke-20260828_20260828_083919_rollout.log`.
- Interrupted drift log: `log/unified-qwen7b-fixed-roles-smoke-20260828_20260828_082935_train.log`.
- Runtime route state: `/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/unified-qwen7b-fixed-roles-smoke-20260828_20260828_083919_role_routes.json`.
- Focused unittest command: `python -m unittest test.test_unified_role_routing test.test_unified_local_roles test.test_vllm_timeout_cleanup test.test_unified_replay_pack -v`.
- `git diff --check` passes for source/tests/report artifacts; the two tracked unified patch files contain required single-space blank context lines, so the equivalent check uses `git -c core.whitespace=-blank-at-eol diff --check`. Both patches independently pass `patch --dry-run` against pristine VERL 0.5.0.
- The successful run's two cleanup markers, BF16/FSDP memory markers, route request telemetry, raw reward vectors, and final GPU/process state are summarized in the companion JSON.

## Observed facts / Hypotheses / Conclusions / Recommended minimal fix boundary

This smoke is an infrastructure feasibility result only. It does not authorize a formal 60-prompt baseline, PPO epoch change, HOB implementation, benchmark evaluation, or any new training run.
