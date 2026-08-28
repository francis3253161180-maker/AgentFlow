# Doubao fixed-role matched smoke handoff

## Observed facts

- Starting point was branch `experiment/flow-grpo-3b-lora` at `6d66cdd1f7ca27f42a94ef321def20852db10fb6`.
- The prior corrected Qwen reference completed 8 groups / 32 rollouts with 32/32 valid, zero retries, reward successes 0/32, group bins `0/4=8`, `1/4=0`, `2/4=0`, `3/4=0`, `4/4=0`, and mixed groups 0/8.
- Its configured protocol was Qwen2.5-7B-Instruct, temperature 0.7, `n=4`, max prompt 3072, max response 1024, max model length 8192, dynamic response padding enabled, rollout-only, optimizer steps 0, and checkpoint disabled.
- Qwen route evidence was `qwen-actor` with the synchronized LoRA for planner_main and `qwen-base` without an adapter for fixed roles. The local deterministic Game24 reward path was used; no DeepSeek judge was enabled.
- After sourcing `/root/.env`, `ARK_API_KEY` was present. Its value was never printed or written to a file.
- Exactly one real Ark request was made using the required endpoint and model, with a short non-streaming prompt and `temperature=0`. It returned `404 NotFoundError` with provider message `InvalidEndpointOrModel.NotFound` in approximately 472.73 ms. Response content and credentials were not recorded.

## Code changes

- Added `agentflow/agentflow/engine/ark.py`, a small OpenAI-compatible provider adapter using `OpenAI(...).chat.completions.create(...)`.
- The adapter resolves credentials from `ARK_API_KEY`, defaults to the required Ark base URL, and uses the exact configured Doubao model string. It does not print or persist credentials; optional cache keys are SHA-256 hashes of prompt inputs.
- Structured requests use the least provider-specific `{"type":"json_object"}` hint. Existing AgentFlow Pydantic parsing and strict Game24 validation remain authoritative; no validator or reward rule was loosened.
- `factory.py` routes only explicit `doubao-*` model names and fail-closes them when external LLMs are disabled.
- `solver.py` adds `AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE` and `AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE`. In unified mode these affect planner_fixed, verifier, executor, and fixed `self`/`frozen` tool engines only. planner_main remains the local `qwen-actor` route with LoRA. KL/reference-policy behavior was not changed.
- The existing rollout harness now preserves its default external-disabled behavior while allowing an explicit future fixed-role provider override.

## Tests and verification

- CPU/mock and regression tests passed: **31 passed, 39 subtests passed**.
- `py_compile` passed for the changed Python modules.
- The earlier Qwen lifecycle evidence showed `drain_complete=1`, `complete=1`, no CUDA illegal memory access, OOM, prefix-cache reset failure, deadlock, or worker death.
- After the sanity failure, GPU usage was 2 MiB and no training, Ray, or vLLM process remained.

## Live results

The paid Ark sanity gate failed at model/endpoint access. Per the task instruction, no alternate endpoint or model was tried and the matched 8-group rollout was **not run**. Therefore there are no Doubao rollout reward, per-role rollout-call, token-usage, or cleanup metrics to report.

## Decision

The provider wiring is covered by mock tests, but the live provider/model combination is not usable in this environment as tested. The evidence gate is unresolved rather than a zero-reward Doubao result. Do not start GRPO/HOB exploration or infer that Doubao fixes the Qwen planner bottleneck from this run.

## Artifacts

- Results JSON: `log/2026-08-28_doubao_fixed_roles_matched_smoke_results.json`
- Qwen reference aggregate: `/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/final-qwen-fixed-role-smoke-20260828_20260828_203939_aggregate.json`
- Qwen reference train log: `log/final-qwen-fixed-role-smoke-20260828_20260828_203939_train.log`
- Qwen reference rollout log: `log/final-qwen-fixed-role-smoke-20260828_20260828_203939_rollout.log`

Raw rollout data and logs remain local and are not included in the commit.
