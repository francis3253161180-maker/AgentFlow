# Typed three-step Game24 atomic-action Qwen-only smoke

## Scope and protocol

This was an isolated prototype smoke at base commit `e65e5d0`. It ran exactly the three requested solvable questions, one rollout per question. The planner was local `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` with the previously captured LoRA snapshot loaded (`lora_hash=2f46d9002978cbbf623f28d5113a3d03634246a9332d308d768fc13b86ddf8c9`). Temperature was `0.7`. There was no training, backward pass, optimizer step, checkpoint, HOB/GRPO update, fixed-role call, or external model call.

## Observed facts

- All 3/3 rollouts ran to the three-action terminal state.
- All 9/9 planner outputs were strict single JSON objects matching the typed action schema on the first attempt; schema retries: `0`.
- All 9/9 actions passed semantic checks: active distinct operands, supported operator, and nonzero divisor. Invalid-action retries: `0`.
- Deterministic state transitions preserved exact `Fraction` values, constructed expressions program-side, and preserved all four input indices in the terminal node.
- Rewards were `0/3`; reward mean `0.0`. The three terminal values were `1/18`, `1/48`, and `28`, so the failures were classified as `SEARCH/ARITHMETIC STRATEGY`, not FORMAT/SCHEMA or INVALID ACTION.
- The trajectories were: `[2,3,4,8] -> (4+8) -> 3*(4+8) -> 2/(3*(4+8))`; `[1,4,6,6] -> (6+6) -> 4*(6+6) -> 1/(4*(6+6))`; and `[1,7,8,12] -> (1+12) -> 8+(1+12) -> 7+(8+(1+12))`.
- Prompt and generated-response token counts, every action, state, deterministic result, and wall time are in the compact trace JSON.
- Runtime was `8.364s` including local model loading. GPU peak allocated memory was `14,688 MiB`; final GPU usage was `2 MiB`. No training/Ray/vLLM process remained.

## Implementation

- Added `agentflow/agentflow/models/game24_atomic.py`: strict `AtomicAction` Pydantic schema, `AtomicNode` with exact `Fraction` and provenance, deterministic state transitions, active-node/zero-divisor validation, strict JSON parsing, and terminal reward.
- Added `test/test_game24_atomic.py` covering state transitions, provenance, division by zero, invalid IDs, strict parsing/no free-form repair, exact three-step success, and terminal reward.
- Added `scripts/run_game24_atomic_qwen_only_20260828.py`: local-files-only Qwen planner runner with exactly one optional structured retry per step, deterministic transition/reward, compact trace, external-provider guard, and LoRA snapshot loading.

During the first runtime attempt, PEFT's convenience loader appended the adapter name twice to already-qualified snapshot keys. This was fixed generically by native PEFT module `load_state_dict(..., strict=False)` plus explicit verification that no LoRA keys were missing or unexpected. No rollout was produced in that failed attempt; the final run loaded all 392 LoRA tensors successfully.

## Diagnosis

The typed action interface removed the old final-expression formatting failure in this smoke: every planner response was machine-parseable and every action was semantically executable. It did not make the planner search correct. Each rollout made a legal but locally poor arithmetic choice and reached a non-24 terminal value. This is evidence of search/strategy failure on these three samples, not evidence of a scorer or transition bug.

The atomic path intentionally bypasses `planner_fixed`, `verifier`, and `executor`; therefore there is no fixed-role routing to validate beyond the explicit absence of those calls. The only model attribution is the local Qwen planner with the verified LoRA snapshot. The runner uses `local_files_only=True` and no OpenAI/Ark/Doubao/DeepSeek/GPT client.

## Tests and artifacts

- `32 passed, 43 subtests passed` for the existing relevant scorer/structured/provider/cleanup tests plus the new atomic tests.
- `py_compile` passed for the atomic module, runner, and test.
- Results: `log/2026-08-28_game24_atomic_qwen_only_smoke_results.json`.
- Compact trace: `log/2026-08-28_game24_atomic_qwen_only_smoke_trace.json`.
- Large model, snapshot, cache, and raw runtime artifacts remain outside Git.

## Recommendation

The prototype is technically valid and demonstrates that typed atomic actions provide clean step-level credit-assignment boundaries, but the 3-question result is `0/3` and is too small for a performance claim. Do not expand to 8×4 or start training from this smoke without a new approval. A next experiment, if approved, should compare a generic search-oriented prompt or action policy on a separately frozen sample while retaining the same deterministic environment and no sample-specific rules.
