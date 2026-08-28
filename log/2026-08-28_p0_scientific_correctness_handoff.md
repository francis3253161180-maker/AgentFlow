# P0 scientific-correctness handoff

## Observed facts

- Scope is the AgentFlow/VERL training path at base revision `8199c20` (no
  rollout, model inference, GPU experiment, optimizer step, checkpoint, or
  external API was run for this change).
- The prior `9447d83` safe-math change is present in the current ancestry. Its
  offline audit identified five genuine deterministic math false negatives and
  four Klein-bottle rows where the ground truth itself was defective; the
  latter must remain a data-quality issue, not be solved by a looser scorer.
- Before this change, the daemon could flatten multiple transitions from one
  rollout and pass them to VERL's row-based GRPO grouping. It also had no
  stable prompt/rollout metadata sufficient to distinguish actor transitions
  from fixed-role transitions, and response truncation was not marked for
  dropping.

## Code changes

- `agentflow/tracer/triplet.py` now carries model/role/trainable identity from
  span attributes (`qwen-actor` → `planner_main`, `qwen-base` → fixed) into
  triplet metadata. `agentflow/runner.py` filters unified-local PPO input to
  explicitly attributed trainable planner transitions and fails closed for
  unattributed tokenized spans; the full trace remains available as evidence.
- `agentflow/verl/identifiers.py` adds a stable prompt id based on source
  identity and canonical prompt content, never on per-batch `data_id` or
  `rollout_id`.
- `agentflow/verl/advantage.py` computes GRPO mean/std over one reward per
  unique `(prompt_id, data_id, rollout_id)` outcome, then broadcasts the scalar
  advantage to every transition in that rollout while preserving the response
  mask. Incomplete groups fail closed.
- `agentflow/verl/daemon.py` tracks retries by `(data_id, logical_slot)`, does
  not increase the logical denominator for retries, rejects incomplete
  training groups before batch creation, propagates prompt/rollout rewards,
  and marks both prompt and response truncation as drop masks.
- `agentflow/verl/trainer.py` uses the rollout-level GRPO path, uses stable
  prompt ids for grouping, and drops flagged transitions after advantage
  computation so truncated transitions are never trained. Offline replay
  update now requires current behavior-snapshot verification plus model,
  sampling, seed, schema, field-digest, complete-field, and LoRA-hash checks.
- `agentflow/agentflow/models/structured_outputs.py`, `train/rollout.py`, and
  `train/utils.py` route identifiable Game24 tasks through strict local
  validation: marked/structured expression, allowed arithmetic grammar, exact
  original four-number multiset with multiplicity, and exact Fraction result
  24. Invalid or unmarked repaired expressions are not rewarded.
- `agentflow/agentflow/engine/vllm.py` now re-raises generation exceptions so
  bounded retry/infra-failure handling can see them; an error dictionary can
  no longer silently become an ordinary zero reward.

## Tests

- New `test/test_p0_scientific_correctness.py` covers actor-only filtering,
  stable role identity, unequal-transition rollout groups, incomplete groups,
  logical-slot retry bounds, truncation drop behavior, strict Game24 reward,
  and replay identity/digest/hash/drop validation.
- Focused suite: **49 passed**, **39 subtests passed**, one pre-existing Ray
  deprecation warning.
- `py_compile` passed for all changed Python modules and tests.
- `git diff --check` passed after removing trailing whitespace.

## Deferred / remaining risks

- This is a code-and-test handoff only. No runtime claim is made about a live
  vLLM/AgentFlow process in this turn.
- The installed VERL 0.5.0 backport is not modified. Replay updates now
  intentionally fail closed unless the existing behavior-snapshot hook can
  verify the actor LoRA hash; old/incomplete packs should therefore be
  rejected rather than silently consumed.
- Role attribution depends on the production OpenTelemetry model attribute;
  unified-local mode rejects tokenized spans when that attribution is absent.
- The prior `safe_math_mismatch` fix remains the general math fix; this change
  does not relax it to accommodate GT defects. DeepSeek semantic-judge
  behavior is unchanged and was not called.

## Delivery

- Base revision: `8199c20`.
- The final delivery commit hash is recorded by the repository history and in
  the final agent handoff. Only the files listed by the final commit should be
  considered part of this P0 change; existing raw logs and unrelated dirty
  files were not staged.
