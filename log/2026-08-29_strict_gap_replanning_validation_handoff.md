# Strict-gap current-step replanning validation — 2026-08-29

Starting commit: `4162280`.  Phase A used the same frozen MuSiQue Barcelona
sample (source index 259), Qwen2.5-7B actor LoRA/Qwen-base fixed roles,
hierarchical planning, `n=4`, `N_WORKERS=4`, `max_num_seqs=4`, three steps,
and a 600-second timeout.  It was rollout-only: no optimizer, training,
checkpoint, reward change, sample expansion, or external judge/API call.

## Observed facts

- Phase A persisted the current unresolved-gap list and Planner emitted target
  for every first and one allowed revised action.  All four trajectories
  received the sole current ID `step_1::initial_objective::1` in their state
  contract, but emitted `target_gap=""` initially and after revision.
- The runtime exact-ID check rejected each candidate after its one revision;
  all **4/4** trajectories terminated `planner_action_invalid`.  No invalid
  action reached Executor, Wikipedia, Memory, step verification, or Final
  generation.  This safely satisfies the "do not silently execute" failure
  path, but Phase A target validity is **0/4 executed actions** because there
  were no executable actions (and 0/4 proposed/revised actions were valid).
- The existing strict hierarchical schema required the field to exist but
  allowed an empty string.  This was the immediate Phase A harness defect.
  After preserving the live evidence, `HierarchicalNextStep.target_gap` was
  tightened to `StrictStr(min_length=1)`.  The dynamic exact-current-ID check
  remains in Solver.  This code-only tightening was not followed by another
  rollout because the task permits only the one Phase A run unless the Phase B
  gate is met.
- Phase B did not run: its prerequisite—valid gap IDs followed by real
  stagnation—was not met.  No stable-gap/progress semantic change was made.
- The run had 4 rollout JSON records, reward vector `[0.0, 0.0, 0.0, 0.0]`,
  peak GPU memory 20,097 MiB, no timeout termination, and no CUDA OOM/illegal
  memory access/prefix-cache failure.  Normal cleanup logged `outstanding=0`,
  `drained=true`, then prefix reset and sleep.
- Cache-isolation code audit: `Memory.reset()` is called at every solver query;
  Wikipedia's response cache belongs to a tool instance, is keyed by all
  MediaWiki request parameters, and deep-copies results.  The initializer and
  Executor pass tool instances within an Agent worker, not across processes.
  This Phase A had zero tool executions, so it cannot empirically reproduce or
  refute the earlier irrelevant-URL observation; it also found no runtime
  evidence of cross-rollout contamination.
- As in prior four-worker smokes, the four known post-run worker PIDs became
  orphaned.  They were explicitly SIGTERM-cleaned; `ray stop --force` followed.
  GPU is 0 MiB and no Ray/vLLM/AgentFlow worker remains.

## Hypotheses

- Requiring a nonempty guided-JSON target field should prevent the exact empty
  output observed in Phase A.  Whether the local vLLM constrained decoder and
  planner then provide an exact dynamic gap ID remains unverified live.
- The prior free-text gap wording churn remains a credible telemetry issue,
  but there is no valid-gap stagnation evidence in this run to authorize the
  Phase B stable-gap/progress revision.

## Conclusions

- Phase A correctly prevented invalid Planner actions from being executed,
  exposing a schema/content gap rather than treating the earlier repeated
  Wikipedia actions as a replanning conclusion.
- The Phase B gate is not satisfied.  It would be premature to infer current-
  step strategy stagnation or change stable-gap semantics from this evidence.
- The nonempty schema constraint is a minimal correctness repair; it preserves
  legacy `NextStep` replay parsing and only affects new hierarchical actions.

## Recommendation

- Stop as instructed.  Before any Phase B work, obtain approval for one fresh
  Phase A confirmation under the `min_length=1` schema.  Only if that run has
  valid exact gap IDs and real no-progress repetitions should the single
  stable-gap/progress revision be considered.

## Evidence and verification

- Result JSON: `log/2026-08-29_strict_gap_replanning_validation_results.json`.
- Local-only logs: `log/20260829_strict_gap_phaseA_20260829_{train,rollout}.log`.
- Local-only rollout evidence:
  `rollout_data/46.38.243.197/strict-gap-phaseA-musique-group4-20260829_20260829-213525/`.
- Verification before commit: focused hierarchical/tool-guidance/scorer unit
  tests, `py_compile`, JSON consistency, `bash -n`, `git diff --check`, and a
  scoped secret scan.
