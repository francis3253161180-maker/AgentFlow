# Current-step replanning diagnostic handoff — 2026-08-29

Commit base: `bcd6196`.  This was one frozen MuSiQue Barcelona question
(source index 259), `n=4`, rollout-only diagnostic.  It used the already
measured bounded setting (`N_WORKERS=4`, vLLM `max_num_seqs=4`), Qwen2.5-7B
actor LoRA, Qwen-base fixed roles, the existing scorer, three tool steps, and
the existing 600-second timeout.  No training, backward pass, optimizer step,
checkpoint, sample expansion, or external judge/API call occurred.

## Observed facts

- The implementation now constructs and persists a generic Current-Step State
  Contract before each Planner-main call.  It contains active step ID/objective/
  success criteria, stable unresolved-gap IDs, verified evidence, known URLs,
  same-step prior attempts, and the last verifier assessment.
- Every action stores its target gap, tool, sub-goal signature, whether the
  verifier observed material evidence/gap progress, and the reason.  The
  persisted result additionally stores each one-time action revision and the
  revised candidate.
- The first live validation exposed a harness defect: the legacy `NextStep`
  schema deliberately accepts absent `target_gap` for old replay artifacts, so
  vLLM guided JSON was not required to emit it.  Across 12 actions, all 12
  invoked the one permitted revision for an invalid/empty gap; 11 were empty
  and one was a quoted, nonmatching ID.  No revised action supplied a valid
  exact gap ID.  Consequently the requested guard could not reliably compare
  same-gap actions in this run.
- The defect is fixed in code after preserving that live evidence: new
  hierarchical generation uses `HierarchicalNextStep`, whose `target_gap` is
  required by the guided JSON schema.  `NextStep` remains backward-compatible
  for parsing historic persisted trajectories.  Unit tests prove both paths.
- The bounded run completed 4/4 valid rollouts with reward vector
  `[0.0, 0.0, 0.0, 0.0]`, zero retries, a 20,193 MiB recorded GPU peak, and no
  CUDA OOM/illegal memory access/prefix-cache error.  vLLM normal cleanup had
  `outstanding=0` and `drained=true` before reset/sleep.
- Structurally, all four trajectories remained on step_1 for three Wikipedia
  actions and terminated `max_steps_with_unresolved_plan`; no step_1 completion
  or step_2 activation occurred.  The verifier marked 4/12 action outcomes as
  material progress and 8/12 as no-progress.  Because the target field was
  absent, this is not valid evidence that the same-gap repetition guard works.
- As in the prior concurrency smoke, four known AgentFlow worker processes
  became PID-1 children after the parent finished.  They were explicitly
  terminated after the run; GPU is 0 MiB and no Ray/vLLM/AgentFlow worker is
  active.  This lifecycle limitation is separate from current-step semantics.

## Hypotheses

- Requiring the target ID at guided-JSON decode time should eliminate the
  observed empty-field failure, allowing the generic no-progress comparison to
  operate on an unambiguous current gap.  This is code/test evidence only; it
  has not yet been revalidated by a second rollout in this task.
- The observed repeated Wikipedia actions may reflect missing relevant local
  evidence rather than a bad preference for repetition.  The current evidence
  does not justify forcing a different tool; the new guard intentionally only
  rejects an unchanged tool+gap+objective after verifier-confirmed no-progress.

## Conclusions

- The state contract, progress telemetry, and one-revision mechanism are now
  implemented generically without dataset-specific tool routing or reward
  changes.
- This single validation is a documented harness-blocker result, not a
  structural success: `target_gap` was optional at runtime and the primary
  success criterion (a materially changed within-step strategy after a
  no-progress attempt) was not demonstrated.
- The post-run schema correction is minimal and preserves historical replay
  compatibility, but needs one newly authorized frozen-sample validation before
  claiming replanning-quality improvement.

## Recommendation

- Stop here as instructed.  On approval, run exactly one fresh frozen-sample
  `n=4` validation with the strict hierarchical schema, then evaluate target-ID
  validity, no-progress revisions, and step advancement.  Do not expand the
  sample or start training as part of that confirmation.

## Evidence and checks

- Results: `log/2026-08-29_current_step_replanning_results.json`.
- Raw local-only logs: `log/20260829_current_step_replanning_20260829_{train,rollout}.log`.
- Raw local-only rollouts:
  `rollout_data/46.38.243.197/current-step-replanning-musique-group4-20260829_20260829-212010/`.
- Focused tests cover state transitions, state-contract construction, unchanged
  no-progress repetition, strict hierarchical target fields, legacy parsing,
  tool guidance, and existing follow-up fixes.  Static/JSON/secret/diff checks
  are run before commit.
