# Stable active-step goal replanning validation — 2026-08-29

## Scope

This bounded diagnostic changes the hierarchical replanning contract only.  It
does **not** change the reward, model, tool set, optimizer, checkpoint policy,
or training data.  Exactly one frozen MuSiQue group was run: source index 259,
benchmark id `2hop__13592_49388`, `n=4`, three tool steps, `N_WORKERS=4`,
`max_num_seqs=4`, and a 600 s per-rollout limit.  No optimizer/backward step
was run.

Base commit before this scoped change: `21e3771`.

Artifacts retained locally (not committed):

- rollout directory: `rollout_data/46.38.243.197/stable-step-goal-musique-group4-20260829_20260829-214935/Qwen2.5-7B-Instruct_20260829-214935/train`
- train log: `log/20260829_stable_step_goal_20260829_train.log`
- rollout log: `log/20260829_stable_step_goal_20260829_rollout.log`
- frozen-input manifest: `/root/autodl-tmp/tmp/stable_step_goal_20260829/input_manifest.json`

Small, tracked aggregate: `log/2026-08-29_stable_step_goal_replanning_results.json`.

## Observed facts

### Code and persistence changes

- The active high-level plan step now supplies the persistent identity:
  `stable_step_id = current_step.step_id`.  The system, rather than Planner,
  attaches it to every action and persisted attempt.  Planner no longer has to
  copy/invent a volatile `target_gap` string.
- The hierarchical Planner prompt now asks for one atomic evidence step in the
  high-level plan.  It also explicitly keeps the active step's identity out of
  the model response schema.
- `made_progress` is now true only for verifier-grounded evidence addition or
  a completion/reopen state change.  A changed `missing_evidence` wording is
  stored as diagnostics but cannot by itself make progress true.
- Each executed attempt persists the stable step id, selected tool, sub-goal,
  context, actual executor command, normalized executable signature,
  evidence before/after, boolean progress, and verifier assessment.  The
  stale-action guard compares the actual normalized query/URL intent for the
  same stable step and tool; it permits a materially different query and never
  requires artificial tool diversity.

### Live validation

- All 4 trajectories accepted the schema and executed three actions; there
  were no `planner_action_invalid` or target-ID/schema failures.
- All 12 persisted attempts have `stable_step_id="step_1"`.  No identity was
  reconstructed from free-text missing-evidence output.
- Every action used the locally configured public Wikipedia retrieval tool.
  The twelve executable signatures were distinct (for example,
  `query barcelona league champions 1948 1949` followed by
  `query barcelona champions 1948 and 1949`).  Therefore the unchanged-query
  rejection path was not exercised; it was not needed for this run.
- In the first trajectory, step 1 changed only the diagnostic
  `missing_evidence` string after action 1.  It had no new verified evidence
  and `made_progress=false`, confirming that wording churn is not treated as
  progress.  Across the group, 7/12 attempts remained no-progress and 5/12
  recorded evidence additions; no attempt completed the active step.
- All trajectories terminated as `max_steps_with_unresolved_plan`, emitted an
  evidence-safe insufficient-evidence final answer, and received reward
  vector `[0, 0, 0, 0]`.  This smoke supplies no training-signal or accuracy
  claim.
- Public Wikipedia retrieval, not an LLM/judge API, was used.  Its telemetry
  recorded 21 HTTP 429 responses and 15 retries across 12 attempts.  This is a
  real evidence-quality limitation for the concurrent smoke; it does not show
  cross-rollout state leakage.  The results retain each query and returned URL
  provenance.

### Lifecycle and cache isolation

- GPU peak was 20,195 MiB of 32,607 MiB.  No CUDA illegal-memory-access,
  OOM, prefix-cache-block, Ray-worker-death, or deadlock marker appeared.
- The normal-complete cleanup reported `outstanding=0`, `drained=true`, then
  `reset_prefix_cache`, then `sleep`; the driver recorded a 0.772 s normal
  cleanup.  GPU usage was 0 MiB after the run.
- `Memory.reset()` is called per solve.  Wikipedia's response cache belongs to
  the per-worker tool instance and keys the request parameters; its returned
  payload is deep-copied.  The observed irrelevant/partial pages are fully
  attributable to the recorded queries and public search ordering/429 errors.
  There is no evidence here for cross-rollout cache contamination.
- The post-run process inspection still showed older PPID=1
  `AgentFlow-AgentOpsServer` processes.  They had no GPU allocation and were
  not killed because this task did not establish that each belonged to this
  smoke.  This is separate from the active vLLM cleanup, which completed.

## Hypotheses

- The one-step high-level plan remains too coarse for this multihop question:
  it combines identifying the league with deriving games-per-season.  The
  strengthened generic planner wording can encourage splitting this in future
  samples, but this bounded run must not be retuned to Barcelona.
- The public Wikipedia 429 rate under four concurrent rollouts likely reduced
  the chance of verifying the required relation.  It is not evidence that
  stable-step semantics caused the unresolved result.
- The unchanged-query guard remains covered by unit tests, but requires a
  future naturally repeated executable query to be exercised end-to-end.

## Conclusions

- The target-ID design defect is fixed structurally: active-step identity is
  stable, system-owned, and recorded without a Planner schema burden.
- Progress accounting now correctly refuses free-text diagnostic churn as
  evidence of advancement.  The run also demonstrates that a stable active
  step may receive different retrieval intents without an unjustified forced
  tool switch.
- The smoke does not establish successful multihop solution quality: all four
  groups were unresolved after the three-step budget, with public-retrieval
  throttling as a material confounder.
- Do not launch training or enlarge this sample based on this evidence.  The
  next decision should separately address high-level plan atomicity and
  retrieval reliability, while retaining the stable-ID/progress semantics.

## Verification

Commands completed successfully:

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python -m py_compile \
  agentflow/agentflow/models/current_step_state.py \
  agentflow/agentflow/models/planner.py \
  agentflow/agentflow/models/executor.py \
  agentflow/agentflow/solver.py \
  scripts/aggregate_wikipedia_tool_priority_smoke_20260829.py
/root/autodl-tmp/conda/envs/agentflow/bin/python -m unittest \
  test.test_hierarchical_planning test.test_tool_priority_guidance \
  test.test_followup_github_audit_fixes test.test_reward_scorer
bash -n scripts/run_wikipedia_tool_priority_smoke_20260829.sh
git diff --check
```

The unittest suite ran 34 tests successfully.  The reward-scorer tests include
mocked failure-path messages; they did not make an external judge request.
