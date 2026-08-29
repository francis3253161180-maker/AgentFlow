# Reward-0 retrieval and plan-revision audit — 2026-08-29

## Scope

This task audited historical positive rewards before changing the hierarchical
planning/retrieval plumbing.  It did not weaken the evidence gate, change the
reward scorer, call an external judge/LLM, train, backpropagate, save a
checkpoint, or expand the frozen sample.  The only live execution was the
authorized one-question MuSiQue group (source index 259, benchmark id
`2hop__13592_49388`), `n=4`, three tool steps, `N_WORKERS=4`,
`max_num_seqs=4`, 600 s limit, with local Qwen actor+LoRA and Qwen-base fixed
roles.

Base commit: `3dbae94`.

Tracked artifacts:

- `log/2026-08-29_reward0_retrieval_plan_revision_pre_audit.json`
- `log/2026-08-29_reward0_retrieval_plan_revision_results.json`

Raw rollout evidence remains local:

- `rollout_data/46.38.243.197/reward0-retrieval-plan-revision-musique-group4-20260829_20260829-221208/Qwen2.5-7B-Instruct_20260829-221208/train`
- `log/20260829_reward0_retrieval_plan_revision_20260829_train.log`
- `log/20260829_reward0_retrieval_plan_revision_20260829_rollout.log`

## Observed facts

### Offline historical-positive audit

- The earlier `a22d286` positive and both `2473baa` positives emitted answer
  `38` despite their own verifier responses stating that the retrieved Memory
  did **not** establish the historical league and/or its games-per-season
  value.  Their raw Wikipedia evidence was introductory material such as
  Barcelona currently competing in La Liga, a Premier League result, or a
  generic Barcelona history; it did not establish the required two-hop chain.
- Their final text explicitly filled the gap with claims such as “we can infer”
  and “typically 38 matches.”  Those are final-generator parametric-knowledge
  completions, not Memory-supported conclusions.
- Therefore the old `reward=1` rows are unsupported reward false positives.
  Their removal by evidence-gated finalization is not an accuracy regression.

### Authorized live confirmation

- The run completed 4/4 valid rollouts with no optimizer/backward/checkpoint.
  Raw scorer result: reward vector `[0, 0, 1, 0]`, mean `0.25`.  Three
  trajectories safely ended with insufficient verified evidence; one emitted
  `38` after the incomplete one-step plan was marked complete.
- This positive is **not** accepted as a grounded success.  The coverage role
  itself enumerated two independently necessary requirements (identify the
  historical league; establish games per season), while the plan had only
  `step_1`.  Yet it returned `sufficient=true`, `covered_step_ids=[step_1]`,
  and a rationale saying that games-per-season was not covered.  The original
  coverage schema had no per-requirement-to-step mapping, so the initial
  implementation could not deterministically reject this contradiction.
- The four live plans therefore did not satisfy the requested plan-coverage
  success criterion.  This was a diagnostic failure, not evidence for a
  reward improvement; no second rollout was started because this task
  authorizes exactly one confirmation.
- The first implementation's global request admission substantially reduced
  public MediaWiki throttling relative to the prior 21 HTTP 429 events, but did
  not eliminate it: 3 HTTP 429s, 3 retries, 35 s shared retry-after delay, and
  43.32 s aggregate throttle wait were recorded.  The results retain the raw
  snippets/intros and query provenance.  No Doubao/OpenAI/DeepSeek calls were
  recorded.
- The run used only Wikipedia retrieval paths (no forced diversity); all 12
  executed query signatures were materially different, so the near-duplicate
  stale-action rejection did not fire.  This is expected rather than a failed
  guard test.
- Peak GPU memory was 20,099 MiB / 32,607 MiB.  Normal cleanup reported
  `outstanding=0`, `drained=true` before prefix-cache reset/sleep; no CUDA OOM,
  illegal-memory-access, prefix-cache-block, deadlock, or worker-death marker
  occurred.  GPU memory was 0 MiB after completion.

### Final code changes

- The fixed Qwen coverage audit now requires an explicit
  `requirement_coverage` mapping from every independently necessary requirement
  to exactly one atomic step.  A positive verdict is accepted only when every
  requirement is mapped to existing steps, mappings are nonempty, and no
  step is silently reused for multiple independent requirements.  Malformed or
  internally inconsistent coverage causes at most one fixed-role plan revision
  and then the safe `high_level_plan_coverage_invalid` termination gate.
- The high-level planner persists the original plan, initial coverage audit,
  optional one-revision plan, final audit, and validated plan.  This does not
  generate factual answers or choose a dataset-specific plan.
- Public Wikipedia now uses a cross-process request admission lock, shares
  server `Retry-After` backoff, and preserves sanitized MediaWiki result
  snippets alongside title, URL, and bounded intro.
- Per the requested isolation boundary, the final code also adds a process-safe
  raw-response cache keyed by provider, endpoint, and canonical HTTP params,
  with per-key singleflight.  Only successful JSON responses are atomically
  cached/deep-copied; 429/timeouts/errors are never cached.  Planner/Memory/
  Verifier/history/final/reward remain rollout-local.  The shared-cache and
  singleflight code is unit-tested but was added after the one authorized live
  confirmation, so it has no live-effect claim in this report.
- Current-step stable IDs and strict verifier-grounded progress semantics from
  `3dbae94` remain unchanged.  The stagnant-action guard now treats cosmetic
  variations of the same executable query as a single retry target while
  retaining queries with changed numeric seasons/entities/relations.

## Hypotheses

- Once the corrected requirement mapping is exercised live, Qwen may produce a
  two-step dependency plan and prevent the unsupported `38` final path.  This
  is a hypothesis; the one completed run used the superseded coverage schema.
- Cross-process singleflight should remove duplicate same-HTTP-request traffic
  among four rollout workers and reduce environment-induced reward variance.
  Its effect on real MediaWiki 429 frequency remains unmeasured here.
- Even with complete planning and reliable retrieval, the public top-two
  introductory retrieval policy may lack the requested historical relation.
  That would be a retrieval limitation, not permission to relax grounding.

## Conclusions

- Do not restore historical free final generation to recover reward positives:
  the audited positives are unsupported false positives.
- The initial coverage check was insufficient and the one live confirmation is
  invalid as a pass/fail test of plan completeness.  The final code corrects
  the identified generic schema/validation defect, but requires explicit
  approval for any new live confirmation.
- Retrieval admission was directionally helpful (21 prior 429s versus 3), but
  the robust shared raw-response cache/singleflight design should be treated as
  infrastructure correctness, not shared agent state or a credit-assignment
  shortcut.
- Stop here.  No training, second confirmation, or sample expansion was run.

## Verification

Completed checks:

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python -m py_compile \
  agentflow/agentflow/models/formatters.py \
  agentflow/agentflow/models/current_step_state.py \
  agentflow/agentflow/models/planner.py \
  agentflow/agentflow/solver.py \
  agentflow/agentflow/tools/wikipedia_search/tool.py \
  scripts/aggregate_wikipedia_tool_priority_smoke_20260829.py \
  scripts/audit_reward0_retrieval_plan_revision_20260829.py
/root/autodl-tmp/conda/envs/agentflow/bin/python -m unittest \
  test.test_hierarchical_planning test.test_tool_priority_guidance \
  test.test_followup_github_audit_fixes test.test_reward_scorer
bash -n scripts/run_wikipedia_tool_priority_smoke_20260829.sh
git diff --check
```

The focused hierarchy/tool suite (17 tests) passed after the final coverage and
shared-cache changes.  The full listed suite is run before commit; reward
failure-path messages are mocks and do not make external judge requests.
