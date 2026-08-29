# Grounded step verification + run-scoped retrieval cache: live confirmation

## Scope

This was exactly one frozen MuSiQue Barcelona question with four concurrent
rollouts (`n=4`, `N_WORKERS=4`, `max_num_seqs=4`, three agent steps, 600 s).
It was rollout-only: no optimizer, backward pass, checkpoint, training, or
external LLM/reward-judge call.  The local Qwen setup and the fixed manifest
are recorded in `2026-08-29_grounded_step_verification_cache_live_results.json`.

## Observed facts

### Audit of the latest historical `[0, 0, 1, 0]` run

The only historical positive was
`rollout_82289c9c-a92a-44ec-b10b-7609e1717c45.json` from
`reward0-retrieval-plan-revision-musique-group4-20260829`.  Its answer was
`38`, but the only recorded retrieval was `Barcelona league titles 1948 1949`.
The returned snippet established only that Barcelona added titles in 1948 and
1949; it did not name the relevant historical league or establish games per
season.  The final text explicitly called 38 a modern-structure estimate.

The old coverage object nevertheless mapped both requirements to `step_1` and
marked itself sufficient, although its own rationale said that games per
season was not covered.  The old verifier marked the step complete without
quoted, traceable evidence.  This is unsupported final generation, not a
grounded success; no reward rule was weakened to change that conclusion.

### Changes

- `StepVerification` now carries per-requirement `Action Step N` references
  and exact evidence quotes.
- The solver attaches the approved requirement-to-plan-step map to live plan
  state and applies a deterministic provenance gate before a verifier may
  complete a step.  It checks that each active requirement has an existing
  action reference and a normalized quote present in that referenced raw tool
  result.  It does **not** decide whether a quote semantically proves an
  answer.  A failed gate retains the step in progress and persists a reason.
- The current-step contract exposes both mapped active requirements and known
  URL `Web_RAG_Search_Tool` deep-read affordances.  It does not force a tool
  sequence; a new Wikipedia query remains permitted.
- The Wikipedia layer now records actual HTTP request counts.  The runner
  makes a fresh, run-scoped raw-success cache and shared throttle lock before
  the run, so this confirmation cannot reuse old responses.  The existing
  shared file cache/singleflight caches successful raw responses only, returns
  copies, and does not cache timeout/HTTP failure responses.  Global admission
  is 0.75 s.

### One live confirmation

All four initial one-step plans failed coverage liveness because they mapped
both requirements to `step_1`.  The single generic plan revision split the
work into `step_1` (identify the league) and dependent `step_2` (games per
season); the final requirement-to-step map was valid in all four rollouts.

Every rollout kept `step_1` active for all three actions because the evidence
never named the historical league.  The deterministic completion gate was
therefore invoked three times per rollout with `completion_not_requested`;
the fixed verifier did not falsely request completion.  All final answers
were the solver's explicit `Insufficient verified evidence...` form, all
rewards were zero, and all terminations were
`max_steps_with_unresolved_plan`.  No unsupported final claim was emitted.

Tool sequences were:

| Rollout | Sequence |
| --- | --- |
| 1 | Wikipedia, Wikipedia, Wikipedia |
| 2 | Wikipedia, Wikipedia, Web RAG |
| 3 | Wikipedia, Wikipedia, Web RAG |
| 4 | Wikipedia, Wikipedia, Wikipedia |

Known URL deep-read affordances were present at step 3 in 4/4 rollouts, and
2/4 elected the affordance.  This is an available evidence-driven option, not
a diversity requirement.

Cache telemetry reports 11 actual HTTP requests, 8 shared-cache writes, 7
shared-cache hits, and three throttle waits (1.673 s combined).  It reports
zero telemetry-level HTTP 429, retry, DeepSeek, or OpenAI calls.  Three
independent initial Wikipedia reads timed out; those failures were not cached.
No identical in-flight key happened in this small run, so live
`singleflight_wait_count` was zero; its owner/waiter behavior is covered by
the focused unit test rather than overstated as live evidence.

For context, the pre-cache revision run reported 3 HTTP 429 and the older
stable-step run reported 21 HTTP 429 / 15 retries.  The configurations are
not a controlled retrieval-quality comparison, but this fresh-cache run did
eliminate recorded 429s while preserving isolated rollout state.

The maximum sampled GPU use was 20,097 MiB.  The normal-completion cleanup
reported `outstanding=0`, `drained=true`, `reset_prefix_cache complete`, then
sleep; there were no CUDA illegal-memory, OOM, deadlock, Ray-worker, or
prefix-cache-free-block markers.  The GPU was 0 MiB at final inspection.

## Hypotheses

- The remaining failures are primarily retrieval/query/evidence-quality
  failures, not a plan-coverage or unsafe-finalization failure: the revised
  plan correctly exposed the prerequisite but did not obtain evidence for it.
- The three initial read timeouts are external endpoint variability.  They
  should not be cached as evidence, so they may still reduce a small rollout's
  useful observations.
- Broader runs with repeated simultaneous queries should exercise live
  singleflight waiting more often; this bounded run demonstrated shared cache
  reuse but not that particular contention path.

## Conclusions

The historical reward-1 trace is invalid as grounded evidence.  The new
coverage contract, live requirement map, and deterministic completion gate
prevent the same unsupported early step completion without judging answer
semantics or changing the reward function.  The run-scoped cache/throttle
reduced the observed MediaWiki 429 failure mode and did not share agent state:
Planner, Memory, Verifier, plan state, final answer, and reward remain per
rollout.

This one-group result is structural safety evidence, not an accuracy result
and not a basis for training.  Do not run additional samples or training from
this task without approval.

## Verification

- Focused unit suite: 39 tests passed (`test_hierarchical_planning`,
  `test_tool_priority_guidance`, `test_followup_github_audit_fixes`, and
  `test_reward_scorer`).
- `py_compile`, `bash -n`, JSON parsing, `git diff --check`, and scoped secret
  scan passed before commit.
- Raw rollout data, raw logs, temporary cache, and locks remain local and are
  not tracked.
