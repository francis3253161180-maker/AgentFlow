# Doubao supervisor calibration with a longer action budget — 2026-08-29

## Observed facts

- Scope was one frozen MuSiQue question only: source index `259`, benchmark id
  `2hop__13592_49388`, four logical rollouts, no optimizer/backward/checkpoint.
  The raw local rollout pack remains untracked. The exact source and question
  hashes are in `2026-08-29_doubao_calibration_longer_budget_results.json`.
- Phase-0 supervisor-only preflight passed before the live run. It used
  `doubao-seed-2-0-lite-260428`, `temperature=0`,
  `ARK_REASONING_EFFORT=minimal`, an independent plan cap of three, and made
  three calls: one initial high-level plan, one permitted supervisor-only
  structural self-revision, and one coverage audit. The initial plan's
  redacted telemetry marked season-date strings as arithmetic; the revision
  was structurally clean. The subsequent generic season-range exclusion keeps
  `YYYY-YY` / `YYYY-YYYY` date ranges out of this diagnostic without relaxing
  tool, URL, command/query, answer, or real arithmetic checks.
- The live calibration used action budget `6`, agent timeout `1200s`, plan cap
  `3`, four workers, four vLLM sequences, a fresh run-scoped Wikipedia cache,
  and the existing Qwen context limits. All four final rollout slots were
  valid. Four initial daemon posts had no triplet/reward and were retried once;
  the logs contain no accompanying exception/trace, and the retried slots all
  completed.
- Role routing was exactly: Qwen actor+LoRA for `planner_main`; local Qwen-base
  for query analysis, Executor, and Final; Doubao only for high-level evidence
  planning, plan coverage, and step verification. All observed Ark metadata
  records `reasoning_effort=minimal` and `temperature=0`. Search/tool telemetry
  reports zero internal Doubao/OpenAI calls; Executor/tools/Final did not make
  a Doubao call.
- All four high-level plans passed coverage and retained a stable three-step
  requirement-to-step mapping. The fixed-role totals were eight supervisor
  calls and 17 step-verifier calls. Redacted supervisor and actor-visible
  verifier audits are persisted as hashes, lengths, schema status, and marker
  categories only. Actor-visible verifier audits had zero prohibited markers.
- Reward vector was `[0, 0, 0, 0]`. Every final output used the grounded
  insufficient-evidence form, so unsupported-final-claim count was `0`.
  No step was marked completed: the deterministic provenance gate correctly
  rejected incomplete support for step 1 rather than accepting a plausible
  La Liga inference.
- Two rollouts used both Wikipedia discovery and known-URL Web_RAG; the other
  two used only Wikipedia. The ending causes were two `max_steps_with_unresolved_plan`
  and two `planner_action_stagnant`. Thus the longer budget allowed more
  exploration but did not resolve the required provenance gap.
- Shared retrieval operated without rate-limit failure: 26 HTTP requests,
  11 shared-cache hits, two singleflight waits (3.82s total), 12 throttle
  waits (7.13s total), zero HTTP 429s, and zero retrieval retries.
- Observed peak GPU memory was 20,195 MiB. Normal cleanup recorded
  `outstanding=0`, `drain_complete=1`, then prefix-cache reset and sleep;
  no illegal-memory-access, OOM, deadlock, prefix-cache-block, Ray-worker, or
  other fatal lifecycle marker occurred. Post-run process inspection found no
  attributable train/Ray/vLLM worker and GPU usage was `0 MiB` / `0%`.

## Hypotheses

- The remaining failure is primarily a retrieval/provenance limitation on this
  frozen question: the search evidence established that Barcelona participates
  in La Liga and exposed a relevant late-1940s snippet, but the strict current
  requirement demanded a traceable direct link between the two title years and
  the league. The system therefore stayed on step 1 instead of advancing.
- The six-action budget removed the prior three-action budget confounder for
  two trajectories, but cannot by itself create missing historical evidence.
  The repeated semantically similar discovery actions indicate that the actor
  still needs stronger observation-conditioned query refinement; this is not
  evidence that Doubao chose actions, because the persisted actor action
  records remain Qwen planner outputs and the sanitized verifier state contains
  no tool/query/action text.
- This single-question calibration cannot establish task accuracy or justify
  training. It establishes bounded routing/grounding behavior only.

## Conclusions

- The supervisor is now executable under the anti-signal-theft boundary: strict
  schemas reject extra fields, telemetry is redacted, and one supervisor-only
  self-revision is available. The live run showed no action, query, URL,
  arithmetic, or final-answer channel from Doubao to `planner_main`.
- The longer action budget and separate small plan cap work as intended. They
  did not weaken the provenance gate: `0/4` is a valid grounded result, not a
  calibration crash or an unsupported positive.
- Retrieval/cache/lifecycle behavior is healthy for this bounded run. There is
  no basis to begin training or to add a second live calibration without a new
  approved change.

## Changes and verification

- Added redacted boundary telemetry and generic supervisor-only structural
  revision in `agentflow/models/role_boundaries.py` and
  `agentflow/models/planner.py`; made supervisor JSON schemas forbid unexpected
  fields in `agentflow/models/formatters.py`.
- Added a separate `AGENTFLOW_HIERARCHICAL_PLAN_MAX_STEPS` cap in `solver.py`
  and the smoke runner, plus `scripts/run_doubao_supervisor_preflight_20260829.py`.
- Sanitized the actor-visible step-verifier state while retaining raw fixed-role
  rationale only as local audit evidence. The aggregation script now records
  redacted role-routing/boundary telemetry and supports a compact handoff mode.
- Focused tests passed (45):
  `python -m unittest test.test_hierarchical_planning test.test_tool_priority_guidance test.test_ark_provider test.test_followup_github_audit_fixes test.test_reward_scorer`.
  `py_compile`, `bash -n`, JSON parsing, `git diff --check`, and scoped secret
  scan are run before commit. Mock scorer failure-path messages in the unit
  suite are not external judge calls.

## Recommendation

Do not train from this result. Preserve the current strict grounding rule and
use the compact result plus local raw pack to diagnose generic
observation-conditioned retrieval refinement in a separately approved task.
Any future change should keep Doubao confined to plan/coverage/verification,
retain Qwen as the sole action policy, and rerun a fresh bounded calibration
rather than altering this completed evidence.
