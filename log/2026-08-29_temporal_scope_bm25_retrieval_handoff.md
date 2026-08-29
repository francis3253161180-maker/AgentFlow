# Temporal-scope planning and deterministic Web_RAG BM25 diagnostic — 2026-08-29

## Observed facts

- This was a bounded, rollout-only diagnostic from commit `f1b31ed`: one
  frozen MuSiQue source index `259` / benchmark id `2hop__13592_49388`,
  requested `n=4`, Qwen2.5-7B actor+LoRA as the only action policy, local
  Qwen-base query-analysis/Executor/Final, and Doubao only for evidence plan,
  coverage, and step verification. No optimizer, backward pass, checkpoint,
  reward change, sample expansion, or extra live run occurred.
- The official MuSiQue dev record was read offline only from the locally
  verified official repository revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`.
  The tracked offline diagnostic retains only hashes, counts, and lexical
  predicates; it contains no answer, support title, paragraph, or decomposition
  text, and no field is imported by AgentFlow runtime. It confirms that the two
  official decomposition questions separate a year-qualified relation from an
  unqualified requested property.
- The prior longer-budget raw trajectories had historical/era wording in later
  property steps in all four plans. That is evidence for temporal-scope
  propagation in the prior supervisor plan, not evidence that the actor caused
  it.
- The generic supervisor change makes the literal query authoritative, requires
  minimal sufficient plans, and asks coverage to flag
  `introduced_constraints` and `redundant_step_ids`. The schema forbids extra
  fields and retains the existing redacted one-revision boundary policy.
- Preflight passed before the live run with two atomic steps, plan cap `3`,
  zero introduced constraints, zero redundant steps, temperature `0`, and
  `ARK_REASONING_EFFORT=minimal`. It made two supervisor calls. Its redacted
  telemetry and request metadata are included in the result JSON.
- Web_RAG now uses deterministic BM25/IDF scoring plus bounded term-coverage,
  phrase, and proximity tie signals. It exposes BM25 score, final lexical
  score, coverage, matched numeric tokens, matched terms, phrase hits, and
  proximity span. It remains a raw known-URL extractor with no LLM calls.
  A synthetic unit test verifies that a chunk containing rare relation terms
  and exact year-like tokens outranks a long generic chunk with frequent common
  terms.
- The one authorized live run kept action budget `6`, timeout `1200s`, plan cap
  `3`, `N_WORKERS=4`, `max_num_seqs=4`, temperature `0.7`, and a fresh
  run-scoped Wikipedia cache. It was only partially retained: two of four
  logical slots were dropped after existing bounded retries returned empty
  daemon posts (`final_reward=None`, no triplet). The train log has no root
  traceback for those two slots. This is an infrastructure-observability issue,
  so the run is not a 4-way performance estimate.
- The two retained trajectories both had reward `0`; final answers used the
  grounded insufficient-evidence form, so unsupported-final-claim count was
  `0`. Both end causes were `planner_action_stagnant`, not timeout.
- In retained trajectory 1, the supervisor still put the historical qualifier
  into the later season-property step even though coverage reported no
  introduced constraint. This is a coverage-auditor false negative; it shows
  the prompt/schema change reduced but did not eliminate semantic-scope error.
- In retained trajectory 2, the plan's later property step had no historical
  qualifier. The actor retrieved direct raw evidence for the first relation;
  the deterministic provenance gate accepted it, step 1 completed, and step 2
  activated. This is the requested structural transition, not a reward success.
- Its Web_RAG deep read ranked an evidence chunk with matched exact numeric
  tokens and high BM25/coverage telemetry. Thus the BM25 path was exercised and
  produced an auditable, relevant ranking. However, on the later property
  step, the local Executor emitted an HTTP URL not present in recorded Memory;
  it received HTTP 404. The tool contract describes known-URL-only deep reads,
  but current execution does not enforce that provenance at the command
  boundary. No repair was applied after the run.
- Retrieval telemetry across retained trajectories: 5 HTTP requests, 1 shared
  cache hit, 1 singleflight wait, 0 HTTP 429, 0 HTTP retries, and zero
  OpenAI/Doubao/tool-internal LLM calls. Fixed-role totals were four supervisor
  calls and five step-verifier calls. Actor-visible verifier audits contained
  zero prohibited markers.
- GPU peak was `20,609 MiB`. Normal cleanup recorded `outstanding=0` and
  `drain_complete=1` before reset/sleep. There were no CUDA illegal-memory,
  OOM, prefix-cache, deadlock, Ray-worker-death, or fatal lifecycle markers;
  post-run GPU use was `0 MiB` / `0%` and no attributable train/Ray/vLLM
  process remained.

## Hypotheses

- The new temporal-scope contract can enable the intended two-stage evidence
  chain, as shown by one retained trajectory, but the coverage model at
  temperature zero is not by itself a reliable semantic validator of introduced
  scope. The remaining scope error is supervisor/coverage behavior, not an
  actor action-selection conclusion.
- The BM25 revision addresses the observed deep-read ranking failure mode. The
  one successful deep-read ranking supports the mechanism, but two retained
  trajectories are insufficient to estimate its general impact.
- The next blocking failure in the structurally improved trajectory is likely
  command-level known-URL provenance: Planner selected a legitimate deep-read
  role, but local Executor substituted an unrecorded URL. That is distinct from
  discovery, BM25 extraction, verifier grounding, or external endpoint quality.
- Empty daemon posts for two logical slots may be a startup/concurrency or
  daemon-reporting issue. The available logs do not establish a causal link to
  the temporal/BM25 changes.

## Conclusions

- Do not interpret `[0, 0]` as a completed n=4 reward result; the requested
  group was incomplete (`2/4` retained). It provides structural diagnostic
  evidence only.
- The temporal-scope hypothesis is supported offline and partly corrected live:
  one trajectory advanced from a correctly grounded historical relation to an
  unqualified property step. Strict grounding remained intact and prevented an
  unsupported positive.
- The current live evidence distinguishes three remaining issues: occasional
  supervisor/coverage semantic-scope false negatives, actor stagnation in one
  trajectory, and non-enforced known-URL provenance in Executor command
  generation. The latter caused the observed 404 after valid step advancement.
- The required one calibration has finished. Per the approved boundary, no
  hot-fix or second calibration was launched.

## Recommendation

Do not train from this partial result. A separately approved next task should
first make the known-URL-only contract enforceable at the Executor command
boundary without selecting an action, URL, query, or fallback for the actor;
it should also improve empty-rollout error attribution. If approved, validate
those changes with a fresh bounded run and keep the temporal-scope coverage
audit independent from gold/support data. No official MuSiQue support content
may be introduced into runtime prompts, memory, cache, or training data.

## Verification

- Offline diagnostic: `scripts/audit_musique_temporal_scope_20260829.py`
- Focused tests: `python -m unittest test.test_hierarchical_planning
  test.test_tool_priority_guidance test.test_ark_provider
  test.test_followup_github_audit_fixes test.test_reward_scorer` (`47` passed).
- Also run: `py_compile`, `bash -n`, JSON consistency, `git diff --check`,
  and scoped secret scan before commit.
