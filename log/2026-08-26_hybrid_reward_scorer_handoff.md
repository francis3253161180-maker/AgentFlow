# Hybrid reward scorer handoff

Implementation/test commit: `0858756a88583f212c7c67c7a320b362457bfc33`  
Branch: `experiment/flow-grpo-3b-lora`  
Scope: offline scorer refactor only; no GPU training, model, rollout.n, Flow-GRPO algorithm, or training budget was changed.

## Observed facts

- The production reward path is `train/rollout.py::eval` → `train/utils.py::compute_score` → binary `bool`, then `1.0` or `0.0`.
- The saved 40-rollout audit for main experiment `20260825_231408` had pre-hybrid TP/TN/FP/FN = `16/3/3/18`, with 19 positive and 21 negative saved rewards. Its prior deterministic fix matched the reviewed labels, but the web-constructed unseen cases exposed discourse false positives.
- The current branch was clean at the start of this task, and no training process was running or started during this task.
- The environment exposes a `DEEPSEEK_API_KEY` variable name without its value being read or written. The current Python environment cannot load the existing DeepSeek client: the import fails on the missing `aiohttp` dependency (and `openai` is also unavailable to the current interpreter). Therefore no live provider verdict was used in the offline results.

### Hypotheses

- Routing only high-confidence structured cases locally and sending unresolved discourse to a judge should reduce the specific unseen FP modes, but offline mock results cannot establish DeepSeek semantic accuracy or production latency.
- A conservative `0` on provider failure will trade some FN for lower-risk FP, which is appropriate for RL reward generation.

### Conclusions supported by this handoff

- The fixed-string/any-shared-number failure mode is no longer the production route: unresolved open answers are triaged instead of being promoted by phrase occurrence alone, and dates/numbers are compared as complete values.
- The implementation is ready for independent review, but it is not evidence that a live DeepSeek judge is calibrated until the provider runtime is repaired and a small live sanity check is reviewed.

## Design

- `deterministic_decision()` is tri-state: high-confidence `True`, high-confidence `False`, or `None` for semantic fallback.
- Local `True`/`False` is limited to normalized exact/final-answer matches, complete structured date/number comparisons, explicit Yes/No, and SymPy-backed math equivalence/mismatch. Multiple dates/numbers, self-correction, rejection, conflicting candidates, and unrestricted entity prose route to the judge.
- `<answer>...</answer>`, final-answer/conclusion/result markers, and boxed math are treated as final candidates. Earlier reasoning is ignored when an explicit final candidate is unambiguous.
- The DeepSeek prompt includes the complete `question`, `groundtruth`, and `answer_extracted`. It explicitly asks for the final semantic answer, not substring occurrence, and calls out denial, correction, replacement, and alternatives.
- Judge output is accepted only after strict JSON/Pydantic validation of a boolean `true_false`. Temperature is fixed at `0`; provider/model/base URL/key are read from environment/project client configuration (`DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` or `AGENTFLOW_REWARD_JUDGE_MODEL`, `DEEPSEEK_BASE_URL`/`DEEPSEEK_API_BASE`, and timeout).
- Cache keys are SHA-256 over the canonical `(question, groundtruth, answer)` tuple. Cache files contain only schema version, hash, and binary verdict; per-key file locking prevents concurrent cache misses from issuing duplicate calls for the same tuple.
- Provider failure, timeout, unavailable configuration, or malformed output is recorded as a typed failure and returns `False`; no failed result is cached as a successful judge verdict. The rollout-facing score remains strictly `0.0/1.0`.

## Code changes

- `train/utils.py`: removed the old GPT-specific scorer path; added tri-state high-confidence deterministic routing while retaining math normalization/equivalence.
- `train/reward_judge.py`: added DeepSeek adapter, strict response parser, cache, per-key lock, conservative failure path, and routing statistics.
- `agentflow/agentflow/engine/deepseek.py`: parameterized API key, base URL, and timeout while preserving the existing OpenAI-compatible client.
- `test/test_reward_scorer.py`: added mock-judge routing, parser, cache, failure, math, date, number, Yes/No, negation, and regression tests.
- `test/hybrid_reward_cases.py`: added 26 independently authored adversarial cases.
- `scripts/hybrid_reward_audit_20260826.py`: reproducible seen/synthetic offline audit plus optional bounded live-check mode.
- `log/2026-08-26_hybrid_reward_audit_results.json`: generated offline metrics and route-level evidence. It contains no prompts or secrets.

No dataset id, saved answer string, or one-off benchmark sample is referenced by production scorer logic.

## Tests

Commands and results:

- `PYTHONPATH=. python -m unittest -v test.test_reward_scorer`: 9 tests passed.
- `python -m py_compile train/utils.py train/reward_judge.py scripts/hybrid_reward_audit_20260826.py agentflow/agentflow/engine/deepseek.py`: passed.
- `git diff --check`: passed before commit.
- `PYTHONPATH=. python scripts/hybrid_reward_audit_20260826.py --output log/2026-08-26_hybrid_reward_audit_results.json --live-count 4`: offline audit completed; live mode reported unavailable before making a call.
- Secret scan found no API key/token value in changed source, tests, report, or result JSON. Environment variable names are referenced only as configuration names.

## Seen regression

The audit oracle is the prior independent semantic review stored in `log/2026-08-26_reward_scorer_fix_results.json`; it is used only to drive a mock judge and is not production logic.

| split | count | TP | TN | FP | FN | FN rate | FP rate | reward 1 / 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| saved reward before hybrid | 40 | 16 | 3 | 3 | 18 | 52.94% | 50.00% | 19 / 21 |
| hybrid overall, mock judge | 40 | 34 | 6 | 0 | 0 | 0.00% | 0.00% | 34 / 6 |
| NQ | 20 | 15 | 5 | 0 | 0 | 0.00% | 0.00% | 15 / 5 |
| mathhard | 20 | 19 | 1 | 0 | 0 | 0.00% | 0.00% | 19 / 1 |

Routing on the 40 seen rows: 20 deterministic hits (50.00%), 20 judge fallbacks (50.00%), 18 mock judge calls, and 2 cache hits. These are routing/regression numbers with an injected oracle, not live API quality measurements.

## Unseen adversarial results

The independent set contains 26 cases: later correction to wrong/right, Yes/No self-correction, target date mentioned then denied, multiple candidates, entity rejection, “near X but not in X”, “thought X, actually Y”, final-answer precedence, and local math/number/date cases.

| count | TP | TN | FP | FN | accuracy | FN rate | FP rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 26 | 15 | 11 | 0 | 0 | 100.00% | 0.00% | 0.00% |

Routing: 12/26 deterministic (46.15%), 14/26 judge fallback (53.85%), 14 mock calls, and zero expected-route mismatches. The three previously observed web FP patterns all route to judge: conflicting dates, ambiguous Yes/No, and rejected `Grant Park` mention.

This 100% figure means the hybrid routing harness returned the labels supplied by its deterministic mock judge; it is not a claim that DeepSeek will be 100% accurate. The test’s purpose is to prove routing and conservative control flow without network dependence.

## DeepSeek live check (if available)

Requested command: `--live-count 4`. Status: not available; calls `0`; successful verdicts `0`; average latency not available; model not recorded because the client could not be constructed. The current interpreter failed while loading the existing client due to `ModuleNotFoundError` for `aiohttp` (and does not have the `openai` package available). No key value, request payload, or provider response was logged. No dependency installation or training run was attempted.

## Cost-latency analysis

- Seen routing would have sent 20 rows to semantic fallback; two repeated tuples were served from cache, so the offline harness made 18 logical judge calls. Synthetic routing would send 14/26 rows.
- Seen judge-route input size was 13,892 characters, approximately 3,473 input tokens at a rough 4-characters/token estimate; synthetic was 1,762 characters, approximately 441 tokens. These are estimates, not provider billing records.
- Judge requests use `max_tokens=256`, temperature `0`, and the full prompt. Actual cost requires the configured DeepSeek model’s current price and actual tokenizer, neither of which was available from a live call.
- For rollout throughput planning, use `paid_calls ≈ unique_uncertain_tuples` and measure provider latency in a bounded live sanity run. With the seen routing pattern, the first-pass paid-call fraction is approximately 18/40 (45%) after cache reuse; deterministic math/structured cases add no provider latency.

## Remaining risks

- The semantic oracle in offline tests is a mock; live DeepSeek can still produce FP/FN, especially on aliases, ambiguous questions, and long tool traces. FP remains the higher-risk error for RL.
- If provider dependencies/configuration are unavailable during rollout, uncertain answers conservatively receive `0`, which can increase FN and reduce reward density. This is safer than accepting an unverified positive.
- Explicit final-answer markers are trusted by design; malformed or adversarial markers may still deserve live calibration. SymPy handling is intentionally bounded and may defer some valid math to the judge.
- Per-key locking prevents duplicate concurrent cache misses for one tuple, but provider outages and malformed responses are not cached, so repeated retries after an outage remain possible.
- The old 40-row regression is a seen regression only and cannot prove generalization.

## Recommendation

Do not start pre-validation or any training from this handoff. First have the independent reviewer inspect commit `0858756a88583f212c7c67c7a320b362457bfc33`, the 26-case routing evidence, and the conservative failure behavior. If approved, repair/verify the existing AgentFlow Python client environment and run only the bounded live judge sanity check; then review FP/FN manually before authorizing a controlled experiment. Keep the scorer binary and prioritize reducing semantic-judge FP over increasing recall.
