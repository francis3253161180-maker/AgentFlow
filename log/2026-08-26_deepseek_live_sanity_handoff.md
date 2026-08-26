# DeepSeek live sanity handoff

Implementation and live-artifact commit: `63bdd0ce0cf5d8685d67934e3865419a09f60113`  
Branch: `experiment/flow-grpo-3b-lora`  
Scope: real DeepSeek reward-judge calls only; no GPU training, validation, model, rollout.n, Flow-GRPO, or training budget changes.

## Observed facts

- The live check used the production `HybridRewardScorer` and the existing `ChatDeepseek` OpenAI-compatible client, not a mock judge.
- The final recorded batch completed 12 primary cases and 18 real API requests. All 12 primary verdicts matched the manually specified expected verdicts; all responses passed the strict Pydantic JSON parser.
- The final provider/model was `deepseek-v4-flash`; the request temperature was `0`.
- Three complex cases were each called three times with cache bypassed. Their raw verdict sequences were respectively `[False, False, False]`, `[True, True, True]`, and `[False, False, False]`.
- The final batch had no API error, timeout, or parse error. The temporary cache contained only SHA-256-keyed result records and was deleted at process exit.
- No training, validation, vLLM, or pre-validation process was started. The final worktree was clean after the live-check commit.

The runner required several bounded reruns while fixing its own bookkeeping and coverage selection. Four preliminary runs made 15, 15, 16, and 17 real requests respectively but failed to persist a final artifact or were superseded by a later coverage-complete run; they were not included in accuracy/latency metrics. Including them, the task made 81 provider requests in total. The final committed artifact below reflects only the last 18-request batch.

## Environment & dependency changes

- The active usable environment was `/root/autodl-tmp/conda/envs/agentflow`, Python `3.10.21`.
- Required packages were already present; no package was installed or upgraded:
  - `openai 1.75.0`
  - `aiohttp 3.14.3`
  - `tenacity 9.0.0`
  - `pydantic 2.13.4`
  - `sympy 1.14.0`
  - `python-dotenv 1.0.1`
- The base Python 3.13 environment was missing `openai`, `aiohttp`, and `tenacity`; it was not modified. The previous “missing dependency” result came from using that base interpreter rather than the AgentFlow conda environment.
- `DEEPSEEK_API_KEY`: present. Its value was never printed, read into a report, or committed.
- `DEEPSEEK_BASE_URL` and `DEEPSEEK_API_BASE`: missing, so the existing client defaulted to its configured DeepSeek endpoint.
- `DEEPSEEK_MODEL` and `AGENTFLOW_REWARD_JUDGE_MODEL`: missing, so the scorer used its existing default `deepseek-v4-flash`.
- `DEEPSEEK_TIMEOUT_SECONDS`: missing, so the scorer used its default 30-second client timeout.

## Live test design

The final primary set had 12 cases: 11 independently authored synthetic cases and one open natural-language NQ rollout. It covered:

| case type | primary case ids | expected outcome |
|---|---|---|
| correct → later correction to wrong | `date_corrected_to_wrong` | false |
| wrong → later correction to correct | `date_wrong_corrected_to_right` | true |
| Yes → No self-correction | `yes_to_no_self_correction` | false |
| No → Yes self-correction | `no_to_yes_self_correction` | true |
| multiple candidates | `multiple_candidate_entities` | true |
| near X but not in X | `near_but_not_in_entity` | false |
| thought X, actually Y | `thought_x_actually_y` | true |
| entity mentioned but rejected | `entity_mentioned_then_rejected` | false |
| final marker over earlier reasoning | `final_marker_overrides_earlier_reasoning` | true, deterministic route |
| math deterministic path | `fraction_local_proof` | true, deterministic route |
| numeric deterministic mismatch | `integer_local_mismatch` | false, deterministic route |
| real rollout | `real_rollout_nq_open_answer` | true, judge route |

For three complex judge cases, three direct uncached calls measured raw provider stability. Each was then passed through the production scorer once with an empty temporary cache and once again to verify a real cache hit. The remaining cases were passed through the production hybrid scorer once.

## Live results

| metric | final result |
|---|---:|
| primary cases | 12 |
| primary successes | 12 |
| primary errors | 0 |
| accuracy against expected labels | 100% (12/12) |
| real API requests | 18 |
| raw repeat cases | 3 |
| raw repeat verdict consistency | 100% (3/3 cases) |
| production deterministic routes | 3/15 scorer invocations |
| production judge fallbacks | 12/15 scorer invocations |
| production cache hits | 3 |
| API/parse failures | 0 |

The 100% accuracy is a small sanity-check result against manually specified labels, not a generalization claim. The final marker, fraction, and integer mismatch cases correctly stayed local; all ambiguous discourse cases and the real NQ open answer reached the judge.

## Consistency & latency

- Raw uncached repeat verdicts were stable for all three cases, including both positive and negative verdicts.
- Raw/uncached latency: average `0.957 s`, median `0.888 s`, P95 `1.304 s` over 18 measured calls.
- Cached latency: average `0.000658 s`, median `0.000649 s`, P95 `0.000693 s` over 3 cache hits.
- The live request path explicitly passed `temperature=0`, `max_tokens=256`, and `top_p=1.0`. All 18 returned objects were accepted only after `SemanticJudgeVerdict` Pydantic validation.
- Cache validation: 9 JSON result records, every key had 64 hex characters, `cache_contains_raw_input=false`, and the cache directory was temporary. No runtime cache was committed.
- The final sample’s 80% judge-fallback rate is intentionally higher than the earlier seen-40 50% routing rate because this batch over-sampled difficult semantic cases. It should not be used as a training-throughput estimate by itself.

## Failure-path checks

The controlled failure tests were run without touching the valid key or provider:

- `TimeoutError` from the injected judge → `conservative_fallback`, score `False`.
- `RuntimeError` representing an HTTP/provider failure → `conservative_fallback`, score `False`.
- Invalid non-JSON judge text → strict parser `ValueError`, `conservative_fallback`, score `False`.

The three cases were added to `test/test_reward_scorer.py`; all 9 unit tests passed in the AgentFlow conda environment. Every failure result remained representable as rollout reward `0.0` or `1.0` through the existing `eval` wrapper. No live failure occurred in the final batch.

## Remaining uncertainties

- Twelve live primary cases are too few to estimate semantic accuracy or production-scale cost. The expected labels were manually supplied and do not constitute an independent benchmark.
- The raw consistency probe used three cases, so it cannot rule out rare nondeterminism on other prompts or provider-side changes.
- The final batch used the default `deepseek-v4-flash` because no model override was configured. A future model/config change requires a new sanity check.
- The final batch’s latency was measured from this server and this provider response condition; it does not establish latency under concurrent rollout workers or rate limiting.
- The preliminary runner reruns incurred extra API cost, as recorded under Observed facts. They do not affect the committed final metrics but should be counted in operational cost review.
- Provider failure remains conservatively negative by design. This protects against RL false positives but can increase false negatives when the API or dependencies are unavailable.

## Recommendation

The real production chain is technically verified for a bounded sanity set: structured responses parse, temperature is deterministic, ambiguous cases route to DeepSeek, cache hits are fast and input-free, and failure paths remain conservative binary rewards. Do not start pre-validation or training from this report. Have the independent reviewer inspect the final JSON and commit; only after approval should a separately authorized controlled experiment use this scorer, with live API error rate, judge-fallback rate, FP/FN samples, and spend monitored explicitly.
