# Structured-output harness A/B smoke handoff

## Observed facts

- Scope was rollout-only on the frozen 8-group Game24 selection: 32 new trajectories, no backward, optimizer step, parameter update, checkpoint, external judge, or DeepSeek call.
- The old pre-harness evidence is unchanged. The new run used Qwen2.5-7B-Instruct, temperature 0.7, `n=4`, max prompt 3072, max response 1024, max model length 8192, dynamic response padding, vLLM utilization 0.50, max sequences 1, max batched tokens 1024, and the saved initial behavior snapshot.
- New run validation completed `32/32` valid rollouts. Cleanup markers reported `complete=1 drained=1` on both cleanup cycles. GPU peak was 16,982 MiB and final sampled usage was 1,224 MiB. No CUDA illegal memory access, OOM, prefix-cache reset failure, context overflow/HTTP 400, Ray worker death, or deadlock was observed.
- The first implementation attempt was stopped after vLLM 0.9.2 returned HTTP 400 because the request explicitly selected `outlines` while the server was initialized with `auto`. The minimal compatibility correction removed request-level backend selection and retained only `guided_json`; the subsequent run completed without this error.

## Code changes

- Added `agentflow/agentflow/models/structured_outputs.py`: strict Pydantic schemas, exact JSON parsing, Game24 AST/Fraction validation, candidate selection, and bounded structured retry prompt.
- Updated `agentflow/agentflow/engine/vllm.py`: Pydantic schemas are converted to vLLM 0.9.2 `extra_body.guided_json`; the unsupported request-level backend selector is not sent.
- Updated `agentflow/agentflow/models/planner.py`: Game24 final/direct output uses deterministic marked-candidate selection first, then at most one structured retry; arbitrary free text is not promoted.
- Updated `train/rollout.py`: structured answers are parsed strictly and a validated expression is passed to the existing numeric scorer inside an explicit `<answer>...</answer>` boundary. This is an interface-preserving bridge, not a reward-rule change.
- Parameterized the existing diagnostic runner for experiment name, response length, and harness flag. No normal training defaults were changed.
- Added focused unit tests and the offline audit script `scripts/audit_structured_output_harness_ab_20260828.py`.

## A/B smoke results

| metric | old pre-harness | new structured harness |
|---|---:|---:|
| groups / rollouts | 8 / 32 | 8 / 32 |
| reward 1.0 / 32 | 0 | 0 |
| reward 0.0 / 32 | 32 | 32 |
| group bin 0/4 | 8 | 8 |
| mixed groups | 0 (0%) | 0 (0%) |
| exact answer duplicate rate | 6.15% | 5.94% |
| internal request records | 260 | 320 |
| response mean / max tokens | 217.53 / 740 | 169.69 / 1024 |
| response cap hits | 0 | 4 (1.25%) |
| finish stop / length | 260 / 0 | 316 / 4 |
| context overflow / HTTP 400 | 0 / 0 | 0 / 0 |
| GPU peak | 16,992 MiB | 16,982 MiB |

Structured telemetry in the new run:

- Exact `{"expression": ...}` objects in traces: 63. Of these, 1 passed the local Game24 semantic validator; 62 failed semantic validation (`not_24`: 41, `wrong_number_multiset`: 21). These are trace-level role responses, not 63 independent final answers.
- Outer final harness: 1 guided response validated and 31 failed after the one allowed retry. There were 31 retry attempts; no schema-parse-failure marker occurred in the harness path. No deterministic candidate was selected.
- The one validated outer answer was `(12-8)*(7-1)` for `[1,7,8,12]`. The log shows it reached rollout extraction, but the old scorer saw an untagged expression and emitted `conflicting_numbers`, reward 0. The post-run bridge now preserves the explicit answer boundary for future runs; this smoke was not rerun after that code change by design.
- All 32 reward events used the local/external-disabled conservative path (`error=unavailable`, no judge fallback/cache call). The aggregate records the configured route as deterministic; the per-event telemetry is the more specific evidence for these all-`None`/uncertain cases.
- The old run emitted 147 legacy `parse response as JSON` warnings. The new run emitted 0 matching legacy warnings; the structured harness itself emitted 0 `schema_parse_failure` markers. Role outputs were accepted through the vLLM guided JSON request path during the successful run.

## Offline causal audit

The new all-zero reward result is not evidence that constrained decoding is ineffective by itself. It has two separable causes:

1. In 31/32 outer finalizations, the model did not provide a semantically valid Game24 expression after one retry. The dominant failures were wrong target value or wrong input-number multiset. This is a model/agent reasoning failure that the validator correctly rejected.
2. One final expression was locally valid but was converted to plain text before the existing scorer. Because the scorer treats an untagged multi-number expression conservatively, this caused an integration false negative. The minimal bridge fix is included and covered by a focused offline assertion, but was not live-rerun in this smoke.

Therefore, the harness did reduce the observed legacy free-form parser-warning class and verified vLLM guided-JSON compatibility, but it did not restore nonzero reward in this run. The run does not justify claiming a reward improvement.

## Tests and evidence

- Focused suite: `27 passed, 1 warning, 39 subtests passed` before the smoke; rerun after the bridge change is required in the final verification below.
- New aggregate: `/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/structured-output-harness-ab-20260828_20260828_175051_aggregate.json`.
- New evidence directory: `/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/structured-output-harness-ab-20260828_20260828_175051_trajectories`.
- New raw train/rollout log SHA256: `465d56e957e2c32428f9c7550bfcd385c24597d1a8663e56318c314b193a3eac` / `758fcd94d19dbeec535a628b4de84fa28666e9457fad50c488e9f84737ac3bf5`.
- New aggregate SHA256: `df35e0370d3346d73cd653b533a7c691e6a7e0eeb2a3b741c0b374486e930be0`.
- Snapshot preflight hash remained verified against the frozen selection and behavior snapshot. Raw evidence/logs remain local and untracked.

## Recommendation

Do not start formal training from this smoke. Before another rollout or baseline, run the focused tests again and, with approval, one narrowly scoped recheck of the same frozen 8 groups after the reward-boundary bridge. Keep the strict validator and one-retry cap; do not loosen it to manufacture reward variance. The next recheck should separately report valid-structured-output rate and reward-bridge correctness, since the current smoke proves the former is still the main bottleneck while also exposing the latter integration defect.
