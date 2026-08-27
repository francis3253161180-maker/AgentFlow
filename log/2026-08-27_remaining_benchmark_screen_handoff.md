# Remaining AgentFlow benchmark screen handoff

## Scope and protocol

This was intended as a first-stage, rollout-only mixed-ratio screen for the seven AgentFlow-paper benchmarks not covered by the completed probe. No optimizer update, backward pass, parameter update, checkpoint, validation run, or variance-aware sampling was started. Benchmark examples are evaluation/probe examples only and must not enter a future formal training pool.

The intended common protocol was Qwen2.5-3B-Instruct at `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`, the current LoRA rank 8 / alpha 16 configuration, the current AgentFlow toolchain, temperature 0.7, rollout.n=4, max prompt/response lengths 1280/384, tool steps 2, `trainer.val_only=true`, `trainer.save_freq=0`, and the patched safe vLLM cleanup lifecycle. Sampling used seed 20260827 and the fixed manifest in `log/2026-08-27_remaining_benchmark_screen_sample_manifest.json`.

## Observed facts

### Authoritative benchmark inventory

The AgentFlow project page and paper list ten benchmarks: Bamboogle, 2Wiki, HotpotQA, and Musique for knowledge-intensive search; GAIA using its textual split; AIME24, AMC23, and GameOf24 for mathematics; and GPQA and MedQA for scientific reasoning. See the [AgentFlow project benchmark description](https://agentflow.stanford.edu/) and [AgentFlow paper](https://arxiv.org/html/2510.05592v2).

The repository contains local fixtures for all ten names. The seven remaining local files were present before this run, so no benchmark download was performed:

| Dataset | Local rows | Native answer shape | Local split/index evidence |
|---|---:|---|---|
| HotpotQA | 100 | list | wrapper indices 0--99; upstream release not encoded |
| Musique | 200 | string | wrapper indices 0--99; upstream release not encoded |
| GAIA | 127 | string | every local row has `split=validation`; wrapper indices 0--126 |
| AIME24 | 30 | integer | full local 30-row fixture; wrapper indices 0--29 |
| GameOf24 | 300 | list | wrapper indices 0--99 of the local 300-row fixture |
| GPQA | 300 | string | wrapper indices 0--99 of the local 300-row fixture |
| MedQA | 300 | string | wrapper indices 0--99 of the local 300-row fixture |

The local files are repository fixtures at the observed repository data revision (`b940064` for these paths). Several upstream version/split identities are not serialized in the files; the manifest records that limitation instead of guessing an official release. The preparation step found no structurally invalid rows. Native list/dict answers were encoded as compact JSON text for the AgentFlow parquet adapter; scalar answers were stringified.

### Fixed sampling

Each remaining dataset was answer-content-independently sampled with `random.Random(20260827).sample` over structurally valid rows, then source indices were sorted. Ten rows were selected for each of the seven remaining datasets. The manifest stores source row indices, pids, source hashes, selected question/ground-truth hashes, and generated parquet hashes; it does not store raw question/answer text.

### Execution status and safety

- HotpotQA: complete, 40/40 valid rollouts, retry 0, cleanup drained.
- Musique: complete, 40/40 valid rollouts, retry 0, cleanup drained.
- GAIA: complete, 40/40 valid rollouts, retry 0, cleanup drained.
- AIME24: task queue completed but only 29 valid out of 46 attempts; the remaining failures repeatedly hit DeepSeek HTTP 402 `Insufficient Balance` through the normal tool/scorer path. Cleanup drained successfully. This is not a complete comparable screen.
- GameOf24: started, but was safely stopped after persistent DeepSeek HTTP 402 failures; 9 attempts and 8 saved reward rows were observed, with only two complete n=4 groups. No valid complete screen was obtained and no cleanup marker was emitted before the external interrupt.
- GPQA and MedQA: not started after the persistent provider failure was established; no API or GPU request was made for either.

The initial HotpotQA launch failed before rollout because the runner referenced a stale temporary parquet directory. This was corrected once in the generic runner; the subsequent HotpotQA run completed normally. No CUDA/Ray/vLLM lifecycle error caused a stop. No OOM, illegal memory access, prefix-cache reset failure, `drained=False`, Ray worker death, or deadlock was observed in the actual completed lifecycles. Raw logs and rollout data remain local and untracked.

## Hypotheses

The HTTP 402 failures explain the AIME24 partial validity and the GameOf24 interruption; they cannot be interpreted as benchmark difficulty, reward sparsity, or scorer quality. AIME24's provisional reward vector statistics are therefore diagnostic only. GameOf24's two observed all-zero groups are not evidence of a benchmark-level all-zero distribution.

For the three complete new screens, the observed differences are consistent with task-format difficulty under the exact 3B/tool configuration: GAIA produced almost exclusively all-zero groups, HotpotQA was mostly all-one, and Musique was closer to a useful mixed regime. This is a screening observation, not a causal claim about the benchmarks or the paper's 7B results.

## Code and artifact changes

- Extended `scripts/prepare_benchmark_difficulty_probe_20260827.py` with explicit source-reference and split metadata while retaining answer-content-independent sampling.
- Extended `scripts/run_benchmark_difficulty_probe_20260827.sh` for the seven remaining names, ten-prompt expected runs, and safe archival of completed-but-partial validity results. It retains temperature 0.7, rollout.n=4, val-only mode, no checkpoint, and the existing cleanup settings.
- Extended `scripts/aggregate_benchmark_difficulty_probe_20260827.py` to merge prior results into a unified ten-benchmark table, preserve source metadata, and represent partial/not-run datasets explicitly rather than failing or inventing complete metrics.
- Tracked artifacts are this report, `log/2026-08-27_remaining_benchmark_screen_results.json`, and `log/2026-08-27_remaining_benchmark_screen_sample_manifest.json`. No raw rollout, cache, generated parquet, or secret was added.

## Unified ten-benchmark comparison

The prior three probes used 20 prompts each; this stage planned 10 prompts per remaining dataset. `mixed` means 1/4, 2/4, or 3/4. Bins for complete rows use the raw hybrid scorer. A dagger marks a partial/non-comparable result; its bin metrics use only the available complete n=4 vectors and must not be ranked as a normal screen.

| Dataset | Stage/status | Planned prompts | Valid rollouts | Reward mean | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | Mixed | Unique answers | Exact dup. | Unique paths |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bamboogle | prior complete | 20 | 80 | 0.7000 | 4 | 2 | 1 | 0 | 13 | 15% | 4.00 | 0.000 | 1.25 |
| AMC23 | prior complete | 20 | 80 | 0.9000 | 1 | 0 | 1 | 2 | 16 | 15% | 1.45 | 0.638 | 1.60 |
| 2Wiki | prior complete | 20 | 80 | 0.6125 | 3 | 3 | 4 | 2 | 8 | 45% | 3.55 | 0.113 | 1.75 |
| HotpotQA | new complete | 10 | 40 | 0.7250 | 2 | 1 | 0 | 0 | 7 | 10% | 4.00 | 0.000 | 1.40 |
| Musique | new complete | 10 | 40 | 0.2750 | 6 | 1 | 1 | 0 | 2 | 20% | 3.50 | 0.125 | 1.70 |
| GAIA textual fixture | new complete | 10 | 40 | 0.1000 | 9 | 0 | 0 | 0 | 1 | 0% | 3.60 | 0.100 | 1.90 |
| AIME24 | new partial† | 10 | 29/40 valid; 46 attempts | 0.4500† | 4 | 1 | 1 | 1 | 3 | 30%† | 2.10 | 0.500 | 1.43 |
| GameOf24 | new interrupted† | 10 | 8 saved rows | 0.0000† | 2 | 0 | 0 | 0 | 0 | 0%† | 1.00 | 0.750 | n/a |
| GPQA | new not run | 10 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| MedQA | new not run | 10 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

The unified JSON records 130 planned prompts across the ten rows, 110 observed prompt groups, and 397 valid rollouts. The two not-run rows contribute no observed metrics. The exact per-dataset data, group vectors, routing, runtime, GPU, and cleanup fields are in the results JSON.

## Complete new-screen results and routing

| Dataset | Runtime | GPU peak | Routing | API/parse errors | Cleanup |
|---|---:|---:|---|---:|---|
| HotpotQA | 10.0 min | 20,312 MiB | deterministic 18; judge 22 | 0 | drained true; outstanding before 0 |
| Musique | 13.1 min | 20,270 MiB | deterministic 4; judge 32; judge cache 4 | 0 | drained true; outstanding before 0 |
| GAIA | 17.2 min | 20,274 MiB | deterministic 8; judge 30; judge cache 2 | 0 | drained true; outstanding before 0 |
| AIME24† | 26.5 min | 20,302 MiB | deterministic 26; judge 2; conservative fallback 18 | 18 `RetryError` events from provider failure | drained true; early-completion cleanup |
| GameOf24† | interrupted | 20,204 MiB observed | conservative fallback 9 | 9 `RetryError` events from provider failure | no marker before interrupt; no lifecycle error observed |

No extra audit/judge calls were made. The route counts are normal production scorer telemetry from the attempted probe paths. The 402 provider failure is not a dataset-level score and makes judge-dependent rows unusable for the intended comparison.

## Conclusions

1. The complete new evidence does not support transferring the paper's 7B accuracy claims to this exact 3B setup. HotpotQA is too close to all-one (`mixed=10%`), GAIA is too close to all-zero (`mixed=0%`), and Musique is borderline secondary (`mixed=20%`, reward mean 0.275).
2. The prior 2Wiki result remains the strongest observed candidate for a useful binary-GRPO regime (`reward mean=0.6125`, `mixed=45%`). Bamboogle and AMC23 remain deprioritized by the same heuristic used previously.
3. AIME24's provisional `mixed=30%` and reward mean 0.45 look promising numerically, but the sample is invalid for ranking because only 29/40 rollouts were valid and 18 scorer events failed with provider `RetryError`. GameOf24 has no usable screen result.
4. There is not enough evidence to rank GPQA or MedQA. Their local data and fixed manifests are prepared, but running them while DeepSeek is returning HTTP 402 would only measure provider failure.

## Recommendation for stage 2

Do not start stage 2 yet. First restore/verify the normal DeepSeek provider balance and rerun the incomplete AIME24/GameOf24 screens under the same fixed manifest/protocol; then run GPQA and MedQA. Once provider health is confirmed, the conditional stage-2 shortlist is:

- 2Wiki, as the current strong candidate and a useful replication/extension target;
- Musique, as the only complete remaining screen in the secondary band;
- AIME24 only if a clean rerun reaches 40/40 valid, because its current number is not evidence.

For any later training experiment, keep all sampled benchmark/evaluation examples out of training. Construct a separate, non-overlapping training pool from an allowed official train split or an explicitly analogous source, and verify content-hash and identifier non-overlap before use. No stage-2 probe, variance-aware sampling, or formal Flow-GRPO training was started here.

## Verification

The final offline checks included JSON consistency, Python compilation for the preparation and aggregation scripts, `bash -n` for the runner, `git diff --check`, and a scoped scan for likely secret literals. The existing related scorer/cleanup unit suite had already passed 14 tests in the prior probe commit; pytest was not installed and was not needed for this verification. GPU state after stopping was approximately 2 MiB used with no compute application, and no Ray/vLLM/AgentFlow process remained.
