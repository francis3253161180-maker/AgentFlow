# Benchmark difficulty probe handoff: Bamboogle, AMC23, and 2Wiki

Probe date: 2026-08-27
Probe source revision: `9447d83` (`experiment/flow-grpo-3b-lora`)
Delivery: the commit containing this handoff and the accompanying small artifacts

## Observed facts

### Scope and protocol

This was a rollout-only difficulty probe. The existing `trainer.val_only` harness was used solely to queue training-mode `n=4` rollouts; there was no separate pre/post validation pass. It did not run backward, `optimizer.step`, parameter updates, or checkpoint writes. Each benchmark used its own process/Ray/vLLM lifecycle and the same current AgentFlow setup:

- model: `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`
- LoRA: rank 8, alpha 16
- one GPU, FSDP2/async vLLM configuration from `train/config_5090_lora_mini20.yaml`
- AgentFlow toolchain: `Base_Generator_Tool`, current configured DeepSeek tool engine, `TOOL_STEPS=2`
- rollout temperature `0.7`, rollout `n=4`
- max prompt/response lengths `1280/384`, current timeout `180s`
- `AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE=1`, `trainer.val_only=true`, `trainer.save_freq=0`, `trainer.test_freq=0`
- current hybrid scorer from `9447d83`, without dataset-specific rules

The benchmark examples were used only as probe/evaluation inputs. They are not approved training data and must not be added to any future formal training pool.

All three probes completed 20 groups x 4 rollouts = 80/80 valid rollouts, with zero retries. The runtime validation summaries were 80/80 for every dataset. No OOM, illegal memory access, Ray/vLLM worker death, deadlock, prefix-cache reset failure, or scorer API/parse error was observed. Each lifecycle emitted one cleanup marker with `drained=True`, `outstanding_before=0`, and `sleep_started=True`; there were no `drained=False` markers.

### Benchmark names, formats, splits, and evaluation conventions

The following facts are directly supported by the local repository unless marked otherwise.

| Benchmark | Direct local format/evidence | Direct paper evidence | Split/index caveat |
|---|---|---|---|
| Bamboogle | `test/bamboogle/data/data.json`; 125 rows; `pid`, `query`, `question`, `answer`, optional image/cache fields; `answer` is a list. `test/solve.py` uses `query` when present and writes `pid/query/answer`. | The AgentFlow paper describes manually constructed multi-step questions with up to four inferential steps and lists Bamboogle in the search-intensive evaluation table. | `test/bamboogle/run.sh` defaults to indices 0--124. The probe sampled 20 of the 125 structurally valid rows using the fixed seed. |
| AMC23 | `test/amc23/data/data.json`; 40 rows; `pid`, `query`, `question`, `answer`, optional image; `answer` is an integer. | The paper describes AMC23 as problems derived from the 2023 American Mathematics Competition and lists it among evaluation datasets. | `test/amc23/run.sh` contains a default 0--99 index range even though the local file has 40 rows. The probe used the observed 40-row file and sampled 20; it did not infer 100 available examples. |
| 2Wiki / 2WikiMultihopQA | `test/2wiki/data/data.json`; 200 rows; `pid`, `query`, optional image; `answer` is a string. | The paper describes 2WikiMultihopQA as combining structured Wikidata and unstructured Wikipedia and says it randomly samples 100 examples as a test set for efficiency. | `test/2wiki/run.sh` defaults to indices 0--99 while the local file contains 200 rows. The probe sampled 20 from the observed local corpus. Treat the local file/path as repository evaluation data; the canonical official split identity is not re-established by this probe. |

The paper's evaluation details report planner temperature 0.7, max turns 10, tool models based on Qwen2.5-7B-Instruct unless otherwise specified, GPT-4o judging, and three trials. The present probe intentionally does not claim to reproduce that evaluation: it uses the current Qwen2.5-3B AgentFlow tool configuration and current hybrid binary reward scorer so that comparison to the recent 3B difficulty audit is protocol-consistent. These distinctions are important.

Paper source: [AgentFlow paper, arXiv HTML v2](https://arxiv.org/html/2510.05592v2). The paper's main training description uses Search-R1 and DeepMath, while these three datasets are evaluation/probe benchmarks here; no benchmark probe row should be treated as formal training data.

### Source integrity and sampling

The source file SHA-256 values and selected row IDs/hashes are in `log/2026-08-27_benchmark_difficulty_probe_sample_manifest.json`. The fixed selection used `seed=20260827`, `random.sample` over structurally valid rows only, then sorted selected source row indices. No answer-content or difficulty criterion was used.

| Dataset | Local source rows | Structurally valid | Selected | Source SHA-256 (prefix) |
|---|---:|---:|---:|---|
| Bamboogle | 125 | 125 | 20 | `647c37dda2f2...` |
| AMC23 | 40 | 40 | 20 | `ad7c8d4930ac...` |
| 2Wiki | 200 | 200 | 20 | `eb4b045ebf8e...` |

The preparation check found no missing question/query, answer, or pid fields. This is only a structural check; it is not a human audit of semantic GT validity, staleness, or ambiguity. Bamboogle's native list answers were preserved as compact JSON text in the AgentFlow `result` column; scalar answers were stringified. That adapter is recorded in the manifest.

## Hypotheses

The probe was designed to test, rather than assume, whether benchmark accuracy reported for AgentFlow transfers to this exact 3B setup and whether the benchmark creates useful binary-GRPO group variance. Higher mixed groups with reward mean away from 0 and 1 are evidence of a more useful probe regime, but are not evidence that the benchmark is a safe training set or that the scorer is semantically perfect.

The exact-duplicate and tool-signature statistics measure surface/path diversity only. A unique answer string can still express the same semantic answer, and a common path can still produce different outcomes. No LLM similarity metric was added.

## Conclusions

### Per-dataset raw scorer results

All numbers below are raw current-hybrid-scorer results, not manually corrected labels.

| Dataset | Reward mean | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | Mixed / nonzero variance | Mean unique answers | Exact duplicate rate | Mean unique paths | Runtime | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bamboogle | 0.7000 | 4 (20%) | 2 (10%) | 1 (5%) | 0 (0%) | 13 (65%) | 3/20 (15%) | 4.00 | 0.0000 | 1.25 | 18.8 min | 20,906 MiB |
| AMC23 | 0.9000 | 1 (5%) | 0 (0%) | 1 (5%) | 2 (10%) | 16 (80%) | 3/20 (15%) | 1.45 | 0.6375 | 1.60 | 26.7 min | 20,374 MiB |
| 2Wiki | 0.6125 | 3 (15%) | 3 (15%) | 4 (20%) | 2 (10%) | 8 (40%) | 9/20 (45%) | 3.55 | 0.1125 | 1.75 | 21.2 min | 20,270 MiB |

Combined across the three equally sized probe samples: 8/60 groups were 0/4, 5/60 were 1/4, 6/60 were 2/4, 2/60 were 3/4, and 37/60 were 4/4. The raw reward mean was 177/240 = 0.7375 and raw mixed ratio was 13/60 = 21.67%. This pooled number is descriptive only because the datasets have different task formats and were not a single benchmark distribution.

### Scorer routing and errors

| Dataset | Deterministic | Judge fallback | Judge API calls | Cache hits | API/parse errors |
|---|---:|---:|---:|---:|---:|
| Bamboogle | 46/80 | 34/80 | 34 | 0 | 0 |
| AMC23 | 80/80 | 0/80 | 0 | 0 | 0 |
| 2Wiki | 15/80 | 65/80 | 60 | 5 | 0 |

The route counts come from the production `HYBRID_REWARD_EVENT` telemetry in the probe logs. The 2Wiki cache hits are normal scorer cache reuse; no extra audit/judge API call was made. The results JSON records the complete route counters, errors, cleanup markers, and local evidence paths.

### Comparison to the recent NQ+DeepMath audit

The same-protocol historical raw scorer baseline from the completed 100-prompt audit was reward mean 0.7775 and mixed 19% overall, with NQ 0.725/14% and mathhard 0.83/24%. The cleaned historical manual/quality-filtered point estimate was approximately 13.10% overall, NQ 10.00%, and mathhard 15.91%; that is a different corrected/manual metric and is not mixed directly with the new raw scorer numbers.

On the primary raw-to-raw comparison:

- 2Wiki is the clear candidate for useful binary-GRPO signal in this exact setup: reward mean 0.6125 is away from both extremes, and mixed/nonzero-variance groups are 45%, versus 19% historical overall, 14% NQ, and 24% mathhard.
- Bamboogle's 0.70 reward mean is usable but its 15% mixed ratio is not higher than the historical overall raw baseline. Its four unique answer strings per group are surface diversity without corresponding reward variance in most groups.
- AMC23 is too easy/concentrated under this setup: reward mean 0.90, 80% 4/4, 15% mixed, and 1.45 unique answers/group with 63.75% mean exact duplicate rate. It does not look like a useful main binary-GRPO training-probe distribution despite its paper-level benchmark relevance.

This is evidence that the current NQ+DeepMath random sample is not the only possible source of sparsity, but it does not prove that 2Wiki is intrinsically better in general. The sample is only 20 groups per dataset, and the scorer's binary outputs were not independently human-audited in this probe.

## Recommended training-pool strategy

Do not train on these sampled benchmark examples. If 2Wiki remains promising after an approved follow-up, use it as a held-out validation/difficulty probe and construct a non-overlapping training pool from an allowed 2Wiki training split or an explicitly licensed/analogous source. Verify train/probe IDs and content hashes do not overlap. Keep Bamboogle/AMC23/2Wiki evaluation samples out of the formal training pool. A later controlled training experiment may compare a clean NQ+DeepMath pool against a separate, non-overlapping multi-hop QA pool and optionally use variance-aware sampling based on observed pass rate/group variance; that is a proposal, not an action taken here.

No scorer change, temperature sweep, variance-aware sampler, or formal Flow-GRPO training was started. The repository is left with no active GPU/Ray/vLLM probe process.

## Reproduction and verification

Preparation:

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python \
  scripts/prepare_benchmark_difficulty_probe_20260827.py \
  --dataset bamboogle=test/bamboogle/data/data.json \
  --dataset amc23=test/amc23/data/data.json \
  --dataset 2wiki=test/2wiki/data/data.json \
  --output-parquet bamboogle=/root/autodl-tmp/tmp/benchmark_difficulty_probe_20260827/bamboogle.parquet \
  --output-parquet amc23=/root/autodl-tmp/tmp/benchmark_difficulty_probe_20260827/amc23.parquet \
  --output-parquet 2wiki=/root/autodl-tmp/tmp/benchmark_difficulty_probe_20260827/2wiki.parquet \
  --manifest log/2026-08-27_benchmark_difficulty_probe_sample_manifest.json \
  --seed 20260827 --sample-count 20
```

Each probe was run with `scripts/run_benchmark_difficulty_probe_20260827.sh {bamboogle|amc23|2wiki}`. Offline aggregation used `scripts/aggregate_benchmark_difficulty_probe_20260827.py` and the three local metadata files under `/root/autodl-tmp/tmp/benchmark_difficulty_probe_20260827/`.

Verification performed before handoff: shell syntax check, Python compilation, deterministic manifest generation, JSON aggregation, `git diff --check`, scorer unit tests, secret review, and final GPU/process check. Raw benchmark logs, generated parquet files, caches, and rollout JSON remain local/untracked.
