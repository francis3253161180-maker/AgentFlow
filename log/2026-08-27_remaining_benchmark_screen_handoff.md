# Remaining AgentFlow benchmark screen handoff (completed)

## Scope and protocol

This handoff completes the first-stage rollout-only screen for the seven remaining AgentFlow-paper benchmarks after the DeepSeek quota was restored. No formal training, backward pass, optimizer step, parameter update, checkpoint write, variance-aware sampling, or new 100-prompt audit was started.

The common protocol was kept fixed: Qwen2.5-3B-Instruct at `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`, the existing LoRA rank 8 / alpha 16 configuration, the existing AgentFlow toolchain and safe vLLM cleanup lifecycle, temperature 0.7, rollout.n=4, max prompt/response lengths 1280/384, tool steps 2, `trainer.val_only=true`, `trainer.save_freq=0`, and zero optimizer steps. The seven datasets used the fixed manifest `log/2026-08-27_remaining_benchmark_screen_sample_manifest.json` with selection seed 20260827 and ten prompts per dataset. The scorer was unchanged and used only its normal deterministic/judge paths.

The final AIME24, GameOf24, GPQA, and MedQA runs used the independent run tag `quota_refill1`; this preserved the earlier quota-blocked raw logs and rollout directories. Only the new complete runs are used for the final metrics below.

## Observed facts

### Authoritative benchmark inventory and data preparation

The AgentFlow project and paper identify these ten benchmark names: Bamboogle, 2Wiki, HotpotQA, Musique, GAIA, AIME24, AMC23, GameOf24, GPQA, and MedQA. The project groups them as knowledge-intensive search (Bamboogle, 2Wiki, HotpotQA, Musique, GAIA), mathematics (AIME24, AMC23, GameOf24), and scientific reasoning (GPQA, MedQA). References: [AgentFlow project](https://agentflow.stanford.edu/) and [AgentFlow paper](https://arxiv.org/html/2510.05592v2).

All seven remaining inputs were already available as local repository fixtures, so no public benchmark download was performed. The manifest records source paths, source SHA-256 hashes, source references, split evidence, selected source indices, and generated parquet hashes. The repository fixture revision observed for these paths was `b940064`. Where a fixture does not serialize an upstream release or official split identity, that limitation is recorded rather than inferred away.

| Dataset | Local fixture / split evidence | Selected prompts |
|---|---|---:|
| HotpotQA | `test/hotpotqa/data/data.json`; wrapper indices 0--99; upstream release not encoded | 10 |
| Musique | `test/musique/data/data.json`; wrapper indices 0--99; upstream release not encoded | 10 |
| GAIA | `test/gaia/data/data.json`; local rows explicitly marked validation; wrapper indices 0--126 | 10 |
| AIME24 | `test/aime24/data/data.json`; full local 30-row fixture | 10 |
| GameOf24 | `test/gameof24/data/data.json`; repository evaluation fixture; wrapper indices 0--99 of 300 | 10 |
| GPQA | `test/gpqa/data/data.json`; repository evaluation fixture; wrapper indices 0--99 of 300 | 10 |
| MedQA | `test/medqa/data/data.json`; repository evaluation fixture; wrapper indices 0--99 of 300 | 10 |

Sampling was answer-content-independent: structurally valid source rows were sampled with `random.Random(20260827).sample`, then selected source indices were sorted. These benchmark/evaluation examples are probe-only and must not be placed in a future formal training pool.

### Completion and safety

All seven remaining datasets completed with 40 valid rollouts from ten prompt groups. GPQA's validation summary recorded 42 attempts and 40 completed valid rollouts, corresponding to two transient empty-rollout retries; its final ten n=4 groups are complete. The other six runs had no final retry count. No run produced an API/parse error after the quota refill.

Every final run emitted a safe cleanup marker with `drained=True`, `outstanding_before=0`, `abort_count=0`, and `abort_errors=0`. No final run emitted `drained=False`, `Failed to reset prefix cache`, `blocks are not freed yet`, CUDA illegal memory access, CUDA OOM, Ray worker death, or deadlock. No optimizer/backward/update marker was found. Raw logs and rollout data remain local and untracked.

## Unified ten-benchmark comparison

The prior Bamboogle, AMC23, and 2Wiki probes used 20 prompts each. The remaining screen used ten prompts each, so the new rows have lower group-level precision. Bins are counts of prompt groups with the indicated number of positive rewards; `mixed` is the sum of 1/4, 2/4, and 3/4 groups and equals the nonzero-variance group ratio for binary rewards.

| Dataset | Stage | Groups | Valid rollouts | Reward mean | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | Mixed | Unique answers | Exact dup. rate | Unique paths |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bamboogle | prior complete | 20 | 80 | 0.7000 | 4 | 2 | 1 | 0 | 13 | 15% | 4.00 | 0.000 | 1.25 |
| AMC23 | prior complete | 20 | 80 | 0.9000 | 1 | 0 | 1 | 2 | 16 | 15% | 1.45 | 0.638 | 1.60 |
| 2Wiki | prior complete | 20 | 80 | 0.6125 | 3 | 3 | 4 | 2 | 8 | 45% | 3.55 | 0.113 | 1.75 |
| HotpotQA | remaining complete | 10 | 40 | 0.7250 | 2 | 1 | 0 | 0 | 7 | 10% | 4.00 | 0.000 | 1.40 |
| Musique | remaining complete | 10 | 40 | 0.2750 | 6 | 1 | 1 | 0 | 2 | 20% | 3.50 | 0.125 | 1.70 |
| GAIA textual fixture | remaining complete | 10 | 40 | 0.1000 | 9 | 0 | 0 | 0 | 1 | 0% | 3.60 | 0.100 | 1.90 |
| AIME24 | remaining complete | 10 | 40 | 0.7000 | 2 | 0 | 1 | 2 | 5 | 30% | 2.10 | 0.475 | 1.50 |
| GameOf24 | remaining complete | 10 | 40 | 0.6000 | 1 | 2 | 3 | 0 | 4 | 50% | 2.60 | 0.350 | 1.50 |
| GPQA | remaining complete | 10 | 40 | 0.6500 | 2 | 2 | 0 | 0 | 6 | 20% | 2.40 | 0.400 | 1.80 |
| MedQA | remaining complete | 10 | 40 | 0.9500 | 0 | 0 | 1 | 0 | 9 | 10% | 1.40 | 0.650 | 1.40 |

The final results JSON records the full group vectors, source metadata, routing, runtime, GPU telemetry, cleanup markers, and the same table as `unified_comparison`. Across the ten rows there are 130 prompt groups and 520 valid rollouts; all ten rows are complete. The 20-prompt prior rows and ten-prompt new rows should not be treated as equal-precision estimates.

## New-screen runtime, routing, and resource evidence

| Dataset | Runtime (min) | GPU peak (MiB) | Retry count | Scorer routing | API/parse errors | Cleanup |
|---|---:|---:|---:|---|---:|---|
| HotpotQA | 10.0 | 20,312 | 0 | deterministic 18; judge 22 | 0 | drained true; outstanding 0 |
| Musique | 13.1 | 20,270 | 0 | deterministic 4; judge 32; judge cache 4 | 0 | drained true; outstanding 0 |
| GAIA | 17.2 | 20,274 | 0 | deterministic 8; judge 30; judge cache 2 | 0 | drained true; outstanding 0 |
| AIME24 | 19.7 | 20,308 | 0 | deterministic 40 | 0 | drained true; outstanding 0 |
| GameOf24 | 21.9 | 20,906 | 0 | deterministic 1; judge 25; judge cache 14 | 0 | drained true; outstanding 0 |
| GPQA | 19.6 | 20,354 | 2 transient empty-rollout retries | deterministic 20; judge 18; judge cache 2 | 0 | drained true; outstanding 0 |
| MedQA | 12.5 | 20,352 | 0 | judge 14; judge cache 26 | 0 | drained true; outstanding 0 |

The new probe had no extra audit/judge calls: DeepSeek calls shown in routing are only ordinary hybrid reward computation. AIME24 was fully deterministic because its sampled numeric answers were resolved locally; GameOf24, GPQA, and MedQA used the normal judge fallback and/or stable cache. The GPQA `42 attempted / 40 completed` summary and two retry events are retained in the JSON runtime evidence, while only the 40 valid rows contribute group metrics.

The prior complete rows retained their original telemetry: Bamboogle 18.8 min / 20,906 MiB, AMC23 26.7 min / 20,374 MiB, and 2Wiki 21.2 min / 20,270 MiB. Their scorer routing and cleanup markers remain in the prior result provenance.

## Hypotheses and interpretation

The observed separation is consistent with task-format difficulty and answer-generation behavior for this exact 3B/tool configuration, but it is not a causal estimate of benchmark difficulty and does not establish transfer of paper-level accuracy. The paper's benchmark results cannot be assumed to predict these binary reward distributions.

GameOf24 and 2Wiki are the clearest useful-regime candidates: their reward means are 0.6000 and 0.6125, with mixed ratios of 50% and 45%. AIME24 is also numerically at the strong-screen boundary (0.7000 mean and 30% mixed), but it has only ten groups and all 40 scores were deterministic, so it needs confirmation rather than immediate training use. Musique and GPQA are secondary: each has a 20% mixed ratio, with means 0.2750 and 0.6500 respectively. HotpotQA, Bamboogle, AMC23, and MedQA are mostly all-one; GAIA is mostly all-zero. These rows are poor main candidates under the stated screening heuristic.

Surface answer diversity does not imply outcome diversity. For example, GameOf24 has mean 2.60 unique normalized answers per group but 35% exact duplicate rate, while MedQA has mean 1.40 unique answers and 65% exact duplicate rate. Tool/path signatures are also relatively concentrated (1.4--1.9 unique signatures per group in the new rows), so the screen supports measuring both answer and path diversity in later probes. This is descriptive evidence only; it does not show data leakage.

## Conclusions

1. The quota-blocked AIME24/GameOf24 results were successfully replaced by complete, comparable runs, and GPQA/MedQA were completed. The final ten-benchmark screen is complete at 520/520 valid rollouts.
2. The strongest observed binary-GRPO candidates are GameOf24 and 2Wiki. AIME24 is a promising boundary candidate, not yet a high-confidence winner because its new sample has only ten groups.
3. The screen does not support using GAIA, MedQA, AMC23, Bamboogle, or HotpotQA as the main 3B baseline under this protocol. Musique and GPQA can remain secondary comparison targets.
4. These are raw scorer metrics. They should not be mixed with the historical manually corrected outcome-reward audit metrics; the comparison above uses the same raw hybrid scorer protocol for all rows.

## Recommendation for stage 2

Do not start stage 2 or formal Flow-GRPO training in this task. After approval, expand the following candidates to 20--30 prompts under the same protocol:

1. GameOf24, to confirm its 50% mixed estimate;
2. AIME24, to confirm its 30% boundary estimate with more groups;
3. 2Wiki, as the current strongest 20-prompt result and a useful replication anchor.

For any later training experiment, keep every probe/evaluation example out of the training pool. Build a separate pool from an allowed non-overlapping official train split or an explicitly analogous source, and run identifier/content-hash overlap checks against all probe manifests before training. If a candidate remains promising, use variance-aware sampling only after the larger probe confirms the signal; do not hand-label difficulty or tune the scorer per benchmark.

## Verification and artifact policy

The updated aggregate was generated offline from the fixed manifest and final raw rollout directories. The runner now supports a validated `AGENTFLOW_PROBE_RUN_TAG` solely to preserve independent rerun evidence; it does not alter model, sampling, scoring, or training behavior. The tracked artifacts are this handoff, the small results JSON, the fixed sample manifest, and the generic preparation/runner/aggregation scripts. Raw logs, generated parquet files, judge cache, and rollout data remain local and untracked.

Before commit, run the existing scorer/cleanup unit tests, Python compilation, `bash -n`, JSON consistency checks, `git diff --check`, and a scoped secret scan. No API key or other credential is included in the report, results, manifest, or source changes.
