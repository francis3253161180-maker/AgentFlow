# MuSiQue + 2Wiki all-Qwen7B rollout-only probe handoff

## Observed facts

- This was a rollout-only difficulty probe. No backward pass, optimizer step, parameter update, checkpoint, or formal training was run. The benchmark dev rows are probe-only and must not enter a future training pool.
- Both datasets completed exactly 10 fixed prompt groups × 4 rollouts: 40/40 valid per dataset, retry count 0, and no missing group.
- The final successful runs used the local Qwen7B route: planner_main=`qwen-actor` with the current LoRA adapter; planner_fixed/verifier/executor/tool route=`qwen-base` with no adapter. The model path was `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` for all roles. No Doubao, DeepSeek, GPT, or other external model call was enabled.
- Final runtime protocol was `temperature=0.7`, fixed-role temperature `0.0`, `rollout.n=4`, seed `20260829`, max prompt/response `1536/1024`, max model length `4096`, vLLM tensor parallel size 1, `max_num_seqs=1`, `max_num_batched_tokens=1024`, and `gpu_memory_utilization=0.60`.
- The first `.60` attempt was correctly stopped because an old orphan worker occupied the GPU. A subsequent attempt with the actor offload override removed still failed before rollout because the actor occupied the GPU. After restoring the already-validated FSDP2 `offload_policy=true` path and cleaning the orphan, both final runs started successfully. These failed attempts produced no probe result and are not included in the metrics.

## Protocol and data provenance

The fixed sample is `log/2026-08-29_multihop_allqwen20_probe_sample_manifest.json`, generated with `seed=20260829` using `random.sample` over structurally valid rows, then sorted by source row index. Selection did not inspect answers, rewards, or apparent difficulty.

| dataset | source split | source commit | source rows | source SHA256 | selected parquet SHA256 |
|---|---|---:|---:|---|---|
| MuSiQue | official MuSiQue-Ans `dev` | `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | 2417 | `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b` | `0dba79b972473a34b27586f24d7e1a0f2b179d5b8a92bb545417d7bc8fc8e450` |
| 2Wiki | official `dev` | `13800e5be57df1b4040b9b1588c6c811779e69e9` | 12576 | `48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6` | `1cbb6f46f8b55cf3fc8587ef8f0eba6b3877ab42c84275f06c99bb1467f12ee8` |

The source files and generated parquet files are under `/root/autodl-tmp/tmp/multihop_official_sources_20260829/` and `/root/autodl-tmp/tmp/multihop_allqwen_probe_20260829/`; they remain local and untracked. The probe uses dev data for engineering measurement only, not for policy training.

## Rollout results

All reward values below are the unchanged current scorer output. `mixed` and `nonzero-variance` are identical here because rewards are binary.

| dataset | groups / valid | reward mean | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | mixed / nonzero variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue | 10 / 40 | 0.0250 | 9 (90%) | 1 (10%) | 0 | 0 | 0 | 1/10 (10%) |
| 2Wiki | 10 / 40 | 0.0000 | 10 (100%) | 0 | 0 | 0 | 0 | 0/10 (0%) |

Additional deterministic diversity statistics:

| dataset | mean unique final answers/group | exact duplicate rate | mean unique tool/path signatures/group | mean steps | runtime | peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| MuSiQue | 3.20 | 20.0% | 1.50 | 1.35 | 21.5 min | 20,467 MiB |
| 2Wiki | 2.40 | 40.0% | 1.30 | 1.325 | 19.8 min | 20,319 MiB |

The exact per-group vectors and answer/path hashes are in the results JSON. MuSiQue's only mixed group was `[1, 0, 0, 0]`; all 2Wiki groups were `[0, 0, 0, 0]`.

## Scorer routing and semantic caveat

- The runner set `AGENTFLOW_REWARD_JUDGE_ENABLED=0`, `AGENTFLOW_DISABLE_EXTERNAL_LLM=1`, and `external_calls=0`. The aggregate therefore records 80 deterministic-scored rollouts, 0 DeepSeek fallback calls, 0 cache hits, 0 API errors, and 0 parse errors.
- The current production logs did not emit per-rollout scorer route events, so the routing counts are configuration-backed, not independently reconstructed from a scorer telemetry event. This is explicitly represented as `per_row_telemetry_observed=false` in the JSON.
- The raw trace span name `openai.chat.completion` is the local OpenAI-compatible AgentFlow/vLLM interface. It is not evidence of an external OpenAI call: route records show only local `qwen-base`/`qwen-actor`, requests go to `127.0.0.1`, and no DeepSeek/Doubao/GPT markers were observed.
- A deterministic scorer limitation is visible in the persisted results. For example, the MuSiQue group whose ground truth is `Charles University` includes answers explicitly stating that employer, yet all four received reward 0 because external semantic judging was disabled. Conversely, the MuSiQue group with ground truth `38` has one exact concise `38` answer and is the only positive. In 2Wiki, some all-zero groups contain clearly wrong direct answers, while other answers give the expected entity in longer prose and still receive 0. These are observations about scorer/output text, not a manual correctness relabeling of all 80 rows.

## Runtime and cleanup evidence

- MuSiQue final evidence: `log/20260829_multihop_allqwen7b_musique_train.log`; rollout directory `rollout_data/46.38.243.197/multihop-allqwen7b-musique-20260829_20260829-165503/Qwen2.5-7B-Instruct_20260829-165504/train`.
- 2Wiki final evidence: `log/20260829_multihop_allqwen7b_2wiki_train.log`; rollout directory `rollout_data/46.38.243.197/multihop-allqwen7b-2wiki-20260829_20260829-171907/Qwen2.5-7B-Instruct_20260829-171908/train`.
- Each final log contains `Validation summary: 40/40 total rollouts (100.0%), 40 valid rollouts`, normal completion cleanup with `outstanding=0`, `drain_complete=1`, `drained=True`, then prefix-cache reset and sleep. Final cleanup durations were 0.764s (MuSiQue) and 0.660s (2Wiki).
- No final-run CUDA OOM, illegal memory access, prefix-cache reset failure, `drained=false`, Ray worker death, deadlock, or HTTP 5xx marker was found. The GPU monitor peaked at the values above and ended at 0 MiB. The local raw logs, rollout JSON, source data, and caches are not tracked.

## Hypotheses

1. The observed near-zero raw reward, especially 2Wiki's 0/40 positives, combines genuine Qwen7B multi-hop failures with deterministic scorer under-recall for long natural-language answers. The probe alone cannot identify their proportions.
2. The low unique tool/path-signature counts and one-tool trajectory pattern suggest the current agent often reaches a short, homogeneous tool path; this may contribute to outcome homogeneity, but it does not prove that the benchmark is intrinsically easy or hard.
3. MuSiQue is a more useful follow-up candidate than 2Wiki under this exact raw-score protocol because it produced the only mixed group and a nonzero reward mean. This is a screening observation with only 10 groups, not evidence that its paper-level accuracy transfers.

## Conclusions

- The all-Qwen7B architecture and safe lifecycle are operationally feasible on the RTX 5090 with `gpu_memory_utilization=0.60` when the existing FSDP2 `offload_policy=true` path is retained. Both bounded datasets completed without lifecycle failure.
- Under the unchanged deterministic scorer, MuSiQue produced 10% mixed groups and 2Wiki produced 0%. These raw numbers should not be interpreted as semantic accuracy because no judge fallback was allowed and the spot evidence shows false negatives for natural-language answers.
- The result does not support choosing 2Wiki as a useful binary-GRPO task distribution. MuSiQue is the only provisional candidate, but its current reward signal is too sparse and scorer-sensitive to justify training directly from this dev probe.

## Recommendation

- Do not add either dev probe sample to a future training pool and do not start formal GRPO from these rows.
- Before any baseline decision, run an approved offline semantic validity audit or use the already-designed hybrid scorer in a separately authorized evaluation, with route telemetry enabled. Keep benchmark dev/eval rows isolated.
- If MuSiQue remains promising after scorer-validity confirmation, construct a future train pool from its official non-overlapping training split (or a separately justified analogous source), using source IDs and normalized content hashes to exclude every probe row. Treat 2Wiki similarly only if a separate train-pool/provenance review supports it.
- No stage-2 probe, variance-aware sampling, or formal training was started by this handoff.

## Reproducibility and verification

- Preparation: `scripts/prepare_multihop_allqwen_probe_20260829.py`
- Runner: `scripts/run_multihop_allqwen_probe_20260829.sh`
- Aggregation: `scripts/aggregate_multihop_allqwen_probe_20260829.py`
- Results: `log/2026-08-29_multihop_allqwen20_probe_results.json`
- Manifest: `log/2026-08-29_multihop_allqwen20_probe_sample_manifest.json`
- Final generated config hashes: MuSiQue `2510e94c03905fd553d324db1ee9b20269b68134300d0adb9d3642322a0e2eab`; 2Wiki `12adeaf5deca52e0508b2812e4c517085bedcfcdb0fbacede8c7764346820aeb`.
