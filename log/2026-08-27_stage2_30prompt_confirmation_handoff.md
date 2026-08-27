# Stage-2 30-prompt confirmation handoff

## Observed facts

- Branch: `experiment/flow-grpo-3b-lora`. This was a rollout-only probe; no
  backward pass, optimizer update, parameter update, or checkpoint was run.
- Protocol was held constant: Qwen2.5-3B-Instruct, existing AgentFlow toolchain
  and scorer, LoRA rank 8 / alpha 16, temperature 0.7, rollout.n=4,
  `val_only=true`, `optimizer_steps=0`, and `save_freq=0`. The launch logs
  show `override_generation_config` with `n=4`, `temperature=0.7`, `top_k=-1`,
  and `top_p=1`.
- The reproducible manifest uses seed `20260827` and samples additional rows
  with `random.sample` after excluding every historical manifest row. It records
  source row IDs, source-record/question/ground-truth hashes, source file hash,
  and output parquet hash. The manifest explicitly marks all benchmark examples
  as probe-only and forbidden from a future formal training pool.
- Source fixtures were unchanged and structurally valid:

  | Dataset | Source | Source SHA-256 | Historical | Incremental | Combined |
  |---|---|---|---:|---:|---:|
  | 2Wiki | `test/2wiki/data/data.json` | `eb4b045e...5312` | 20 | 10 | 30 |
  | GameOf24 | `test/gameof24/data/data.json` | `ffb0c950...14ff` | 10 | 20 | 30 |
  | AIME24 | `test/aime24/data/data.json` | `92843562...feb2` | 10 | 20 | 30 |

- Historical and incremental source-row sets were checked for identifier and
  content-hash overlap. Each combined set has exactly 30 unique source rows.
- All three incremental runs completed fully: 2Wiki 40/40, GameOf24 80/80,
  and AIME24 80/80 valid rollouts, with zero retries. The merged totals are
  therefore exactly 120 valid rollouts per dataset.
- Safe cleanup completed on every run. The incremental logs contain one
  `VLLM_CLEANUP_DRIVER reason=normal_complete` marker each with
  `drained: True`, `outstanding_before: 0`, `abort_count: 0`, and
  `sleep_started: True`. No `drained=False`, CUDA illegal access/OOM,
  prefix-cache reset failure, Ray worker/deadlock, or scorer API/parse error
  was observed. GPU monitoring showed incremental peaks of approximately
  20,304 MiB (2Wiki), 20,348 MiB (GameOf24), and 20,362 MiB (AIME24); after
  cleanup `nvidia-smi` showed about 2 MiB used and no active training/Ray/vLLM
  compute process.

## Incremental and combined results

The primary comparison is the combined 30-group result. A group is mixed when
its four binary rewards are 1/4, 2/4, or 3/4; this is also a nonzero
group-normalized GRPO variance group for binary rewards.

| Dataset / phase | Groups | Reward mean | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | Mixed | Wilson 95% CI | Unique answers/group | Exact duplicate rate | Unique paths/group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2Wiki historical | 20 | 0.6125 | 3 | 3 | 4 | 2 | 8 | 45.00% | 25.82–65.79% | 3.55 | 11.25% | 1.75 |
| 2Wiki incremental | 10 | 0.5750 | 3 | 1 | 0 | 2 | 4 | 30.00% | 10.78–60.32% | 3.20 | 20.00% | 1.60 |
| **2Wiki combined** | **30** | **0.6000** | **6** | **4** | **4** | **4** | **12** | **40.00%** | **24.59–57.68%** | **3.43** | **14.17%** | **1.70** |
| GameOf24 historical | 10 | 0.6000 | 1 | 2 | 3 | 0 | 4 | 50.00% | 23.66–76.34% | 2.60 | 35.00% | 1.50 |
| GameOf24 incremental | 20 | 0.6625 | 2 | 3 | 4 | 2 | 9 | 45.00% | 25.82–65.79% | 3.20 | 20.00% | 1.95 |
| **GameOf24 combined** | **30** | **0.6417** | **3** | **5** | **7** | **2** | **13** | **46.67%** | **30.23–63.86%** | **3.00** | **25.00%** | **1.80** |
| AIME24 historical | 10 | 0.7000 | 2 | 0 | 1 | 2 | 5 | 30.00% | 10.78–60.32% | 2.10 | 47.50% | 1.50 |
| AIME24 incremental | 20 | 0.6000 | 3 | 4 | 3 | 2 | 8 | 45.00% | 25.82–65.79% | 2.40 | 40.00% | 1.80 |
| **AIME24 combined** | **30** | **0.6333** | **5** | **4** | **4** | **4** | **13** | **40.00%** | **24.59–57.68%** | **2.30** | **42.50%** | **1.70** |

For all combined 30-group datasets, the nonzero-variance ratio equals the mixed
ratio: 2Wiki 40.00%, GameOf24 46.67%, and AIME24 40.00%. Reward counts are
respectively 72/120, 77/120, and 76/120 positive. The Wilson intervals are
descriptive only; 30 groups is still a small sample.

## Scorer routing, runtime, and cleanup

These counts include historical plus incremental rollouts in the 30-group
total. They are normal hybrid-scorer calls only; no extra judge/audit calls
were made.

| Dataset | Deterministic | Judge API | Judge cache | Judge fallback total | API/parse errors | Incremental runtime | Cleanup true/false | Retry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 32 | 81 | 7 | 88 | 0 | 10.2 min | 2 / 0 | 0 |
| GameOf24 | 3 | 87 | 30 | 117 | 0 | 38.4 min | 2 / 0 | 0 |
| AIME24 | 112 | 8 | 0 | 8 | 0 | 51.0 min | 2 / 0 | 0 |

The combined scorer-event denominator is 120 per dataset. The corresponding
combined scorer latency means were approximately 628.9 ms (2Wiki), 906.7 ms
(GameOf24), and 82.7 ms (AIME24); these are telemetry means, not end-to-end
generation latency. Incremental GPU observed peaks were 20,304 / 20,348 /
20,362 MiB, respectively.

## Interpretation and recommendation

### Conclusions

Under the same 3B protocol, all three datasets provide materially more group
reward variance than the cleaned NQ+DeepMath baseline (historical corrected
mixed estimate about 13.10%; raw and manually corrected metrics are not
interchangeable). GameOf24 is the strongest screening result: it has the
highest combined mixed ratio (14/30), a mid-range reward mean (0.6417), and
the narrowest practical alignment with the useful binary-reward regime among
these three. 2Wiki and AIME24 are also viable variance sources, but 2Wiki has
slightly lower reward mean and AIME24 has much higher exact answer duplication
and a longer rollout runtime.

The stage-2 choice for a Standard GRPO baseline should therefore be
**GameOf24 as the primary task distribution candidate**, with **2Wiki as a
cross-task validation/probe candidate**. AIME24 is a reasonable secondary
mathematical probe, but should not be preferred over GameOf24 solely from this
small sample. This decision is based on observed rewards and mixed groups, not
on paper-level accuracy assumptions.

### Hypotheses and remaining uncertainty

- The added rows changed the point estimates relative to the historical small
  probes (for example, AIME24 increased from 30% to 40% mixed), so the
  incremental and combined tables must remain separate. The 30-group intervals
  still overlap substantially; this is screening evidence, not a significance
  claim.
- High path-signature reuse and answer duplication indicate correlated samples
  may reduce the effective number of independent outcome draws. Exact duplicate
  rate is a surface metric and does not prove semantic duplication.
- GameOf24 relies heavily on the DeepSeek fallback (117/120 fallback events),
  so provider availability/latency is an operational dependency. AIME24 is
  mostly deterministic and operationally cheaper per scorer event, but its
  group answers are more repetitive.
- These probes do not establish transfer to formal training, nor do they
  identify an optimal mixture or sampling policy. No claim of data leakage is
  made by this audit.

### Recommendation for next controlled experiment

Do not start training automatically. If approved, use GameOf24 as the primary
candidate for a controlled Standard GRPO baseline and retain 2Wiki as an
out-of-task validation/probe set. Construct the future training pool only from
an explicitly non-overlapping official train split or an analogous source;
perform row-ID and content-hash overlap checks against this stage-2 manifest.
Never place these benchmark evaluation/probe rows in the training pool. AIME24
can be retained for math-specific validation or a separately approved mixture
study. HOB and variance-aware sampling were not implemented or run here.

## Verification and artifacts

- New preparation script: `scripts/prepare_stage2_30prompt_confirmation_20260827.py`.
- New offline merge script: `scripts/aggregate_stage2_30prompt_confirmation_20260827.py`.
- Runner compatibility update: `scripts/run_benchmark_difficulty_probe_20260827.sh`
  accepts `2wiki` and retains the existing run-tag/expected-prompt controls.
- Manifest: `log/2026-08-27_stage2_30prompt_confirmation_sample_manifest.json`.
- Results: `log/2026-08-27_stage2_30prompt_confirmation_results.json`.
- Raw logs and rollout data remain local and untracked.
- JSON parsing, source overlap/count checks, Python compilation, scorer/cleanup
  tests, shell syntax, `git diff --check`, and scoped secret review were run
  before delivery. Commit hash is recorded in the final handoff response.
