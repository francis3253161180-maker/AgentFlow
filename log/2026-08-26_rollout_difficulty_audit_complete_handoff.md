# Rollout difficulty audit complete handoff

## Observed facts

- Scope was rollout-only. The effective launch overrides were `trainer.val_only=true`, `trainer.val_before_train=true`, `trainer.save_freq=0`, `rollout.n=4`, and rollout `temperature=0.7`. The model was `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`, with LoRA rank 8 / alpha 16. Prompt/response lengths and tool configuration came from the existing mini20 config. No optimizer step, backward, parameter update, or checkpoint marker was observed.
- The fixed manifest `log/2026-08-26_rollout_difficulty_audit_sample_manifest.json` was reused without resampling: seed `20260826`, 100 prompts, NQ 50 / mathhard 50, selected parquet SHA-256 `48b93b2ca6162f71dad4f1fa8f967a86e2058dac8379ed46eb562e119c722c5f`.
- Four sequential chunks were run in manifest order, each containing 25 prompts and 100 rollouts. Every chunk reported `100/100` completed and `100` valid, with zero retries. The aggregate contains exactly 100 unique prompt groups and 400 rollout JSON files.
- Every counted chunk emitted `VLLM_CLEANUP_DRIVER` with `drained=True`, `outstanding_before=0`, `abort_count=0`, `abort_errors=0`, and `sleep_started=True`. There were four `drained=True` and zero `drained=False` markers. No `CUDA illegal memory access`, CUDA OOM, prefix-cache reset failure, Ray worker death, deadlock, or unrecoverable 5xx was found in the counted train/rollout logs.
- GPU peak across the four monitoring logs was 20,370 MiB; final GPU usage was 2 MiB. No Ray, vLLM, training, `resource_tracker`, or `spawn_main` process remained after chunk3 cleanup.
- Phase 0 orphan cleanup was conservative. The confirmed old AgentFlow PPID=1 pairs were `70056/70057`, `146162/146163`, `157940/157941`, `163776/163777`, `169643/169644`, and `175532/175533`; all received SIGTERM and exited, so no SIGKILL was needed. The later old-smoke pair `205101/205102` was also terminated after the aborted preliminary launch. The long-lived `python -i` PID 39104 and unrelated Desktop Commander/tmux/Jupyter processes were not killed because they were not unambiguously orphan rollout workers.

The complete group distribution is:

| subset | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | mixed | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall (100) | 14 (14%) | 4 (4%) | 6 (6%) | 9 (9%) | 67 (67%) | 19 (19%) | 0.7775 |
| NQ (50) | 11 (22%) | 0 (0%) | 4 (8%) | 3 (6%) | 32 (64%) | 7 (14%) | 0.7250 |
| mathhard (50) | 3 (6%) | 4 (8%) | 2 (4%) | 6 (12%) | 35 (70%) | 12 (24%) | 0.8300 |

Overall reward counts were 311 positive and 89 negative out of 400. Mixed groups are exactly the groups that can produce non-zero group-relative GRPO advantage: 19/100 overall, 7/50 NQ, and 12/50 mathhard. The deterministic duplicate audit found mean 3.04 normalized unique answers per group, mean exact duplicate rate 24%, and an exact-duplicate group proportion of 46%. All groups had a stable tool-path signature; mean unique signatures per group was 1.32. These are deterministic exact/path measures, not an LLM similarity score.

Hybrid scorer routing was observed for all 400 rewards: deterministic 192 (48.0%), uncached DeepSeek judge 185 (46.25%), judge-cache 23 (5.75%), and API/parse errors 0. The 23 cache hits are routing events, not extra API calls; observed judge API calls were 185.

## Hypotheses

- The 19% overall mixed-group rate is higher than the previous mini20 observation of complete collapse, but it is below the approximately 30% level that would make a larger baseline an obvious first move. NQ is particularly sparse at 14%; mathhard is less sparse at 24%.
- Mathhard has more mixed groups and a higher reward mean in this sample, while also showing more exact answer duplication. This is an observed split, not evidence that temperature or tool paths alone caused the difference.
- The old runner's first correct chunk0 completion returned a false `missing_rollout_data_directory` after training cleanup because its path matcher did not account for timestamped run directories. The actual timestamped chunk0 directory was verified directly and used in the aggregate. Chunks 1–3 used the corrected matcher. This runner issue did not invalidate the completed rollout data or cleanup marker.
- A preliminary chunk0 launch before the counted run exposed an incorrect temporary `val_before_train=false` setting and was terminated/excluded before the audit. All four counted chunks used the explicit `val_before_train=true` override and passed the no-update marker checks.

## Theoretical advantage check

The offline check used the installed GRPO convention:

`(reward - group_mean) / (torch.std(reward, unbiased=True) + 1e-6)`.

All 19 mixed groups had non-zero theoretical advantages. Three representative vectors were:

| source / idx | reward vector | theoretical advantages |
|---|---|---|
| NQ / 60799 | `[0, 0, 1, 1]` | `[-0.866024, -0.866024, 0.866024, 0.866024]` |
| NQ / 47281 | `[1, 0, 1, 1]` | `[0.499999, -1.499997, 0.499999, 0.499999]` |
| NQ / 39820 | `[1, 0, 1, 1]` | `[0.499999, -1.499997, 0.499999, 0.499999]` |

This confirms that the binary reward vectors in this rollout-only audit are capable of producing a non-zero group signal; no optimizer or advantage update was run here.

## Conclusions

- The 100-prompt audit is complete and internally consistent: 100/100 prompt groups have four valid training rollouts, with no retries or runtime cleanup failure.
- Mini20's all-equal reward pattern was too small to characterize the broader sample, but current binary reward variance is still sparse: 81% of groups are all-0 or all-1. The result is not consistent with declaring the current setup broadly variance-rich.
- The evidence does not support attributing the result to a cleanup race or an advantage formula failure. Cleanup drained before sleep on all four chunks, and every observed mixed group has non-zero theoretical advantage.

## Recommendation

Do not start formal baseline GRPO training solely on the assumption that mini20 was an unlucky collapse. The overall mixed rate is 19% (NQ 14%, mathhard 24%), below the roughly 30% expansion heuristic and with a notably sparse NQ signal. The next controlled decision should be a general difficulty-/variance-aware data or group-sampling experiment, or an explicitly larger baseline-data audit, with the same scorer and no sample-specific rules. This handoff itself does not authorize or start that experiment.

## Reproduction and artifacts

- Aggregator: `scripts/aggregate_rollout_difficulty_audit_complete_20260826.py`
- Results: `log/2026-08-26_rollout_difficulty_audit_complete_results.json`
- Fixed chunk runner: `scripts/run_rollout_difficulty_audit_chunk_20260826.sh`
- Raw chunk train/rollout logs and rollout JSON directories remain local evidence and are intentionally not committed.
- Verification used the AgentFlow environment's Python, `py_compile`, `bash -n`, `git diff --check`, and the existing scorer/cleanup unit tests. The final commit hash is supplied in the handoff response.
