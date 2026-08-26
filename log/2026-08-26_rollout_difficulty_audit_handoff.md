# Rollout difficulty / reward-variance audit handoff

## Observed facts

- Audit status: **aborted_incomplete_sample_set**. The run was stopped after the existing rollout-only process reached its 60.1-minute timeout and vLLM reported a CUDA illegal-memory-access error. No rerun, training, validation, model switch, or parameter update was started.
- Repository state before this audit: branch `experiment/flow-grpo-3b-lora`, HEAD `5300e27`.
- Source data: `data/train/combined_train.parquet`, 182,190 rows: 79,168 NQ and 103,022 mathhard.
- Fixed sampling: seed `20260826`, 50 NQ + 50 mathhard, 100 unique global data IDs. The selected parquet SHA-256 is recorded in the JSON result and the small committed sample manifest. Original questions/results were not modified; source-local `extra_info.idx` collisions were avoided so rollout paths remain unambiguous.
- Fixed runtime configuration: Qwen2.5-3B-Instruct, LoRA rank 8, alpha 16, one GPU, temperature 0.7, rollout.n=4, max prompt 1280, max response 384, the mini20 tool/model configuration, and `trainer.save_freq=0`. The launch command and resolved config in the train log show the actual n and temperature overrides.
- This repository's rollout-only instrumentation necessarily enters the existing `_validate` method to queue training-mode groups, then returns through `trainer.val_only=true` before `_train_step`. Its `Initial validation metrics`/`step:0` line is therefore an instrumentation summary for this rollout-only queue, not a pre/post validation pair and not a training update.
- The run queued 400 tasks and completed 219 valid rollouts (219/400, 54.8%); retries were 0 for the completed work. There are 54 complete n=4 groups and one incomplete group with 3 rollouts. The remaining 46 mathhard prompts were not observed.
- No optimizer step, backward pass, training advantage computation, `pg_loss`, `grad_norm`, `old_log_prob`, or training `global_step` was executed. No checkpoint file was written. These quantities are not numerically zero; they are not applicable because the run stopped before training.
- GPU monitor evidence: 727 samples, observed peak 20,310 MiB of 32,607 MiB, peak reported utilization 45%. The failure was not an OOM.
- At 14:43:22 the vLLM worker failed in `gpu_model_runner.py` while converting sampled token IDs, with `RuntimeError: CUDA error: an illegal memory access was encountered`; the worker then reported a possible deadlock and the proxy returned HTTP 502. This is the stopping anomaly.

## Hypotheses

- The immediate run failure is consistent with a vLLM/CUDA runtime failure after partial completion. The available log does not establish whether the trigger was a particular prompt, a long-lived cache/state issue, or an upstream runtime defect.
- The partial completed-group distribution may look sparse, but it is not an unbiased 100-prompt estimate: all 50 NQ groups completed while only the first 4 mathhard groups completed before the timeout. Treating it as a general Qwen2.5-3B signal-sparsity result would be misleading.
- The scorer did not show API/parse failures in the completed events, so the observed abort is not currently attributable to the DeepSeek judge fallback.

## Conclusions

- The requested larger audit did **not** complete and cannot answer whether mini20 collapse was a small-sample accident or a general binary-reward GRPO sparsity phenomenon.
- Among the 54 complete groups only, the partial evidence is:

  | population | groups | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | mixed | reward mean |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | complete groups observed | 54 | 12 (22.22%) | 1 (1.85%) | 2 (3.70%) | 2 (3.70%) | 37 (68.52%) | 5 (9.26%) | 0.7361 |
  | NQ complete groups | 50 | 12 (24.00%) | 0 | 2 (4.00%) | 1 (2.00%) | 35 (70.00%) | 3 (6.00%) | 0.7350 |
  | mathhard complete groups | 4 | 0 | 1 (25.00%) | 0 | 1 (25.00%) | 2 (50.00%) | 2 (50.00%) | 0.7500 |

- The table is explicitly **partial evidence**, not the requested 100-prompt result. In particular, the mathhard row has only 4/50 groups and must not be extrapolated.
- For complete groups, average normalized unique answers/group was 3.685 and mean normalized exact duplicate rate was 7.87%; simple structural tool-path signatures were available for all 54 groups, with mean unique signatures 1.185/group. These are deterministic duplicate/path metrics; no LLM similarity metric was used.
- The theoretical production GRPO formula was independently applied to the recorded binary reward vectors. All 5 complete mixed groups had non-zero theoretical advantages. Examples:
  - `[0,0,0,1]` → `[-0.499999, -0.499999, -0.499999, 1.499997]`.
  - `[1,1,1,0]` → `[0.499999, 0.499999, 0.499999, -1.499997]`.
  - `[1,0,0,1]` → `[0.866024, -0.866024, -0.866024, 0.866024]`.
  This confirms the formula on observed mixed groups; it does not test a training update because no update ran.

## Scorer routing and resources

- 219 hybrid scorer events were observed: deterministic 49, uncached judge 158, judge-cache route 12. Thus 0 API/parse errors were recorded; 12 cache hits were recorded. Mean event latency was 861.0 ms and median 984.6 ms across the telemetry events. The cache key and raw prompt/answer contents were not emitted into the committed result.
- All 219 completed rollouts were valid, but “all 400 valid” is false because 181 expected rollouts were never completed. The incomplete 3-rollout group is excluded from the 5-bin summaries.
- Raw rollout JSON and large train/rollout logs remain local evidence and are not committed.

## Recommended next step

Do not use this partial run to choose a formal training change or to label binary GRPO as generally sparse. First independently audit the vLLM illegal-memory-access failure and the 60.1-minute partial-completion behavior. A future rerun should be separately approved after that diagnosis; it should preserve this manifest and use an explicit abort check for Traceback/illegal-memory-access markers. No formal training or validation should start from this handoff.

## Reproducibility and artifacts

- Sample manifest: `log/2026-08-26_rollout_difficulty_audit_sample_manifest.json`.
- Aggregated evidence: `log/2026-08-26_rollout_difficulty_audit_results.json`.
- Preparation script: `scripts/prepare_rollout_difficulty_audit_20260826.py`.
- Runner: `scripts/run_rollout_difficulty_audit_20260826.sh`.
- Aggregator: `scripts/audit_rollout_difficulty_20260826.py`.
- Test/verification commands run after the abort are recorded in the final repository handoff: Python compilation, `bash -n`, scorer unit tests, `git diff --check`, and secret scan. No new GPU process was started after the abort.
