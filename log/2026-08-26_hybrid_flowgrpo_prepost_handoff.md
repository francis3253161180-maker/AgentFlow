# Controlled Flow-GRPO hybrid scorer pre/post handoff

## Observed facts

- Run: `20260826_hybrid_prepost_112050`; branch: `experiment/flow-grpo-3b-lora`; HEAD before this run: `c1d78f2`.
- Model and fixed training settings were preserved: Qwen2.5-3B-Instruct, LoRA rank 8 / alpha 16, mini20 seed20260825, train batch 2, PPO mini batch 2, micro batch 1, `rollout.n=2`, max prompt 1280, max response 384, LR `1e-5`, 1 GPU, 1 epoch. No model, GPU, algorithm, rollout.n, or budget change was made.
- Runtime overrides were limited to `trainer.val_before_train=true`, `trainer.save_freq=0`, a unique experiment name, and `data.val_files` pointing to the same 20-row mini20 file. The validation override was intentional so pre/post results can be split into NQ and mathhard; it is not held-out generalization.
- Pre-validation and post-validation each completed 20/20 valid rollouts. The framework reported 0 retries for all observed rollout progress records.
- The 10 training global steps completed. The 40 saved training rollout JSON records correspond to 20 unique examples with `rollout.n=2` each.

| phase | overall | NQ | mathhard |
| --- | ---: | ---: | ---: |
| pre reward mean (positive/negative) | 0.80 (16/4) | 0.70 (7/3) | 0.90 (9/1) |
| post reward mean (positive/negative) | 0.80 (16/4) | 0.70 (7/3) | 0.90 (9/1) |
| training rollout reward mean (positive/negative) | 0.85 (34/6) | 0.70 (14/6) | 1.00 (20/0) |

Training-step metrics from the console log:

| step | reward mean | advantage mean | pg_loss | grad_norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.50 | 0.00 | 0.00 | 0.00 |
| 3 | 0.75 | 0.00 | 0.00 | 0.00 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.25 | 0.00 | 0.00 | 0.00 |
| 6 | 1.00 | 0.00 | 0.00 | 0.00 |
| 7 | 1.00 | 0.00 | 0.00 | 0.00 |
| 8 | 1.00 | 0.00 | 0.00 | 0.00 |
| 9 | 1.00 | 0.00 | 0.00 | 0.00 |
| 10 | 1.00 | 0.00 | 0.00 | 0.00 |

- Additional recorded training signals: entropy loss was present; old-log-prob and actor-update timing were present; `actor/lr` remained `1e-5`. The log did not provide a scalar old-log-prob value, only `timing_s/old_log_prob`.
- GPU peak from trainer metrics: `perf/max_memory_allocated_gb=21.608757972717285`; reserved peak `27.08203125` GiB. After cleanup the RTX 5090 was idle at 2 MiB used.
- No OOM, `OutOfMemoryError`, `No valid rollout(s)`, or zero-valid-rollout matches were found. The experiment checkpoint directory contains no files, consistent with `save_freq=0`.
- Hybrid scorer events: 80 total; 46 deterministic; 34 judge fallback (30 uncached DeepSeek calls plus 4 cache hits); 0 API errors. Overall uncached judge latency mean/median/P95 was 1103.48/1046.41/1697.38 ms; cached mean/median/P95 was 1.11/1.13/1.24 ms. Phase-level details are in the JSON result.
- One non-fatal framework warning was observed: cleanup reported that `lsof` was unavailable while trying to kill port 9999. It did not stop task processing or affect validity counts.

Evidence:

- Trainer log: `log/20260826_hybrid_prepost_112050_train.log`
- Rollout/scorer log: `log/20260826_hybrid_prepost_112050_rollout.log`
- Rollout JSON root: `rollout_data/107.173.39.105/qwen25-3b-lora-mini20-seed20260825_20260826-112203/Qwen2.5-3B-Instruct_20260826-112204`
- Machine-readable result: `log/2026-08-26_hybrid_flowgrpo_prepost_results.json`

## Hypotheses

- The unchanged pre/post reward on this same 20-row validation set is consistent with little or no effective policy update, but the logs alone do not establish the cause.
- The all-zero advantage, policy-gradient loss, and gradient norm are a material signal that the actor update may have been numerically inactive or that the logged batch advantages collapsed to zero. This requires an independent code/path audit before treating the run as a learning result.
- The single `5/5` valid-response trace observation occurred while the corresponding task batch still reported 4/4 completed and valid; it may reflect an internal trace-counting detail rather than a retry. No retry was reported by the rollout progress logger.

## Conclusions

- The requested controlled pre-validation → fixed 10-step training → post-validation workflow completed safely with the requested model and resource budget.
- The production hybrid scorer was exercised in the real rollout path, including deterministic routing, DeepSeek fallback, cache hits, binary rewards, and zero API failures for this run.
- Operational validity is strong: all 80 framework rollout results were valid, no retry was recorded, and no checkpoint was written.
- There is no evidence in this run of a pre/post reward improvement. Because the training update metrics are all zero, this run must not be used as evidence that Flow-GRPO improved the policy.

## Recommendation

- Stop here as requested. Do not start pre-validation, another seed, 7B, or another training run automatically.
- Before any next controlled experiment, independently audit why `critic/advantages/mean`, `actor/pg_loss`, and `actor/grad_norm` are zero at every global step, including the interaction between grouped rewards, mini-batch dropping, and the current trainer update path.
- Keep the raw rollout and logs as evidence, but do not commit runtime cache, model files, or checkpoints. The report and aggregate JSON are the durable handoff artifacts.

## Reproduction and audit commands

```bash
PYTHONPATH=. /root/autodl-tmp/conda/envs/agentflow/bin/python -m unittest -v test.test_reward_scorer
/root/autodl-tmp/conda/envs/agentflow/bin/python scripts/audit_controlled_hybrid_prepost_20260826.py \
  --train-log log/20260826_hybrid_prepost_112050_train.log \
  --rollout-log log/20260826_hybrid_prepost_112050_rollout.log \
  --rollout-root rollout_data/107.173.39.105/qwen25-3b-lora-mini20-seed20260825_20260826-112203/Qwen2.5-3B-Instruct_20260826-112204 \
  --data data/train/flowgrpo_mini_20_seed20260825.parquet \
  --checkpoint-dir checkpoints/agentflow-mini-baseline/qwen25-3b-lora-mini20-seed20260825-hybrid-prepost-20260826 \
  --output log/2026-08-26_hybrid_flowgrpo_prepost_results.json
git diff --check
```
