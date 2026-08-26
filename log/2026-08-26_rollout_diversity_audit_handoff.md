# Rollout diversity audit handoff — 2026-08-26

## Observed evidence (A: previous controlled training)

The audit read only `train/step_*/idx_*/rollout_*.json` under the previous run's rollout root. Validation directories were excluded. The source run was `b76d29f` on `experiment/flow-grpo-3b-lora`; it used the same mini20 seed20260825 data and produced 40 training rollout files for 20 data groups, exactly two files per group.

The effective sampling evidence is split between the trainer and AgentFlow planner:

- The trainer-resolved vLLM actor configuration was `rollout.n=2`, `do_sample=True`, `temperature=1.0`, `top_p=1`, `top_k=-1`.
- The actual AgentFlow Qwen planner requests used `TRAIN_TEMPERATURE=0.7`, `top_p=0.99`, max tokens 384, and no separately emitted `top_k` or `do_sample` request field. The planner temperature is the parameter varied in B; the trainer actor configuration stayed unchanged.
- Validation used the separate `TEST_TEMPERATURE=0.0` path and was not included in these group statistics.

Training-only group results:

| subset | groups | all-1 | all-0 | mixed | non-zero-variance | mean unique answers | normalized duplicate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 20 | 17 (0.85) | 3 (0.15) | 0 (0.00) | 0 (0.00) | 1.80 | 0.10 |
| NQ | 10 | 7 (0.70) | 3 (0.30) | 0 (0.00) | 0 (0.00) | 1.90 | 0.05 |
| mathhard | 10 | 10 (1.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 1.70 | 0.15 |

All 20 groups had a deterministic structural tool/path signature. The mean number of unique signatures was 1.15 overall (1.10 NQ, 1.20 mathhard). Thus there was some answer/path variation, but no group-level reward variation.

The production GRPO implementation was inspected in the AgentFlow conda environment. It computes outcome scores per uid group, uses the default unbiased `torch.std` plus `epsilon=1e-6`, and broadcasts the normalized scalar over the response mask. Independent recomputation gave zero theoretical advantage for every group. The training log also had advantage min/max/mean all zero at every step; this is consistent with the observed all-1/all-0 groups, not evidence of a mixed-group advantage pipeline bug. The audit gate therefore passed: no mixed group existed that should have produced a non-zero advantage.

## B: rollout-only 2×2 sweep

The four conditions used the same fixed 10 rows from `flowgrpo_mini_20_seed20260825.parquet` (5 NQ, 5 mathhard), the same model, LoRA, tools, prompt/response limits, and scorer. The selected prompt parquet SHA-256 is recorded in `log/2026-08-26_rollout_diversity_audit_results.json`. No optimizer, backward pass, training step, or checkpoint was run. The explicit opt-in route queued training-mode rollouts solely to honor `rollout.n`, then returned through `trainer.val_only=true`.

| condition | planner temp | n | valid | reward | all-1 / all-0 / mixed | non-zero-variance groups | unique answers/group | duplicate rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.7 | 2 | 20/20 | 0.800 | 0.80 / 0.20 / 0.00 | 0/10 | 1.9 | 0.05 |
| B0 | 1.0 | 2 | 20/20 | 0.750 | 0.70 / 0.20 / 0.10 | 1/10 | 1.7 | 0.15 |
| C0 | 0.7 | 4 | 40/40 | 0.775 | 0.70 / 0.20 / 0.10 | 1/10 | 2.9 | 0.275 |
| D0 | 1.0 | 4 | 40/40 | 0.750 | 0.70 / 0.20 / 0.10 | 1/10 | 2.9 | 0.275 |

NQ/mathhard reward breakdown:

- A0: NQ 0.60 (3/5 groups all-1, 2/5 all-0, no mixed), mathhard 1.00 (5/5 all-1).
- B0: NQ 0.60 (same group classes, no mixed), mathhard 0.90 (4/5 all-1, 1/5 mixed).
- C0: NQ 0.60 (same group classes, no mixed), mathhard 0.95 (4/5 all-1, 1/5 mixed).
- D0: NQ 0.60 (same group classes, no mixed), mathhard 0.90 (4/5 all-1, 1/5 mixed).

The only mixed groups were mathhard. Their deterministic reward vectors were B0 `[0,1]`, C0 `[0,1,1,1]`, and D0 `[1,1,0,0]`; they produced non-zero theoretical GRPO advantages. With the production unbiased-std rule, B0's pair is approximately `[-0.707, +0.707]`, C0 is `[-1.500, +0.500, +0.500, +0.500]`, and D0 is `[+0.866, +0.866, -0.866, -0.866]`. These are offline calculations only; no advantage tensor was constructed or applied in the sweep.

All four conditions had final validation summaries with 100% completed and valid rollouts, zero reported retries, zero hybrid reward error events, and no training-update marker. The rollout-only logs show no `actor/pg_loss`, `Training data keys`, or `optimizer.step`; no checkpoint files were created. Sampled `nvidia-smi` memory during the runs was about 20.2 GiB; there was no OOM. The approximate wall times were A0 6.3 min, B0 6.5 min, C0 11.8 min, and D0 13.0 min. The scorer event counters were A0 20 events/8 uncached judge/1 cache, B0 20/5/1, C0 40/13/2, D0 40/8/7; all logged scorer events had `error=none`.

## Temperature vs n conclusions

### Observed facts

- Increasing planner temperature from 0.7 to 1.0 changed A0→B0 from 0/10 to 1/10 mixed groups at n=2, but reduced mean reward from 0.800 to 0.750 and did not increase mean unique answers (1.9→1.7).
- Increasing n from 2 to 4 at temperature 0.7 changed A0→C0 from 0/10 to 1/10 mixed groups and increased mean unique answers from 1.9 to 2.9, while mean reward moved to 0.775.
- At temperature 1.0, B0→D0 kept mixed groups at 1/10 and mean unique answers at 2.9, but doubled rollout cost and mean reward remained 0.750.
- NQ was completely reward-homogeneous in this small sample under all four conditions. Diversity did not produce a mixed NQ reward group.

### Conclusion

The evidence supports both effects, but n is the cleaner lever in this sample: n=4 produced more distinct answers and a non-zero-variance group at the lower temperature, whereas temperature=1.0 did not improve the aggregate reward or answer uniqueness. This is a small 10-prompt experiment, so it demonstrates a routing/sampling effect rather than a general accuracy estimate.

## Remaining uncertainties

- There are only 10 prompts and one seed; the single mixed group in each of B0/C0/D0 is not statistically stable evidence.
- All observed mixed groups are mathhard, so the sweep does not establish that NQ will gain useful group variance.
- “Unique answer” and the structural tool signature are deterministic diversity proxies, not semantic quality measures. No LLM similarity metric was used.
- The trainer's vLLM actor configuration reports `temperature=1.0/top_p=1/top_k=-1/do_sample=True`, while the AgentFlow planner request carries `TRAIN_TEMPERATURE` and `top_p=0.99`. The sweep intentionally changed only the latter, which is the generation parameter confirmed in the planner request; a future refactor should continue to keep these two control planes explicit.
- `perf/max_memory_allocated_gb` is a training-step metric and was not emitted in val-only mode. The reported ~20.2 GiB is a sampled GPU monitor value, not an exact allocator peak.

## Recommendation for next controlled experiment

If a formal training run is approved later, the best observed tradeoff is `TRAIN_TEMPERATURE=0.7` with `rollout.n=4`: it preserved 100% rollout validity, kept reward closer to A0 than either temperature-1.0 condition, and was the only setting that combined higher answer diversity with a non-zero-variance group at the conservative temperature. It costs roughly 1.9× the n=2 wall time in this run, so the cost should be accepted explicitly.

Do not infer that this result guarantees useful learning: the gain was only 1/10 groups and only mathhard. Keep the next run controlled, retain `save_freq=0` unless final-weight retention is explicitly requested, and instrument per-sample advantage min/max so a future mixed group can be verified in-process. No pre-validation, formal training, model change, rollout.n change, or budget change was started after this audit.

## Reproducibility and files

Committed analysis code:

- `scripts/audit_rollout_diversity_20260826.py`
- `scripts/audit_rollout_diversity_sweep_20260826.py`
- `scripts/run_rollout_diversity_sweep_20260826.sh`
- `agentflow/verl/trainer.py` and `agentflow/verl/daemon.py`: explicit, opt-in rollout-only group mode; normal validation/training behavior is unchanged.

Commands used:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/agentflow
python scripts/audit_rollout_diversity_20260826.py ...
python scripts/audit_rollout_diversity_sweep_20260826.py ...
git diff --check
```

Large raw rollout and runtime log files remain local evidence and are intentionally not included in the commit.
