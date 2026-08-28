# Reward audit and 2048/8192 probe handoff

## Observed facts

- The saved random30 replay was audited offline only. It contains 30 groups, 120 trajectories, four records per group, and stored reward `0.0` for all 120 records. The original evidence and replay pack were not overwritten.
- An independent exact-Fraction Game24 solver found a valid expression for all 30 puzzle groups. After the generic scorer change, production deterministic scoring accepted all 30 known-valid tagged expressions. This is a scorer-path regression check, not a relabeling of the model's original outputs.
- The audit's conservative zero-reward cause counts are: `invalid_or_no_expression=23`, `wrong_number_multiset=95`, and `context_error_affected=2`. These classifications describe reconstructed saved final responses and can overlap with the interpretation of an affected context event; they do not prove that every generated answer was semantically wrong.
- The offline overlay is separate and was never written back. It preserves the original reward evidence. One reconstructed response disagreed with the independent oracle/production comparison; it was not used to edit the original run.

## Scorer change

`train/utils.py` now recognizes an explicit arithmetic answer candidate for a scalar numeric target and proves the whole expression with the existing SymPy path before falling through to numeric-token matching. The rule is generic and only applies to explicit candidates; unmarked prose is not promoted. Focused tests cover a valid expression, a mismatching expression, and incidental prose. No dataset/sample-specific rule or reward-range change was made.

## Phase B selection and protocol

The selection was prepared from persisted length metadata before generation, using the fixed selection manifest and its SHA256. It contains eight groups (32 intended rollouts), prioritizing the previously observed 1024-token response cap and largest prompt context. The intended protocol was Qwen2.5-7B-Instruct, temperature `0.7`, `n=4`, prompt/response `3072/2048`, vLLM `max_model_len=8192`, dynamic response padding, vLLM utilization `0.50`, local deterministic scorer, external calls disabled, rollout-only/val-only, optimizer steps `0`, and checkpoint disabled.

## Probe result and blocker

The runner reached model initialization but stopped before launching `train/rollout.py`. It failed while restoring the required initial LoRA snapshot into the already FSDP2-wrapped actor. The snapshot contains CPU `torch.Tensor` LoRA values, while the wrapped model state contains DTensor parameters; `module.load_state_dict` raised:

`aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!`

The error was reported for 392 LoRA tensors. Therefore there are zero new generation requests, zero probe rollouts, and no valid 2048/8192 response-length or overflow statistics. The exact initial adapter-state restore requirement was not bypassed, so this probe cannot be treated as a partial length result.

## Lifecycle and resources

This run did not reach rollout generation or prefix-cache reset. No CUDA OOM, illegal memory access, `blocks are not freed yet`, `drained=false`, or vLLM worker-death marker was observed. The runner's failure cleanup stopped Ray; final GPU usage was 2 MiB and no relevant train/rollout/Ray/vLLM process remained. Peak observed GPU usage during initialization was 15,970 MiB (16% sampled utilization).

## Previous probe comparison

The prior 1024/4096 probe had 18 response-at-cap events across 120 trajectories and two recorded context-limit events while still completing its saved run. The new run generated no trajectories, so it supplies no evidence about whether 2048 or 8192 removes those events. A FSDP2-safe, hash-verifying snapshot restore is required before retrying; that is a separate engineering change and was not invented or applied in this task.

## Verification and boundary

Relevant tests passed: `19 passed, 1 warning, 34 subtests passed`; Python compilation and shell syntax checks passed. No optimizer, backward, formal training, checkpoint, or external judge call was performed. Raw train logs, temporary config, model/cache, saved replay evidence, and failed-run artifacts remain local and are not part of the tracked handoff. This task stops at the restore blocker and does not authorize a rerun or formal experiment.

See `log/2026-08-28_reward_audit_len2048_probe_results.json` for hashes and machine-readable status.
