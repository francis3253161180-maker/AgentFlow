# FSDP2 restore + role audit handoff

## Observed facts

The first 2048/8192 attempt stopped before generation because the old restore path loaded ordinary CPU tensors into already wrapped FSDP2 DTensor parameters (`aten.copy_.default: got mixed torch.Tensor and DTensor`). The source behavior snapshot is the saved random30 snapshot, SHA256 `88be88645ccb737de45c30750d3560255ea5d953d5b0005d185636cf0c6c53a3`, with 392 LoRA tensors and expected hash `2f46d9002978cbbf623f28d5113a3d03634246a9332d308d768fc13b86ddf8c9`.

After the fix, the exact retry completed all 8 selected groups and 32 outer rollouts. No optimizer update, backward pass, checkpoint, external call, CUDA illegal access, CUDA OOM, or prefix-cache reset failure occurred. Final GPU memory was 4 MiB and no relevant Ray/vLLM/training process remained.

## FSDP2-safe restore fix

`agentflow/verl/unified_smoke_capture.py` now separates restore from verification. `restore_behavior_snapshot()` copies the 392-tensor CPU snapshot into the ordinary PEFT actor before `apply_fsdp2()`. The pinned VERL 0.5.0 `verl/workers/fsdp_workers.py` calls this pre-wrap and records the pre-FSDP marker. Its registered post-wrap method only calls `verify_behavior_snapshot()`, which hashes the wrapped actor and never performs a CPU Tensor-to-DTensor copy. The reproducible site-package backport is recorded in `patches/verl_behavior_snapshot_backport.patch`.

Evidence in the successful train log shows both `PRE_FSDP status=restored` and post-wrap `status=verified`, each with 392 tensors and the exact expected hash. The focused test also recreates the 392-parameter PEFT tree and verifies the hash after restore.

## 2048/8192 probe

The retry used the frozen 8-group manifest, Qwen2.5-7B-Instruct, temperature 0.7, rollout.n=4, max prompt 3072, max response 2048, max model length 8192, dynamic response padding, vLLM utilization 0.50, max sequences 1, max batched tokens 1024, rollout/validation-only mode, optimizer steps 0, and checkpoint disabled. The successful run had 32/32 evidence files and 8 groups of four. All stored rewards were 0, hence 8/8 groups were 0/4 and there were no mixed groups. The persisted evidence contains 260 internal role requests: all finished with `stop`; response-token mean/p50/p95/max were 217.5/167.5/509.25/740, with zero responses at the 2048 cap. Prompt-token mean/p50/p95/max were 938.3/804.5/2079.95/2609, and context-overflow/HTTP-400 events were zero.

The scorer was deterministic-only for this intentionally external-disabled probe (32 deterministic, zero judge fallback, zero cache hit, zero API/parse errors). There were 147 local AgentFlow tool-response JSON parse warnings in the raw rollout log; they did not prevent complete evidence files and are distinct from scorer/API errors. Retry markers were absent and the run reported no invalid rollout.

Cleanup evidence is ordered as `abort_start` -> `drain_complete=1` -> `reset_prefix_cache complete=1` -> `sleep_complete=1` -> `complete=1 drained=1`, both before wake/generation and after normal completion. Peak GPU memory was 16,992 MiB (peak sampled utilization 79%). The run lasted approximately 14:12:04–14:29:46 UTC.

## Offline role-level audit

The original 120 saved random30 trajectories were decoded with the local Qwen tokenizer and checked with the existing independent Fraction arithmetic oracle. No rollout was regenerated and no model/API was called. The historical cause summary remains 95 wrong-number-multiset, 23 invalid/no-expression, and 2 context-error-affected rows. The role script’s direct oracle categories are 96 wrong-number-multiset and 24 invalid/no-expression because the two context-error rows overlap those categories.

The first *observable* non-target expression stage among wrong-number cases was: planner_main 1, planner_fixed 1, verifier/revision 19, final assembly 38, and final extraction 37. The 24 invalid/no-expression cases were attributed to final assembly/extraction in 17 cases and insufficient evidence in 7. These counts identify where deterministic parsing first sees evidence; they are not causal proof. In particular, the persisted trace has repeated malformed arithmetic prose, untagged answers, and `agent_name="*"`, so role names are inferred from stable AgentFlow prompt templates. The full 120-row local audit is retained outside Git at `/root/autodl-tmp/tmp/reward_audit_len2048_probe_20260828/random30_role_failure_audit.json` (SHA256 `00e2c74c7eba045ce0ca25f828b42801f10d305d061022fb3dc72e435aec0370`).

The independent oracle found a known solution for all 30 groups, and the current generic scorer accepts all 30 known-valid constructions in the prior offline check. This supports the earlier conclusion that fixed Qwen7B role generation and final-answer assembly, rather than an unresolved snapshot restore problem, are the dominant bottlenecks in this saved random30 behavior. It does not establish that any one role is the sole cause of the final reward.

## Conclusion and recommendation

The FSDP2-safe snapshot restore is proven for all 392 LoRA tensors: exact hash match before FSDP2 and exact hash verification after wrapping. The 8-group 2048/8192 diagnostic is complete, uncapped, and lifecycle-clean. The role audit shows predominantly invalid final arithmetic/extraction, with a smaller but nontrivial set of verifier/planner-stage bad candidates; fixed-role Qwen7B execution is not shown to be a memory bottleneck in this run.

The probe should not be repeated for restore debugging. A future architecture/model experiment may investigate structured final-answer extraction or role prompts, but this handoff deliberately makes no scorer rule change and starts no training. The full 100-prompt audit is not recommended until a separate controlled decision is made about the observed zero-reward behavior and final-answer quality.

## Verification and evidence

Tracked code/test changes are limited to the pre-FSDP restore, post-FSDP hash-only verification, the focused 392-tensor test, the reproducible role-audit script, this handoff, and the small results JSON. Raw rollout files, snapshots, caches, and large logs remain local and untracked. Tests: `python -m pytest -q test/test_unified_smoke_capture.py test/test_reward_scorer.py test/test_unified_replay_pack.py test/test_vllm_timeout_cleanup.py` -> 20 passed, 1 warning, 34 subtests passed; `py_compile`, `bash -n`, `git diff --check`, and scoped secret scan passed.

Key local logs:

- `log/reward-audit-len2048-context8192-20260828_20260828_141154_train.log` SHA256 `e3e0b0e84c15d145b5fb2f3a798ea7504f177a2ee38d6bc06357a249d778ce70`
- `log/reward-audit-len2048-context8192-20260828_20260828_141154_rollout.log` SHA256 `119b4805ed247c1c6481ac1079a07c303069af3b08c0706e49ef9748d7ee2018`
- aggregate JSON SHA256 `a908de96c5c4be7183e0b9117294af99fcad476cc8499e57f8efb2a709c68991`
- GPU log SHA256 `de3c9fe8525bf1cd6e061047730224a8ee7e13f23e2fc4c0e137310521659aa9`
