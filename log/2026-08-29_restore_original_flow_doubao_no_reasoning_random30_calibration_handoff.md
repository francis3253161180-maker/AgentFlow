# Restore-original-flow Doubao fixed-role calibration handoff

## Observed facts

- The approved calibration was not completed. The target was 30 fixed Game24 prompts × 4 rollouts (120 logical rollouts), rollout-only, with no optimizer update, backward pass, checkpoint, HOB, or formal GRPO training.
- The fixed manifest was not changed: `log/2026-08-28_random30_len1024_context4096_probe_sample_manifest.json`.
- The intended runtime configuration was Qwen2.5-7B-Instruct actor + LoRA (rank 8, alpha 16, all-linear) with temperature 0.7 and `rollout.n=4`; fixed roles were `doubao-seed-2-0-lite-260428`, temperature 0.0, and `ARK_REASONING_EFFORT=minimal`.
- The first run (`20260828_232358`) entered the standard AgentFlow path and produced 13 partial rollout evidence files: 3 complete prompt groups and 1 partial group. All 13 recorded rewards were 0.0. This is not a complete group-level result and is excluded from calibration statistics.
- Partial trace telemetry from that run recorded 19 `qwen-actor` requests at temperature 0.7 and 83 fixed-role requests at temperature 0.0, all fixed-role requests using exactly `doubao-seed-2-0-lite-260428`. The rollout triplets identify `planner_main` as `trainable: true`, `model_name: qwen-actor`.
- The run route state registered the actor LoRA adapter (`adapter_id=2067897179`, version `1787930736948226623`). The captured behavior snapshot contains 392 LoRA tensors; its recorded hash is `bd73d2887403e45266eee0843b906ca60921b450a3516c6a29bd10ff81554311`.
- A prior explicit Ark sanity request succeeded with `reasoning_effort=minimal`, model `doubao-seed-2-0-lite-260428`, and observed `reasoning_tokens=0`; the API key was not printed or stored.
- The first run was later lost at the session boundary without a complete-run marker. Its final device check showed 0 MiB GPU use, but a normal end-of-run aggregate/validation was not produced.
- A detached retry (`20260829_091347`) loaded the 7B actor and applied LoRA, then stopped producing logs after `NCCL version 2.26.2+cuda12.2`. The actor worker and TaskRunner remained waiting for approximately six minutes with no rollout files. It was terminated with SIGTERM and its cleanup completed.
- A second retry (`20260829_092236`) used only the single-GPU compatibility environment deltas `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`. It reproduced the same post-NCCL initialization stall before any rollout, so it was terminated safely. No rollout files were created by either retry.
- Fatal scans of the preserved logs found no CUDA OOM, CUDA illegal memory access, prefix-cache reset race, `drained=false`, Ray worker death, or API/parse error. The observed failure is an initialization stall, not a proven memory OOM.
- After cleanup, `nvidia-smi` reported 0 MiB used and no AgentFlow train process, Raylet, or vLLM worker remained. The machine was not shut down.

## Hypotheses

- The two repeated stalls at the same actor initialization boundary are consistent with a FSDP2/NCCL single-GPU initialization compatibility or lifecycle issue in this environment. The evidence does not isolate whether the trigger is process startup ordering, distributed initialization, or an interaction with this VERL configuration.
- The first run's partial progress shows that the same code path can initialize and serve requests at least once, but it does not establish stable reproducibility.
- The stale reference-model path printed by the base configuration is not evidence of a call: partial traces contained only the requested Qwen actor and Doubao fixed-role model. It should nevertheless be cleaned up separately before a future run if it can be done without changing semantics.

## Conclusions

- The requested actor/fixed-role split is verified for the partial run: Qwen2.5-7B + trainable LoRA for `planner_main`; Doubao lite for `planner_fixed`, `verifier`, and `executor`; fixed roles are frozen in the configured role map. No DeepSeek/Doubao reward judge was used; Game24 reward was local deterministic checking.
- The 30-group calibration result is **blocked and unavailable**. No claim about reward mean, mixed groups, or group distribution is scientifically valid from the 13 partial rollouts.
- Qwen2.5-7B + LoRA is not yet demonstrated as a stable full calibration on this host. The two post-NCCL stalls are sufficient reason not to start another unapproved retry or switch models silently.

## Code and evidence

- Source changes: `agentflow/agentflow/engine/ark.py`, `test/test_ark_provider.py`, and `scripts/run_random30_fresh_rollout_replay_20260828.sh`.
- Partial evidence run: `log/random30-fresh-rollout-replay-20260828_20260828_232358_{train,rollout}.log` and `/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828/random30-fresh-rollout-replay-20260828_20260828_232358_trajectories/`.
- First stalled retry: `log/random30-fresh-rollout-replay-20260828_20260829_091347_train.log`.
- NCCL-disabled retry: `log/random30-fresh-rollout-replay-20260828_20260829_092236_train.log`.
- Retry driver evidence: `log/restore_original_flow_doubao_no_reasoning_random30_20260829_driver.log` and `log/restore_original_flow_doubao_no_reasoning_random30_20260829_nccl_retry_driver.log`.
- Machine-readable status: `log/2026-08-29_restore_original_flow_doubao_no_reasoning_random30_calibration_results.json`.

## Recommendation

Do not use the partial run as a calibration result and do not begin formal training. The next approved action should be a narrowly scoped CPU/single-GPU FSDP initialization diagnosis (with the exact same Qwen7B actor and frozen Doubao role configuration) before another 120-rollout attempt. Any future run must again verify the actor adapter route and fixed-role model IDs from runtime traces before being counted.
