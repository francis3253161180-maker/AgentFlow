# Random30 fresh rollout + pre-update replay handoff

## Observed facts

- Run: `random30-fresh-rollout-replay-20260828_20260828_115632`.
- Frozen manifest: `log/2026-08-28_random30_len1024_context4096_probe_sample_manifest.json`, SHA256 `f2c8db2b44bf1d8e0879565a2c49215c212090e48cd15d267529f14ea098136d`.
- Model: local `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`; prepared parquet SHA256 `6938703ad24ae934ac211b459ea9aae4b7a7b96d78b5da685b44d74d0d6334a2`.
- Protocol was unchanged for this diagnostic: seed `20260828`, temperature `0.7`, rollout `n=4`, max prompt/response `1536/1024`, vLLM max model length `4096`, dynamic response padding enabled, rollout-only/`val_only`, optimizer steps `0`, checkpoint disabled, external calls disabled.
- 120/120 outer rollouts were valid. There were exactly 30 groups, each with 4 records and unique rollout IDs (120 unique IDs). No trajectory, model cache, snapshot, or replay pack is tracked by Git.
- The local deterministic scorer returned reward `0.0` for all 120 rollouts and all 30 groups. This is recorded as observed output; no scorer rule was changed for this run.
- The evidence aggregate reports 980 replay transitions. Its answer-text field is unavailable (`answer_texts_available=false`), so answer duplicate statistics are not inferred.

## Initial behavior state and routing

- The pre-rollout behavior snapshot was captured before the first request: 392 trainable LoRA tensors, 20,185,088 parameters, LoRA state hash `2f46d9002978cbbf623f28d5113a3d03634246a9332d308d768fc13b86ddf8c9`.
- The snapshot includes Python, NumPy, CPU Torch, CUDA Torch RNG state and explicit seed metadata. Snapshot SHA256: `88be88645ccb737de45c30750d3560255ea5d953d5b0005d185636cf0c6c53a3`.
- The recorded role route was `base_model=qwen-base`, `actor_role=qwen-actor`, with actor LoRA route metadata present. Fixed/base role routing remained local; no external judge/provider was enabled.

## Replay pack

- A reusable authentic pre-update replay pack was built only after all 120 trajectories had been persisted. It contains the required prompts/responses, token tensors, masks, old log-probs, scores/rewards, GRPO advantages/returns, drop masks, IDs and metadata.
- Pack SHA256: `aca07732a172cebb6fef34c2634c68d0dd5674cd462d7a3ed33ef5f4317a6089`.
- Fresh-process validation returned `status=ok`, `evidence_count=120`, `pack_batch_size=980`, `rollout_requests=0`, `external_calls=0`, and matched the snapshot LoRA hash.

## Runtime and cleanup

- GPU peak sampled memory: `23888 MiB / 32607 MiB`; peak observed GPU utilization: `100%` during read-only log-prob replay calculation. Final GPU state: `2 MiB`, utilization `0%`.
- Safe cleanup completed with `drained=1` on both the manager-sleep path and final normal-complete path. In both cases `drain_complete` precedes prefix-cache reset and sleep.
- No CUDA illegal memory access, CUDA OOM, prefix-cache “blocks are not freed” failure, `drained=false`, RayTaskError, or deadlock was observed.
- Two vLLM HTTP 400 context-limit events were recorded for internal role requests (requested 4106 and 4225 tokens against a 4096 context). The outer run recovered and still produced 120/120 valid persisted rollouts; these events are a known limitation, not hidden as zero failures.
- Final process check found no `train_agent`, rollout, Ray WorkerDict, or PatchedvLLMServer process. Raw train/rollout logs remain local for evidence.

## Hypotheses and limitations

- The all-zero reward distribution may reflect the task/scorer/model interaction, but this replay task does not authorize a semantic audit or scorer modification. The captured replay remains valid for later offline analysis only if its reward interpretation is independently reviewed.
- The current aggregate cannot calculate answer-text duplicate rates because the persisted evidence schema exposes tokenized traces but no direct answer-text field. No similarity model or new API was used.
- The two context-limit events indicate that the requested `1536 + 1024` nominal lengths can exceed the 4096 runtime context for some expanded internal prompts. The outer validation and evidence count remain complete, but future long-context runs should treat this as a monitored risk.

## Conclusions and handoff gate

- The requested fresh random30 generation completed: all 30 frozen prompts have four persisted trajectories, and the pre-update replay pack is reloadable and independently validated.
- No optimizer, backward/update, formal training, checkpoint save, external judge, or new generation was performed after the requested rollout/replay workflow.
- This task is complete and stopped. Any replay training or formal experiment requires separate approval; this handoff does not authorize it.

## Reproduction and evidence paths

- Runner: `scripts/run_random30_fresh_rollout_replay_20260828.sh`
- Aggregate/validator: `scripts/aggregate_random30_fresh_rollout_replay_20260828.py`, `scripts/validate_random30_replay_pack_20260828.py`
- Train log SHA256 `f522c63935597081f5092851d3c87a10b9a200306dde25936f0bca6be51daa22`
- Rollout log SHA256 `de46109523afa56324183363105e438c34e5f0e2576e12cd4968324182cc9944`
- Aggregate JSON SHA256 `aa0fbc53b09636c9070d2adfee17f61065fdf928c0c167ca6014dd9e86a8780f`
- Validation JSON SHA256 `fd75ebe41ab4698bed03380165e0d2af1560e9f0d9c8de8c82c500e0b7ead6af`
- Length audit SHA256 `1cdf523dd3710231ac134115f81f5f35b1c8793825c9fa2836f073af67ffbcf3`
- Evidence-file manifest SHA256 `68be50fd9e26e0821d18786c5a459c27d96ab731a5ddfe46c18c5ca562d84a19`
