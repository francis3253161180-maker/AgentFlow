# Final LoRA checksum + authentic Replay Pack smoke handoff

## Scope and boundary

This was the final infrastructure smoke requested before any formal baseline. It reused the existing non-eval four-row mixed-signal smoke set with Qwen2.5-7B-Instruct, the unified local-role route, BF16 FSDP2/offload, one vLLM base engine, and external LLM/judge disabled. No formal baseline, HOB run, benchmark evaluation, checkpoint, or new rollout sweep was started.

Run id: `unified-qwen7b-fixed-roles-smoke-20260828_20260828_093406`.

## Gate 1 — direct LoRA parameter-change evidence: PASS

The pinned VERL actor worker now invokes an opt-in capture hook at the real `DataParallelPPOActor.update_policy()` entry and immediately after the first gradient-bearing `_optimizer_step()`. Only `requires_grad` parameters whose canonical name contains `lora_` are hashed, in sorted name order; each digest includes name, shape, dtype, and raw CPU tensor bytes.

Evidence from step 2:

- `pg_loss=0.0002660676836967468`, `grad_norm=0.51171875` in the capture record (the aggregated step line reports `grad_norm=0.59619140625`); both are nonzero.
- 392 LoRA tensors, 20,185,088 total elements.
- pre hash `157a12bb25be81199d718dd83df0c2f086467ee8803fabc99c66ee6cfd0c888b`.
- post hash `95eaa9b123292ab203182579b7cd0a443733994155e656048bb6946cd9748d34`.
- hashes differ; 196 per-tensor hashes changed.

Step 1 was all-zero reward/advantage and therefore had zero gradient. Step 2 had reward mean 0.25 and advantages from -0.9765625 to 0.9765625, so the observed parameter change is tied to a real GRPO signal rather than an initialization-only checksum.

## Gate 2 — authentic runtime Replay Pack: PASS

The pack was captured from the actual `DataProto` immediately before the actor policy update, before VERL's internal field selection/splitting. It is a `torch.save` serialization of runtime tensors and arrays, not an export reconstructed from rollout JSON.

Pack: `/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/unified-qwen7b-fixed-roles-smoke-20260828_20260828_093406_authentic_replay_pack.pt`

SHA256: `fd0f12c4b5bef1cf8ec07d378b2472a6c4e98ef6ec2b7602e30838e0250a085f`.

Token ids are present (`input_ids`, `responses`, and `prompts`) with preserved dtype/shape. The captured update tensors include `response_mask`, `old_log_probs`, `advantages`, `returns`, `token_level_rewards`, and `token_level_scores`; non-tensor group identity includes `uid`, plus data/rollout/turn identifiers. The pack also preserves `attention_mask`, `position_ids`, `is_drop_mask`, runtime meta-info, route snapshot, model path, temperature, rollout n, seed, scorer mode, and the LoRA pre-update hash.

The fresh validator process checked the captured field digest (`0258031c6f6c0adf7cb253e584e23ea9e18a4cc62763e55bf86e69ff157a7a35`), saved/reloaded a second copy, and reported field-by-field equality. Its offline dry-run inspected masks, old log-probs, rewards, and advantage shapes without model inference, rollout generation, optimizer access, or external calls. The pack is the first real pre-update batch and is all-zero reward; the later step-2 batch is the one that provided the nonzero update. This is a limitation of capturing one pack, not fabricated data or a failed authenticity check.

## Runtime role and safety evidence

The run used `qwen-base` for fixed-role requests and `qwen-actor` for the synchronized LoRA route. Rollout telemetry counted 75 `qwen-base` and 23 `qwen-actor` requests. The client-side log showed five fixed-role engine clients and one actor client, all pointing to the same local vLLM base endpoint; no duplicate fixed-model process was observed. External judge/reward calls were disabled by environment and no DeepSeek call was made.

The two normal-completion cleanup markers both report `drained=true`, `outstanding_before=0`, `abort_count=0`, `abort_errors=0`, then sleep. No CUDA OOM, illegal memory access, prefix-cache reset race, deadlock, or worker death occurred. FSDP reported 22.225 GiB max allocated and 25.621 GiB max reserved; sampled GPU peak was 21,034 MiB. Final `nvidia-smi` was 2 MiB and no matching train/Ray/vLLM processes remained.

The existing launcher emitted one non-fatal `lsof`-not-found message while attempting to clear port 9998. It did not affect the run, and is recorded as a tooling warning rather than a model/lifecycle failure.

## Changes and verification

- Added opt-in capture module: `agentflow/verl/unified_smoke_capture.py`.
- Added runtime worker backport patch: `patches/verl_unified_replay_capture.patch`; applied to the pinned VERL site-package `verl/workers/actor/dp_actor.py`.
- Extended `scripts/run_unified_qwen_fixed_roles_smoke_20260828.sh` with per-run checksum/pack paths and fresh-process validation.
- Added `scripts/validate_unified_replay_pack_20260828.py`.
- Added `test/test_unified_smoke_capture.py`.
- Focused suite: 25 tests passed, including scorer, role routing, replay-capture, and cleanup tests.
- `py_compile`, `bash -n`, JSON validation, and scoped secret review passed.

The large raw logs, rollout data, model/cache files, checksum, and Replay Pack remain local and are not committed. The small report/results/source/test/patch artifacts are the only intended additions. The final artifact commit SHA is supplied in the completion handoff after commit/push; the code was based on `b2bf488`.

## Recommendation

Both evidence gates pass for the 7B unified architecture smoke. The result supports proceeding to an independently approved next experiment design, but does not authorize or start formal training. In particular, the authentic pack currently represents an all-zero first update batch, so any future replay-based analysis that requires a mixed batch should capture a separate explicitly selected runtime batch under the same opt-in instrumentation.
