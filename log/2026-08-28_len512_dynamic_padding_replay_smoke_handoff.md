# Len-512 dynamic-padding and authentic replay smoke handoff

## Observed facts

- Scope was a diagnostic infrastructure smoke only. No formal baseline, HOB run, benchmark sweep, checkpoint, external LLM call, or DeepSeek call was started. The implementation is in commit `6c6fe91` (delivery/report commit follows).
- Source HEAD before this delivery was `50480e9`, on `experiment/flow-grpo-3b-lora`. The model was `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`; the local role route reported one `qwen-base` engine, `qwen-actor` with the LoRA adapter, and fixed roles as base/no-LoRA.
- Final online run: `unified-qwen7b-fixed-roles-smoke-20260828_20260828_103253`. It completed two training steps and 82 valid actor transitions (`34`, then `48`). The first batch was all-zero reward/advantage; the second was a genuine nonzero-signal batch with reward mean `0.125`, advantage range `[-0.56640625, 1.6953125]`, `pg_loss=-0.00016723014414310455`, and aggregate `grad_norm=0.9227701822916666`.
- Both normal online cleanup events reported `drained=true`, `outstanding_before=0`, `abort_count=0`, then `sleep_started=true`. No successful-run CUDA illegal access, CUDA OOM, prefix-cache reset failure, or deadlock was observed. Final `nvidia-smi` was 2 MiB used and no matching Ray/vLLM/training processes remained.
- The online rollout log contains context-length HTTP 400s for requests whose prompt plus 512-token completion exceeded the fixed vLLM `max_model_len=2048` (examples include requested 2126 and 2249 tokens). These were recorded as runtime errors, not silently counted as successful semantic results; the run nevertheless finished with the stated valid transitions.

## Phase A — 512-cap length audit

`data.max_response_length=512` was passed both to the AgentFlow training batch and, through the rollout override, to vLLM as `max_new_tokens=512`. The vLLM log also recorded `temperature=0.7`, `top_k=-1`, `top_p=1`, and `ignore_eos=false`. `max_model_len` remained 2048.

The audit covers 82 online train transitions from the non-eval `mixed_signal_smoke_4.parquet`. Using nearest-rank percentiles:

| field | mean | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw response tokens | 153.70 | 101 | 209 | 360 | 512 | 512 | 512 |
| prompt tokens | 596.93 | 659 | 847 | 953 | 968 | 1145 | 1467 |

Five responses were exactly 512 (`6.10%`); none was observed above the cap and none was truncated by the local post-processing path. Exact-512 values are censored unless finish reason/EOS proves natural termination. Therefore `512` is the smallest listed cap with zero observed `>cap` samples, but not a general safety proof. The observed context 400s show that response cap and total-context budgeting must be handled together before a formal run. A lower cap such as 384 would trade this risk for a likely non-negligible response truncation rate because the observed p95 is already above 384.

The raw length audit is preserved at `/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/unified-qwen7b-fixed-roles-smoke-20260828_20260828_103253_response_lengths.json` (SHA256 `19718f2344fed20f0e5d2d21d26f802f321d60ed3ca744c517a1d85825a7b72b`).

## Phase B — dynamic response padding

`agentflow/verl/daemon.py` now keeps response sequences unpadded until all valid transitions are collected. With `AGENTFLOW_DYNAMIC_RESPONSE_PADDING=1`, it truncates only at the hard cap and pads the batch to `min(raw_batch_max, 512)`. Masks, token scores/rewards, advantages, returns, old log-probs, and identifiers use the same response width.

Online widths were:

| batch | transitions | raw max | fixed elements | dynamic elements | saved elements | padding ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 512 | 17,408 | 17,408 | 0 | 0.5053 |
| 2 | 48 | 240 | 24,576 | 11,520 | 13,056 | 0.6535 |

The same-data comparison script reconstructs a fixed-width representation from the captured dynamic pack and verifies equality for non-pad response IDs/masks, token scores, rewards, advantages, returns, old log-probs, and `uid`. It reports `13,056` response slots saved (`53.125%` for that batch) and approximately `365,568` bytes saved across the aligned response-width tensors, based on their actual dtypes. This is a tensor-memory accounting result, not a claim of faster model computation; remove-padding remains separately enabled.

## Phase C — authentic Replay Pack and real offline update

The first all-zero warmup batch was deliberately not captured. The capture hook waited for a pre-update batch with nonzero advantage/reward signal and wrote:

- pack: `/root/autodl-tmp/tmp/unified_qwen_fixed_roles_20260828/unified-qwen7b-fixed-roles-smoke-20260828_20260828_103253_authentic_replay_pack.pt`
- pack SHA256: `0a48d4e685f859261448f351a1b286c6a426e45fde69ed0970489084b0329226`
- batch: 48; response tokens: 3992; response reward mean: `0.0005208333604969084`; response advantage mean: `-0.011537724174559116`
- fields include token IDs, masks, old log-probs, token-level scores/rewards, advantages, returns, `uid`, data/rollout/turn identifiers, temperature, and multi-turn metadata.

The fresh offline run `unified-qwen7b-offline-replay-20260828_104133` loaded this pack and invoked the real `update_actor` path. Its log explicitly reports `rollout_requests=0 external_calls=0`; no AgentFlow task was queued. The current VERL worker topology still initializes a local vLLM engine as part of worker construction, but no inference request was made in the offline branch.

Offline evidence: `pg_loss=0.0012388012061516445`, aggregate `grad_norm=0.9849446614583334`, checksum-hook gradient norm `3.59375`, and 196 changed LoRA tensors. LoRA hash changed from `8e5bc87630bce72ff9345f18383f92151e3943f7d41b1b5d97b6f049c5bf1fa8` to `51a9c9cb3521db749714763e76a2d31f2a2e72ad6f8cdea0052af1c9466dd93b`. Thus the offline path performed a real parameter update with zero rollout calls.

The online run also changed 196 LoRA tensors (`920e1228...9365` to `7b8d6e22...3ff0f`) and had nonzero update metrics on step 2. Exact online/offline bitwise replay equivalence was **not** demonstrated: the fresh offline process did not restore the exact online initial LoRA/RNG/optimizer state. The result proves update-path viability, not numerical equivalence.

## PPO and configuration

The resolved key was `actor_rollout_ref.actor.ppo_epochs=2`; `trainer.total_epochs=1`; `save_freq=0`. The same rollout DataProto is reused by the pinned VERL 0.5.0 actor loop for both PPO passes. Per-pass metrics are not separately exposed by this version, so the recorded `pg_loss`, KL, clipfrac, and grad norm are aggregate actor metrics over the update, not two independently reported pass values. No large checkpoint was written.

The online config SHA256 is `2144ecd030fa617ec1f9b147bf3b9da0212d938efd5dac7c282ac7b0a7bbd11a`; the offline config SHA256 is `90a797c66f92ab570323e432fd370a23ba0db391b8e8135203c43e88789f0169`.

## Failures and cleanup evidence

Two preliminary attempts are preserved locally. The first used vLLM utilization `0.10` and failed at vLLM initialization with “No available memory for the cache blocks”. A subsequent attempt encountered host Ray OOM because an old PPID=1 worker from the failed attempt remained alive; that orphan was identified and terminated before retry. The successful `0.60` run then completed cleanly. This is an operational limitation of the single-GPU setup, not a change to the requested model or algorithm. The successful run’s cleanup markers are in its train log; no `drained=false` marker occurred.

## Tests and delivery

Added/changed only the opt-in dynamic-padding, replay-capture, offline-replay diagnostic path, its focused test, and the generic same-pack comparison script. Relevant checks completed before delivery include:

- `python -m unittest test.test_dynamic_padding test.test_unified_smoke_capture test.test_unified_replay_pack`: 6 passed.
- `py_compile` for changed Python modules and `bash -n` for both runners: passed.
- Same-data comparison: `status=ok`, all alignment checks true.
- Final checks include JSON parsing, `git diff --check`, and scoped source secret review; no credentials are stored in the repository.

Raw logs, rollout data, model/cache files, replay pack, checksum files, and GPU traces remain local and are not part of the commit. No formal training is recommended automatically. Before any formal use, first add prompt-plus-completion context budgeting (or otherwise resolve the observed 2048-context 400s), then obtain approval for a separate controlled experiment.
