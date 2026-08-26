# vLLM timeout cleanup smoke handoff

## Scope and provenance

- Branch: `experiment/flow-grpo-3b-lora`
- Source parent before this change: `950c7236a5b180910360137328d657e886c5099f`
- Model/runtime: Qwen2.5-3B-Instruct, LoRA rank 8 / alpha 16, one RTX 5090, async vLLM.
- Installed versions: `verl==0.5.0`, `vllm==0.9.2`, `torch==2.7.1+cu128`.
- No dependency, model, LoRA, FSDP, optimizer, or checkpoint configuration upgrade was made.
- The final implementation commit SHA is supplied with the Git handoff; this report is part of that commit.

## Observed facts

The installed VERL 0.5.0 `AsyncLLMServerManager` exposes only `wake_up()` and `sleep()`. Its old async vLLM server `sleep()` directly called `reset_prefix_cache()` followed by optional engine `sleep()`; it had no drain or abort step.

The installed vLLM 0.9.2 `AsyncLLM` exposes per-request `abort(request_id)`, `output_processor.request_states`, `get_num_unfinished_requests()` / `has_unfinished_requests()`, `reset_prefix_cache()`, and `sleep()`. It does not expose the newer `wait_for_requests_to_drain()`, `abort_all_requests()`, `pause_generation()`, or `resume_generation()` API. The backport therefore uses the current per-request abort and state polling surface.

Forced-timeout smoke:

- 8 prompts, 32 training-mode rollout tasks (`rollout.n=4`), temperature `0.7`.
- `val_only=true`, `trainer.save_freq=0`, no optimizer/update path.
- Test-only wait timeout: 180 seconds; cleanup drain timeout: 30 seconds.
- One test-only completed-response HTTP handler was held after four requests. This made the outer request lifecycle deterministic without occupying vLLM generation slots.

Key lines from `log/20260826_vllm_timeout_cleanup_forced_train.log`:

```text
Timeout after 3.0 minutes. Completed 9.4% (3/32)
Validation summary: 3/32 total rollouts (9.4%), 3 valid rollouts
VLLM_CLEANUP_DRIVER reason=wait_timeout results=[{'drained': False, 'outstanding_before': 1, 'outstanding_after': 0, 'abort_count': 0, 'abort_errors': 0, 'sleep_started': False, ...}]
VLLM_CLEANUP_HEALTH_CHECK status=ok results=[{'status': 'ok', 'outstanding': 0}]
VLLM_CLEANUP drain_complete=0 outstanding_after=0 active_http=1 abort_errors=0 sleep=skipped reason=drain_timeout
```

The forced path observed a nonzero in-flight lifecycle count (`outstanding_before=1`, `active_http=1`). The held request had already completed engine generation, so this live run had `abort_count=0`; the real-style request-id abort path is covered by the no-GPU unit test.

Normal mini smoke:

- 4 prompts, 16 training-mode rollout tasks (`rollout.n=4`), temperature `0.7`.
- 16/16 completed and valid; no timeout, traceback, CUDA illegal memory access, or OOM.
- Cleanup returned `drained=true`, `outstanding_before=0`, `abort_count=0`, then performed prefix-cache reset and sleep.

Resource evidence:

- Forced GPU peak: 20,264 MiB / 32,607 MiB.
- Normal mini GPU peak: 20,268 MiB / 32,607 MiB.
- Final `nvidia-smi`: about 2–4 MiB used, 0% utilization.
- Final process check: no train, rollout, Ray, vLLM, or AgentOps process remained.
- No checkpoint was written for either smoke experiment.

Raw smoke logs retained locally:

- `log/20260826_vllm_timeout_cleanup_forced_train.log`
- `log/20260826_vllm_timeout_cleanup_forced_rollout.log`
- `log/20260826_vllm_timeout_cleanup_mini_train.log`
- `log/20260826_vllm_timeout_cleanup_mini_rollout.log`
- Earlier runner/config failure copies are retained locally with the `_dir_failure.log` suffix and are not part of the accepted result.

## Hypotheses

The original crash was a lifecycle race: timeout returned while rollout-side work could still be active, then the trainer cleared state and called the old `sleep()` implementation, which reset the prefix cache before all work had quiesced. The observed `Failed to reset prefix cache because some blocks are not freed yet` followed by CUDA illegal memory access is consistent with that ordering.

## Code changes

### `agentflow/verl/async_server.py`

- Added actor-local request acceptance and active HTTP request gates.
- Added `cleanup(reason)`: reject new requests → snapshot vLLM state → abort each request with vLLM 0.9.2 `engine.abort(request_id)` → bounded polling until engine and HTTP layers are idle → reset prefix cache → optional engine sleep.
- If drain does not complete, cleanup returns without reset or sleep and logs the conservative skip path.
- Added cleanup trigger, outstanding count, abort/drain, reset, sleep, skip, duration, wake, and health-check logs.
- Overrode `sleep()` so legacy manager callers receive safe cleanup behavior.
- Added a smoke-only request hold hook, disabled by default.

### `agentflow/verl/trainer.py`

- Replaced direct `async_rollout_manager.sleep()` calls in validation and training-cycle cleanup with a helper invoking server `cleanup()` before clearing AgentFlow state.
- Cleanup runs in `finally` blocks, including timeout/partial-result paths.
- Added test-only post-cleanup wake/health-check support.
- Normal training and optimizer semantics are unchanged.

### `agentflow/verl/daemon.py` and `agentflow/server.py`

- Added a test-only wait-timeout override; the production dynamic timeout remains unchanged unless explicitly overridden by environment.
- Timeout, early-completion, and accepted no-progress exits stop new AgentFlow task intake.
- The task queue rejects new work while draining and re-opens at the next rollout cycle.

### Tests and runner

- Added `test/test_vllm_timeout_cleanup.py` for request-id abort ordering, active-handler safe skip, and task-queue rejection.
- Added `scripts/run_vllm_timeout_cleanup_smoke_20260826.sh` for reproducible forced and normal rollout-only smokes.
- No site-packages file was modified; no `patches/verl_vllm_timeout_cleanup_backport.patch` is required.

## Conclusions

1. Reset/sleep is now reachable only after a successful drain. If drain cannot be proven, reset/sleep is skipped.
2. The forced timeout path observed an active request lifecycle, safely took the bounded skip path, and passed the post-cleanup wake/health check.
3. The normal mini path completed 16/16 valid rollouts and exercised drain → reset → sleep.
4. No CUDA illegal memory access, CUDA OOM, deadlock marker, or `blocks are not freed yet` message appeared in the accepted forced/mini logs.
5. This was lifecycle repair and smoke validation only; it did not run training or change Flow-GRPO semantics.

## Remaining limitations

- The forced live smoke held an HTTP handler after vLLM generation, so it validated the outer “do not reset while an active request handler remains” boundary but did not capture a live nonzero vLLM `request_states` set at the exact cleanup instant. The unit test verifies two request IDs are aborted before reset/sleep.
- The forced smoke intentionally accepted 3/32 partial results under a smoke-only 1% completion threshold; these are cleanup evidence, not quality or training evidence.
- No full 100-prompt audit was run in this task.
- A non-AgentFlow custom server still has a compatibility fallback to the old manager sleep; the configured production path uses `PatchedvLLMServer` and the new cleanup RPC.

## Verification commands

```text
python -m unittest -v test.test_vllm_timeout_cleanup       # 3 passed
python -m unittest -v test.test_reward_scorer               # 9 passed
python -m py_compile agentflow/verl/async_server.py agentflow/server.py agentflow/verl/daemon.py agentflow/verl/trainer.py test/test_vllm_timeout_cleanup.py
bash -n scripts/run_vllm_timeout_cleanup_smoke_20260826.sh
git diff --check
```

Secret review found no API key, token, or credential added to the tracked diff. The smoke runner reads the existing environment file but does not print or persist its contents.

## Recommendation

The forced and normal lifecycle smokes pass the requested safety criteria, so a new controlled 100-prompt rollout-only audit may proceed under the existing model and sampling configuration. It should remain rollout-only, preserve `save_freq=0`, monitor the new cleanup markers, and stop if cleanup reports `drained=false` outside the intentional forced-smoke test or if any CUDA/Ray/vLLM error recurs. This is a handoff recommendation only; no 100-prompt audit was started automatically.
