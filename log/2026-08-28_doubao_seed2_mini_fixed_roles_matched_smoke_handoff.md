# Doubao Seed 2.0 mini fixed-role matched smoke handoff

## Scope and boundary

The previously running `doubao-seed-2-0-lite-260428` matched smoke was stopped on request and preserved as interrupted evidence. It produced only 3 partial trajectories (`[1, 0, 1]`) and is not included in the new metrics. No formal training, backward pass, optimizer update, checkpoint, DeepSeek call, GPT call, or HOB work was performed.

## New live sanity check

The exact model `doubao-seed-2-0-mini-260428` was called once through the existing OpenAI-compatible Ark client at the configured official endpoint. The request used `temperature=0`, `max_tokens=8`, and a short exact-response probe. It succeeded in approximately 1352 ms. The API key was present but was never printed or persisted.

## Matched smoke result

The new run completed all 8 fixed groups and 32/32 valid rollouts with zero retries and zero errors. Every reward was `1.0`: reward mean `1.0000`, group distribution `0/4=0`, `1/4=0`, `2/4=0`, `3/4=0`, `4/4=8`, and mixed groups `0/8 (0%)`. This is a small GameOf24 matched smoke and does not establish generalization or training benefit.

Runtime was approximately 23m42s. GPU peak was 16,998 MiB / 32,607 MiB and final usage was 4 MiB. The actor route was local Qwen2.5-7B-Instruct with the synchronized LoRA adapter. `planner_fixed`, `verifier`, and `executor` used the exact requested Doubao mini model and remained frozen external roles; no separate reference model was instantiated because KL was disabled.

Raw trace evidence counted 32 `qwen-actor` requests and 160 exact `doubao-seed-2-0-mini-260428` requests. The local GameOf24 deterministic scorer handled all 32 rewards; DeepSeek fallback and judge cache were both unused. All 32 structured-output validations passed.

## Lifecycle and safety evidence

The patched vLLM lifecycle emitted two cleanup sequences: initial manager sleep and final normal completion. Both reported drain complete, prefix-cache reset complete, sleep complete, and `complete=1 drained=1`. The final cleanup had `outstanding_before=0`, `abort_count=0`, `abort_errors=0`, and duration about 0.680s. No CUDA illegal memory access, CUDA OOM, prefix-cache “blocks are not freed” warning, deadlock, worker death, HTTP 4xx/5xx, provider not-found error, or parse error was observed. After completion no train, Ray, or vLLM process remained and the GPU was idle.

The harness stdout still labels its mode line `external_calls=disabled` even when the explicit fixed-role override is enabled; this is stale bookkeeping text. The authoritative evidence is the role-routing line and the 160 exact Ark model traces in the rollout log. No behavior or result was inferred from that stale field.

## Verification

The existing partial provider changes were retained because they are required for explicit model-configured Ark routing. The exact model is supplied at runtime via `AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE`; no secret or model credential was added to the repository.

Results JSON: `log/2026-08-28_doubao_seed2_mini_fixed_roles_matched_smoke_results.json`.

The raw run metadata, route state, GPU TSV, rollout log, train log, and trajectory directory are listed in that JSON and remain local/untracked. The old interrupted run remains local as separate evidence.

## Conclusion

The requested Doubao mini fixed-role integration is live and stable for this 8×4 smoke, and it changes the observed matched GameOf24 outcome from the Qwen fixed-role reference’s `0/32` to `32/32`. Because this is a tiny rollout-only sample and fixed roles are external, it is not evidence to start formal GRPO. Stop here pending approval; do not begin the 60-prompt baseline or any HOB experiment automatically.
