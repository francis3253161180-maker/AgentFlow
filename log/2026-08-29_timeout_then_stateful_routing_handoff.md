# Timeout then stateful routing smoke — 2026-08-29

## Observed facts

- The previous n=4 result was confounded: three one-step trajectories had persisted execution times of 194.41–229.52 seconds against `AGENT_MAX_TIMEOUT=180`, while their sole verifier response was `CONTINUE`.
- Phase A initially exposed a separate runner limit: a 4096-token vLLM context rejected third-step verifier prompts (HTTP 400; 4314–4489 requested tokens). I stopped that invalid retrying attempt, preserved its local logs, and changed only the smoke runner/config override to `max_model_len=8192`.
- Clean Phase A then used the same frozen MuSiQue row (`id=4`, source index 259, `2hop__13592_49388`), Qwen2.5-7B actor LoRA, Qwen-base adapter-off fixed roles, n=4, `TOOL_STEPS=3`, `AGENT_MAX_TIMEOUT=600`, and no optimizer/checkpoint or external LLM calls.
- Phase A completed 4/4 valid rollouts with zero retries. All four paths were Wikipedia → Wikipedia → Wikipedia; all twelve verifier decisions were `CONTINUE`; every trajectory terminated at `max_steps`, not `max_time` or verifier STOP. Eight verifier-CONTINUE decisions were followed by another planner action. Times were 33.09, 27.71, 31.03, and 30.81 seconds. Rewards were `[0, 0, 0, 0]`.
- This is a clean stagnation case: e.g. Phase-A trajectory 1 repeated the same joint league-and-games sub-goal and the same FC Barcelona/list-record URLs at steps 2–3. Other trajectories had material query wording changes but still retained the same unresolved evidence gap with known URLs. Raw trajectory JSON preserves every planner context/sub-goal, executor command, returned URLs/evidence, verifier response, final answer, and reward at the untracked locations in the results JSON.
- Phase B added one generic, advisory routing-state snapshot to every Planner next-action prompt: prior verifier assessment, prior tool/sub-goal signature, and URLs extracted from Memory. The prompt explicitly permits repeated use for a genuinely new entity/sub-goal and treats a known URL as a deep-read candidate; it does not force Web_RAG or forbid Wikipedia.
- Phase B completed 4/4 valid rollouts with zero retries, all routing-state snapshots present, and clean normal completion (`drained=True`). It again produced four Wikipedia → Wikipedia → Wikipedia paths, 12 verifier `CONTINUE` decisions, four `max_steps` terminations, `[0, 0, 0, 0]` reward, and zero Wikipedia→Web_RAG transitions. Peak GPU memory was 20,099 MiB in A and 20,609 MiB in B. There were no CUDA, prefix-cache, OOM, Ray, 429, OpenAI, or Doubao markers.

## Hypotheses

- The timeout was a real confounder for the earlier one-step outputs, but it was not the cause of the clean three-step all-Wikipedia behavior.
- On this exact actor/prompt/tool configuration, merely making verifier feedback and known URLs explicit is insufficient to make the actor regard deep-reading as preferable to another Wikipedia discovery query.
- One question and four stochastic rollouts are diagnostic evidence, not a general routing benchmark.

## Conclusions

- The Phase-A causal question is resolved: a verifier `CONTINUE` does lead to another Planner action when both time and steps remain.
- A clean stagnation pattern remains after removing the 180-second confounder.
- The single authorized generic state-visibility change was correctly routed into the live prompts but did not improve this case. The bounded task therefore stops here; no hard tool ordering, no forced multi-tool diversity, no scorer change, and no further retry was made.

## Recommendation

- Keep the termination telemetry and state snapshot as auditable infrastructure, but do not claim the snapshot solves routing.
- Before broader experiments, seek approval for a separate design decision: either improve how the actor is trained/prompted to value evidence transitions, or evaluate a diverse multi-hop probe set. Do not infer that a generic forced Wikipedia→Web sequence would be correct.
- Raw rollout/log evidence remains local and untracked. The committed small results file is [2026-08-29_timeout_then_stateful_routing_results.json](2026-08-29_timeout_then_stateful_routing_results.json).
