# Doubao fixed-role isolation calibration

## Scope

This was the one authorized frozen Barcelona MuSiQue calibration: `n=4`, four
workers, three tool steps, 600 seconds, rollout-only. No optimizer, backward,
checkpoint, training, extra sample, or second live retry was run. Raw logs,
temporary cache, configuration, and rollout directory remain local; compact
evidence is in `2026-08-29_doubao_fixed_role_isolation_results.json`.

## Observed facts

### Preserved all-Qwen baseline

The immediately preceding `8fac51a` baseline remains unchanged: 4/4 valid
rollouts, reward `[0,0,0,0]`, one coverage revision producing a valid two-step
dependency plan in every trajectory, no completed step 1 or activated step 2,
two step-3 Web RAG choices, no unsupported final claim, seven shared cache
hits, and zero MediaWiki 429. Retrieval/evidence quality remained the
bottleneck.

### Role-routing audit and scoped fix

The prior wiring had two coupling defects: Planner used its fixed engine for
query analysis, high-level planning/coverage, and final generation; Verifier
constructed its fixed engine from `planner_fixed_engine`, not its own role.
The new opt-in supervisor route is:

| Function | This calibration |
| --- | --- |
| Planner-main action choice | local `vllm-qwen-actor` with synced LoRA |
| Query analysis | local `vllm-qwen-base` |
| High-level evidence plan / coverage | `doubao-seed-2-0-lite-260428` |
| Step verification | `doubao-seed-2-0-lite-260428` |
| Executor / tools / final generator | local `vllm-qwen-base` / deterministic raw tools |

Only the indicated Doubao model was allowlisted. Temperature was 0.0 and
`ARK_REASONING_EFFORT=minimal`, the requested Seed 2.0 no-thinking setting.
No DeepSeek, OpenAI, Google, or tool-internal Doubao route was enabled.
`ChatArk` records request metadata without credentials and the focused mocked
provider test confirms that `reasoning_effort=minimal` is passed in the request.

The code now audits structural leakage from supervisor/verifier outputs (tool
names, URLs, query/command syntax, answer markers, and arithmetic absent from
recorded evidence). High-level planning/coverage outputs with such an
action/answer channel are rejected. This is not a factual-correctness judge.
Planner-main receives only sanitized verifier state and provenance already in
Memory, never free-form verifier rationale. The deterministic provenance gate
from `8fac51a` remains authoritative.

### One live calibration

All four logical slots performed local Qwen query analysis, then failed at the
post-response high-level-supervisor boundary with
`ValueError: supervisor high-level plan violated capability boundary`. The
daemon reported 0/4 valid rollouts. Combined local logs contain eight duplicate
stack-marker lines; the authoritative outcome is four invalid logical slots,
not eight independent examples.

The exception happens after the supervisor call returns, so a high-level
supervisor response was received for each failed slot. Coverage and step
verification were not reached. Because the safety exception predated
trajectory persistence, no plan/map/tool sequence/verifier state/final answer
or per-call request metadata was written. There is therefore no valid reward,
HTTP/cache statistic, or outcome comparison against the all-Qwen baseline.

GPU peak was 20,095 MiB. Normal lifecycle cleanup recorded `outstanding=0`,
`drained=true`, then cache reset/sleep; no CUDA OOM, illegal memory access,
prefix-cache failure, or deadlock occurred. A clearly attributable PPID=1
`ray::WorkerDict` (PID 343455) remained after the failed run; SIGTERM did not
exit it within five seconds, so SIGKILL was used. GPU ended at 0 MiB.

## Hypotheses

- The Doubao plan likely contained at least one prohibited structural marker.
  The rejected raw response was not persisted, so the exact marker cannot be
  claimed from this run.
- The next engineering need is a redacted supervisor-rejection telemetry path.
  This is not evidence that Doubao improves or worsens fixed-environment
  reasoning quality.

## Conclusions

Role isolation is now explicit: Qwen remains the sole action policy and
query-analysis/executor/final roles remain local Qwen. The live calibration is
**blocked**, not successful. Strict anti-signal-theft enforcement prevented a
possible action/answer leak from reaching Planner-main, but yielded zero valid
trajectories. Thus the all-Qwen baseline remains the only valid outcome
comparison, and no reward/variance conclusion is justified.

No hot fix or second run was made, as required. A future approved task may add
redacted rejection telemetry and refine the generic supervisor schema/prompt,
but must not weaken action-control or evidence-grounding merely to obtain valid
rollouts.

## Verification

- 44 focused tests passed: hierarchical planning, tool priority, Ark provider,
  prior audit fixes, and reward scorer.
- `py_compile`, `bash -n`, JSON consistency, `git diff --check`, and scoped
  secret review were run before commit.
- The runner used `trainer.val_only=true`, `save_freq=0`, and no checkpoint.
  No update occurred.
