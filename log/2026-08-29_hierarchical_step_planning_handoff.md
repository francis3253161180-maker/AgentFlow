# Hierarchical step-planning bounded smoke handoff — 2026-08-29

## Observed facts

- This task was rollout-only: one frozen MuSiQue Barcelona question (`source_idx=259`, `benchmark_id=2hop__13592_49388`), Qwen2.5-7B actor+LoRA, Qwen-base adapter-off fixed roles, `n=4`, temperature `0.7`, three tool steps, 600 s agent limit, no optimizer/backward/checkpoint, and no external LLM or reward-judge calls.
- The first implementation attempt exposed a production adapter mismatch: the OpenAI-compatible vLLM engine returns guided JSON as text, while the new high-level-plan path passed that text directly to the state machine. The resulting type mismatch was safely rejected as an empty plan before any tool action. The fix uses the existing strict JSON/Pydantic adapter (`parse_strict_json`) and records both raw and validated plan telemetry. This was a constructor/serialization blocker, not a policy result.
- The corrected bounded validation completed `4/4` valid rollouts with `0` retry. Safe lifecycle markers recorded `drained=1` before both sleep/reset operations; no CUDA, prefix-cache, Ray, or vLLM fatal marker appeared. GPU peak was `20,097 MiB`.
- All four rollouts produced a staged dependency-aware plan and activated `step_1`; no trajectory advanced solely because a tool returned. Each step verifier kept `step_1` in progress with an explicit missing fact. Every exact termination reason was `max_steps_with_unresolved_plan`, not timeout or verifier STOP. Execution times were 32.32–37.74 s, far below 600 s.
- Every trajectory used Wikipedia three times, with reward vector `[0, 0, 0, 0]`. Search telemetry reports 0 Doubao/OpenAI/internal-LLM calls, 0 HTTP 429, and 20 local cache hits.
- The final path correctly withheld an answer when plan evidence was incomplete: all four final answers are the local `Insufficient verified evidence; unresolved plan steps: ...` result. `unsupported_final_claim_rollout_count=0`.
- The direct raw-trajectory addendum is incorporated: the earlier `2473baa` one-step runs had 194–230 s execution with a 180 s harness limit, so their one-step length was timeout-confounded; reward is not taken as routing success; unsupported final claims are audited separately; repeated Wikipedia is only legitimate for a materially new unresolved evidence target.
- In this smoke the step state exposed the missing fact, but the trainable planner still repeated materially equivalent Wikipedia objectives. Two trajectories briefly made a composite subgoal by combining league identification and game-count lookup. Therefore the smoke demonstrates state-machine containment, not successful observation-driven re-planning or completed-step advancement.

## Hypotheses

- The Wikipedia result supplied generic FC Barcelona/La Liga context but not the year-specific title evidence required by `step_1`; this explains the verifier remaining false rather than advancing to the dependent game-count step.
- The planner has not yet made enough use of the current step's `missing_evidence` and known-URL state to reformulate a retrieval query or choose a warranted deep-read. This is a routing/prompt-quality limitation, not evidence that multiple tools should be forced.

## Conclusions

- The optional hierarchical control path is operational and preserves the original legacy path when disabled. It provides a fixed-role high-level plan, one active dependency-satisfied step, evidence-only step verification, explicit remain/activate/reopen transitions, and evidence-gated final behavior.
- It prevented the addendum's unsupported-final-claim failure mode in this bounded run. It did not establish the stronger structural success criterion of a completed-step transition, because the requisite first fact was never verified.
- This smoke must not be interpreted as a reward or accuracy improvement. It is a negative routing result with clean infrastructure evidence.

## Recommendation

- Do not train or expand samples from this result.
- The next approved diagnostic, if any, should test the generic state prompt change now present in code: it explicitly names `missing_evidence`, verified evidence, and known URLs and requires one atomic current-step subgoal. It should audit whether repeated retrieval has a materially changed objective/evidence state; it must not force tool diversity.
- Keep final-generation changes separate. The current task only flags unsupported claims and confirms withholding behavior; it does not alter final synthesis.

## Files and checks

- Results: `log/2026-08-29_hierarchical_step_planning_results.json`
- Raw local evidence (untracked): `log/20260829_hierarchical_step_planning_finalparse_20260829_{train,rollout}.log` and `rollout_data/46.38.243.197/hierarchical-step-planning-finalparse-musique-group4-20260829_20260829-203749/`.
- Focused static/unit checks: `py_compile`, `bash -n`, and 32 focused unit tests (hierarchical state, tool guidance, GitHub audit fixes, reward scorer).
