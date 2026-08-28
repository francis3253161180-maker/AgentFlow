# Follow-up GitHub audit fixes handoff

Base revision: `519904e`
Branch: `experiment/flow-grpo-3b-lora`

## Scope and safety boundary

This change implements only the four scoped code/test fixes from the follow-up audit. No model inference, rollout, GPU/Ray/vLLM process, optimizer step, training run, checkpoint write, or external API call was performed.

## Observed audit findings

- Agent execution exceptions in `train/rollout.py::_solve_and_evaluate()` were converted to the string answer `None`, allowing an infrastructure failure to reach reward evaluation as an ordinary zero-reward answer instead of AgentRunner's invalid-rollout/retry/drop path.
- `ChatVLLM` accepted role sampling values through construction, but text-generation defaults were hard-coded in the request path. This could erase fixed-role `temperature=0` or planner overrides. Cache keys also did not distinguish sampling settings.
- Planner, verifier, and executor prompts requested legacy section formats while also passing structured response schemas.
- The non-unified triplet filter returned un-tokenized spans, while unified filtering needed to keep only explicitly attributed trainable `planner_main` actor transitions.

## Code changes

- `train/rollout.py`: propagate exceptions from `rollout.solve()`; retain conservative `None` only for missing/invalid model output, so AgentRunner can post an empty invalid rollout on infrastructure failure.
- `agentflow/agentflow/engine/vllm.py`: store constructor defaults for temperature, top-p, frequency penalty, and max tokens; resolve explicit call overrides with `None`-aware logic; include resolved sampling settings/content/schema in cache keys; preserve vLLM `guided_json` forwarding.
- `agentflow/agentflow/models/planner.py`: make `NextStep` prompts explicitly JSON-only with all four schema fields.
- `agentflow/agentflow/models/verifier.py`: make `MemoryVerification` prompts explicitly JSON-only with `analysis` and `stop_signal`; retain legacy extraction only as compatibility fallback.
- `agentflow/agentflow/models/executor.py`: make `ToolCommand` prompts explicitly JSON-only with `analysis`, `explanation`, and `command`; retain legacy extraction only as compatibility fallback.
- `agentflow/runner.py`: discard spans without both prompt and response token IDs in all modes; in unified-local mode additionally require explicit `planner_main`, trainable, `qwen-actor` attribution. Fixed-role empty spans no longer invalidate a rollout that has a valid actor path.
- `test/test_followup_github_audit_fixes.py`: focused fake-client/CPU tests for exception propagation, sampling defaults and overrides, guided JSON, all three structured prompts, and non-unified filtering.

## Verification

The focused and existing related suites passed:

```text
56 passed, 1 warning, 39 subtests passed in 7.54s
```

The warning is the existing Ray state API deprecation warning. The new tests use a mocked OpenAI-compatible client and do not make a network request.

Additional checks performed for this handoff: Python compilation, shell syntax checks for touched smoke entrypoints, JSON parsing of tracked JSON files, `git diff --check`, and a scoped secret scan. No credentials are present in the change.

## Remaining risks and next step

There is no live vLLM runtime evidence from this code-only task. The request-path behavior is covered with a fake client, including explicit zero-valued parameters and guided JSON. The compatibility parsers remain available for non-structured/legacy responses, so a future minimal runtime smoke should verify production server behavior before any formal experiment. No formal training is authorized by this handoff.
