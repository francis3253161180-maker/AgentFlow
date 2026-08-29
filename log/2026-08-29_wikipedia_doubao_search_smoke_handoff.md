# Public Wikipedia tool-priority smoke

Commit under test: `46edf50` plus the scoped uncommitted public-retrieval change documented below.  This was one frozen MuSiQue probe prompt (`2hop__13592_49388`, source index 259), four rollout-only trajectories, seed 20260829.  It is a structural diagnostic, not a training or benchmark result.

## Observed facts

- The prior all-Qwen MuSiQue probe exposed only `Generalist_Solution_Generator_Tool`; its first actions were therefore all Generalist.  This run exposed Generalist fallback, Python calculation, and Wikipedia retrieval together.
- `wikipedia` package requests returned non-JSON in this environment, while the public MediaWiki API returned valid JSON for the identical FC Barcelona query.  The scoped tool change therefore replaces the package/OpenAI/Web-RAG path with raw public MediaWiki search plus page excerpts.  Ranking is the public API result order; no semantic reranking, embeddings, LLM answer generation, OpenAI API, or API key is used.
- Architecture evidence in the rollout log: `planner_main` requests used `qwen-actor` with the synced LoRA; query analysis, executor, verifier, final generation, and LLM-backed local tools used `qwen-base` with no adapter.  Search had `raw-wikipedia`, not an LLM engine.
- Search telemetry across every executed search reports `provider=public_wikipedia`, `search_internal_llm_calls=0`, `openai_calls=0`, and `doubao_calls=0`.  The approved Doubao-in-Search exception was not needed and was not exercised.
- All four first actions selected `Wikipedia_RAG_Search_Tool`, not Generalist.  Each trajectory used three Wikipedia steps; none used two distinct tools.  The complete step-level choices, subgoals, executor commands, evidence URLs/excerpts, verifier decisions, final answers, and rewards are in the [results JSON](2026-08-29_wikipedia_doubao_search_smoke_results.json).

| rollout | tool sequence | verifier stop signals | final answer | reward |
| --- | --- | --- | --- | --- |
| 1 | Wikipedia → Wikipedia → Wikipedia | false, false, false | `38` | 1.0 |
| 2 | Wikipedia → Wikipedia → Wikipedia | false, false, false | qualified natural-language answer mentioning 38 | 0.0 |
| 3 | Wikipedia → Wikipedia → Wikipedia | false, false, false | insufficient-evidence response | 0.0 |
| 4 | Wikipedia → Wikipedia → Wikipedia | false, false, false | insufficient-evidence response | 0.0 |

- Aggregate reward vector is `[1.0, 0.0, 0.0, 0.0]` (mean 0.25); 4/4 rollouts were valid and no retry was recorded.  This reward is the existing deterministic scorer, unchanged.
- The public endpoint began returning HTTP 429 responses under the four concurrent trajectories: one page-fetch step in rollout 2, one search step in rollout 3, and two steps in rollout 4.  These error strings are preserved in the untracked raw rollout files; they were not converted into correct evidence.
- The first launch did not load vLLM or emit a trajectory: the one-row diagnostic data with inherited `train_batch_size=2` caused a dataloader assertion.  The run-only batch and PPO-mini-batch were then set to 1.  A second partial attempt reached retrieval but exceeded the 4096-token context with 3×1200-character excerpts and 1024 completion tokens; it was stopped safely after preserving local logs.  The final valid run uses a generic two-page, 600-character raw-evidence bound and 512 maximum completion tokens.
- Final valid run lifecycle was clean: normal-complete cleanup reported `drained=True`, `outstanding_before=0`, then reset-prefix-cache and sleep.  No CUDA OOM, illegal memory access, deadlock, or prefix-cache error occurred.  Peak sampled GPU use was 20,097 MiB; final `nvidia-smi` showed no compute process and no experiment Ray/vLLM process remained.

## Code changes

- `agentflow/tools/wikipedia_search/tool.py`: implements direct public MediaWiki retrieval, preserves search order, emits explicit zero-call telemetry, accepts unified tool constructor arguments, and marks itself `require_llm_engine=False`.
- `scripts/run_wikipedia_tool_priority_smoke_20260829.sh`: fixed existing group 4 provenance, creates only a one-row local parquet/config, uses Qwen 7B actor/frozen-base aliases, enables only Generalist/Python/Wikipedia, enforces `val_only`, `save_freq=0`, and aborts on lifecycle failures.
- `scripts/aggregate_wikipedia_tool_priority_smoke_20260829.py`: writes the reproducible per-trajectory structural record without using an LLM or judging output semantics.
- `test/test_tool_priority_guidance.py`: now proves raw Wikipedia retrieval succeeds without `OPENAI_API_KEY` or an LLM factory, with mocked public API responses.

## Tests

```text
python -m pytest -q test/test_tool_priority_guidance.py test/test_unified_local_roles.py test/test_vllm_timeout_cleanup.py test/test_reward_scorer.py
24 passed, 1 warning, 34 subtests passed
python -m py_compile ...; bash -n scripts/run_wikipedia_tool_priority_smoke_20260829.sh; git diff --check
passed
```

## Hypotheses

- The role/tool wording is sufficient to correct the specific structural failure—Generalist-first factual delegation—when a usable retrieval tool is present.
- The repeated Wikipedia choice and all verifier `CONTINUE` outcomes suggest that raw top-two excerpts alone are not enough for robust multi-hop evidence management.  HTTP 429 under concurrent requests is a confounder, so this smoke cannot distinguish retrieval ranking limits from public-endpoint rate limiting.

## Conclusions

- Structural success: the bounded smoke meets its stated criterion.  At least one—and in fact all four—Planner trajectories chose specialized factual retrieval first instead of blindly delegating the composite factual question to Generalist.
- Boundary success: all AgentFlow roles remained Qwen-only; no OpenAI, DeepSeek, GPT, or Doubao call occurred.  Generalist was available only as fallback and was not used as a fact source.
- This does **not** establish accuracy or a multi-tool policy: 1/4 exact reward and zero Python/tool diversity are too small and too affected by endpoint 429 errors for that claim.

## Recommendation

Do not expand this smoke automatically.  Before a larger factual multi-hop rollout, obtain approval for a conservative rate-limit/cache policy for public MediaWiki retrieval and separately evaluate whether Planner can select Python only when a retrieved subgoal supplies a calculation.  Neither change should allow Generalist to substitute for missing factual evidence.
