# Tool-role priority and fixed-role boundary correction

## Observed facts

This task did not start a model process, rollout, training update, checkpoint,
or external LLM/judge call. The intended one-question MuSiQue smoke was not
run because the clarified prerequisite—a usable all-Qwen local factual
retrieval tool—is not met.

The prior all-Qwen7B MuSiQue probe used only
`ENABLE_TOOLS: ['Base_Generator_Tool']`; its planner therefore could not choose
a specialist. The original trace records all 40 MuSiQue first actions as
`Generalist_Solution_Generator_Tool`.

The currently implemented factual-retrieval candidates are not usable under
the requested all-Qwen/no-external-LLM boundary:

- The actual repository tool is named `Wikipedia_RAG_Search_Tool` (not
  `Wikipedia_Knowledge_Searcher_Tool`). Its `execute()` exits when
  `OPENAI_API_KEY` is absent before calling Wikipedia
  ([tool.py](/root/autodl-tmp/AgentFlow/agentflow/agentflow/tools/wikipedia_search/tool.py:211)).
  It then invokes an LLM relevance selector and creates `Web_Search_Tool`
  ([tool.py](/root/autodl-tmp/AgentFlow/agentflow/agentflow/tools/wikipedia_search/tool.py:224)).
- `Web_RAG_Search_Tool` declares an OpenAI requirement for embeddings and LLM
  generation ([tool.py](/root/autodl-tmp/AgentFlow/agentflow/agentflow/tools/web_search/tool.py:23)).
- No PubMed tool module is present under `agentflow/agentflow/tools/`.
- `Python_Code_Generator_Tool` is a locally routable calculation specialist,
  but cannot supply the missing factual hop for the selected Barcelona
  MuSiQue question. Testing only Generalist versus Python would not verify
  specialist retrieval and would violate the clarified smoke gate.

The offline test `test_current_wikipedia_rag_cannot_run_without_openai_key`
confirms the Wikipedia tool exits before network retrieval with a clean
environment. It makes no external call.

## Code changes

The minimal generic code change is limited to prompt/metadata boundaries and
local Python-tool wiring:

- `planner.py` adds a relevance-based (not fixed-order) specialist priority:
  factual/entity/relation lookup → available Wikipedia/knowledge/web retrieval;
  arithmetic → Python; biomedical lookup → PubMed; Generalist only for
  reasoning/synthesis/fallback when no suitable specialist exists.
- The Generalist metadata now explicitly says it is a fallback and is neither
  a factual retrieval tool nor a calculator.
- Planner-fixed query analysis is limited to decomposing/selecting tools;
  final/direct generation is limited to synthesis from memory. Neither may
  fill missing facts or perform hidden nontrivial calculation.
- The executor prompt now only permits translating the planner-selected action
  into `tool.execute(...)`; it cannot solve, retrieve, calculate, change the
  tool, or substitute Generalist output.
- The verifier prompt now requires judgments solely from recorded memory; it
  cannot invent/search facts or calculate unrecorded results.
- `Python_Coder_Tool` accepts the existing local vLLM `base_url` and
  `max_tokens`, so a future smoke can route it through frozen `qwen-base`
  rather than an external default engine.

No tool sequence, dataset rule, reward/scorer behavior, solver loop, or
training algorithm was changed.

## Tests and static checks

Run in `/root/autodl-tmp/conda/envs/agentflow`:

```text
python -m pytest -q test/test_tool_priority_guidance.py \
  test/test_unified_local_roles.py test/test_vllm_timeout_cleanup.py \
  test/test_reward_scorer.py
24 passed, 34 subtests passed
```

`py_compile` passed for the changed modules and `git diff --check` passed.
The new tests cover generic priority wording, role boundaries, Generalist
fallback metadata, local Python endpoint wiring, and the current Wikipedia
OpenAI-key blocker.

## Hypotheses

1. With a factual retrieval tool that can return locally retrieved evidence,
   the new planner guidance should make a factual relation subgoal prefer that
   tool over Generalist. This remains untested because enabling the present
   Wikipedia/Web path would break the zero-external constraint.
2. The new verifier wording may reduce premature stops on unsupported claims,
   but no behavioral claim is made without a valid retrieval-enabled smoke.

## Conclusions

The one-question n=4 smoke is correctly blocked, not failed: there is no
usable local factual retrieval tool in the currently enabled/implemented tool
set. Running it with Generalist as a stand-in would directly contradict the
role-specific boundary clarification. No GPU/Ray/vLLM process was started;
there are no new trajectory artifacts or cleanup events to report.

## Recommendation

Before retrying the authorized smoke, implement or approve a narrowly scoped
local factual retrieval backend—for example, a Wikipedia raw-fetch/retrieval
tool that does not require OpenAI selection, embeddings, or Web RAG—and test
that it is wired to frozen `qwen-base` only for any LLM-assisted formatting.
Then enable that tool plus Python and rerun the exact existing MuSiQue group-4
question with n=4 once. Do not use this held-out probe row for training.
