# Tool boundaries and search-escalation validation

Commit under test: `a22d286` plus the scoped changes in this handoff.  This was
one previously frozen MuSiQue prompt (`id=4`, source index `259`, benchmark id
`2hop__13592_49388`), four rollout-only trajectories, seed `20260829`.  It is
a structural diagnostic, not training or a benchmark result.  Raw rollout data
and raw logs remain local and untracked:

`rollout_data/46.38.243.197/tool-boundaries-search-escalation-musique-group4-20260829_20260829-192827/Qwen2.5-7B-Instruct_20260829-192827/train`

## Observed facts

- The enabled tools were Generalist, Python, public Wikipedia, and Web_RAG.
  `planner_main` used Qwen2.5-7B-Instruct with the actor LoRA; Planner-fixed,
  Executor, Verifier, Final, and any LLM-backed local tool used Qwen-base with
  the adapter disabled.  `trainer.val_only=true`, `save_freq=0`, and no
  optimizer/backward/checkpoint marker was observed.
- `GOOGLE_API_KEY` was **missing**.  Ground_Google was therefore not enabled or
  invoked.  No OpenAI, DeepSeek, GPT, or Doubao request was made.  The
  OpenAI-compatible trace label in the local logs refers to loopback vLLM calls
  to `127.0.0.1`, not an OpenAI API call.
- The four valid rollout rewards were `[1.0, 1.0, 0.0, 0.0]` (mean `0.50`),
  versus a22d286's `[1.0, 0.0, 0.0, 0.0]` (mean `0.25`).  This one-group
  difference is not evidence of an accuracy improvement.

| rollout | planner tool sequence | final answer | reward |
| --- | --- | --- | ---: |
| 1 | Wikipedia → Wikipedia | `38` | 1.0 |
| 2 | Wikipedia | `38` | 1.0 |
| 3 | Wikipedia | insufficient-evidence response | 0.0 |
| 4 | Wikipedia | insufficient-evidence response | 0.0 |

- All first actions selected Wikipedia and no trajectory invoked Generalist,
  Python, Web_RAG, Google, or a search-internal LLM.  Thus there was no
  Generalist factual shortcut and Python was appropriately unused: no grounded
  arithmetic operands were available.
- Rollout 1 is direct counter-evidence to the intended escalation behaviour.
  Step 1 returned `https://en.wikipedia.org/wiki/FC_Barcelona`; the verifier
  said the excerpt lacked the league/game-count evidence.  Step 2 nevertheless
  made a near-duplicate Wikipedia query and returned the same two pages, rather
  than deep-reading the known URL with Web_RAG.  The query change removed only
  `in`; it did not materially narrow the subgoal or evidence state.
- Wikipedia's per-call telemetry total was `cache_hits=6`, `retries=0`,
  `http_429=0`, and all internal/external LLM counters were zero.  In a22d286
  the otherwise comparable run exposed four public-endpoint 429 failures.  The
  new bounded cache/backoff path is exercised by the unit test; this smoke
  exercised cache hits but did not need a live retry.
- The run completed 4/4 valid trajectories with zero retries.  Peak sampled GPU
  use was `20,099 MiB`.  Normal-complete cleanup recorded `drained=True`,
  `outstanding_before=0`, reset-prefix-cache only after drain, and sleep
  completed.  No CUDA OOM/illegal-memory-access, prefix-cache failure,
  deadlock, Ray worker death, or retained experiment process remained; final
  GPU use was 0 MiB.

## Code changes

- `planner.py`: replaces name-only preference wording with explicit generic
  capability boundaries and an observation-based stagnation guard.  The guard
  prohibits same/near-identical retrieval with unchanged evidence and permits
  URL deep-read or source switch only where justified; it does not require
  multi-tool diversity.
- `wikipedia_search/tool.py`: makes public MediaWiki responses process-local
  cached/deduplicated, adds bounded Retry-After-aware 429 backoff, and emits
  cache/retry/429/provider/LLM telemetry.  The cache stores only successful
  HTTP payloads, never planner/verifier state, final answers, or rewards.
- `web_search/tool.py`: removes OpenAI embedding and answer-generation code.
  It now deep-reads one already-known `http(s)` URL and returns bounded raw
  chunks ranked deterministically by lexical overlap, with zero LLM/OpenAI/
  Doubao telemetry.  It is not an open-web discovery tool.
- `python_coder/tool.py`: documents the existing intended boundary that
  operands must be present in memory/evidence before calculation.
- The runner accepts a scoped run tag/tool list and stops its completed local
  rollout daemon promptly; aggregation records Web and Wikipedia telemetry.

## Hypotheses

- The unchanged Qwen actor can read the new capability text but did not comply
  with the soft stagnation instruction in the only clearly eligible case.  The
  problem is likely policy-following/decision quality, not missing Web_RAG
  construction: Web_RAG was loaded as `raw-web-lexical` with the known-URL
  metadata and no constructor error.
- The reduced live 429 count is consistent with cache/dedup reducing repeated
  MediaWiki requests.  It is not a reliability estimate because this smoke is
  only four trajectories and no 429 occurred.

## Conclusions

- The minimal generic boundaries are correctly exposed and safe: the original
  Generalist-as-factual-retrieval failure did not recur, Web_RAG no longer has
  an OpenAI dependency, and public retrieval is more robust and observable.
- The validation does **not** demonstrate successful observation-driven
  Wikipedia → Web_RAG escalation.  The only duplicate-retrieval trajectory
  ignored the prompt-level guard; the other three stopped after one Wikipedia
  action despite verifier `CONTINUE`.
- No further smoke was run.  The task permits only a rerun for a trivial
  metadata/constructor bug, and the observed issue is not that category.

## Recommendation

Do not claim multi-tool planning is solved and do not use this result to tune
reward or start training.  If approved, inspect a minimal generic enforcement
mechanism at action selection (for example, represent returned URLs and a
stable evidence signature explicitly) before another bounded validation.  Such
work should preserve the rule that repeated same-tool use is allowed when a
new, justified subgoal/evidence state exists; it must not force tool diversity.

## Verification

- Focused tool-boundary tests: `6` passed.
- `py_compile`, `bash -n`, JSON parsing, and `git diff --check` passed before
  the smoke; final checks are recorded with the commit.
