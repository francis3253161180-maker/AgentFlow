# Offline MuSiQue single-policy boundary

Runtime semantics come only from `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` with the existing rank-8 LoRA actor. The same weights run both fixed textual modes: `DECISION` emits one search/answer action, and `EVIDENCE_UPDATE` emits extractive selections from only the immediately preceding local observation. There are no special mode tokens.

The environment is mechanical: a question-local candidate corpus, deterministic BM25 plus cached normalized BGE CLS embeddings plus RRF, exact normalized-substring provenance, memory/budget enforcement, official-style answer normalization, scorer-only support matching, and binary terminal reward. The runtime module imports no legacy solver, planner, verifier, generator, web tool, external client, or semantic judge. Runtime explicitly removes common external API credential variables and forces Hugging Face/Transformers offline mode.

Compact policy memory stores at most six `[pid] exact quote` entries (each at most 300 characters) and six query/outcome history rows. Raw paragraphs, scores, hashes, and scorer annotations stay in local audit artifacts. The maximum measured compact memory in the passing smoke was 284 Qwen tokens.

One rollout is `DECISION -> local search -> EVIDENCE_UPDATE -> ... -> DECISION(answer)`. Every actor generation is persisted as its own transition with prompt hash, response, token/cumulative-logprob metadata, memory before/after, action or selection, observation references, and validation outcome. Existing `agentflow.verl.advantage.compute_rollout_group_advantage` computes group statistics once per unique rollout and broadcasts the trajectory advantage across each transition response mask; focused existing tests verify unequal transition multiplicity and fail-closed incomplete groups.

Terminal reward remains `1` only when normalized answer EM/alias match and validated selected evidence covers every gold support pid. The only positive Phase-C rollout met both conditions. Retrieved-but-not-selected support never counts.

Large scorer corpus, embedding arrays, raw prompts, paragraphs, trajectories, GPU samples, and runtime logs remain local under `/root/autodl-tmp/offline_musique_artifacts_20260830`.
