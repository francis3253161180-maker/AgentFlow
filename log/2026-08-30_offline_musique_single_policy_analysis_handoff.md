# Offline MuSiQue single-policy baseline — analysis handoff

Implementation/evidence commit: `c75bddc9bdeda6fb6235ce9147ddd281a27ab45d` (short `c75bddc`; branch base `02fc10b06d29d70f740819d77409d5ef411fe2da`). Branch: `experiment/offline-musique-single-policy`. Worktree: `/root/autodl-tmp/AgentFlow-offline-musique`. Seed: `20260830`.

## Preservation and boundary

The original `/root/autodl-tmp/AgentFlow` remained on `experiment/flow-grpo-3b-lora` at `02fc10b06d29d70f740819d77409d5ef411fe2da`; it had 15 tracked modifications and 285 untracked status entries. Nothing there was stashed, reset, deleted, staged, or committed. Initial GPU state was RTX 5090 UUID `GPU-f3dcba7f-d559-96e7-b0a6-8879840f9d5a`, 32,607 MiB total, 2 MiB used, 0% utilization; no Ray/vLLM/torchrun/deepspeed workload was running.

Runtime semantics came only from local `Qwen2.5-7B-Instruct` plus the existing rank-8 LoRA actor, shared by `DECISION` and `EVIDENCE_UPDATE`. The environment performed only deterministic local search, parsing, provenance, memory/budget enforcement, normalization, scorer-only matching, and reward. External semantic/API calls and fixed semantic roles were both zero. HOB, reranking, reward shaping, transition rewards, variance sampling, special mode tokens, open-web retrieval, and training on dev were not used.

## Observed facts

### Artifacts and Phase A

Official MuSiQue repository provenance is `https://github.com/StonyBrookNLP/musique.git` at `922ac98f19a201998dbdae6d7f2887a5258dbdeb`. Dev source SHA256 is `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`; it produced 2,417 answerable questions and 21,100 unique paragraph hashes. The local 384-d float32 normalized-CLS BGE cache SHA256 is `c6e77cadba495117863ca01cf453c446dbc25f09aabb898cd92b3713ae575ca4`. Full actor/BGE/LoRA hashes are in `log/2026-08-30_offline_musique_artifact_manifest.json`.

Phase A passed: 23 focused plus existing grouping tests passed; py_compile, direct entry-point resolution, diff-check, JSON/schema checks, and secret scan passed. Structural audits checked all 2,417 actor payloads/prompts with zero scorer-key leakage. Cached BGE ranking equaled fresh ranking, and repeated BM25/dense/RRF calls matched exactly. GPU returned to 2 MiB after the one-time GPU embedding precompute.

### Phase B retrieval gate

The fixed 64-question subset contained 22 two-hop, 21 three-hop, and 21 four-hop questions. RRF was exactly repeatable and unsaturated.

| Query | RRF Recall@1 | Recall@2 | Recall@5 |
|---|---:|---:|---:|
| full question | 0.3047 | 0.4284 | 0.5781 |
| compact question-only | 0.2734 | 0.3685 | 0.4935 |
| weak truncated | 0.1198 | 0.1797 | 0.3190 |
| unrelated | 0.0508 | 0.1055 | 0.2865 |

Full-minus-unrelated Recall@2 was +0.3229; compact-minus-unrelated was +0.2630. RRF was not materially worse than both components. Phase B passed without a retrieval iteration.

### Phase C iterations

- `phase_c_v1`: failed before the first generation because vLLM 0.9.2 V1 xgrammar rejected its cached slow-tokenizer wrapper. Completed/valid rollouts were 0/0. Cleanup returned GPU memory to 2 MiB.
- `phase_c_v2`: removing only the incompatible generation constraint allowed 128 rollouts, while strict no-repair validation remained. DECISION validity was 99.37%; EVIDENCE_UPDATE was 70.90%. Rewards were all zero. Evidence failures were primarily valid-looking JSON followed by generated suffixes.
- `phase_c_v3`: stopping raw evidence generation with the schema-closing `]}` included improved EVIDENCE_UPDATE validity to 91.27% and support-selection recall from 0.0697 to 0.1784, but rewards remained all zero. The gate was corrected to evaluate each mode separately.
- `phase_c_v4`: one generic evidence-protocol clarification (exact outer suffix, at most two selections, shortest exact clause, and intermediate-fact usefulness) passed the gate. DECISION validity was 98.74%; EVIDENCE_UPDATE validity was 97.21%; invalid pid/quote rate was 2.60%; maximum compact memory was 284 Qwen tokens. There were 104 distinct query-sequence signatures.

The persisted v4 result's historical `config_hash` did not include prompt text. The actual passing DECISION and EVIDENCE_UPDATE system-prompt SHA256 values are `197e2e115395b78c0e84097574b08a56815f04813f665833cabe8f0e06529ed0` and `6f8708037ec10948a216bd638f3d4604bfeec9ad5bb2cc049ecb71023f8d4920`; every generated transition also persists its exact rendered prompt hash. The runner now includes protocol and schema hashes in future config hashes. No experimental result was rewritten.

Passing-smoke completed rollouts: 128. Valid answer-terminated rollouts: 112. Dropped/fail-closed trajectories: 16 (6 DECISION format failures, 10 EVIDENCE_UPDATE format failures). Reward vector in persisted trajectory order is exactly `109 zeros, 1, 18 zeros` (one at zero-based index 109); summary `{0: 127, 1: 1}`, mean `0.0078125`. One of 32 n=4 groups was mixed with rewards `[0, 1, 0, 0]`; 31 groups were `[0, 0, 0, 0]`. The positive had answer EM and full validated selected-support coverage; positive grounding violations were zero.

Other passing-smoke metrics: answer EM 0.046875, selected-support recall 0.18685, full selected-support coverage 0.0078125, retrieval recall 0.43490, distractor-selection rate 0.30435, repeated-query rate 0.28212, premature-answer rate 0.99107, 49.96 rollouts/min, 153.71 s end-to-end wall time, and 22,748 MiB peak GPU memory.

### Phase D systems gate and stop

The same eight prompts were compared end to end, including engine startup:

| Setting | Rollouts | Wall s | Rollouts/min | Mean/p95 trajectory s | Peak MiB | Final MiB |
|---|---:|---:|---:|---:|---:|---:|
| n=4, max_num_seqs=4 | 32 | 93.21 | 20.60 | 84.07 / 93.21 | 22,568 | 2 |
| n=8, max_num_seqs=8 | 64 | 157.44 | 24.39 | 147.35 / 157.44 | 21,432 | 2 |

n=8 had no OOM/CUDA/Ray failure and reset prefix cache before clean process exit. vLLM emitted a process-group-not-destroyed warning after both otherwise drained exits. Configured `max_num_seqs` is recorded; the synchronous local API did not expose live in-flight counts. Throughput gain was 1.1841x, below the 1.2x adoption target. This is the explicit poor-scaling stop condition; n=4 remains the only candidate and Phase F was not run.

### Phase E actor-causality audit

The causality gate passed. Identical `(qid, normalized query)` calls had zero observation nondeterminism. Twenty-eight of 32 questions varied first queries; 20 changed first observations. Of 133 distinct first-query pairs, 87 (65.4%) changed retrieval. Seventeen questions varied support-retrieval counts across rollouts. There were 81 trajectories where retrieved support was missing from selected evidence, including 10 where all support was retrieved but not fully selected. Fifteen no-useful results were followed by a distinct useful reformulation; 84 were followed by a repeated query. One answer followed full selected support, while 111 answers were premature.

The mixed group provides the direct causal example: the reward-1 rollout used four actor queries, retrieved and selected 2/2 supports, and passed answer EM. Its three sibling rollouts retrieved only 1/2 support or selected 0/2 or 1/2 and/or failed answer EM. Retriever randomness, fixed roles, and external semantic calls were all zero.

### Training fields

Training occurred: **no**. Official train data was not obtained because the Phase-D stop happened before Phase F. No optimizer step or actor update ran. Consequently actual GRPO advantages, `pg_loss`, `grad_norm`, entropy, `old_log_prob`, optimizer/update status, global step, GPU-hours for training, and pre/post train evaluation are **not recorded**, not numerical zeros. The existing rollout-level advantage implementation and its focused tests passed, but no live training batch was materialized. For the sole mixed group, an advantage vector could be computed only in a future authorized Phase F; it is not reported here as an observed training metric.

## Hypotheses tested

- Meaningful question-derived queries would beat unrelated queries: supported by +0.323/+0.263 RRF Recall@2 margins.
- Native unconstrained generations would meet strict schema validity: unsupported for EVIDENCE_UPDATE in v2.
- Trailing evidence generation was a major format cause: supported directionally in v3 (70.90% to 91.27%).
- A minimal generic evidence-protocol clarification would pass per-mode format and improve selection: supported in v4 (97.21% evidence validity; support-selection recall 0.18685; first grounded positive).
- n=8 would provide at least 1.2x throughput with safe headroom: unsupported (stable 1.1841x).
- Reward variance was attributable to actor actions: supported by deterministic retrieval, query-to-observation differences, support selection differences, and the grounded mixed group.

## Conclusions

The offline two-mode boundary, corpus isolation, deterministic retrieval, strict provenance, transition persistence, and grounded reward are functioning. The actor has a genuine but extremely sparse causal learning signal. The dominant remaining behavioral failures are evidence selection and stopping: 81 returned-support selection misses and 111 premature answers. These are observed policy weaknesses; they do not justify weakening grounding or adding shaping.

The approved stopping point is Phase E. No GRPO or HOB ran. The next decision belongs to the reviewer: retain n=4 and explicitly authorize a later Phase-F pilot despite the n=8 scaling miss, or require a separate systems investigation first. If Phase F is later authorized, first obtain and verify the official MuSiQue-Ans train split, preserve the fixed dev set, and confirm the existing rollout-group advantage bridge with the new transition pack. Do not change reward, provenance, corpus isolation, or actor-only semantics.

## Key local evidence

- Corpus/cache/raw trajectories/GPU/runtime logs: `/root/autodl-tmp/offline_musique_artifacts_20260830`
- Passing Phase-C pack: `/root/autodl-tmp/offline_musique_artifacts_20260830/phase_c_v4_trajectories.json`
- Phase-D raw systems results: `/root/autodl-tmp/offline_musique_artifacts_20260830/phase_d_n4_results.json`, `/root/autodl-tmp/offline_musique_artifacts_20260830/phase_d_n8_results.json`
- Tracked gate results: `log/2026-08-30_offline_musique_phase_{a,b_v1,c_v1_failure,c_v2,c_v3,c_v4,d,e_v1}_results.json` (individual filenames in Git)
