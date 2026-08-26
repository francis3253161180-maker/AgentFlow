# Outcome-reward validity audit handoff

## Observed facts

- This audit is offline-only. It read the completed fixed 100-prompt audit results, the four existing training-rollout directories, and scorer telemetry. No rollout, GPU process, scorer call, training update, checkpoint, or reward-rule change was performed.
- Source audit commit: `03b0886`; source manifest: `log/2026-08-26_rollout_difficulty_audit_sample_manifest.json`; source result: `log/2026-08-26_rollout_difficulty_audit_complete_results.json`.
- A fixed `random.Random(20260826)` stratified sample was created without inspecting answer content: 12 all-zero groups (9 NQ + 3 mathhard) and 20 all-one groups (10 NQ + 10 mathhard), for 32 groups / 128 rollout answers. The exact selection is in `log/2026-08-26_outcome_reward_validity_audit_sample_manifest.json`.
- Per-rollout scorer route was recovered by sorting each chunk's JSON timestamp and matching it to the ordered `HYBRID_REWARD_EVENT` telemetry. All 400/400 route event scores matched the stored reward; no new scorer execution was made.
- I directly reviewed every selected question, ground truth, answer, reward, route, and relevant tool/path signature. Manual labels are in `log/2026-08-26_outcome_reward_validity_audit_results.json`; the compact label source is `log/2026-08-26_outcome_reward_validity_audit_manual_labels.json`.
- Of 128 reviewed rollouts, 92 were clear binary judgments and 36 were marked ambiguous/uncertain. The primary confusion matrix excludes the 36 ambiguous rows rather than forcing them into FP/FN.

### Primary manual comparison

| subset | clear rows | ambiguous excluded | TP | TN | FP | FN | agreement | FN rate | FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 92 | 36 | 79 | 4 | 1 | 8 | 90.2% | 9.2% | 20.0% |
| NQ | 44 | 32 | 39 | 4 | 1 | 0 | 97.7% | 0.0% | 20.0% |
| mathhard | 48 | 4 | 40 | 0 | 0 | 8 | 83.3% | 16.7% | — |
| scorer 0/4 groups | 12 | 36 | 0 | 4 | 0 | 8 | 33.3% | 100.0% | 0.0% |
| scorer 4/4 groups | 80 | 0 | 79 | 0 | 1 | 0 | 98.8% | 0.0% | 100.0%* |
| deterministic route | 43 | 3 | 35 | 0 | 0 | 8 | 81.4% | 18.6% | — |
| judge fallback (judge + cache) | 49 | 33 | 44 | 4 | 1 | 0 | 98.0% | 0.0% | 20.0% |

`FP rate` and `FN rate` use the manually judged negative and positive denominators; the 4/4 FP rate is 1/1 because this selected subset has only one clear negative. It is not a stable population estimate.

The sampled route counts were deterministic 46, uncached judge 73, and judge-cache 9. The one clear FP was an uncached judge result; there were no clear deterministic FPs in this sample. The eight clear FNs all used deterministic routing.

## Manual review findings

### All-zero groups

- `mathhard idx=33486`: all four answers say “Yes, a=b” for the divisibility question. These are semantically correct. The production deterministic route returned `safe_math_mismatch`, so this is a direct scorer false-negative pattern caused by an over-conservative math/yes-no equivalence path.
- `mathhard idx=9398`: all four answers give 8 for the minimal Klein-bottle triangulation. The standard mathematical answer is 8; the stored GT is 6. This is a ground-truth error, not evidence that the hybrid scorer itself systematically rejects a correct answer.
- `nq idx=44579`: all four answers identify actors other than the sampled GT Matt Bennett for Ariana Grande’s “One Last Time” video. This is a clear true-negative all-zero group.
- `mathhard idx=50939` has no defensible numeric answer because `T(x,y)` is unspecified while GT is `3`; the derivative-form answers are reasonable conditional answers. It is excluded as ambiguous.
- The remaining sampled NQ zero-groups contain missing time anchors, missing source context, changing government/economic facts, an underspecified “hexagon” question, or classification-dependent mineral counts. They are recorded as ambiguous rather than counted as scorer errors.

Thus, in this sample, one of 12 all-zero groups directly demonstrates scorer FN behavior, one demonstrates a GT defect, one is a clear true all-zero group, and nine cannot be safely classified without resolving prompt/GT ambiguity. The sample does not support explaining the whole 0/4 population as scorer bias.

### All-one groups

- 79/80 clear answers were manually correct.
- One clear FP occurred in `nq idx=31767`, “who wrote the song Baby by Justin Bieber?” Rollout 3 received reward 1 through the uncached judge but added false writer names (including Ludacris, Nasri Atweh, and Adam Messinger) to otherwise correct credits. This is a semantic-judge false positive for a contaminated multi-answer list: mentioning the GT among additional false candidates was accepted.
- The other sampled NQ and mathhard all-one answers were semantically correct, including natural-language answers, exact dates, aliases, math expressions, and structured derivations.

### Group-level outcome and diversity

Of the 32 originally all-equal groups:

- 21 remained clear manual all-1 groups;
- 1 remained clear manual all-0;
- 1 became mixed under manual labels (`Baby` writer credits);
- 9 remained ambiguous because the prompt/GT did not support a reliable binary judgment.

For the sampled 20 scorer-4/4 groups, 19 remained all-equal and one became mixed. For the sampled 12 scorer-0/4 groups, one remained all-0, two became clear all-1 (`Klein bottle`, divisibility), and nine were ambiguous.

Across the 32 groups, mean normalized unique answers per group was 3.16, mean exact duplicate rate 21.1%, and mean unique structural tool/path signatures 1.25. NQ had mean 3.63 unique answers and 9.2% exact duplicate rate; mathhard had mean 2.46 unique answers and 38.5% exact duplicate rate. These are deterministic surface/path measures, not semantic-similarity scores. The lower mathhard answer diversity is consistent with more correlated mathematical outputs, but does not by itself prove semantic collapse.

## Hypotheses

- The deterministic scorer appears more likely to be over-conservative than over-broad in math/boolean cases: the clear deterministic FNs all came from `safe_math_mismatch`, while no clear deterministic FP was found. This is a targeted observation from 43 clear deterministic-route rows, not a population proof.
- The judge fallback appears high precision on this sample but not perfectly conservative: the contaminated “Baby” writer list shows that a judge can accept a correct phrase embedded in an answer containing false additional claims. The existing “ground truth merely mentioned” protection does not fully solve false multi-answer lists.
- A large part of the apparent all-zero behavior is compatible with stale, incomplete, or definition-sensitive ground truths rather than model difficulty alone. The selected NQ zero-groups contain several time-sensitive or underspecified questions, so their binary reward cannot be treated as a clean difficulty label.
- The sampled all-one precision is high enough to argue that 4/4 is often genuine correctness, but the sample is small and was not blind to the scorer class. It does not establish that all 67 original 4/4 groups are easy.
- Tool/path signatures are highly similar within many groups, especially mathhard, and answer duplication is higher there. This supports a generation-correlation hypothesis, but the audit did not use an LLM similarity judge and cannot separate shared prompt difficulty from shared planner/tool behavior.
- No evidence of answer leakage or ground-truth exposure was found in the reviewed final answers/tool paths. This was not a dedicated leakage test, so the hypothesis is not proven either way.

## Conclusions

- The 4/4 count is not primarily explained by a broad scorer FP problem in this sample: 79/80 clear all-one answers were correct, and 19/20 sampled all-one groups stayed all-equal. It is reasonable to treat much of `4/4=67` as genuine easy/correct outcome behavior, while retaining uncertainty about the unsampled population.
- The 0/4 count is not clean evidence of prompt difficulty. The audit found one direct scorer FN group, one likely ground-truth error group, one clear true-negative group, and nine ambiguous groups among the 12 sampled zero groups. Therefore, scorer/GT validity issues materially affect the interpretation of 0/4, even though they do not explain all of it.
- The observed 19% mixed rate remains directionally useful evidence that group reward variance is sparse, but it should not be treated as an exact clean estimate until label ambiguity and scorer edge cases are audited more broadly. The clearest known scorer defect would turn one sampled 0/4 group into all-1, while the judge FP would turn one sampled 4/4 group into mixed; these effects are non-zero but do not overturn the existence of substantial all-equal groups.

## Recommendation

Do not modify the scorer or start training based on this audit. First run an independent blind outcome audit on a new, answer-content-independent sample with reward/category hidden from the reviewer, and stratify by dataset/source and question age/definition sensitivity. Include dedicated checks for:

- deterministic math/yes-no equivalence false negatives;
- judge acceptance of answers containing extra false candidates;
- stale or incomplete ground truths and time-dependent NQ questions;
- semantic rather than exact answer duplication.

Until that blind audit is complete, retain the conservative interpretation: binary GRPO signal is sparse and 19% mixed is plausible but noisy; expanding data or difficulty-aware sampling may be useful, but this result alone does not authorize a new rollout or formal baseline training.

## Reproduction and checks

- Preparation: `scripts/prepare_outcome_reward_validity_audit_20260826.py`
- Summarization: `scripts/summarize_outcome_reward_validity_audit_20260826.py`
- Tests/checks: existing scorer and cleanup tests, JSON validation, `py_compile`, `bash -n`, `git diff --check`, and secret scan.
- Large raw rollout/log directories remain local evidence and are not included in the commit.
