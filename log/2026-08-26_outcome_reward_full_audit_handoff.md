# Outcome-reward validity audit: full 100 groups / 400 rollouts

## Observed facts

- Audit base commit: `4cc7090` on `experiment/flow-grpo-3b-lora`.
- This was an offline audit of the already completed 100-prompt / 400-rollout evidence. No rollout, training, optimizer step, GPU/Ray/vLLM process, checkpoint write, DeepSeek call, or scorer change was performed.
- The original scorer distribution was 0/4=14, 1/4=4, 2/4=6, 3/4=9, 4/4=67; original mixed=19/100.
- Exposure provenance was reconstructed by exact `(source, idx)` keys: 32 groups from the prior 4cc7090 audit, 24 groups from the same-session metadata audit, and 44 remaining groups. There were no duplicates between the two prior manifests.
- The final unified table contains exactly 100 groups and 400 rows. The 44-group blind phase contains 176 rows and no reward/class/source/id/idx/route/cache/path/timestamp fields.
- The first full-blind preparation used a set before seeded shuffling. A second invocation therefore changed the opaque mapping despite the same nominal seed. That mismatched intermediate result was discarded. The mapping was recovered from the recorded manual-review question order and the RNG stream, and the preparation script was fixed to sort before shuffling. Final unblind checks are 100 groups x 4 rows and unique group/candidate keys.
- Manual labels across all rows are: 320 correct, 35 incorrect, and 45 ambiguous. Ambiguous means the question/GT could not support a defensible binary semantic label; it is not counted as a scorer FP/FN.

## Exposure and blind protocol

The 44 remaining groups were randomized with seed `20260826`; only their question, ground truth, candidate answers, and opaque/candidate positions were shown during the new review. Candidate order was independently randomized. Manual labels were saved and validated before the sealed mapping was opened.

The 56 prior groups are explicitly marked exposed/non-blind in the final table. The 44-group phase is metadata-blinded and non-overlapping, but it was performed in the same Codex session that had seen the earlier audits. It is therefore not a strictly independent blind human audit.

## Manual label and scorer comparison

The all-row confusion matrix, excluding ambiguous rows from the rates, is:

| partition | rows | clear | ambiguous | TP | TN | FP | FN | agreement on clear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all 100 groups | 400 | 355 | 45 | 304 | 33 | 2 | 16 | 94.93% |
| legacy exposed 32 | 128 | 92 | 36 | 79 | 4 | 1 | 8 | 90.22% |
| metadata exposed 24 | 96 | 88 | 8 | 80 | 6 | 0 | 2 | 97.73% |
| new non-overlap 44 | 176 | 175 | 1 | 145 | 23 | 1 | 6 | 96.00% |

Overall clear-row FP rate is 2/355=0.56%; FN rate is 16/355=4.51%. By source, mathhard is 166/17/0/9 (TP/TN/FP/FN, 192 clear) and NQ is 138/16/2/7 (163 clear). By route:

| route | rows | clear | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 192 | 187 | 168 | 10 | 0 | 9 |
| DeepSeek judge | 185 | 149 | 119 | 21 | 2 | 7 |
| judge cache | 23 | 19 | 17 | 2 | 0 | 0 |

The two clear FPs are both judge-route NQ cases: the previously observed contaminated songwriter list, and a new alcohol-age answer that incorrectly says 16/17-year-olds may purchase alcohol. No deterministic FP was observed. The 16 clear FNs include five deterministic `safe_math_mismatch` rows (four semantically equivalent `a = b` answers and one valid quaternion-group answer), four rows where the GT itself appears wrong (`GT=6` versus the manually judged mathematical answer 8), and seven conservative/open-answer judge FNs involving broad date/role/entity explanations.

## Corrected group outcomes

The original group bins are 0/4=14, 1/4=4, 2/4=6, 3/4=9, 4/4=67. After manual review, groups with any ambiguous row are unresolved. Among 85 resolvable groups:

| corrected bin | groups | proportion of resolvable groups |
|---|---:|---:|
| 0/4 | 3 | 3.53% |
| 1/4 | 3 | 3.53% |
| 2/4 | 3 | 3.53% |
| 3/4 | 5 | 5.88% |
| 4/4 | 71 | 83.53% |

There are 15 unresolved groups. Corrected mixed groups are 11/85=12.94% among resolvable groups. Treating unresolved groups as all non-mixed gives a lower bound of 11%; treating all unresolved groups as mixed gives an upper bound of 26%.

By source, mathhard has 45 resolvable groups with corrected mixed=7 (15.56%, bounds 14%–24%); NQ has 40 resolvable groups with corrected mixed=4 (10.00%, bounds 8%–28%).

Original all-equal groups total 81. Two clear groups changed from all-equal to manual mixed, both original 4/4 judge FPs. Ten all-equal groups are unresolved because of GT/question defects; no clear original 0/4 group became manual mixed. Original mixed groups total 19: five became manual all-equal, five are unresolved, and nine remain manual mixed. The corrected mixed total is 11 because two originally all-equal groups also became manual mixed.

For the original 4/4=67 specifically: 64 remain clear manual 4/4, two become manual 3/4 because of the two judge FPs, and one is unresolved. For original 0/4=14: three remain clear manual 0/4, two become manual 4/4 because the supplied GT is wrong, and nine are unresolved.

## GT/question quality and recurrence checks

- There are 15 unresolved groups / 45 rows. Recurring defects include missing time anchors for changing facts, missing definitions or classification conventions, contradictory premises, a character-vs-actor GT mismatch, an underspecified mathematical parameter, and ambiguous negative-base real limits. These are recorded row-by-row in the unified manual-label file.
- `safe_math_mismatch` recurs in five rows. This is a genuine general scorer matching defect for the four `a = b` natural-language equivalents and one equivalent quaternion answer; it is not evidence that those groups are difficult.
- No new yes/no-specific FP was identified in the full manual labels. Twenty rows used a scorer yes/no-related route/reason in the telemetry; that count is not a semantic error count.
- Surface diversity is not outcome independence: mean unique normalized answers is 3.04/4, 46/100 groups contain an exact normalized duplicate, and mean unique tool/path signatures is 1.32/4. Path signatures were available for all 100 groups after incorporating the existing 24-group diversity telemetry. These are deterministic surface/structural measures only; no LLM similarity was used.
- Shared tool/path signatures and repeated answer templates establish generation correlation, but there is no direct evidence here of data leakage. Leakage is therefore not concluded.

## Hypotheses

- The high 4/4 mass is primarily real answer correctness plus easy/low-ambiguity prompts: 64/67 original 4/4 groups are manually confirmed all-correct, and the only two clear deviations are judge FPs. The 0/4 mass is mixed: three groups remain all-wrong, two are explained by defective GTs, and nine remain unresolved.
- Binary reward group collapse is therefore not mainly caused by a scorer FP flood. The larger validity risk is conservative FN behavior (especially deterministic safe-math matching) plus GT/question defects.
- Similar tool paths and surface-diverse but semantically repetitive answers plausibly increase within-group correlation. This is evidence for correlated generation, not proof of data leakage.

## Conclusions: requested answers

**A. How much all-equal behavior can scorer bias explain?**

At group level, 2 of 81 original all-equal groups (2.47%) are clear scorer-induced all-equal changes, both 4/4→3/4 judge FPs. Ten more all-equal groups are unresolved because of GT/question quality; they cannot be assigned to scorer bias. No clear 0/4→mixed scorer error was found.

**B. Is 4/4=67 mainly real correctness/easy prompts?**

Yes, with the stated manual-review uncertainty: 64/67 are clear manual 4/4, two have one incorrect answer accepted by the judge, and one is unresolved. This supports “mostly real correctness / relatively easy or correlated prompts,” not “mostly scorer artifact.”

**C. What is the corrected mixed value and is 19% directionally credible?**

The corrected point estimate is 11/85=12.94% on resolvable groups, with a defensible all-group interval of 11%–26% due to 15 unresolved groups. The new 44-group non-overlap metadata-blinded partition independently in this same session has 8/43=18.60% resolvable mixed, with bounds 18.18%–20.45%. Thus the original 19% is directionally credible as a group-variance signal, but its exact value is not established and the 12.94% point estimate shows that GT ambiguity and manual resolution materially affect it.

**D. Is it enough to enter variance-aware/difficulty-aware sampling?**

It is enough to design a controlled variance-aware sampling ablation, but not enough to treat the current reward as fully validated for a formal training claim. Before using it for the next training run, apply and offline-test the general safe-math scorer fix and establish a GT policy for the unresolved time-sensitive/underspecified questions. Do not add sample-specific rules and do not infer leakage from path correlation.

## Recommendation and reproducibility

1. Keep the current scorer/model/algorithm unchanged for this audit record; do not retrain from these labels.
2. Fix and regression-test the general `safe_math_mismatch` path, and separately curate or exclude the 15 unresolved GT/question groups under an explicit policy.
3. Then run the already-designed variance-aware/difficulty-aware sampling comparison using the corrected deterministic reward policy. The present audit supports that direction, but the next training decision should use the post-fix offline audit, not the raw 19% alone.

Primary artifacts:

- `log/2026-08-26_outcome_reward_full_audit_results.json`
- `log/2026-08-26_outcome_reward_full_audit_manual_labels.json`
- `log/2026-08-26_outcome_reward_full_audit_exposure_manifest.json`
- `log/2026-08-26_outcome_reward_full_audit_blinded_review.json`
- `log/2026-08-26_outcome_reward_full_audit_unexposed_manual_labels.json`

Preparation/aggregation scripts:

- `scripts/prepare_outcome_reward_full_audit_20260826.py`
- `scripts/unblind_outcome_reward_full_audit_20260826.py`
- `scripts/recover_full_audit_blind_mapping_20260826.py` (artifact-recovery utility for the discarded set-order run)
- `scripts/rebuild_full_audit_sealed_mapping_20260826.py` (raw-evidence rebuild with candidate-order validation)

The sealed mapping containing raw path/tool metadata remains outside the repository. The final result and label files contain no API key or DeepSeek response. Reproduction is offline: the committed blind file plus `rebuild_full_audit_sealed_mapping_20260826.py` exactly validates the final candidate order against the four existing chunk directories/logs, after which the unblind script reproduces the result. A fresh preparation run uses the fixed sorted-before-shuffle rule and is a new deterministic manifest; it must not be mixed with the labels from the discarded set-order run.

Validation completed: final JSON consistency checks passed; the blinded-file forbidden-metadata scan passed; `python -m unittest -v test.test_reward_scorer` passed all 9 tests; `git diff --check` passed. The unrelated vLLM lifecycle test was not runnable in the current base environment because importing `agentflow.server` requires missing `aiohttp`; no dependency was installed for this offline audit. No GPU/Ray/vLLM process was active at handoff.
