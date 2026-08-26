# Outcome-reward data-quality handoff

## Observed facts

- The source is the completed full audit at commit `96ef69f`: 100 fixed groups / 400 saved rollouts. This task used only existing JSON/manual labels; no new rollout or judge call was made.
- The full audit had 15 unresolved groups (45 ambiguous rows) and one explicit benchmark GT defect: Klein bottle `GT=6` versus the standard answer 8.
- The quality layer therefore excludes 16 groups. It does not exclude a prompt merely because the model was wrong, the prompt was difficult, the scorer had an error, or answers had surface diversity.

## Quality policy

- `clean`: no supported question/GT quality defect in the full semantic audit; eligible.
- `ambiguous_question`: the question/answer target cannot be resolved to one binary semantic interpretation from the supplied context.
- `stale_or_time_sensitive`: a missing date/season anchor makes the target change across time.
- `underspecified`: required function, source, aggregation, or geometric context is missing.
- `definition_sensitive`: the answer changes with domain, classification, or definition conventions.
- `gt_defect`: the supplied ground truth is contradicted by the supported standard interpretation.

The manifest is group-level and versioned (`2026-08-26.v1`). It retains evidence and the original/corrected audit classes but contains no raw answer text. The 15 unresolved groups are all excluded; the Klein bottle group is separately marked `gt_defect`. Conditional definition issues such as the unqualified regular-hexagon target are excluded conservatively rather than silently treated as clean.

## Manifest results

| primary category | groups |
|---|---:|
| clean / eligible | 84 |
| ambiguous_question | 2 |
| stale_or_time_sensitive | 4 |
| underspecified | 4 |
| definition_sensitive | 5 |
| gt_defect | 1 |
| excluded total | 16 |

The eligible subset has 44 mathhard and 40 NQ groups. The exclusion set has 6 mathhard and 10 NQ groups. `nq:39070` carries both definition-sensitive and conditional-GT-defect evidence; its primary category is definition-sensitive.

## Static re-statistics from existing labels

### Eligible 84 groups — original saved rewards

- Overall: `0/4=4`, `1/4=3`, `2/4=6`, `3/4=5`, `4/4=66`; mixed `14/84 = 16.67%`.
- NQ (40): `0/4=3`, `1/4=0`, `2/4=4`, `3/4=2`, `4/4=31`; mixed `6/40 = 15.00%`.
- mathhard (44): `0/4=1`, `1/4=3`, `2/4=2`, `3/4=3`, `4/4=35`; mixed `8/44 = 18.18%`.

### Eligible 84 groups — manual corrected outcomes

- Overall: `0/4=3`, `1/4=3`, `2/4=3`, `3/4=5`, `4/4=70`; mixed `11/84 = 13.10%`.
- NQ (40): `0/4=3`, `1/4=0`, `2/4=1`, `3/4=3`, `4/4=33`; mixed `4/40 = 10.00%`.
- mathhard (44): `0/4=0`, `1/4=3`, `2/4=2`, `3/4=2`, `4/4=37`; mixed `7/44 = 15.91%`.

These are static numbers from the audit, not regenerated outcomes. The separate scorer replay fixes five deterministic false negatives, including two groups that become all-correct under the reward. The manual corrected distribution already treats those answers as correct; the quality filter is independent of that scorer change.

## Limitations and recommendation

- This is a conservative manifest review, not an automatic universal benchmark validator. A clean label means “no supported defect found in this audit,” not “permanently timeless or unambiguous.”
- The excluded set contains question/GT quality issues, not a subjective difficulty curriculum. Difficulty should later be estimated from reproducible pass-rate and group-variance measurements.
- The corrected eligible mixed signal is 13.10% overall, with 10.00% NQ and 15.91% mathhard. That is lower than the raw eligible 16.67% and below the previously observed 19% full-set raw rate. First use the clean/eligible manifest for the next controlled variance-aware sampling study; do not use the excluded prompts to infer GRPO signal quality.

## Files and checks

- `log/2026-08-26_outcome_reward_data_quality_manifest.json`
- `log/2026-08-26_outcome_reward_clean_eligible_manifest.json`
- `log/2026-08-26_outcome_reward_exclusion_manifest.json`
- `scripts/build_outcome_reward_data_quality_manifest_20260826.py`
- JSON was parsed after generation. No external API, GPU, rollout, optimizer, or checkpoint was used.
