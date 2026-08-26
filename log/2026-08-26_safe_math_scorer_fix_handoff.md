# Safe math scorer fix handoff

## Observed facts

- Scope was the saved full outcome-reward audit: 100 prompt groups / 400 rollout rows from audit commit `96ef69f`. No rollout, GPU process, judge API, or new external call was used.
- The previous deterministic route had 192 rows and 9 clear false negatives. Four rows were the Klein bottle case (`GT=6`, answer `8`); the full audit identified this as a GT defect, not a scorer defect. The other five were generic math matching failures: four affirmative `a=b` conclusions and one explicit final `2^{n+1}` answer in a longer derivation.
- Before the change, all clear rows were `TP/TN/FP/FN = 304/33/2/16`. The deterministic subset was `168/10/0/9` for `TP/TN/FP/FN`.
- Existing math cases such as `\dfrac{1}{2}`, `1+\sqrt{2}`, and `x+1` were included in regression tests and remained locally correct.

## Code changes

- `train/utils.py` now extracts math spans from `\(...\)`, `\[...\]`, dollar delimiters, and `\boxed{...}` before comparing expressions. Equation right-hand sides are also considered when a labelled left side cannot be parsed (for example `|H_n| = 2^{n+1}`).
- Simple equality claims are parsed as SymPy equalities and compared by their left-minus-right form, allowing orientation-equivalent claims without shared-number-token matching.
- Multiple unmarked bare equations remain judge-routed as `ambiguous_math_candidates`; the code does not promote an arbitrary candidate list to a positive reward.
- Existing conflict/negation guards remain active. A rejected or corrected equation is not locally rewarded.
- Added focused tests for affirmative equality conclusions, explicit long math answer blocks, rejected equations, multi-candidate equations, and the existing scorer suite.

## Offline regression

The replay substituted only a new high-confidence local decision. Saved judge outcomes were retained; this is not a new semantic-judge run.

| scope | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| saved reward, all clear rows | 304 | 33 | 2 | 16 |
| post-fix replay, all clear rows | 309 | 33 | 2 | 11 |
| post-fix local deterministic rows | 173 | 10 | 0 | 4 |

The five changed rows were exactly the five scorer FNs described above. They belong to two groups: `mathhard:33486` changes from `0/4` to `4/4`, and `mathhard:19926` changes from `3/4` to `4/4`. All five had manual label `correct`. The four remaining deterministic FNs are the Klein bottle GT defect and are intentionally not changed by the scorer.

There were zero new clear FPs and zero new clear FNs among the changed rows. The deterministic route count stayed 192 in the offline replay; the change improves matching within the existing high-confidence route rather than broadening open natural-language phrase matching.

## FP/FN analysis

- The fix addresses a parser/extraction limitation, not a sample-specific answer. It recognizes an equivalent formula inside a final math claim and preserves the prior safe rejection/uncertainty behavior for conflicting or multi-candidate prose.
- The two saved clear FPs remain judge-route failures (`nq:31767` and `nq:54331`); this change does not claim to solve semantic judge contamination.
- The Klein bottle case remains a data-quality exclusion. Relaxing numeric matching to turn `6` and `8` into a match would be incorrect.

## Remaining uncertainties

- The replay does not re-run DeepSeek and therefore cannot measure judge behavior after the local route changes. It only proves that saved judge results are unchanged and that the five newly local decisions agree with the existing manual labels.
- SymPy equivalence remains intentionally limited to parseable, high-confidence forms. Domain-sensitive identities and long unmarked derivations remain on the semantic route.

## Recommendation

Keep this minimal scorer change and add the data-quality filter described in the companion handoff. Do not start training from the raw 100-prompt set until the excluded/flagged quality groups are removed or separately handled. After the clean subset is accepted, a controlled variance-aware sampling study is reasonable; this patch alone is not evidence to change rollout temperature, `n`, or Flow-GRPO.

## Verification

- `python -m unittest test.test_reward_scorer` — 11 tests passed.
- Offline replay: `python scripts/audit_safe_math_scorer_fix_20260826.py --audit log/2026-08-26_outcome_reward_full_audit_results.json --output log/2026-08-26_safe_math_scorer_fix_results.json`.
- No DeepSeek/API calls, GPU work, rollout, optimizer update, or checkpoint write.
