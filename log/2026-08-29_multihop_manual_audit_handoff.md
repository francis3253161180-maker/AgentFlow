# All-Qwen7B MuSiQue + 2Wiki manual trajectory audit

## Scope and evidence

This is an offline observational audit of the 80 persisted trajectories from
commit `3eae7da`; it made no model/API call and did not modify the planner,
verifier, tools, scorer, or benchmark data.  The final raw inputs are the 40
MuSiQue files under
`rollout_data/46.38.243.197/multihop-allqwen7b-musique-20260829_20260829-165503/Qwen2.5-7B-Instruct_20260829-165504/train`
and the 40 2Wiki files under
`rollout_data/46.38.243.197/multihop-allqwen7b-2wiki-20260829_20260829-171907/Qwen2.5-7B-Instruct_20260829-171908/train`.

The complete per-rollout audit, including raw file path/SHA256, final answer,
planner subgoal/context, tool-result excerpt, verifier analysis/decision, stop
reason, and manual label, is in
`log/2026-08-29_multihop_manual_audit_results.json`.  Its input probe manifest
SHA256 is `922b73e10a4724d61163d71f6a12e2e57494134a84311c720f71cc9a664d61a1`.
Manual labels mean semantic agreement with the supplied benchmark gold, not an
independent re-verification of every benchmark fact.

## Observed facts

### Semantic reward audit

| Dataset | Rollouts | Clearly correct | Clearly wrong | Ambiguous | TP / TN / FP / FN (clear only) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuSiQue | 40 | 4 | 34 | 2 | 1 / 34 / 0 / 3 |
| 2Wiki | 40 | 0 | 40 | 0 | 0 / 40 / 0 / 0 |
| Total | 80 | 4 | 74 | 2 | 1 / 74 / 0 / 3 |

The deterministic reward has no observed false positive in the 78 clear
rows, but it recognizes only 1 of 4 clearly correct answers: clear-positive
false-negative rate `3/4 = 75%`.  Its agreement on clear rows is `75/78 =
96.15%`, a class-imbalanced number dominated by true negatives and therefore
not evidence of good semantic recall.  MuSiQue has the three clear false
negatives; 2Wiki has no manually clear positive with which to estimate recall.

The following persisted MuSiQue trajectories are deterministic false
negatives.  Each had `reward=0`, despite a final answer semantically matching
the supplied gold and a verifier `stop_signal=true`:

- `step_1/idx_112/rollout_1a9c019b-bba5-4853-a5b7-885a44ed7cb0.json`
  (`sha256=309b3a6b…117413bf`): gold `Charles University`; answer says
  “Charles University (formerly known as the University of Prague)”.
- `step_2/idx_112/rollout_b0f7aaf3-e0a4-4e64-a1e3-a7c0f8f0172d.json`
  (`sha256=66a7e800…d5ced905`): gold `Charles University`; answer says
  “Charles University in Prague”.
- `step_1/idx_259/rollout_64df77c2-9b75-44ad-af36-c55b36df2d20.json`
  (`sha256=b6ef6e7a…63dac54`): gold `38`; answer says “38 games per team”.

Two additional Charles-University answers were intentionally left
`ambiguous_needs_semantic_judgment`: they mention University of
Prague/Charles University but also make an unscoped plural-employer claim that
includes Leiden.  They were not converted into either scorer errors or extra
correct examples.

### Planner and verifier behavior

All 80 trajectories selected `Generalist_Solution_Generator_Tool` as their
first tool.  There was no trajectory with two distinct tools.  The only tool
sequences were:

| Dataset | One broad-generator call | Two calls, same broad generator | Second-step rate | Distinct-tool rate |
| --- | ---: | ---: | ---: | ---: |
| MuSiQue | 26 | 14 | 35.0% | 0.0% |
| 2Wiki | 27 | 13 | 32.5% | 0.0% |
| Total | 53 | 27 | 33.75% | 0.0% |

For every two-step trajectory, the second step repeated the same broad
generator, rather than collecting an independent relation/evidence source.
The final verifier stopped on a non-gold claim in 25/40 MuSiQue and 30/40
2Wiki trajectories (55/80 total, 68.75%).  These cases are marked
`verifier_stop_true_after_non_gold_claim` in the JSON: the raw tool output
asserted a specific but non-gold answer and the verifier nevertheless declared
the memory complete.  In the remaining 19 trajectories the verifier kept
`stop_signal=false`, but the two-step harness ended them anyway.  One MuSiQue
first tool call had a command-syntax error; the next call retried the same
tool and then stopped.

The relational structure of the prompts is often multi-hop (for example,
film → director → nationality/workplace, person → birthplace → administrative
entity, or performer → birthplace → castle).  Yet the planner commonly sent
the full composite task to one generalist call.  Representative examples in
the raw JSON are:

- MuSiQue `idx_112`, Mach → employer: the planner’s one-call subgoal is
  “Identify the person Mach's principle was named after and find their
  employer.”  This did produce a correct answer in two trajectories, but no
  relation was independently checked.
- 2Wiki `idx_1145`, prince → mother → birthplace: one rollout first proposed
  “Find the place of birth of Frederick…'s mother”, then repeated the same
  generator for “Find the place of birth of Princess Sophie…”.  It stopped on
  `Berlin, Germany`, while gold is `Zürich`.
- MuSiQue `idx_171`, organization → area: one two-step trajectory did make a
  structurally sensible first subgoal (identify the Census organization) and
  second subgoal (locate/extract area), but the same generator supplied
  `100388.2`, not gold `17.037 square miles`, and the verifier stopped.

Thus the last example is a partial textual decomposition, not evidence of a
successful verified multi-hop tool path.  No clearly successful trajectory
used a separate retrieval/verification tool or two distinct evidence sources.

### Credit-assignment evidence

The only clear within-group success/failure contrast is MuSiQue `idx_259`
(Barcelona league games): candidate answers were `38` (reward 1), `38 games
per team` (reward 0; semantic false negative), `34`, and `306`.  All four
selected the same one-call generalist tool, used near-identical composite
subgoals, and received verifier stop signals.  Their divergence appears in
the broad generator's factual assertion (20-team versus 18-team / per-team
versus league-total interpretation), not in a distinct planner tool choice.

MuSiQue `idx_112` provides a weaker contrast: the two clearly correct
Charles-University answers and two unresolved plural-employer answers all use
the same broad tool and receive `stop_signal=true` except for one retry after a
tool-command syntax error.  It likewise does not isolate a planner action that
explains outcome variance.

## Hypotheses

1. The primary observed failure is likely a lack of grounded multi-hop
evidence: a frozen broad generator is asked to solve the entire composed query
and the frozen verifier accepts its own-style factual assertions.
2. On these samples, outcome differences inside a group are more consistent
with downstream frozen Qwen base generation variance than with a materially
different trainable `planner_main` tool-selection policy.  This is not a
causal estimate because planner text, generator sampling, and verifier output
were not independently intervened upon.
3. The raw reward mixed rate from the prior probe understates semantic mixed
outcomes at least for the three demonstrated false negatives.  A full semantic
relabeling of a larger sample would be required to quantify that bias beyond
these 80 trajectories.

## Conclusions

- The observed all-Qwen7B MuSiQue/2Wiki probe is not a useful positive
  binary-GRPO regime: it has only four clearly correct outcomes in 80, and the
  scorer misses three of those four.
- The 2Wiki all-zero reward result is directionally supported by the direct
  audit: all 40 answers are clearly non-gold relative to the supplied labels.
- The MuSiQue raw `1/40` reward count is semantically incomplete, not a
  reliable correctness estimate, because at least three additional answers
  are clearly correct.  The two unresolved temporal/multiple-employer answers
  should remain excluded rather than silently inflating either side.
- No planner/verifier or reward change was made.  The persisted dev/probe
  examples remain evaluation-only and must not be added to a training pool.

## Recommendation

Do not launch a training follow-up from these probe rows.  Before a new
multihop baseline is considered, obtain an approved task/tool setup that can
produce independently checkable intermediate relations, then perform a
pre-registered semantic reward audit on a fresh sample.  Any future training
pool must come from a non-overlapping official training split (with
identifier/content-hash checks), never these dev probe trajectories.

## Reproducibility and validation

- Collector: `scripts/audit_multihop_manual_audit_20260829.py`
- Command: `python scripts/audit_multihop_manual_audit_20260829.py`
- The collector asserts 10 groups × 4 trajectories per dataset and records
  SHA256 for every raw JSON.
- This task ran offline only; no rollout, training, optimizer, checkpoint,
  external model, or reward judge call was made.
