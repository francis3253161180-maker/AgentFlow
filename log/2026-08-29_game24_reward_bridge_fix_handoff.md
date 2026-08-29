# Game24 reward bridge correctness handoff (updated after final-answer extraction)

## Observed facts

- The non-structured path in `train/rollout.py` now extracts a conservative
  final-answer boundary and passes that candidate to `eval()`.
  `train/utils.py::compute_score()` and the rollout `eval()` both dispatch an
  identifiable four-number Game24 question to `game24_reward_decision()`.
- Before this change, `game24_reward_decision()` accepted strict JSON and
  marked candidates, but did not try the already-extracted bare expression.
  Therefore a valid expression such as `3 * 8 * (13 - 12) = 24` was rejected
  while `<answer>3 * 8 * (13 - 12)</answer>` was accepted.
- A direct local reproduction on the production function gave `bare=False,
  tagged=True` at the base revision and `bare=True, tagged=True` with the
  worktree change. No provider or model was used.
- The preserved complete random30 evidence contains 120 raw rollout JSONs in
  `/root/autodl-tmp/AgentFlow/rollout_data/46.38.243.197/random30-fresh-rollout-replay-20260828_20260829-104119`.
  The raw stored reward is `0.0` for all 120 rows, consistent with the bridge
  failure. The run itself was already complete (30 prompt groups, four rows
  per group); this task did not regenerate it.

## Root cause and code change

- Added `extract_final_answer()` in
  `agentflow/agentflow/models/structured_outputs.py` and routed the
  non-structured final output path in `train/rollout.py` through it. It
  prefers the last `<answer>...</answer>`, then an explicit Answer heading or
  section, and otherwise preserves the legacy full-output fallback.
- After JSON and explicitly marked candidates, `game24_reward_decision()`
  validates the complete input string as one bare expression with the
  existing `validate_game24_expression()` implementation.
- Validation is unchanged and remains strict: Python AST, exact
  `fractions.Fraction` arithmetic, exactly four supplied numbers with the
  original multiset, and exact value 24. Arbitrary prose is not searched or
  promoted, generic non-Game24 scoring is unchanged, and reward remains
  binary `0.0/1.0`.
- Added `test/test_game24_reward_bridge.py` covering bare, tagged, JSON,
  invalid arithmetic, wrong multiset, unrelated prose, the production
  `compute_score()` bridge, and non-Game24 generic-score routing.
- Added the reusable offline script
  `scripts/rescore_gameof24_reward_bridge_20260829.py`. It invokes production
  `compute_score()` and never writes to raw rollout files or calls a provider.

## Tests and static checks

The following completed successfully during this task:

```text
python -m py_compile agentflow/agentflow/models/structured_outputs.py train/utils.py train/rollout.py scripts/rescore_gameof24_reward_bridge_20260829.py test/test_game24_reward_bridge.py
python -m unittest -v test.test_game24_reward_bridge test.test_p0_scientific_correctness test.test_reward_scorer
```

The unit command passed all 33 tests in the final worktree. The environment was explicitly set to
`AGENTFLOW_DISABLE_EXTERNAL_LLM=1` and `AGENTFLOW_REWARD_JUDGE_ENABLED=0`.
The remaining delivery checks (`git diff --check`, JSON parsing and scoped
secret scan) are recorded in the final handoff command output.

## Offline production rescore

The complete result is in
`log/2026-08-29_game24_reward_bridge_fix_results.json`.

| Measure | Stored run | Production rescore |
|---|---:|---:|
| Rollouts | 120 | 120 |
| Prompt groups | 30 | 30 |
| Successful rewards | 0 | 62 |
| Zero rewards | 120 | 58 |
| Mean reward | 0.0000 | 0.5167 |
| Group 0/4 | 30 | 5 |
| Group 1/4 | 0 | 5 |
| Group 2/4 | 0 | 6 |
| Group 3/4 | 0 | 11 |
| Group 4/4 | 0 | 3 |

All 120 rows routed through `game24_strict_deterministic`; generic scorer,
DeepSeek judge, cache, and API were not used. There were 62 stored-to-fixed
reward discrepancies, all `0.0 -> 1.0`, because the stored run predates the
bridge. Relative to the previous 59/120 field rescore, exactly three rows
changed `0.0 -> 1.0` after explicit Answer-section extraction: groups 568,
782, and 1115. The individual file, group, answer excerpt, hash and strict
decision reason are retained in the JSON result.

The requested reference `61/120` cannot be reproduced as written. Its own
reference bins `0/4=4, 1/4=9, 2/4=10, 3/4=3, 4/4=4` sum to 30 groups but imply
only 54 successful rollouts, not 61. The fixed production result is therefore
reported as observed, not adjusted toward that reference.

## Intermediate-stage audit

Using the same raw `total_result` objects, the script searched non-prompt
intermediate fields with the same strict validator. `direct_output` was
excluded from intermediate evidence; this is an observational audit, not a
causal role attribution and not a generic AgentFlow verifier correctness
claim.

- 15 trajectories had at least one strict legal expression in an intermediate
  field but an invalid final answer under the fixed production scorer.
- Wrong-final trajectory observations by stage (a trajectory can occur in
  more than one stage): `planner_fixed_analysis=8`,
  `executor_tool_result=5`, `verifier_feedback=10`.
- `memory_state` also repeated valid expressions in 4 of these trajectories;
  it is a copied state field and is not counted as an independent role.
- Representative cases and the first 12 compact examples are in the result
  JSON. They include valid expressions in planner analysis/tool output or
  verifier feedback followed by a wrong-number or wrong-arithmetic final
  expression. The observed count is 17, not the earlier informal “about six”
  degradation; the evidence schema and strict raw-field audit are different,
  so the earlier number is not silently substituted.

## Calibration cleanup boundary

The separate `reasoning_effort=low` 3x4 calibration was stopped before this
audit because its reward bridge was invalid. Its partial log/trajectory
evidence was preserved under `/root/autodl-tmp/tmp/gameof24_low_calibration_20260829/`
and the corresponding small `log/` files. No partial calibration rows were
included in the 120-row rescore. The two clearly associated orphan
`multiprocessing` processes were targeted with SIGTERM and the surviving
entry became a PPID=1 defunct process; it held no GPU allocation and could not
be removed by signaling. No active Ray/vLLM/training process or GPU compute
allocation remained after cleanup.

## Recommendation

The bridge and conservative final-answer extraction fix are suitable for
review. This handoff intentionally stops after the offline rescore and the
separate bounded causal sanity check; no new training or reward judge call is
started here. The historical 61/120 reference still requires reconciliation
with its inconsistent bins.

Implementation base revision: `42e4120`.
Delivery commit: reported by the final repository handoff after commit/push.
