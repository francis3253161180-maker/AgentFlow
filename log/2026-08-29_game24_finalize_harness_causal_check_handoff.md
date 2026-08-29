# Game24 final-answer harness and planner-temperature causal sanity handoff

## Observed facts

- The preserved source manifest is the first three prompts of the frozen
  random30 calibration: groups 43 `[3,8,12,13]`, 58 `[2,3,4,4]`, and 181
  `[6,9,9,11]`. The 3×4 run produced exactly 12 valid rollouts.
- Run evidence is preserved in:
  - train log:
    `log/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_train.log`
  - rollout log:
    `log/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_rollout.log`
  - raw rollout directory:
    `rollout_data/46.38.243.197/gameof24-planner-temp0-causal-sanity-20260829_20260829-135524/Qwen2.5-7B-Instruct_20260829-135525/train`
  - run metadata:
    `/root/autodl-tmp/tmp/gameof24_planner_temp0_causal_sanity_20260829/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_run_meta.json`
- Code revision at launch was
  `2727c73d4c4b614357ad077c121d3b395d307ba4`.

## Code changes

- Added `extract_final_answer()` in
  `agentflow/agentflow/models/structured_outputs.py` and routed the
  non-structured `train/rollout.py` final answer through it. Parsing is
  limited to the last answer tag, explicit Answer heading/section, or legacy
  full-output fallback. No arbitrary prose mining or intermediate-field
  rescue was added; strict Game24 AST/Fraction/multiset/value validation is
  unchanged.
- Corrected offline audit stage labels in
  `scripts/rescore_gameof24_reward_bridge_20260829.py` according to verified
  source call paths. Added focused extraction and bridge tests.
- Added bounded runner and offline analyzer:
  `scripts/run_game24_planner_temp0_causal_sanity_20260829.sh` and
  `scripts/analyze_game24_planner_temp0_causal_sanity_20260829.py`.

## Configuration and protocol

The causal run changed only planner-main temperature relative to the
calibrated target: `0.7 -> 0.0`. Other recorded settings were:

| Item | Value |
|---|---|
| Flow | original Initializer → Planner → Executor/tool → Memory → Verifier → final → reward |
| Actor | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` + current LoRA snapshot |
| LoRA | r=8, alpha=16, all-linear; snapshot hash `c841011c9800784d97d381e0c5781b299c4d679aa7f8e824ad4773bc61bf8b38` |
| Fixed roles | `doubao-seed-2-0-lite-260428`, temperature 0, `reasoning_effort=minimal` |
| Planner-main temperature | 0.0, explicitly in runtime config and environment |
| Rollouts | n=4, 3 prompts, 12 expected; rollout-only, optimizer steps 0 |
| Lengths | max prompt 1536, max response 1024, vLLM max model length 4096 |
| vLLM smoke limits | GPU utilization 0.50, max sequences 1, max batched tokens 1024 |
| Persistence | `val_only=true`, `save_freq=0`, checkpoint disabled |
| Reward | strict deterministic Game24; external reward judge disabled |

Config SHA256 is
`1a0c72b57cb9f1c11d1f73826943e1231cca5b11f5f35ff59c8a290384366837`.
Prepared data SHA256 is
`4ffbb758a9cbce4fceee2b3f38129bde7319f047011d83cb8c297ef0899482e5`; the
three-prompt manifest SHA256 is
`39c2d8d3c006999253746abce0598bd9acb4167a3bd721c9e36f312307ba9083`.

## Offline audit (Tasks A–C)

The same preserved 120 trajectories were rescored offline with external
calls disabled. Current production result: 62/120 positives, mean reward
0.5167, group bins `0/4=5`, `1/4=5`, `2/4=6`, `3/4=11`, `4/4=3`, hence
22/30 mixed groups (73.33%). Compared with the previous 59/120 field rescore,
three explicit Answer-section rows changed from 0 to 1. All 120 rows used
the strict deterministic route; no judge/API/cache calls occurred. Corrected
stage observations are `planner_fixed_analysis=8`,
`executor_tool_result=5`, `verifier_feedback=10`, and copied
`memory_state=4` among 15 wrong-final trajectories.

## Causal sanity check (Task D)

The per-group reward vectors were:

| Group | Reward vector | Bin |
|---|---|---|
| 43 | `[1,0,1,1]` | 3/4 |
| 58 | `[0,1,0,1]` | 2/4 |
| 181 | `[0,0,1,0]` | 1/4 |

Valid rollouts were 12/12, mean reward was 0.5000, and bins were
`0/4=0`, `1/4=1`, `2/4=1`, `3/4=1`, `4/4=0`. Mixed and nonzero-variance
group rates were both 3/3 (100%).

## Actor versus downstream evidence

- Within every group, persisted planner-main action sequences had exact
  uniqueness 4/4. Semantic action uniqueness was 3, 4, and 4 for groups 43,
  58, and 181. No group had effectively identical planner actions with mixed
  rewards.
- Final answer uniqueness was 4, 4, and 3; tool-path uniqueness was 2 in
  each group. Downstream fixed-role signatures were unique 4/4 in every
  group, and all three groups had downstream variation.
- Therefore planner temperature 0 did not collapse actor action diversity in
  this small run. Mixed outcomes co-occurred with downstream/fixed-role
  output variation, so the observed reward variance cannot be attributed to
  planner sampling alone. The old temperature-0.7 run for these groups was
  incomplete and used the invalid historical reward bridge; its persisted
  action sequences were also non-identical, so it is not a valid paired
  reward control.

## Runtime, cleanup, and safety

- Runtime inferred from logs: approximately 665.441 seconds. GPU monitor:
  134 samples, peak 16,899 MiB; final `nvidia-smi` allocation was 0 MiB.
- Final cleanup markers show `trigger=normal_complete`,
  `outstanding=0`, `drain_complete=1`, `reset_prefix_cache` only after drain,
  `sleep_start`, `sleep_complete`, and `complete=1 drained=1` with duration
  0.685 seconds. No `blocks are not freed yet`, CUDA illegal-memory, CUDA OOM,
  Ray/vLLM fatal error, deadlock, or forbidden training marker was observed.
- Route evidence records trainable Qwen actor LoRA and fixed Doubao roles. No
  external Game24 reward judge was called. No formal training, backward,
  optimizer update, checkpoint, GRPO, or HOB update was run. GPU/Ray/vLLM
  were clean at completion; unrelated historical AgentOpsServer processes
  were left untouched.

## Remaining uncertainties

This is three groups and 12 rollouts, not a benchmark or a formal causal
estimate. There is no complete matched temperature-0.7 run with the same
fixed-role configuration and valid current reward bridge. Fixed-role
reasoning and tool responses varied, so downstream stochasticity is a
confounder for attributing reward variance. The run verifies lifecycle and
routing more strongly than it estimates temperature effects.

## Recommendation

The bounded answer-extraction and audit-label fixes are suitable for review.
The temperature-0 sanity run supplies no evidence that setting the planner to
zero removes useful group diversity; it also does not establish that
temperature 0 is better than 0.7. Do not infer a causal winner or start
formal training from this smoke. If a paired causal estimate is needed,
obtain explicit approval for a matched temperature comparison with the same
fixed roles and a valid deterministic reward path. No automatic follow-up was
started.

Delivery commit: the final repository commit hash is reported in the
completion handoff; no other worktree changes are part of this delivery.
