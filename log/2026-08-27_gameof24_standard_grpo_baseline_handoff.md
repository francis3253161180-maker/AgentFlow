# GameOf24 Standard GRPO baseline — provenance blocker handoff

## Observed facts

- Checked branch: `experiment/flow-grpo-3b-lora`.
- Checked commit: `a13c9d649542063305b512cfd1eeca1aad8d7c4b`.
- The requested experiment was **not started**. There was no new rollout,
  training process, backward pass, optimizer step, parameter update, post-eval,
  checkpoint write, or DeepSeek judge call for this task.
- The stage-2 fixed 30-group GameOf24 probe remains the pre-eval reference:
  120/120 valid rollouts, reward mean `0.6417`, group bins
  `0/4=3, 1/4=5, 2/4=7, 3/4=2, 4/4=13`, and mixed ratio `46.67%`. Those
  30 groups are probe/evaluation data and are forbidden from any future train
  pool.
- The only local GameOf24 data found is
  `test/gameof24/data/data.json`:
  - 300 rows, SHA-256
    `ffb0c95018ba400931f2940a2dabae666f1ed6d0383beadd969f0447869a14ff`;
  - fields are `pid`, `question`, `answer`, `image`, and `query`; there is no
    train/eval split field or training provenance in the file;
  - Git history shows the file first appears in commit
    `b94006436b8712ab8682846fb0d886a5f174f2d4`, with no earlier repository
    history for this path.
- `test/gameof24/run.sh` is an evaluation runner. It reads
  `gameof24/data/data.json`, invokes `solve.py`, writes benchmark results, and
  defines benchmark indices `0..99`; this is direct repository evidence that
  the local file is being used as a benchmark fixture, not a documented train
  pool.
- `data/train/combined_train.parquet` contains 182,190 rows whose only source
  values are `nq` and `mathhard`; it contains zero GameOf24 rows. No official
  or analogous GameOf24 training source is present in the repository or local
  training data inventory.
- The 30 stage-2 eval/probe rows and historical probe rows are already recorded
  in `log/2026-08-27_stage2_30prompt_confirmation_sample_manifest.json`, with
  identifier/content hashes and zero internal overlap. Selecting another 60
  rows from the same 300-row fixture would still be selecting benchmark
  evaluation data, regardless of whether those rows were not previously
  rolled out.

## Provenance gate and decision

The requested 60-row training pool cannot be validated as an official train
split, licensed analogous training source, or documented development split.
The local evidence instead supports only the conservative interpretation that
it is a benchmark/evaluation fixture. Therefore the provenance gate is
**blocked**.

Per the explicit task constraint, I did not silently use the other 60 rows (or
any other rows) as formal training data. No config or scorer rule was changed,
and no baseline run was attempted. There are consequently no train-step
metrics, gradient metrics, post-eval metrics, runtime, GPU peak, or cleanup
markers to report for a baseline; these fields are represented as null or zero
in the results JSON rather than inferred from the old probe.

## Hypotheses

- The local 300-row fixture may have an upstream source or a separate train
  split outside this checkout, but that provenance is not verifiable from the
  current repository/local inventory. This is an uncertainty, not evidence that
  the rows are safe for training.
- The prior stage-2 mixed-rich result indicates that GameOf24 is a promising
  *probe/eval distribution*, but it does not authorize benchmark-test
  contamination or establish a train split.

## Recommendation

Obtain and record a verifiable official GameOf24 train split or a separately
licensed analogous training source. Before any future baseline run:

1. record source/version/split and file hash;
2. construct a fixed seed-20260827 60-row train manifest without answer-based
   selection;
3. compare row identifiers and content hashes against all 30 stage-2 eval rows
   and every historical GameOf24 probe row;
4. keep the stage-2 30 prompts as the unchanged pre/post eval set; and
5. only then request/execute the Standard GRPO baseline under the specified
   protocol.

Do not start HOB, variance-aware sampling, or formal GRPO training from this
  blocked state. The requested task stops here.

## Artifacts and checks

- Blocked manifest:
  `log/2026-08-27_gameof24_standard_grpo_train_manifest.json`
- Blocked results:
  `log/2026-08-27_gameof24_standard_grpo_baseline_results.json`
- This handoff:
  `log/2026-08-27_gameof24_standard_grpo_baseline_handoff.md`
- Raw logs, rollout data, caches, and model files were not added to Git.
- Relevant scorer/cleanup tests and static checks were run after writing these
  small blocker artifacts; no new compute run was launched.
