# Offline MuSiQue n=8 outcome-vs-transition diagnostic — analysis handoff

Generation parent commit: `af79d1da1872e35c65f2f1344e51ec34591adc57`. Diagnostic implementation/evidence commits: `fd8699ea71a5bb0ffcb118bb344883c66d385f6b`, followed by measurement-completeness commit `b67c3a5`. Branch: `experiment/offline-musique-single-policy`. Worktree: `/root/autodl-tmp/AgentFlow-offline-musique`. Seed: `20260830`.

Configuration and compact results are in `log/2026-08-30_offline_musique_n8_transition_diagnostic_results.json`. Dataset/corpus path is `/root/autodl-tmp/offline_musique_artifacts_20260830/offline_musique_corpus_v1.json`, SHA256 `9f6b7fc8e6c3180ae4d2c11e86c76fdc00c39847d0f68b3784b1a05806fc1014`.

## Boundary and freeze audit

This was one explicitly authorized diagnostic run of the same fixed 32 Phase-C-v4 qids with `n=8`, for 256 rollouts. It was not training. The generation runner SHA256 remained `62e5e93d9f1cdc273f8e5606c2f39393c2a881920057e714158bd594d8dfdabd`, and the actor/protocol module SHA256 remained `915e5aa1136e6960ee14b8676c99769e20813515d30fcbc7003c86a5add5b87e`; both equal their versions at `af79d1d`. DECISION and EVIDENCE_UPDATE system-prompt hashes remained `197e2e115395b78c0e84097574b08a56815f04813f665833cabe8f0e06529ed0` and `6f8708037ec10948a216bd638f3d4604bfeec9ad5bb2cc049ecb71023f8d4920`.

The post-hoc analyzer verified identical qid order and all frozen policy/retrieval/configuration fields. It copied the 256 actor trajectories unchanged, then attached scorer annotations at a separate top-level field with `actor_visible=false`, `computed_post_generation=true`, and `training_weight=0`. An independent structural scan checked 1,147 actor prompts plus every compact-memory before/after payload and found zero transition-diagnostic or scorer-label keys. Terminal reward was not changed. No GRPO, optimizer step, actor update, HOB, reranker, external LLM, web retrieval, or reward shaping ran.

## Observed facts

### Execution and validity

- Completed rollouts: 256.
- Format-valid/answer-terminated rollouts: 215. Dropped fail-closed rollouts: 41, comprising 7 DECISION and 34 EVIDENCE_UPDATE format failures.
- DECISION validity: 904/911 = 99.2316%. EVIDENCE_UPDATE validity: 655/689 = 95.0653%. Overall transition schema validity: 97.4375%.
- Wall time: 177.2115 s; throughput: 86.6761 rollouts/min.
- GPU peak/final memory: 21,272/2 MiB. There was no OOM, CUDA error, Ray error, traceback, or cleanup error. vLLM reset its prefix cache and exited; it emitted the previously observed process-group-not-destroyed warning after the drained exit.
- Answer EM: 0.042969. Gold support retrieval recall: 0.438477. Gold support selection recall: 0.190430. Count-based distractor-selection rate: 0.294737. Repeated-query rate: 0.235123. Premature-answer rate among 215 answers: 0.981395.
- Grounded terminal positives: 4/256; mean reward 0.015625. Exact final support set: 4/256 = 0.015625. Exact-support-set was diagnostic only and did not replace terminal reward.

### Question outcome groups

The outcome histogram is: `0/8: 30`, `1/8: 1`, `2/8: 0`, `3/8: 1`, and `4/8` through `8/8: 0`. Thus 30/32 groups were all-zero, 2/32 were mixed, and 0/32 were all-one.

| qid | terminal reward vector | population variance |
|---|---|---:|
| `2hop__75714_21969` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop1__58323_375563_161848_67585` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop3__270458_88460_30152_20999` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__56851_343058` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop3__668721_132409_371500_35031` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__56806_7298` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__145427_106426_157788` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__337705_132457_47686` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop1__88342_75218_128008_86588` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop2__90098_60649_10557` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__129499_85379` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop1__443274_17130_70784_79935` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__640171_228453_86925` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__41865_55331_34700` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__57638_615257` | [1, 0, 0, 1, 0, 1, 0, 0] | 0.234375 |
| `2hop__32254_84601` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop3__601548_836463_161616_77103` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop3__387712_132409_223216_35031` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__507722_124896` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop3__152056_698586_1926_54362` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__389955_132457_47686` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop2__71753_623626_70784_61381` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__426806_42197_18397` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `4hop1__726391_153080_33952_34053` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__604991_339990_15538` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__123190_519940_18967` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__532383_768138` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__50910_177869` | [0, 0, 1, 0, 0, 0, 0, 0] | 0.109375 |
| `2hop__3739_13529` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__53690_161697_73916` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `3hop1__222497_737465_93723` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |
| `2hop__551235_310309` | [0, 0, 0, 0, 0, 0, 0, 0] | 0.000000 |

These are diagnostic question groups, not executed GRPO batches. No advantage vector was materialized. Advantage min/max/mean/std, `pg_loss`, `grad_norm`, entropy, training `old_log_prob`, and global step are not recorded, not numerical zeros. `optimizer.step=false` and `update_actor=false`.

### Final support-score and transition distributions

Final cumulative F1 varied within 23/32 qids = 71.875%; final cumulative F2 varied within the same 23/32 = 71.875%. Terminal outcome varied within only 2/32 = 6.25%. Each final-score mixed rate is therefore 65.625 percentage points higher and 11.5 times the terminal mixed rate. Full per-qid mean/std/min/max/value/unique-with-tolerance records are in the compact JSON.

Across 655 valid EVIDENCE_UPDATE transitions:

| Signal | Positive | Zero | Negative | Question signal | Trajectory with positive |
|---|---:|---:|---:|---:|---:|
| delta_F1 | 134/655 = 20.4580% | 520/655 = 79.3893% | 1/655 = 0.1527% | 23/32 = 71.875% | 124/256 = 48.4375% |
| delta_F2 | 134/655 = 20.4580% | 520/655 = 79.3893% | 1/655 = 0.1527% | 23/32 = 71.875% | 124/256 = 48.4375% |

One qid (3.125%) contained both positive and negative deltas for each metric. Evidence-update ordinal positive rates for both F1 and F2 were: ordinal 1, 94/211 = 44.55%; ordinal 2, 16/116 = 13.79%; ordinal 3, 5/94 = 5.32%; ordinal 4, 8/86 = 9.30%; ordinal 5, 8/81 = 9.88%; ordinal 6, 3/67 = 4.48%. The sole negative occurred at ordinal 6. These pooled transitions have heterogeneous states and are labeled only `question-level transition signal availability`; they are not asserted to be valid GRPO normalization groups.

### Overlapping failure taxonomy

- Retrieval miss (not all gold support retrieved): 235/256 = 91.797%.
- At least one returned gold support not selected: 160/256 = 62.5%.
- Any distractor selected: 56/256 = 21.875%.
- Premature answer: 211/256 = 82.422% overall, or 211/215 = 98.140% conditional on answering.

On the common trajectory denominator, incomplete retrieval coverage is the most frequent failure. Premature stopping is almost universal conditional on the actor answering. These categories overlap and should not be summed.

## Required A–G analysis

**A. Did n=8 increase outcome mixed-group rate?** Yes, from Phase-C-v4 n=4's 1/32 = 3.125% to 2/32 = 6.25%, an increase of 3.125 percentage points and 2x. This remains sparse: 30/32 groups are all-zero.

**B. What is the outcome histogram?** `0/8: 30`, `1/8: 1`, `2/8: 0`, `3/8: 1`, `4/8: 0`, `5/8: 0`, `6/8: 0`, `7/8: 0`, `8/8: 0`.

**C. How much larger are final-score mixed rates?** Final_F1_mixed and final_F2_mixed are both 23/32 = 71.875%, versus 2/32 = 6.25% terminal mixed: +65.625 percentage points and 11.5x each.

**D. How dense are delta signals?** For both delta_F1 and delta_F2, 20.458% are positive, 79.389% zero, and 0.153% negative. The question-level signal rate is 23/32 = 71.875% for each; 124/256 = 48.438% of trajectories have a positive delta. Both signs occur within one qid.

**E. Which failure dominates?** Incomplete retrieval coverage dominates on the common trajectory denominator at 91.797%. Premature answering is the strongest conditional behavioral failure at 98.140% of answers. Returned-but-not-selected support is also substantial at 62.5%; distractor incidence is lower at 21.875%.

**F. Is n=8 a reasonable baseline for a later GRPO pilot?** It is operationally stable and doubled the observed terminal mixed-group rate, but the current terminal-only signal still leaves 93.75% of question groups with zero variance. Based only on these measurements, n=8 is not yet a strong terminal-reward-only GRPO baseline; it would be a narrowly reasonable feasibility baseline only if a later authorization explicitly accepts mostly zero-advantage groups. The earlier systems throughput adoption stop remains in force outside this diagnostic.

**G. Does evidence support the transition-progress hypothesis?** Yes at the signal-availability level: 71.875% of qids show final F1/F2 variation and question-level transition signal, compared with 6.25% terminal outcome variation, and 48.438% of trajectories contain positive progress. This does not establish an optimization method or training benefit. No transition reward has been used for optimization, the transition states are heterogeneous, and training weight remained zero.

## Hypothesis and conclusion

**Hypothesis:** post-hoc scorer-side support progress is denser than grounded binary outcome under frozen Phase-C-v4 behavior.

**Conclusion:** the measured distributions support the hypothesis as a credit-availability diagnostic. They do not authorize reward shaping, define a valid transition-level GRPO grouping scheme, or demonstrate improved learning. The dominant evidence bottlenecks remain retrieval coverage and premature stopping, with a large support-selection gap when gold evidence is returned.

No further run is recommended automatically. The reviewer must separately decide whether to design a causally valid transition-credit method, authorize a terminal-only n=8 feasibility pilot despite sparsity, or investigate retrieval/policy behavior. Do not weaken grounding.

## Evidence paths

- Full enriched raw trajectory pack: `/root/autodl-tmp/offline_musique_artifacts_20260830/n8_transition_diag_v1_enriched_trajectories.json`, SHA256 `83c8dd0cf28a6a22a376673422e3008d03210071c5a44d696178a028af13c523`.
- Original raw actor pack: `/root/autodl-tmp/offline_musique_artifacts_20260830/n8_transition_diag_v1_trajectories.json`, SHA256 `db2debdd15174c9db4cea209db9547c892074534c627f3b678ed017c3bfab95c`.
- Runner summary: `/root/autodl-tmp/offline_musique_artifacts_20260830/n8_transition_diag_v1_runner_results.json`, SHA256 `99709c7c0963facb0a44c011df052db5582fb97137c242d88bcabb6a9b6bd6f6`.
- GPU log: `/root/autodl-tmp/offline_musique_artifacts_20260830/n8_transition_diag_v1_gpu.csv`, SHA256 `de73caccc6d21872a13fd3bce34bc1f63edc68faafca0c71e2c54187625158a5`.
- Runtime log: `/root/autodl-tmp/offline_musique_artifacts_20260830/n8_transition_diag_v1_runtime.log`, SHA256 `780236ac36d6004a632dc5c2920b2c82da9381cbdfdcfee97795497c6e05ea83`.
