# Random30 reusable-rollout addendum handoff

## Observed facts

- The addendum was read in full. It requires preserving an existing 30×4 rollout corpus and, before rollout, persisting the exact behavior-policy adapter state and RNG metadata.
- The frozen random30 selection is present and internally consistent: 30 rows, seed `20260828`, XoT train commit `d69d3b8ddb75f888732e893394e8e5e5df4f4e6f`, train CSV SHA256 `8fb0cbd833ed083bd26c24d9f952d7952189d251bb2905bc58171939b6c3c803`, and manifest SHA256 `f2c8db2b44bf1d8e0879565a2c49215c212090e48cd15d267529f14ea098136d`.
- The 30 selected row IDs, puzzle tuples, and normalized puzzle hashes match the preparation output. No selected puzzle overlaps the prior stage-2 content set.
- No random30 rollout directory, 30×4 trajectory corpus, or random30 replay pack exists in the repository or local rollout-data tree. The only prepared local inputs are the manifest and a small parquet file.
- The prior 7B unified smoke recorded LoRA checksums, but it used a different four-prompt smoke dataset/run. It did not persist a reloadable adapter state plus RNG snapshot for this random30 behavior policy. Its checksum cannot establish the exact initial state required by the addendum.

## Blocker and decision

The exact initial adapter state is not feasible to restore from the available evidence, and there are no 30×4 trajectories to preserve. Therefore this run stops before rollout and before replay materialization. No new rollout, optimizer step, formal training, checkpoint, GPU/Ray/vLLM process, or external LLM call was started.

This is intentional: generating fresh random30 rollouts now would violate “do not rerun” and would make a later replay pack incomparable to the requested behavior-policy identity. Likewise, using the unrelated four-prompt smoke checksum would falsely label a different policy state as the random30 initial state.

## Frozen manifest details

The tracked manifest is [random30 sample manifest](/root/autodl-tmp/AgentFlow/log/2026-08-28_random30_len1024_context4096_probe_sample_manifest.json). Its selection is answer/outcome independent and remains suitable as a future candidate tranche A, subject to a new approved run that persists the adapter/RNG identity before generating trajectories. The XoT train source is explicitly the official XoT train split, not the canonical ToT split.

## Not produced

- 120 preserved trajectories and per-request telemetry: unavailable because no random30 rollout ran.
- Authentic pre-update replay pack: not created.
- Fresh-process replay load validation: not run.
- Any claim that this corpus is a reusable D0/update dataset: not made.

## Recommendation

Do not reuse the current parquet as if trajectories already existed. For a future approved attempt, first export a small reloadable actor LoRA adapter state, deterministic tensor hash, RNG states, resolved config/route state, and then run exactly the frozen manifest with the requested 1024/4096 settings. Only after all 120 trajectories and their hashes are present should replay materialization and fresh-process zero-request validation proceed.

The addendum result is machine-readable in [results JSON](/root/autodl-tmp/AgentFlow/log/2026-08-28_random30_reuse_addendum_results.json).
