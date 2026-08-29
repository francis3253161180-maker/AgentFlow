# Rollout concurrency smoke handoff — 2026-08-29

Commit under test: `4f4d0b1` plus the observability-only changes in this handoff.
This is a systems-only, rollout-only diagnostic: no optimizer step, backward pass,
checkpoint, external judge, or agent-semantic change occurred.

## Observed facts

- The frozen single MuSiQue Barcelona prompt (source index 259) was run with
  Qwen2.5-7B, the existing LoRA actor, Qwen base fixed roles, hierarchical
  planning, `n=4`, `temperature=0.7`, three tool steps, and a 600-second agent
  timeout.  The only scheduler deltas were `N_WORKERS` and vLLM
  `max_num_seqs`.
- `N_WORKERS=1, max_num_seqs=2` had a previously recorded group wall time of
  **140.66 s** (1.706 rollouts/min).  It did not create logical overlap.
- `N_WORKERS=2, max_num_seqs=2` completed 4/4 valid JSON rollouts, with two
  worker-owned rollouts each.  Persisted task intervals show a maximum of two
  concurrent rollouts; the vLLM log-derived in-flight-request proxy also peaks
  at two.  Group wall time was **78.387 s** (3.062 rollouts/min), a **1.794x**
  wall-clock speedup versus the one-worker baseline.
- After the explicit authorization to test four-way concurrency, the identical
  frozen sample was rerun with `N_WORKERS=4, max_num_seqs=4`.  All four
  rollouts started within 39 ms, completed as four valid distinct JSON files,
  and were uniquely owned by Workers 0–3.  Persisted task intervals peak at
  **four** simultaneous rollouts.  The vLLM request-start/HTTP-200 log proxy
  also peaks at **four** in-flight requests (44 completed requests; proxy is
  labeled because the server log has no request correlation id).
- The four-worker group wall time was **59.267 s** (4.050 rollouts/min):
  **2.373x** faster than one worker and **1.323x** faster than two workers.
  Individual four-worker solve durations were 54.00, 54.31, 58.42, and 59.26 s;
  their growth versus two workers confirms GPU/serving contention rather than
  a linear fourfold acceleration.
- Both two- and four-worker runs had reward vector `[0.0, 0.0, 0.0, 0.0]`.
  That outcome is unchanged and is not interpreted as a learning result.
- Peak recorded GPU memory was **20,195 MiB / 32,607 MiB** for both runs.
  No CUDA OOM, illegal-memory-access, prefix-cache-reset failure, Ray/vLLM
  worker death, duplicate ownership, or malformed rollout JSON was found.
- Normal-complete cleanup recorded `outstanding=0`, `drained=1`, prefix-cache
  reset before sleep, and 0.694 s (two workers) / 0.755 s (four workers) final
  cleanup.  Search telemetry records 20 cache hits and zero retries, 429s, or
  external LLM calls in each run.
- The current runner leaves its known `AgentFlow-Worker-*` children reparented
  to PID 1 after the parent returns.  For each completed smoke their exact,
  known PIDs were terminated with SIGTERM (no SIGKILL required), followed by
  `ray stop --force`.  GPU memory is now 0 MiB and no Ray/vLLM/AgentFlow worker
  remains.  Two older PID-1 multiprocessing helper processes (270205/270206)
  were not attributable to this smoke and were deliberately not killed.

## Hypotheses

- The near-constant memory peak is consistent with the 7B model and configured
  vLLM reservation dominating this short workload; the extra concurrent request
  KV/cache footprint fits in the remaining memory.  This is a bounded-sample
  observation, not a general capacity proof for longer contexts or more workers.
- The sublinear 2→4 gain is plausibly caused by shared single-GPU decoding and
  simultaneous fixed-role calls.  The collected timings establish the effect
  but do not separately attribute every source of contention.

## Conclusions

- Four logical AgentFlow rollouts and four vLLM requests did overlap on the
  single RTX 5090, and `N_WORKERS=4, max_num_seqs=4` is feasible for this
  1536/512-token, one-question bounded smoke without increasing observed peak
  GPU memory beyond 20,195 MiB.
- It improves the measured group wall time from 140.66 s (one worker) to 59.27
  s (four workers), a 2.37x speedup, but it is not a 4x speedup.
- The orphan-worker behavior means four workers should **not** become the
  unattended default until the runner process lifecycle is fixed or the launch
  wrapper reliably joins/terminates worker children.  It is safe for explicitly
  monitored bounded runs with the documented cleanup procedure.

## Recommendation

- Keep `N_WORKERS=4, max_num_seqs=4` as an evidence-backed, monitored smoke or
  rollout-only setting for similarly bounded contexts.  Before a longer
  benchmark/training job, first repair the parent/worker join lifecycle, then
  rerun a bounded lifecycle regression; do not extrapolate this memory result
  to higher token limits.
- No training or follow-up rollout was started after this measurement.

## Evidence and verification

- Two-worker result: `log/2026-08-29_two_rollout_concurrency_results.json`.
- Four-worker result: `log/2026-08-29_four_rollout_concurrency_results.json`.
- Raw local-only logs: `log/20260829_{two,four}_rollout_concurrency_20260829_{train,rollout}.log`.
- Raw local-only rollout directories:
  `rollout_data/46.38.243.197/{two,four}-rollout-concurrency-musique-group4-20260829_*`.
- Code-only checks recorded after the run: `py_compile`, `bash -n`, JSON parse,
  focused unit tests, `git diff --check`, and a scoped secret scan.
