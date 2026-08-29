# vLLM `max_num_seqs=2` one-group capacity smoke — 2026-08-29

## Observed facts

- One frozen MuSiQue group (`id=4`, four rollouts) was run rollout-only with the same Qwen2.5-7B LoRA actor, Qwen-base frozen roles, tools, temperature, context, and three-step/600-second limits as the preceding hierarchical smoke. There was no optimizer, backward pass, checkpoint, or external LLM/reward-judge call.
- The only runtime delta was `actor_rollout_ref.rollout.max_num_seqs: 1 -> 2`; the generated config and VERL command both record `max_num_seqs=2`.
- The run completed `4/4` valid, with `0` retry, rewards `[0,0,0,0]`, all four terminations `max_steps_with_unresolved_plan`, and no CUDA OOM, illegal-memory, prefix-cache, Ray, or vLLM fatal marker.
- vLLM cleanup was safe: both manager sleep and normal completion logged `drained=1`; final GPU process list was empty. Peak memory was `20,097 MiB`, equal to the immediately preceding `max_num_seqs=1` smoke.
- Per-trajectory execution times were 40.25, 35.81, 33.24, and 31.36 seconds (140.66 s total). The matched `max_num_seqs=1` smoke measured 37.74, 34.50, 34.15, and 32.32 seconds (138.71 s total). This one-sample comparison has no useful end-to-end speedup.
- Request timestamps remain effectively serialized through alternating fixed-role and actor calls. In this multi-turn AgentFlow workflow, later actions depend on tool output and step verification, so capacity for two sequences does not make each trajectory's dependency chain parallel.

## Hypotheses

- `max_num_seqs=2` is a stable engine capacity setting for this configuration, but the current rollout server/tool pipeline does not submit enough simultaneously ready model calls to benefit materially on one 4-trajectory, three-step group.
- Increasing `max_num_batched_tokens`, changing async scheduling, or testing a larger independent prompt batch might expose throughput gains, but each would be a separate controlled systems experiment.

## Conclusions

- `max_num_seqs=2` passed the bounded stability check with no additional observed memory cost in this workload.
- It is not yet evidence for a faster end-to-end AgentFlow rollout. Do not claim a throughput improvement from this test and do not increase further on the basis of this single group.

## Recommendation

- Retain the new runner environment knob for a future approved systems-only throughput study. Any next test should compare independent groups with fixed prompt/response lengths and measure request concurrency; it should not be combined with a routing, reward, or training change.

## Evidence

- Results: `log/2026-08-29_vllm_parallel2_onegroup_results.json`
- Local raw logs (untracked): `log/20260829_vllm_parallel2_onegroup_20260829_{train,rollout}.log`.
