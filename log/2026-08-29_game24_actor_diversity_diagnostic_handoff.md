# Game24 actor-only diversity diagnostic — 2026-08-29

## Scope and safety boundary

This is an actor-policy diagnostic only. It ran exactly 24 Qwen actor generations: three frozen prompts (groups 43, 58, and 181), four repetitions at planner temperature 0.0 and four at 0.7 per group. It did not call reward/scoring, Executor, Memory, Verifier, Final, fixed-role models, Doubao, DeepSeek, or any external API. There was no backward pass, optimizer step, checkpoint, GRPO update, HOB update, or full AgentFlow rollout.

## Observed facts

- Base checkpoint: `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`.
- Frozen behavior snapshot: `.../gameof24_planner_temp0_causal_sanity-20260829_20260829_135323_behavior_snapshot.pt`.
- Snapshot SHA256: `654d8a7698ef70a27d9dff15a93c55979acfa3a2532c4fb8d68f0d9ebfaebcc8`.
- Snapshot LoRA hash: `c841011c9800784d97d381e0c5781b299c4d679aa7f8e824ad4773bc61bf8b38` (392 tensors, 20,185,088 parameters).
- The temporary PEFT adapter has `r=8`, `lora_alpha=16`, all-linear target modules. Its adapter config SHA256 is `1aa2085f8cb04dd08ac8887b1e88fbd0c934cce0ade9321b324defbe343e2fbc`; weights SHA256 is `335dc4eaccc19131c82dc96e7996bf59e6468c453440428467a44a9148f40654`.
- The final run used one local vLLM 0.9.2 OpenAI-compatible server. The server exposed `qwen-base` and `qwen-actor`; its log contains 24 `LoRARequest(lora_name='qwen-actor', ..., base_model_name='qwen-base')` requests and one loaded `qwen-actor` adapter.
- All 24 requests returned HTTP 200 and all 24 outputs parsed with the production `agentflow.models.formatters.NextStep` parser. No final-run vLLM error, CUDA error, OOM, illegal memory access, or prefix-cache warning was observed.
- The exact frozen prompt SHA was identical across all four repetitions for each group. The prompt SHA values and full outputs are in the JSON artifact.

Frozen prompt provenance:

| group | selected source file | prompt bytes | prompt SHA256 |
|---|---|---:|---|
| 43 | `rollout_7362b8d9-a0a4-48fa-b39c-adf3b4cbb272.json` | 4120 | `74269bdba605a9a7489e51bd049998dab5cf74117fbf71eb6d9977f0e1cfec6f` |
| 58 | `rollout_0c8f58dd-ec0d-4d21-9bff-04f1680ab61f.json` | 4121 | `980099584b6ebd4b8b6af1882cc0b535772888b6b41c1c166061c60c41904ff3` |
| 181 | `rollout_17d7920d-29c3-49cc-a4ac-70477d6018b2.json` | 4112 | `3b25354a6192f424e1c5686044d50ff54770fde8de2129f45b85160cb6235aa3` |

## Protocol and serving details

The persisted `action_predictor_1_prompt` was used as the exact user-content string for every request; it was not regenerated. The unchanged system prompt was `You are a helpful, creative, and smart assistant.` The server used `qwen-base` as the base and `qwen-actor` as the only LoRA request model. No fixed role was invoked.

Sampling observed in the vLLM log:

| condition | temperature | top_p | top_k | repetition penalty | max tokens | seed |
|---|---:|---:|---:|---:|---:|---|
| T0 | 0.0 | 1.0 | 0 | 1.05 | 1024 | None |
| T0.7 | 0.7 | 1.0 | 20 | 1.05 | 1024 | None |

The server context setting was `max_model_len=8192`, `max_num_seqs=1`, `max_num_batched_tokens=1024`, `gpu_memory_utilization=0.50`, tensor parallel size 1, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The 8192 setting was a diagnostic-only compatibility adjustment: vLLM 0.9.2's OpenAI path counted the serialized chat string length (about 4.2k characters) before tokenization, while the actual frozen prompt token counts were 966–1047 without added special tokens. The prior causal smoke used 4096; this diagnostic did not change model, LoRA, prompt content, or sampling semantics.

## Results

All entries below are `parsed_valid=4/4`; `unique_tool_names=1` and the only tool was `Generalist_Solution_Generator_Tool`.

| group | temperature | raw unique / 4 | byte-identical? | unique sub-goals | unique contexts | semantic action unique / 4 |
|---|---:|---:|---|---:|---:|---:|
| 43 | 0.0 | 2 | no | 2 | 1 | 1 |
| 43 | 0.7 | 4 | no | 3 | 2 | 1 |
| 58 | 0.0 | 1 | yes | 1 | 1 | 1 |
| 58 | 0.7 | 4 | no | 4 | 1 | 1 |
| 181 | 0.0 | 1 | yes | 1 | 1 | 1 |
| 181 | 0.7 | 4 | no | 3 | 1 | 1 |

The deterministic semantic signature is `(lowercased tool_name, sorted content-token signature of sub_goal)`. It applies NFKC/lowercase, direct generic synonym normalization (`generate/create/construct/produce/find`, `calculate/determine`, `evaluate/evaluates/equals`, `expression/expressions`), removes structural words such as “step-by-step”, “valid”, “solution”, “using”, “once”, “arithmetic”, and preserves the puzzle numbers and target. It is not an embedding, LLM judge, reward scorer, or claim of full semantic equivalence. The exact normalization and all raw outputs are recorded in the JSON.

## Interpretation

At temperature 0.0, groups 58 and 181 repeated byte-for-byte. Group 43 had one first-request wording variant and three identical later outputs; its semantic signature was still identical in all four repetitions. This is a small server/model warm-up or deterministic-output variation, not a different tool or planning action.

At temperature 0.7, all three groups produced four distinct raw JSON strings, and some produced distinct sub-goal/context wording. After transparent deterministic normalization, every group still had exactly one semantic action: use `Generalist_Solution_Generator_Tool` to pursue an arithmetic expression for the same four numbers and target 24. Thus T=0.7 created lexical and field-level variation but no evidence of substantively different planner actions, tool selection, or sub-goals on these fixed states.

The result supports the narrow conclusion that this actor prompt/state combination has a degenerate planner action space for Game24. It does not measure final Game24 correctness or reward variance, because the task explicitly excluded Executor, Memory, Verifier, Final, and reward computation. It also does not prove that every Game24 state or every downstream tool state is degenerate.

## Runtime and cleanup evidence

- Total wall time reported by the runner: 181.634 seconds, including vLLM startup/graph capture.
- Per-request latency: mean 2.0004 seconds; range 1.3183–5.4994 seconds. The 5.4994-second request was the first request in group 43; later requests were about 1.3–1.97 seconds.
- GPU monitor samples: 47; peak 17,227 MiB; post-run query: `0, 0 MiB, 0 %`.
- Cleanup called server process-group termination after all synchronous requests. No vLLM/Ray/training/rollout process remained in the post-run process query.
- Final server log SHA256: `1d0fdf00445ddaa25f4dbdf7a4105dbed85e3ab36e4260b3e63e28f955c7e67e`.
- Local evidence (not committed): `/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/server_final/vllm_actor.log` and `gpu.tsv`.

Earlier local-only attempts are retained outside Git as evidence. They exposed vLLM 0.9.2 issues with the direct API (forked CUDA startup, added-token validation, and V1 guided tokenizer type handling); they produced no final diagnostic rows and are not included in the reported measurements. The final server run avoided those issues and is the sole source of the results above.

## Verification and delivery

Final result: `log/2026-08-29_game24_actor_diversity_diagnostic_results.json` (92 KiB; includes all 24 parsed outputs, output hashes, prompt hashes, route metadata, and sampling observations).

Reproduction command:

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python scripts/run_game24_actor_diversity_diagnostic_server_20260829.py
/root/autodl-tmp/conda/envs/agentflow/bin/python scripts/annotate_game24_actor_diversity_diagnostic_20260829.py \
  --vllm-log /root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/server_final/vllm_actor.log \
  --gpu-log /root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/server_final/gpu.tsv
```

The diagnostic scripts compile successfully; `git diff --check` is clean for the changed files. No secret or external API credential was used or written. Raw vLLM logs, GPU traces, model cache, adapter directory, and rollout data remain local and are not Git artifacts.

## Recommendation

Do not infer useful policy diversity from the temperature-0.7 raw variation alone. Before any diversity-oriented algorithm change, inspect whether the actor is being trained to select one generic tool because the fixed prompt always exposes only that tool. A future approved diagnostic could test a small set of states with genuinely different available-tool/action choices, still actor-only, or assess downstream outcomes separately. This handoff itself does not authorize training or any follow-up experiment.
