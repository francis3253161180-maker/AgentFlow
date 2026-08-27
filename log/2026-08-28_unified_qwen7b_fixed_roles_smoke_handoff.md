# Unified local Qwen fixed-role architecture smoke handoff

## Scope and outcome

This was an engineering smoke only. No formal 60-prompt training, HOB, sweep, validation, checkpoint, DeepSeek, Doubao, GPT, or other external judge call was started. The 7B path was attempted first, as required. It failed at actor/FSDP2 initialization on the single RTX 5090 32 GB. The same unified role design was then checked with 3B and completed safely.

The tracked aggregate is in `log/2026-08-28_unified_qwen7b_fixed_roles_smoke_results.json`. The large raw logs and rollout JSON remain local and are intentionally not tracked.

## Architecture audit

`agentflow/agentflow/solver.py::construct_solver` is the role assembly point. It receives `[planner_main, planner_fixed, verifier, executor]`, creates the `Initializer`, `Planner`, `Verifier`, and `Executor`, and passes the local vLLM endpoint to each local role. The new opt-in contract is `['trainable', 'frozen', 'frozen', 'frozen']`, with `TOOL_ENGINE=['frozen']`.

`train/train_agent.py`/VERL instantiate the actor through PyTorch FSDP2 and apply the existing PEFT LoRA to the actor. The rollout path is a VERL/vLLM engine. The fixed roles use the same local Qwen endpoint and are marked base-only/frozen; they do not create separate fixed-agent model processes. Role behavior remains prompt/role logic, not separate paid providers or model copies.

KL is disabled in this smoke, so no separate `RefPolicy` is created. When reference semantics are needed by the existing actor path, the intended semantic is the same base with the LoRA adapter disabled; this task did not enable KL merely to manufacture a reference model.

One pinned-stack limitation is material: VERL 0.5.0/vLLM 0.9.2's async OpenAI-compatible request path does not provide a per-request LoRA selector. Therefore the smoke proves the actor's LoRA wiring and a shared local inference endpoint, but it does not prove that direct AgentFlow planner_main HTTP calls select the synchronized actor adapter. This must be resolved or explicitly accepted before a formal unified baseline.

## Model source

Qwen2.5-7B-Instruct was downloaded first through ModelScope using `Qwen/Qwen2.5-7B-Instruct` into `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`. The local four safetensors shards and SHA256 values are recorded in the JSON artifact. The download command did not pin a ModelScope revision, so no unverified revision claim is made. The existing local Qwen2.5-3B-Instruct model was reused for fallback; no model artifacts were deleted.

## 7B failure evidence

Two 7B attempts were made before fallback:

* `20260827_233700`, vLLM GPU utilization 0.14;
* `20260827_233813`, vLLM GPU utilization 0.10.

Both failed before rollout during actor FSDP2 initialization with the same CUDA OOM: an attempted 2.03 GiB allocation with 1.17 GiB free, 30.18 GiB in use, and 28.58 GiB allocated by PyTorch. Lowering vLLM reservation did not change this actor-init failure, so the 7B unified design is not feasible on this single 32 GB card under the current FSDP2+vLLM layout and conservative smoke settings.

Evidence is preserved in:

* `log/unified-qwen7b-fixed-roles-smoke-20260828_20260827_233700_train.log`
* `log/unified-qwen7b-fixed-roles-smoke-20260828_20260827_233813_train.log`

## 3B fallback smoke

The final successful run used `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`, vLLM utilization 0.24, TP=1, max model length 2048, max sequences 1, max batched tokens 1024, and a smoke-only response cap of 64 tokens. The cap was propagated through the local ChatVLLM factory because the AgentFlow role context otherwise exceeded the intentionally small vLLM context budget; formal experiment configuration was not changed.

The four-prompt, `n=2` run produced 8 raw rollout files, all valid, with zero retries and no errors. The role log reports:

`planner_main=trainable_actor_lora planner_fixed=frozen_base_no_lora verifier=local_base_no_lora executor=local_base_no_lora tools=local_base_no_lora`

The run loaded one local vLLM engine path and did not spawn a separate fixed-model process. Six role/tool client objects are visible in the role construction logs, but they are clients to the one local endpoint, not six model instances. The actor log reports PEFT LoRA application and a 3.10B-parameter actor.

All eight rewards were 0.0. With no mixed reward group, theoretical group-normalized advantages were all zero; both logged steps consequently had `actor/pg_loss=0.0`, `actor/grad_norm=0.0`, and `critic/advantages/mean=0.0`. This is an uninformative all-zero sample, not evidence that the LoRA backward path is broken, and the smoke does not claim a nonzero update was observed. No extra prompts were run after seeing this outcome because the requested architecture smoke was limited to 2–4 prompts.

The local scorer was explicitly run with external judge disabled. Its eight events were conservative fallback events (`reason=conflicting_numbers`, `error=unavailable`), not DeepSeek calls. This keeps the smoke within the zero-external-LLM requirement; it also means this run does not validate the paid judge route.

## Cleanup and resource evidence

The final 3B train log contains two normal-completion markers. Each reports `drained=True`, `outstanding_before=0`, `abort_count=0`, then `sleep_started=True`; cleanup duration was about 0.49–0.51 seconds. There were no prefix-cache reset failures, “blocks are not freed yet” messages, CUDA illegal memory accesses, deadlocks, Ray worker deaths, or CUDA OOMs in the successful run.

The local GPU monitor recorded a peak of 20,916 MiB used out of 32,607 MiB. Actor metrics recorded approximately 20.86–20.87 GiB allocated and 24.11–24.40 GiB reserved. After cleanup, `nvidia-smi` reported 2 MiB used and no matching Ray/vLLM/training process. `trainer.save_freq=0`; no checkpoint was written. Validation was disabled (`trainer.test_freq=0`, final validation metrics `None`).

There was one non-fatal environment warning because the port cleanup helper could not find `lsof`; the control server and the safe cleanup path still completed. It is recorded for follow-up, but it did not cause a model or lifecycle failure.

## Code changes

The isolated changes:

* forward the local endpoint and response cap into initializer tools, planner/verifier fixed engines, executor, and local ChatVLLM;
* add the opt-in unified local role contract and external-provider safety guard;
* retain only planner_main as the trainable actor-LoRA role while fixed roles are explicitly frozen/base-only;
* add four small wiring/guard contract tests;
* add the reproducible smoke launcher with no-validation/no-checkpoint and external-LLM-disabled safeguards.

No Flow-GRPO algorithm, reward range, model weights, optimizer hyperparameters, or formal training configuration was changed.

## Recommendation

Do not launch the formal GameOf24 60-prompt baseline or HOB from this smoke. The 7B unified architecture should be rejected for this single 32 GB GPU unless the memory layout is materially redesigned. The 3B unified endpoint is operationally feasible and cleanup-safe, but a controlled follow-up is required to verify how planner_main's synchronized LoRA is selected on this pinned async vLLM API. Only after that adapter-selection question is answered should a formal retraining decision be made.
