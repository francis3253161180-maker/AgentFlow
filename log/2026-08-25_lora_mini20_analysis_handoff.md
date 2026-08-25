# LoRA mini20 baseline：实验数据审查交接

分支：`experiment/flow-grpo-3b-lora`  
主实验运行：`20260825_231408`  
模型：Qwen2.5-3B-Instruct + LoRA（rank 8, alpha 16）  
数据：`data/train/flowgrpo_mini_20_seed20260825.parquet`（mathhard 10 + nq 10）

## 审查范围

请以严格的 LLM/Agent 算法审稿人视角审查本次原始日志与 rollout 数据：

1. reward 的 0/1 差异是否在 GRPO group 内形成了有效的非零 advantage；
2. `actor/pg_loss` 一直打印为 `0.0` 是否符合当前实现与首轮 old/new policy 相同的预期，或暴露了训练信号问题；
3. 非零 `grad_norm` 是否足以支持发生了非零 policy gradient 的结论；
4. 只建议一个成本最低、信息增益最大的下一步验证；不要建议大规模重构或改写 reward/算法。

## Observed facts

- 主训练日志显示 10 个训练 rollout batch 均完成；每个 batch 的汇总为 4/4 completed、4 valid、0 retries。
- 训练后自动验证完成 30/30 rollouts，30 valid。
- 已记录 reward 的 batch 统计包含 `min=0.0`、`max=1.0`；例如 step 7 的 reward mean 为 `0.75`，min/max 为 `0.0/1.0`。
- step 7 记录了非零 advantage：mean `-0.034912109375`、min `-0.70703125`、max `0.70703125`。
- 同一 step 7 的 `actor/grad_norm=0.23865246772766113`；其余已打印的 step 1–9 metric 行为 `0.0`。
- 已打印的 step 1–9 `actor/pg_loss` 均为 `0.0`。因此，日志支持“至少一次非零 advantage 与非零梯度范数”，但**不支持**“已观察到非零打印 pg_loss”的结论。
- 日志记录了 `timing_s/old_log_prob`、`timing_s/adv` 和 `timing_s/update_actor`，说明相应阶段被调用；最终 checkpoint 路径为 `global_step_10`。
- GPU 的训练期已记录最大 allocated memory 为 `21.608510971069336 GB`、reserved memory 为 `26.28125 GB`，未出现 CUDA OOM。
- 最终 checkpoint 未写成：数据盘在 checkpoint 保存时已满，`torch.save` 报 `PytorchStreamWriter failed writing file`。这发生在训练与验证均完成之后；不影响已落盘的训练/验证 rollout 证据。
- 随后启动的 `20260825_235120` 运行在用户要求优先分析首轮数据后被主动停止，只包含启动与极少量 rollout，不应用于训练结论。

## Hypotheses（未验证）

- 在初始 old/new policy 一致时，按 token/group 聚合后的 surrogate pg loss 标量可能为零，而梯度仍可非零；需直接检查实现与参数差异才能确认。
- step 7 的非零 advantage 可能来自同一 GRPO group 内的 0/1 reward 差异，但应通过 `rollout_data` 中同一训练 index 的成对 rollout 复核，不能仅依据 batch min/max 推断。
- 当前多数 batch 的 advantage/grad_norm 为零，可能是 group 内 reward 相同，也可能与 trace 过滤/聚合方式有关；尚未区分。

## Conclusions（当前可成立）

- LoRA mini20 已证明单卡工程路径可完成 rollout、reward、old log prob、advantage、actor update 阶段及验证，且未触发显存错误。
- 当前证据不足以宣称“稳定、非零 policy-gradient baseline 成功”，因为打印的 `actor/pg_loss` 仍为零且非零 gradient 仅在已记录的一个 step 中出现。
- checkpoint 失败是存储容量问题，不应被表述为算法失败；checkpoint 本身也不是分析训练信号的前置条件。

## 原始产物

- 主训练日志：`log/20260825_231408_lora_mini20_train.log`
- 主 rollout 日志：`log/20260825_231408_lora_mini20_rollout.log`
- 主 rollout 数据：`rollout_data/46.38.243.197/qwen25-3b-lora-mini20-seed20260825_20260825-231521/`
- 已停止重跑日志：`log/20260825_235120_lora_mini20_train.log`、`log/20260825_235120_lora_mini20_rollout.log`
- 更早 endpoint-staleness 失败日志：`log/20260825_225127_lora_mini20_train.log`

为满足安全边界，两个 rollout 日志中原本由 AgentOps 打印的 API key 已替换为 `[REDACTED]`；未提交 checkpoint、模型权重、缓存、Conda 环境或 `.env`。
