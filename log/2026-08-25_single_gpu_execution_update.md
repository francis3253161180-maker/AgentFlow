# 单卡 AgentFlow 冒烟执行更新

日期：2026-08-25  
环境：单张 RTX 5090（约 32 GiB），`/root/autodl-tmp/conda/envs/agentflow`

## 执行结果

- no-LoRA rollout 使用 4 条 smoke 样本完成：4/4 成功、0 次重试、4 个 rollout 有效。
- 4 条样本的 reward 均为 `1.0`，答案评分链路正常。
- 训练器已完成 rollout 收集并进入 optimizer step，但在单卡显存峰值处失败：约 31.3 GiB 已占用，额外申请 86 MiB 时发生 CUDA OOM。
- 已知 LoRA/FSDP-vLLM 隔离日志确认原错误：
  `KeyError: layers.0.self_attn.qkv_proj.base_layer.weight`。

## 代码与配置更新

- 修复 `train/train_agent.py`，使 Hydra 风格 positional overrides 不再被静默丢弃。
- 修复本地模型路径到 verl/vLLM 服务模型名的映射。
- 修复 rollout 中固定模型 span 的空 token id 导致整条样本无效的问题，并保留 reward span。
- 压缩 planner 的本地 action-predictor 上下文，并正确传递 `max_tokens`。
- 默认关闭外部 GPT scorer，smoke 环境使用本地答案归一化评分；如需外部 scorer，可设置 `AGENTFLOW_USE_LLM_SCORER=1`。
- 为 `Memory` 增加 `reset()`，每个 Solver query 开始时清理跨任务状态。
- 单卡 smoke 配置启用参数/优化器 offload，并将 vLLM 显存预算从 `0.36` 调整为 `0.24`。
- 在环境中补充安装 `socksio`，以支持 SOCKS 代理下的 HTTP 客户端。

## 验证

- Memory reset smoke test：通过。
- 相关 Python 文件编译检查：通过。
- `git diff --check`：通过。

## 提交

- 本次更新提交：`e865d5e Fix single-GPU AgentFlow smoke compatibility`
- 前一提交：`5ffccd0 Add single-GPU smoke training setup and compatibility fixes`
- 本文档与上述代码一并推送到远端 `main`。

## 剩余限制

当前 32 GiB 单卡可以稳定完成 rollout，但 3B 模型、vLLM 与训练 optimizer 共存时仍无法完成一次训练更新。正式训练需要降低模型/训练显存占用、进一步拆分 rollout 与 trainer，或使用更大显存/多卡资源。

## DeepSeek 模型更新

- DeepSeek 模型已切换为 `deepseek-v4-flash`。
- 通过 OpenAI-compatible Chat Completions 请求的 `extra_body` 显式设置 `thinking.type` 为 `disabled`。
- 未设置或发送 `reasoning_effort`。
- 已更新 smoke 配置、引擎工厂识别、引擎单元测试及相关文档。
