# AgentFlow 服务器 Codex 交接说明（单卡最小复现与一次 A/B 改进）

更新时间：2026-08-25  
工作目录：`/root/autodl-tmp/AgentFlow`

## 1. 任务边界

本任务的最终交付是一个**可复现、成本受控、证据充分**的小型 AgentFlow/Flow-GRPO 实验：

1. 在单张 RTX 5090 上跑通最小 Flow-GRPO baseline；
2. 基于真实 rollout 失败轨迹，选择并实现一个小而实的改进；
3. 给出 baseline 与改进版的定量 A/B 对比、成本和失败案例。

不是目标：重构 AgentFlow、堆砌多个奖励机制、宣称论文级创新、修改论文/简历表述、对外发布结果、提交 GitHub PR、上传数据或泄露环境变量。任何对外发送、push、发布、租用额外算力或改用更大模型都需要用户单独确认。

## 2. 当前环境与硬约束

- 当前硬件：**1 × RTX 5090，约 32 GiB 显存**；严禁直接使用 8 卡或多卡默认配置。
- Conda 环境：`/root/autodl-tmp/conda/envs/agentflow`。
- 本地模型：`/root/autodl-tmp/models/Qwen2.5-3B-Instruct`。
- 数据、模型、缓存、日志均写入 `/root/autodl-tmp`；不要占用系统盘下载大文件。
- API 变量位于 `/root/.env`：只有在确需固定 Planner/Executor/Verifier 调用时才 `source /root/.env`，禁止打印、复制或写入密钥。
- 单卡首轮配置：`train/config_5090_lora_smoke.yaml`；启动脚本：`train/run_5090_lora_smoke.sh`。
- 深度学习框架已为 RTX 5090 做过适配：Torch 2.7.1 + CUDA 12.8、torchvision 0.22.1、vLLM 0.9.2；不要随意降级或大规模重装。

## 3. 已完成事实与当前阻塞点

以下是已实际验证到的链路，不能遗漏也不能夸大为“训练完成”：

- GPU CUDA kernel、Qwen2.5-3B 加载、LoRA attach、FSDP2、Ray、vLLM 与 AgentFlow 服务端已先后启动；
- 为 Ada/5090 兼容，环境中存在 `flash_attn` 的 torch-only 兼容层，并将 verl 的 `flash_attention_2` 路径改为 `sdpa`；
- 单卡 smoke 在训练进度刚开始时失败于 LoRA 权重同步到 vLLM：

```text
KeyError: layers.0.self_attn.qkv_proj.base_layer.weight
```

失败位置是 verl 的 FSDP → vLLM 参数同步：PEFT 包装出的 `base_layer` 命名与 vLLM Qwen loader 期待的键不一致。该失败说明**尚未完成 LoRA + vLLM 的端到端 rollout/step**；它不是显存不足或 GPU 数量不足。

## 4. 第一优先级：以最小实验定位阻塞

先做下面的分层验证，禁止把多个变量一次性混在一起：

1. 检查当前分支、`git diff`、配置和已有 patch，保留此前单卡兼容修复；
2. 进行一个**不启用 LoRA**的单卡 AgentFlow + Ray + vLLM rollout 冒烟实验；
3. 如果无 LoRA 跑通，则确认 blocker 是 PEFT/FSDP-vLLM loader mapping；
4. 只在能定位到明确兼容层转换逻辑时，尝试最小 mapping/同步修复；不要为了绕过报错切回 7B 全参训练、改多卡或删除 vLLM；
5. 每次尝试限定在 1 个小任务或极小任务集，记录命令、关键日志、显存和退出码。

如果两到三次有依据的最小修复仍无法使 LoRA + vLLM 同步通过，应停止继续盲修，提交清晰的根因与可行替代路径（例如 baseline 先用 no-LoRA rollout，或改用已兼容的训练/rollout 组合）供用户决策。

## 5. Flow-GRPO baseline 与一次改进的路线

当最小 rollout 能完成后：

1. 用小而固定的任务子集做 baseline；先保证 rollout JSON、最终 reward 与轨迹字段可落盘；
2. 按 `AGENTFLOW_V11_IMPROVEMENT_SUMMARY.md` 的 taxonomy 统计失败；
3. 选择**一个**高频且可归因的问题作为改进目标。优先顺序：Memory 跨任务污染验证、`model_engine` 配置一致性、重复工具/子目标、Verifier 停止错误、Executor/Tool 噪声归因；
4. 改动仅覆盖必要文件，并补充最小单测或可重复的 smoke 命令；
5. 用同一模型、同一数据子集、同一 rollout 数、同一 max steps 和同一评测规则做 A/B；
6. 同时报成功率/奖励、失败构成、平均步骤/延迟或 token 成本。若改进不显著或变差，也如实记录。

不允许先设计“复杂奖励”再寻找好看的指标。若基础 rollout 未通，不能启动大规模 GRPO 训练。

## 6. 成本控制与运行纪律

- 固定使用 Qwen2.5-3B；不下载/加载 7B 以上模型，除非用户明确批准；
- 固定使用单张 5090；只有单卡完整 baseline 跑通后，才讨论多卡；
- 首轮实验任务数与 rollout 数保持极小，先验证完整数据流；
- 使用 `tmux` 运行可能持续的任务，确保 SSH 断开不终止任务；
- 每次执行前后记录 `nvidia-smi`、磁盘占用和日志路径；
- 遇到明显无进展的下载、训练或修复，不要无限重试；定位原因后给出证据与下一步；
- 完成一轮验证后停掉不需要的 Ray/vLLM/AgentFlow 进程，避免后台占卡。

## 7. 交付与汇报格式

每个阶段都使用以下结构汇报，避免“看起来跑了”但没有证据：

```markdown
### 阶段名称
- 目的：
- 修改文件：
- 实际命令：
- 结果：通过 / 失败
- 关键证据：日志路径、退出码、指标、显存
- 根因或结论：
- 下一步：
```

最终输出至少包括：

- 一条可复现的 baseline 命令；
- 一个明确、最小的改进 commit（仅本地，除非用户批准提交）；
- A/B 实验配置、原始日志/结果路径与简表；
- 失败案例和技术边界；
- 对“是否已完成 Flow-GRPO + LoRA + vLLM 端到端训练”的真实状态说明。

## 8. 参考材料

- `AGENTFLOW_V11_IMPROVEMENT_SUMMARY.md`：从 v11 清单人工整理的可执行候选、证据标准与阅读路径；
- `AgentFlow_潜在改进与探索清单_v11.docx`：原始清单，仅供追溯；
- 当前仓库源代码、`train/config_5090_lora_smoke.yaml` 与历史日志：运行现状的唯一事实来源。
