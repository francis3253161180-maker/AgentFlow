# AgentFlow：实验执行与审计指令

本文件是本仓库中服务器 Codex 的持久项目指令。采用“网页审查、服务器执行、GitHub 交接”的实验闭环；所有研究结论必须由可复现的配置和原始证据支撑。

## 职责边界

### 网页版 ChatGPT

负责实验设计、指标定义、改进方向与审计结论：

- 设计实验、对照组、预算和成功标准。
- 从 GitHub 读取审计报告与关键原始数据，复核证据是否支持结论。
- 决定下一轮的唯一优先实验，并向服务器 Codex 下达范围明确的指令。

### 服务器 Codex

负责执行实验与保全证据：

- 严格执行已批准的实验；不得为得到预期结果擅自改变算法、奖励定义、数据集或关键超参数。
- 收集完整 train/rollout 日志，汇总完整 `rollout_data`。
- 编写实验审计报告并提交、推送到实验分支。
- 在报告中严格区分 **Observed facts**、**Hypotheses** 和 **Conclusions**。假设不得写成结论。

### 人工负责人

负责资源授权、最终研究判断与对外表述。

## 每轮工作流

```text
网页版 ChatGPT：实验设计 / 指标定义 / 改进方向 / 审计结论
        ↓
服务器 Codex：执行 → 完整日志与 rollout_data → 审计报告 → Git 提交与推送
        ↓
网页版 ChatGPT：从 GitHub 复核 → 决定下一轮 → 下达新指令
```

Remote Desktop Commander 仅用于向服务器 Codex 下达任务、检查进程和 GPU。GitHub 是稳定的实验审计与数据阅读入口。

## 实验执行要求

- 使用描述性的实验分支；不要直接在 `main` 累积实验迭代。
- 每一阶段使用可读的 commit，便于按 diff 审查。
- 保留失败的日志与 rollout 数据，除非人工负责人明确要求删除。
- 环境兼容修复（包括 site-packages 补丁）必须整理为项目内补丁或可执行说明，并明确标注为环境修复，不得表述为算法贡献。
- 完成后停止实验进程，记录异常，不进行未经批准的追加运行或算法改动。

## 固定审计报告

每轮实验结束，创建：

```text
log/<date>_<experiment>_analysis_handoff.md
```

报告至少包含：

- 实验 commit SHA、配置路径、数据集路径和 seed。
- completed rollouts、valid rollouts、dropped samples。
- 每个 GRPO group 的 reward 分布与组内 reward variance。
- advantage 的 min/max/mean/std。
- `pg_loss`、`grad_norm`、entropy、`old_log_prob`。
- `optimizer.step` / `update_actor` 是否成功，以及 `global_step`。
- GPU peak memory。
- 异常、traceback、OOM 与 rollout failure taxonomy。
- 关键 `rollout_data` 路径。
- **Observed facts**、**Hypotheses**、**Conclusions** 和仅供网页审查者决策的下一步建议。

报告必须说明指标缺失是“未记录”还是“数值为零”，不得用成功完成的工程步骤推断算法有效性。

## GitHub 交接

提交本轮真实修改与可审查证据：源码、相关配置、runner、兼容性补丁、小型实验数据、完整日志、完整 `rollout_data` 和审计报告。

```bash
git add log rollout_data train/<relevant-config> <relevant-source-or-runner>
git commit -m "Add <experiment> analysis and rollout evidence"
git push origin experiment/flow-grpo-3b-lora
```

严禁提交模型权重、checkpoint、HF/Ray/vLLM 缓存、Conda 环境、`.env` 或 API key。提交前检查日志与配置，确保不含任何密钥。

## 研究节奏

后续实验按以下证据链推进：

```text
baseline → failure analysis → algorithm improvement → A/B → ablation → resume evidence
```

在 baseline 尚未证明稳定训练信号前，不提前宣称完成算法复现或算法改进。
