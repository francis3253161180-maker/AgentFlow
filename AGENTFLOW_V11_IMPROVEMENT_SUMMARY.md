# AgentFlow：潜在改进与探索清单（可执行摘要）

> 来源：`AgentFlow_潜在改进与探索清单_v11.docx`（2026-08-22）。
>
> 本文是供工程执行与复盘使用的人工整理版，不替代原始文档。**源代码阅读能确认的事实**、**尚待实验验证的假设**和**最终实验结果**必须严格区分；没有 rollout 数据支撑的想法不能写成创新或结论。

## 目标与选择原则

目标不是把 AgentFlow 重做一遍，也不是堆叠强化学习术语，而是：

1. 在受控成本下跑通一个最小的 Flow-GRPO 基线；
2. 从真实失败轨迹中发现一个高频、可定位的问题；
3. 仅实现一个最小改进；
4. 用相同任务集、模型、预算与评测口径做 A/B 比较，报告收益和代价。

改进候选的优先级由“失败出现频率 × 对最终奖励/成本的影响 × 修改范围可控性”决定，而不是由概念听起来是否新颖决定。

## P0：先验证的代码/配置一致性问题

### 1. Memory 是否跨任务污染

- **已读源码线索：**`Solver.solve()` 未见显式 `memory.reset()`；同一个 `Solver` 复用同一 `Memory`。训练路径中同一 worker 还可能复用同一 `training_agent/Solver`。
- **待验证：**连续使用同一 `Solver` 跑两个不同 query，检查第二个任务的 memory、Planner prompt 与 action history 是否残留前一个任务内容；再检查训练 worker 的复用路径。
- **若证实：**这是一个范围小、工程价值清晰的可靠性修复，可形成独立 PR；不要在未证实前把它写成已修复 bug。

### 2. `model_engine` 默认值与 `construct_solver()` 的接口一致性

- **已读源码线索：**`train/rollout.py` 的默认/注释配置表现为 3 个 engine（`planner_main, planner_fixed, executor`），而 `construct_solver()` 会按 4 个 engine（含 `verifier`）读取。
- **待验证：**先检查当前 YAML 的实际值，再用最小配置测试默认值是否触发索引错误；同步核对 README 与样例配置。
- **若证实：**优先修正默认配置与文档，使 main 分支的最小路径可运行；这是低风险、可上游化的改进候选。

## P1：必须先做轨迹统计再决定是否实现的候选

### 3. 重复工具调用 / 重复子目标

- **假设：**Planner prompt 虽要求避免冗余，但策略可能仍重复选择相同工具或相同子目标。
- **先记录：**重复 `tool_name`、重复 `sub_goal`、相同 query/command、成功/失败轨迹差异、token 和耗时。
- **可选改进：**只有重复确实高频且与失败或成本显著相关时，才尝试重复惩罚、轨迹效率项或去重约束。

### 4. 稀疏二元奖励与信用分配

- **已读源码线索：**当前任务最终得分主要是答案正确 `1.0`、错误 `0.0`，没有现成的 step-level reward。
- **先记录：**同一 reward 下不同轨迹的长度、工具错误、Verifier 状态和有效 action，确认稀疏奖励是否真是主要瓶颈。
- **可选改进：**在不改变最终正确性主指标的前提下，探索与证据/效率有关的轻量 shaping；不能只报告 shaping 后的单一有利指标。

### 5. Verifier 的过早或过晚停止

- **先标注：**`premature_stop`、`late_stop`、`correct_stop`，并同时记录 semantic STOP 和 `max_steps/max_time` 等硬截止原因。
- **可选改进：**如果失败主要来自终止决策，再考虑 confidence、动态步数预算或 trajectory filtering；否则不应优先改 Verifier。

### 6. Executor / Tool 错误污染 Planner 的 RL 信号

- **问题：**Planner 选择 action 后，Executor 仍需生成/解析 tool command；工具本身也可能失败。最终 reward 为零不必然意味着 Planner 选错。
- **先分类：**Planner 选择、Executor 生成/解析、Tool 执行、Verifier、最终答案抽取、超时。
- **可选改进：**只有当非 Planner 失败占比高时，才考虑剔除/降权无效轨迹或修复相应接口，避免把 Executor 噪声错误归因给 Planner policy。

## 必要的最小日志字段

每次 baseline 或 A/B 运行至少保存下列字段，优先复用现有 rollout JSON，不要重构日志系统：

- `task / idx / rollout_id / groundtruth / answer_extracted / reward`；
- 每步 Planner 的 `context / sub_goal / tool_name`；
- Executor 的 command、解析路径、tool result 与 exception；
- Verifier 的 `STOP/CONTINUE`、`step_count`、`termination_reason`；
- 重复工具/子目标计数、token 与耗时；
- failure taxonomy：`planner / executor / tool / verifier / final_extraction / timeout`。

## 其他可靠性检查（不作为首轮功能开发）

- `ToolCommand` prompt、结构化 response 与 JSON/regex fallback 的一致性和频率；
- `split_commands()`、多命令执行与工具输入安全边界；
- timeout/cancel 是否留下后台 thread 或造成后续 rollout 不稳定；
- `query_cache_dir` 是否会在不同任务间覆写或复用；
- Initializer 的工具名映射、demo command 与实际 `available_tools` 是否一致；
- 固定 Query Analysis / Final Synthesizer、以及仅训练 `tool_name` 对信用分配的限制。

## 推荐执行顺序

1. 先解决“能否完成最小 rollout”的运行时阻塞；
2. 用小任务集获得可解析的 baseline 轨迹；
3. 完成 P0 的两项一致性/污染验证；
4. 对失败类型与成本做统计；
5. 只选择一个由数据支持的候选改进；
6. 以同模型、同数据、同采样预算、同随机种子策略完成 baseline 与改进版 A/B；
7. 汇报绝对指标、相对变化、运行成本、失败案例和仍未解决的限制。

## 当前源码阅读路径

优先按调用链阅读，不要一开始陷入分布式训练细节：

1. `agentflow/solver.py`：系统调度入口；
2. `agentflow/models/planner.py`：固定 Query Analysis、可训练 Action Prediction、固定 Final Synthesis；
3. `agentflow/models/executor.py`：高层 action 到 tool command/observation；
4. `memory.py`、Verifier、Initializer；
5. `train/rollout.py` 与训练配置。
