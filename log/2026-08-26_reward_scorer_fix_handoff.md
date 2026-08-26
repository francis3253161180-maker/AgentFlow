# Deterministic reward scorer fix handoff

实验分支：`experiment/flow-grpo-3b-lora`  
基线 HEAD：`7a7fb68`  
实现 commit：`f7cfd2b` (`Fix deterministic reward scorer and add regression`)  
配置：`train/config_5090_lora_mini20.yaml`  
数据：`data/train/flowgrpo_mini_20_seed20260825.parquet`（NQ 10、mathhard 10）  
模型：Qwen2.5-3B-Instruct + LoRA rank 8；本轮未修改模型、GPU、rollout.n、Flow-GRPO 或 reward 数值范围。

完整逐条前后结果见 [`2026-08-26_reward_scorer_fix_results.json`](2026-08-26_reward_scorer_fix_results.json)。

## Observed facts

- 已阅读并确认生产路径：[`train/rollout.py`](../train/rollout.py) 调用 `compute_score(question, groundtruth, answer_extracted)`；本次也修复了 [`train/utils.py`](../train/utils.py) 自带 `eval()` 的参数交换，使其使用相同签名。该交换不是主实验的主要根因，因为生产路径原本已使用正确顺序。
- 主实验原始 `train/` 有 40 条 rollout、20 个唯一样本；validation 为此前日志记录的 30/30 completed、30 valid，本轮没有重新运行 validation。
- 原始训练日志仍显示 10 个训练 batch、每批 4/4 completed、4 valid、无新增 rollout failure；没有 dropped sample 的新变化。
- 原始日志可解析到 global step 1–9：`actor/pg_loss` 均为 `0.0`；step 7 的 advantage 为 mean `-0.034912109375`、min `-0.70703125`、max `0.70703125`，`grad_norm=0.23865246772766113`；其余已记录 step 的 advantage/grad_norm 为零。
- 原始日志记录的最大 GPU allocated memory 约 `21.61 GB`；`old_log_prob` 只有 timing，没有独立数值汇总；`update_actor` 有 timing，但没有单独的 `optimizer.step succeeded` 字段。
- 原始训练最后在 checkpoint 写入阶段失败，约写入 `9.43 GB` 时出现 `PytorchStreamWriter ... file write failed`；本轮未重启训练或修复 checkpoint。

## Code changes

只修改 reward scorer、离线审计脚本和测试：

1. `train/utils.py::deterministic_fallback_score` 现在按以下顺序工作：
   - 优先提取最后一个 `<answer>...</answer>`，其次提取最后一个 `Answer` / `Final Answer` / `Conclusion` / `Result` 结构。
   - `Yes/No` 使用开头或显式答案结构判定，并拒绝 `not yes`、`not no` 等否定形式。
   - 日期解析为完整 `(year, month, day)` 三元组；不再以共享年份判定正确。
   - 纯数字只接受简短答案中的完整值或明确的 `answer/result/value/there are` 结构；删除任意共享数字 token 逻辑。
   - 文本答案使用规范化后的 token-boundary phrase matching；仅对 groundtruth 自带 possessive 的短实体允许有限有序 token gap，并加入前置/后置否定保护。
   - 保留原有紧凑 LaTeX 规范化，并用受限 deterministic SymPy 表达式等价支持分数、根式、简单等式 RHS，例如 `1 + sqrt(2)` 与 `sqrt(2) + 1`。
   - `eval()` 改为 `compute_score(question_str, groundtruth_str, answer_extracted_str)`，最终输出仍严格为 `1.0` 或 `0.0`。
2. `scripts/audit_reward_20260826.py` 同时复现 legacy scorer 与 fixed scorer，输出逐条前后 reward、source、TP/TN/FP/FN、新增 positive 以及原始训练 metric。
3. 新增 [`test/test_reward_scorer.py`](../test/test_reward_scorer.py)，覆盖自然语言实体、否定、Yes/No、日期、数字、分数、根式、等式 RHS 和 `eval()` 参数顺序。

## Regression results

测试命令：

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python -m unittest discover \
  -s test -p 'test_reward_scorer.py' -v

/root/autodl-tmp/conda/envs/agentflow/bin/python scripts/audit_reward_20260826.py \
  --run-root rollout_data/46.38.243.197/qwen25-3b-lora-mini20-seed20260825_20260825-231521/Qwen2.5-3B-Instruct_20260825-231521 \
  --dataset data/train/flowgrpo_mini_20_seed20260825.parquet \
  --train-log log/20260825_231408_lora_mini20_train.log \
  --output log/2026-08-26_reward_scorer_fix_results.json
```

结果：8/8 unit tests passed；40/40 offline records processed；legacy scorer 与保存 reward 仍为 40/40 一致。

| source | 维度 | 修复前 | 修复后 |
|---|---|---:|---:|
| 全部 | TP | 16 | 34 |
| 全部 | TN | 3 | 6 |
| 全部 | FP | 3 | 0 |
| 全部 | FN | 18 | 0 |
| 全部 | reward=1 / reward=0 | 19 / 21 | 34 / 6 |
| NQ | TP/TN/FP/FN | 3/2/3/12 | 15/5/0/0 |
| mathhard | TP/TN/FP/FN | 13/1/0/6 | 19/1/0/0 |

按 `FN/(TP+FN)` 与 `FP/(TN+FP)` 计算：

- 全部：FN rate `52.9% → 0.0%`；FP rate `50.0% → 0.0%`。
- NQ：FN rate `80.0% → 0.0%`；FP rate `60.0% → 0.0%`。
- mathhard：FN rate `31.6% → 0.0%`；FP rate `0.0% → 0.0%`。

### Conclusions

在这 40 条固定审计样本上，fixed scorer 与既有人工语义标签 40/40 一致；输出范围仍是 binary `0.0/1.0`。这证明修复解决了已观察到的 scorer 工程缺陷，但不等价于证明 scorer 对未见数据具有同样的准确率。

## New FP-FN analysis

修复后从 0 变为 1 的 18 条全部是 TP，具体类别为：

- NQ：Daimler-Benz 2 条、John McCrae 2 条、bust 2 条、democratic / Democratic-Republicans 2 条、Grant Park 2 条、Oscar the Grouch 2 条。
- mathhard：两个 separating-vector `Yes` 2 条、自然数存在性 `Yes` 2 条、稀疏正交不等式 `Yes` 2 条。

针对新增 positive 的抽查结论：

- 正确实体都出现在答案段或答案句中，并通过 token-boundary / 有限 possessive span；没有使用数据集 id 或答案字符串硬编码。
- 三个原始日期 FP 均没有新增为 positive：`May 19, 2017` 不匹配 `October 3, 2017`；`March 31, 2018` 不匹配 `7 April 2018`。
- 原始 `Sophie Sumner` / 未确认身份与 `Louise Glover` 仍为 TN；`20` 与 groundtruth `13` 仍为 TN。
- `not Oscar the Grouch`、`Oscar the Grouch is not ...`、`not yes`、`The answer is 20, not 13` 均由单元测试拒绝。
- mathhard 的分数、根式、`x+1` 和 `x_0=1/3` 样例均保持 positive；没有用共享数字 token 来判定数学等价。

因此，fixed regression 中没有观察到新的 FP；原始 3 个 FP 均被清除，原始 18 个 FN 均被恢复。

## Remaining uncertainties

- 上述 TP/TN/FP/FN 仍继承前一份 audit 对这 20 个样本的人工语义标注，不是独立外部裁判；尤其 Linkin Park 问题存在专辑日期与 title-track 日期的自然语言歧义，本回归以 parquet groundtruth `October 3, 2017` 为 benchmark oracle。
- phrase matching 只处理短实体、token 边界和有限 possessive gap；更复杂的同义改写、跨句推理和多答案问题可能保守返回 0。
- Yes/No scorer 只验证显式结论，不验证长数学证明本身；SymPy 等价也只覆盖可安全解析的有限表达式。
- 若未来显式设置 `AGENTFLOW_USE_LLM_SCORER=1`，`compute_score` 会进入原有外部 LLM 分支；本次修复和本次 offline regression 针对的是默认 deterministic fallback 路径。
- 本轮未重新计算新的 GRPO group variance、advantage 或 policy update；这些训练指标仍是原始实验的记录，不能由离线 scorer 修复反推训练已有效。

### Hypotheses

- 修复后的 reward 分布更接近语义正确性，可能改变后续 GRPO group 的 advantage 分布；但这需要一个单独批准的受控实验验证，不能从本次离线重打分直接推断。

## Recommendation for next controlled experiment

本轮建议到此停止。先由网页端独立审计代码和回归结果；若批准下一步，使用同一模型、同一 GPU、同一 rollout.n、同一数据和同一 Flow-GRPO 配置，仅将 scorer 作为唯一变量，进行一个明确记录的受控验证。不要在此报告之后自动启动 pre-train validation、训练或扩大数据规模。

本轮未启动任何 GPU 训练，未切换 7B，未修改 rollout.n、Flow-GRPO、模型或 reward 数值范围。
