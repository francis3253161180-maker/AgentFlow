# 主实验 `20260825_231408` offline reward audit

审计日期：2026-08-26。审计范围严格限定为主实验 rollout 根目录下的 `train/`：40 条训练 rollout、20 个唯一数据样本；`validation/` 的 30 条 rollout 不混入下面的训练统计。逐条完整的 `groundtruth`、`answer_extracted`、保存的 `reward`、fallback 复现结果、source 和判定见 [`2026-08-26_reward_audit_results.json`](2026-08-26_reward_audit_results.json)，复现脚本见 [`audit_reward_20260826.py`](../scripts/audit_reward_20260826.py)。

复现命令：

```bash
/root/autodl-tmp/conda/envs/agentflow/bin/python scripts/audit_reward_20260826.py \
  --run-root rollout_data/46.38.243.197/qwen25-3b-lora-mini20-seed20260825_20260825-231521/Qwen2.5-3B-Instruct_20260825-231521 \
  --dataset data/train/flowgrpo_mini_20_seed20260825.parquet \
  --train-log log/20260825_231408_lora_mini20_train.log \
  --output log/2026-08-26_reward_audit_results.json
```

## 实验标识与范围

- 分支：`experiment/flow-grpo-3b-lora`
- 审计开始时源码 HEAD：`f074aacf55cdac6f2bf4fd67c55e708779692349`
- 配置：`train/config_5090_lora_mini20.yaml`
- 数据：`data/train/flowgrpo_mini_20_seed20260825.parquet`，source 为 `mathhard` 10 条、`nq` 10 条
- 模型/LoRA：Qwen2.5-3B-Instruct，rank 8，alpha 16；本次未更换模型、GPU 或 rollout.n
- 训练 rollout：40 条；validation：30/30 completed、30 valid，未纳入训练 reward audit

## Observed facts

### 逐条 reward 与语义审计汇总

下面每行代表一个唯一数据样本；`rewards` 与 `verdicts` 按保存 rollout 顺序列出。TP/TN/FP/FN 是本次离线人工语义复核标签，不是重新训练或外部 LLM judge 的结果。

| source | id | groundtruth | rollouts | rewards | verdicts | answer 摘要 |
|---|---:|---|---:|---|---|---|
| nq | 66972 | Daimler-Benz | 2 | 0,0 | FN,FN | 两次答案都明确说 BMW nearly sold to Daimler-Benz。|
| nq | 34128 | 22 | 2 | 1,1 | TP,TP | `There are 22 episodes...` |
| mathhard | 80293 | Yes | 2 | 0,0 | FN,FN | 两次都回答 admits a separating vector，结论为 Yes。|
| nq | 98332 | Louise Glover | 2 | 0,0 | TN,TN | Sophie Sumner；或称未被正式确认，均不是 Louise Glover。|
| mathhard | 116769 | 6 | 2 | 1,1 | TP,TP | `6` |
| mathhard | 108190 | `\dfrac{1}{2}` | 2 | 1,1 | TP,TP | `\frac{1}{2}` |
| mathhard | 101323 | 13 | 2 | 1,0 | TP,TN | `13`；`20`。|
| nq | 121873 | John McCrae | 2 | 0,0 | FN,FN | 两次均明确写出 John McCrae。|
| mathhard | 138626 | Yes | 2 | 0,0 | FN,FN | 两次均以 Yes 开头并给出满足条件的例子。|
| mathhard | 5991 | `\dfrac{1}{6}` | 2 | 1,1 | TP,TP | `\frac{1}{6}`；带解释的同一答案。|
| nq | 129667 | bust | 2 | 0,0 | FN,FN | 两次均说明 over 21 is called “bust”。|
| mathhard | 68772 | x + 1 | 2 | 1,1 | TP,TP | `f(x)=x+1` |
| nq | 35924 | democratic | 2 | 0,0 | FN,FN | 两次给出 Democratic-Republicans / Jeffersonian democracy 的正确历史结论。|
| nq | 61526 | October 3, 2017 | 2 | 1,1 | FP,TP | 一次只答专辑 May 19；一次同时指出 title track 为 October 3。|
| nq | 147614 | Chicago's Grant Park | 2 | 0,0 | FN,FN | 两次都指出 Art Institute 位于 Grant Park。|
| mathhard | 132919 | Yes | 2 | 0,0 | FN,FN | 两次均回答存在绝对常数 `C'`，结论为 Yes。|
| nq | 6418 | 7 April 2018 | 2 | 1,1 | FP,FP | 两次都答 March 31, 2018，与该 benchmark groundtruth 不同。|
| mathhard | 51978 | `1 + \sqrt{2}` | 2 | 1,1 | TP,TP | `\sqrt{2}+1`；`√2+1`。|
| nq | 90379 | Oscar the Grouch | 2 | 0,0 | FN,FN | 两次都明确写出 Oscar the Grouch。|
| mathhard | 67873 | `\dfrac{1}{3}` | 2 | 1,1 | TP,TP | `x_0=\frac{1}{3}` |

总 reward 计数为 `reward=1: 19`、`reward=0: 21`。人工语义标签为：

| source | TP | TN | FP | FN | 合计 |
|---|---:|---:|---:|---:|---:|
| nq | 3 | 2 | 3 | 12 | 20 |
| mathhard | 13 | 1 | 0 | 6 | 20 |
| 合计 | 16 | 3 | 3 | 18 | 40 |

按通常定义，false-negative rate 为 `FN/(TP+FN)`，false-positive rate 为 `FP/(TN+FP)`：

- 总体：FN `18/34 = 52.9%`；FP `3/6 = 50.0%`。
- NQ：FN `12/15 = 80.0%`；FP `3/5 = 60.0%`。
- mathhard：FN `6/19 = 31.6%`；FP `0/1 = 0.0%`。
- 若以全部 40 条为分母，FN 占 `45.0%`，FP 占 `7.5%`。

### 当前 scorer 的逐条复现

- `scripts/audit_reward_20260826.py` 在无网络、无 LLM judge 条件下逐条复现 `train/utils.py` 当前 fallback：40/40 条与 JSON 保存的 reward 一致，mismatch `0`。
- fallback 先把整个 `answer_extracted` 作为一个字符串做规范化后比较；它没有真正抽取最后答案或 `<answer>...</answer>` 内的答案。
- fallback 的数值兜底在 `train/utils.py:53-56` 比较任意共同数字 token，而不是完整日期/完整数值。例如 `October 3, 2017` 与 `May 19, 2017` 共享 `2017`，`7 April 2018` 与 `March 31, 2018` 共享 `2018`，因此产生本次 3 个 FP 中的 3 个日期型 FP。
- 生产 rollout 路径 [`train/rollout.py:45`](../train/rollout.py#L45) 以 `(question, groundtruth, answer_extracted)` 的正确顺序调用 `compute_score`。[`train/utils.py:103`](../train/utils.py#L103) 的自带 `eval` 测试函数却交换了后两个参数；在本批数据上，由于当前 fallback 的主要比较操作是集合交集，这个交换导致的差异为 0/40，不是本次主要原因。

### 训练日志中与 reward audit 直接相关的事实

- 日志记录的训练 batch 为 10 个，每批 4/4 completed、4 valid；原始训练 JSON 共 40 条。
- 已解析到 global step 1–9 的 metric 行；`actor/pg_loss` 九步均为 `0.0`。
- step 1–6、8–9 的 `critic/advantages` min/max/mean 均为 `0.0`；step 7 为 mean `-0.034912109375`、min `-0.70703125`、max `0.70703125`，同一步 `actor/grad_norm=0.23865246772766113`。
- 记录到的 reward min/max 为 `0.0/1.0`；`timing_s/update_actor` 有数值，但日志没有单独的 `optimizer.step succeeded` 字段，因此不能据此断言 optimizer step 成功。
- 记录到的最大 GPU allocated memory 为约 `21.61 GB`；old-log-prob 只记录了 timing，不存在独立的 `old_log_prob` 数值汇总字段。
- 日志最后显示 validation 30/30 valid，随后在 checkpoint 写入阶段失败：`PytorchStreamWriter ... file write failed`，位置约 `9.43 GB`。原始 YAML 为 `trainer.save_freq: 0`，但这次实际启动命令的日志 override 为 `trainer.save_freq=1000`。该失败是 checkpoint/storage 事件，不是本次 reward scorer 结论。

## Hypotheses

- fallback 的设计目标仍是 compact smoke-test answers；把它用于自然语言 Agent 输出，是本次大量 FN 的最可能工程原因：正确答案嵌在解释段落中时，整个字符串不会与短 groundtruth 相等。
- 日期型 FP 的直接原因是“任意共同数字 token”兜底；这比模型知识错误更能解释两个样本各两次都被判为 1。
- 训练中大多数 group 的 reward/advantage 为零，可能同时受到 scorer 错误和组内回答分布影响；仅凭本 audit 不能把全部非零/零 policy-gradient 现象归因于模型或 Flow-GRPO 算法。

## Conclusions

1. 当前 `train/utils.py::compute_score` fallback 是这批 40 条 reward 错判的主要原因：它复现了全部保存 reward，并产生 18 FN、3 FP。
2. 已确认的 Oscar 样本不是模型答错：`groundtruth="Oscar the Grouch"`，答案明确为 `Oscar the Grouch lives in the trash can on Sesame Street.`，却得到 0，是确定的 false negative。
3. 这批训练 reward 不能直接当作 answer accuracy：人工语义上 34/40 条正确，但 scorer 只给了其中 16 条 TP；同时给了 3 条明显错误答案 reward=1。
4. 证据支持一个 scorer 工程缺陷结论；证据不支持改变 Flow-GRPO、模型、GPU、rollout.n 或 reward 数值范围。

## Recommended minimal fix

暂不实施、暂不启动训练。最小 deterministic 修复应保持 reward 输出仍为 `0.0/1.0`，并只收敛 scorer 行为：

1. 统一生产调用与函数签名，移除 `train/utils.py::eval` 的参数交换；补充离线单元测试。
2. 先抽取 `<answer>...</answer>` 或明确的最后答案段，再做比较；不要把整段 reasoning 当作 candidate。
3. 文本答案采用大小写/标点归一化后的完整短语或 token-boundary 匹配，并处理 `Yes/No`；必要时加入最小否定保护，避免“not Oscar”误判为正确。
4. 数字/日期必须比较完整结构化值（完整日期、完整整数/小数/分数），删除“共享任意数字 token 即正确”的兜底。数学表达式继续使用确定性的规范化/等价比较。
5. 用本报告中的 Oscar、Daimler-Benz、John McCrae、bust、Art Institute、两个日期 FP、Sophie Sumner 和 `20 vs 13` 作为回归样例；修复后只做 offline scorer regression，不重新训练。

本轮未修改算法、模型、GPU、rollout.n 或启动新的训练进程。
