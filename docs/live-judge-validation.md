# GLM Judge 实际验证与兼容修复

本记录补充 Sprint 10 发布检查，区分接口能否运行和裁判评分是否可信。未更换模型、修改 Prompt、调整评分规则或改动正式 Corpus/Dataset。

## 根因与修复

旧 `JUDGE_RESPONSE_FORMAT=json` 只在 Prompt 中要求 JSON，并未请求供应商的 JSON 模式。GLM 返回 Markdown 代码围栏，严格解析失败；首次10条样例各重试至3次后仍失败。

新增显式 `json_object` 模式，发送 `response_format={"type":"json_object"}`。继续执行严格 JSON/Pydantic 校验和有限重试，不剥离围栏、不补字段、不修正模型分数。原 `json` 与 `json_schema` 保持兼容。

本地配置已切换为 `JUDGE_RESPONSE_FORMAT=json_object`；模型、密钥不变。仓库示例保留通用默认值并说明 GLM 配置。缓存键已包含输出模式，因此旧模式缓存不会复用。Provider 与 Runner 已支持透传该参数，无需改动。

## 10条真实 Judge 验证

- 模型：`glm-4-flash-250414`，Prompt：`judge_v2`，temperature=0。
- 修复后：10/10通过格式与Schema校验，0次重试。
- 每条保存 model、prompt version、latency、raw output 和尝试记录。
- 此处 PASS 仅表示调用和结构校验成功，不是裁判准确性验收。

观察到的语义问题：

| 样例 | 输入事实 | 当前裁判输出 | 人工观察 |
| --- | --- | --- | --- |
| smoke_context_conflict | 标准答案30分钟，上下文和回答60分钟 | correctness=4、faithfulness=5 | 正确性明显高估，reason还声称符合标准答案 |
| wrong_refund | 政策只支持原路退款，回答任意指定银行卡 | correctness=0、faithfulness=4、hallucination=true | faithfulness过高，与理由中指出的矛盾不一致 |
| smoke_unanswerable_fabrication | 回答只有“会员售价是每期9.9元。” | hallucination=true、refusal_detected=true | 捕捉到了编造，但把没有拒答的回答误标为拒答 |

当前 GLM 可用于验证真实链路；以上样例说明它还不适合作为未经人工复核的质量判定依据。生成与裁判使用同一模型，也不能视为独立评审。

原始运行产物位于本地被 Git 忽略的目录：

- `artifacts/live-validation/judge_samples.json`：修复前10条失败记录。
- `artifacts/live-validation/json_object_probe.json`：保持评分规则不变的2条接口对照实验。
- `artifacts/live-validation/judge_samples_json_object.json`：修复后10条完整结果。

## 自动化验证

先运行6项新回归用例：旧实现6项失败；加入兼容支持后，Judge客户端与配置测试47项通过。

完整测试（Python 3.11，无真实 API）：

```bash
python -m pytest -q --basetemp .cache/judge-json/full-311-temp -o cache_dir=.cache/judge-json/full-311-cache
```

```text
356 passed, 20 warnings in 32.90s
IMPORT CHECK: PASS (45 modules)
```

20条警告来自 Chroma 第三方弃用提示。测试覆盖请求透传、v1/v2元数据、非法JSON、Schema/分数错误、有限修复重试和修复成功。既有测试继续覆盖原两种模式、Provider与缓存。

## 60条真实 RAG + Judge Benchmark

时间（UTC）：2026-09-06T05:55:58.698873+00:00。RAG_MODE=LIVE、JUDGE_MODE=LIVE，
生成模型和裁判均为 `glm-4-flash-250414`，Prompt=`judge_v2`，
Embedding=`BAAI/bge-small-zh-v1.5`，Top-K=3。无 mock、无 Judge 缓存命中。

| 指标 | 本轮结果 |
| --- | ---: |
| Total / Success / Failure | 60 / 60 / 0 |
| Failure breakdown | 空，无执行失败 |
| 重试 / Judge缓存命中 | 0 / 0 |
| Recall@3（48条可回答） | 1.0000 |
| Precision@3 | 0.3403 |
| MRR@3 | 0.9583 |
| Correctness（0–5） | 4.0500 |
| Faithfulness（0–5） | 3.9833 |
| Relevance（0–5） | 4.6000 |
| Completeness（0–5） | 4.0167 |
| Judge报告的幻觉率 | 0/60（0%） |
| Judge报告的拒答准确率 | 12/12（100%） |
| 平均case延迟 | 4578.01 ms |

**这些生成和安全分数是当前 GLM 的真实输出，但未经过人工校准。**
上述人工样例已经暴露裁判误判，不能据此宣称系统没有幻觉或拒答完全可靠。
流水线成功率是工程执行结果，不是答案正确率。当前仅完成接口兼容与真实链路验证，裁判质量验收仍有待完成。

复现命令：

```bash
python -m src.judge.live_validation --output artifacts/live-validation/judge_samples_json_object.json
python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/live-evaluation --judge-mode LIVE
```

以上命令会真实调用已配置的供应商；再次运行可能命中Judge缓存，应核对报告中的缓存计数。
完整报告与逐条原始证据保存在本地 `artifacts/live-evaluation/latest_report.json` 和
`artifacts/live-evaluation/case_results.json`，保持 Git 忽略。历史 MOCK 报告与Baseline均未替换。
Dashboard可选择 `artifacts/live-evaluation/latest_report.json` 查看本轮结果；历史截图仍展示旧 MOCK 报告。

正式数据版本：

- Corpus SHA-256：`989ef975f4b8ded581b74eb75624dc19358e015b1c5b6396471fa4334227cc2e`
- Dataset SHA-256：`7bf63862ff4730d5884f92c20eb63888f58407d3785c24c8d1750e1d4cc7ba3f`

后续建议先为人工样例建立独立评分标注，再在同一组输入上验证其他裁判模型；
不通过修改正式数据、放宽校验或调整阈值来掩盖当前模型误判。
