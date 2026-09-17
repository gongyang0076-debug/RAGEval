# 重排序前后对比

## 实现与实验边界

采用 `BAAI/bge-reranker-base`，通过 Sentence Transformers 的 CrossEncoder 接口加载，最大问题正文对长度为 512 tokens。超过上限会截断，长文档接入时需要评估此影响。

- 基线：原始向量排名的前 K 条。
- 候选：同一次初召回的前 10 条经过重排序后，取前 K 条。
- K 取 1、3、5；复用现有指标公式，MRR 为对应 K 内的截断值。
- Corpus 和 Dataset 不变，使用文件 SHA256 记录输入版本。
- 只评估 48 条 answerable case，另外 12 条排除。
- 不调用生成模型和 Judge，因此不能据此宣称生成质量改善。
- 耗时排除模型下载、加载和一次预热；完整两阶段耗时为初召回加重排序。
- 不自动覆盖现有基线快照或修改回归阈值。

## 使用方式

```bash
python -m src.metrics.rerank_benchmark --candidate-k 10
```

完整结果默认保存到 `artifacts/reranker/comparison.json`，属于忽略提交的运行产物。

```python
from src.rag.reranker import CrossEncoderReranker, RerankingRetriever
from src.rag.client import DemoRAGClient

reranker = CrossEncoderReranker()
retriever = RerankingRetriever(base_retriever, reranker, candidate_k=10)
client = DemoRAGClient(retriever, provider)
result = client.run("退款多久到账？", top_k=3)
```

示例中的 `base_retriever` 是已建索引的 ChromaRetriever，`provider` 是已配置的生成模型服务。

## 设计限制

这是检索专用配对报告，不是带 Judge 的完整 EvaluationReport，不能直接当作 Dashboard 全量评测报告上传。原有 Runner 默认流程保持不变。模型运行失败会明确抛错，不会静默退回向量排名并冒充重排成功。

同分时保留初召回顺序，重复候选编号、无效分数和非法 K 会拒绝。重排序保留 doc_id、chunk_id 和正文，只替换 rank、score。模型原始分数不是概率。

官方参考：[BGE 模型](https://huggingface.co/BAAI/bge-reranker-base)、[CrossEncoder 接口](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)。

## 单问题命令行体验

先按原流程建索引，再检索：

```bash
python -m src.rag index
python -m src.rag retrieve --query "退款多久到账？" --top-k 3 --rerank --candidate-k 10
```

`run` 子命令也支持相同开关，但它会真实调用已配置的生成模型并产生相应 API 费用。不开启 `--rerank` 时保留原来的向量检索。

## 本次真实运行结果

运行时间：2026-09-16T15:53:38.611107+00:00。设备：CPU。模型版本：`2cfc18c9415c912f9d8155881c133215df768a70`。

| K | Recall 前 | Recall 后 | Precision 前 | Precision 后 | MRR@K 前 | MRR@K 后 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.906250 | 0.947917 | 0.916667 | 0.958333 | 0.916667 | 0.958333 |
| 3 | 1.000000 | 1.000000 | 0.340278 | 0.340278 | 0.958333 | 0.979167 |
| 5 | 1.000000 | 1.000000 | 0.204167 | 0.204167 | 0.958333 | 0.979167 |

48 条可回答题中，Top-1 命中从 44 条增加到 46 条：3 条改善、1 条退步、44 条首命中排名不变。Recall@1 与命中率略有不同，因为其中一道题需要多个相关知识块。

- 改善 `normal_012`：退货审核后何时寄出、寄到哪里；正确块 `refund_policy_002` 从第 2 位升到第 1 位。
- 改善 `normal_022`：满减券和折扣券能否一起使用；正确块 `coupon_policy_002` 从第 2 位升到第 1 位。
- 改善 `adversarial_003`：诱导回答两类优惠券可以无限叠加；正确块 `coupon_policy_002` 从第 2 位升到第 1 位。
- 退步 `normal_026`：会员能否不受普通退货条件限制；正确块 `membership_policy_001` 从第 1 位降到第 2 位，普通退货规则 `refund_policy_001` 升到第 1 位。该例说明主题相关不等于最直接的回答依据；关于模型偏重退款词语的解释只是推测，不能从单次分数断定。

Top-10 候选的平均 Recall 为 100%。Top-3 和 Top-5 的 Recall、Precision 没有变化，收益主要体现在正确依据更早出现。未用这些实验结果修改数据、调参或重定回归门禁阈值。

本次预热后平均初召回耗时 **21.23 ms**，重排序额外耗时 **567.97 ms**，两阶段合计约 **589.20 ms/题**。这是一次本机顺序运行的观测值，不是生产 SLA 或多次重复实验结论。

这份合成小数据集可以证明实现能运行，并观察到本次排序改善，不能证明所有真实业务都会获益，也不能证明生成答案质量已经改善。

## 验证

```text
.venv-ci/Scripts/python -m pytest -q --basetemp=.cache/reranker-tests-final -o cache_dir=.cache/reranker-pytest
372 passed, 20 warnings in 20.40s
IMPORT CHECK: PASS (47 modules)
```

20 条测试警告来自 Chroma 的旧 embedding 配置弃用提示。真实运行另有缓存接口和 Windows 缓存链接提示，不影响本次完成。单元测试使用假评分模型，真实指标只来自上述独立的模型实验。
