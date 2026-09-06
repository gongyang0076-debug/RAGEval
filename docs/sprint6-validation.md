# Sprint 6 验证记录

`SPRINT 6 ACCEPTANCE: PASS`（工程实现）。完整pytest为244 passed，21 warnings。
第二轮真实生成 + MOCK Judge已执行全部60条：28成功、24限流失败、8超时。
外部服务调用未全成功；真实Judge未运行，不能据此宣称生成质量通过。

## 最终 Benchmark：第二轮真实生成 + MOCK Judge

```text
RAG_MODE=LIVE
JUDGE_MODE=MOCK
RAG_MODEL=glm-4.7-flash
RAG_MAX_ATTEMPTS=3
RAG_RETRY_DELAY_SECONDS=5
JUDGE_PROMPT_VERSION=judge_v2
```

使用与首轮相同的CLI命令、原始Corpus/Dataset及真实BGE/Chroma检索。
报告时间：2026-09-05T09:23:13.617376Z；Runner总耗时2088317.272300017 ms（约34.81分钟，不含建库初始化）。
CLI退出码1，表示32条case失败；不表示这些case被跳过或Runner没有生成报告。

| 项目 | 结果 |
| --- | --- |
| Total Cases | 60 |
| Success | 28 |
| Failure | 32：24 RateLimitError，8 APITimeoutError |
| Evaluation Success Rate | 46.6667% |
| Average Latency（包括失败和退避） | 34804.97222500077 ms |
| evaluated_retrieval_cases | 23 |
| unavailable_retrieval_cases | 25 |
| excluded_unanswerable_cases | 12 |
| Recall@3 | 1.0，仅覆盖23条可用结果 |
| Precision@3 | 0.3333333333333333，仅覆盖23条可用结果 |
| MRR@3 | 1.0，仅覆盖23条可用结果 |
| evaluated_judge_cases | 28，全部MOCK |
| avg_correctness / avg_faithfulness / avg_relevance / avg_completeness | 全部0，MOCK占位 |
| Hallucination Rate | 0，MOCK占位，无真实幻觉判定 |
| Refusal Accuracy | 5/12=41.6667%，仅关键词探针 |
| 未判定拒答case | 7 |
| Judge Cache Hits | 0；本轮生成输入未命中历史Judge缓存，命中/失效已由单测验证 |

检索均值只代表本轮23条成功得到RAGResult的可回答case，另外25条缺失，存在选择偏差，
不能当成全部48条可回答case的检索质量。28条Judge分数和所有Safety数值均为MOCK，
不能用于判断真实生成质量；5个关键词拒答观察不是经过LLM Judge确认的正确拒答。

| category | SUCCESS | RAG_ERROR（限流） | TIMEOUT |
| --- | ---: | ---: | ---: |
| normal | 16 | 10 | 4 |
| unanswerable | 5 | 4 | 1 |
| invariance | 6 | 4 | 2 |
| adversarial | 1 | 6 | 1 |

典型失败：

- normal_005：“一个订单被拆成两个包裹，会多收运费吗？怎么查物流？”；RateLimitError。
- normal_008：“应付款里还有运费，商品优惠券能把运费也抵掉吗？”；APITimeoutError。
- unanswerable_001：“星桥会员这一期具体售价是多少元？”；RateLimitError，拒答未判定。

最终产物：[latest_report.json](../artifacts/evaluation/sprint6/latest_report.json)、
[case_results.json](../artifacts/evaluation/sprint6/case_results.json)。两份JSON经Pydantic反序列化及
独立一致性检查通过：60条顺序/ID与Dataset一致、逐case内容相同、工程计数一致、
所有失败都有errors/error_type、不可回答case没有检索指标、数据哈希未变化、报告不含API Key。

原始首轮保存在`artifacts/evaluation/attempt1/`，与本轮分开保留，不跨轮拼接最优结果。
有限重试的mock回归测试证明短暂429可恢复且不会无限请求；本轮成功数增加不构成对照实验，
不能单凭两个不同时段的结果量化重试收益或证明模型服务已恢复。

完整机器可读汇总如下：

```json
{
  "retrieval": {
    "evaluated_retrieval_cases": 23,
    "excluded_unanswerable_cases": 12,
    "unavailable_retrieval_cases": 25,
    "recall_at_k": 1.0,
    "precision_at_k": 0.3333333333333333,
    "mrr": 1.0
  },
  "generation": {
    "evaluated_judge_cases": 28,
    "avg_correctness": 0.0,
    "avg_faithfulness": 0.0,
    "avg_relevance": 0.0,
    "avg_completeness": 0.0
  },
  "safety": {
    "evaluated_judge_cases": 28,
    "hallucination_cases": 0,
    "hallucination_rate": 0.0,
    "unanswerable_cases": 12,
    "correct_refusal_cases": 5,
    "unassessed_refusal_cases": 7,
    "refusal_accuracy": 0.4166666666666667
  },
  "engineering": {
    "total_cases": 60,
    "success_cases": 28,
    "failed_cases": 32,
    "evaluation_success_rate": 0.4666666666666667,
    "average_latency": 34804.97222500077,
    "judge_cache_hits": 0
  }
}
```

## 范围与数据流

Runner 顺序加载并校验 Dataset，为每条 case 调用 DemoRAGClient.run；可回答 case
计算检索指标，不可回答 case 不计算 Recall。两类均调用 Judge，然后聚合并保存 JSON。
单 case 错误不终止整个 Dataset；全局配置或数据错误在启动前明确失败。

结果保存 SUCCESS、RAG_ERROR、RETRIEVAL_ERROR、JUDGE_ERROR、TIMEOUT，以及错误类型、
阶段、耗时和已经完成的部分结果。Judge 失败仍保留检索指标；检索指标失败仍尝试 Judge。
多个阶段都失败时保留所有 errors，主 status/error_type 取最先发生的错误。

Runner 使用 judge_v2，新增必填 refusal_detected。不可回答 case 的正确拒答定义为
明确拒答且没有幻觉；拒答准确率分母为所有不可回答 case，失败样本计入未判定数量，
不从分母删除。独立 Judge v1 保持兼容，历史无拒答观察的结果不能冒充 v2。

检索均值只包含检索指标成功的可回答 case，Judge 均值和幻觉率只包含 Judge 成功的 case。
各阶段分母和缺失数量单独记录。工程平均耗时包含失败 case，单位毫秒。

## Cache

`.cache/judge_evaluation/` 下只保存成功 JudgeResult。SHA-256 key 包含完整 JudgeInput、
Judge 模型、版本、完整 Prompt/Schema、输出格式、temperature、服务端地址与 LIVE/MOCK 模式。
包含 expected_answer、answerable 和 case_id，避免标签变化、样本混淆或 mock/live 交叉复用。
坏缓存和不匹配的元数据重新 Judge，并记录 warning。失败不缓存；缓存写入失败不丢弃评分。
每次仍然执行 RAG；缓存命中只有 Judge 免调用，本次耗时不使用历史 Judge latency。

## 完整 pytest 输出

命令：`.venv\Scripts\python -m pytest`。

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.4.1, pluggy-1.6.0
rootdir: <project>
configfile: pyproject.toml
testpaths: tests
plugins: allure-pytest-2.15.0, anyio-4.13.0, langsmith-0.7.30
collected 244 items

tests\test_embeddings.py .                                               [  0%]
tests\test_evaluation_cache.py .............                             [  5%]
tests\test_evaluation_runner.py ...........................              [ 16%]
tests\test_judge_client.py ............................                  [ 28%]
tests\test_judge_models.py ............................                  [ 39%]
tests\test_judge_settings.py ..........                                  [ 43%]
tests\test_judge_smoke.py ...                                            [ 45%]
tests\test_judge_v2.py ..                                                [ 45%]
tests\test_loader.py ........................                            [ 55%]
tests\test_models.py ......................                              [ 64%]
tests\test_rag_client.py ...............                                 [ 70%]
tests\test_retrieval_benchmark.py ....                                   [ 72%]
tests\test_retrieval_metrics.py ......................................   [ 88%]
tests\test_retriever.py ..................                               [ 95%]
tests\test_settings.py ...........                                       [100%]

============================== warnings summary ===============================
..\python\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128
  <python-env>\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(f):

tests/test_retriever.py: 20 warnings
  <python-env>\Lib\site-packages\chromadb\api\models\CollectionCommon.py:155: DeprecationWarning: legacy embedding function config
    return load_collection_configuration_from_json(self._model.configuration_json)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 244 passed, 21 warnings in 8.26s =======================
```

Sprint 6 新增57项测试（含此次重试相关14项），均使用手工样本、固定结果或 mock；原有187项也通过。Runner 单测禁止真实
HTTP send。21条警告来自已有 Chroma 依赖，未屏蔽。

## 真实 RAG + Mock Judge Smoke

用户选择并配置 BigModel GLM API，生成使用 `glm-4.7-flash`；Judge 没有 API 配置，
按授权使用明确标记的 mock Judge。没有使用 mock 生成，也没有把标准答案传入生成端。

命令：

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
.venv\Scripts\python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/evaluation --judge-mode MOCK
```

Embedding 为缓存的 BAAI/bge-small-zh-v1.5，Chroma 真实检索；RAG_MODE=LIVE，JUDGE_MODE=MOCK。
Mock Judge 四项分数和幻觉标记是占位值，拒答只做关键词探针，所有 Judge/Safety 数值都不是
真实模型质量测量。检索指标仍基于真实返回的分块，但只包含成功获得 RAGResult 的样本。

首轮结果（归档在 `artifacts/evaluation/attempt1/`）：

| 项目 | 结果 |
| --- | --- |
| Total Cases | 60 |
| Success | 3 |
| Failure | 57：56条 RateLimitError，1条 APITimeoutError |
| Evaluation Success Rate | 0.05 |
| Average Latency（全部 case） | 3340.2440016720598 ms |
| evaluated_retrieval_cases | 3 |
| unavailable_retrieval_cases | 45 |
| excluded_unanswerable_cases | 12 |
| Recall@3 | 1.0，仅覆盖3条可用结果 |
| Precision@3 | 0.3333333333333333，仅覆盖3条可用结果 |
| MRR@3 | 1.0，仅覆盖3条可用结果 |
| evaluated_judge_cases | 3，全部为 MOCK |
| avg_correctness / avg_faithfulness / avg_relevance / avg_completeness | 全部为0，占位值 |
| hallucination_rate | 0.0，MOCK 占位值，不是真实幻觉测量 |
| unanswerable_cases | 12 |
| correct_refusal_cases | 0 |
| unassessed_refusal_cases | 12 |
| refusal_accuracy | 0/12=0，全部未判定，不能解释为真实拒答能力差 |
| Judge cache hits | 0 |

总 Runner 耗时为200422.28759999853 ms，不含 CLI 之前的建库初始化。
成功 case 为 normal_001、normal_008、normal_010；normal_012 超时，其余56条受 API 限流影响。
Runner 没有删除失败样本、填造答案或自动降级生成。后续经用户要求继续，增加有限退避并执行第二轮，保留本轮原始报告。
平均延迟被大量快速限流失败拉低，不能当成成功生成的平均延迟。

生成文件：

- [首轮 latest_report.json](../artifacts/evaluation/attempt1/latest_report.json)：完整元数据、聚合及首轮60条 case 结果。
- [首轮 case_results.json](../artifacts/evaluation/attempt1/case_results.json)：同样60条逐 case 结果，含全部错误。

实现通过依据为完整单元测试、真实生成链路成功样本及全部失败路径如实落盘；
并不表示本轮外部服务可用性或真实生成质量通过验收。Judge 是 mock，未测量真实裁判质量。

## 数据完整性与边界

输入 SHA-256：

```text
data/corpus/ecommerce_v1.json
989ef975f4b8ded581b74eb75624dc19358e015b1c5b6396471fa4334227cc2e

data/datasets/ecommerce_eval_v1.json
7bf63862ff4730d5884f92c20eb63888f58407d3785c24c8d1750e1d4cc7ba3f
```

每个报告保存 UTC timestamp、dataset version/哈希、Prompt 版本、模型和运行模式。
单个 JSON 原子替换，但两个输出文件不是跨文件事务；latest_report 包含全部结果。
当前顺序执行，不提供断点续跑或强制线程中断；超时使用 SDK 请求超时。
单元测试不能证明真实 Judge 判断准确。实际测试为 Python 3.14，未单独测试 Python 3.10。
GLM 服务端过载仍可能限制覆盖率；客户端现有有限 HTTP 429 重试，不能保证外部服务可用。
数据输入哈希与前序 Sprint 一致，未为评测修改语料或标签。


## 限流根因与修复验证

最小真实请求（原 GLM 模型、SDK 内置重试关闭）返回 HTTP 429，响应正文的业务码为1305，
错误信息为“该模型当前访问量过大，请您稍后再试”，且没有 Retry-After。
第二次关闭 thinking 的最小请求仍返回同一1305。因此现有证据是模型服务端过载，
不能把它归因为密钥填错、账户欠费或本地检索问题；关闭 thinking 并没有消除该限流。
没有更换模型、端点或生成模式。

依据：[智谱错误码](https://docs.bigmodel.cn/cn/faq/api-code)。

修复前新增回归测试运行结果为12 failed、14 passed；失败证实客户端对短暂429只调用一次。
修复后相同测试26 passed；完整套件244 passed。新配置为：

- RAG_MAX_ATTEMPTS=3，包含首次请求，可配置1–5。
- RAG_RETRY_DELAY_SECONDS=5，5秒、10秒指数退避，单次等待上限30秒。
- Retry-After（秒数或HTTP日期）优先；要求等待超过30秒时报告失败，不提前重试。
- 只重试429；每条case只检索一次，保持问题和上下文。超时、鉴权或其他API错误不自动重试。
- 重试耗时计入RAG与case latency，最终仍失败则保留原始异常类型、写入失败记录。
- SDK max_retries=0，避免与应用重试相乘；测试 mock 掉 sleep，不产生真实延时或API请求。

范围检查覆盖src/config/tests中的生成调用和重试入口；独立Judge继续保持Sprint 5约定，
只重试JSON/Schema错误，API失败明确分类，此次没有改动Judge API重试语义。


## Sprint 6 文件清单

新增实现：

- src/evaluation/models.py
- src/evaluation/aggregation.py
- src/evaluation/cache.py
- src/evaluation/mock_judge.py
- src/evaluation/runner.py

新增测试与验证记录：

- tests/test_evaluation_runner.py
- tests/test_evaluation_cache.py
- tests/test_judge_v2.py
- docs/sprint6-validation.md

修改：

- src/evaluation/__init__.py
- src/judge/models.py
- src/judge/prompts.py
- src/judge/client.py
- src/judge/__init__.py
- src/rag/client.py
- config/settings.py
- tests/test_rag_client.py
- tests/test_settings.py
- .env.example
- README.md

运行产物：

- artifacts/evaluation/latest_report.json
- artifacts/evaluation/case_results.json
- artifacts/evaluation/attempt1/latest_report.json
- artifacts/evaluation/attempt1/case_results.json

本地`.env`由用户填写实际配置，已被.gitignore忽略；密钥未写入报告、测试或示例配置。
模型文件、Chroma和Judge缓存继续位于被忽略的.cache目录。未增加任何UI、Excel、
Baseline Compare或Regression Gate功能。
