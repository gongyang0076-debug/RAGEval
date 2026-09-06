# Sprint 7 验证记录

`SPRINT 7 ACCEPTANCE: PASS`

本次真实检索 + 真实RAG生成 + 明确标记MOCK Judge的完整60条运行全部成功。
真实Judge尚未配置：`LIVE JUDGE NOT AVAILABLE`。这不是一次真实LLM裁判质量验收。

## Provider 架构

```text
DemoRAGClient → LLMProvider.generate(messages) → OpenAICompatibleProvider → AsyncOpenAI
LLMJudgeClient → 同一Provider → JSON/Schema校验或有限修复
EvaluationRunner → CaseEvaluationResult → 分阶段聚合 → EvaluationReport
```

DemoRAGClient只负责检索、上下文和Prompt，不导入OpenAI SDK，不承担HTTP重试。
CLI装配具体Provider；构造接口更新为DemoRAGClient(retriever, provider)。
Judge的可注入依赖也更新为LLMProvider，原Prompt v1/v2评分口径保持不变。

LLMResponse包含text和ProviderTrace；ProviderError包含可序列化ProviderFailure。
Trace记录model、无凭据base_url、timeout、最大尝试数、thinking选项、retry_count、
总延迟和逐次尝试。尝试记录包括HTTP状态、业务码、错误类型、类别、耗时和等待时间。
API错误正文和密钥不写入Trace。Judge仍保留raw output、model、prompt_version和latency。

RAGResult新增retry_count和generation_trace；Case新增rag_retry_count、judge_retry_count及
总retry_count，错误新增provider_failure。Engineering新增retry_count和failure_breakdown。
旧结果缺少新字段时可使用默认值反序列化；Judge缓存命中时不重复统计历史重试。

## Retry 策略

- 可重试：429、SDK/本地请求超时、HTTP408/504、连接错误、其他5xx。
- 不可重试：401认证、403权限、400无效请求、404或业务码1211/model_not_found。
- 已知欠费或额度耗尽代码（如1113、insufficient_quota）即使HTTP429也不重试。
- 最大尝试次数包含首次调用，RAG_MAX_ATTEMPTS和JUDGE_MAX_ATTEMPTS默认3，范围1–5。
- 初始退避由RAG_RETRY_DELAY_SECONDS/JUDGE_RETRY_DELAY_SECONDS控制，默认5秒，指数增长，单次最多30秒。
- Retry-After支持秒数和HTTP日期；超过30秒等待预算直接报告失败，不提前请求。
- SDK max_retries=0；Judge API重试与JSON修复共用总请求预算，不做3×3叠加。
- retry_count是实际请求数减1，包括Judge JSON修复请求；最终失败也记录重试次数。
- 终止错误决定case最终状态；前序错误仍完整保存在尝试历史。

## Timeout 策略

RAG_GENERATION_TIMEOUT_SECONDS对应Settings.generation_timeout；JUDGE_TIMEOUT_SECONDS
对应JudgeSettings.judge_timeout；默认均60秒，范围(0,120]。

Provider同时设置SDK网络超时和asyncio.wait_for单请求截止时间。到期取消异步请求，关闭客户端，
按预算重试；最终仍超时则保存TIMEOUT。测试通过可取消的30秒挂起协程验证：20ms截止时间
确实触发取消，RAG/Judge分别保留timeout结果。没有借助无法取消的后台线程。

每次generate调用创建并关闭一个异步客户端，避免跨已关闭事件循环复用连接。公开接口仍为同步，
当前CLI使用asyncio.run，不直接支持在已有事件循环内调用该同步接口。
请求总时长还包括有限重试、退避及本地工作，不把单请求timeout冒充整个Dataset截止时间。

## 根因与运行配置

Sprint 6的源代码只重试RateLimitError，生成请求timeout固定60秒；超时、连接异常、5xx直接失败，
且RAG报告缺少业务错误码。本阶段统一了这些行为并加入逐次追溯。

新Provider的真实短探测仍确认glm-4.7-flash返回429/1305（模型访问量过大），同平台免费
模型glm-4-flash-250414约1秒成功回答。诊断不是认证错误；更长重试无法保证恢复服务端容量。
证据保存在[provider_probe.json](../artifacts/sprint7/provider_probe.json)，旧模型的失败没有删除。

本地.env的RAG_MODEL已改为glm-4-flash-250414，整轮固定此模型，没有逐case自动回退或混用模型。
API Key和base_url保持原配置；示例配置不包含密钥。RAG_THINKING为可选扩展，本轮不传该参数。

依据：[智谱错误码](https://docs.bigmodel.cn/cn/faq/api-code)、
[GLM-4-Flash-250414官方说明](https://docs.bigmodel.cn/cn/guide/models/free/glm-4-flash-250414)。

本轮没有发生重试，所以60/60成功不能归因于重试策略本身；现场恢复主要来自选择可用模型。
重试和超时机制的恢复、终止行为由mock回归测试验证；并不证明原glm-4.7-flash已恢复服务。
更换生成模型可能改变答案质量；本阶段没有实现Baseline/Compare或质量对比结论。

## Live Judge

命令：

```powershell
.venv\Scripts\python -m src.judge.live_validation
```

```text
LIVE JUDGE NOT AVAILABLE
Saved 0 live results to artifacts\sprint7\live_judge_validation.json
```

验证模块准备10条独立人工样例，覆盖正确、错误、上下文与标准答案冲突、幻觉、正确拒答及
拒答后继续编造。未修改Corpus/Dataset。缺少Judge配置时results=[]，不创建Provider也不填假分。
配置齐全时逐条调用真实Judge v2，保存原始输出/版本/模型/延迟或错误，并继续处理后续样例。
本轮未将RAG密钥擅自复用为Judge配置，也未将Mock结果写入live验证文件。

## Full Benchmark

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/evaluation --judge-mode MOCK
```

```text
RAG_MODE=LIVE
JUDGE_MODE=MOCK
RAG_MODEL=glm-4-flash-250414
CLI exit code: 0
```

Report timestamp：2026-09-05T14:09:28.066575Z。Embedding为真实BAAI/bge-small-zh-v1.5，Chroma真实检索。
Runner耗时87456.50490000844ms（约87.46秒，不含建库初始化），60个真实生成请求均成功。

| 项目 | 结果 |
| --- | --- |
| Total | 60 |
| Success / Failure | 60 / 0 |
| Failure breakdown | {} |
| Evaluation success rate | 100% |
| Average latency | 1457.4682383332402ms |
| Retry count | 0 |
| Judge cache hits | 2（仅复用匹配的历史MOCK Judge） |
| Evaluated retrieval cases | 48，缺失0，排除不可回答12 |
| Recall@3 | 1.0 |
| Precision@3 | 0.34027777777777773 |
| MRR@3 | 0.9583333333333334 |
| Generation correctness / faithfulness / relevance / completeness | 全部0，MOCK占位 |
| Hallucination rate | 0，MOCK占位 |
| Refusal accuracy | 12/12=1.0，MOCK关键词探针 |
| Unassessed refusal cases | 0 |

Generation/Safety并非真实Judge评分：0分不代表真实质量差、0幻觉不代表没有幻觉，
12/12关键词观察不等于真实拒答准确率。报告quality_metrics_are_synthetic=true，每条reason明确标记MOCK。

完整汇总：

```json
{
  "retrieval": {
    "evaluated_retrieval_cases": 48,
    "excluded_unanswerable_cases": 12,
    "unavailable_retrieval_cases": 0,
    "recall_at_k": 1.0,
    "precision_at_k": 0.34027777777777773,
    "mrr": 0.9583333333333334
  },
  "generation": {
    "evaluated_judge_cases": 60,
    "avg_correctness": 0.0,
    "avg_faithfulness": 0.0,
    "avg_relevance": 0.0,
    "avg_completeness": 0.0
  },
  "safety": {
    "evaluated_judge_cases": 60,
    "hallucination_cases": 0,
    "hallucination_rate": 0.0,
    "unanswerable_cases": 12,
    "correct_refusal_cases": 12,
    "unassessed_refusal_cases": 0,
    "refusal_accuracy": 1.0
  },
  "engineering": {
    "total_cases": 60,
    "success_cases": 60,
    "failed_cases": 0,
    "evaluation_success_rate": 1.0,
    "average_latency": 1457.4682383332402,
    "judge_cache_hits": 2,
    "retry_count": 0,
    "failure_breakdown": {}
  }
}
```

最终文件：[latest_report.json](../artifacts/evaluation/latest_report.json)、
[case_results.json](../artifacts/evaluation/case_results.json)。本阶段快照保存在
[artifacts/sprint7/evaluation](../artifacts/sprint7/evaluation/latest_report.json)。
Sprint 6最终报告归档至artifacts/evaluation/sprint6，初次失败报告仍保留在attempt1。

一致性检查通过：两份JSON可Pydantic反序列化、case列表与Dataset的60个ID和顺序一致，
所有结果SUCCESS且有真实模型trace、累计请求数60、retry_count和聚合一致、输入哈希不变，
报告/探测记录中不含API Key。单次成功不保证未来外部服务持续可用。

## pytest 完整结果

命令：`.venv\Scripts\python -m pytest`。

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.4.1, pluggy-1.6.0
rootdir: <project>
configfile: pyproject.toml
testpaths: tests
plugins: allure-pytest-2.15.0, anyio-4.13.0, langsmith-0.7.30
collected 277 items

tests\test_embeddings.py .                                               [  0%]
tests\test_evaluation_cache.py .............                             [  5%]
tests\test_evaluation_runner.py .............................            [ 15%]
tests\test_judge_client.py ...............................               [ 26%]
tests\test_judge_live_validation.py ...                                  [ 27%]
tests\test_judge_models.py ............................                  [ 37%]
tests\test_judge_settings.py ..........                                  [ 41%]
tests\test_judge_smoke.py ...                                            [ 42%]
tests\test_judge_v2.py ..                                                [ 43%]
tests\test_loader.py ........................                            [ 51%]
tests\test_models.py ......................                              [ 59%]
tests\test_provider.py ...........................                       [ 69%]
tests\test_rag_client.py .......                                         [ 72%]
tests\test_retrieval_benchmark.py ....                                   [ 73%]
tests\test_retrieval_metrics.py ......................................   [ 87%]
tests\test_retriever.py ..................                               [ 93%]
tests\test_settings.py .................                                 [100%]

============================== warnings summary ===============================
..\python\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128
  <python-env>\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(f):

tests/test_retriever.py: 20 warnings
  <python-env>\Lib\site-packages\chromadb\api\models\CollectionCommon.py:155: DeprecationWarning: legacy embedding function config
    return load_collection_configuration_from_json(self._model.configuration_json)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 277 passed, 21 warnings in 8.39s =======================
```

全部单测使用mock SDK/Provider，无真实LLM调用。SDK相关原RAG测试迁至Provider边界，
不是删除其覆盖；Judge测试改为注入Provider，并保留评分/JSON/Schema/异常验证。
HTTP408/504回归测试先运行得到2 failed，确认分类遗漏，修正后完整套件通过。
21条警告来自既有Chroma依赖，未屏蔽。实际验证Python3.14，未单独运行Python3.10。

范围检查：src中实际chat.completions.create只剩Provider一处，DemoRAG/两个CLI/Judge均接入；
独立Judge smoke和十条live验证都走相同Judge客户端，没有遗漏的直接SDK生成入口。
Runner仍保留SDK异常类型识别以兼容已有调用方，但不负责SDK请求。

## 文件清单

新增：

- src/llm/__init__.py
- src/llm/provider.py
- src/llm/models.py
- src/llm/openai_compatible.py
- src/judge/live_validation.py
- tests/test_provider.py
- tests/provider_fakes.py
- tests/test_judge_live_validation.py
- docs/sprint7-validation.md

修改：

- config/settings.py、config/judge.py
- src/rag/client.py、src/rag/__main__.py
- src/dataset/models.py
- src/judge/client.py、src/judge/models.py
- src/evaluation/runner.py、src/evaluation/models.py、src/evaluation/aggregation.py
- tests/test_rag_client.py、tests/test_judge_client.py、tests/test_judge_v2.py
- tests/test_judge_smoke.py、tests/test_evaluation_runner.py、tests/test_settings.py
- .env.example、README.md
- docs/sprint6-validation.md（仅归档链接）
- 本地.env（RAG_MODEL；文件仍被Git忽略）

产物：artifacts/sprint7/provider_probe.json、live_judge_validation.json、evaluation两份快照；
artifacts/evaluation两份当前报告及sprint6归档。没有增加Baseline、Compare、Regression Gate、
Streamlit或Excel功能，没有改动Corpus或Dataset。

数据SHA-256：

```text
corpus: 989ef975f4b8ded581b74eb75624dc19358e015b1c5b6396471fa4334227cc2e
dataset: 7bf63862ff4730d5884f92c20eb63888f58407d3785c24c8d1750e1d4cc7ba3f
```
