# Sprint 5 验证记录

结论：`SPRINT 5 ACCEPTANCE: PASS`。Live 状态：`LIVE JUDGE: NOT RUN`。

## 实现范围

独立 Judge 输入/输出模型、版本化 `judge_v1` Prompt、OpenAI-compatible client、
有限 JSON/Schema 修复重试、网络/API 分类异常及四条人工 smoke 样例。
未实现全 Dataset 生成评测、完整 Runner、Refusal Accuracy 聚合、Baseline/Compare、
Regression Gate、Streamlit 或 Excel。既有 Corpus、Dataset 和 RAG 检索器未修改。

Correctness 只对照标准答案；Faithfulness 只对照检索上下文。相关性衡量回应问题的程度，
完整性衡量要点覆盖。四项独立取0–5整数。幻觉必须列出不受上下文支持的具体断言，
不可回答输入仍交给裁判检查拒答/编造，而不排除。

## 重试及追溯

默认最多3次总调用，可配置1–5次；仅 JSON/Schema 失败重试。SDK 自动重试为0，
超时、连接、认证、权限、限流、请求错误和其他 API 错误立即分类抛出。
结果及失败详情均保留模型、Prompt 版本、输出模式、temperature=0、耗时、原文和历史尝试。
错误不会转换为正常评分。请求超时默认60秒，不等于整个含重试操作的硬性总时限。

默认通用 JSON 模式；配置 `JUDGE_RESPONSE_FORMAT=json_schema` 可使用服务端结构化输出，
两种模式均执行严格本地校验。未验证任何真实服务端的模式兼容性。

## 完整 pytest 输出

命令：`.venv\Scripts\python -m pytest`。

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.4.1, pluggy-1.6.0
rootdir: <project>
configfile: pyproject.toml
testpaths: tests
plugins: allure-pytest-2.15.0, anyio-4.13.0, langsmith-0.7.30
collected 187 items

tests\test_embeddings.py .                                               [  0%]
tests\test_judge_client.py ............................                  [ 15%]
tests\test_judge_models.py ............................                  [ 30%]
tests\test_judge_settings.py ..........                                  [ 35%]
tests\test_judge_smoke.py ...                                            [ 37%]
tests\test_loader.py ........................                            [ 50%]
tests\test_models.py ......................                              [ 62%]
tests\test_rag_client.py ........                                        [ 66%]
tests\test_retrieval_benchmark.py ....                                   [ 68%]
tests\test_retrieval_metrics.py ......................................   [ 88%]
tests\test_retriever.py ..................                               [ 98%]
tests\test_settings.py ...                                               [100%]

============================== warnings summary ===============================
..\python\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128
  <python-env>\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(f):

tests/test_retriever.py: 20 warnings
  <python-env>\Lib\site-packages\chromadb\api\models\CollectionCommon.py:155: DeprecationWarning: legacy embedding function config
    return load_collection_configuration_from_json(self._model.configuration_json)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 187 passed, 21 warnings in 8.81s =======================
```

新增69项测试覆盖模型、独立维度、幻觉约束、不可回答、错误 JSON/Schema、分数类型/范围、
有限修复、API 分类、配置、元数据及 smoke 跳过行为。Judge client 测试还禁止真实 HTTP
send。原有118项测试也通过。警告来自已有 Chroma 依赖，未屏蔽。

## Live Judge

只检查变量是否有值，未打印密钥。三个必需变量均缺失：

```json
{
  "JUDGE_API_KEY": false,
  "JUDGE_BASE_URL": false,
  "JUDGE_MODEL": false
}
```

已执行 `.venv\Scripts\python -m src.judge.smoke`，得到：

```text
LIVE JUDGE: NOT RUN
Missing Judge configuration: JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL
```

[live_judge_smoke.json](../artifacts/sprint5/live_judge_smoke.json) 保存四个输入样例，
`status="NOT RUN"`、`results=[]`。没有真实 Judge JSON 分数，也未把 mock 输出当成真实结果。
四个样例分别为：

1. 标准答案、上下文和回答都是30分钟关闭订单。
2. 标准答案30分钟，上下文和回答60分钟，用于区分 correctness 与 faithfulness。
3. 回答在30分钟关闭之外添加上下文不存在的100元赔付。
4. 不可回答的会员价格问题，回答编造每期9.9元。

这些样例独立于原始数据文件。配置可用 API 后可重跑相同命令，每个样例都保存真实结果或错误。

## 数据完整性及风险

开始和结束时 SHA-256 一致：

```text
data/corpus/ecommerce_v1.json
989ef975f4b8ded581b74eb75624dc19358e015b1c5b6396471fa4334227cc2e

data/datasets/ecommerce_eval_v1.json
7bf63862ff4730d5884f92c20eb63888f58407d3785c24c8d1750e1d4cc7ba3f
```

本阶段实现无未完成项；按任务允许的验收规则，live 缺配置不阻塞 Sprint 通过。
真实 API 连通性、结构化输出兼容性和裁判语义质量尚未验证。mock 测试证明接口和校验行为，
不能证明真实裁判准确；temperature=0 不能保证完全可重复或免受提示注入影响。
实际测试环境是 Python 3.14，未单独验证 Python 3.10。无新增依赖。
