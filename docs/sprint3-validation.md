# Sprint 3 验证记录

验证日期：2026-09-05。结论：`SPRINT 3 ACCEPTANCE: PASS`。

## 环境

Windows，Python 3.14.0，pytest 8.4.1。
本次使用项目 `.venv`，通过 `--system-site-packages` 复用本机已有 PyTorch、Chroma
和 OpenAI SDK，新增依赖安装在 `.venv` 内。模型与向量库放在已忽略的 `.cache/`。

| 组件 | 实际版本 |
| --- | --- |
| sentence-transformers | 5.7.0 |
| transformers | 5.16.1 |
| chromadb | 1.5.7 |
| openai | 2.31.0 |
| torch | 2.9.0 |

模型：`BAAI/bge-small-zh-v1.5`；CPU；归一化向量；查询加中文检索指令。
本次下载的模型 revision 为 `7999e1d3359715c523056ef9478215996d62a620`。
Chroma collection：`ecommerce_v1`，cosine 距离，36 条记录。

## 完整 pytest 输出

执行命令：`.venv\Scripts\python -m pytest`。

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.4.1, pluggy-1.6.0
rootdir: <project>
configfile: pyproject.toml
testpaths: tests
plugins: allure-pytest-2.15.0, anyio-4.13.0, langsmith-0.7.30
collected 76 items

tests\test_embeddings.py .                                               [  1%]
tests\test_loader.py ........................                            [ 32%]
tests\test_models.py ......................                              [ 61%]
tests\test_rag_client.py ........                                        [ 72%]
tests\test_retriever.py ..................                               [ 96%]
tests\test_settings.py ...                                               [100%]

============================== warnings summary ===============================
..\python\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128
  <python-env>\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(f):

tests/test_retriever.py: 20 warnings
  <python-env>\Lib\site-packages\chromadb\api\models\CollectionCommon.py:155: DeprecationWarning: legacy embedding function config
    return load_collection_configuration_from_json(self._model.configuration_json)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 76 passed, 21 warnings in 6.81s =======================
```

单元测试未下载模型，真实 Chroma 测试使用确定性向量替身，LLM 调用均为 mock。
上述警告来自 Chroma 依赖，未屏蔽。原有 Sprint 1–2 测试包含在本次完整测试中。

## 真实 retrieval-only smoke

执行命令：`.venv\Scripts\python -m src.rag.smoke`。
首次下载时为将辅助缓存留在项目内，设置 `HF_HOME` 为项目 `.cache/huggingface`，
并设置 `HF_HUB_DISABLE_XET=1` 使用 HTTP 下载。没有调用外部 LLM。
选取策略为每个主题第一条 answerable normal case，共六个主题，不按结果挑题。

```text
Indexed 36 chunks

case_id: normal_001 / order
query: 下单后一直没付款，订单多久会关闭？关闭后还能恢复吗？
expected relevant chunk_id: [order_policy_001]
top-3 retrieved chunk_id: [order_policy_001, coupon_policy_006, order_policy_004]

case_id: normal_006 / payment
query: 能用货到付款吗？支持哪些支付方式？
expected relevant chunk_id: [payment_policy_001]
top-3 retrieved chunk_id: [payment_policy_001, payment_policy_003, payment_policy_004]

case_id: normal_011 / refund
query: 普通商品签收5天、没使用且包装配件赠品都完整，可以无理由退货吗？
expected relevant chunk_id: [refund_policy_001]
top-3 retrieved chunk_id: [refund_policy_001, membership_policy_003, refund_policy_004]

case_id: normal_016 / logistics
query: 现货订单付款后多久发货？预售也一样吗？
expected relevant chunk_id: [logistics_policy_001]
top-3 retrieved chunk_id: [logistics_policy_001, refund_policy_003, payment_policy_006]

case_id: normal_021 / coupon
query: 满减券的门槛包含运费和不适用的商品吗？
expected relevant chunk_id: [coupon_policy_001]
top-3 retrieved chunk_id: [coupon_policy_001, coupon_policy_002, payment_policy_003]

case_id: normal_026 / membership
query: 买会员以后可以不受普通退货条件限制吗？
expected relevant chunk_id: [membership_policy_001]
top-3 retrieved chunk_id: [membership_policy_001, refund_policy_001, coupon_policy_002]
```

## 离线重开验证

首次下载和建库完成后，在独立进程设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，执行：

```powershell
.venv\Scripts\python -m src.rag retrieve --query "退货运费谁承担？" --top-k 2
```

成功复用缓存模型和持久化索引，退出码为0：

| rank | doc_id | chunk_id | score |
| --- | --- | --- | --- |
| 1 | refund_policy | refund_policy_004 | 0.6444593667984009 |
| 2 | membership_policy | membership_policy_004 | 0.5246431231498718 |

## 边界

未进行真实 LLM 调用，服务端连通性、模型兼容性及生成质量尚未验证；本阶段不要求。
Prompt 要求基于证据回答和拒绝编造，但不能证明生成一定正确。
Python 3.10+ 已声明，实际只在 Python 3.14.0 验证。依赖和模型使用版本范围/模型 ID，
本记录保存了实际版本及 revision，未来重新下载或升级后应重新验证。
未实现评测指标、Judge、Evaluation Runner、Baseline/Compare、UI 或 Excel Report。
