# RAGEval 详细使用说明

RAG 知识库质量自动化评测框架，用于建立可复用的评测数据契约和后续评测流程。
本项目不是 RAG 聊天应用。

## 当前范围：Sprint 1–9

已提供 Python 工程骨架、环境配置、Pydantic 数据模型、36 个电商客服语料分块、
60 条评测 case、JSON 加载与交叉校验，以及作为被测系统的本地 Demo RAG。
Demo RAG 使用 SentenceTransformer + Chroma 检索，通过独立的 LLMProvider 生成答案。
已实现基于 chunk_id 的 Recall@K、Precision@K、RR、MRR 聚合及 retrieval-only benchmark。
已实现独立 LLM-as-Judge、版本化评分 Prompt、严格 JSON 校验及有限输出修复重试。
已实现逐 case 完整 Evaluation Runner、拒答统计、Judge 缓存及 JSON 报告。
Sprint 7 增加共享 Provider、可取消请求超时、分类重试与完整重试追溯。
Sprint 8 增加 Baseline 快照、11项指标对比、YAML Regression Gate 和离线模拟版本实验。
Sprint 9 增加只读 Streamlit Evaluation Dashboard 与 JSON/Excel 导出，不改变核心评测逻辑。

## 安装与测试

要求 Python 3.10+。在项目根目录执行以下命令（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
```

单元测试使用固定向量替身和临时真实 Chroma 库，不下载 Embedding 模型，LLM 调用
全部使用 mock。指标测试使用手工构造数据，不调用 Embedding 或 LLM。
真实检索验证由下方独立 smoke / retrieval benchmark 命令执行。

macOS/Linux 可使用 `source .venv/bin/activate` 激活环境，使用
`cp .env.example .env` 复制配置。可选执行 `python -m pip install -e .`
进行可编辑安装。

## 目录

```text
RAGEval/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── app.py
├── .streamlit/config.toml
├── config/
│   ├── __init__.py
│   ├── judge.py
│   ├── regression_gate.yaml
│   ├── regression_gate_demo.yaml
│   └── settings.py
├── data/
│   ├── .gitkeep
│   ├── README.md
│   ├── corpus/ecommerce_v1.json
│   └── datasets/ecommerce_eval_v1.json
├── artifacts/sprint4/retrieval_benchmark.json
├── artifacts/sprint5/live_judge_smoke.json
├── artifacts/evaluation/
│   ├── latest_report.json
│   └── case_results.json
├── docs/
│   ├── sprint3-validation.md
│   ├── sprint4-validation.md
│   ├── sprint5-validation.md
│   ├── sprint6-validation.md
│   ├── sprint7-validation.md
│   ├── sprint8-validation.md
│   ├── sprint9-validation.md
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── models.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   ├── client.py
│   │   └── smoke.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py
│   │   ├── models.py
│   │   └── openai_compatible.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── retrieval.py
│   │   └── benchmark.py
│   ├── judge/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── prompts.py
│   │   ├── client.py
│   │   ├── smoke.py
│   │   └── live_validation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── aggregation.py
│   │   ├── cache.py
│   │   ├── mock_judge.py
│   │   └── runner.py
│   ├── comparison/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── snapshot.py
│   │   ├── engine.py
│   │   ├── gate.py
│   │   └── demo.py
│   └── report/
│       ├── __init__.py
│       ├── __main__.py
│       ├── data.py
│       ├── dashboard.py
│       └── export.py
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_loader.py
    ├── test_embeddings.py
    ├── test_retriever.py
    ├── test_rag_client.py
    ├── test_provider.py
    ├── provider_fakes.py
    ├── test_judge_live_validation.py
    ├── test_retrieval_metrics.py
    ├── test_retrieval_benchmark.py
    ├── test_judge_models.py
    ├── test_judge_client.py
    ├── test_judge_settings.py
    ├── test_judge_smoke.py
    ├── test_judge_v2.py
    ├── test_evaluation_cache.py
    ├── test_evaluation_runner.py
    ├── test_comparison.py
    ├── test_report_export.py
    ├── test_dashboard.py
    └── test_settings.py
```

`data/` 是数据目录，不是 Python 包。所有 Python 包均有 `__init__.py`。

## 配置

通过 `config.settings.load_settings()` 显式加载配置：进程环境变量优先于
`.env`，未配置时使用默认值。默认读取当前工作目录的 `.env`，也可传入具体路径；
传入 `None` 时仅读取进程环境变量。相对数据路径以当前工作目录为基准。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `RAGEVAL_DATA_DIR` | `data` | 数据存放目录 |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 模型 ID 或本地模型目录 |
| `RAG_EMBEDDING_DEVICE` | `cpu` | SentenceTransformer 运行设备 |
| `RAG_EMBEDDING_CACHE_DIR` | `.cache/huggingface` | Embedding 模型缓存 |
| `RAG_EMBEDDING_QUERY_INSTRUCTION` | `为这个句子生成表示以用于检索相关文章：` | 只加在查询前的提示，换模型时可设为空 |
| `RAG_CHROMA_DIR` | `.cache/chroma` | Chroma 本地持久化目录 |
| `RAG_CHROMA_COLLECTION` | `ecommerce_v1` | 当前语料索引的 collection 名称 |
| `RAG_API_KEY` | 空 | 生成服务密钥，仅 `run()` 必填 |
| `RAG_BASE_URL` | 空 | OpenAI-compatible API 基础 URL，仅 `run()` 必填，通常含 `/v1` |
| `RAG_MODEL` | 空 | 服务端提供的生成模型名称，仅 `run()` 必填 |

进程环境变量优先于 `.env`。建库、检索和 retrieval smoke 不需要 API Key；生成前
检查上述三个 LLM 变量，缺少或为空时抛出包含变量名的 `ConfigurationError`。
无硬编码密钥或默认生成供应商，配置对象的 repr 不显示密钥。
`.env`、`.cache/`、常见模型和 Chroma 缓存目录已被 Git 忽略。若自定义缓存到其他
项目内目录，应同步更新 `.gitignore`，不要提交模型权重和向量库。

## 核心数据契约

统一从 `src.dataset` 导入，基于 Pydantic 2。模型拒绝未知字段。
标识符、问题、语料内容和分类去除首尾空白后不能为空。

| 模型 | 字段及含义 |
| --- | --- |
| `CorpusDocument` | `doc_id: str` 文档 ID；`chunk_id: str` 分块 ID；`content: str` 内容；`metadata: dict[str, Any]` 元数据，默认独立空字典 |
| `EvalCase` | `id: str` 用例 ID；`query: str` 问题；`expected_answer: str` 标准答案；`relevant_doc_ids: list[str]` 相关 chunk_id 列表；`answerable: bool` 是否可回答；`category: str` 分类；`variant_group: str \| None` 变体分组，默认 `None` |
| `RetrievedDocument` | `doc_id: str` 文档 ID；`chunk_id: str` 分块 ID（必填）；`content: str` 内容；`score: float` 有限数值且越高越相关；`rank: int` 从 1 开始的严格整数排名 |
| `RAGResult` | `query: str` 问题；`retrieved_docs: list[RetrievedDocument]` 检索结果；`rag_answer: str` 生成答案；`latency_ms: float` 非负且有限的总耗时（毫秒） |

标准答案和生成答案允许为空字符串，相关文档和检索结果允许为空列表，以表达
不可回答或无检索命中的情况；这些字段仍需显式提供。分数不限制在 0–1 区间，
本地检索具体使用下述 cosine similarity。`doc_id` 可对应多个 `chunk_id`，模型不做跨记录
唯一性或标签引用校验，这些检查由 Loader 执行。元数据需要导出 JSON 时，应使用
可 JSON 序列化的值。

测试覆盖模型创建、默认值隔离、可选分组、空结果、嵌套 JSON 往返、非法字段、
排名和数值约束，以及环境配置的默认值与优先级。

## Corpus 与 Dataset 加载

从项目根目录运行以下 Python 示例：

```python
from config.settings import load_settings
from src.dataset import load_corpus, load_dataset

data_dir = load_settings().data_dir
corpus = load_corpus(data_dir / "corpus" / "ecommerce_v1.json")
cases = load_dataset(data_dir / "datasets" / "ecommerce_eval_v1.json", corpus)
print(len(corpus), len(cases))  # 36 60
```

两个 Loader 均接受 JSON 文件路径，返回 Pydantic 模型列表；文件须为 UTF-8
（支持 BOM），顶层为数组。加载评测集时必须提供已加载的 corpus，避免跳过引用校验。
错误统一抛出 `DataValidationError`（继承 `ValueError`），包含文件路径；JSON
语法错误附行列号，结构错误附 Pydantic 字段位置，语义错误附 case/chunk ID。

校验规则：

- Pydantic 校验每条记录的字段、类型和既有模型约束。
- 同一 corpus 的 `chunk_id` 全局唯一，允许同一 `doc_id` 包含多个 chunk。
- 同一 dataset 的 case `id` 唯一。
- `answerable=true` 必须有非空 `relevant_doc_ids`，且每个值均为 corpus 中存在的 `chunk_id`。
- `answerable=false` 必须有空 `relevant_doc_ids`，适用于所有 category。
- `category=unanswerable` 必须同时满足 `answerable=false`。
- `variant_group` 沿用可选字段，Loader 正常解析；v1 数据测试另行验证六组同义问题的组内一致性。

字段名 `relevant_doc_ids` 保持兼容，但其语义从 Sprint 1 的文档级说明改为分块级
引用；Sprint 3 为 `RetrievedDocument` 增加必填 `chunk_id`，与这些标注直接对齐。
旧的检索结果 JSON 如缺少 `chunk_id`，现在会明确校验失败，需要补齐真实分块 ID。
详细主题、分类数量、标注约定和合成数据边界见 [data/README.md](../data/README.md)。

## 本地 Demo RAG

```text
Corpus JSON → Loader → SentenceTransformer（文档向量归一化）
                         ↓
                Chroma 持久化 cosine 索引
                         ↑
query → 查询指令 + SentenceTransformer → Top-K chunk
                         ↓
              JSON 上下文 + 约束 Prompt
                         ↓
              OpenAI-compatible Chat Completions
                         ↓
                     RAGResult
```

`RAGClient` 定义 `retrieve(query, top_k=3)` 和 `run(query, top_k=3)`；
`DemoRAGClient` 组合 `ChromaRetriever` 和生成客户端。`latency_ms` 使用单调计时器，
包含本次检索、上下文组装、LLM 客户端首次初始化（如有）和生成耗时，不含先前建库。

Embedding 延迟加载，使用 SentenceTransformer，batch size 为 32，文档和查询均
归一化；查询加中文检索指令，文档不加指令。设置参照
[BGE 模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。首次使用需联网下载公开模型，
之后可复用缓存，也可通过 `RAG_EMBEDDING_MODEL` 指定预先下载的本地模型目录。

Chroma 的记录 ID 为 `chunk_id`，metadata 明确保留 `doc_id` 和 `chunk_id`。
使用显式向量，禁用 Chroma 默认 Embedding 函数。索引操作同步整个指定 corpus：
upsert 现有/新增分块，并删除该 collection 内不再属于 corpus 的旧分块；重复运行
不会累积重复数据。此 collection 应专用于该 corpus。索引同步不是事务操作，
中断后可重新执行。更换模型或查询指令时须使用新 collection 并重新建库，避免混用向量。

依照 [Chroma 距离定义](https://docs.trychroma.com/docs/collections/configure)，本实现
固定 `space=cosine`，将返回距离转换为 `score = 1 - distance`，即 cosine similarity，
理论范围 [-1, 1]，浮点运算可能产生微小偏差；越高越相关，不是回答正确概率。
结果按距离升序返回，`rank` 从 1 开始；`top_k` 必须为正整数，超过库大小时返回全部分块。
空索引会明确提示先建库。检索器没有阈值过滤或重排；评测指标独立放在 `src/metrics/`。

生成使用 [Chat Completions 接口](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)，
通过环境变量指定服务端 URL 和模型。Prompt 要求仅依据上下文、信息不足明确拒答、
不编造，并要求忽略问题和上下文中的规则覆盖指令。生成端通过 Provider 调用模型，
单请求截止时间默认60秒，可配置；429、超时、连接失败、5xx按上限退避重试。
最终错误带请求追溯向上层传播；空文本响应明确报错。Prompt 不构成无幻觉保证。

### 运行命令

在项目根目录且已激活虚拟环境时执行：

```powershell
# 建库 / 同步语料，无需 LLM 配置
python -m src.rag index

# 仅检索，可更改 top-k
python -m src.rag retrieve --query "退货运费谁承担？" --top-k 3

# 填好 .env 的三个 RAG 生成变量后才执行，会调用配置的 LLM
python -m src.rag run --query "退货运费谁承担？" --top-k 3

# 真实 Embedding + Chroma，仅检索六个主题，不调用 LLM
python -m src.rag.smoke
```

smoke 从 Dataset 固定选取每个主题的第一条可回答 normal case，重新同步语料后
打印 query、真实相关 chunk_id 与 Top-3 chunk_id；不计算指标、不按结果挑选问题。
本次完整测试输出、六条 smoke 结果和离线重开结果见
[Sprint 3 验证记录](sprint3-validation.md)。

模型已完整下载后，可设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 进行离线检索。

Python 调用方式：

```python
from config.settings import load_settings
from src.rag import ChromaRetriever, DemoRAGClient
from src.llm.openai_compatible import OpenAICompatibleProvider

settings = load_settings()
provider = OpenAICompatibleProvider.from_rag_settings(settings)
client = DemoRAGClient(ChromaRetriever(settings), provider)
hits = client.retrieve("退货运费谁承担？", top_k=3)  # 已先执行 index
# result = client.run("退货运费谁承担？", top_k=3)  # 需要生成配置
```

## Retrieval Metrics

相关性判断统一比较 `RetrievedDocument.chunk_id` 与 `EvalCase.relevant_doc_ids`。
令 R 为去重后的相关 chunk 集合，T 为返回列表的前 K 条记录，H 为 T 中与 R 匹配的
唯一 chunk 数：

| 指标 | 定义 |
| --- | --- |
| Recall@K | H / len(R) |
| Precision@K | H / len(T)，分母为实际返回条数，不固定为 K |
| RR | 1 / 首个相关 chunk 的 rank；当前排名内无命中则为0 |
| MRR | 所有参与计算 case 的 RR 算术平均值 |

`reciprocal_rank()` 检查传入的完整排名；`evaluate_retrieval(..., k)` 只检查前 K 条，
因此该结果聚合得到的 `mrr` 是 Top-K 截断下的 MRR（即 MRR@K），不使用 K 之后的命中。
平均 Recall 和 Precision 使用逐 case 算术平均（macro average），不按相关分块数加权。

边界约定：

- 可回答 case 的检索结果为空：Recall、Precision、RR 均为0，仍参与聚合。
- K 大于返回条数：只计算实际返回条数，不补空位。
- 多个相关 chunk：Recall 分母为全部不同相关 chunk 数，不只看首个命中。
- 重复返回的 chunk：先取前 K 条，重复项保留位置但仅计一次命中，不去重后补取 K 之外的结果。
  例如 `[a, a, b]`、相关集合 `{a, b}`、K=2，得到 Recall=0.5、Precision=0.5、RR=1。
- 重复相关标签按集合处理，结果中的 `relevant_chunk_ids` 去重并保留首次出现顺序。
- `answerable=false`：三个指标均为 `None`，聚合排除并记录数量；即使返回了 chunk 也不计为0分。
- `answerable=true` 但相关标签为空：抛出明确 `ValueError`；低层指标函数遇空相关标签同样报错。
- 全部 case 被排除或输入结果为空：均值与 MRR 为 `None`，不人为设为0。
- K 必须为正整数，0、负数、浮点数和布尔值均报错。
- 输入检索列表必须按 rank 排序且从1连续编号，矛盾排名会报错，不静默改写 RR 分母。
- 聚合拒绝不同 K 混合或重复 case ID，避免重复计权。

### 结果结构与使用

`RetrievalMetricResult` 包含 `case_id`、`k`、`answerable`、`recall_at_k`、
`precision_at_k`、`rr`、`retrieved_chunk_ids`、`relevant_chunk_ids`。
`retrieved_chunk_ids` 保存实际参与计算的前 K 条，保留重复项便于核查。
不可回答记录使用 `answerable=false` 和空相关标签，三个指标在 JSON 中为 `null`。

`RetrievalMetricSummary` 包含 `k`、`evaluated_cases`、`excluded_unanswerable_cases`、
`mean_recall_at_k`、`mean_precision_at_k`、`mrr`。两种结果均为 Pydantic 模型，
指标值必须在 [0,1] 内且有限。

```python
from src.metrics import evaluate_retrieval, aggregate_retrieval_metrics

# cases 来自 load_dataset；client 为已建库的 DemoRAGClient。
results = [
    evaluate_retrieval(case, client.retrieve(case.query, 3) if case.answerable else [], k=3)
    for case in cases
]
summary = aggregate_retrieval_metrics(results, k=3)
print(summary.model_dump())
```

### 真实检索 Benchmark

```powershell
# 本机已缓存完整模型时可离线执行；首次下载时不要设置这两个离线变量。
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m src.metrics.benchmark
```

该命令加载原有 corpus 和 dataset，同步索引，对所有可回答 case 各检索一次 Top-5，
用同一排名的前缀分别计算 K=1/3/5。不可回答 case 不调用检索器。
它不调用 LLM，不加载标准答案作为检索上下文，也不修改数据或检索参数。

默认输出 `artifacts/sprint4/retrieval_benchmark.json`，可通过 `--output` 指定其他输出文件。
文件保存运行配置、依赖版本、输入 SHA-256、全部逐 case 指标、排除 ID、各 K 聚合和失败明细；
运行前后比较输入哈希，变化时直接报错。
失败定义为可回答 case 的 Recall@K < 1，包含部分命中；`no_hit_cases` 另行统计完全未命中。
不要求 Precision@K=1 才算成功，因为多返回的候选通常多于相关标签数量。

本次 48 条参与、12 条排除：

| K | mean Recall@K | mean Precision@K | MRR（Top-K 截断） | 未覆盖全部相关分块 | 完全未命中 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.906250 | 0.916667 | 0.916667 | 5 | 4 |
| 3 | 1.000000 | 0.340278 | 0.958333 | 0 | 0 |
| 5 | 1.000000 | 0.204167 | 0.958333 | 0 | 0 |

完整测试输出、失败案例与结果边界见 [Sprint 4 验证记录](sprint4-validation.md)。

## LLM-as-Judge

`src/judge/` 只评测传入的一条答案，不调用 RAG、不读取整个 Dataset，也不聚合拒答指标。
Judge 与 RAG 生成服务使用各自的环境变量，不自动借用 RAG 的密钥或模型。

### 评分维度

| 字段 | 含义 |
| --- | --- |
| answer_correctness | 是否符合 expected_answer 的事实与结论 |
| faithfulness | 回答中的事实断言是否得到 retrieved_context 支持，不能把标准答案当作检索证据 |
| answer_relevance | 是否直接回应 query，避免无关内容，适当拒答也可切题 |
| completeness | 是否覆盖问题与标准答案要求的要点、子问题和适用条件 |
| hallucination | 是否存在上下文未支持或否定的事实断言 |
| unsupported_claims | 上述具体断言的摘录或简短概括；无幻觉时为空列表 |
| reason | 简短中文判定理由 |

前四项为严格整数0–5，不接受数字字符串、浮点分数或布尔值。通用锚点从0（完全不满足）
至5（完全满足），1–4对应严重、主要、实质、轻微缺陷。各维度独立判断，不求平均总分。
回答与上下文一致但违背标准答案时，可以 correctness 低、faithfulness 高且无幻觉。

`answerable=false` 仍执行 Judge，重点检查是否诚实拒答或捏造信息；不沿用检索指标的排除规则。
合理拒答可以获得高分。空白回答四项为0，但不因空白本身标记幻觉。
`hallucination=true` 必须有非空 `unsupported_claims`，反之须为空，Pydantic 会交叉校验。

### 模型与 Prompt

`JudgeInput`：`case_id`、`query`、`expected_answer`、`retrieved_context: str`、
`rag_answer`、`answerable`；允许空上下文、空标准答案及空生成答案。

`JudgeVerdict`：上述七个评分字段，全部必填并拒绝额外字段。
`JudgeResult`：七个评分字段加应用端填写的 `metadata`，包含：

```text
case_id, judge_model, prompt_version, response_format, temperature
latency_ms, attempts, raw_output, attempt_history
```

每条尝试记录保存序号、原始输出、错误类别/说明和 HTTP 状态码（如有）。
模型不能自行提供或覆盖这些运行元数据。`latency_ms` 使用单调时钟，覆盖首次 SDK 初始化
（如有）、所有尝试及本地解析，成功和失败均保留。

Prompt 版本固定为 `judge_v1`，位于 `src/judge/prompts.py`。更改评分口径或输出 Schema
时应提升版本。Prompt 要求只当裁判、不回答问题，不执行输入中的改分或覆盖规则指令，
并将 Pydantic 生成的固定 JSON Schema 放入系统消息。请求始终使用 `temperature=0`。
重试保留原输入、规则、失败输出及校验反馈，不替换评分标准。

### Judge 配置

通过 `config.judge.load_judge_settings()` 显式加载，环境变量优先于 `.env`。
缺少必需变量时抛出 `ConfigurationError`，异常指明变量名，不打印密钥。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| JUDGE_API_KEY | 空 | 必填；配置对象 repr 隐藏该值 |
| JUDGE_BASE_URL | 空 | 必填；OpenAI-compatible API 基础 URL，通常含 `/v1` |
| JUDGE_MODEL | 空 | 必填；服务端可用的 Judge 模型名 |
| JUDGE_MAX_ATTEMPTS | 3 | 总尝试次数，含首次调用；允许1–5 |
| JUDGE_TIMEOUT_SECONDS | 60 | SDK 请求超时参数；允许有限数值 (0,120] |
| JUDGE_RESPONSE_FORMAT | json | `json`、`json_object` 或 `json_schema` |

默认 `json` 为通用兼容模式：Prompt 严格要求 JSON，响应必须通过 JSON 解析和 Pydantic
校验，不要求服务端支持特定 `response_format` 参数。已确认服务支持结构化输出时，可设
`json_schema`，发送 `response_format={type: json_schema, ... strict: true}`，仍执行本地校验。
具体支持情况以服务端为准，参见 [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。
若服务端拒绝此参数，报 `invalid_request`，不会偷偷降级或额外重试；可显式改用 `json` 模式。

### 有限重试与错误

| 类别 | 行为 |
| --- | --- |
| invalid_json | 非法 JSON、代码围栏、空输出、重复键或非标准 NaN/Infinity：在总次数内重试 |
| invalid_schema | 缺字段、额外字段、越界/错误类型分数或幻觉字段矛盾：在总次数内重试 |
| timeout / connection | 在剩余请求预算内退避重试，耗尽后抛出 JudgeAPIError |
| authentication / permission | 认证或权限失败，立即抛出 JudgeAPIError |
| rate_limit / server | 429或5xx，在剩余请求预算内退避重试 |
| invalid_request / model_not_found / quota | 请求无效、模型不存在、已知额度耗尽，立即抛出 JudgeAPIError |
| api_response / api_error | SDK 响应结构异常或其他 API 异常，立即抛出 JudgeAPIError |

SDK 自动重试设置为0，网络重试和JSON修复共用 JUDGE_MAX_ATTEMPTS 总请求预算；不会相乘。
输出校验次数耗尽抛出 `JudgeOutputError`，终止于网络/API错误则抛出 `JudgeAPIError`。
两种异常均继承 `JudgeError`，其 `details` 为可 JSON 序列化的 `JudgeFailure`：
`category`、`message`、`metadata`。不会将失败伪装成全0评分。
API 错误记录类型和状态码，不持久化供应商错误 body；之前的解析失败原文仍保留在 history。
每次请求同时设置 SDK 超时与 asyncio.wait_for 截止时间；超时取消异步请求。
这是单请求截止时间，整个 Judge 操作还包含有限重试和JSON修复。

### 调用及 Live Smoke

```python
from src.judge import JudgeInput, LLMJudgeClient, JudgeError

# 需先配置三个 JUDGE 变量；不需要 RAG 生成配置。
client = LLMJudgeClient()
sample = JudgeInput(
    case_id="manual-1", query="多久关闭未支付订单？",
    expected_answer="30分钟未支付自动关闭。",
    retrieved_context="订单提交后30分钟未支付自动关闭。",
    rag_answer="30分钟未支付会自动关闭。", answerable=True,
)
try:
    result = client.judge(sample)
    print(result.model_dump_json(indent=2))
except JudgeError as exc:
    print(exc.details.model_dump_json(indent=2))
```

```powershell
python -m src.judge.smoke
```

该命令仅使用四条手工样例：正确且有依据、上下文与标准答案冲突、明显幻觉、不可回答但编造。
样例存在于 smoke 模块内，不修改 Corpus 或 Dataset。配置齐全时会真实调用 Judge；
配置缺失时打印 `LIVE JUDGE: NOT RUN`，输出保留样例但 `results=[]`，不伪造 Judge JSON。
配置齐全但 API/解析失败时保存错误记录并以退出码1结束，不能当作成功运行。
默认记录为 `artifacts/sprint5/live_judge_smoke.json`。

本次环境未配置 Judge API，live 未运行；单元测试全部使用 mock，不证明真实裁判评分可靠。
`temperature=0` 也不保证跨运行完全一致。完整结果见 [Sprint 5 验证记录](sprint5-validation.md)。

## Evaluation Runner

```text
Load Corpus + Dataset（结构、引用、唯一性校验）
  → 每个 case 调用 DemoRAGClient.run（包括不可回答 case）
  → 可回答 case 计算检索指标；不可回答 case 跳过检索指标
  → JudgeInput → Judge 缓存 → Judge v2
  → 不可回答 case 判定明确拒答且无幻觉
  → 按阶段聚合 → EvaluationReport → JSON 文件
```

CLI 会在评测前同步当前 corpus 到 Chroma。Runner 顺序处理每条记录，单条失败不阻止
后续 case；全局数据或服务配置错误则在开始前明确报错。不会传标准答案给 RAG 生成端。
检索指标阶段失败但已有 RAGResult 时，仍尝试 Judge 并保留两阶段的独立结果。

### 调用

```powershell
# 真实生成 + 真实 Judge，需要 RAG_* 和 JUDGE_* 两组 API 配置
python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/evaluation

# 真实生成 + 明确标记的 mock Judge；仍然必须有有效 RAG 配置
python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/evaluation --judge-mode MOCK
```

`--output` 是输出目录。`--judge-mode` 默认读取 `JUDGE_MODE`，未配置时为 `LIVE`。
mock 只能显式启用，API 失败不会自动降级为 mock。退出码0表示所有 case 成功，存在失败时
保存完整报告后退出码1。模型已缓存时，可设置 Hugging Face 离线变量避免再次联网下载，
它们不会禁止 RAG/Judge 的 API 请求。

```python
from src.evaluation.runner import evaluate_dataset

report = evaluate_dataset("data/datasets/ecommerce_eval_v1.json", top_k=3)
```

需要注入已有客户端时使用 `EvaluationRunner(...).evaluate_dataset(dataset_path, top_k)`。
单 case API 为 `evaluate_case(case, top_k)`。RAG 检索阶段异常包装为 `RAGRetrievalError`，
Runner 因而能区分检索失败与生成失败，并保留原始异常类型。

### 结果与失败

`CaseEvaluationResult` 保存 `case_id`、`query`、`category`、`answerable`、`rag_result`、
`retrieval_metrics`、`judge_result`、`refusal_correct`、`status`、`error_type`、
`errors`、`latency_ms`、`judge_cache_hit`、`cache_warning`。

状态为 `SUCCESS`、`RAG_ERROR`、`RETRIEVAL_ERROR`、`JUDGE_ERROR` 或 `TIMEOUT`。
有多阶段错误时，主状态和 error_type 取第一个错误，所有错误仍保存在 errors；其中 Judge
异常还保留其原始结构化详情。timeout 来自底层客户端超时，不创建无法取消的线程伪装硬超时。
单 case latency 使用单调时钟，包含本次 RAG、指标、Judge/缓存与拒答判定，不把缓存记录中
历史 Judge latency 当成本次耗时。平均 latency 单位为毫秒，包含失败 case。

`latest_report.json` 是完整 `EvaluationReport`，包含全部 case 结果与聚合；
`case_results.json` 单独保存同一结果数组以便检查。单个文件使用临时文件替换，避免半写 JSON；
两个文件不构成跨文件事务，完整 latest_report 是权威报告。保存失败会显式报错。
元数据包含 UTC timestamp、内容哈希形式的 dataset version、dataset/corpus SHA-256、
embedding model、RAG model、judge model、judge prompt version、top_k、运行模式和耗时。
运行前后比较数据哈希，防止评测中数据发生变化。

### 聚合口径

| 分组 | 指标与分母 |
| --- | --- |
| Retrieval | 只对 answerable=true 且检索指标阶段成功的 case，计算 macro Recall@K、Precision@K、MRR@K；记录 evaluated_retrieval_cases、excluded_unanswerable_cases 和 unavailable_retrieval_cases |
| Generation | 只对拿到有效 JudgeResult 的 case，计算四项平均分；记录 evaluated_judge_cases |
| Hallucination | Judge 判为幻觉的数量 / evaluated_judge_cases |
| Refusal | 正确拒答数量 / **全部** answerable=false case 数，包含 RAG/Judge 失败的不可回答 case |
| Engineering | total、success、failed、success/total、全部 case 平均 latency、Judge cache hit 数量 |

分母为0时输出 null。失败不会填入虚假的 Judge 0分，也不会当作检索0分；对应阶段缺失数量
明确记录。拒答准确率按任务约定保留全部不可回答 case 为分母，尚未判定的 case 计入
unassessed_refusal_cases，不计入正确拒答分子。

### Judge v2 拒答观察

独立 Judge 保持默认 `judge_v1`，Runner 显式使用 `judge_v2`。v2 增加必填布尔值
`refusal_detected`，要求 Judge 观察回答是否明确说明信息不足/无法回答，不能从低评分、
空白回答、仅建议联系客服或没有事实断言推导。v1 历史结果的该值为 null，Runner 不接受其
作为有效 v2 结果。原有四项评分与幻觉规则不变。

对于不可回答 case：`refusal_correct = refusal_detected and not hallucination`。
若先拒答又编造细节，仍不是正确拒答；不能仅靠“没有幻觉”认定拒答。

### Judge Cache

默认目录 `.cache/judge_evaluation/` 已被 Git 忽略。只缓存成功 JudgeResult，不缓存生成、
超时或失败判定。key 为排序后的 JSON 的 SHA-256，包含完整 JudgeInput（含 case ID、
query、expected answer、retrieved context、rag answer、answerable），还包含 Judge model、
Prompt version、完整系统 Prompt/Schema、模式 LIVE/MOCK、API base URL、输出格式及 temperature。
这些内容任一变化都会失效；MOCK 与 LIVE 绝不共用缓存。不把 API Key 放进 key 或缓存文件。

每次评测仍调用 RAG，只有答案和上下文等完全匹配才复用 Judge。缓存损坏、旧 Schema 或
元数据不匹配会重新 Judge，并在 case 中记录 warning。缓存写入失败不丢弃已经成功的评分。
返回缓存时保留原始 Judge 追溯信息，并通过 judge_cache_hit 明确标记为历史结果复用。

### Smoke 模式的边界

`JUDGE_MODE=MOCK` 的 Judge 四项分数均为0占位，幻觉标记为 false 占位；拒答字段只通过
“无法确定/无法回答/信息不足/未提供”关键词探针生成，不能当成真实裁判结果。
报告同时标记 `quality_metrics_are_synthetic=true`，每条 mock reason 也明确说明。
此模式可以验证真实生成、报告、缓存和聚合链路，但 Judge 均分、幻觉率和拒答准确率均不构成
真实质量测量；检索指标仍来自实际返回的 chunk。

### Sprint 6 历史记录

以下为Sprint 6历史结果，报告已归档至artifacts/evaluation/sprint6。
Sprint 6 的运行结果与完整测试输出见 [验证记录](sprint6-validation.md)。

首轮真实 GLM 生成 + MOCK Judge 处理60条：3成功、56限流失败、1超时。首轮报告保留在
`artifacts/evaluation/attempt1/`。诊断请求返回 HTTP 429、业务码1305（模型当前访问量过大）。
补充有限退避后完成第二轮60条：28成功、24限流失败、8超时。报告为真实生成 + MOCK Judge，
可用检索结果23条：Recall@3=1、Precision@3=1/3、MRR@3=1；另25条可回答case缺失，
这些指标不能代表全部48条可回答case。Judge四项均分和幻觉率为MOCK占位；
拒答关键词探针5/12，另7条未判定，不能解释为真实拒答准确率。CLI按设计返回退出码1，
两份JSON包含全部60条记录并通过一致性校验。完整单元测试244项通过，21条依赖警告。

### Provider 与请求可靠性

`DemoRAGClient` 只依赖 `LLMProvider.generate()`。`OpenAICompatibleProvider` 封装 AsyncOpenAI，
负责 endpoint、model、timeout、retry，返回 `LLMResponse(text, trace)` 或 `ProviderError`。
Judge 使用同一个 Provider；JSON/Schema修复仍由 Judge 负责。CLI负责组装具体 Provider。
同步 generate() 通过 asyncio.run 执行，每次逻辑调用创建并关闭异步客户端；适用于当前同步CLI，
不直接在已有事件循环内调用。没有无法取消的后台线程，也没有跨事件循环复用网络客户端。

| 环境变量 | 默认值与含义 |
| --- | --- |
| RAG_API_KEY / RAG_BASE_URL / RAG_MODEL | 生成服务配置，无密钥默认值 |
| RAG_GENERATION_TIMEOUT_SECONDS | 60，每次生成请求截止时间，范围(0,120] |
| RAG_MAX_ATTEMPTS | 3，包含首次请求，范围1–5 |
| RAG_RETRY_DELAY_SECONDS | 5，指数退避初始等待，范围(0,30] |
| RAG_THINKING | 默认不传；仅支持该扩展的模型可填enabled/disabled，本轮未启用 |
| JUDGE_TIMEOUT_SECONDS | 60，每次Judge请求截止时间，范围(0,120] |
| JUDGE_MAX_ATTEMPTS | 3，API重试和JSON修复共用总请求预算 |
| JUDGE_RETRY_DELAY_SECONDS | 5，Judge传输重试的初始等待 |

429、超时（包括HTTP408/504）、连接错误、5xx可重试；认证/权限错误、无效请求、模型不存在不重试。
已知额度或欠费错误（即使HTTP 429）也立即终止；稍后重试无法恢复其额度。
指数退避默认5、10秒，单次最多30秒。Retry-After秒数或HTTP日期优先；若等待超过30秒，
直接报告失败而不提前重试。SDK自动重试为0，最终失败保留最后错误分类及全部尝试。

RAGResult增加retry_count和generation_trace。追溯包含模型、无凭据endpoint、timeout、
每次请求延迟、状态码、业务码、错误类别及退避等待。Case记录RAG/Judge各自和总retry_count；
工程汇总包含retry_count及failure_breakdown。Judge缓存命中时不把历史retry_count计为本轮重试。
重试计数=实际请求次数减1，JSON修复请求也属于Judge重试。API错误body和密钥不写入追溯。

### Sprint 7 验证

```powershell
# 10条人工构造Judge样例；未配置时明确输出 LIVE JUDGE NOT AVAILABLE
python -m src.judge.live_validation

# 真实生成 + MOCK Judge，全量60条，使用.env中的模型
python -m src.evaluation.runner --dataset data/datasets/ecommerce_eval_v1.json --top-k 3 --output artifacts/evaluation --judge-mode MOCK
```

原glm-4.7-flash当前探测仍返回429/1305（模型过载）。同平台免费glm-4-flash-250414探测成功，
本地.env已将RAG_MODEL设为后者；整轮固定该模型，代码不会在case之间自动切换模型。
模型可用性选择独立于评测数据，没有修改Corpus/Dataset。切换模型后的质量不能直接等同原模型。

本轮60条真实生成全部成功，0失败、0重试，平均1457.47ms；真实检索覆盖48条可回答case：
Recall@3=1.0、Precision@3=0.3402778、MRR@3=0.9583333。Judge仍为MOCK，
四项均分和幻觉率为占位值，拒答12/12为关键词探针，不代表真实生成/安全质量。
真实Judge未配置，live验证results=[]。完整测试277 passed、21条Chroma依赖警告。

新报告位于artifacts/evaluation，Sprint 7快照位于artifacts/sprint7/evaluation；
Sprint 6最终报告保存在artifacts/evaluation/sprint6。完整输出和限制见
[Sprint 7 验证记录](sprint7-validation.md)。一次全量成功不保证后续外部服务持续可用。

## Baseline Compare 与 Regression Gate

Sprint 8 独立消费已有 `EvaluationReport`，不重新调用 RAG 或 Judge。
`save_baseline(report)` 原子保存到 `artifacts/comparison/baseline.json`；`load_baseline()`
读取并校验快照，也可给两个函数传入自定义路径。显式保存会替换该路径的旧基线，
建议用不同文件名保留历史版本；运行 candidate 不会自动晋升基线。

快照保存 dataset version、以 SHA-256 表示的 corpus version、embedding/RAG/Judge model、
top_k、Judge prompt version、原始评测时间戳、11项聚合指标、LIVE/MOCK 标记和参与计算的 case IDs。
缺失或 null 指标保留缺失状态，不填0。保存的是聚合快照，完整 case 证据仍在源报告中。

```python
from pathlib import Path
from src.evaluation.models import EvaluationReport
from src.comparison import (
    save_baseline, load_baseline, compare_reports, apply_gate, load_gate_config,
)

# 首次建立基线；后续保留此文件，只加载新的 candidate report。
report = EvaluationReport.model_validate_json(
    Path("artifacts/evaluation/latest_report.json").read_text(encoding="utf-8")
)
save_baseline(report)
candidate = EvaluationReport.model_validate_json(
    Path("artifacts/sprint8/candidate.json").read_text(encoding="utf-8")
)
comparison = compare_reports(load_baseline(), candidate)
gate = apply_gate(comparison, load_gate_config("config/regression_gate.yaml"))
print(comparison.model_dump_json(indent=2))
print(gate.model_dump_json(indent=2))
# CI 中可使用：raise SystemExit(0 if gate.status == "PASS" else 1)
```

示例 candidate 是下方生成的模拟报告，与当前真实生成基线的模式不同，真实门禁会拒绝它。
实际版本验证时，将路径换成由 Runner 生成的 candidate 报告。

`delta = candidate - baseline`。Recall、Precision、MRR、四项 Judge 均分、拒答准确率和
评测成功率越高越好；幻觉率和平均延迟越低越好。`improvement`/`regression` 表示指标方向，
不直接决定门禁：候选值仍满足配置阈值时，小幅下降也可以 PASS。

| 规则 | 含义 |
| --- | --- |
| `min` / `max` | candidate 的绝对下限/上限，等于边界通过 |
| `max_increase: 0.2` | `(candidate - baseline) / baseline <= 20%`，用于延迟等越低越好的指标 |
| `recall_at_3` / `precision_at_3` | 除阈值外还要求 candidate.top_k=3；通用名称为 `recall_at_k` / `precision_at_k` |
| `latency` | `avg_latency` 的别名，绝对值单位为毫秒 |
| `allow_synthetic` | 默认 false；仅模拟实验显式设为 true，结果仍标记 `simulation=true` |

默认规则见 `config/regression_gate.yaml`：Recall@3≥0.85、MRR≥0.8、幻觉率≤0.1、
评测成功率≥0.95、延迟增幅≤20%。0基线到正值的相对增幅无法定义，明确 FAIL；0到0通过。
YAML 重复/未知字段、空规则、非法范围、字符串代替数字等均抛出 `GateConfigurationError`。

门禁对所配置指标逐项判定；缺失、无有效样本或不可比较的指标均 FAIL。数据版本或样本集合
不同会阻止比较；Judge 模型/评分 Prompt 改变时，Judge 指标不认定改善。变更 embedding、
RAG model 或 top_k 会记录配置差异；不同 K 的数值差异只表达配置取舍，不代表固定 K 的提升。
已有 MRR 是截断到对应 K 的 MRR@K。原始 delta 可以保留，但不可比较时方向标记为 null。

MOCK Judge 的占位分数、安全指标不能通过真实门禁；真实检索仍可单独验证。包含 MOCK
步骤的整条流水线成功率和延迟也不视为全量 LIVE 评测证据。当前 Sprint 7 快照保留
`JUDGE_MODE=MOCK`，因此默认完整门禁 FAIL 是正确行为。

### 离线模拟版本实验

```powershell
python -m src.comparison.demo --output artifacts/sprint8
python -m pytest -q
```

实验读取正式60条 case 与36个 chunk，以 ground truth 构造固定排名替身：每第四条可回答
case 在第2位首次命中，其余首先返回相关 chunk。基线 top_k=3，candidate top_k=1。
**这是手工控制的模拟检索，不能当成真实 Retriever Benchmark**；生成答案、Judge评分和
延迟也为模拟值，报告明确记录 `RAG_MODE=MOCK`、`JUDGE_MODE=MOCK` 及 synthetic 标记。
不调用 Embedding/LLM，不写 Dataset/Corpus，并核对运行前后文件哈希。

输出 `baseline.json`、`candidate.json`（完整报告）、`baseline_snapshot.json`、
`comparison.json`、`gate_pass.json` 和 `gate_fail.json`。基线自检 PASS，candidate 因
Recall=0.739583<0.85、MRR=0.75<0.8 而 FAIL。demo 退出码0表示这两个预期行为验证成功，
并不表示 candidate 已获准发布。实验数值、完整文件列表和测试输出见
[Sprint 8 验证记录](sprint8-validation.md)。

## Evaluation Dashboard 与 Report Export

在项目根目录执行：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
# 未激活环境时：.venv\Scripts\python -m streamlit run app.py
```

浏览器访问 `http://127.0.0.1:8501`。配置默认只绑定本机且关闭使用统计。
页面从已保存的报告读取结果，不执行 Runner、不建索引、不调用 LLM，不需要 API Key。

页面结构：

- 侧栏：选择已有 Top-K 和报告，或上传完整 EvaluationReport JSON。
- Overview：显示总数、成功/失败数、成功率；展开版本区可查看 dataset/corpus version 和模型信息。
- Metrics：Recall@K、Precision@K、MRR@K 卡片、检索指标条形图、case latency 直方图，
  以及四项 Judge 均分、幻觉率、拒答准确率。缺失值显示 N/A，不填0。
- Regression：选择 baseline 报告/快照，也可上传；当前报告作为 candidate，复用 Sprint 8
  对比引擎，展示 baseline、candidate、delta、status、来源和不可比原因，并展示默认门禁结论。
- Case detail：按 category、SUCCESS/FAIL、hallucination筛选；FAIL包括RAG_ERROR、
  RETRIEVAL_ERROR、JUDGE_ERROR、TIMEOUT。无Judge结果属于“未评判”，不当作“无幻觉”。
  单条展示问题、标准答案、检索chunk内容/ID/排名/分数、生成答案、指标、Judge JSON及错误记录。
- Report export：下载当前完整报告的JSON/Excel；case筛选不裁剪汇总或下载内容。

默认扫描 `artifacts/` 下名为 `latest_report.json`、`baseline.json`、`candidate.json`、
`evaluation_report.json` 的有效完整报告。其他文件名可以直接上传。可用
`RAGEVAL_DASHBOARD_ARTIFACTS` 环境变量指定另一个报告根目录。坏文件不进入报告选择列表；
上传无效结构会显示错误。候选报告与基线支持不同K，但界面保留不同K/模式/版本警告，
不重新计算指标，也不把选择K理解成运行新评测。

Judge为MOCK时，页面显式显示 **JUDGE MODE: MOCK**，均分和安全结果标明为模拟占位。
RAG也为MOCK时，额外说明检索和耗时来自模拟。现有Sprint 7报告的真实检索仍可查看，
但MOCK Judge评分不能解释为真实质量，其完整Regression Gate保持FAIL。

标准答案不在历史CaseEvaluationResult中。展示层仅从 `data/datasets/` 寻找与报告
`dataset_sha256` 完全匹配的数据集，再按case_id关联；不会信任上传报告中的任意文件路径。
数据缺失/哈希不同会提示不可用，不使用其他版本的标准答案。

### 导出命令与格式

```powershell
python -m src.report --report artifacts/evaluation/latest_report.json --output artifacts/report
```

可选 `--dataset-dir data/datasets` 指定标准答案目录。输出：

```text
artifacts/report/
├── evaluation_report.json
└── evaluation_report.xlsx
```

Python API：

```python
from src.report import load_report, export_report
report = load_report("artifacts/evaluation/latest_report.json")
json_path, excel_path = export_report(report, "artifacts/report")
```

JSON保留完整EvaluationReport及嵌套case/追溯结构，可直接再上传Dashboard。
Excel导出由项目Python模块使用openpyxl完成，用户安装requirements即可运行，无额外Node依赖。

| Sheet | 内容 |
| --- | --- |
| Summary | 版本、模型、时间、运行模式、完整聚合指标 |
| Case Results | 每条case的query、expected answer、rag answer、检索chunks、状态、错误、延迟及重试 |
| Retrieval Metrics | 每条case的K、Recall、Precision、RR、预测/相关chunk IDs、排除/不可用状态 |
| Judge Results | 四项评分、幻觉、unsupported claims、reason、拒答观察、模型/Prompt、延迟和raw output |

每张表保留MOCK提示；表头可筛选并冻结。数字保持数值类型，缺失值为空白；文本作为字面量写入，
以 `=` 开头也不会执行公式。Excel单元格32767字符上限会触发显式TRUNCATED标记，
XML不支持的控制字符会转为可见转义，完整原文以JSON为准。两份文件分别原子替换，不构成跨文件事务。

### Dashboard 截图

以下是实际启动本地Streamlit后截取的页面，来源为Sprint 7已保存报告，**真实RAG + MOCK Judge**：

![RAGEval Dashboard Overview](screenshots/dashboard-overview.png)

- [Regression](screenshots/dashboard-regression.png)
- [Case detail](screenshots/dashboard-case-detail.png)
- [375px窄屏](screenshots/dashboard-mobile.png)

截图统一保存于 `docs/screenshots/`。本轮导出示例位于 `artifacts/sprint9/export/`。
测试命令、结果和完整变更列表见 [Sprint 9 验证记录](sprint9-validation.md)。

### GLM Judge JSON 模式

GLM 平台可在本地 `.env` 设置 `JUDGE_RESPONSE_FORMAT=json_object`，客户端会发送
`response_format={"type": "json_object"}`。该模式约束 JSON 格式，字段、分数与语义一致性仍由本地
Pydantic 校验；它不能保证裁判评分准确。`json` 保持仅用 Prompt 约束的兼容行为，
`json_schema` 仅用于支持服务端 Schema 的模型。切换模式会改变 Judge 缓存键。
