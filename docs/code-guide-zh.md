# RAGEval 中文代码导航

这份文档回答两个问题：每个代码文件负责什么，为什么项目需要它。你不必逐行读代码，先看这里，再看方法上方的中文注释即可。

## 先理解项目在做什么

RAGEval 像一套自动批改系统：评测数据集是试卷，Demo RAG 是被测试的答题系统，检索指标检查它有没有找到正确资料，LLM Judge 检查它答得好不好。Runner 负责组织整个过程，报告记录结果，Dashboard 展示结果，版本对比和回归门禁判断新版本是否退步。

## 建议阅读顺序

1. `src/dataset/models.py`：认识一份资料、一道题和一次回答分别包含什么。
2. `src/evaluation/runner.py`：看整场评测怎样串起来。
3. `src/rag/client.py`：看被测系统怎样找资料并生成答案。
4. `src/metrics/retrieval.py`：看怎样判断有没有找到正确资料。
5. `src/judge/client.py`：看怎样让裁判模型给回答打分。
6. `src/evaluation/aggregation.py`：看怎样把每道题的结果汇总。
7. `src/comparison/` 和 `src/report/`：最后看版本对比、页面展示和文件导出。

## 怎样看代码里的注释

每个文件开头都有“文件作用”和“为什么有它”。每个方法上方都有“做什么”和“为什么需要”，数据类也有相应说明。原有英文说明仍然保留。

- **类**：把一类相关的数据或操作放在一起。例如一张评测结果表，或者一个负责调用模型的工具。
- **方法/函数**：完成一件具体工作，例如读文件、检索资料、计算分数。
- **`models.py`**：通常规定数据表需要哪些字段，防止传入的内容缺项或格式不对。
- **`__init__.py`**：把目录标记为 Python 包，让模块之间可以按统一路径导入。
- **`__main__.py`**：提供命令行入口，让某个包可以直接运行。
- **`test_*.py`**：自动检查业务代码是否符合预期；其中的假对象用于代替真实外部服务。

## 每个 Python 文件的作用

下面覆盖项目自身的全部 72 个 Python 文件，不包括虚拟环境和第三方依赖。点击文件名可以打开对应代码。

### 入口与配置

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [app.py](../app.py) | 启动评测结果网页，把页面工作交给 report.dashboard。 | 提供一个容易找到的 Streamlit 启动入口，打开页面不会重新运行评测。 |
| [config/__init__.py](../config/__init__.py) | 标记“项目配置”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [config/judge.py](../config/judge.py) | 读取并检查 Judge 使用的模型、地址、密钥、超时和输出格式。 | 裁判与生成模型需要独立配置，配置错误应在调用服务前发现。 |
| [config/settings.py](../config/settings.py) | 读取 RAG、Embedding 和 Chroma 的运行配置，并检查必要参数。 | 让密钥、模型和路径通过环境配置切换，不散落在业务代码里。 |
| [src/__init__.py](../src/__init__.py) | 标记“项目业务代码”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |

### 数据：资料与试题

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/dataset/__init__.py](../src/dataset/__init__.py) | 标记“数据模型与加载”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/dataset/loader.py](../src/dataset/loader.py) | 从 JSON 文件读取知识库与试卷，并检查编号、结构和相关块引用。 | 避免错误标注或重复数据进入检索和评测流程。 |
| [src/dataset/models.py](../src/dataset/models.py) | 规定知识块、评测题、检索结果和 RAG 输出应该包含哪些字段。 | 让各模块交换相同结构的数据，尽早发现字段缺失和非法值。 |

### RAG：被测试的答题系统

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/rag/__init__.py](../src/rag/__init__.py) | 标记“被测 RAG 系统”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/rag/__main__.py](../src/rag/__main__.py) | 提供从命令行运行一次 Demo RAG 的入口。 | 便于单独验证被测系统，而不必先跑完整评测。 |
| [src/rag/client.py](../src/rag/client.py) | 规定被测 RAG 的接口，并实现检索资料、组装上下文和生成答案。 | 把被测系统与评测工具分开，换生成服务时不改指标公式。 |
| [src/rag/embeddings.py](../src/rag/embeddings.py) | 把知识文本和查询问题编码成向量，并按需加载本地模型。 | 向量检索需要统一的文本表示，而查看报告时不应该加载模型。 |
| [src/rag/retriever.py](../src/rag/retriever.py) | 将知识块同步到 Chroma，并按问题返回带编号和排名的 Top-K 结果。 | 为 Demo RAG 提供可持久化、可核对标注的本地检索能力。 |
| [src/rag/smoke.py](../src/rag/smoke.py) | 抽取不同主题的问题，打印真实检索结果与标准相关块。 | 快速确认本地 Embedding 和 Chroma 连通，不调用生成模型。 |

### LLM Provider：访问生成模型

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/llm/__init__.py](../src/llm/__init__.py) | 标记“模型服务访问”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/llm/models.py](../src/llm/models.py) | 规定模型调用结果、每次尝试、总轨迹和失败信息的数据结构。 | 让失败原因、耗时和重试次数能够被上层统一保存。 |
| [src/llm/openai_compatible.py](../src/llm/openai_compatible.py) | 通过兼容 SDK 调用模型，并统一处理超时、错误分类和有限重试。 | 把外部服务的不稳定性集中处理，避免每个业务模块重复实现。 |
| [src/llm/provider.py](../src/llm/provider.py) | 定义模型服务调用的统一接口和通用错误对象。 | 让 RAG 和 Judge 依赖统一约定，便于替换供应商和注入测试替身。 |

### Metrics：检查检索质量

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/metrics/__init__.py](../src/metrics/__init__.py) | 标记“检索指标”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/metrics/benchmark.py](../src/metrics/benchmark.py) | 只运行真实检索，对同一排名分别计算 K=1、3、5 并记录失败题。 | 单独测量检索能力，避免生成和裁判影响这一阶段的结论。 |
| [src/metrics/models.py](../src/metrics/models.py) | 规定单题检索指标和整组检索汇总的数据格式。 | 区分有分数的可回答题与应排除的不可回答题，避免错误统计。 |
| [src/metrics/retrieval.py](../src/metrics/retrieval.py) | 按知识块编号计算 Recall、Precision、RR 和多题平均结果。 | 使用确定的公式衡量检索表现，不让模型来猜测是否命中。 |

### Judge：检查回答质量

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/judge/__init__.py](../src/judge/__init__.py) | 标记“答案质量裁判”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/judge/client.py](../src/judge/client.py) | 调用模型判卷，严格校验 JSON 和评分结构，并在有限预算内修复输出。 | 把不合规输出和 API 故障明确记录，不能把坏数据当成有效评分。 |
| [src/judge/live_validation.py](../src/judge/live_validation.py) | 扩展为十条人工样例，保存真实裁判结果或逐条错误。 | 在全量评测前暴露格式、接口和评分问题，保留原始证据。 |
| [src/judge/models.py](../src/judge/models.py) | 规定裁判输入、评分字段、拒答信息及调用元数据。 | 先检查裁判输出是否结构合法，再允许进入报告和缓存。 |
| [src/judge/prompts.py](../src/judge/prompts.py) | 保存不同版本的裁判规则，并把规则和待评数据组装成请求消息。 | 让正确性、忠实性和拒答的评分依据有明确版本、便于追溯。 |
| [src/judge/smoke.py](../src/judge/smoke.py) | 构造四条人工样例，单独验证真实 Judge。 | 接入裁判时先检查典型场景，未配置 API 时明确跳过而不编造结果。 |

### Evaluation：组织评测与汇总

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/evaluation/__init__.py](../src/evaluation/__init__.py) | 标记“评测调度与汇总”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/evaluation/aggregation.py](../src/evaluation/aggregation.py) | 把逐条结果按检索、生成、安全和工程表现分别汇总。 | 每类指标分母不同，不能把未评估或不可回答的题随意当作零分。 |
| [src/evaluation/cache.py](../src/evaluation/cache.py) | 按评分输入与配置保存和读取成功 Judge 结果，并提供安全写 JSON 的方法。 | 减少重复模型调用，同时避免损坏缓存或半写入文件误导评测。 |
| [src/evaluation/mock_judge.py](../src/evaluation/mock_judge.py) | 提供明确标记为模拟的裁判，用占位分数和简单拒答探针验证流程。 | 没有真实 API 时仍可测试流水线，但这些结果不能冒充真实质量。 |
| [src/evaluation/models.py](../src/evaluation/models.py) | 规定逐题结果、分项汇总和整份评测报告的数据结构。 | 让成功、失败、缺失分数和运行版本都有固定位置可查询。 |
| [src/evaluation/runner.py](../src/evaluation/runner.py) | 串起数据加载、RAG、指标、Judge、缓存、失败记录和报告保存。 | 把零散能力变成可批量执行的评测流程，并让单条失败不影响其余题目。 |

### Comparison：版本对比与回归判断

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/comparison/__init__.py](../src/comparison/__init__.py) | 标记“版本对比与回归判断”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/comparison/demo.py](../src/comparison/demo.py) | 离线构造两个模拟版本，生成对比结果和门禁通过、失败示例。 | 不花真实模型调用成本即可演示比较流程，并明确区分模拟与真实结论。 |
| [src/comparison/engine.py](../src/comparison/engine.py) | 核对两份报告是否可比，并计算每项指标的变化方向。 | 数据或有效样本变化时，单纯比较平均数可能误判进步。 |
| [src/comparison/gate.py](../src/comparison/gate.py) | 读取阈值配置并逐项给出 PASS 或 FAIL 及原因。 | 让版本验收按明确规则执行，缺指标和错误配置不能默认为通过。 |
| [src/comparison/models.py](../src/comparison/models.py) | 规定基线、指标差值、门禁规则和门禁结论的数据结构。 | 统一指标名称、方向和取值，避免错误配置产生虚假通过。 |
| [src/comparison/snapshot.py](../src/comparison/snapshot.py) | 把完整报告转换成版本快照，并提供基线保存和加载能力。 | 留下稳定的历史参照，供后续候选版本进行对比。 |

### Report：页面展示与报告导出

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/report/__init__.py](../src/report/__init__.py) | 标记“报告展示与导出”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [src/report/__main__.py](../src/report/__main__.py) | 提供已有报告的命令行导出入口。 | 用户可以直接生成 JSON 和 Excel，不需要打开网页或调用模型。 |
| [src/report/dashboard.py](../src/report/dashboard.py) | 用 Streamlit 展示总览、指标、回归比较、单题详情和下载入口。 | 让不读代码的人也能查看评测证据，并明确识别 MOCK 报告。 |
| [src/report/data.py](../src/report/data.py) | 读取报告、匹配同版本标准答案、筛选题目并生成模式说明。 | 展示层只查看已有证据，避免误用其他数据版本或重新评测。 |
| [src/report/export.py](../src/report/export.py) | 把已有报告导出成完整 JSON 和四张 Excel 工作表。 | 方便分享与查看结果，同时处理长文本、非法字符和公式样式字符串。 |

### Scripts：工程检查

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [scripts/check_imports.py](../scripts/check_imports.py) | 尝试导入配置、业务模块和页面入口。 | 在本地和 CI 提前发现缺依赖、导入错误或模块入口问题。 |
| [scripts/check_release.py](../scripts/check_release.py) | 检查拟发布文件中的密钥、个人路径、模型缓存、体积和 Git 忽略规则。 | 推送前检查哪些文件会公开，并确认正式数据不会被 Git 换行处理改变。 |

### Tests：自动化验证

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [tests/__init__.py](../tests/__init__.py) | 标记“自动化测试”目录为 Python 包，供其他模块导入。 | 让相关文件能够通过统一的包路径组织和引用。 |
| [tests/provider_fakes.py](../tests/provider_fakes.py) | 提供假的 SDK 响应和可注入测试的 Provider 适配器。 | 保留真实重试逻辑，同时阻止单元测试访问外部模型或真的等待退避。 |
| [tests/test_comparison.py](../tests/test_comparison.py) | 测试基线读写、指标可比性、差值方向和回归门禁。 | 避免错误样本、缺失指标或 MOCK 分数被判成真实改进。 |
| [tests/test_dashboard.py](../tests/test_dashboard.py) | 通过 Streamlit AppTest 检查页面元素、过滤、报告切换和下载入口。 | 验证页面展示现有报告，不依赖浏览器交互或真实模型。 |
| [tests/test_embeddings.py](../tests/test_embeddings.py) | 测试模型按需加载、向量归一化和查询指令处理。 | 防止模型重复加载或把查询指令错误加到文档上。 |
| [tests/test_evaluation_cache.py](../tests/test_evaluation_cache.py) | 测试相同输入的缓存键稳定，以及输入和裁判配置变化后的失效。 | 防止将旧答案或旧评分规则的结果错误复用。 |
| [tests/test_evaluation_runner.py](../tests/test_evaluation_runner.py) | 测试逐题调度、失败隔离、分母、缓存和报告保存。 | 保证成功与失败都被记录，局部故障不污染整体统计。 |
| [tests/test_judge_client.py](../tests/test_judge_client.py) | 测试裁判请求参数、结构化输出、有限重试、错误分类和元数据。 | 不调用真实模型也能覆盖合法、非法和超时等接口分支。 |
| [tests/test_judge_live_validation.py](../tests/test_judge_live_validation.py) | 测试十条真实验证样例的范围、配置缺失和执行调度。 | 验证 live 检查脚本按约定运行，测试本身仍使用替身。 |
| [tests/test_judge_models.py](../tests/test_judge_models.py) | 测试裁判输入输出的类型、分数范围和幻觉证据约束。 | 防止看似是 JSON 但不符合评分契约的数据进入报告。 |
| [tests/test_judge_settings.py](../tests/test_judge_settings.py) | 测试裁判环境配置读取、优先级和参数边界。 | 让缺失或非法配置在真正访问模型前清楚报错。 |
| [tests/test_judge_smoke.py](../tests/test_judge_smoke.py) | 测试四条冒烟样例以及缺配置时的处理。 | 确保真实验证不会自动伪造结果，单元测试也不发外部请求。 |
| [tests/test_judge_v2.py](../tests/test_judge_v2.py) | 测试 judge_v2 的显式拒答字段和 Prompt 版本限制。 | 让 Runner 使用的拒答统计有真实字段来源，不凭其他分数猜测。 |
| [tests/test_loader.py](../tests/test_loader.py) | 测试知识库和试卷的读取、编号、结构、标注关系和正式数据规模。 | 防止错误测试数据让后续指标失去意义。 |
| [tests/test_models.py](../tests/test_models.py) | 测试知识块、评测题和 RAG 结果的基础字段约束及 JSON 往返。 | 保证模块间数据格式一致，空值与非法值有明确区别。 |
| [tests/test_provider.py](../tests/test_provider.py) | 测试模型接口参数、重试、Retry-After、超时取消和永久错误。 | 让网络故障能受控结束，并保留真实尝试次数。 |
| [tests/test_rag_client.py](../tests/test_rag_client.py) | 测试 RAG 接口、上下文组装、输出结果和检索异常。 | 确认被测系统正确连接检索器与 Provider，而不直接依赖真实服务。 |
| [tests/test_report_export.py](../tests/test_report_export.py) | 测试报告读取、过滤、JSON 与 Excel 导出和标准答案匹配。 | 避免展示丢失信息、误把文本当公式或读取不应访问的文件。 |
| [tests/test_retrieval_benchmark.py](../tests/test_retrieval_benchmark.py) | 测试检索基准实验的调用次数、K 截断和失败明细。 | 保证不同 K 使用同一次排名，错误标注在检索前被发现。 |
| [tests/test_retrieval_metrics.py](../tests/test_retrieval_metrics.py) | 用手算数据测试检索公式、边界值和宏平均。 | 确保分数来自正确的编号、排名和分母，而不是模型判断。 |
| [tests/test_retriever.py](../tests/test_retriever.py) | 用固定向量和真实本地 Chroma 测试写入、排序、更新和参数校验。 | 在不下载 Embedding 模型的情况下验证向量库接入行为。 |
| [tests/test_settings.py](../tests/test_settings.py) | 测试 RAG 环境配置、超时、重试和可选模型参数。 | 防止配置覆盖错误或非法数值进入实际调用。 |

## 代码之外，还会看到什么

这些文件不包含业务方法，但会影响项目怎样安装、配置和运行。

| 文件或目录 | 作用 |
| --- | --- |
| `.env.example` | 环境变量填写示例，不保存真实密钥。 |
| `.env` | 你本机的实际配置，例如模型地址和密钥；不应提交到 Git。 |
| `.gitignore` | 告诉 Git 哪些本机文件、缓存和输出不用上传。 |
| `requirements.txt` | 安装项目所需的第三方 Python 库。 |
| `pyproject.toml` | 项目的工具配置，例如测试设置。 |
| `.github/workflows/test.yml` | 让 GitHub 自动安装依赖并执行测试。 |
| `data/corpus/` | 被测 RAG 查阅的知识库资料。 |
| `data/datasets/` | 评测问题、标准答案以及应命中的资料编号。 |
| `artifacts/` | 保存运行产生的报告等结果。 |
| `docs/` | 使用说明、发布检查和面试学习文档。 |

## 一个容易混淆的区别

`src/rag/` 是“被测试的系统”，`src/evaluation/` 是“负责组织测试的系统”。修改 RAG 可能影响回答质量；评测框架负责把这种变化测出来。Dashboard 读取报告供你查看，打开页面本身不等于重新执行评测。

测试文件中的 `mock` 或 `fake` 是模拟对象，用来稳定地验证流程和异常处理。通过这些测试不能证明真实模型的回答质量，真实模型质量需要另行运行真实评测。

## 后续新增：重排序实验

| 文件 | 做什么 | 为什么需要 |
| --- | --- | --- |
| [src/rag/reranker.py](../src/rag/reranker.py) | 联合评估问题与候选知识块，重新排序；提供两阶段检索器。 | 让正确资料有机会排得更靠前，同时保留原始身份。 |
| [src/metrics/rerank_benchmark.py](../src/metrics/rerank_benchmark.py) | 比较相同候选在重排前后的指标和耗时。 | 用数据判断重排序是否值得，保存改善与退步案例。 |
| [tests/test_reranker.py](../tests/test_reranker.py) | 使用假模型检查排序、边界、命令行开关与实验公平性。 | 不依赖下载和外部服务也能稳定验证新功能。 |
