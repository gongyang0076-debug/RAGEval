# Sprint 9 验证记录

## 交付范围

增加只读 Streamlit Dashboard 和 Python JSON/Excel 导出模块。没有修改 Runner、
Dataset/Corpus、指标定义、Judge 或比较引擎，没有新增聊天功能或真实模型调用。

```text
app.py → src.report.dashboard
              ↓
本地/上传 EvaluationReport → Pydantic校验 → 卡片/图表/case详情
              ↓                           ↓
哈希匹配的Dataset → expected_answer     Sprint 8 compare/gate
              ↓
src.report.export → 完整JSON + 四表Excel
```

Dashboard使用 Streamlit 原生控件与少量CSS，浅色背景、深绿指标条、无装饰动画。
默认绑定127.0.0.1，不加载.env或调用生成模型。侧栏选择已有报告与K；切换报告不重算分数。
版本/模型信息在Overview可展开查看；Metrics、Regression、Case detail分成三个标签页。

检索条形图范围固定0–1，延迟分布采用直方图；两个图表都消费已有结果。
缺失指标显示N/A，空报告显示空状态。Judge为MOCK时Overview、Generation及导出均明确标记。
Regression表保留指标方向、真实/模拟来源、不可比说明，并复用默认门禁规则。

标准答案只从哈希匹配的数据集读取，不使用上传报告的任意dataset_path。
Category、SUCCESS/FAIL和hallucination过滤只影响case列表；下载与整体指标保持全报告。
没有Judge结果归入“未评判”，不会混进“没有幻觉”类别。

## 启动与导出

```powershell
.venv\Scripts\python -m streamlit run app.py
.venv\Scripts\python -m src.report --report artifacts/evaluation/latest_report.json --output artifacts/sprint9/export
```

本地页面：http://127.0.0.1:8501。依赖已加入requirements：Streamlit、Altair、pandas、openpyxl。
本机验证版本：Streamlit1.63.0、openpyxl3.1.5、Python3.14；项目保留Python3.10+声明。

导出产物：

- `artifacts/sprint9/export/evaluation_report.json`
- `artifacts/sprint9/export/evaluation_report.xlsx`

JSON是完整EvaluationReport，能通过Pydantic往返校验。Excel严格四表：

| Sheet | 已验证内容 |
| --- | --- |
| Summary | 39行元数据/聚合指标，含SHA-256语料版本、dataset版本、模型、模式与评测时间 |
| Case Results | 60条，含标准答案、生成答案、检索证据JSON、状态/错误/耗时/重试 |
| Retrieval Metrics | 60条，48条有效检索、12条EXCLUDED，排除项指标为空 |
| Judge Results | 60条，含MOCK模式、分数、unsupported claims、reason、refusal、模型/Prompt与raw output |

Excel前四行为标题、说明、空行和表头；冻结A5、启用筛选、使用数值单元格。
没有重新计算指标，没有生成公式；以`=`开头的文本仍为字符串。
长文本超过Excel上限时显式标注截断，完整JSON保持原文。每个sheet重复模式提示。

使用openpyxl重新读取文件核对内容、类型、行数、空值和冻结/筛选配置；同时通过
独立Artifact Tool导入并渲染四张表检查布局，错误扫描匹配0项。

## 截图与页面验证

使用本机Edge headless实际打开Streamlit，等待指标及下载按钮渲染后截图：

- `docs/screenshots/dashboard-overview.png`：总览、指标卡、条形图、直方图、Generation与下载。
- `docs/screenshots/dashboard-regression.png`：对比表和MOCK门禁失败原因。
- `docs/screenshots/dashboard-case-detail.png`：筛选、标准答案、RAG答案、chunk与Judge详情入口。
- `docs/screenshots/dashboard-mobile.png`：375px宽度，侧栏折叠、指标纵向排列。

已逐张打开检查，无页面异常。截图源为现有Sprint 7报告：60成功、0失败，
Recall@3=1、Precision@3=0.3402778、MRR@3=0.9583333。
Judge仍为MOCK；页面显示的0分/0幻觉率为占位值，拒答率100%是既有关键词探针结果。
本阶段没有重新执行Benchmark，也没有把这些数值宣称为真实Judge评分。

## pytest

```text
.venv\Scripts\python -m pytest -q
350 passed, 21 warnings in 29.57s
```

退出码0。完整输出：[pytest.txt](../artifacts/sprint9/pytest.txt)。
20项新增测试覆盖报告/JSON往返、错误格式、四表Excel、数值/空值、公式文本、长文本、
哈希匹配与错误dataset、过滤、只读性、模块导入、MOCK提示、图表/下载控件、K切换与空页面。
AppTest使用模拟报告，禁止httpx同步/异步网络调用；不依赖浏览器交互测试。
21条警告为已有Chroma弃用提示。

UI测试使用官方 [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)，
图表使用 [st.altair_chart](https://docs.streamlit.io/develop/api-reference/charts/st.altair_chart)。

## 新增/修改文件

新增源码/配置：

- `app.py`
- `.streamlit/config.toml`
- `src/report/data.py`
- `src/report/dashboard.py`
- `src/report/export.py`
- `src/report/__main__.py`
- `tests/test_report_export.py`
- `tests/test_dashboard.py`
- `docs/sprint9-validation.md`
- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/dashboard-regression.png`
- `docs/screenshots/dashboard-case-detail.png`
- `docs/screenshots/dashboard-mobile.png`

修改：`src/report/__init__.py`、`requirements.txt`、`README.md`。
新增验证产物：`artifacts/sprint9/input_hashes.json`、`artifacts/sprint9/pytest.txt`和上述两份导出文件。
临时浏览器/Excel验证脚本和预览保存在已忽略的`.cache/sprint9/`，不是应用运行依赖。

数据和核心模块共35个文件在本轮开始时记录SHA-256，结束时逐个比对，变更数0。
范围包括data、evaluation、metrics、dataset、comparison、rag、judge；基准见input_hashes.json。

## 风险与边界

- 当前真实Judge未配置，Dashboard不会改善或补齐既有评分；真实门禁仍应拒绝MOCK质量结果。
- 无匹配数据集时标准答案不可用，报告其他字段仍可查看和导出。
- Excel是供人查看的扁平视图，极长文本与XML控制字符以完整JSON为准；用户可调整行高查看长单元格。
- 页面用于本地、当前60条规模；未增加鉴权、远程部署或大规模分页，默认只监听本机。
- 截图记录本轮报告状态，后续报告变化后需重新截图。不同K可对应不同模型/模拟实验，不能混为同一配置性能曲线。

SPRINT 9 ACCEPTANCE: PASS
