# Sprint 8 验证记录

本阶段新增 Baseline Compare 与 Regression Gate，消费已有 EvaluationReport。
没有修改 Dataset/Corpus，没有调用真实 Embedding 或 LLM，没有新增 UI 或 Excel。

## 架构与数据流

```text
EvaluationReport → to_snapshot → BaselineSnapshot → save_baseline/load_baseline
                                              ↓
baseline + candidate → compare_reports → ReportComparison
                                              ↓
YAML → load_gate_config → GateConfig → apply_gate → GateResult(PASS/FAIL)
```

- `snapshot.py`：Pydantic 校验、原子 JSON 保存、加载时拒绝重复 JSON key；错误为 SnapshotError。
- `engine.py`：计算11项 delta、改善/下降方向、配置变化和可比性说明。
- `gate.py`：安全 YAML 解析、严格规则校验、逐项判定并汇总所有失败原因。
- `demo.py`：离线生成两份完整60条模拟报告，验证正常通过与预期回归失败。

BaselineSnapshot 保存 schema_version、dataset_version、corpus_version、embedding_model、
rag_model、top_k、judge_model、judge_prompt_version、metrics、timestamp、运行模式和样本集合。
corpus_version 使用 `sha256:<corpus_sha256>`，timestamp 保留原始评测时间且必须含时区。
metrics 值必须有限、非负；比例≤1、Judge 均分≤5；缺失指标/null 不补零。
cohorts 保存全体、有效检索和有效 Judge 的 case IDs，防止失败样本集合变化导致虚假改善。

MetricComparison 包含 metric、direction、baseline_value、candidate_value、delta、
improvement、regression、comparable、synthetic、reason。delta 始终为 candidate-baseline；
越高越好的指标正差为改善，幻觉率与延迟负差为改善，绝对差≤1e-12视为不变。
不可比较时保留可计算的原始差值，但 improvement/regression 为 null。

GateResult 包含 status、simulation、checks 和 reasons，每条失败都保留说明，不静默跳过。
同一数据/样本集合下可以比较不同 RAG/embedding/top_k 配置；不同 K 有显式警告。
Dataset/Corpus 版本不同、相关样本集合不同或为空、运行模式不同，相关指标不可比较。
Judge 模型或 Prompt version 变化时，Judge 指标不可比较；检索指标仍可用。

## Gate 规则

`config/regression_gate.yaml`：

```yaml
allow_synthetic: false
thresholds:
  recall_at_3:
    min: 0.85
  mrr:
    min: 0.8
  hallucination_rate:
    max: 0.1
  evaluation_success_rate:
    min: 0.95
  latency:
    max_increase: 0.2
```

- min/max 限制 candidate 绝对值，等于边界通过，比较容差1e-12。
- max_increase 限制相对增长 `(candidate-baseline)/baseline`，0.2是20%，不是20个百分点。
  基线0且候选正值时明确FAIL；两者0时相对增长为0。
- 支持 `recall_at_N`、`precision_at_N`，校验 candidate 的 K；通用规则以 `*_at_k` 命名。
- 只有配置的指标参与 Gate。MetricComparison.regression=true 不一定触发 Gate FAIL。
- 缺失/null指标、不可比较、空样本、MOCK指标默认使对应规则FAIL，不输出虚假PASS。
- 配置错误抛出 GateConfigurationError：空规则、未知名称、重复key/别名、非有限或负数、
  超出分数范围、min>max、字符串/布尔值代替数字，以及给越高越好指标配置max_increase。
- `allow_synthetic=true` 仅用于演示；结果保留 simulation=true，不能据此宣称真实质量验收。

## 模拟版本实验

命令：`.venv\Scripts\python -m src.comparison.demo --output artifacts/sprint8`

两份报告均60条、60 SUCCESS、0失败；48条计算检索指标，12条不可回答被排除。
固定 ground truth 排名替身每第四条可回答case先放一个干扰chunk，再返回相关chunk。
只改变 top_k：baseline=3，candidate=1。没有修改任何正式标签，也没有真实检索/生成调用。
使用标签构造排名仅服务于受控测试，不能当成检索性能证据。
所有生成回答是固定模拟拒答，Judge使用已有MOCK占位结果，延迟是人为设置的确定值。

| Metric | Baseline K=3 | Candidate K=1 | Delta |
| --- | ---: | ---: | ---: |
| Recall@K | 1.000000 | 0.739583 | -0.260417 |
| Precision@K | 0.340278 | 0.750000 | +0.409722 |
| MRR@K | 0.875000 | 0.750000 | -0.125000 |
| Correctness | 0 | 0 | 0 |
| Faithfulness | 0 | 0 | 0 |
| Relevance | 0 | 0 | 0 |
| Completeness | 0 | 0 | 0 |
| Hallucination rate | 0 | 0 | 0 |
| Refusal accuracy | 1 | 1 | 0 |
| Evaluation success rate | 1 | 1 | 0 |
| Avg latency（ms） | 25 | 17 | -8 |

评分和幻觉率0是MOCK占位；拒答率1来自固定拒答的关键词探针；延迟不是实测性能。
K=1的Precision上升是实际返回数量分母变小产生的取舍，不能解读为固定K的系统改进。

演示配置 `config/regression_gate_demo.yaml` 显式允许synthetic，使用通用recall_at_k：

```text
Baseline self-check: PASS
Candidate gate: FAIL
recall_at_k: 1 -> 0.739583; below min 0.85
mrr: 0.875 -> 0.75; below min 0.8
```

demo退出码0表示PASS/FAIL两个预期均验证成功。单元测试另验证Recall 0.91→0.82和延迟
100→125ms同时失败，失败说明包含25%实际增幅及20%上限；延迟100→120ms在边界通过。

## 现有报告快照

同时对已有 `artifacts/evaluation/latest_report.json` 做了保存/加载往返校验，
新快照位于 `artifacts/comparison/baseline.json`。没有覆盖或重跑Sprint 7报告。
该报告真实生成、MOCK Judge，默认完整门禁自检为FAIL，输出保存在
`artifacts/sprint8/current_report_gate.json`：

```text
hallucination_rate: synthetic/MOCK metric cannot pass a live gate
evaluation_success_rate: synthetic/MOCK metric cannot pass a live gate
latency: synthetic/MOCK metric cannot pass a live gate
```

真实检索阈值通过；MOCK裁判下的安全结果与整条流水线成功率/延迟不能代表全LIVE流水线。
这是预期的数据来源保护，不是把Sprint 7运行成功误报为调用失败。

## 测试与完整输出

最终命令：`.venv\Scripts\python -m pytest -q`

结果：`330 passed, 21 warnings in 8.75s`，退出码0。新增比较测试53项。
21条警告来自现有Chroma依赖的弃用提示，无测试失败。

完整控制台输出保存在 [pytest.txt](../artifacts/sprint8/pytest.txt)。
新增比较测试使用手工构造数据，并禁止 httpx 同步/异步网络请求。
覆盖快照往返、非法文件、11项delta/方向、PASS/FAIL、缺失指标、空样本、阈值错误、
零基线、不同K/版本/评分样本集合/MOCK模式，以及完整模拟报告的JSON往返。

## 新增和修改文件

新增源码/配置/测试/文档：

- `src/comparison/models.py`
- `src/comparison/snapshot.py`
- `src/comparison/engine.py`
- `src/comparison/gate.py`
- `src/comparison/demo.py`
- `config/regression_gate.yaml`
- `config/regression_gate_demo.yaml`
- `tests/test_comparison.py`
- `docs/sprint8-validation.md`

修改：`src/comparison/__init__.py`、`requirements.txt`（PyYAML>=6.0,<7）、`README.md`。

新增产物：

- `artifacts/comparison/baseline.json`（现有Sprint 7报告快照）
- `artifacts/sprint8/baseline.json`
- `artifacts/sprint8/candidate.json`
- `artifacts/sprint8/baseline_snapshot.json`
- `artifacts/sprint8/comparison.json`
- `artifacts/sprint8/gate_pass.json`
- `artifacts/sprint8/gate_fail.json`
- `artifacts/sprint8/current_report_gate.json`
- `artifacts/sprint8/pytest.txt`

数据文件保持原SHA-256：

```text
corpus:  989ef975f4b8ded581b74eb75624dc19358e015b1c5b6396471fa4334227cc2e
dataset: 7bf63862ff4730d5884f92c20eb63888f58407d3785c24c8d1750e1d4cc7ba3f
```

## 边界与风险

本阶段提供确定性阈值门禁，不做统计显著性判断。样本集合可比性检查采取保守策略，
Judge可用样本变化时应先检查失败记录，再判断版本效果。阈值是示例配置，需要业务方按目标调整。
快照不签名，也不重新执行原评测；应保留原始report作为证据。单文件原子替换不提供多进程锁。
真实Judge仍未配置，模拟报告不能替代真实RAG+Judge版本对比；本阶段验收基于实现、
离线测试和模拟实验，真实质量门禁没有获准通过。
