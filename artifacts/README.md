# 已审核的报告示例

这里只发布 `.gitignore` 白名单中明确列出的报告。新报告和原始日志默认忽略；
加入白名单前须检查密钥、个人路径和业务文本。

| 路径 | 来源与用途 |
| --- | --- |
| `evaluation/latest_report.json` | Sprint 7真实BGE/Chroma检索及GLM生成，MOCK Judge，默认Dashboard数据 |
| `evaluation/case_results.json` | 同一轮逐case明细 |
| `evaluation/attempt1/`、`evaluation/sprint6/` | Sprint 6调用失败的历史证据，不代表当前版本 |
| `sprint4/retrieval_benchmark.json` | 48条answerable case的真实Top1/3/5检索结果，无LLM |
| `sprint5/live_judge_smoke.json`、`sprint7/live_judge_validation.json` | 当时没有Judge配置，未伪造live评分 |
| `sprint7/provider_probe.json`、`sprint7/evaluation/` | Provider诊断及同轮报告冻结归档 |
| `comparison/baseline.json` | Sprint 7报告的快照，保留MOCK标记 |
| `sprint8/` | 受控MOCK排名、生成、Judge及延迟的K=3/K=1门禁演示 |
| `sprint9/export/` | Dashboard导出示例：完整JSON与Excel四表 |

没有模型权重或Chroma数据库。历史快照有少量有意重复，便于复核各Sprint；
默认Dashboard只读这些产物，不调用API。公开日志已将本机目录替换为通用占位路径，
评测数值及成功/失败状态不变。这些是合成电商语料上的历史实验，不是生产质量承诺。
