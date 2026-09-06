# Release Check

## 项目状态

**Sprint 10：本地发布准备完成。** 目标仓库为 [gongyang0076-debug/RAGEval](https://github.com/gongyang0076-debug/RAGEval)。

本地已初始化 `main` 并配置该仓库为 `origin`。本轮只读 `git ls-remote` 成功且未返回任何ref：
检查时远程为空。本地尚无提交，未执行commit、push、创建tag或GitHub Release；线上Actions尚未运行。
本记录的PASS表示发布准备与本地验证通过，不表示代码已经公开发布，也不是生成质量门禁PASS。

项目版本字段仍为 `0.1.0`，交付方式是GitHub源码仓库。未发布PyPI包，未附开源许可证。
如需允许第三方按开源许可证使用，应由维护者选择并补充许可证。

## 本阶段变更

- README改为项目介绍、Features、Mermaid架构、Quick Start、数据集、真实/模拟Benchmark、
  三张Dashboard截图、开发与CI、目录结构。复杂API/边界说明保留到 `docs/usage-guide.md`。
- 新增 `.github/workflows/test.yml`，Python3.11、CPU依赖、模块导入、发布文件扫描、pytest。
- 新增 `scripts/check_imports.py` 与 `scripts/check_release.py`，可本地和CI复用。
- 扩展 `.gitignore`，发布产物使用明确文件白名单；保留小规模历史失败报告作为证据。
- 新增 `.gitattributes`，数据/报告JSON禁用Git换行转换，防止Windows/Linux间版本哈希失配。
- 历史验证文档与两份公开pytest日志中的绝对目录替换为通用占位符；原件留在忽略目录。
- 没有修改 `src/`、`config/`、`data/` 或已有 `tests/` 的任何文件字节。

## GitHub发布检查

| 检查项 | 结果与证据 |
| --- | --- |
| .env | 已忽略，不在拟发布集合；.env.example仅含空密钥配置 |
| 缓存/虚拟环境 | .cache、.venv、.venv-*、pytest缓存被忽略 |
| 模型/向量库 | 权重、模型目录、Chroma及数据库扩展名不进入发布集合 |
| artifacts | 仅白名单报告公开；新运行产物默认忽略，见artifacts/README.md |
| API Key | 对Git候选集合进行密钥特征与本地.env密钥精确匹配，命中0；不输出密钥 |
| 个人路径 | 公开文本和Excel XML绝对路径扫描命中0；截图已人工检查 |
| 临时文件 | 验证环境、日志原件、截图辅助脚本位于忽略目录 |
| 未使用代码 | Ruff F401/F841/F821/F822/F823检查通过 |
| Debug输出 | 未发现pdb/breakpoint或遗留调试输出；CLI及Runner进度print按设计保留 |
| 数据跨平台一致性 | 两份正式数据git hash-object过滤前/后结果一致，text属性为unset |
| 历史提交 | 本地与远程均无历史提交，因此不存在需要扫描的旧提交 |
| 推送/线上CI | 本轮未执行推送，线上CI未运行，不宣称绿色构建 |

文件检查使用 `git ls-files --cached --others --exclude-standard`，包含尚未提交的候选文件，
不是仅检查Git索引。拟发布集合约3.4MiB，最大单文件不足1MiB，不含大模型文件。
扫描是已定义规则的检查，不是对未来引入数据或所有编码形式秘密的绝对保证。

## CI配置

触发：main分支push、pull_request、workflow_dispatch。使用Ubuntu runner、Python3.11，
`contents: read` 权限，checkout不持久化凭据，25分钟任务超时。

依赖安装先从PyTorch CPU索引安装torch，再安装requirements。设置Hugging Face离线变量，
确保测试使用固定向量/mock，不下载Embedding模型；没有RAG/Judge密钥需求。

```text
checkout → setup-python 3.11 → CPU dependencies
         → check_imports.py → check_release.py → pytest -q
```

Action用法依据官方 [checkout](https://github.com/actions/checkout) 与
[setup-python](https://github.com/actions/setup-python)；当前配置使用v6。
本地Windows Python3.11验证不能代替实际Ubuntu Actions结果，首次推送后应查看任务日志。

## 测试结果

### 隔离发布副本，Python 3.11.15

从Git拟发布集合复制119个当时已有文件到全新目录，副本不含.env、虚拟环境或模型缓存。
使用项目内另建的Python3.11环境安装requirements，执行：

```text
python scripts/check_imports.py
IMPORT CHECK: PASS (45 modules)

python -m pytest -q --basetemp .test-temp
........................................................................ [ 20%]
........................................................................ [ 41%]
........................................................................ [ 61%]
........................................................................ [ 82%]
..............................................................           [100%]
============================== warnings summary ===============================
tests/test_retriever.py: 20 warnings
  <python-env>/Lib/site-packages/chromadb/api/models/CollectionCommon.py:155:
  DeprecationWarning: legacy embedding function config
    return load_collection_configuration_from_json(self._model.configuration_json)

350 passed, 20 warnings in 22.04s
```

退出码0。随后增加的内容仅为.gitattributes、发布检查规则与本记录，已分别通过文件/换行检查，
没有改变被测试的评测代码。副本验证的是源码发布依赖的完整性，不是一次真实RAG/LLM Benchmark。

### 当前开发环境，Python 3.14

```text
python scripts/check_imports.py
IMPORT CHECK: PASS (45 modules)

python -m pytest -q --basetemp .cache/sprint10/temp-314
350 passed, 22 warnings in 25.80s
```

退出码0。21条为既有Chroma相关弃用提示；另1条是本机pytest缓存目录写入权限提示。
首次两个环境使用系统默认Temp时，各有118个fixture初始化权限错误、232项通过；
改用独立项目临时目录后全部通过，没有修改测试断言或业务代码来掩盖错误。
普通环境和CI仍使用 `python -m pytest -q`，只有本机权限冲突时需要指定独立 `--basetemp`。

### 静态检查

```text
python -m ruff check src config app.py tests scripts --select F401,F841,F821,F822,F823
All checks passed!
```

Ruff仅作为本轮开发检查工具，不是应用运行依赖。没有为了满足格式工具重写核心代码。
README与docs相对链接检查通过；正式Corpus/Dataset及核心模块、已有测试哈希逐文件保持不变。

## Release Gate 2.0记录

| 项目 | 状态 |
| --- | --- |
| 检查基线 | 首次Git初始化，无HEAD/tag；按工作区全部拟发布文件检查 |
| 工作区 | 候选文件均未提交；本地配置与缓存已排除 |
| origin | 已配置用户指定仓库；检查时无refs |
| 版本 | pyproject.toml为0.1.0，无其他发布版本需同步 |
| 发布形式 | GitHub源码准备；PyPI、安装器、签名、appcast不适用 |
| 依赖 | Python3.11隔离环境安装成功，导入45模块通过 |
| 产物 | 已审核JSON、Excel、截图及白名单规则齐全 |
| 发布文件内容 | 拟发布集合隔离复制并运行完整测试；数据Git过滤校验通过 |
| CI | 工作流已创建，本地等价Python版本通过；线上尚未运行 |
| 公开动作 | 未commit/push/tag/release；无issue/PR操作要求 |

## 已知限制

1. **Judge真实质量依赖外部模型**及Prompt。当前公开报告为MOCK Judge，不能宣称高正确率、
   零幻觉或真实拒答100%；默认真实质量门禁仍FAIL。
2. **Demo corpus为合成数据**：36块、60条case、48条参与检索评测。小样本结果不代表生产泛化。
3. **Chroma warning来自第三方依赖**；不同Python版本的警告数不同，本轮未通过屏蔽警告掩盖问题。
4. CI尚未在远程运行；依赖采用兼容版本范围而非完全锁定，未来解析结果可能变化。
5. Dashboard面向本地使用，无远程鉴权；同步Provider不直接运行于已有asyncio事件循环中。
6. Excel超长单元格会显式截断，完整JSON为权威记录；平均延迟受缓存、服务端与硬件影响。
7. 未附开源许可证，当前只声明可公开展示源码，不擅自选择第三方再分发授权。

## 简历建议

- 开发RAGEval自动化评测框架，覆盖RAG检索、生成、安全与工程稳定性，使用Pydantic建立统一数据契约。
- 实现Recall/Precision/MRR、版本化LLM Judge、分类重试、超时追溯、Judge缓存及Baseline回归门禁。
- 基于36个合成知识块和60条标注case构建可复核实验，完成350项自动化测试、
  Python3.11 CI配置与Streamlit/JSON/Excel结果展示。

面试说明：真实检索结果可引用；MOCK Judge仅用于验证流水线，尚未完成真实生成质量验收。

SPRINT 10 ACCEPTANCE: PASS

## 后续 GLM Judge 兼容验证

Sprint 10 上述测试与 MOCK 报告是历史记录。后续已补齐 GLM `json_object` 模式，
完整测试更新为356项通过，真实 Judge 的10条调用均通过结构校验；人工核查发现评分与拒答识别误判，
因此不能把调用成功当作评分质量验收。详见 [Live Judge 验证记录](live-judge-validation.md)。
