# 文件作用：读取报告、匹配同版本标准答案、筛选题目并生成模式说明。
# 为什么有它：展示层只查看已有证据，避免误用其他数据版本或重新评测。
"""Read-only report views; never run evaluation or recompute metrics."""

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.dataset import EvalCase
from src.evaluation.models import EvaluationReport


# 这个类：表示报告或匹配数据不可用。
# 为什么需要：页面与导出能够给出统一错误提示。
class ReportError(ValueError):
    """Invalid report or unavailable matching source data."""


# 做什么：把上传或读取的 JSON 校验成完整报告对象。
# 为什么需要：拒绝结构不合法的展示输入。
def parse_report(content: bytes | str) -> EvaluationReport:
    try:
        return EvaluationReport.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise ReportError(f"Invalid EvaluationReport: {exc}") from exc


# 做什么：读取报告文件并统一文件错误信息。
# 为什么需要：页面和导出共享同样的加载行为。
def load_report(path: str | Path) -> EvaluationReport:
    try:
        return parse_report(Path(path).read_bytes())
    except OSError as exc:
        raise ReportError(f"Cannot read report {path}: {type(exc).__name__}") from exc


# 做什么：扫描目录中指定名称的合法完整报告。
# 为什么需要：避免把快照、逐题列表或坏文件误当报告。
def report_catalog(root: str | Path) -> dict[str, EvaluationReport]:
    reports = {}
    for path in sorted(Path(root).rglob("*.json")):
        # Snapshots and individual case exports are not complete reports.
        if path.name not in {"latest_report.json", "baseline.json", "candidate.json", "evaluation_report.json"}:
            continue
        try:
            reports[str(path)] = load_report(path)
        except ReportError:
            continue
    return reports


# 做什么：只从给定目录中匹配数据哈希完全一致的试卷取得标准答案。
# 为什么需要：避免把其他版本答案或报告中的任意路径用于展示。
def expected_answers(report: EvaluationReport, dataset_dir: str | Path) -> tuple[dict[str, str], str | None]:
    """Only read an exact-hash dataset in the supplied directory, never a report's arbitrary path."""
    for path in sorted(Path(dataset_dir).glob("*.json")):
        try:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != report.dataset_sha256:
                continue
            cases = TypeAdapter(list[EvalCase]).validate_json(content)
            answers = {case.id: case.expected_answer for case in cases}
            if len(answers) != len(cases):
                raise ValueError("Duplicate dataset case IDs")
            if any(row.case_id not in answers for row in report.case_results):
                raise ValueError("Report case IDs do not match dataset")
            return answers, None
        except (OSError, ValueError) as exc:
            return {}, f"标准答案不可用：{type(exc).__name__}，未使用其他版本的数据。"
    return {}, "标准答案不可用：未找到 SHA-256 匹配的数据集。"


# 做什么：生成真实与模拟模式的展示提示。
# 为什么需要：防止用户把 MOCK 分数或模拟耗时当作真实质量。
def mode_notice(report: EvaluationReport) -> str:
    parts = [f"RAG MODE: {report.rag_mode}", f"JUDGE MODE: {report.judge_mode}"]
    if report.judge_mode == "MOCK" or report.quality_metrics_are_synthetic:
        parts.append("Judge 分数和安全结果为模拟数据，不能解释为真实质量。")
    if report.rag_mode == "MOCK":
        parts.append("检索、生成与延迟来自模拟实验。")
    return " · ".join(parts)


# 做什么：按类别、成功失败和幻觉观察筛选题目。
# 为什么需要：帮助用户定位问题，不改变整份报告的汇总。
def filter_cases(report: EvaluationReport, category: str = "全部", status: str = "全部",
                 hallucination: str = "全部"):
    rows = []
    for row in report.case_results:
        observed = "未评判" if row.judge_result is None else ("是" if row.judge_result.hallucination else "否")
        if category != "全部" and row.category != category:
            continue
        if status == "SUCCESS" and row.status != "SUCCESS":
            continue
        if status == "FAIL" and row.status == "SUCCESS":
            continue
        if hallucination != "全部" and observed != hallucination:
            continue
        rows.append(row)
    return rows


# 做什么：把复杂字段转换成保留中文的严格 JSON 字符串。
# 为什么需要：在页面和表格中完整显示列表或错误等结构。
def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
