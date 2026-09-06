"""Report visualization only: no Runner, Retriever, or LLM client imports."""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.comparison import apply_gate, compare_reports, load_baseline, load_gate_config
from .data import ReportError, expected_answers, filter_cases, mode_notice, parse_report, report_catalog
from .export import excel_bytes, json_bytes

ROOT = Path(__file__).resolve().parents[2]
GREEN = "#285C4D"


def _fmt(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _cards(values):
    for column, (label, value) in zip(st.columns(len(values)), values):
        column.metric(label, value, border=True)


def regression_rows(comparison):
    return [dict(metric=name, baseline=item.baseline_value, candidate=item.candidate_value,
                 delta=item.delta,
                 status="不可比较" if not item.comparable else ("REGRESSION" if item.regression else "IMPROVEMENT" if item.improvement else "UNCHANGED"),
                 source="SIMULATION" if item.synthetic else "LIVE", reason=item.reason)
            for name, item in comparison.metrics.items()]


def _regression(report, catalog, artifacts):
    st.subheader("Regression")
    st.caption("当前报告作为 candidate；指标方向与门禁结论分别展示。")
    options = dict(catalog)
    for path in sorted(artifacts.rglob("*snapshot.json")) + [artifacts / "comparison/baseline.json"]:
        if not path.exists():
            continue
        try:
            options[str(path)] = load_baseline(path)
        except ValueError:
            continue
    upload = st.file_uploader("上传 baseline 快照或完整报告（可选）", type=["json"], key="baseline_upload")
    baseline = None
    if upload is not None:
        from src.comparison.models import BaselineSnapshot
        try:
            raw = upload.getvalue()
            try:
                baseline = parse_report(raw)
            except ReportError:
                baseline = BaselineSnapshot.model_validate_json(raw)
        except ValueError as exc:
            st.error(f"Baseline 无效：{exc}")
            return
    elif options:
        default = str(artifacts / "comparison/baseline.json")
        keys = list(options)
        selected = st.selectbox("Baseline", keys, index=keys.index(default) if default in keys else 0,
                                format_func=lambda p: str(Path(p).relative_to(artifacts)), key="baseline")
        baseline = options[selected]
    else:
        st.info("没有 baseline。可上传 Sprint 8 快照或完整 EvaluationReport。")
        return
    comparison = compare_reports(baseline, report)
    if comparison.warnings:
        st.warning("；".join(comparison.warnings))
    if any(metric.synthetic for metric in comparison.metrics.values()):
        st.warning("SIMULATION：标记为模拟的指标不能作为真实质量提升证据。")
    st.dataframe(pd.DataFrame(regression_rows(comparison)), hide_index=True, width="stretch")
    try:
        gate = apply_gate(comparison, load_gate_config(ROOT / "config/regression_gate.yaml"))
        (st.success if gate.status == "PASS" else st.error)(f"REGRESSION GATE: {gate.status}")
        for reason in gate.reasons:
            st.text(reason)
    except ValueError as exc:
        st.error(f"Gate 配置无效：{exc}")


def _case_detail(report, answers):
    st.subheader("Case detail")
    category_col, status_col, hallucination_col = st.columns(3)
    category = category_col.selectbox("Category", ["全部", *sorted({r.category for r in report.case_results})])
    status = status_col.selectbox("Status", ["全部", "SUCCESS", "FAIL"])
    hallucination = hallucination_col.selectbox("Hallucination（MOCK 时仅为占位标记）", ["全部", "是", "否", "未评判"])
    rows = filter_cases(report, category, status, hallucination)
    st.caption(f"匹配 {len(rows)} / {len(report.case_results)} 条；筛选仅影响 case 列表，汇总及下载保留完整报告。")
    if not rows:
        st.info("没有符合筛选条件的 case。")
        return
    st.dataframe(pd.DataFrame([dict(case_id=r.case_id, category=r.category, status=r.status,
                                  hallucination=r.judge_result.hallucination if r.judge_result else None,
                                  latency_ms=r.latency_ms, query=r.query) for r in rows]),
                 hide_index=True, width="stretch", height=240)
    selected = st.selectbox("Case ID", [row.case_id for row in rows])
    row = next(row for row in rows if row.case_id == selected)
    st.caption(f"{row.category} · {row.status} · answerable={row.answerable} · {row.latency_ms:.1f} ms")
    st.markdown("**Query**")
    st.text(row.query)
    left, right = st.columns(2)
    with left:
        st.markdown("**Expected answer**")
        st.text(answers.get(row.case_id, "N/A：匹配的数据集不可用"))
    with right:
        st.markdown("**RAG answer**")
        st.text(row.rag_result.rag_answer if row.rag_result else "N/A：未生成答案")
    st.markdown("**Retrieved chunks**")
    if row.rag_result and row.rag_result.retrieved_docs:
        for doc in row.rag_result.retrieved_docs:
            with st.expander(f"#{doc.rank} · {doc.chunk_id} · score {doc.score:.4f}"):
                st.text(f"doc_id: {doc.doc_id}")
                st.text(doc.content)
    else:
        st.info("没有检索结果。")
    left, right = st.columns(2)
    with left:
        st.markdown("**Retrieval metrics**")
        if row.retrieval_metrics:
            st.json(row.retrieval_metrics.model_dump(mode="json"), expanded=False)
        else:
            st.text("EXCLUDED：不可回答 case" if not row.answerable else "N/A：指标不可用")
    with right:
        st.markdown(f"**Judge result · {report.judge_mode}**")
        if row.judge_result:
            st.json(row.judge_result.model_dump(mode="json"), expanded=False)
        else:
            st.text("N/A：Judge 未成功")
    with st.expander("Error status / execution detail", expanded=row.status != "SUCCESS"):
        st.json(dict(status=row.status, error_type=row.error_type,
                     errors=[e.model_dump(mode="json") for e in row.errors], retry_count=row.retry_count))


def render_report(report, catalog, artifacts):
    notice = mode_notice(report)
    if report.judge_mode == "MOCK" or report.rag_mode == "MOCK" or report.quality_metrics_are_synthetic:
        st.warning(notice)
    else:
        st.info(notice)
    e, r, g, safety = report.summary.engineering, report.summary.retrieval, report.summary.generation, report.summary.safety
    st.subheader("Overview")
    st.caption(f"评测时间 {report.timestamp.isoformat()} · Top-K {report.top_k}")
    _cards([("Total cases", e.total_cases), ("Success cases", e.success_cases),
            ("Failure cases", e.failed_cases), ("Evaluation success", _fmt(e.evaluation_success_rate, True))])
    with st.expander("Dataset / corpus version & models"):
        st.text(f"Dataset version: {report.dataset_version}")
        st.text(f"Corpus version: sha256:{report.corpus_sha256}")
        st.text(f"Embedding: {report.embedding_model}\nRAG: {report.rag_model}\nJudge: {report.judge_model}\nPrompt: {report.judge_prompt_version}")
    overview, regression, detail = st.tabs(["Metrics", "Regression", "Case detail"])
    with overview:
        st.subheader("Retrieval metrics")
        st.caption(f"{r.evaluated_retrieval_cases} 条已评测 · {r.excluded_unanswerable_cases} 条不可回答排除 · {r.unavailable_retrieval_cases} 条不可用。MRR 截断至 K={report.top_k}。")
        _cards([(f"Recall@{report.top_k}", _fmt(r.recall_at_k)), (f"Precision@{report.top_k}", _fmt(r.precision_at_k)),
                (f"MRR@{report.top_k}", _fmt(r.mrr))])
        left, right = st.columns(2)
        with left:
            st.markdown("**检索指标对照**")
            values = [{"metric": name, "value": value} for name, value in
                      [(f"Recall@{report.top_k}", r.recall_at_k), (f"Precision@{report.top_k}", r.precision_at_k),
                       (f"MRR@{report.top_k}", r.mrr)] if value is not None]
            if values:
                chart = alt.Chart(pd.DataFrame(values)).mark_bar(color=GREEN, cornerRadiusEnd=3).encode(
                    x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                    y=alt.Y("metric:N", title=None, sort=None), tooltip=["metric", alt.Tooltip("value:Q", format=".4f")]).properties(height=190)
                st.altair_chart(chart, width="stretch")
            else:
                st.info("没有可展示的检索指标。")
        with right:
            st.markdown("**Case latency 分布**")
            if report.case_results:
                chart = alt.Chart(pd.DataFrame({"latency_ms": [c.latency_ms for c in report.case_results]})).mark_bar(color="#758D6C").encode(
                    x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=12), title="Latency (ms)"),
                    y=alt.Y("count():Q", title="Cases", axis=alt.Axis(tickMinStep=1)), tooltip=["count():Q"]).properties(height=190)
                st.altair_chart(chart, width="stretch")
                st.caption("包含失败 case；MOCK 步骤下的耗时不能代表全 LIVE 流水线。")
            else:
                st.info("没有 case，无法展示分布。")
        st.subheader("Generation metrics")
        if report.judge_mode == "MOCK" or report.quality_metrics_are_synthetic:
            st.warning("JUDGE MODE: MOCK" if report.judge_mode == "MOCK" else "SYNTHETIC QUALITY METRICS")
            st.caption("以下为模拟占位分数和幻觉标记，不是真实裁判结果。")
        st.caption(f"Judge 成功样本数：{g.evaluated_judge_cases}，四项评分范围 0–5；缺失显示 N/A。")
        _cards([("Correctness", _fmt(g.avg_correctness)), ("Faithfulness", _fmt(g.avg_faithfulness)),
                ("Relevance", _fmt(g.avg_relevance)), ("Completeness", _fmt(g.avg_completeness))])
        _cards([("Hallucination rate", _fmt(safety.hallucination_rate, True)),
                ("Refusal accuracy", _fmt(safety.refusal_accuracy, True)), ("Average latency (ms)", _fmt(e.average_latency))])
    answers, warning = expected_answers(report, ROOT / "data/datasets")
    with regression:
        _regression(report, catalog, artifacts)
    with detail:
        if warning:
            st.warning(warning)
        _case_detail(report, answers)
    st.divider()
    st.subheader("Report export")
    st.caption("下载当前完整报告；Excel 保留运行模式和缺失状态，JSON 保留全部嵌套追溯。")
    left, right = st.columns(2)
    left.download_button("下载完整 JSON", json_bytes(report), "evaluation_report.json", "application/json")
    right.download_button("下载 Excel", excel_bytes(report, answers, warning), "evaluation_report.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main():
    st.set_page_config(page_title="RAGEval · Evaluation Dashboard", page_icon="📊", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top:2rem; padding-bottom:2rem; max-width:1400px}
    h1,h2,h3 {font-family:'Segoe UI','Microsoft YaHei',sans-serif; letter-spacing:-.02em}
    [data-testid="stMetric"] {box-shadow:none; border-radius:6px; font-variant-numeric:tabular-nums}
    [data-testid="stMetricValue"] {font-size:1.8rem}
    button:active {transform:scale(.98)}
    @media (prefers-reduced-motion:reduce) {button:active {transform:none}}
    @media (max-width:640px) {h1 {font-size:1.8rem} .block-container {padding-top:1rem}}
    </style>""", unsafe_allow_html=True)
    st.title("RAG Evaluation Dashboard")
    st.caption("RAGEval / 已保存报告的质量、回归与 case 证据")
    artifacts = Path(os.environ.get("RAGEVAL_DASHBOARD_ARTIFACTS", ROOT / "artifacts"))
    catalog = report_catalog(artifacts)
    with st.sidebar:
        st.header("RAGEval")
        st.caption("Evaluation workspace")
        upload = st.file_uploader("上传 EvaluationReport JSON", type=["json"], key="report_upload")
        report = None
        if upload is not None:
            try:
                report = parse_report(upload.getvalue())
            except ReportError as exc:
                st.error(str(exc))
                st.stop()
        elif catalog:
            ks = sorted({r.top_k for r in catalog.values()})
            k = st.selectbox("Top-K（已有报告）", ks, index=ks.index(3) if 3 in ks else 0)
            keys = [name for name, r in catalog.items() if r.top_k == k]
            default = str(artifacts / "evaluation/latest_report.json")
            selected = st.selectbox("Report", keys, index=keys.index(default) if default in keys else 0,
                                    format_func=lambda name: str(Path(name).relative_to(artifacts)))
            report = catalog[selected]
            st.caption("只切换已有报告，不触发评测。不同版本的 K 不能当成同一系统的重算结果。")
        st.divider()
        st.caption("报告来源：本地 artifacts 或上传文件。页面不读取 API Key。")
    if report is None:
        st.info("未找到完整 EvaluationReport。请上传报告，或将 Runner 输出放入 artifacts。")
        return
    render_report(report, catalog, artifacts)
