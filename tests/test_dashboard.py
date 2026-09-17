# 文件作用：通过 Streamlit AppTest 检查页面元素、过滤、报告切换和下载入口。
# 为什么有它：验证页面展示现有报告，不依赖浏览器交互或真实模型。
"""Streamlit AppTest validates rendered elements without a browser or LLM."""

import importlib
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.comparison.demo import simulated_report
from src.evaluation.aggregation import aggregate_results
from src.report.export import json_bytes


# 做什么：隔离报告目录并阻止页面测试联网。
# 为什么需要：检查页面行为时不能读取正式运行结果或调用外部 API。
@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    # 做什么：遇到真实 HTTP 请求立即报测试失败。
    # 为什么需要：确保展示报告没有隐式启动模型调用。
    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard must not call APIs")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setenv("RAGEVAL_DASHBOARD_ARTIFACTS", str(tmp_path))
    return tmp_path


# 做什么：把指定 K 或空结果的模拟报告写入测试目录。
# 为什么需要：为页面切换和空状态测试准备可控数据。
def write_report(directory, k=3, empty=False):
    report = simulated_report(Path("data/corpus/ecommerce_v1.json"), Path("data/datasets/ecommerce_eval_v1.json"), k)
    if empty:
        report.case_results = []
        report.summary = aggregate_results([])
    (directory / ("baseline.json" if k == 3 else "candidate.json")).write_bytes(json_bytes(report))
    return report


# 做什么：验证页面与导出模块可以直接导入。
# 为什么需要：看报告不应启动评测或调用模型。
def test_key_modules_import_without_evaluation():
    for module in ("app", "src.report.dashboard", "src.report.export"):
        assert importlib.import_module(module)


# 做什么：验证模拟报告显示来源提示、指标、图表、详情和下载入口。
# 为什么需要：页面必须完整可用且不能隐藏 MOCK 身份。
def test_mock_report_renders_notice_cards_charts_details_and_exports(dashboard):
    write_report(dashboard)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert any("JUDGE MODE: MOCK" in w.value for w in app.warning)
    assert app.title[0].value == "RAG Evaluation Dashboard"
    assert any(m.label == "Recall@3" and m.value == "1.000" for m in app.metric)
    assert len(app.get("vega_lite_chart")) + len(app.get("arrow_vega_lite_chart")) == 2
    assert len(app.get("download_button")) == 2
    assert any("Expected answer" in m.value for m in app.markdown)
    assert any("REGRESSION GATE: FAIL" in e.value for e in app.error)


# 做什么：验证切换 K 时读取对应已有报告。
# 为什么需要：切换页面选项不能悄悄重新计算或调用模型。
def test_k_selection_displays_existing_report_without_recalculation(dashboard):
    write_report(dashboard, 3)
    write_report(dashboard, 1)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    select = next(s for s in app.selectbox if s.label == "Top-K（已有报告）")
    select.select(1).run()
    assert not app.exception
    assert any(m.label == "Recall@1" and m.value == "0.740" for m in app.metric)


# 做什么：验证筛选无结果时显示说明。
# 为什么需要：避免空白页面让用户误以为程序崩溃。
def test_empty_filter_shows_message(dashboard):
    write_report(dashboard)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(s for s in app.selectbox if s.label == "Status").select("FAIL").run()
    assert not app.exception
    assert any("没有符合筛选条件" in i.value for i in app.info)


# 做什么：验证没有报告时显示明确提示。
# 为什么需要：首次打开项目也应知道缺少什么。
def test_no_report_renders_clear_empty_state(dashboard):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert any("未找到完整 EvaluationReport" in i.value for i in app.info)


# 做什么：验证空报告把指标显示为不可用。
# 为什么需要：没有评测不能显示成真实零分。
def test_empty_report_renders_na_instead_of_zero(dashboard):
    write_report(dashboard, empty=True)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label == "Recall@3") == "N/A"
