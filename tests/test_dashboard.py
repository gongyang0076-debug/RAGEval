"""Streamlit AppTest validates rendered elements without a browser or LLM."""

import importlib
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.comparison.demo import simulated_report
from src.evaluation.aggregation import aggregate_results
from src.report.export import json_bytes


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard must not call APIs")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setenv("RAGEVAL_DASHBOARD_ARTIFACTS", str(tmp_path))
    return tmp_path


def write_report(directory, k=3, empty=False):
    report = simulated_report(Path("data/corpus/ecommerce_v1.json"), Path("data/datasets/ecommerce_eval_v1.json"), k)
    if empty:
        report.case_results = []
        report.summary = aggregate_results([])
    (directory / ("baseline.json" if k == 3 else "candidate.json")).write_bytes(json_bytes(report))
    return report


def test_key_modules_import_without_evaluation():
    for module in ("app", "src.report.dashboard", "src.report.export"):
        assert importlib.import_module(module)


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


def test_k_selection_displays_existing_report_without_recalculation(dashboard):
    write_report(dashboard, 3)
    write_report(dashboard, 1)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    select = next(s for s in app.selectbox if s.label == "Top-K（已有报告）")
    select.select(1).run()
    assert not app.exception
    assert any(m.label == "Recall@1" and m.value == "0.740" for m in app.metric)


def test_empty_filter_shows_message(dashboard):
    write_report(dashboard)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    next(s for s in app.selectbox if s.label == "Status").select("FAIL").run()
    assert not app.exception
    assert any("没有符合筛选条件" in i.value for i in app.info)


def test_no_report_renders_clear_empty_state(dashboard):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert any("未找到完整 EvaluationReport" in i.value for i in app.info)


def test_empty_report_renders_na_instead_of_zero(dashboard):
    write_report(dashboard, empty=True)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label == "Recall@3") == "N/A"
