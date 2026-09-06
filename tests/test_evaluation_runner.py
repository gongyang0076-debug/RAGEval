import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from config.judge import JudgeSettings
from src.dataset import EvalCase, RAGResult, RetrievedDocument
from src.evaluation.aggregation import aggregate_results
from src.evaluation.cache import JudgeCache
from src.evaluation.models import EvaluationReport
from src.evaluation.runner import EvaluationRunner
from src.judge import JudgeAPIError, JudgeResult
from src.judge.models import JudgeAttempt, JudgeFailure, JudgeMetadata
from src.rag.client import RAGRetrievalError


def make_case(case_id="a", answerable=True):
    return EvalCase(id=case_id, query=f"Question {case_id}", expected_answer="Expected answer",
                    relevant_doc_ids=["a"] if answerable else [], answerable=answerable,
                    category="normal" if answerable else "unanswerable")


def rag_result(query):
    return RAGResult(query=query, rag_answer="无法确定。", latency_ms=10, retrieved_docs=[
        RetrievedDocument(doc_id="parent", chunk_id=chunk, content=f"Content {chunk}", score=0.8, rank=rank)
        for rank, chunk in enumerate(["a", "b", "c"], 1)
    ])


def judge_result(value, *, refusal=True, hallucination=False):
    return JudgeResult(
        answer_correctness=4, faithfulness=5, answer_relevance=3, completeness=2,
        hallucination=hallucination, unsupported_claims=["unsupported"] if hallucination else [],
        refusal_detected=refusal, reason="Test fixture, not real judgment",
        metadata=JudgeMetadata(case_id=value.case_id, judge_model="fixture-judge", prompt_version="judge_v2",
                               response_format="json", latency_ms=20, attempts=1, raw_output="fixture",
                               attempt_history=[JudgeAttempt(attempt=1, raw_output="fixture")]),
    )


@pytest.fixture(autouse=True)
def forbid_llm_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Runner unit tests cannot call a real LLM")
    monkeypatch.setattr(httpx.Client, "send", forbidden)


@pytest.fixture
def setup(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([
        dict(doc_id="parent", chunk_id=chunk, content=f"Content {chunk}", metadata={}) for chunk in ["a", "b", "c"]
    ]), encoding="utf-8")
    rag = Mock()
    rag.run.side_effect = lambda query, top_k: rag_result(query)
    judge = Mock()
    judge.settings = JudgeSettings(model="fixture-judge", base_url="https://example.invalid/v1")
    judge.prompt_version = "judge_v2"
    judge.judge.side_effect = judge_result
    runner = EvaluationRunner(
        rag, judge, corpus_path=corpus_path, embedding_model="fixture-embedding", rag_model="fixture-rag",
        output_dir=tmp_path / "output", cache=JudgeCache(tmp_path / "cache"), rag_mode="MOCK", judge_mode="MOCK",
    )
    return SimpleNamespace(runner=runner, rag=rag, judge=judge, tmp=tmp_path)


def write_dataset(setup, cases):
    path = setup.tmp / "dataset_v1.json"
    path.write_text(json.dumps([case.model_dump() for case in cases]), encoding="utf-8")
    return path


def test_single_case_full_pipeline_and_serialization(setup):
    value = make_case()
    row = setup.runner.evaluate_case(value, 3)
    assert row.status == "SUCCESS" and row.error_type is None
    assert row.retrieval_metrics.recall_at_k == 1
    assert row.retrieval_metrics.precision_at_k == pytest.approx(1 / 3)
    assert row.judge_result.answer_correctness == 4
    assert row.refusal_correct is None
    assert row.latency_ms >= 0
    setup.rag.run.assert_called_once_with(value.query, 3)
    sent = setup.judge.judge.call_args.args[0]
    assert sent.expected_answer == value.expected_answer
    assert json.loads(sent.retrieved_context)[0] == {"doc_id": "parent", "chunk_id": "a", "content": "Content a"}


def test_dataset_report_files_and_aggregation(setup):
    cases = [make_case(), make_case("unknown", False)]
    report = setup.runner.evaluate_dataset(write_dataset(setup, cases), 3)
    assert report.summary.engineering.total_cases == 2
    assert report.summary.engineering.success_cases == 2
    assert report.summary.engineering.failed_cases == 0
    assert report.summary.engineering.evaluation_success_rate == 1
    assert report.summary.retrieval.evaluated_retrieval_cases == 1
    assert report.summary.retrieval.excluded_unanswerable_cases == 1
    assert report.summary.generation.avg_correctness == 4
    assert report.summary.generation.avg_faithfulness == 5
    assert report.summary.generation.avg_relevance == 3
    assert report.summary.generation.avg_completeness == 2
    assert report.summary.safety.refusal_accuracy == 1
    assert report.timestamp.tzinfo is not None
    assert report.dataset_version.startswith("dataset_v1@sha256:")
    assert report.judge_prompt_version == "judge_v2"
    assert report.embedding_model == "fixture-embedding"
    assert report.quality_metrics_are_synthetic is True
    stored = EvaluationReport.model_validate_json((setup.tmp / "output/latest_report.json").read_text(encoding="utf-8"))
    assert stored == report
    assert len(json.loads((setup.tmp / "output/case_results.json").read_text(encoding="utf-8"))) == 2


def test_rag_failure_does_not_stop_other_cases(setup):
    setup.rag.run.side_effect = [RuntimeError("generation failed"), rag_result("Question b")]
    report = setup.runner.evaluate_dataset(write_dataset(setup, [make_case(), make_case("b")]), 3)
    failed, succeeded = report.case_results
    assert failed.status == "RAG_ERROR"
    assert failed.error_type == "RuntimeError"
    assert failed.rag_result is None
    assert succeeded.status == "SUCCESS"
    assert report.summary.engineering.failed_cases == 1
    assert report.summary.engineering.evaluation_success_rate == 0.5
    assert report.summary.retrieval.unavailable_retrieval_cases == 1
    assert report.summary.generation.evaluated_judge_cases == 1


def test_retrieval_inside_rag_run_is_classified(setup):
    error = RAGRetrievalError("retrieval failed")
    error.__cause__ = ValueError("empty index")
    setup.rag.run.side_effect = error
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "RETRIEVAL_ERROR"
    assert row.errors[0].stage == "retrieval"
    assert row.error_type == "ValueError"
    setup.judge.judge.assert_not_called()


def test_bad_retrieval_ranks_preserve_rag_and_still_judge(setup):
    bad = rag_result("Question a")
    bad.retrieved_docs[0].rank = 2
    setup.rag.run.side_effect = None
    setup.rag.run.return_value = bad
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "RETRIEVAL_ERROR"
    assert row.rag_result is not None
    assert row.retrieval_metrics is None
    assert row.judge_result is not None
    assert aggregate_results([row]).generation.evaluated_judge_cases == 1


def test_judge_failure_keeps_successful_retrieval(setup):
    setup.judge.judge.side_effect = RuntimeError("Judge unavailable")
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "JUDGE_ERROR"
    assert row.retrieval_metrics.recall_at_k == 1
    assert row.judge_result is None
    summary = aggregate_results([row])
    assert summary.retrieval.evaluated_retrieval_cases == 1
    assert summary.generation.avg_correctness is None
    assert summary.safety.hallucination_rate is None


@pytest.mark.parametrize("stage", ["rag", "retrieval", "judge"])
def test_timeout_status_preserves_stage(setup, stage):
    error = TimeoutError("test timeout")
    if stage == "retrieval":
        wrapped = RAGRetrievalError("retrieval failed")
        wrapped.__cause__ = error
        setup.rag.run.side_effect = wrapped
    elif stage == "rag":
        setup.rag.run.side_effect = error
    else:
        setup.judge.judge.side_effect = error
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "TIMEOUT"
    assert row.errors[0].stage == stage


def test_judge_api_error_metadata_is_preserved(setup):
    value = SimpleNamespace(case_id="a")
    metadata = judge_result(value).metadata
    failure = JudgeFailure(category="timeout", message="APITimeoutError", metadata=metadata)
    setup.judge.judge.side_effect = JudgeAPIError(failure)
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "TIMEOUT"
    assert row.errors[0].judge_failure == failure


@pytest.mark.parametrize("refusal,hallucination,correct", [(True, False, True), (False, False, False), (True, True, False)])
def test_unanswerable_refusal_requires_explicit_refusal_without_hallucination(setup, refusal, hallucination, correct):
    setup.judge.judge.side_effect = lambda value: judge_result(value, refusal=refusal, hallucination=hallucination)
    with patch("src.evaluation.runner.evaluate_retrieval") as metric:
        row = setup.runner.evaluate_case(make_case("unknown", False), 3)
        metric.assert_not_called()
    setup.rag.run.assert_called_once()
    setup.judge.judge.assert_called_once()
    assert row.retrieval_metrics is None
    assert row.refusal_correct is correct


def test_refusal_denominator_includes_failed_unanswerable_cases(setup):
    good = setup.runner.evaluate_case(make_case("u1", False), 3)
    setup.judge.judge.side_effect = RuntimeError("Judge failure")
    failed = setup.runner.evaluate_case(make_case("u2", False), 3)
    summary = aggregate_results([good, failed])
    assert summary.safety.unanswerable_cases == 2
    assert summary.safety.correct_refusal_cases == 1
    assert summary.safety.unassessed_refusal_cases == 1
    assert summary.safety.refusal_accuracy == 0.5
    assert summary.generation.evaluated_judge_cases == 1


def test_cache_hit_avoids_judge_but_still_runs_rag(setup):
    first = setup.runner.evaluate_case(make_case(), 3)
    second = setup.runner.evaluate_case(make_case(), 3)
    assert first.judge_cache_hit is False
    assert second.judge_cache_hit is True
    assert setup.rag.run.call_count == 2
    assert setup.judge.judge.call_count == 1
    assert second.judge_result == first.judge_result


@pytest.mark.parametrize("changed", ["expected_answer", "rag_answer", "context"])
def test_changed_case_or_rag_input_rejudges(setup, changed):
    case = make_case()
    setup.runner.evaluate_case(case, 3)
    if changed == "expected_answer":
        case.expected_answer = "Changed expected answer"
    else:
        value = rag_result(case.query)
        if changed == "rag_answer":
            value.rag_answer = "Changed answer"
        else:
            value.retrieved_docs[0].content = "Changed context"
        setup.rag.run.side_effect = None
        setup.rag.run.return_value = value
    row = setup.runner.evaluate_case(case, 3)
    assert not row.judge_cache_hit
    assert setup.judge.judge.call_count == 2


def test_corrupt_cache_rejudges_and_reports_warning(setup):
    setup.runner.evaluate_case(make_case(), 3)
    file = next((setup.tmp / "cache").glob("*.json"))
    file.write_text("broken", encoding="utf-8")
    row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "SUCCESS"
    assert not row.judge_cache_hit
    assert row.cache_warning is not None
    assert setup.judge.judge.call_count == 2


def test_judge_failures_are_not_cached(setup):
    setup.judge.judge.side_effect = RuntimeError("failure")
    setup.runner.evaluate_case(make_case(), 3)
    setup.runner.evaluate_case(make_case(), 3)
    assert setup.judge.judge.call_count == 2
    assert not list((setup.tmp / "cache").glob("*.json"))


def test_cache_write_failure_does_not_discard_success(setup):
    with patch.object(setup.runner.cache, "put", return_value="cache write failed"):
        row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "SUCCESS"
    assert row.cache_warning == "cache write failed"


def test_missing_v2_refusal_is_judge_error(setup):
    def old_result(value):
        result = judge_result(value)
        result.refusal_detected = None
        return result
    setup.judge.judge.side_effect = old_result
    assert setup.runner.evaluate_case(make_case("unknown", False), 3).status == "JUDGE_ERROR"


def test_multiple_stage_errors_are_not_lost(setup):
    with patch("src.evaluation.runner.evaluate_retrieval", side_effect=ValueError("bad metrics")):
        setup.judge.judge.side_effect = RuntimeError("bad judge")
        row = setup.runner.evaluate_case(make_case(), 3)
    assert row.status == "RETRIEVAL_ERROR"
    assert [error.stage for error in row.errors] == ["retrieval", "judge"]


def test_empty_dataset_null_aggregates(setup):
    report = setup.runner.evaluate_dataset(write_dataset(setup, []), 3)
    assert report.summary.engineering.total_cases == 0
    assert report.summary.engineering.evaluation_success_rate is None
    assert report.summary.safety.refusal_accuracy is None


@pytest.mark.parametrize("k", [0, -1, True])
def test_invalid_top_k_fails_before_work(setup, k):
    with pytest.raises(ValueError, match="positive integer"):
        setup.runner.evaluate_dataset(write_dataset(setup, [make_case()]), k)
    setup.rag.run.assert_not_called()


def test_real_provider_timeout_trace_is_saved_and_runner_continues(setup):
    from openai import APITimeoutError
    from unittest.mock import AsyncMock
    from src.llm.openai_compatible import OpenAICompatibleProvider
    from src.rag import DemoRAGClient
    from tests.provider_fakes import async_sdk, completion
    sdk = async_sdk()
    error = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    sdk.chat.completions.create.side_effect = [error, error, error, completion("Recovered")]
    retriever = Mock()
    retriever.retrieve.return_value = rag_result("Question").retrieved_docs
    provider = OpenAICompatibleProvider(api_key="fake", base_url="https://example.invalid", model="fixture-rag")
    setup.runner.rag_client = DemoRAGClient(retriever, provider)
    with patch("src.llm.openai_compatible.AsyncOpenAI", return_value=sdk), patch("src.llm.openai_compatible.sleep", new_callable=AsyncMock):
        report = setup.runner.evaluate_dataset(write_dataset(setup, [make_case("a"), make_case("b")]), 3)
    first, second = report.case_results
    assert first.status == "TIMEOUT" and second.status == "SUCCESS"
    assert first.retry_count == first.rag_retry_count == 2
    assert first.errors[0].provider_failure.category == "timeout"
    assert len(first.errors[0].provider_failure.trace.attempts) == 3
    assert report.summary.engineering.failure_breakdown == {"APITimeoutError": 1}
    assert report.summary.engineering.retry_count == 2
    assert sdk.chat.completions.create.call_count == 4 and retriever.retrieve.call_count == 2
    saved = EvaluationReport.model_validate_json((setup.tmp / "output/latest_report.json").read_text(encoding="utf-8"))
    assert saved.case_results[0].errors[0].provider_failure == first.errors[0].provider_failure


def test_cache_hit_does_not_count_historical_judge_retries(setup):
    def judged(value):
        result = judge_result(value)
        result.metadata.retry_count = 2
        return result
    setup.judge.judge.side_effect = judged
    first = setup.runner.evaluate_case(make_case(), 3)
    cached = setup.runner.evaluate_case(make_case(), 3)
    assert first.judge_retry_count == first.retry_count == 2
    assert cached.judge_cache_hit and cached.judge_result.metadata.retry_count == 2
    assert cached.judge_retry_count == cached.retry_count == 0
