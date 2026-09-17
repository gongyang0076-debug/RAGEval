# 文件作用：串起数据加载、RAG、指标、Judge、缓存、失败记录和报告保存。
# 为什么有它：把零散能力变成可批量执行的评测流程，并让单条失败不影响其余题目。
"""Sequential RAG evaluation with isolated case failures and explicit mock labeling."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from openai import APIError, APITimeoutError

from config.judge import load_judge_settings
from config.settings import load_settings
from src.dataset import EvalCase, RAGResult, load_corpus, load_dataset
from src.judge import JudgeError, JudgeInput, LLMJudgeClient
from src.judge.prompts import RUNNER_PROMPT_VERSION
from src.metrics import evaluate_retrieval
from src.rag import ChromaRetriever, DemoRAGClient, RAGClient
from src.rag.client import RAGRetrievalError
from src.llm import ProviderError
from src.llm.openai_compatible import OpenAICompatibleProvider

from .aggregation import aggregate_results
from .cache import JudgeCache, atomic_write_json
from .models import CaseError, CaseEvaluationResult, EvaluationReport


# 做什么：把各层异常转换成带阶段和错误类别的统一记录。
# 为什么需要：让报告区分检索、生成、裁判和超时问题。
def _case_error(stage: str, exc: Exception) -> CaseError:
    cause = exc.__cause__ if isinstance(exc, RAGRetrievalError) and exc.__cause__ else exc
    timeout = isinstance(cause, (TimeoutError, APITimeoutError))
    if isinstance(exc, ProviderError):
        timeout = exc.details.category == "timeout"
        error_type = exc.details.trace.attempts[-1].error_type or exc.details.category
        message = exc.details.message
    elif isinstance(exc, JudgeError):
        timeout = exc.details.category == "timeout"
        error_type, message = exc.details.category, exc.details.message
    else:
        error_type = type(cause).__name__
        message = error_type if isinstance(cause, APIError) else str(cause)
    return CaseError(
        stage=stage, status="TIMEOUT" if timeout else {"rag": "RAG_ERROR", "retrieval": "RETRIEVAL_ERROR", "judge": "JUDGE_ERROR"}[stage],
        error_type=error_type, message=message,
        judge_failure=exc.details if isinstance(exc, JudgeError) else None,
        provider_failure=exc.details if isinstance(exc, ProviderError) else None,
    )


# 这个类：连接 RAG、指标和 Judge 的批量调度器。
# 为什么需要：让评测流程集中执行并隔离单题失败。
class EvaluationRunner:
    # 做什么：接收被测客户端、裁判、数据路径、缓存和运行身份。
    # 为什么需要：集中连接依赖，允许单元测试替换外部服务。
    def __init__(self, rag_client: RAGClient, judge_client, *, corpus_path: str | Path,
                 embedding_model: str, rag_model: str, output_dir: str | Path = "artifacts/evaluation",
                 cache: JudgeCache | None = None, judge_mode: str = "LIVE", rag_mode: str = "LIVE", verbose: bool = False,
                 execution_config: dict | None = None):
        if judge_mode not in {"LIVE", "MOCK"} or rag_mode not in {"LIVE", "MOCK"}:
            raise ValueError("RAG/Judge mode must be LIVE or MOCK")
        if judge_client.prompt_version != RUNNER_PROMPT_VERSION:
            raise ValueError("Evaluation Runner requires judge_v2 with explicit refusal_detected")
        self.rag_client, self.judge_client = rag_client, judge_client
        self.corpus_path = Path(corpus_path)
        self.embedding_model, self.rag_model = embedding_model, rag_model
        self.output_dir = Path(output_dir)
        self.cache = cache
        self.judge_mode, self.rag_mode = judge_mode, rag_mode
        self.verbose = verbose
        self.execution_config = execution_config or {}

    # 做什么：先读取匹配缓存，未命中时评分并保存成功结果。
    # 为什么需要：减少重复请求，同时记录缓存异常和来源。
    def _judge(self, value: JudgeInput, row: CaseEvaluationResult):
        settings = self.judge_client.settings
        key = self.cache.key(
            value, judge_model=settings.model, prompt_version=self.judge_client.prompt_version,
            judge_mode=self.judge_mode, base_url=settings.base_url, response_format=settings.response_format,
        ) if self.cache else None
        if self.cache:
            result, row.cache_warning = self.cache.get(key)
            if result is not None:
                if (result.metadata.case_id == value.case_id and result.metadata.judge_model == settings.model
                        and result.metadata.prompt_version == self.judge_client.prompt_version):
                    row.judge_cache_hit = True
                    return result
                row.cache_warning = "Ignored Judge cache with incompatible execution metadata"
        result = self.judge_client.judge(value)
        if (result.refusal_detected is None or result.metadata.case_id != value.case_id
                or result.metadata.judge_model != settings.model
                or result.metadata.prompt_version != self.judge_client.prompt_version):
            raise ValueError("Judge result has missing refusal observation or incompatible execution metadata")
        if self.cache:
            warning = self.cache.put(key, result)
            if warning:
                row.cache_warning = warning
        return result

    # 做什么：执行一条题的 RAG、指标和裁判，记录拒答、错误与耗时。
    # 为什么需要：把单题成功和失败都保留，并隔离局部故障。
    def evaluate_case(self, case: EvalCase, top_k: int) -> CaseEvaluationResult:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        start = perf_counter()
        row = CaseEvaluationResult(case_id=case.id, query=case.query, category=case.category, answerable=case.answerable)
        try:
            row.rag_result = RAGResult.model_validate(self.rag_client.run(case.query, top_k))
            row.rag_retry_count = row.rag_result.retry_count
        except Exception as exc:
            row.errors.append(_case_error("retrieval" if isinstance(exc, RAGRetrievalError) else "rag", exc))
            if isinstance(exc, ProviderError):
                row.rag_retry_count = exc.details.trace.retry_count
        if row.rag_result is not None:
            if case.answerable:
                try:
                    row.retrieval_metrics = evaluate_retrieval(case, row.rag_result.retrieved_docs, top_k)
                except Exception as exc:
                    row.errors.append(_case_error("retrieval", exc))
            try:
                context = json.dumps([
                    {"doc_id": doc.doc_id, "chunk_id": doc.chunk_id, "content": doc.content}
                    for doc in row.rag_result.retrieved_docs
                ], ensure_ascii=False)
                value = JudgeInput(case_id=case.id, query=case.query, expected_answer=case.expected_answer,
                                   retrieved_context=context, rag_answer=row.rag_result.rag_answer, answerable=case.answerable)
                row.judge_result = self._judge(value, row)
                row.judge_retry_count = 0 if row.judge_cache_hit else row.judge_result.metadata.retry_count
                if not case.answerable:
                    row.refusal_correct = row.judge_result.refusal_detected and not row.judge_result.hallucination
            except Exception as exc:
                row.errors.append(_case_error("judge", exc))
                if isinstance(exc, JudgeError):
                    row.judge_retry_count = exc.details.metadata.retry_count
        if row.errors:
            row.status, row.error_type = row.errors[0].status, row.errors[0].error_type
        row.latency_ms = (perf_counter() - start) * 1000
        row.retry_count = row.rag_retry_count + row.judge_retry_count
        return row

    # 做什么：逐条评测已校验的试卷，汇总后保存报告与明细。
    # 为什么需要：形成完整实验结果，并防止运行期间数据版本发生变化。
    def evaluate_dataset(self, dataset_path: str | Path, top_k: int = 3) -> EvaluationReport:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        start = perf_counter()
        dataset_path = Path(dataset_path)
        dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        corpus_hash = hashlib.sha256(self.corpus_path.read_bytes()).hexdigest()
        corpus = load_corpus(self.corpus_path)
        cases = load_dataset(dataset_path, corpus)
        results = []
        for index, case in enumerate(cases, 1):
            row = self.evaluate_case(case, top_k)
            results.append(row)
            if self.verbose:
                print(f"[{index}/{len(cases)}] {case.id}: {row.status} ({row.latency_ms:.0f} ms)", flush=True)
        if (hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_hash
                or hashlib.sha256(self.corpus_path.read_bytes()).hexdigest() != corpus_hash):
            raise RuntimeError("Corpus or Dataset changed during evaluation")
        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc), dataset_path=str(dataset_path),
            dataset_version=f"{dataset_path.stem}@sha256:{dataset_hash}",
            dataset_sha256=dataset_hash, corpus_sha256=corpus_hash, top_k=top_k,
            embedding_model=self.embedding_model, rag_model=self.rag_model,
            judge_model=self.judge_client.settings.model, judge_prompt_version=self.judge_client.prompt_version,
            rag_mode=self.rag_mode, judge_mode=self.judge_mode,
            quality_metrics_are_synthetic=self.rag_mode == "MOCK" or self.judge_mode == "MOCK",
            run_latency_ms=(perf_counter() - start) * 1000, summary=aggregate_results(results), case_results=results,
            execution_config=self.execution_config,
        )
        atomic_write_json(self.output_dir / "case_results.json", [row.model_dump(mode="json") for row in results])
        atomic_write_json(self.output_dir / "latest_report.json", report.model_dump(mode="json"))
        return report


# 做什么：从环境配置创建真实检索器、RAG 和选定裁判后启动 Runner。
# 为什么需要：给外部调用提供一个简单的完整评测入口。
def evaluate_dataset(dataset_path: str | Path, top_k: int = 3, *, output: str | Path = "artifacts/evaluation",
                     judge_mode: str = "LIVE") -> EvaluationReport:
    settings = load_settings()
    settings.require_llm()
    if judge_mode == "MOCK":
        from .mock_judge import MockJudge
        judge = MockJudge()
    elif judge_mode == "LIVE":
        judge = LLMJudgeClient(load_judge_settings(), prompt_version=RUNNER_PROMPT_VERSION)
    else:
        raise ValueError("judge_mode must be LIVE or MOCK")
    # Validate before expensive embedding or external generation work.
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    corpus_path = settings.data_dir / "corpus/ecommerce_v1.json"
    corpus = load_corpus(corpus_path)
    load_dataset(dataset_path, corpus)
    retriever = ChromaRetriever(settings)
    retriever.index(corpus)
    runner = EvaluationRunner(
        DemoRAGClient(retriever, OpenAICompatibleProvider.from_rag_settings(settings)), judge, corpus_path=corpus_path,
        embedding_model=settings.embedding_model, rag_model=settings.rag_model,
        output_dir=output, cache=JudgeCache(".cache/judge_evaluation"), judge_mode=judge_mode, verbose=True,
        execution_config={"generation_timeout_seconds": settings.generation_timeout,
                          "rag_max_attempts": settings.rag_max_attempts,
                          "rag_retry_delay_seconds": settings.rag_retry_delay_seconds,
                          "rag_thinking": settings.rag_thinking,
                          "judge_timeout_seconds": judge.settings.timeout_seconds if judge_mode == "LIVE" else None,
                          "judge_max_attempts": judge.settings.max_attempts if judge_mode == "LIVE" else None},
    )
    return runner.evaluate_dataset(dataset_path, top_k)


# 做什么：解析数据集、K、输出目录和裁判模式参数，打印运行结果。
# 为什么需要：支持命令行自动化，并在存在失败题时返回非零退出码。
def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run full RAG evaluation, preserving all case outcomes")
    parser.add_argument("--dataset", type=Path, default=settings.data_dir / "datasets/ecommerce_eval_v1.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"), help="Output directory")
    parser.add_argument("--judge-mode", choices=["LIVE", "MOCK"], default=os.environ.get("JUDGE_MODE", "LIVE"))
    args = parser.parse_args()
    report = evaluate_dataset(args.dataset, args.top_k, output=args.output, judge_mode=args.judge_mode)
    print(f"RAG_MODE={report.rag_mode}\nJUDGE_MODE={report.judge_mode}")
    if report.quality_metrics_are_synthetic:
        print("Judge/Safety values are MOCK placeholders/probes, not measured model quality.")
    print(report.summary.model_dump_json(indent=2))
    print(f"Saved report to {args.output}")
    if report.summary.engineering.failed_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
