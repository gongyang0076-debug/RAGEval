"""Four hand-authored live Judge examples; explicitly skip missing configuration."""

import argparse
import json
from pathlib import Path

from config.judge import load_judge_settings
from config.settings import ConfigurationError

from .client import JudgeError, LLMJudgeClient
from .models import JudgeInput
from .prompts import PROMPT_VERSION


def smoke_samples() -> list[JudgeInput]:
    return [
        JudgeInput(case_id="smoke_supported", query="订单多久未支付会关闭？",
                   expected_answer="30分钟未支付自动关闭。", retrieved_context="订单提交后30分钟未支付自动关闭。",
                   rag_answer="订单提交后30分钟未支付会自动关闭。", answerable=True),
        JudgeInput(case_id="smoke_context_conflict", query="订单多久未支付会关闭？",
                   expected_answer="30分钟未支付自动关闭。", retrieved_context="订单提交后60分钟未支付自动关闭。",
                   rag_answer="订单提交后60分钟未支付会自动关闭。", answerable=True),
        JudgeInput(case_id="smoke_hallucination", query="订单多久未支付会关闭？",
                   expected_answer="30分钟未支付自动关闭。", retrieved_context="订单提交后30分钟未支付自动关闭。",
                   rag_answer="30分钟未支付会自动关闭，商城还会自动赔付100元。", answerable=True),
        JudgeInput(case_id="smoke_unanswerable_fabrication", query="会员售价是多少？",
                   expected_answer="知识库未提供会员售价，无法确定。", retrieved_context="会员有效期30天。本知识库未提供会员售价。",
                   rag_answer="会员售价是每期9.9元。", answerable=False),
    ]


def run_live_smoke() -> dict:
    samples = smoke_samples()
    report = {"prompt_version": PROMPT_VERSION, "samples": [sample.model_dump() for sample in samples], "results": []}
    try:
        settings = load_judge_settings()
        client = LLMJudgeClient(settings)
    except ConfigurationError as exc:
        return {**report, "status": "NOT RUN", "reason": str(exc)}
    report["judge_model"] = settings.model
    for sample in samples:
        try:
            result = client.judge(sample)
            report["results"].append({"case_id": sample.case_id, "result": result.model_dump()})
        except JudgeError as exc:
            report["results"].append({"case_id": sample.case_id, "error": exc.details.model_dump()})
    report["status"] = "FAIL" if any("error" in row for row in report["results"]) else "PASS"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Judge smoke on four synthetic examples")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint5/live_judge_smoke.json"))
    args = parser.parse_args()
    report = run_live_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LIVE JUDGE: {report['status']}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
