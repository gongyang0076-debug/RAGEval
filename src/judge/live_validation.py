# 文件作用：扩展为十条人工样例，保存真实裁判结果或逐条错误。
# 为什么有它：在全量评测前暴露格式、接口和评分问题，保留原始证据。
"""Sprint 7 live-only validation on ten synthetic examples, separate from Dataset."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from config.judge import load_judge_settings
from config.settings import ConfigurationError
from src.evaluation.cache import atomic_write_json
from .client import JudgeError, LLMJudgeClient
from .models import JudgeInput
from .prompts import RUNNER_PROMPT_VERSION
from .smoke import smoke_samples


# 做什么：准备十条覆盖正确、错误、编造和拒答的人工样例。
# 为什么需要：先用可核对的小样本检查真实裁判。
def validation_samples():
    samples = smoke_samples()
    samples.extend([
        JudgeInput(case_id="supported_refund", query="退款退到哪里？", expected_answer="原路退回。",
                   retrieved_context="退款只支持原支付渠道退回。", rag_answer="退款将原路退回。", answerable=True),
        JudgeInput(case_id="supported_tracking", query="在哪里查看物流？", expected_answer="订单详情页。",
                   retrieved_context="订单详情页展示物流信息。", rag_answer="请在订单详情页查看。", answerable=True),
        JudgeInput(case_id="wrong_refund", query="退款退到哪里？", expected_answer="原路退回。",
                   retrieved_context="退款只支持原支付渠道退回。", rag_answer="退到任意指定银行卡。", answerable=True),
        JudgeInput(case_id="invented_compensation", query="包裹延误会赔钱吗？", expected_answer="没有赔偿承诺，无法确定。",
                   retrieved_context="包裹延误时请联系人工客服核查。", rag_answer="每延误一天赔100元。", answerable=False),
        JudgeInput(case_id="proper_refusal", query="会员多少钱？", expected_answer="未提供价格，无法确定。",
                   retrieved_context="会员有效期30天。", rag_answer="知识库未提供会员售价，无法确定。", answerable=False),
        JudgeInput(case_id="refusal_with_fabrication", query="会员多少钱？", expected_answer="未提供价格，无法确定。",
                   retrieved_context="会员有效期30天。", rag_answer="无法确定，但官方一般是9.9元。", answerable=False),
    ])
    return samples


# 做什么：加载真实配置并逐条判卷，保存成功结果或错误。
# 为什么需要：不因单条失败丢失其余证据，缺配置时明确说明。
def run_validation():
    samples = validation_samples()
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "prompt_version": RUNNER_PROMPT_VERSION,
              "samples": [s.model_dump() for s in samples], "results": [], "judge_model": None}
    try:
        settings = load_judge_settings()
        settings.validate()
    except ConfigurationError as exc:
        return {**report, "status": "LIVE JUDGE NOT AVAILABLE", "reason": str(exc)}
    client = LLMJudgeClient(settings, prompt_version=RUNNER_PROMPT_VERSION)
    report["judge_model"] = settings.model
    for sample in samples:
        try:
            result = client.judge(sample)
            report["results"].append({"case_id": sample.case_id, "result": result.model_dump(mode="json")})
        except JudgeError as exc:
            report["results"].append({"case_id": sample.case_id, "error": exc.details.model_dump(mode="json")})
    report["status"] = "FAIL" if any("error" in row for row in report["results"]) else "PASS"
    return report


# 做什么：接收保存路径并执行十条真实裁判验证。
# 为什么需要：提供独立验证命令，失败时向调用者返回失败状态。
def main():
    parser = argparse.ArgumentParser(description="Ten live Judge validation examples; never substitutes a mock")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint7/live_judge_validation.json"))
    args = parser.parse_args()
    report = run_validation()
    atomic_write_json(args.output, report)
    print(report["status"])
    print(f"Saved {len(report['results'])} live results to {args.output}")
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
