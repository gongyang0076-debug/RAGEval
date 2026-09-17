# 文件作用：标记“评测调度与汇总”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Case evaluation and reporting; CLI: python -m src.evaluation.runner."""

from .models import CaseEvaluationResult, EvaluationReport

__all__ = ["CaseEvaluationResult", "EvaluationReport"]
