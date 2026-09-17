# 文件作用：标记“答案质量裁判”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Independent LLM-as-Judge; no dataset runner or score aggregation."""

from .client import JudgeAPIError, JudgeError, JudgeOutputError, LLMJudgeClient
from .models import JudgeInput, JudgeResult, JudgeVerdict, JudgeVerdictV2
from .prompts import PROMPT_VERSION, RUNNER_PROMPT_VERSION

__all__ = [
    "JudgeInput", "JudgeResult", "JudgeVerdict", "LLMJudgeClient",
    "JudgeError", "JudgeAPIError", "JudgeOutputError", "PROMPT_VERSION", "RUNNER_PROMPT_VERSION", "JudgeVerdictV2",
]
