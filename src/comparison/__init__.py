# 文件作用：标记“版本对比与回归判断”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Baseline snapshots, direction-aware comparison and configured regression gates."""

from .engine import compare_reports
from .gate import GateConfigurationError, apply_gate, load_gate_config
from .models import BaselineSnapshot, GateConfig, GateResult, MetricComparison, ReportComparison, Threshold
from .snapshot import SnapshotError, load_baseline, save_baseline, to_snapshot

__all__ = ["BaselineSnapshot", "MetricComparison", "ReportComparison", "GateConfig", "GateResult", "Threshold",
           "SnapshotError", "GateConfigurationError", "save_baseline", "load_baseline", "to_snapshot",
           "compare_reports", "apply_gate", "load_gate_config"]
