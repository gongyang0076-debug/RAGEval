# 文件作用：标记“报告展示与导出”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Read-only evaluation report loading and JSON/Excel export."""
from .data import ReportError, load_report, parse_report
from .export import excel_bytes, export_report, json_bytes

__all__ = ["ReportError", "load_report", "parse_report", "excel_bytes", "export_report", "json_bytes"]
