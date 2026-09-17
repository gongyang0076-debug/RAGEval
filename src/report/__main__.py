# 文件作用：提供已有报告的命令行导出入口。
# 为什么有它：用户可以直接生成 JSON 和 Excel，不需要打开网页或调用模型。
"""Export an existing report without calling RAG/Judge."""

import argparse
from .data import load_report
from .export import export_report


# 做什么：读取报告路径和输出目录参数并调用导出功能。
# 为什么需要：不打开网页也能生成 JSON 与 Excel。
def main():
    parser = argparse.ArgumentParser(description="Export an EvaluationReport to JSON and Excel")
    parser.add_argument("--report", default="artifacts/evaluation/latest_report.json")
    parser.add_argument("--output", default="artifacts/report")
    parser.add_argument("--dataset-dir", default="data/datasets")
    args = parser.parse_args()
    for path in export_report(load_report(args.report), args.output, args.dataset_dir):
        print(path)


if __name__ == "__main__":
    main()
