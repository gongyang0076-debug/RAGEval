"""Export an existing report without calling RAG/Judge."""

import argparse
from .data import load_report
from .export import export_report


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
