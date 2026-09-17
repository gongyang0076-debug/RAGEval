# 文件作用：启动评测结果网页，把页面工作交给 report.dashboard。
# 为什么有它：提供一个容易找到的 Streamlit 启动入口，打开页面不会重新运行评测。
"""RAGEval read-only Streamlit entry point. Launch: streamlit run app.py."""

from src.report.dashboard import main


if __name__ == "__main__":
    main()
