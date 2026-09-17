# 文件作用：抽取不同主题的问题，打印真实检索结果与标准相关块。
# 为什么有它：快速确认本地 Embedding 和 Chroma 连通，不调用生成模型。
"""Manual retrieval-only smoke test; no generation or evaluation metrics."""

import json

from config.settings import load_settings
from src.dataset import load_corpus, load_dataset

from .retriever import ChromaRetriever


# 做什么：按主题选择常规可回答题并打印标准块与 Top-3 结果。
# 为什么需要：快速检查真实检索是否找对资料，不计算生成评分。
def main() -> None:
    settings = load_settings()
    corpus = load_corpus(settings.data_dir / "corpus" / "ecommerce_v1.json")
    cases = load_dataset(settings.data_dir / "datasets" / "ecommerce_eval_v1.json", corpus)
    retriever = ChromaRetriever(settings)
    print(f"Indexed {retriever.index(corpus)} chunks", flush=True)
    topics = {doc.chunk_id: doc.metadata["topic"] for doc in corpus}
    seen = set()
    # The first answerable normal case of each topic, selected before retrieval.
    for case in cases:
        if not case.answerable or case.category != "normal":
            continue
        topic = topics[case.relevant_doc_ids[0]]
        if topic in seen:
            continue
        seen.add(topic)
        hits = retriever.retrieve(case.query, top_k=3)
        print(json.dumps({
            "case_id": case.id, "topic": topic, "query": case.query,
            "expected_relevant_chunk_ids": case.relevant_doc_ids,
            "top_3_retrieved_chunk_ids": [hit.chunk_id for hit in hits],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
