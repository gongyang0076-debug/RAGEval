"""Manual retrieval-only smoke test; no generation or evaluation metrics."""

import json

from config.settings import load_settings
from src.dataset import load_corpus, load_dataset

from .retriever import ChromaRetriever


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
