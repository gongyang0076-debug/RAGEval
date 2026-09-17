# 文件作用：给初步召回的知识块重新评分，并提供可替换的两阶段检索器。
# 为什么需要：向量召回之后进一步判断问题与资料是否真正相关。
from math import isfinite
from typing import Protocol

from src.dataset import RetrievedDocument


# 这个类：约定检索器需要提供的行为。
# 为什么需要：让 DemoRAGClient 同时接受向量检索和两阶段检索。
class Retriever(Protocol):
    # 做什么：约定统一检索接口。
    # 为什么需要：原始检索器与重排序检索器可以接入同一个 RAG Client。
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]: ...


# 这个类：使用模型判断问题与候选知识块的相关程度。
# 为什么需要：对初召回结果进行更精细的排序。
class CrossEncoderReranker:
    # 做什么：保存模型配置，允许测试注入假模型。
    # 为什么需要：默认延迟加载，导入模块不会下载模型。
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu",
                 cache_dir=".cache/huggingface", model=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = str(cache_dir)
        self._model = model

    # 做什么：首次使用时加载本地 CrossEncoder。
    # 为什么需要：重复请求复用权重，禁止执行模型仓库中的自定义代码。
    def load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from torch import nn
            self._model = CrossEncoder(
                self.model_name, device=self.device, cache_folder=self.cache_dir,
                max_length=512, activation_fn=nn.Identity(), trust_remote_code=False,
            )
        return self._model

    # 做什么：对问题和每个候选正文联合评分，降序返回前 K 条。
    # 为什么需要：保留原始块编号供评测对齐；logit 越高越相关，但不是概率。
    def rerank(self, query: str, documents: list[RetrievedDocument], top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if len({d.chunk_id for d in documents}) != len(documents):
            raise ValueError("duplicate candidate chunk_id")
        if not documents:
            return []
        scores = self.load().predict(
            [(query.strip(), d.content) for d in documents],
            batch_size=8, show_progress_bar=False, convert_to_numpy=True,
        )
        if len(scores) != len(documents):
            raise ValueError("Reranker score count does not match candidates")
        scores = [float(score) for score in scores]
        if not all(isfinite(score) for score in scores):
            raise ValueError("Reranker scores must be finite")
        # 相同分数保持初召回顺序；不原地修改原始结果。
        ordered = sorted(zip(documents, scores), key=lambda item: -item[1])[:top_k]
        return [doc.model_copy(update={"score": score, "rank": rank})
                for rank, (doc, score) in enumerate(ordered, 1)]


# 这个类：把初召回与重排序封装成一次检索。
# 为什么需要：接入现有 RAG 接口而不改变生成步骤。
class RerankingRetriever:
    # 做什么：把原始检索器与重排序器组合起来。
    # 为什么需要：作为 DemoRAGClient 的可选检索器，不改变默认检索行为。
    def __init__(self, retriever: Retriever, reranker: CrossEncoderReranker, candidate_k=10):
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        self.retriever, self.reranker, self.candidate_k = retriever, reranker, candidate_k

    # 做什么：先多召回候选，再重排序得到最终 K 条。
    # 为什么需要：让重排序有足够的候选空间，并保持统一的检索接口。
    def retrieve(self, query: str, top_k=3):
        if type(top_k) is not int or top_k <= 0 or top_k > self.candidate_k:
            raise ValueError("top_k must be positive and at most candidate_k")
        candidates = self.retriever.retrieve(query, self.candidate_k)
        return self.reranker.rerank(query, candidates, top_k)
