# 文件作用：把知识文本和查询问题编码成向量，并按需加载本地模型。
# 为什么有它：向量检索需要统一的文本表示，而查看报告时不应该加载模型。
"""Local SentenceTransformer embeddings with a query-only retrieval instruction."""

from config.settings import Settings


# 这个类：负责按统一配置把文本变成向量。
# 为什么需要：检索前需要一致的文本表示。
class SentenceTransformerEmbedder:
    # 做什么：保存模型、查询指令和运行配置，暂不加载权重。
    # 为什么需要：创建对象时不触发昂贵的模型加载。
    def __init__(self, settings: Settings):
        self.model_name = settings.embedding_model
        self.query_instruction = settings.embedding_query_instruction
        self.settings = settings
        self._model = None

    # 做什么：首次需要时加载模型，并批量生成归一化向量。
    # 为什么需要：统一编码方式，后续调用复用已加载的模型。
    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.settings.embedding_device,
                cache_folder=str(self.settings.embedding_cache_dir),
                trust_remote_code=False,
            )
        return self._model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).tolist()

    # 做什么：对知识块原文生成向量。
    # 为什么需要：写入向量库时保持文档侧编码方式一致。
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    # 做什么：给问题添加查询指令后生成单个向量。
    # 为什么需要：让检索查询按当前模型配置表达搜索意图。
    def embed_query(self, query: str) -> list[float]:
        return self._encode([self.query_instruction + query])[0]
