"""Local SentenceTransformer embeddings with a query-only retrieval instruction."""

from config.settings import Settings


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings):
        self.model_name = settings.embedding_model
        self.query_instruction = settings.embedding_query_instruction
        self.settings = settings
        self._model = None

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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([self.query_instruction + query])[0]
