# 文件作用：测试模型按需加载、向量归一化和查询指令处理。
# 为什么有它：防止模型重复加载或把查询指令错误加到文档上。
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from config.settings import Settings
from src.rag.embeddings import SentenceTransformerEmbedder


# 做什么：验证模型仅按需加载、向量归一化且指令只加在查询上。
# 为什么需要：避免重复加载及文档与问题编码规则混乱。
def test_embedding_model_is_lazy_normalized_and_query_instruction_is_query_only(monkeypatch):
    factory = Mock()
    factory.return_value.encode.side_effect = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    settings = Settings(data_dir=Path("data"))
    embedder = SentenceTransformerEmbedder(settings)
    factory.assert_not_called()
    assert embedder.embed_documents(["商品政策"]) == [[1.0, 0.0]]
    assert embedder.embed_query("如何退货") == [0.0, 1.0]
    factory.assert_called_once_with(
        "BAAI/bge-small-zh-v1.5", device="cpu", cache_folder=str(settings.embedding_cache_dir),
        trust_remote_code=False,
    )
    calls = factory.return_value.encode.call_args_list
    assert calls[0].args[0] == ["商品政策"]
    assert calls[1].args[0] == [settings.embedding_query_instruction + "如何退货"]
    assert all(call.kwargs["normalize_embeddings"] for call in calls)
