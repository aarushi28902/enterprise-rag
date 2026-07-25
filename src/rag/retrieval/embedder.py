from __future__ import annotations
import asyncio
from dataclasses import dataclass


@dataclass
class SparseEmbedding:
    indices: list[int]
    values: list[float]


class EmbedderService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(self.model_name, use_fp16=False)

    async def embed_query(self, text: str) -> tuple[list[float], SparseEmbedding]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._embed_sync, [text])
        dense = result["dense_vecs"][0].tolist()
        sparse_weights = result["lexical_weights"][0]
        indices = [int(k) for k in sparse_weights.keys()]
        values = [float(v) for v in sparse_weights.values()]
        return dense, SparseEmbedding(indices=indices, values=values)

    def _embed_sync(self, texts: list[str]) -> dict:
        assert self._model is not None, "Model not loaded. Call load() first."
        return self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
