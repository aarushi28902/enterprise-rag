from __future__ import annotations
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, SearchRequest

from src.rag.retrieval.embedder import EmbedderService

RRF_K = 60


@dataclass
class SearchResult:
    chunk_id: str
    parent_id: str
    text: str
    score: float


class HybridSearcher:
    def __init__(self, qdrant_client: AsyncQdrantClient, embedder: EmbedderService):
        self.qdrant = qdrant_client
        self.embedder = embedder

    async def search(self, query: str, collection: str, top_k: int = 6) -> list[SearchResult]:
        dense_vec, sparse_vec = await self.embedder.embed_query(query)

        # dense search
        dense_results = await self.qdrant.search(
            collection_name=collection,
            query_vector=NamedVector(name="dense", vector=dense_vec),
            limit=top_k * 3,
            with_payload=True,
        )

        # sparse search
        sparse_results = await self.qdrant.search(
            collection_name=collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            ),
            limit=top_k * 3,
            with_payload=True,
        )

        # manual RRF fusion
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, point in enumerate(dense_results, start=1):
            pid = str(point.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (RRF_K + rank)
            payloads[pid] = point.payload or {}

        for rank, point in enumerate(sparse_results, start=1):
            pid = str(point.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (RRF_K + rank)
            payloads[pid] = point.payload or {}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        hits = []
        for pid, score in ranked:
            payload = payloads[pid]
            hits.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    parent_id=payload.get("parent_id", ""),
                    text=payload.get("text", ""),
                    score=score,
                )
            )
        return hits
