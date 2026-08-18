"""Debug which chunks are retrieved for each question."""
import asyncio
from qdrant_client import AsyncQdrantClient
from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.hybrid_search import HybridSearcher


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    embedder = EmbedderService()
    await embedder.load()
    searcher = HybridSearcher(qdrant_client=client, embedder=embedder)

    questions = [
        "What are the names of the three characters in the love triangle?",
        "Which Evermore song features Bon Iver?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        results = await searcher.search(q, collection="enterprise_rag", top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  Chunk {i} (score: {r.score:.4f}):")
            print(f"  {r.text[:200]}")

asyncio.run(main())
