import asyncio
from qdrant_client import AsyncQdrantClient

from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.hybrid_search import HybridSearcher


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    embedder = EmbedderService()

    print("Loading embedder...")
    await embedder.load()

    searcher = HybridSearcher(qdrant_client=client, embedder=embedder)

    query = "What is RAG?"
    print(f"Searching: {query}")
    results = await searcher.search(query, collection="enterprise_rag", top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
        print(f"chunk_id: {r.chunk_id}")
        print(f"text: {r.text[:100]}")


asyncio.run(main())
