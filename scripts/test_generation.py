import asyncio
from qdrant_client import AsyncQdrantClient

from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.hybrid_search import HybridSearcher
from src.rag.generation.generator import StreamingGenerator


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    embedder = EmbedderService()

    print("Loading embedder...")
    await embedder.load()

    searcher = HybridSearcher(qdrant_client=client, embedder=embedder)
    generator = StreamingGenerator()

    query = "What is RAG?"
    print(f"\nQuery: {query}")

    print("Retrieving chunks...")
    chunks = await searcher.search(query, collection="enterprise_rag", top_k=3)
    print(f"Found {len(chunks)} chunks")

    print("\nGenerating answer:\n")
    async for token in generator.stream(query, chunks):
        print(token, end="", flush=True)
    print()


asyncio.run(main())
