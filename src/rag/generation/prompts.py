from src.rag.retrieval.hybrid_search import SearchResult

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question
using ONLY the provided context. If the context does not contain the answer, say so."""


def build_rag_prompt(query: str, chunks: list[SearchResult]) -> list[dict]:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(f"[{i}] {chunk.text}")

    context = "\n\n---\n\n".join(context_blocks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
