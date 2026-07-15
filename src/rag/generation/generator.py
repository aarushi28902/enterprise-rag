from __future__ import annotations
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.rag.generation.prompts import build_rag_prompt
from src.rag.retrieval.hybrid_search import SearchResult


class StreamingGenerator:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def stream(self, query: str, chunks: list[SearchResult]) -> AsyncIterator[str]:
        messages = build_rag_prompt(query=query, chunks=chunks)
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta
            if delta.content:
                yield delta.content
