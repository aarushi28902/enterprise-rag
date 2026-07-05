from __future__ import annotations
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def deterministic_uuid(text: str) -> str:
    """Same text always produces the same UUID - enables deduplication."""
    hash_bytes = hashlib.sha256(text.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


class ParentChildChunker:
    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        overlap: int = 32,
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def _split(self, text: str, size: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

    def chunk(
        self,
        text: str,
        metadata: dict,
    ) -> Iterator[tuple[Chunk, list[Chunk]]]:
        parent_texts = self._split(text, self.parent_size)

        for parent_text in parent_texts:
            parent_id = str(uuid.uuid4())

            parent = Chunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                text=parent_text,
                metadata={**metadata, "level": "parent"},
            )

            child_texts = self._split(parent_text, self.child_size)
            children = [
                Chunk(
                    chunk_id=deterministic_uuid(ct),
                    parent_id=parent_id,
                    text=ct,
                    metadata={**metadata, "level": "child"},
                )
                for ct in child_texts
            ]

            yield parent, children
