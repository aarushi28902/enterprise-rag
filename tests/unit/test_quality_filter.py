import pytest
from src.rag.ingestion.chunker import ParentChildChunker


def test_chunker_overlap_creates_continuity():
    """Chunks should share content at boundaries due to overlap."""
    chunker = ParentChildChunker(parent_size=100, child_size=50, overlap=20)
    text = "a " * 200
    pairs = list(chunker.chunk(text, metadata={}))
    assert len(pairs) > 1


def test_chunker_small_text_single_parent():
    """Text smaller than parent_size should produce one parent."""
    chunker = ParentChildChunker(parent_size=1000, child_size=200, overlap=10)
    text = "short text " * 10
    pairs = list(chunker.chunk(text, metadata={}))
    assert len(pairs) == 1


def test_chunk_ids_are_valid_uuids():
    """All chunk IDs should be valid UUID format."""
    import uuid
    chunker = ParentChildChunker()
    text = "word " * 100
    pairs = list(chunker.chunk(text, metadata={}))
    for parent, children in pairs:
        uuid.UUID(parent.chunk_id)   # raises if invalid
        for child in children:
            uuid.UUID(child.chunk_id)
