import pytest
from src.rag.ingestion.chunker import ParentChildChunker, Chunk, deterministic_uuid


@pytest.fixture
def chunker():
    return ParentChildChunker(parent_size=200, child_size=50, overlap=10)


def test_chunk_returns_pairs(chunker):
    text = "word " * 300
    pairs = list(chunker.chunk(text, metadata={"source": "test.pdf"}))
    assert len(pairs) > 0


def test_each_pair_has_parent_and_children(chunker):
    text = "word " * 300
    pairs = list(chunker.chunk(text, metadata={"source": "test.pdf"}))
    for parent, children in pairs:
        assert isinstance(parent, Chunk)
        assert len(children) > 0


def test_child_parent_id_links_to_parent(chunker):
    text = "word " * 300
    pairs = list(chunker.chunk(text, metadata={"source": "test.pdf"}))
    for parent, children in pairs:
        for child in children:
            assert child.parent_id == parent.chunk_id


def test_metadata_propagated_to_children(chunker):
    text = "word " * 300
    pairs = list(chunker.chunk(text, metadata={"source": "doc.pdf", "page": 1}))
    for _, children in pairs:
        for child in children:
            assert child.metadata["source"] == "doc.pdf"
            assert child.metadata["page"] == 1


def test_child_ids_are_deterministic(chunker):
    text = "The quick brown fox jumps over the lazy dog. " * 20
    pairs1 = list(chunker.chunk(text, metadata={}))
    pairs2 = list(chunker.chunk(text, metadata={}))
    ids1 = {c.chunk_id for _, children in pairs1 for c in children}
    ids2 = {c.chunk_id for _, children in pairs2 for c in children}
    assert ids1 == ids2


def test_parent_points_to_itself(chunker):
    text = "word " * 300
    pairs = list(chunker.chunk(text, metadata={}))
    for parent, _ in pairs:
        assert parent.chunk_id == parent.parent_id


def test_deterministic_uuid_same_text():
    uuid1 = deterministic_uuid("hello world")
    uuid2 = deterministic_uuid("hello world")
    assert uuid1 == uuid2


def test_deterministic_uuid_different_text():
    uuid1 = deterministic_uuid("hello world")
    uuid2 = deterministic_uuid("goodbye world")
    assert uuid1 != uuid2
