from insightlens.ingestion.chunker import ChunkingError, RecursiveTokenChunker

import pytest


def test_short_text_yields_single_chunk():
    chunker = RecursiveTokenChunker(chunk_size_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk_page("This is a short paragraph.", page_number=1, starting_chunk_index=0)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_empty_text_yields_no_chunks():
    chunker = RecursiveTokenChunker(chunk_size_tokens=100, overlap_tokens=10)
    assert chunker.chunk_page("   ", page_number=2, starting_chunk_index=5) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ChunkingError):
        RecursiveTokenChunker(chunk_size_tokens=50, overlap_tokens=50)


def test_long_text_produces_multiple_chunks():
    chunker = RecursiveTokenChunker(chunk_size_tokens=20, overlap_tokens=5)
    long_text = " ".join(["sentence number {}".format(i) for i in range(60)])
    chunks = chunker.chunk_page(long_text, page_number=3, starting_chunk_index=10)
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 10
    assert all(chunk.page_number == 3 for chunk in chunks)
