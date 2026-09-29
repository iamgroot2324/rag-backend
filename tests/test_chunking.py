"""Tests for the two chunking strategies and their parameter validation."""

import pytest

from app.schemas.ingestion import ChunkStrategy
from app.services.chunking import chunk_text, fixed_size_chunks, sentence_chunks


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = "a" * 250
    chunks = fixed_size_chunks(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)  # no chunk exceeds the limit
    assert len(chunks) == 4  # windows start at 0, 80, 160, 240
    assert "".join(chunks) != text  # overlap means some content is repeated


def test_fixed_chunks_empty_text() -> None:
    assert fixed_size_chunks("", 100, 10) == []


def test_sentence_chunks_never_split_sentences() -> None:
    text = "First one is here. Second one follows. Third one ends. Fourth closes it."
    chunks = sentence_chunks(text, chunk_size=45, overlap=0)
    assert len(chunks) > 1  # text is long enough to need several chunks
    for chunk in chunks:
        assert chunk.endswith(".")  # every chunk ends on a sentence boundary


def test_sentence_chunks_carry_overlap() -> None:
    text = "Aaaa aaaa. Bbbb bbbb. Cccc cccc. Dddd dddd."
    chunks = sentence_chunks(text, chunk_size=25, overlap=12)
    assert len(chunks) >= 2
    # The last sentence of a chunk should reappear at the start of the next one.
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_sentence_chunks_long_sentence_kept_whole() -> None:
    # A single sentence longer than chunk_size is kept intact rather than cut.
    long_sentence = "word " * 50 + "end."
    chunks = sentence_chunks(long_sentence, chunk_size=50, overlap=0)
    assert chunks == [long_sentence.strip()]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_chunk_text_rejects_bad_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("hello world", ChunkStrategy.FIXED, size, overlap)


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_chunk_text_dispatches_each_strategy(strategy: ChunkStrategy) -> None:
    # Every strategy in the enum must be wired up and return non-empty chunks.
    chunks = chunk_text("One. Two. Three. Four. Five.", strategy, 12, 2)
    assert chunks and all(isinstance(c, str) and c for c in chunks)
