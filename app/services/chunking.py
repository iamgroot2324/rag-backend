"""Chunking strategies used to split documents before embedding."""

import re
from collections.abc import Callable

from app.schemas.ingestion import ChunkStrategy

# Split after ., ! or ? followed by whitespace (a simple, dependency-free sentence splitter).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding character window; simple and predictable.

    Each window starts `chunk_size - overlap` characters after the previous one,
    so neighbouring chunks share `overlap` characters of context.
    """
    step = chunk_size - overlap
    chunks = (text[i : i + chunk_size].strip() for i in range(0, len(text), step))
    return [c for c in chunks if c]  # drop windows that were only whitespace


def sentence_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole sentences up to chunk_size, carrying trailing sentences as overlap.

    Unlike the fixed strategy this never cuts a sentence in half, which usually
    gives more coherent chunks for retrieval.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []  # sentences in the chunk being built
    length = 0  # running character count of `current` (+1 per joining space)

    for sentence in sentences:
        # Adding this sentence would overflow: close the current chunk first.
        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))

            # Start the next chunk with trailing sentences totalling <= overlap characters.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, carried_len

        current.append(sentence)
        length += len(sentence) + 1

    if current:  # flush the final chunk
        chunks.append(" ".join(current))
    return chunks


# Strategy registry: adding a new strategy only requires a new entry here.
_STRATEGIES: dict[ChunkStrategy, Callable[[str, int, int], list[str]]] = {
    ChunkStrategy.FIXED: fixed_size_chunks,
    ChunkStrategy.SENTENCE: sentence_chunks,
}


def chunk_text(text: str, strategy: ChunkStrategy, chunk_size: int, overlap: int) -> list[str]:
    """Validate parameters and dispatch to the selected chunking strategy."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
    return _STRATEGIES[strategy](text, chunk_size, overlap)
