"""Sliding-window text chunker that splits by token count."""

from __future__ import annotations


def _simple_tokenize(text: str) -> list[str]:
    """Whitespace-based tokenizer. Good enough for chunking; actual token counts
    are approximated (1 word ~ 1.3 tokens). For exact counts we'd need a
    model-specific tokenizer, but this keeps the processing pipeline
    independent of the embedding model."""
    return text.split()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split text into overlapping chunks of approximately `chunk_size` tokens.

    Returns list of dicts with keys: text, start_token, end_token, chunk_index.
    """
    if not text:
        return []

    words = _simple_tokenize(text)
    if not words:
        return []

    if len(words) <= chunk_size:
        return [
            {
                "text": text.strip(),
                "start_token": 0,
                "end_token": len(words),
                "chunk_index": 0,
            }
        ]

    chunks = []
    start = 0
    chunk_idx = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text_str,
                "start_token": start,
                "end_token": end,
                "chunk_index": chunk_idx,
            }
        )
        chunk_idx += 1

        if end >= len(words):
            break
        start += step

    return chunks
