"""Token-aware recursive text chunking."""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken


class ChunkingError(Exception):
    """Raised when chunking parameters are invalid or text cannot be tokenized."""


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int
    page_number: int
    chunk_index: int


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveTokenChunker:
    """Splits text into chunks bounded by token count, preferring natural boundaries."""

    def __init__(self, chunk_size_tokens: int, overlap_tokens: int, encoding_name: str = "cl100k_base") -> None:
        if chunk_size_tokens <= 0:
            raise ChunkingError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
        if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
            raise ChunkingError(
                f"overlap_tokens ({overlap_tokens}) must be in [0, chunk_size_tokens={chunk_size_tokens})"
            )

        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)

    def chunk_page(self, text: str, page_number: int, starting_chunk_index: int) -> list[TextChunk]:
        if not text.strip():
            return []

        pieces = self._split_recursively(text, _SEPARATORS)
        chunks = self._merge_pieces_to_chunks(pieces)

        return [
            TextChunk(
                text=chunk_text,
                token_count=len(self._encoder.encode(chunk_text)),
                page_number=page_number,
                chunk_index=starting_chunk_index + offset,
            )
            for offset, chunk_text in enumerate(chunks)
        ]

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        if len(self._encoder.encode(text)) <= self._chunk_size:
            return [text]

        if not separators:
            return self._hard_split_by_tokens(text)

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            return self._hard_split_by_tokens(text)

        parts = text.split(separator)
        result: list[str] = []
        for part in parts:
            if len(self._encoder.encode(part)) <= self._chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursively(part, rest))
        return result

    def _hard_split_by_tokens(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        return [
            self._encoder.decode(tokens[i : i + self._chunk_size])
            for i in range(0, len(tokens), self._chunk_size)
        ]

    def _merge_pieces_to_chunks(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current_tokens: list[int] = []

        for piece in pieces:
            piece_tokens = self._encoder.encode(piece)
            if not piece_tokens:
                continue

            if len(current_tokens) + len(piece_tokens) <= self._chunk_size:
                if current_tokens:
                    current_tokens.extend(self._encoder.encode(" "))
                current_tokens.extend(piece_tokens)
            else:
                if current_tokens:
                    chunks.append(self._encoder.decode(current_tokens))
                    overlap_start = max(0, len(current_tokens) - self._overlap)
                    current_tokens = current_tokens[overlap_start:] + piece_tokens
                else:
                    current_tokens = piece_tokens

        if current_tokens:
            chunks.append(self._encoder.decode(current_tokens))

        return chunks
