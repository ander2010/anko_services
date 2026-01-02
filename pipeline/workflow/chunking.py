from __future__ import annotations

from typing import Iterable, List

from pipeline.utils.types import ChunkCandidate, Paragraph


class Chunker:
    """Utility class for paragraph chunking and quality enforcement."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.split()

    @staticmethod
    def _should_extend(tokens: List[str]) -> bool:
        if not tokens:
            return False
        return tokens[-1][-1] not in ".?!"

    def adaptive_chunk_paragraphs(self, paragraphs: Iterable[Paragraph], max_tokens: int = 220, overlap: int = 40) -> List[ChunkCandidate]:
        all_tokens: List[str] = []
        page_markers: List[int] = []
        for paragraph in paragraphs:
            tokens = self._tokenize(paragraph.text)
            all_tokens.extend(tokens)
            page_markers.extend([paragraph.page] * len(tokens))

        chunks: List[ChunkCandidate] = []
        start = 0
        total_tokens = len(all_tokens)

        while start < total_tokens:
            end = min(total_tokens, start + max_tokens)
            chunk_tokens = all_tokens[start:end]
            if self._should_extend(chunk_tokens) and end < total_tokens:
                extension = min(total_tokens, end + 20)
                chunk_tokens = all_tokens[start:extension]
                end = extension

            chunk_text = " ".join(chunk_tokens)
            chunk_page = page_markers[start] if page_markers else 1
            chunks.append(ChunkCandidate(page=chunk_page, text=chunk_text, tokens=len(chunk_tokens)))

            if end >= total_tokens:
                break

            start = max(0, end - overlap)

        return chunks

    @staticmethod
    def split_long_chunk(chunk: ChunkCandidate, max_tokens: int) -> List[ChunkCandidate]:
        tokens = chunk.text.split()
        result: List[ChunkCandidate] = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + max_tokens)
            subset = tokens[start:end]
            subset_text = " ".join(subset)
            result.append(ChunkCandidate(page=chunk.page, text=subset_text, tokens=len(subset)))
            start = end
        return result

    @staticmethod
    def _chunk_noisy(text: str) -> bool:
        if not text:
            return True
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / len(text) < 0.35

    def enforce_chunk_quality(self, chunks: List[ChunkCandidate], min_tokens: int = 40, max_tokens: int = 260) -> List[ChunkCandidate]:
        filtered: List[ChunkCandidate] = []
        carry: ChunkCandidate | None = None

        for chunk in chunks:
            if self._chunk_noisy(chunk.text):
                continue

            current = chunk

            if current.tokens > max_tokens:
                filtered.extend(self.split_long_chunk(current, max_tokens))
                continue

            if current.tokens < min_tokens:
                if carry:
                    merged_text = f"{carry.text} {current.text}".strip()
                    merged_tokens = len(merged_text.split())
                    carry = ChunkCandidate(page=carry.page, text=merged_text, tokens=merged_tokens)
                else:
                    carry = current
                continue

            if carry:
                merged_text = f"{carry.text} {current.text}".strip()
                merged_tokens = len(merged_text.split())
                filtered.append(ChunkCandidate(page=carry.page, text=merged_text, tokens=merged_tokens))
                carry = None
            else:
                filtered.append(current)

        if carry and carry.tokens >= min_tokens:
            filtered.append(carry)

        return filtered
