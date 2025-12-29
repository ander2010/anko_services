from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from pipeline.types import OCRPageResult, Paragraph


class TextNormalizer:
    """Provides text normalization and paragraph segmentation utilities."""

    HEADER_PATTERN = re.compile(r"^(page|\d+)\b", re.IGNORECASE)
    FOOTER_PATTERN = re.compile(r"(page\s*\d+|\d+/\d+)$", re.IGNORECASE)

    def normalize_unicode(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\ufeff", "").replace("\u00ad", "")
        return normalized

    def remove_headers_footers(self, text: str, page_number: int) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        filtered: List[str] = []
        for idx, line in enumerate(lines):
            if idx == 0 and self.HEADER_PATTERN.search(line):
                continue
            if idx == len(lines) - 1 and self.FOOTER_PATTERN.search(line):
                continue
            filtered.append(line)
        return "\n".join(filtered)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        no_extra_space = re.sub(r"\s+", " ", text)
        return no_extra_space.strip()

    def lowercase_and_cleanup(self, text: str) -> str:
        text = self.normalize_unicode(text)
        return self.normalize_whitespace(text.lower())

    def segment_into_paragraphs(self, pages: Iterable[OCRPageResult], min_chars: int = 40) -> List[Paragraph]:
        paragraphs: List[Paragraph] = []
        for page in pages:
            normalized_page = self.normalize_unicode(page.cleaned_text)
            cleaned = self.remove_headers_footers(normalized_page, page.page)
            blocks = re.split(r"(?:\n\s*)(?:\n\s*)", cleaned)
            for block in blocks:
                normalized = self.normalize_whitespace(block.lower())
                if len(normalized) < min_chars:
                    continue
                paragraphs.append(Paragraph(page=page.page, text=normalized))
        return paragraphs
