from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from pipeline.logging_config import get_logger
from pipeline.types import OCRPageResult
from pdf_ocr import extract_sections_from_pdf_with_progress, iter_sections_from_pdf_with_progress

logger = get_logger(__name__)


class SectionReader:
    """Reads source documents into OCR sections."""

    @staticmethod
    def read(source: Path, input_type: str, dpi: int, lang: str, on_progress: Optional[Callable[[int, int], Any]] = None) -> List[OCRPageResult]:
        inferred_type = input_type
        if input_type == "auto":
            inferred_type = "pdf" if source.suffix.lower() == ".pdf" else "text"

        if inferred_type == "pdf":
            # When on_progress is provided, consume iteratively so callers can flush progress.
            if on_progress:
                sections: List[OCRPageResult] = []
                for section in iter_sections_from_pdf_with_progress(source, dpi=dpi, lang=lang, on_progress=on_progress):
                    sections.append(section)
                return sections

            return extract_sections_from_pdf_with_progress(source, dpi=dpi, lang=lang, on_progress=on_progress)

        text = source.read_text(encoding="utf-8")
        return [OCRPageResult(page=1, raw_text=text, cleaned_text=text, confidence=1.0)]
