from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

import pytesseract
from PIL.Image import Image
from pytesseract import Output

from workflow.ingestion import stream_pdf_pages, stream_pdf_pages_with_progress
from pipeline.types import OCRPageResult

logger = logging.getLogger(__name__)


def _basic_cleanup(text: str) -> str:
    cleaned = text.replace("\x0c", "").strip()
    return cleaned


def _extract_confidence(image: Image, lang: str) -> float:
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    confidences = [float(value) for value in data.get("conf", []) if value not in ("-1", "-2")]
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences), 2)


def _ocr_images(images: Iterable[tuple[int, Image]], lang: str) -> List[OCRPageResult]:
    """Run OCR over iterable of PIL images and return structured sections."""
    sections: List[OCRPageResult] = []
    for index, image in images:
        raw_text = pytesseract.image_to_string(image, lang=lang)
        confidence = _extract_confidence(image, lang)
        cleaned = _basic_cleanup(raw_text)
        sections.append(OCRPageResult(page=index, raw_text=raw_text, cleaned_text=cleaned, confidence=confidence))
    return sections


def _run_ocr_single(index: int, image: Image, lang: str) -> OCRPageResult:
    raw_text = pytesseract.image_to_string(image, lang=lang)
    confidence = _extract_confidence(image, lang)
    cleaned = _basic_cleanup(raw_text)
    return OCRPageResult(page=index, raw_text=raw_text, cleaned_text=cleaned, confidence=confidence)


def _ocr_images_iter(images: Sequence[tuple[int, Image]], lang: str, on_progress=None, max_workers: int = 4) -> Iterable[OCRPageResult]:
    """Iterate OCR results, yielding after each page and invoking progress callback using a thread pool."""
    total = len(images)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_ocr_single, index, image, lang): index for index, image in images}
        for future in as_completed(futures):
            index = futures[future]
            section = future.result()
            if on_progress:
                try:
                    on_progress(index, total)
                except TypeError:
                    on_progress(index)
            yield section


def extract_sections_from_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> List[OCRPageResult]:
    """Return OCR content per page so downstream tooling can build sections."""
    pages = list(stream_pdf_pages(pdf_path, dpi=dpi))
    return _ocr_images(pages, lang)


def extract_sections_from_pdf_with_progress(pdf_path: Path, dpi: int = 300, lang: str = "eng", on_progress=None) -> List[OCRPageResult]:
    """Return OCR content per page and invoke progress callbacks during OCR."""
    pages = stream_pdf_pages_with_progress(pdf_path, dpi=dpi, on_progress=on_progress)
    return _ocr_images(pages, lang)


def iter_sections_from_pdf_with_progress(pdf_path: Path, dpi: int = 300, lang: str = "eng", on_progress=None, max_workers: int = 4) -> Iterable[OCRPageResult]:
    """Yield OCRPageResult per page with progress callback support."""
    pages = list(stream_pdf_pages_with_progress(pdf_path, dpi=dpi, on_progress=on_progress))
    yield from _ocr_images_iter(pages, lang=lang, on_progress=on_progress, max_workers=max_workers)


def extract_text_from_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> str:
    """High-level helper to read a PDF and return extracted text."""
    sections = extract_sections_from_pdf(pdf_path, dpi=dpi, lang=lang)
    return sections_to_text(sections)


def sections_to_text(sections: List[OCRPageResult]) -> str:
    """Format multiple OCR sections into the legacy plain-text form."""
    if not sections:
        return ""
    joined = [f"Page {section.page} (conf {section.confidence}) ---\n{section.cleaned_text}".strip() for section in sections if section.cleaned_text]
    if not joined:
        return ""
    return ("\n\n".join(joined)).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Utilities for running OCR over PDF files with pytesseract.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("-o", "--output", type=Path, help="Optional output text file")
    parser.add_argument("--lang", default="eng", help="Language for Tesseract")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering")
    args = parser.parse_args()

    sections = extract_sections_from_pdf(args.pdf, dpi=args.dpi, lang=args.lang)
    text = sections_to_text(sections)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"pages": len(sections), "output": str(args.output)}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
