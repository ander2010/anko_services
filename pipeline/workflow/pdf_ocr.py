from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Callable

import pytesseract
from PIL.Image import Image
from pytesseract import Output

from pipeline.workflow.ingestion import PdfIngestion
from pipeline.utils.types import OCRPageResult

logger = logging.getLogger(__name__)

_requested_workers = int(os.getenv("OCR_MAX_WORKERS", "4"))
_cpu_count = os.cpu_count() or 1
OCR_MAX_WORKERS = max(1, min(_requested_workers, _cpu_count))
logger.info("OCR worker pool size set to %s (requested=%s, cpu_count=%s)", OCR_MAX_WORKERS, _requested_workers, _cpu_count)

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


def _ocr_images_iter(images: Sequence[tuple[int, Image]], lang: str, on_progress=None, max_workers: int = OCR_MAX_WORKERS) -> Iterable[OCRPageResult]:
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


class Ocr:
    """OCR helper for PDFs using pytesseract."""

    def __init__(self, lang: str = "eng", dpi: int = 300, max_workers: int = OCR_MAX_WORKERS):
        self.lang = lang
        self.dpi = dpi
        self.max_workers = max_workers

    def extract_sections_from_pdf(self, pdf_path: Path) -> List[OCRPageResult]:
        """Return OCR content per page so downstream tooling can build sections."""
        return list(self.iter_sections_from_pdf_with_progress(pdf_path, on_progress=None))

    def extract_sections_from_pdf_with_progress(self, pdf_path: Path, on_progress: Optional[Callable[[int, int], None]] = None, start_page: int | None = None, end_page: int | None = None) -> List[OCRPageResult]:
        """Return OCR content per page and invoke progress callbacks during OCR, optionally over a page slice."""
        pages = PdfIngestion.stream_pdf_pages_with_progress(
            pdf_path,
            dpi=self.dpi,
            on_progress=on_progress,
            start_page=start_page,
            end_page=end_page,
        )
        return _ocr_images(pages, self.lang)

    def iter_sections_from_pdf_with_progress(self, pdf_path: Path, on_progress=None, start_page: int | None = None, end_page: int | None = None) -> Iterable[OCRPageResult]:
        """Yield OCRPageResult per page with progress callback support without pre-loading all pages."""
        total_est = None
        if start_page is not None and end_page is not None and end_page >= start_page:
            total_est = end_page - start_page + 1
        else:
            # Fall back to counting pages up front so progress is stable.
            total_est = PdfIngestion.count_pages(pdf_path)

        pages = PdfIngestion.stream_pdf_pages_with_progress(
            pdf_path, dpi=self.dpi, on_progress=on_progress, start_page=start_page, end_page=end_page
        )

        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            for index, image in pages:
                futures.add(executor.submit(_run_ocr_single, index, image, self.lang))
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    section = fut.result()
                    processed += 1
                    if on_progress:
                        try:
                            total = total_est or (processed + len(futures))
                            on_progress(section.page, total)
                        except TypeError:
                            on_progress(section.page)
                    yield section

            for fut in as_completed(futures):
                section = fut.result()
                processed += 1
                if on_progress:
                    try:
                        total = total_est or processed
                        on_progress(section.page, total)
                    except TypeError:
                        on_progress(section.page)
                yield section

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """High-level helper to read a PDF and return extracted text."""
        sections = self.extract_sections_from_pdf(pdf_path)
        return sections_to_text(sections)


# Backward-compatible function aliases
def extract_sections_from_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> List[OCRPageResult]:
    return Ocr(lang=lang, dpi=dpi).extract_sections_from_pdf(pdf_path)


def extract_sections_from_pdf_with_progress(
    pdf_path: Path,
    dpi: int = 300,
    lang: str = "eng",
    on_progress=None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> List[OCRPageResult]:
    return Ocr(lang=lang, dpi=dpi).extract_sections_from_pdf_with_progress(
        pdf_path,
        on_progress=on_progress,
        start_page=start_page,
        end_page=end_page,
    )


def iter_sections_from_pdf_with_progress(
    pdf_path: Path,
    dpi: int = 300,
    lang: str = "eng",
    on_progress=None,
    max_workers: int = OCR_MAX_WORKERS,
    start_page: int | None = None,
    end_page: int | None = None,
) -> Iterable[OCRPageResult]:
    return Ocr(lang=lang, dpi=dpi, max_workers=max_workers).iter_sections_from_pdf_with_progress(pdf_path, on_progress=on_progress, start_page=start_page, end_page=end_page)


def extract_text_from_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> str:
    return Ocr(lang=lang, dpi=dpi).extract_text_from_pdf(pdf_path)


def sections_to_text(sections: List[OCRPageResult]) -> str:
    """Format multiple OCR sections into the legacy plain-text form."""
    if not sections:
        return ""
    joined = [f"Page {section.page} (conf {section.confidence}) ---\n{section.cleaned_text}".strip() for section in sections if section.cleaned_text]
    if not joined:
        return ""
    return ("\n\n".join(joined)).strip() + "\n"
