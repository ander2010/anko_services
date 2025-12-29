from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Callable, Generator, Tuple, Optional

from pdf2image import convert_from_path
from PIL import Image

from pipeline.logging_config import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 150


def validate_pdf(pdf_path: Path) -> None:
    """Basic sanity checks before OCR."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Expected a file: {pdf_path}")

    mime, _ = mimetypes.guess_type(pdf_path)
    if mime not in {"application/pdf", None}:
        raise ValueError(f"Unsupported MIME type '{mime}' for {pdf_path}")

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"{pdf_path.name} is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit")

    logger.info("Validated PDF %s (%.1f MB)", pdf_path, size_mb)


def stream_pdf_pages(pdf_path: Path, dpi: int = 300) -> Generator[Tuple[int, Image.Image], None, None]:
    """Stream pages as PIL images; pdf2image converts lazily per page so we wrap in a generator."""
    validate_pdf(pdf_path)
    images = convert_from_path(pdf_path.as_posix(), dpi=dpi)
    for index, image in enumerate(images, 1):
        yield index, image


def stream_pdf_pages_with_progress(pdf_path: Path, dpi: int = 300, on_progress: Optional[Callable[[int, int], None]] = None) -> Generator[Tuple[int, Image.Image], None, None]:
    """Stream pages with an optional progress callback that receives (page_idx, total_pages)."""
    validate_pdf(pdf_path)
    images = convert_from_path(pdf_path.as_posix(), dpi=dpi)
    total = len(images)
    for index, image in enumerate(images, 1):
        if on_progress:
            on_progress(index, total)
            logger.info("Streaming page %s/%s at %s dpi", index, total, dpi)
        yield index, image
