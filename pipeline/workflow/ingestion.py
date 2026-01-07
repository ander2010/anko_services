from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Callable, Generator, Tuple, Optional

from pdf2image import convert_from_bytes, convert_from_path, pdfinfo_from_path
from PIL import Image

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 150


class PdfIngestion:
    """Validate and stream PDFs from local path or remote storage."""

    max_file_size_mb = MAX_FILE_SIZE_MB

    @classmethod
    def validate_pdf(cls, pdf_path: Path) -> Path:
        """Basic sanity checks before OCR; supports local files and Supabase paths (key strings)."""
        if pdf_path.exists():
            if not pdf_path.is_file():
                raise ValueError(f"Expected a file: {pdf_path}")

            mime, _ = mimetypes.guess_type(pdf_path)
            if mime not in {"application/pdf", None}:
                raise ValueError(f"Unsupported MIME type '{mime}' for {pdf_path}")

            size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if size_mb > cls.max_file_size_mb:
                raise ValueError(f"{pdf_path.name} is {size_mb:.1f} MB, which exceeds the {cls.max_file_size_mb} MB limit")

            logger.info("Validated PDF %s (%.1f MB)", pdf_path, size_mb)
            return pdf_path

        # Remote path (e.g., Supabase object key); validate via metadata.
        key = pdf_path.as_posix()
        try:
            from pipeline.db.supabase_storage import get_object_metadata  # defer import to avoid boto3 cost when unused
        except Exception as exc:
            raise FileNotFoundError(f"PDF not found: {pdf_path}") from exc

        metadata = get_object_metadata(key)
        mime = metadata.get("ContentType")
        content_length = metadata.get("ContentLength")
        size_mb = (content_length or 0) / (1024 * 1024)

        if mime not in {"application/pdf", "binary/octet-stream", None}:
            raise ValueError(f"Unsupported MIME type '{mime}' for {pdf_path}")
        if size_mb and size_mb > cls.max_file_size_mb:
            raise ValueError(f"{pdf_path} is {size_mb:.1f} MB, which exceeds the {cls.max_file_size_mb} MB limit")

        logger.info("Validated remote PDF %s (%.1f MB)", pdf_path, size_mb or 0)
        return pdf_path

    @classmethod
    @classmethod
    def count_pages(cls, pdf_path: Path) -> int:
        """Return total number of pages for a PDF without rendering them."""
        local_path = cls.validate_pdf(pdf_path)
        if local_path.exists():
            info = pdfinfo_from_path(local_path.as_posix(), userpw=None, poppler_path=None)
        else:
            from pipeline.db.supabase_storage import download_object_bytes
            from pdf2image import pdfinfo_from_bytes

            pdf_bytes = download_object_bytes(pdf_path.as_posix(), max_bytes=cls.max_file_size_mb * 1024 * 1024)
            info = pdfinfo_from_bytes(pdf_bytes, userpw=None, poppler_path=None)
        try:
            return int(info.get("Pages", 0) or 0)
        except Exception:
            return 0

    @classmethod
    def stream_pdf_pages(
        cls,
        pdf_path: Path,
        dpi: int = 300,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """Stream pages as PIL images without loading entire PDFs into memory."""
        local_path = cls.validate_pdf(pdf_path)
        page_start = start_page or 1
        total_pages = None
        page_end = end_page
        if local_path.exists():
            info = pdfinfo_from_path(local_path.as_posix(), userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0) or 0)
            if page_end is None:
                page_end = total_pages
            for page in range(page_start, (page_end or 0) + 1):
                images = convert_from_path(local_path.as_posix(), dpi=dpi, first_page=page, last_page=page)
                if images:
                    yield page, images[0]
            return

        # Remote PDF (Supabase): fetch bytes in-memory and stream pages page-by-page.
        from pipeline.db.supabase_storage import download_object_bytes
        from pdf2image import pdfinfo_from_bytes

        pdf_bytes = download_object_bytes(pdf_path.as_posix(), max_bytes=cls.max_file_size_mb * 1024 * 1024)
        info = pdfinfo_from_bytes(pdf_bytes, userpw=None, poppler_path=None)
        total_pages = int(info.get("Pages", 0) or 0)
        if page_end is None:
            page_end = total_pages
        for page in range(page_start, (page_end or 0) + 1):
            images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page, last_page=page)
            if images:
                yield page, images[0]

    @classmethod
    def stream_pdf_pages_with_progress(
        cls,
        pdf_path: Path,
        dpi: int = 300,
        on_progress: Optional[Callable[[int, int], None]] = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """Stream pages with an optional progress callback that receives (page_idx, total_pages) without holding all pages."""
        local_path = cls.validate_pdf(pdf_path)
        page_start = start_page or 1
        total_pages = None
        page_end = end_page
        pdf_bytes: bytes | None = None

        if local_path.exists():
            info = pdfinfo_from_path(local_path.as_posix(), userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0) or 0)
            if page_end is None:
                page_end = total_pages
            for page in range(page_start, (page_end or 0) + 1):
                images = convert_from_path(local_path.as_posix(), dpi=dpi, first_page=page, last_page=page)
                if not images:
                    continue
                if on_progress:
                    on_progress(page, total_pages)
                    logger.info("Streaming page %s/%s at %s dpi", page, total_pages, dpi)
                yield page, images[0]
        else:
            from pipeline.db.supabase_storage import download_object_bytes
            from pdf2image import pdfinfo_from_bytes

            pdf_bytes = download_object_bytes(pdf_path.as_posix(), max_bytes=cls.max_file_size_mb * 1024 * 1024)
            info = pdfinfo_from_bytes(pdf_bytes, userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0) or 0)
            if page_end is None:
                page_end = total_pages
            for page in range(page_start, (page_end or 0) + 1):
                images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page, last_page=page)
                if not images:
                    continue
                if on_progress:
                    on_progress(page, total_pages)
                    logger.info("Streaming page %s/%s at %s dpi", page, total_pages, dpi)
                yield page, images[0]


# Backward-compatible function aliases
def validate_pdf(pdf_path: Path) -> Path:
    return PdfIngestion.validate_pdf(pdf_path)


def stream_pdf_pages(pdf_path: Path, dpi: int = 300) -> Generator[Tuple[int, Image.Image], None, None]:
    yield from PdfIngestion.stream_pdf_pages(pdf_path, dpi=dpi)


def stream_pdf_pages_with_progress(pdf_path: Path, dpi: int = 300, on_progress: Optional[Callable[[int, int], None]] = None) -> Generator[Tuple[int, Image.Image], None, None]:
    yield from PdfIngestion.stream_pdf_pages_with_progress(pdf_path, dpi=dpi, on_progress=on_progress)
