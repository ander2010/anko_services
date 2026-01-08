from __future__ import annotations

import mimetypes
import os
import atexit
import hashlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from pathlib import Path
from typing import Callable, Generator, Tuple, Optional

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 150
MAX_REMOTE_CACHE = int(os.getenv("MAX_REMOTE_CACHE", "6"))
SHARED_CACHE_DIR = Path(os.getenv("SUPABASE_CACHE_DIR", "/tmp/supabase_pdfs"))


class PdfIngestion:
    """Validate and stream PDFs from local path or remote storage."""

    max_file_size_mb = MAX_FILE_SIZE_MB
    _validated_cache: dict[str, Path] = {}
    _remote_file_cache: "OrderedDict[str, Path]" = OrderedDict()
    _remote_prefetch_futures: dict[str, Future] = {}
    _remote_executor: ThreadPoolExecutor | None = None
    _lock = Lock()
    _remote_refcounts: dict[str, int] = {}
    _max_remote_cache = MAX_REMOTE_CACHE

    @classmethod
    def _shared_cache_path(cls, key: str) -> Path:
        SHARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pdf"
        return SHARED_CACHE_DIR / name

    @classmethod
    def _get_remote_pdf_file(cls, pdf_path: Path) -> Path:
        """Download remote PDF once per process into a temp file via streaming to avoid memory bloat."""
        key = pdf_path.as_posix()
        with cls._lock:
            cls._remote_refcounts[key] = cls._remote_refcounts.get(key, 0) + 1
            shared_path = cls._shared_cache_path(key)
            if shared_path.exists():
                cls._remote_file_cache[key] = shared_path
                cls._remote_file_cache.move_to_end(key)
                return shared_path
            cached = cls._remote_file_cache.get(key)
            if cached and cached.exists():
                cls._remote_file_cache.move_to_end(key)
                return cached

            from pipeline.db.supabase_storage import download_object_to_tempfile

            fut = cls._remote_prefetch_futures.get(key)
            path: Path
            if fut:
                try:
                    path = fut.result()
                except Exception:
                    # Fallback to direct download if prefetch failed.
                    logger.debug("Prefetch failed for %s, retrying download inline", key, exc_info=True)
                    path = download_object_to_tempfile(
                        pdf_path.as_posix(), max_bytes=cls.max_file_size_mb * 1024 * 1024
                    )
                finally:
                    cls._remote_prefetch_futures.pop(key, None)
            else:
                logger.info("Downloading remote PDF %s to temp file", key)
                path = download_object_to_tempfile(
                    pdf_path.as_posix(), max_bytes=cls.max_file_size_mb * 1024 * 1024
                )

            # Move into shared cache for reuse across worker processes.
            try:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                path.replace(shared_path)
                path = shared_path
            except Exception:
                logger.debug("Could not move %s to shared cache %s", path, shared_path, exc_info=True)

            cls._remote_file_cache[key] = path
            cls._remote_file_cache.move_to_end(key)
            cls._remote_refcounts.setdefault(key, 0)
            while len(cls._remote_file_cache) > cls._max_remote_cache:
                old_key, old_path = cls._remote_file_cache.popitem(last=False)
                if cls._remote_refcounts.get(old_key, 0) > 0:
                    # Put back if still in use.
                    cls._remote_file_cache[old_key] = old_path
                    break
                try:
                    old_path.unlink(missing_ok=True)
                except Exception:
                    pass
                cls._remote_refcounts.pop(old_key, None)
                cls._remote_prefetch_futures.pop(old_key, None)
            return cls._remote_file_cache[key]

    @classmethod
    def prefetch_remote_pdf_file(cls, pdf_path: Path) -> None:
        """Kick off an async download for remote PDFs to overlap I/O with upstream work."""
        if pdf_path.exists():
            return
        key = pdf_path.as_posix()
        with cls._lock:
            if key in cls._remote_file_cache and cls._remote_file_cache[key].exists():
                return
            if key in cls._remote_prefetch_futures:
                return
            from pipeline.db.supabase_storage import download_object_to_tempfile

            if cls._remote_executor is None:
                cls._remote_executor = ThreadPoolExecutor(max_workers=2)

            def _task() -> Path:
                return download_object_to_tempfile(
                    key, max_bytes=cls.max_file_size_mb * 1024 * 1024
                )

            cls._remote_prefetch_futures[key] = cls._remote_executor.submit(_task)

    @classmethod
    def cleanup_cached_file(cls, pdf_path: Path | None = None) -> None:
        """Remove cached temp files to avoid leaking disk space."""
        if pdf_path is None:
            items = list(cls._remote_file_cache.items())
        else:
            key = pdf_path.as_posix()
            items = [(key, cls._remote_file_cache.get(key))]

        with cls._lock:
            for key, cached in items:
                if not cached:
                    continue
                cls._remote_refcounts[key] = max(0, cls._remote_refcounts.get(key, 1) - 1)
                try:
                    cached.unlink(missing_ok=True)
                finally:
                    cls._remote_file_cache.pop(key, None)
                    fut = cls._remote_prefetch_futures.pop(key, None)
                    if fut and not fut.done():
                        fut.cancel()
                    if cls._remote_refcounts.get(key, 0) <= 0:
                        cls._remote_refcounts.pop(key, None)

    @classmethod
    def cleanup_all(cls) -> None:
        """Best-effort cleanup of all cached remote files."""
        cls.cleanup_cached_file()
        try:
            if cls._remote_executor:
                cls._remote_executor.shutdown(wait=False, cancel_futures=True)
                cls._remote_executor = None
        except Exception:
            pass

    @classmethod
    def validate_pdf(cls, pdf_path: Path) -> Path:
        """Basic sanity checks before OCR; supports local files and Supabase paths (key strings)."""
        key = pdf_path.as_posix()
        cached = cls._validated_cache.get(key)
        if cached:
            return cached

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
            cls._validated_cache[key] = pdf_path
            return cls._validated_cache[key]

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
        cls._validated_cache[key] = pdf_path
        return cls._validated_cache[key]

    @classmethod
    def count_pages(cls, pdf_path: Path) -> int:
        """Return total number of pages for a PDF without rendering them."""
        local_path = cls.validate_pdf(pdf_path)
        if local_path.exists():
            info = pdfinfo_from_path(local_path.as_posix(), userpw=None, poppler_path=None)
        else:
            remote_file = cls._get_remote_pdf_file(pdf_path)
            info = pdfinfo_from_path(remote_file.as_posix(), userpw=None, poppler_path=None)
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

        # Remote PDF (Supabase): stream from a cached temp file page-by-page (full download required by pdf2image).
        remote_file = cls._get_remote_pdf_file(pdf_path)
        info = pdfinfo_from_path(remote_file.as_posix(), userpw=None, poppler_path=None)
        total_pages = int(info.get("Pages", 0) or 0)
        if page_end is None:
            page_end = total_pages
        for page in range(page_start, (page_end or 0) + 1):
            images = convert_from_path(remote_file.as_posix(), dpi=dpi, first_page=page, last_page=page)
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
            remote_file = cls._get_remote_pdf_file(pdf_path)
            info = pdfinfo_from_path(remote_file.as_posix(), userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0) or 0)
            if page_end is None:
                page_end = total_pages
            for page in range(page_start, (page_end or 0) + 1):
                images = convert_from_path(remote_file.as_posix(), dpi=dpi, first_page=page, last_page=page)
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


# Ensure cached temp files are cleaned up when the process exits.
atexit.register(PdfIngestion.cleanup_all)
