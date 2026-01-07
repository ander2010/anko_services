from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from redis import Redis

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

from celery_app import celery_app  # type: ignore
from pipeline.workflow.utils.progress import emit_progress, PROGRESS_REDIS_URL  # type: ignore

from pipeline.workflow.sections import SectionReader  # type: ignore

DEFAULT_OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_BATCH_PAGES = max(1, int(os.getenv("OCR_BATCH_PAGES", "10")))

@dataclass
class OCRPageResult:
    page: int
    raw_text: str
    cleaned_text: str
    confidence: float


def serialize_section(section: OCRPageResult) -> Dict[str, Any]:
    return {
        "page": int(section.page),
        "raw_text": str(section.raw_text),
        "cleaned_text": str(section.cleaned_text),
        "confidence": float(section.confidence or 0.0),
    }


def deserialize_sections(sections: Iterable[dict]) -> List[OCRPageResult]:
    """Turn a list of section-like dicts into OCRPageResult objects.

    Accepts missing keys and normalizes them to safe defaults.
    """
    out: List[OCRPageResult] = []
    for item in sections or []:
        # Support both dict-like and object-like sections (including different OCRPageResult classes).
        if hasattr(item, "get"):
            page_val = item.get("page", 0)
            raw_val = item.get("raw_text", "")
            cleaned_val = item.get("cleaned_text", item.get("raw_text", ""))
            conf_val = item.get("confidence", 0.0)
        else:
            page_val = getattr(item, "page", 0)
            raw_val = getattr(item, "raw_text", "")
            cleaned_val = getattr(item, "cleaned_text", getattr(item, "raw_text", ""))
            conf_val = getattr(item, "confidence", 0.0)

        out.append(
            OCRPageResult(
                page=int(page_val),
                raw_text=str(raw_val),
                cleaned_text=str(cleaned_val),
                confidence=float(conf_val or 0.0),
            )
        )
    return out


class OCRTaskService:
    """Encapsulates OCR task flows."""

    @staticmethod
    def _run_ocr(payload: Dict[str, Any], dpi: int, lang: str) -> Dict[str, Any]:
        path = Path(payload.get("file_path") or payload.get("file path", ""))
        job_id = payload.get("job_id")
        doc_id = payload.get("doc_id") or path.stem

        logger.info("OCR start | job=%s doc=%s path=%s dpi=%s lang=%s", job_id, doc_id, path, dpi, lang)

        total_pages_seen = 0

        def on_progress(page_idx: int, total_pages: int) -> None:
            nonlocal total_pages_seen
            total_pages_seen = int(total_pages or total_pages_seen)
            step_pct = round((page_idx / total_pages) * 100, 2) if total_pages else 0.0
            overall = round(min(40.0, (step_pct / 100.0) * 40.0), 2)
            emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, step_progress=step_pct, status="OCR", current_step="ocr", extra={"page": page_idx, "total_pages": total_pages})
            logger.info("OCR progress | job=%s doc=%s page=%s/%s step=%s%% overall=%s%%", job_id, doc_id, page_idx, total_pages, step_pct, overall)

        sections = SectionReader.read(path, input_type="pdf", dpi=dpi, lang=lang, on_progress=on_progress)

        normalized = deserialize_sections(sections)
        serialized = [serialize_section(s) for s in normalized]

        emit_progress(job_id=job_id, doc_id=doc_id, progress=40, step_progress=100, status="OCR", current_step="ocr", extra={"pages": total_pages_seen or len(serialized)})
        logger.info("OCR done   | job=%s doc=%s pages=%s", job_id, doc_id, total_pages_seen or len(serialized))

        page_confidence = {s["page"]: s.get("confidence", 0.0) for s in serialized}

        return {
            "file_path": str(path),
            "job_id": job_id,
            "doc_id": doc_id,
            "sections": serialized,
            "page_confidence": page_confidence,
        }

    def run_pdf(self, payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
        """Run OCR over a PDF and return serialized page sections."""
        return self._run_ocr(payload, dpi or DEFAULT_OCR_DPI, lang)

    def run_paragraphs(self, payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
        """Alternate OCR entrypoint; returns page sections and ensures progress is emitted."""
        return self._run_ocr(payload, dpi or DEFAULT_OCR_DPI, lang)


@celery_app.task(name="pipeline.ocr.pages")
def ocr_pdf_task(payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
    """Run OCR over a PDF and return serialized page sections.

    Emits progress updates (0 -> 40%) while the OCR runs.
    """
    dpi_val = dpi or DEFAULT_OCR_DPI
    return OCRTaskService().run_pdf(payload, dpi=dpi_val, lang=lang)


@celery_app.task(name="pipeline.ocr.paragraphs")
def ocr_par_task(payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
    """Alternate OCR entrypoint; returns page sections and ensures progress is emitted."""
    dpi_val = dpi or DEFAULT_OCR_DPI
    return OCRTaskService().run_paragraphs(payload, dpi=dpi_val, lang=lang)


@celery_app.task(name="pipeline.ocr.batch")
def ocr_batch_task(payload: Dict[str, Any], start_page: int, end_page: int, total_pages: int, batch_index: int = 1, total_batches: int = 1, dpi: int = DEFAULT_OCR_DPI, lang: str = "eng") -> Dict[str, Any]:
    """Perform OCR on a page range and return serialized sections for that batch."""
    path = Path(payload.get("file_path") or payload.get("file path") or "")
    job_id = payload.get("job_id")
    doc_id = payload.get("doc_id") or path.stem
    dpi_val = dpi or DEFAULT_OCR_DPI

    logger.info("OCR batch start | job=%s doc=%s path=%s pages=%s-%s/%s dpi=%s lang=%s", job_id, doc_id, path, start_page, end_page, total_pages, dpi_val, lang)

    band_start, band_end = 0.0, 40.0
    band_width = band_end - band_start

    # Emit a start-of-batch progress to avoid apparent stalling at 5%.
    try:
        start_progress = round(band_start + ((max(0, batch_index - 1) / max(1, total_batches)) * band_width), 2)
        emit_progress(job_id=job_id, doc_id=doc_id, progress=start_progress, step_progress=0, status="OCR", current_step="ocr", extra={"batch": batch_index, "total_batches": total_batches})
    except Exception:
        pass

    def compute_overall(step_pct: float) -> float:
        if total_batches <= 0:
            return min(band_end, band_start + (step_pct / 100.0) * band_width)
        return band_start + (min(total_batches, max(0, batch_index - 1) + (step_pct / 100.0)) / total_batches) * band_width

    sections = SectionReader.read(path, input_type="pdf", dpi=dpi_val, lang=lang, on_progress=None, start_page=start_page, end_page=end_page)
    serialized = [serialize_section(s) for s in sections]
    page_confidence = {s["page"]: s.get("confidence", 0.0) for s in serialized}

    try:
        redis_client = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        batches_done = int(redis_client.hincrby(f"job:{job_id}:batches", "ocr", 1))
    except Exception:
        batches_done = batch_index
    fraction = min(total_batches, batches_done) / max(1, total_batches)
    overall = round(band_start + fraction * band_width, 2)

    try:
        redis_client = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        existing = redis_client.hget(f"job:{job_id}", "progress")
        if existing is not None:
            try:
                overall = max(overall, float(existing))
            except Exception:
                pass
    except Exception:
        pass

    emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, step_progress=100, status="OCR", current_step="ocr", extra={"pages": len(serialized), "batch": batch_index, "total_batches": total_batches, "batches_done": batches_done})
    logger.info("OCR batch done  | job=%s doc=%s pages=%s-%s/%s count=%s batch=%s/%s", job_id, doc_id, start_page, end_page, total_pages, len(serialized), batch_index, total_batches)

    return {
        "file_path": str(path),
        "job_id": job_id,
        "doc_id": doc_id,
        "sections": serialized,
        "page_confidence": page_confidence,
        "batch_index": batch_index,
        "total_batches": total_batches,
    }


@celery_app.task(name="pipeline.ocr.collect")
def ocr_collect_task(batch_results: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect OCR batch results into a single payload for downstream tasks."""
    page_map: Dict[int, Dict[str, Any]] = {}
    page_confidence: Dict[int, float] = {}

    for batch in batch_results or []:
        for item in batch.get("sections") or []:
            try:
                page_num = int(item.get("page", 0))
            except Exception:
                page_num = 0
            if page_num <= 0:
                continue
            page_map[page_num] = item  # last write wins for duplicates
        for page, conf in (batch.get("page_confidence") or {}).items():
            try:
                page_confidence[int(page)] = float(conf)
            except Exception:
                continue

    sections = [page_map[p] for p in sorted(page_map.keys())]

    merged_payload = {
        "file_path": payload.get("file_path"),
        "job_id": payload.get("job_id"),
        "doc_id": payload.get("doc_id"),
        "sections": sections,
        "page_confidence": page_confidence,
    }

    logger.info("OCR collect done | job=%s doc=%s batches=%s pages=%s", merged_payload.get("job_id"), merged_payload.get("doc_id"), len(batch_results or []), len(sections))
    return merged_payload
