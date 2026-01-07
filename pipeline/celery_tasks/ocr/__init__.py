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


def _update_units(job_id: str, count: int, total_pages: int) -> float:
    """Update OCR counters and return overall percent across stages."""
    if not job_id:
        return 0.0
    try:
        OCR_WEIGHT = 0.5
        CHUNK_STAGES = 3

        r = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        units_key = f"job:{job_id}:units"
        try:
            existing_tp = int(r.hget(units_key, "total_pages") or 0)
        except Exception:
            existing_tp = 0
        r.hset(units_key, mapping={"total_pages": max(existing_tp, int(total_pages))})
        if count > 0:
            r.hincrby(units_key, "done_ocr", count)

        data = r.hgetall(units_key)
        total_chunks = int(data.get("total_chunks", 0) or 0)
        total_pages_val = int(data.get("total_pages", 0) or 0)
        done_ocr = int(data.get("done_ocr", 0) or 0)
        done_embed = int(data.get("done_embed", 0) or 0)
        done_persist = int(data.get("done_persist", 0) or 0)
        done_tag = int(data.get("done_tag", 0) or 0)
        frozen_work = data.get("total_work")
        frozen_flag = data.get("total_work_frozen")
        if frozen_work is not None and frozen_flag:
            try:
                total_work = float(frozen_work)
            except Exception:
                total_work = None
        else:
            chunk_weight = total_chunks if total_chunks > 0 else total_pages_val
            effective_chunks = max(1, chunk_weight)
            total_work = total_pages_val * OCR_WEIGHT + effective_chunks * CHUNK_STAGES
            if total_chunks > 0:
                r.hset(units_key, mapping={"total_work": total_work, "total_work_frozen": 1})

        total_work = max(1.0, float(total_work or 1.0))
        done_work = (done_ocr * OCR_WEIGHT) + done_embed + done_persist + done_tag
        raw = min(1.0, max(0.0, done_work / total_work))
        try:
            base = float(r.hget(f"job:{job_id}:progress", "progress") or 0.0)
        except Exception:
            base = 0.0
        scaled = base + (100.0 - base) * raw
        progress = round(min(100.0, max(base, scaled)), 2)
        r.hset(f"job:{job_id}:progress", mapping={"progress": progress})
        return progress
    except Exception:
        return 0.0

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
            overall = _update_units(job_id, 1, total_pages)
            emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, status="OCR", current_step="ocr", extra={"page": page_idx, "total_pages": total_pages})
            logger.info("OCR progress | job=%s doc=%s page=%s/%s overall=%s%%", job_id, doc_id, page_idx, total_pages, overall)

        sections = SectionReader.read(path, input_type="pdf", dpi=dpi, lang=lang, on_progress=on_progress)

        normalized = deserialize_sections(sections)
        serialized = [serialize_section(s) for s in normalized]

        overall = _update_units(job_id, len(serialized), total_pages_seen or len(serialized))
        emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, status="OCR", current_step="ocr", extra={"pages": total_pages_seen or len(serialized)})
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

    sections = SectionReader.read(path, input_type="pdf", dpi=dpi_val, lang=lang, on_progress=None, start_page=start_page, end_page=end_page)
    serialized = [serialize_section(s) for s in sections]
    page_confidence = {s["page"]: s.get("confidence", 0.0) for s in serialized}

    pages_in_batch = len(serialized)
    overall = _update_units(job_id, pages_in_batch, total_pages)

    emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, status="OCR", current_step="ocr", extra={"pages": pages_in_batch, "batch": batch_index, "total_batches": total_batches})
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
