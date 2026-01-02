from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

from celery_app import celery_app  # type: ignore
from pipeline.workflow.utils.progress import emit_progress  # type: ignore

from pipeline.workflow.sections import SectionReader  # type: ignore


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
        return self._run_ocr(payload, dpi, lang)

    def run_paragraphs(self, payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
        """Alternate OCR entrypoint; returns page sections and ensures progress is emitted."""
        return self._run_ocr(payload, dpi, lang)


@celery_app.task(name="pipeline.ocr.pages")
def ocr_pdf_task(payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
    """Run OCR over a PDF and return serialized page sections.

    Emits progress updates (0 -> 40%) while the OCR runs.
    """
    return OCRTaskService().run_pdf(payload, dpi=dpi, lang=lang)


@celery_app.task(name="pipeline.ocr.paragraphs")
def ocr_par_task(payload: Dict[str, Any], dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
    """Alternate OCR entrypoint; returns page sections and ensures progress is emitted."""
    return OCRTaskService().run_paragraphs(payload, dpi=dpi, lang=lang)
