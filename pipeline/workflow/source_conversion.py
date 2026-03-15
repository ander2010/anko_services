from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".log",
    ".ini",
    ".cfg",
}
SOFFICE_EXTENSIONS = {
    ".doc",
    ".odt",
    ".rtf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _render_text_to_pdf(text: str, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    page_width, page_height = 1654, 2339  # A4-ish at ~150 dpi
    margin = 80
    line_height = 24
    max_chars = 95

    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw:
            lines.append("")
            continue
        while len(raw) > max_chars:
            lines.append(raw[:max_chars])
            raw = raw[max_chars:]
        lines.append(raw)
    if not lines:
        lines = [""]

    pages: list[Image.Image] = []
    current = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(current)
    y = margin
    for line in lines:
        if y + line_height > page_height - margin:
            pages.append(current)
            current = Image.new("RGB", (page_width, page_height), "white")
            draw = ImageDraw.Draw(current)
            y = margin
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height
    pages.append(current)
    pages[0].save(output_pdf, "PDF", save_all=True, append_images=pages[1:])


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path, "r") as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in para.findall(".//w:t", ns)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs).strip()


def _convert_via_soffice(source_path: Path, output_pdf: Path) -> None:
    if shutil.which("soffice") is None:
        raise ValueError(
            f"Cannot convert '{source_path.suffix}' without LibreOffice (soffice). "
            "Install LibreOffice in the worker image."
        )
    outdir = output_pdf.parent
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(outdir),
        str(source_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"soffice conversion failed for {source_path.name}: {result.stderr.strip() or result.stdout.strip()}"
        )
    generated = outdir / f"{source_path.stem}.pdf"
    if not generated.exists():
        raise RuntimeError(f"soffice did not produce expected output: {generated}")
    if generated != output_pdf:
        generated.replace(output_pdf)


def ensure_pdf_source(source_path: Path) -> Tuple[Path, Optional[Path]]:
    """
    Ensure the source is a PDF before pipeline validation.

    Returns (pdf_path, original_source_path_if_converted).
    For converted inputs:
      - original file remains unchanged
      - resulting PDF is saved as <name>.pdf
    """
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return source_path, None

    if not source_path.exists():
        # Non-local path: treat as Supabase object key and write both outputs back to storage.
        from pipeline.db.supabase_storage import download_object, object_exists, upload_object

        source_key = source_path.as_posix()
        if not object_exists(source_key):
            raise ValueError(f"Non-PDF input was not found locally or in Supabase: {source_path}")

        pdf_key = str(Path(source_key).with_suffix(".pdf"))

        with tempfile.TemporaryDirectory(prefix="source-convert-") as tmpdir:
            local_input = Path(tmpdir) / Path(source_key).name
            download_object(source_key, local_input)
            local_pdf = local_input.with_suffix(".pdf")

            try:
                if suffix in TEXT_EXTENSIONS:
                    text = local_input.read_text(encoding="utf-8", errors="replace")
                    _render_text_to_pdf(text, local_pdf)
                elif suffix == ".docx":
                    text = _extract_docx_text(local_input)
                    _render_text_to_pdf(text, local_pdf)
                elif suffix in SOFFICE_EXTENSIONS:
                    _convert_via_soffice(local_input, local_pdf)
                else:
                    raise ValueError(
                        f"Unsupported input type '{suffix}'. "
                        "Supported directly: pdf, txt/md/csv/json/xml/log, docx. "
                        "Other office formats require LibreOffice (soffice)."
                    )
            except Exception:
                raise

            upload_object(local_pdf, pdf_key)

        logger.info(
            "Remote source converted to PDF | source_key=%s pdf_key=%s",
            source_key,
            pdf_key,
        )
        return Path(pdf_key), None

    if not source_path.is_file():
        raise ValueError(f"Expected a file path: {source_path}")

    pdf_path = source_path.with_suffix(".pdf")

    try:
        if suffix in TEXT_EXTENSIONS:
            text = source_path.read_text(encoding="utf-8", errors="replace")
            _render_text_to_pdf(text, pdf_path)
        elif suffix == ".docx":
            text = _extract_docx_text(source_path)
            _render_text_to_pdf(text, pdf_path)
        elif suffix in SOFFICE_EXTENSIONS:
            _convert_via_soffice(source_path, pdf_path)
        else:
            raise ValueError(
                f"Unsupported input type '{suffix}'. "
                "Supported directly: pdf, txt/md/csv/json/xml/log, docx. "
                "Other office formats require LibreOffice (soffice)."
            )
    except Exception:
        raise

    logger.info("Source converted to PDF | source=%s pdf=%s", source_path, pdf_path)
    return pdf_path, None
