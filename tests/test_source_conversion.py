from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict

from pipeline.workflow.source_conversion import ensure_pdf_source


def _create_minimal_docx(path: Path, text: str) -> None:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)


def test_ensure_pdf_source_converts_txt_and_archives_original(tmp_path: Path) -> None:
    source = tmp_path / "lesson.txt"
    source.write_text("line 1\nline 2\nline 3", encoding="utf-8")

    pdf_path, original_path = ensure_pdf_source(source)

    assert pdf_path == tmp_path / "lesson.pdf"
    assert pdf_path.exists()
    assert original_path is None
    assert source.exists()


def test_ensure_pdf_source_converts_docx_and_archives_original(tmp_path: Path) -> None:
    source = tmp_path / "module.docx"
    _create_minimal_docx(source, "Hello DOCX")

    pdf_path, original_path = ensure_pdf_source(source)

    assert pdf_path == tmp_path / "module.pdf"
    assert pdf_path.exists()
    assert original_path is None
    assert source.exists()


def test_ensure_pdf_source_keeps_pdf_untouched(tmp_path: Path) -> None:
    source = tmp_path / "ready.pdf"
    source.write_bytes(b"%PDF-1.4\n%fake\n")

    pdf_path, original_path = ensure_pdf_source(source)

    assert pdf_path == source
    assert original_path is None
    assert source.exists()


def test_ensure_pdf_source_converts_remote_txt_and_uploads_both(monkeypatch, tmp_path: Path) -> None:
    remote_store: Dict[str, bytes] = {"uploads/chapter.txt": b"hello from remote text"}
    uploaded: Dict[str, bytes] = {}

    def _exists(key: str) -> bool:
        return key in remote_store

    def _download(key: str, destination: Path) -> Path:
        destination.write_bytes(remote_store[key])
        return destination

    def _upload(source: Path, key: str) -> str:
        uploaded[key] = source.read_bytes()
        return key

    monkeypatch.setattr("pipeline.db.supabase_storage.object_exists", _exists)
    monkeypatch.setattr("pipeline.db.supabase_storage.download_object", _download)
    monkeypatch.setattr("pipeline.db.supabase_storage.upload_object", _upload)

    pdf_path, original_path = ensure_pdf_source(Path("uploads/chapter.txt"))

    assert str(pdf_path) == "uploads/chapter.pdf"
    assert original_path is None
    assert "uploads/chapter.pdf" in uploaded
    assert uploaded["uploads/chapter.pdf"].startswith(b"%PDF")
