"""PDF ingestion helpers for SFIM documentation."""

from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency fallback
    fitz = None

from app.utils.logger import get_logger


logger = get_logger(__name__)


class PDFParser:
    """Extract text from PDF files and split it into RAG-friendly chunks."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_file(self, pdf_path: str | Path) -> List[str]:
        path = Path(pdf_path)
        if fitz is None:
            logger.warning("PyMuPDF is unavailable; skipping PDF extraction for %s", path.name)
            return []
        document = fitz.open(str(path))
        pages = [page.get_text("text") for page in document]
        full_text = "\n".join(page.strip() for page in pages if page.strip())
        return self.chunk_text(full_text)

    def chunk_text(self, text: str) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        text_length = len(cleaned)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(0, end - self.overlap)

        return chunks
