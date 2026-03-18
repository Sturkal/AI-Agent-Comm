"""High-level ingestion pipeline for raw SAP SFIM PDFs and XML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from app.data_pipeline.indexer import ChromaIndexer
from app.data_pipeline.pdf_parser import PDFParser
from app.data_pipeline.xml_parser import SAPXMLParser
from app.utils.logger import get_logger


logger = get_logger(__name__)


class RawDataIngestor:
    """Read `data/raw`, parse files, and persist vectors plus JSON artifacts."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        indexer: ChromaIndexer | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.indexer = indexer or ChromaIndexer()
        self.xml_parser = SAPXMLParser()
        self.pdf_parser = PDFParser()

    def ingest_all(self) -> Dict[str, int]:
        xml_count, pdf_count = 0, 0
        for path in sorted(self.raw_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".xml":
                xml_count += self.ingest_xml_file(path)
            elif suffix == ".pdf":
                pdf_count += self.ingest_pdf_file(path)
        return {"xml_records": xml_count, "pdf_chunks": pdf_count}

    def ingest_xml_file(self, path: str | Path) -> int:
        file_path = Path(path)
        parsed = self.xml_parser.parse_file(file_path)
        if not parsed:
            logger.info("No XML records extracted from %s", file_path.name)
            return 0

        self._write_processed_json(file_path.stem, parsed)
        self.indexer.index_xml_rules(parsed)
        logger.info("Indexed %s XML records from %s", len(parsed), file_path.name)
        return len(parsed)

    def ingest_pdf_file(self, path: str | Path) -> int:
        file_path = Path(path)
        chunks = self.pdf_parser.parse_file(file_path)
        if not chunks:
            logger.info("No text chunks extracted from %s", file_path.name)
            return 0

        self.indexer.index_pdf_chunks(chunks, source_name=file_path.name)
        self._write_processed_json(file_path.stem, [{"chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)])
        logger.info("Indexed %s PDF chunks from %s", len(chunks), file_path.name)
        return len(chunks)

    def _write_processed_json(self, stem: str, payload: List[dict]) -> None:
        output = self.processed_dir / f"{stem}.json"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_raw_data(raw_dir: str | Path = "data/raw") -> Dict[str, int]:
    """Convenience wrapper for the full raw-data ingest flow."""

    return RawDataIngestor(raw_dir=raw_dir).ingest_all()

