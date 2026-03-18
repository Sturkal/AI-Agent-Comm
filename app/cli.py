"""Command-line utilities for SAP SFIM ingestion and app operations."""

from __future__ import annotations

import argparse
from typing import Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> None:
        return None

from app.agent.llm_engine import SFIMLLMEngine
from app.agent.refinement_memory import SelfEvaluationMemoryStore
from app.agent.retriever import SFIMRetriever
from app.data_pipeline.ingest import RawDataIngestor
from app.data_pipeline.indexer import ChromaIndexer
from app.utils.logger import configure_root_logger, get_logger


logger = get_logger(__name__)
configure_root_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfim-agent", description="SAP SFIM AI Agent utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest raw XML and PDF files into the knowledge base")
    ingest.add_argument("--raw-dir", default="data/raw", help="Directory with source XML/PDF files")
    ingest.add_argument("--processed-dir", default="data/processed", help="Directory for parsed JSON output")
    ingest.add_argument("--chroma-dir", default="chromadb_store", help="Directory for vector store persistence")

    query = subparsers.add_parser("query", help="Ask a question against the knowledge base")
    query.add_argument("question", help="Question to ask the SFIM agent")
    query.add_argument("--phone-number", default=None, help="Optional user phone number for context")
    query.add_argument("--chroma-dir", default="chromadb_store", help="Directory for vector store persistence")
    query.add_argument(
        "--show-hits",
        action="store_true",
        help="Print retrieved XML/PDF hits and self-evaluation details",
    )

    memory = subparsers.add_parser("memory", help="Inspect or export self-evaluation memory")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)

    memory_show = memory_subparsers.add_parser("show", help="Show stored self-evaluation memory entries")
    memory_show.add_argument("--memory-path", default="data/processed/self_eval_memory.jsonl", help="Path to the memory JSONL file")
    memory_show.add_argument("--limit", type=int, default=20, help="Maximum number of entries to display")
    memory_show.add_argument("--question", default="", help="Optional question to filter relevant entries")
    memory_show.add_argument("--chroma-dir", default="chromadb_store", help="Directory for vector store persistence")

    memory_export = memory_subparsers.add_parser("export", help="Export self-evaluation memory")
    memory_export.add_argument("--memory-path", default="data/processed/self_eval_memory.jsonl", help="Path to the memory JSONL file")
    memory_export.add_argument("--output", required=True, help="Output file path")
    memory_export.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Export format",
    )

    return parser


def run_ingest(args: argparse.Namespace) -> None:
    indexer = ChromaIndexer(persist_dir=args.chroma_dir)
    ingestor = RawDataIngestor(raw_dir=args.raw_dir, processed_dir=args.processed_dir, indexer=indexer)
    result = ingestor.ingest_all()
    logger.info("Ingest complete: %s", result)
    print(result)


def run_query(args: argparse.Namespace) -> None:
    indexer = ChromaIndexer(persist_dir=args.chroma_dir)
    engine = SFIMLLMEngine(SFIMRetriever(indexer))
    result = engine.answer(args.question, user_phone_number=args.phone_number)

    print(result.answer)
    if not args.show_hits:
        return

    print("\n--- XML Hits ---")
    for hit in result.xml_hits:
        print(f"{hit.get('id', '')}: {hit.get('document', '')}")

    print("\n--- PDF Hits ---")
    for hit in result.pdf_hits:
        print(f"{hit.get('id', '')}: {hit.get('document', '')}")

    if result.self_evaluation_summary:
        print("\n--- Self Evaluation ---")
        print(result.self_evaluation_summary)
        if result.self_evaluation:
            print(result.self_evaluation)

    if result.refinement_memory:
        print("\n--- Refinement Memory ---")
        for item in result.refinement_memory:
            print(item)


def run_memory_show(args: argparse.Namespace) -> None:
    store = SelfEvaluationMemoryStore(args.memory_path)
    summary = store.render_summary(limit=args.limit, question=args.question)
    print(summary)


def run_memory_export(args: argparse.Namespace) -> None:
    store = SelfEvaluationMemoryStore(args.memory_path)
    if args.format == "json":
        output_path = store.export_json(args.output)
    else:
        output_path = store.export_jsonl(args.output)
    logger.info("Exported self-evaluation memory to %s", output_path)
    print({"output": str(output_path), "format": args.format})


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "memory":
        if args.memory_command == "show":
            run_memory_show(args)
        elif args.memory_command == "export":
            run_memory_export(args)


if __name__ == "__main__":
    main()
