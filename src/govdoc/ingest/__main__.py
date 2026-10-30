"""CLI entry point for PDF ingestion into ChromaDB."""

import argparse
import sys
from pathlib import Path
from typing import List

from govdoc.config import load_config, resolve_path
from govdoc.logging_utils import append_log
from govdoc.ingest.embed_index import BGEM3Embedder, ingest_pdf
from govdoc.ingest.chunker import DocumentChunker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="GovDoc-RAG PDF Ingestion Pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a single PDF file to ingest",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to a directory containing PDF files to ingest",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="govdoc_corpus",
        help="ChromaDB collection name",
    )
    args = parser.parse_args()

    if not args.pdf and not args.dir:
        parser.error("Either --pdf or --dir must be specified.")

    pdf_files: List[Path] = []
    if args.pdf:
        p = resolve_path(args.pdf)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        pdf_files.append(p)

    if args.dir:
        d = resolve_path(args.dir)
        if not d.exists() or not d.is_dir():
            raise NotADirectoryError(f"Directory not found: {d}")
        pdf_files.extend(sorted(d.glob("*.pdf")))

    print("=" * 65)
    print("GovDoc-RAG Document Ingestion")
    print(f"Target Collection: {args.collection}")
    print(f"Found {len(pdf_files)} PDF file(s) to process.")
    print("=" * 65)

    chunker = DocumentChunker()
    embedder = BGEM3Embedder()

    total_chunks = 0
    total_embed_time = 0.0

    for pdf_path in pdf_files:
        stats = ingest_pdf(
            pdf_path=pdf_path,
            collection_name=args.collection,
            chunker=chunker,
            embedder=embedder,
        )
        total_chunks += stats.get("chunks", 0)
        total_embed_time += stats.get("embed_time_s", 0.0)

    time_per_1000 = (total_embed_time / max(total_chunks, 1)) * 1000.0

    print("\n" + "=" * 65)
    print("Ingestion Summary:")
    print(f"  Total PDFs Processed:  {len(pdf_files)}")
    print(f"  Total Chunks Indexed:  {total_chunks}")
    print(f"  Total Embedding Time:  {total_embed_time:.2f}s")
    print(f"  Time per 1000 Chunks:  {time_per_1000:.2f}s")
    print("=" * 65)

    append_log(
        f"Ingested {len(pdf_files)} PDFs ({total_chunks} chunks) into '{args.collection}'. "
        f"Embedding time: {total_embed_time:.2f}s ({time_per_1000:.2f}s/1000 chunks).",
        module="M3:INGEST",
    )


if __name__ == "__main__":
    main()
