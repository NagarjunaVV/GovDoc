"""Unified Command-Line Interface for GovDoc-RAG."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from govdoc.config import load_config, resolve_path
from govdoc.logging_utils import append_log
from govdoc.ingest.embed_index import ingest_pdf, get_chroma_collection
from govdoc.generate.rag_pipeline import RAGPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def handle_ask(args):
    """Handle 'ask' command: answer query on specified PDF or default corpus."""
    collection_name = args.collection

    # If a specific PDF is provided, ensure it is ingested
    if args.pdf:
        pdf_path = resolve_path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF file not found at {pdf_path}")
            sys.exit(1)

        doc_collection = f"doc_{pdf_path.stem}"
        collection_name = doc_collection
        col = get_chroma_collection(collection_name)
        if col.count() == 0:
            print(f"Ingesting {pdf_path.name} into collection '{collection_name}'...")
            ingest_pdf(pdf_path, collection_name=collection_name)

    pipeline = RAGPipeline(collection_name=collection_name)
    result = pipeline.answer_query(query=args.q, mode=args.mode)

    print("\n" + "=" * 65)
    print("GovDoc-RAG Grounded Response")
    print(f"Question:        {result['query']}")
    print(f"Trust Decision:  {result['decision']} (Score: {result['top_score']:.4f})")
    print(f"LLM Called:      {result['llm_called']}")
    print("-" * 65)
    print(f"Answer:\n{result['answer']}")
    print("-" * 65)

    if result["citations"]:
        print("Citations:")
        for c in result["citations"]:
            doc_id = c.get("doc_id", "doc")
            page = c.get("page", 1)
            score = c.get("score", 0.0)
            print(f"  - [{c.get('citation_id', 1)}] Document: {doc_id}, Page: {page} (Cosine: {score})")

    verif = result.get("verifier", {})
    print(f"\nHallucination Verifier: {'PASSED' if verif.get('pass') else 'FLAGGED'}")
    if verif.get("missing"):
        print(f"  Missing / Unverified Tokens: {verif['missing']}")
    print("=" * 65)

    append_log(
        f"CLI ask Q='{args.q}' Mode='{args.mode}' Decision='{result['decision']}' LLM={result['llm_called']}",
        module="M5:CLI_ASK",
    )


def main():
    parser = argparse.ArgumentParser(description="GovDoc-RAG CLI Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against a PDF document or corpus")
    ask_parser.add_argument("--pdf", type=str, default=None, help="Path to PDF document")
    ask_parser.add_argument("--q", "--query", dest="q", type=str, required=True, help="Question text")
    ask_parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "en", "hi", "kn", "hinglish", "kanglish"],
        help="Language mode",
    )
    ask_parser.add_argument(
        "--collection",
        type=str,
        default="govdoc_corpus",
        help="Target vector database collection",
    )

    args = parser.parse_args()

    if args.command == "ask":
        handle_ask(args)


if __name__ == "__main__":
    main()
