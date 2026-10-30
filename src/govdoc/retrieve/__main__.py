"""CLI entry point for GovDoc-RAG retrieval and trust gate evaluation."""

import argparse
import sys
from typing import Optional

from govdoc.config import load_config
from govdoc.logging_utils import append_log
from govdoc.retrieve.retriever import MultilingualRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="GovDoc-RAG Multilingual Query Retrieval")
    parser.add_argument(
        "--q",
        "--query",
        dest="query",
        type=str,
        required=True,
        help="Search query text (English, Hindi, Kannada, Hinglish, or Kanglish)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="govdoc_corpus",
        help="ChromaDB target collection name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "en", "hi", "kn", "hinglish", "kanglish"],
        help="Language / transliteration mode override",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence chunks to display (default: 3)",
    )
    args = parser.parse_args()

    retriever = MultilingualRetriever(collection_name=args.collection)
    result = retriever.retrieve(query=args.query, mode=args.mode, top_k=args.top_k)

    print("=" * 65)
    print("GovDoc-RAG Retrieval & Trust Gate Report")
    print(f"Query:            {result['query']}")
    print(f"Detected Script:  {result['detected_script']}")
    print(f"Resolved Mode:    {result['mode']}")
    print(f"Query Views:      {result['views']}")
    print(f"Trust Decision:   {result['decision']}")
    print(f"Top Score:        {result['top_score']}")
    print(f"Gate Details:     {result['gate_details']}")
    print("=" * 65)

    print(f"\nTop-{len(result['chunks'])} Retrieved Evidence Chunks:")
    for idx, c in enumerate(result["chunks"]):
        print(f"\n[{idx + 1}] Score: {c['score']} (Cosine) | Doc: {c['doc_id']} | Page: {c['page']} | Lang: {c['lang_guess']}")
        print(f"    Text: {c['text'][:200]}...")

    append_log(
        f"Retrieval Query='{args.query}' -> Decision={result['decision']}, Top Score={result['top_score']}, Chunks={len(result['chunks'])}",
        module="M4:RETRIEVE",
    )


if __name__ == "__main__":
    main()
