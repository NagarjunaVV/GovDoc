"""GovDoc-RAG ingestion package."""

from govdoc.ingest.pdf_extract import extract_pages_from_pdf
from govdoc.ingest.quality_gate import check_page_quality, filter_valid_pages
from govdoc.ingest.chunker import DocumentChunker
from govdoc.ingest.embed_index import (
    distance_to_cosine,
    BGEM3Embedder,
    get_chroma_collection,
    index_chunks,
    ingest_pdf,
)

__all__ = [
    "extract_pages_from_pdf",
    "check_page_quality",
    "filter_valid_pages",
    "DocumentChunker",
    "distance_to_cosine",
    "BGEM3Embedder",
    "get_chroma_collection",
    "index_chunks",
    "ingest_pdf",
]
