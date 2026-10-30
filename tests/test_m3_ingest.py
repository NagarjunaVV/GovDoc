"""Unit tests for M3 PDF ingestion, quality gate, chunking, and ChromaDB indexing."""

import pytest
from pathlib import Path

from govdoc.ingest.embed_index import distance_to_cosine, sanitize_collection_name
from govdoc.ingest.pdf_extract import extract_pages_from_pdf, get_pdf_page_count
from govdoc.ingest.quality_gate import check_page_quality, filter_valid_pages
from govdoc.ingest.chunker import DocumentChunker


def test_distance_to_cosine():
    """
    NON-NEGOTIABLE RULE 3:
    Chroma returns distance = 1 - cosine for cosine space.
    Verify distance_to_cosine() strictly handles mathematical conversion.
    """
    # Identical vectors: distance 0.0 -> cosine 1.0
    assert distance_to_cosine(0.0) == pytest.approx(1.0, rel=1e-6)

    # Orthogonal vectors: distance 1.0 -> cosine 0.0
    assert distance_to_cosine(1.0) == pytest.approx(0.0, rel=1e-6)

    # Opposite vectors: distance 2.0 -> cosine -1.0
    assert distance_to_cosine(2.0) == pytest.approx(-1.0, rel=1e-6)

    # Arbitrary intermediate distance
    assert distance_to_cosine(0.35) == pytest.approx(0.65, rel=1e-6)
    assert distance_to_cosine(0.80) == pytest.approx(0.20, rel=1e-6)


def test_chunk_length_bounds():
    """Verify sentence-aware chunker adheres to token bounds and overlap."""
    sample_text = (
        "The Digital India program is a flagship programme of the Government of India with a vision "
        "to transform India into a digitally empowered society and knowledge economy. "
        "Faceless, Paperless, Cashless is one of the professed roles of Digital India. "
        "The programme weaves together a large number of ideas and thoughts into a single, comprehensive vision. "
        "Each idea stands on its own while being part of a larger whole. "
        "The focus is on making technology central to enabling change. "
        "It is coordinated by the Department of Electronics and Information Technology. "
        "All existing and ongoing e-Governance initiatives have been revamped to align with Digital India. "
        "Scope of Mobile connectivity: Universal access to mobile connectivity across all rural villages. "
        "Public Internet Access Programme: A national rural internet mission across 250,000 panchayats. "
        "Information for All: Transparent access to open government data and records."
    )
    chunker = DocumentChunker(target_tokens=50, overlap_tokens=15, min_chunk_tokens=10)
    page_dict = {"doc_id": "test_doc", "page": 1, "text": sample_text, "lang_guess": "en"}
    chunks = chunker.chunk_page(page_dict)

    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    for c in chunks:
        tok_count = c["token_count"]
        assert tok_count >= 10, f"Chunk {c['chunk_id']} below minimum tokens ({tok_count})"
        assert tok_count <= 85, f"Chunk {c['chunk_id']} exceeded upper token bound ({tok_count})"
        assert c["doc_id"] == "test_doc"
        assert c["page"] == 1
        assert len(c["text"]) > 0


def test_page_mapping_and_quality_gate():
    """Verify PDF extraction, page numbering, quality filtering, and chunk metadata mapping."""
    pdf_path = Path("data/sample_pdfs/doc4_finance_tax_circular_en.pdf")
    assert pdf_path.exists(), f"Sample PDF missing: {pdf_path}"

    # Extract pages
    pages = extract_pages_from_pdf(pdf_path)
    assert len(pages) == 3, f"Expected 3 pages, found {len(pages)}"

    # Check page 1
    assert pages[0]["page"] == 1
    assert "MINISTRY OF FINANCE" in pages[0]["text"]

    # Check page 2 (sparse page)
    assert pages[1]["page"] == 2
    assert "Page 2" in pages[1]["text"]

    # Check page 3
    assert pages[2]["page"] == 3
    assert "ANNEXURE I" in pages[2]["text"]

    # Quality Gate filtering
    valid_pages, reports = filter_valid_pages(pages, min_chars=15)
    assert len(reports) == 3

    # Page 1 & 3 should pass, Page 2 should fail (EMPTY_OR_SPARSE_TEXT)
    assert reports[0]["passed"] is True
    assert reports[1]["passed"] is False
    assert "EMPTY_OR_SPARSE_TEXT" in reports[1]["flags"]
    assert reports[2]["passed"] is True

    assert len(valid_pages) == 2

    # Chunker page mapping
    chunker = DocumentChunker(target_tokens=150, overlap_tokens=30, min_chunk_tokens=10)
    chunks = chunker.chunk_document(valid_pages)

    # Check that chunks retain proper page mapping
    chunk_pages = {c["page"] for c in chunks}
    assert 1 in chunk_pages
    assert 3 in chunk_pages
    assert 2 not in chunk_pages  # Page 2 was filtered out by quality gate


def test_pdf_page_selection_and_count():
    """Verify get_pdf_page_count and partial extraction (page range / explicit pages)."""
    pdf_path = Path("data/sample_pdfs/doc4_finance_tax_circular_en.pdf")
    assert pdf_path.exists()

    # 1. Total page count
    total = get_pdf_page_count(pdf_path)
    assert total == 3

    # 2. Extract slice with start_page and end_page
    sliced_pages = extract_pages_from_pdf(pdf_path, start_page=1, end_page=2)
    assert len(sliced_pages) == 2
    assert [p["page"] for p in sliced_pages] == [1, 2]

    # 3. Extract explicit page numbers
    custom_pages = extract_pages_from_pdf(pdf_path, page_numbers=[1, 3])
    assert len(custom_pages) == 2
    assert [p["page"] for p in custom_pages] == [1, 3]


def test_sanitize_collection_name():
    """Verify collection name sanitization satisfies ChromaDB validation rules."""
    # 1. Names with spaces and punctuation (from the user error trace)
    name1 = "doc_Press Release: Press Information Bureau"
    clean1 = sanitize_collection_name(name1)
    assert clean1 == "doc_Press_Release_Press_Information_Bureau"
    assert clean1[0].isalnum() and clean1[-1].isalnum()

    # 2. Leading/trailing underscores and dots
    name2 = "..__test_doc--__"
    clean2 = sanitize_collection_name(name2)
    assert clean2 == "test_doc"

    # 3. Very short name (< 3 chars)
    name3 = "a"
    clean3 = sanitize_collection_name(name3)
    assert len(clean3) >= 3
    assert clean3[0].isalnum() and clean3[-1].isalnum()

    # 4. Empty / None
    clean_empty = sanitize_collection_name("")
    assert len(clean_empty) >= 3
    assert clean_empty == "govdoc_default"

    # 5. Overly long name (> 512 chars)
    name_long = "a" * 600
    clean_long = sanitize_collection_name(name_long)
    assert len(clean_long) <= 512

