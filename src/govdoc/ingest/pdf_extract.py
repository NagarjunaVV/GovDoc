"""PDF extraction using PyMuPDF (fitz) with per-page text and metadata."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Return total number of pages in a PDF without reading or parsing full text."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def extract_pages_from_pdf(
    pdf_path: Path | str,
    doc_id: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    page_numbers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text per page from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: Optional identifier for the document. If None, uses file stem.
        start_page: Optional 1-indexed start page (inclusive).
        end_page: Optional 1-indexed end page (inclusive).
        page_numbers: Optional explicit list of 1-indexed page numbers to extract.

    Returns:
        List of dicts per page:
        {
            "doc_id": str,
            "page": int,          # 1-indexed page number
            "text": str,           # Extracted raw text
            "char_count": int,
            "word_count": int,
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc_id = doc_id or pdf_path.stem

    pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)

    try:
        total = len(doc)
        if page_numbers is not None:
            target_pages = [p for p in page_numbers if 1 <= p <= total]
        elif start_page is not None or end_page is not None:
            s = max(1, start_page or 1)
            e = min(total, end_page or total)
            target_pages = list(range(s, e + 1))
        else:
            target_pages = list(range(1, total + 1))

        for p_num in target_pages:
            page = doc[p_num - 1]
            # Extract plain text and normalize non-breaking spaces
            text = page.get_text("text") or ""
            text = text.replace("\xa0", " ")
            # Clean non-printable control characters except whitespace
            cleaned_text = "".join(
                ch for ch in text if ch.isprintable() or ch in "\n\r\t "
            ).strip()

            pages.append({
                "doc_id": doc_id,
                "page": p_num,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
            })
    finally:
        doc.close()

    return pages
