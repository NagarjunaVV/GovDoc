"""Sentence-aware document chunker adhering to BGE-M3 token bounds."""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

from govdoc.config import load_config
from govdoc.ingest.quality_gate import analyze_script_distribution

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Regex to split on English full stop, Devanagari danda (। / ॥), question/exclamation, or newlines
RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥\n])\s+")


class DocumentChunker:
    """Sentence-aware chunker targeting ~300 tokens with 50-token overlap."""

    def __init__(
        self,
        tokenizer_name_or_path: Optional[str] = None,
        target_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
    ):
        cfg = load_config()
        chunk_cfg = cfg.get("chunking", {})
        models_cfg = cfg.get("models", {})

        self.tokenizer_name = (
            tokenizer_name_or_path
            or models_cfg.get("embeddings", {}).get("name", "BAAI/bge-m3")
        )
        self.target_tokens = (
            target_tokens
            if target_tokens is not None
            else int(chunk_cfg.get("target_tokens", 300))
        )
        self.overlap_tokens = (
            overlap_tokens
            if overlap_tokens is not None
            else int(chunk_cfg.get("overlap_tokens", 50))
        )
        self.min_chunk_tokens = (
            min_chunk_tokens
            if min_chunk_tokens is not None
            else int(chunk_cfg.get("min_chunk_tokens", 30))
        )

        print(f"Loading chunker tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the BGE-M3 tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving sentence boundary punctuation."""
        raw_parts = RE_SENTENCE_SPLIT.split(text.strip())
        sentences = []
        for part in raw_parts:
            s = part.strip()
            if s:
                sentences.append(s)
        return sentences

    def chunk_page(self, page_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single page into overlapping chunks of ~target_tokens.

        Args:
            page_dict: Dict with fields 'doc_id', 'page', 'text', and optionally 'lang_guess'.

        Returns:
            List of chunk dicts:
            {
                "doc_id": str,
                "page": int,
                "chunk_id": str,
                "lang_guess": str,
                "text": str,
                "token_count": int,
            }
        """
        text = page_dict.get("text", "").strip()
        if not text:
            return []

        doc_id = page_dict.get("doc_id", "doc")
        page_num = page_dict.get("page", 1)
        lang_guess = page_dict.get("lang_guess")
        if not lang_guess:
            dist = analyze_script_distribution(text)
            lang_guess = dist.get("dominant_script", "en")

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # Tokenize all sentences upfront
        sent_tokens = [
            (s, self.count_tokens(s)) for s in sentences
        ]

        chunks: List[Dict[str, Any]] = []
        curr_sentences: List[str] = []
        curr_tokens = 0
        chunk_idx = 0

        for s, s_tok in sent_tokens:
            # Handle exceptionally long sentences (> target_tokens) by word-level slicing
            if s_tok > self.target_tokens:
                if curr_sentences:
                    chunk_text = " ".join(curr_sentences).strip()
                    c_tokens = self.count_tokens(chunk_text)
                    if c_tokens >= self.min_chunk_tokens:
                        chunks.append({
                            "doc_id": doc_id,
                            "page": page_num,
                            "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                            "lang_guess": lang_guess,
                            "text": chunk_text,
                            "token_count": c_tokens,
                        })
                        chunk_idx += 1
                    curr_sentences = []
                    curr_tokens = 0

                # Slice long sentence into pieces
                words = s.split()
                sub_curr: List[str] = []
                sub_tok = 0
                for w in words:
                    w_tok = self.count_tokens(w + " ")
                    if sub_tok + w_tok > self.target_tokens and sub_curr:
                        sub_text = " ".join(sub_curr).strip()
                        c_tokens = self.count_tokens(sub_text)
                        chunks.append({
                            "doc_id": doc_id,
                            "page": page_num,
                            "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                            "lang_guess": lang_guess,
                            "text": sub_text,
                            "token_count": c_tokens,
                        })
                        chunk_idx += 1
                        sub_curr = []
                        sub_tok = 0
                    sub_curr.append(w)
                    sub_tok += w_tok

                if sub_curr:
                    curr_sentences = sub_curr
                    curr_tokens = sub_tok
                continue

            if curr_tokens + s_tok > self.target_tokens and curr_sentences:
                chunk_text = " ".join(curr_sentences).strip()
                c_tokens = self.count_tokens(chunk_text)
                if c_tokens >= self.min_chunk_tokens:
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                        "lang_guess": lang_guess,
                        "text": chunk_text,
                        "token_count": c_tokens,
                    })
                    chunk_idx += 1

                # Calculate overlap: take trailing sentences up to overlap_tokens
                overlap_sents: List[str] = []
                overlap_tok_count = 0
                for trailing_s in reversed(curr_sentences):
                    t_tok = self.count_tokens(trailing_s)
                    if overlap_tok_count + t_tok <= self.overlap_tokens:
                        overlap_sents.insert(0, trailing_s)
                        overlap_tok_count += t_tok
                    else:
                        break

                curr_sentences = overlap_sents + [s]
                curr_tokens = overlap_tok_count + s_tok
            else:
                curr_sentences.append(s)
                curr_tokens += s_tok

        # Append final chunk if non-empty
        if curr_sentences:
            chunk_text = " ".join(curr_sentences).strip()
            c_tokens = self.count_tokens(chunk_text)
            if c_tokens >= self.min_chunk_tokens or not chunks:
                chunks.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                    "lang_guess": lang_guess,
                    "text": chunk_text,
                    "token_count": c_tokens,
                })

        return chunks

    def chunk_document(self, valid_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a multi-page document while preserving page mappings and metadata."""
        all_chunks = []
        for page_dict in valid_pages:
            page_chunks = self.chunk_page(page_dict)
            all_chunks.extend(page_chunks)
        return all_chunks
