"""
Document Digest Generator (plan.md section 4, M7).

Groups document chunks into <=512-token sections, generates a gist per section in the target
language using IndicBART (with Rule 1 script unification), ranks gists by centrality + MMR deduplication
using BGE-M3 embeddings, and returns the final structured digest with page references.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from govdoc.config import load_config
from govdoc.summarize.summarizer import IndicBARTSummarizer
from govdoc.summarize.translator import IndicTrans2Translator
from govdoc.ingest.embed_index import BGEM3Embedder


def group_chunks_into_sections(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 512,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Group sequential document chunks into sections not exceeding max_tokens.

    Args:
        chunks: List of chunk dicts containing 'text', 'page', and optionally 'chunk_id'.
        max_tokens: Maximum token budget per section (default: 512).
        tokenizer: Optional tokenizer; if None, approximates with whitespace count.

    Returns:
        List of section dicts: {'text': str, 'pages': List[int], 'chunk_ids': List[str]}
    """
    if not chunks:
        return []

    def get_token_count(text: str) -> int:
        if tokenizer is not None and hasattr(tokenizer, "tokenize"):
            return len(tokenizer.tokenize(text))
        # Conservative approximation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)

    sections: List[Dict[str, Any]] = []
    current_texts: List[str] = []
    current_pages: List[int] = []
    current_ids: List[str] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue

        c_tokens = get_token_count(chunk_text)
        page = chunk.get("page", 1)
        cid = chunk.get("chunk_id", f"chk_{len(current_ids)}")

        if current_tokens + c_tokens > max_tokens and current_texts:
            sections.append({
                "text": " ".join(current_texts),
                "pages": sorted(list(set(current_pages))),
                "chunk_ids": list(current_ids),
            })
            current_texts = [chunk_text]
            current_pages = [page]
            current_ids = [cid]
            current_tokens = c_tokens
        else:
            current_texts.append(chunk_text)
            current_pages.append(page)
            current_ids.append(cid)
            current_tokens += c_tokens

    if current_texts:
        sections.append({
            "text": " ".join(current_texts),
            "pages": sorted(list(set(current_pages))),
            "chunk_ids": list(current_ids),
        })

    return sections


def rank_gists_mmr(
    gists: List[Dict[str, Any]],
    embedder: Optional[BGEM3Embedder] = None,
    lambda_mmr: float = 0.7,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rank section gists using Centroid Centrality and Maximal Marginal Relevance (MMR).

    Args:
        gists: List of gist dicts: {'gist': str, 'pages': List[int], 'chunk_ids': List[str]}
        embedder: BGEM3Embedder instance.
        lambda_mmr: Diversity vs relevance trade-off (0.7 favours centrality, 0.3 diversity).
        top_k: Optional number of top gists to retain.

    Returns:
        List of ranked gist dicts with 'centrality' and 'mmr_score'.
    """
    if not gists:
        return []
    if len(gists) == 1:
        gists[0]["centrality"] = 1.0
        gists[0]["mmr_score"] = 1.0
        return gists

    if embedder is None:
        embedder = BGEM3Embedder()

    gist_texts = [g["gist"] for g in gists]
    # Vectors are L2 normalized (N, D)
    vectors = embedder.embed_texts(gist_texts)

    # Document centroid vector
    centroid = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-6:
        centroid = centroid / norm

    # Centrality = cosine similarity to document centroid
    centralities = np.dot(vectors, centroid)

    # Pairwise similarity matrix
    sim_matrix = np.dot(vectors, vectors.T)

    # MMR selection
    n = len(gists)
    k = min(top_k or n, n)

    selected_indices: List[int] = []
    unselected_indices = list(range(n))

    # First pick: highest centrality
    best_idx = int(np.argmax(centralities))
    selected_indices.append(best_idx)
    unselected_indices.remove(best_idx)

    mmr_scores = {best_idx: float(centralities[best_idx])}

    while len(selected_indices) < k and unselected_indices:
        best_mmr = -float("inf")
        best_candidate = unselected_indices[0]

        for cand in unselected_indices:
            sim_to_selected = max(sim_matrix[cand, sel] for sel in selected_indices)
            mmr = lambda_mmr * centralities[cand] - (1.0 - lambda_mmr) * sim_to_selected
            if mmr > best_mmr:
                best_mmr = mmr
                best_candidate = cand

        selected_indices.append(best_candidate)
        unselected_indices.remove(best_candidate)
        mmr_scores[best_candidate] = float(best_mmr)

    ranked_gists = []
    for idx in selected_indices:
        item = dict(gists[idx])
        item["centrality"] = round(float(centralities[idx]), 4)
        item["mmr_score"] = round(float(mmr_scores[idx]), 4)
        ranked_gists.append(item)

    return ranked_gists


def generate_digest(
    chunks: List[Dict[str, Any]],
    src_lang: str = "en",
    tgt_lang: str = "hi",
    summarizer: Optional[IndicBARTSummarizer] = None,
    translator: Optional[IndicTrans2Translator] = None,
    embedder: Optional[BGEM3Embedder] = None,
    llm_client: Optional[Any] = None,
    max_tokens_per_section: int = 512,
    lambda_mmr: float = 0.7,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an offline executive document digest with page citations.

    Workflow (IndicBART Summarise -> IndicTrans2 Translate):
    1. Group sequential chunks into <=512 token sections.
    2. Generate section-level monolingual gist using IndicBART (en-en, hi-hi, kn-kn).
    3. If cross-lingual (e.g. English -> Hindi/Kannada), translate the concise gist using IndicTrans2
       (or active SLM fallback).
    4. Rank and deduplicate gists using BGE-M3 Centrality + MMR.
    5. Compile final cohesive digest with page citations.

    Returns:
        {
            "digest": str,
            "gists": List[Dict[str, Any]],
            "num_sections": int,
            "target_language": str
        }
    """
    if not chunks:
        return {
            "digest": "No content provided to summarize.",
            "gists": [],
            "num_sections": 0,
            "target_language": tgt_lang,
        }

    if summarizer is None:
        summarizer = IndicBARTSummarizer()

    # Step 1: Section grouping
    sections = group_chunks_into_sections(
        chunks=chunks,
        max_tokens=max_tokens_per_section,
        tokenizer=summarizer.tokenizer,
    )

    # Step 2: Gist generation per section (IndicBART Monolingual -> IndicTrans2 Translate)
    raw_gists: List[Dict[str, Any]] = []
    src_clean = src_lang.strip().lower()
    tgt_clean = tgt_lang.strip().lower()
    is_cross_lingual = (src_clean != tgt_clean)

    # Initialize translator if cross-lingual and not provided
    if is_cross_lingual and translator is None:
        try:
            translator = IndicTrans2Translator()
        except Exception:
            translator = None

    mono_lang = src_clean if is_cross_lingual else tgt_clean
    sec_texts = [sec["text"] for sec in sections]

    # Batch summarize all sections concurrently on GPU for high throughput
    if hasattr(summarizer, "summarise_batch") and len(sec_texts) > 1:
        try:
            mono_gists = summarizer.summarise_batch(
                sec_texts,
                src_lang=mono_lang,
                tgt_lang=mono_lang,
                max_length=64,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        except Exception:
            mono_gists = [
                summarizer.summarise(
                    t,
                    src_lang=mono_lang,
                    tgt_lang=mono_lang,
                    max_length=64,
                    min_length=8,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
                for t in sec_texts
            ]
    else:
        mono_gists = [
            summarizer.summarise(
                t,
                src_lang=mono_lang,
                tgt_lang=mono_lang,
                max_length=64,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
            for t in sec_texts
        ]

    # Batch translate if cross-lingual
    translated_gists: List[str] = []
    if is_cross_lingual:
        if translator is not None and hasattr(translator, "translate_batch"):
            try:
                translated_gists = translator.translate_batch(
                    mono_gists,
                    src_lang=src_clean,
                    tgt_lang=tgt_clean,
                    max_length=128,
                    num_beams=4,
                )
            except Exception:
                translated_gists = []

        if not translated_gists or len(translated_gists) != len(mono_gists):
            translated_gists = []
            for mg in mono_gists:
                t_gist = ""
                if translator is not None:
                    try:
                        t_gist = translator.translate(mg, src_lang=src_clean, tgt_lang=tgt_clean)
                    except Exception:
                        t_gist = ""
                if not t_gist and llm_client is not None:
                    try:
                        p = f"Translate the following text into clear, official {tgt_clean.upper()}:\n{mg}\nTranslation:"
                        res = llm_client.chat(prompt=p, max_tokens=128, temperature=0.1)
                        t_gist = res.get("content", "").strip()
                    except Exception:
                        t_gist = ""
                translated_gists.append(t_gist or mg)

    final_gists = translated_gists if is_cross_lingual else mono_gists

    for i, sec in enumerate(sections):
        gist_text = final_gists[i] if i < len(final_gists) else ""
        if gist_text:
            raw_gists.append({
                "gist": gist_text,
                "pages": sec["pages"],
                "chunk_ids": sec["chunk_ids"],
                "section_index": i,
            })

    if not raw_gists:
        return {
            "digest": "Unable to generate digest from document.",
            "gists": [],
            "num_sections": len(sections),
            "target_language": tgt_lang,
        }

    # Step 3: Centrality Ranking & MMR Deduplication
    ranked_gists = rank_gists_mmr(
        gists=raw_gists,
        embedder=embedder,
        lambda_mmr=lambda_mmr,
        top_k=top_k,
    )

    # Sort selected gists back to natural document reading order
    chronological_gists = sorted(ranked_gists, key=lambda x: x.get("section_index", 0))

    # Step 4: Assemble formatted digest text with page references
    digest_lines: List[str] = []
    for g in chronological_gists:
        pages_str = ", ".join(str(p) for p in g["pages"])
        digest_lines.append(f"• {g['gist']} [Page {pages_str}]")

    full_digest = "\n".join(digest_lines)

    return {
        "digest": full_digest,
        "gists": ranked_gists,
        "num_sections": len(sections),
        "target_language": tgt_lang,
    }
