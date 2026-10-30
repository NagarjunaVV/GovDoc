"""
Unit tests for Module M7: Summarization, Digest, and Evaluation (Non-Negotiable Rules 1, 2, 6, 10).
"""

import json
from pathlib import Path
import pytest

from govdoc.eval.rouge_indic import (
    compute_corpus_rouge_all,
    compute_ngram_rouge,
    compute_rouge_all,
    compute_rouge_l,
    self_test_rouge,
    tokenize_indic,
)
from govdoc.eval.stats import bootstrap_ci_single, paired_bootstrap_test
from govdoc.summarize.digest import group_chunks_into_sections, rank_gists_mmr
from govdoc.summarize.summarizer import IndicBARTSummarizer


def test_rouge_unicode_aware_and_selftest():
    """Non-Negotiable Technical Rule 2: Unicode-aware ROUGE self-test and Indic token preservation."""
    # 1. Verify self-test identity property
    assert self_test_rouge() is True

    # 2. Verify tokenization preserves Hindi and Kannada characters
    hi_text = "प्रधानमंत्री नरेंद्र मोदी ने नई योजना का शुभारंभ किया।"
    kn_text = "ಕರ್ನಾಟಕ ಸರ್ಕಾರವು ರೈತರಿಗೆ ಹೊಸ ಯೋಜನೆಯನ್ನು ಘೋಷಿಸಿದೆ."

    hi_tokens = tokenize_indic(hi_text)
    kn_tokens = tokenize_indic(kn_text)

    assert len(hi_tokens) >= 7
    assert "प्रधानमंत्री" in hi_tokens
    assert len(kn_tokens) >= 5
    assert "ಕರ್ನಾಟಕ" in kn_tokens

    # 3. Partial overlap test
    cand = "प्रधानमंत्री नरेंद्र मोदी ने योजना शुरू की"
    ref = "प्रधानमंत्री नरेंद्र मोदी ने नई योजना शुरू की"
    res = compute_rouge_all(cand, ref)
    assert res["rouge1"]["f1"] > 0.8
    assert res["rougeL"]["f1"] > 0.8


def test_stats_paired_bootstrap():
    """Test paired bootstrap resampling and hypothesis testing."""
    # When System A and System B are identical, delta is 0 and p-val is non-significant
    scores_a = [0.4, 0.5, 0.3, 0.6, 0.45]
    scores_b = [0.4, 0.5, 0.3, 0.6, 0.45]
    res_null = paired_bootstrap_test(scores_a, scores_b, n_resamples=500, seed=42)
    assert res_null["delta"] == 0.0
    assert res_null["significant"] is False

    # When System A is uniformly superior to System B
    better_a = [0.75, 0.80, 0.70, 0.85, 0.90, 0.82, 0.78, 0.88]
    worse_b = [0.25, 0.30, 0.20, 0.35, 0.40, 0.32, 0.28, 0.38]
    res_sig = paired_bootstrap_test(better_a, worse_b, n_resamples=500, seed=42)
    assert res_sig["delta"] > 0.4
    assert res_sig["p_value"] < 0.05
    assert res_sig["significant"] is True

    # Single bootstrap CI
    mean_val, low_ci, high_ci = bootstrap_ci_single(better_a, n_resamples=500, seed=42)
    assert low_ci <= mean_val <= high_ci


def test_chunk_grouping_and_section_budget():
    """Verify group_chunks_into_sections respects max_tokens budget (<= 512)."""
    chunks = [
        {"chunk_id": f"chk_{i}", "page": i // 2 + 1, "text": f"Sentence number {i} with several words for testing chunk section grouping."}
        for i in range(25)
    ]
    sections = group_chunks_into_sections(chunks, max_tokens=100)
    assert len(sections) > 1
    for sec in sections:
        assert "text" in sec
        assert "pages" in sec
        assert "chunk_ids" in sec
        # Conservative word count check
        assert len(sec["text"].split()) <= 120


def test_mmr_ranking_deduplication():
    """Verify section gist ranking by centrality and MMR."""
    sample_gists = [
        {"gist": "Prime Minister inaugurated the highway project in Assam.", "pages": [1], "chunk_ids": ["c1"]},
        {"gist": "The road infrastructure will boost connectivity across eastern states.", "pages": [1, 2], "chunk_ids": ["c2"]},
        {"gist": "Tax audit deadlines extended by thirty days for corporations.", "pages": [3], "chunk_ids": ["c3"]},
    ]

    class MockEmbedder:
        def embed_texts(self, texts):
            import numpy as np
            # Return distinct synthetic normalized vectors
            vecs = np.zeros((len(texts), 8))
            vecs[0] = [1, 1, 0, 0, 0, 0, 0, 0]
            vecs[1] = [1, 0.9, 0, 0, 0, 0, 0, 0]
            vecs[2] = [0, 0, 0, 0, 1, 1, 0, 0]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

    ranked = rank_gists_mmr(sample_gists, embedder=MockEmbedder(), lambda_mmr=0.7)
    assert len(ranked) == 3
    for item in ranked:
        assert "centrality" in item
        assert "mmr_score" in item
