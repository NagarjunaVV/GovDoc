"""Unit tests for M5 generation, hallucination verifier, and sequential residency."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from govdoc.generate.verifier import verify_grounded_answer, normalize_digits
from govdoc.generate.rag_pipeline import RAGPipeline
from govdoc.generate.llm_client import LLMClient
from govdoc.runtime.model_manager import ModelManager


def test_verifier_grounded_and_hallucinated():
    """Verify factual verifier correctly validates numbers and flags invented amounts."""
    evidence = [
        {
            "doc_id": "test_doc",
            "page": 1,
            "text": "The Central Government approved an outlay of Rs. 4,500 Crores by June 2026 for Digital India.",
        }
    ]

    # Case 1: Grounded answer with exact figures
    grounded_ans = "The capital outlay approved is Rs. 4,500 Crores by June 2026 under Digital India [1]."
    res1 = verify_grounded_answer(grounded_ans, evidence)
    assert res1["pass"] is True, f"Expected pass, got missing: {res1['missing']}"
    assert len(res1["missing"]) == 0

    # Case 2: Devanagari numerals normalized to ASCII
    deva_ans = "सरकार ने ४,५०० करोड़ रुपये का बजट स्वीकृत किया [1]।"
    res2 = verify_grounded_answer(deva_ans, evidence)
    assert res2["pass"] is True, f"Devanagari numerals should pass after normalization: {res2['missing']}"

    # Case 3: Invented amount / hallucination
    hallucinated_ans = "The Central Government approved an outlay of Rs. 99,999 Crores by December 2035 [1]."
    res3 = verify_grounded_answer(hallucinated_ans, evidence)
    assert res3["pass"] is False, "Invented amount must be flagged by verifier"
    assert any("99" in item for item in res3["missing"]), f"Expected 99,999 in missing: {res3['missing']}"


def test_gate_abstains_enforces_rule_4():
    """
    NON-NEGOTIABLE RULE 4:
    Never call the LLM when the trust gate says ABSTAIN. Count LLM calls in tests.
    """
    # Create mock LLM client with counter
    mock_llm = MagicMock()
    mock_llm.chat = MagicMock()
    mock_llm.call_count = 0

    pipeline = RAGPipeline(collection_name="govdoc_corpus", llm_client=mock_llm)

    # Off-topic query known to yield ABSTAIN (< 0.45 cosine)
    off_topic_q = "interstellar astrophysics quantum gravity Hawking radiation"
    res = pipeline.answer_query(off_topic_q)

    assert res["decision"] == "ABSTAIN"
    assert res["llm_called"] is False
    assert "Information not found" in res["answer"]

    # Strictly verify 0 LLM calls occurred
    assert mock_llm.chat.call_count == 0, f"Violation of Rule 4: LLM called {mock_llm.chat.call_count} times on ABSTAIN"


def test_model_manager_sequential_residency():
    """Verify sequential residency model manager switches modes and updates vram_trace."""
    mm = ModelManager(initial_mode="qa")
    assert mm.get_active_mode() == "qa"

    res = mm.set_mode("ingest")
    assert res["mode"] == "ingest"
    assert mm.get_active_mode() == "ingest"

    trace_file = Path("results/raw/vram_trace.jsonl")
    assert trace_file.exists(), "vram_trace.jsonl should be created on mode switches"
    assert trace_file.stat().st_size > 0
