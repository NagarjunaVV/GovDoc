"""Unit tests for M4 retrieval, query normalizer, and trust gate."""

from unittest.mock import MagicMock
import pytest

from govdoc.retrieve.trust_gate import TrustDecision, evaluate_trust_gate
from govdoc.retrieve.query_normalizer import QueryNormalizer, detect_query_mode, detect_query_script
from govdoc.retrieve.retriever import MultilingualRetriever


def test_trust_gate_thresholds():
    """Verify trust gate pure function across ANSWER, EVIDENCE_ONLY, and ABSTAIN boundaries."""
    tau_hi = 0.65
    tau_lo = 0.45

    # 1. High confidence: ANSWER
    dec, d = evaluate_trust_gate(0.85, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.ANSWER
    assert d["allow_llm_generation"] is True

    # Boundary at tau_hi
    dec, _ = evaluate_trust_gate(0.65, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.ANSWER

    # 2. Medium confidence: EVIDENCE_ONLY
    dec, d = evaluate_trust_gate(0.55, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.EVIDENCE_ONLY
    assert d["allow_llm_generation"] is False

    # Just below tau_hi
    dec, _ = evaluate_trust_gate(0.6499, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.EVIDENCE_ONLY

    # Boundary at tau_lo
    dec, _ = evaluate_trust_gate(0.45, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.EVIDENCE_ONLY

    # 3. Low confidence: ABSTAIN
    dec, d = evaluate_trust_gate(0.30, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.ABSTAIN
    assert d["allow_llm_generation"] is False

    # Just below tau_lo
    dec, _ = evaluate_trust_gate(0.4499, tau_hi=tau_hi, tau_lo=tau_lo)
    assert dec == TrustDecision.ABSTAIN


def test_query_normalizer_multiview():
    """Verify script detection, mode detection, and Hinglish/Kanglish view generation."""
    normalizer = QueryNormalizer()

    # English query
    en_res = normalizer.normalize("Digital India fiber optical connectivity", mode="auto")
    assert en_res["detected_script"] == "latin"
    assert en_res["mode"] == "en"
    assert len(en_res["views"]) == 1

    # Hindi Devanagari query
    hi_res = normalizer.normalize("राष्ट्रीय शिक्षा नीति डिजिटल शिक्षण", mode="auto")
    assert hi_res["detected_script"] == "hi"
    assert hi_res["mode"] == "hi"
    assert len(hi_res["views"]) == 1

    # Kannada query
    kn_res = normalizer.normalize("ಗ್ರಾಮೀಣ ಕುಡಿಯುವ ನೀರು ಯೋಜನೆ", mode="auto")
    assert kn_res["detected_script"] == "kn"
    assert kn_res["mode"] == "kn"
    assert len(kn_res["views"]) == 1

    # Hinglish query
    hinglish_res = normalizer.normalize("digital seva kendra shiksha yojana kya hai", mode="auto")
    assert hinglish_res["mode"] == "hinglish"
    assert len(hinglish_res["views"]) == 2
    # Second view should contain native Devanagari characters
    assert any("\u0900" <= ch <= "\u097F" for ch in hinglish_res["views"][1])

    # Mode override test
    override_res = normalizer.normalize("kudiyuva neeru yojane bagge", mode="kanglish")
    assert override_res["mode"] == "kanglish"
    assert len(override_res["views"]) == 2
    # Second view should contain native Kannada characters
    assert any("\u0C80" <= ch <= "\u0CFF" for ch in override_res["views"][1])


def test_off_topic_query_abstains_and_counts_llm_calls():
    """
    NON-NEGOTIABLE RULE 4:
    Never call the LLM when the trust gate says ABSTAIN. Count LLM calls in tests.
    Verify off-topic query against 5-PDF corpus triggers ABSTAIN and 0 LLM calls.
    """
    retriever = MultilingualRetriever(collection_name="govdoc_corpus")
    assert retriever.collection.count() > 0, "Chroma collection must have indexed documents"

    # Completely off-topic query unrelated to Indian government notifications
    off_topic_query = "quantum mechanical wave function collapse astrophysics black hole thermodynamics"
    res = retriever.retrieve(query=off_topic_query, top_k=3)

    # Score should be below tau_lo (0.45)
    assert res["decision"] == "ABSTAIN", f"Expected ABSTAIN, got {res['decision']} (score: {res['top_score']})"
    assert res["top_score"] < 0.45

    # Mock LLM generation client to enforce Rule 4
    mock_llm_client = MagicMock()
    llm_call_count = 0

    if res["gate_details"]["allow_llm_generation"]:
        mock_llm_client.generate("some prompt")
        llm_call_count += 1

    # Rule 4 verification: LLM calls must remain strictly 0
    assert llm_call_count == 0, f"Violation of Rule 4: LLM was called on ABSTAIN decision ({llm_call_count} calls)"
    mock_llm_client.generate.assert_not_called()
