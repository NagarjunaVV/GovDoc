"""Unit tests for M8: Retrieval eval, abstain eval, verifier eval."""

import json
import math
import sys
from pathlib import Path
from typing import List
import numpy as np
import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Recall@K ──────────────────────────────────────────────────────────────────


from govdoc.eval.eval_retrieval import recall_at_k, reciprocal_rank, cosine_sim_matrix


def test_recall_at_k_hit():
    ranked = ["doc_A", "doc_B", "doc_C"]
    assert recall_at_k(ranked, "doc_A", 1) == 1.0
    assert recall_at_k(ranked, "doc_B", 1) == 0.0
    assert recall_at_k(ranked, "doc_B", 2) == 1.0
    assert recall_at_k(ranked, "doc_C", 3) == 1.0


def test_recall_at_k_miss():
    ranked = ["doc_A", "doc_B"]
    assert recall_at_k(ranked, "doc_X", 5) == 0.0


def test_reciprocal_rank():
    ranked = ["doc_A", "doc_B", "doc_C"]
    assert reciprocal_rank(ranked, "doc_A") == pytest.approx(1.0)
    assert reciprocal_rank(ranked, "doc_B") == pytest.approx(0.5)
    assert reciprocal_rank(ranked, "doc_C") == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, "doc_X") == pytest.approx(0.0)


def test_cosine_sim_matrix_identical():
    """Cosine similarity of a vector with itself should be 1.0."""
    v = np.random.randn(1, 64).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    scores = cosine_sim_matrix(v, v)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_cosine_sim_matrix_orthogonal():
    """Cosine similarity of orthogonal vectors should be 0.0."""
    a = np.zeros((1, 4), dtype=np.float32)
    a[0, 0] = 1.0
    b = np.zeros((1, 4), dtype=np.float32)
    b[0, 1] = 1.0
    scores = cosine_sim_matrix(a, b)
    assert scores[0] == pytest.approx(0.0, abs=1e-5)


# ── AUROC ──────────────────────────────────────────────────────────────────────


from govdoc.eval.eval_abstain import compute_auroc, compute_sweep_metrics, calibrate_tau


def test_auroc_perfect():
    """Perfect separation: AUROC should be 1.0."""
    answerable = [0.9, 0.85, 0.8]
    unanswerable = [0.3, 0.2, 0.1]
    auroc = compute_auroc(answerable, unanswerable)
    assert auroc == pytest.approx(1.0, abs=0.01)


def test_auroc_random():
    """Random scores: AUROC should be ~0.5."""
    rng = np.random.RandomState(42)
    scores = rng.uniform(0, 1, 200).tolist()
    auroc = compute_auroc(scores[:100], scores[100:])
    assert 0.3 <= auroc <= 0.7, f"Expected ~0.5, got {auroc}"


def test_tau_calibration_selects_far_within_budget():
    """calibrate_tau should find tau with FAR <= target."""
    ans_scores = [0.8, 0.7, 0.75, 0.65, 0.85]
    unans_scores = [0.3, 0.2, 0.25, 0.4, 0.35]
    tau_values = [round(0.20 + 0.05 * i, 2) for i in range(15)]
    sweep = compute_sweep_metrics(ans_scores, unans_scores, tau_values)
    tau = calibrate_tau(sweep, target_far=0.05)
    # Verify that chosen tau meets the budget on the sweep
    row = min(sweep, key=lambda r: abs(r["tau"] - tau))
    assert row["far"] <= 0.05, f"Expected FAR <= 0.05, got {row['far']}"


def test_sweep_metrics_shape():
    """Sweep should return one row per tau value."""
    ans = [0.7, 0.8, 0.6]
    unans = [0.3, 0.4, 0.2]
    tau_values = [0.3, 0.5, 0.7]
    rows = compute_sweep_metrics(ans, unans, tau_values)
    assert len(rows) == 3
    for row in rows:
        assert "tau" in row
        assert "far" in row
        assert "recall_answerable" in row
        assert "balanced_accuracy" in row


def test_sweep_extreme_taus():
    """At tau=0 everything is ANSWER → recall=1, far=0. At tau=1 everything ABSTAINS → recall=0, far=1."""
    ans = [0.5, 0.6, 0.7]
    unans = [0.2, 0.3]
    rows_low = compute_sweep_metrics(ans, unans, [0.0])
    assert rows_low[0]["recall_answerable"] == pytest.approx(1.0)
    assert rows_low[0]["far"] == pytest.approx(0.0)

    rows_high = compute_sweep_metrics(ans, unans, [1.01])
    assert rows_high[0]["recall_answerable"] == pytest.approx(0.0)
    assert rows_high[0]["far"] == pytest.approx(1.0)


# ── Score JSONL schema ─────────────────────────────────────────────────────────


def test_scores_jsonl_schema(tmp_path):
    """Validate scores_val.jsonl and scores_test.jsonl field schema."""
    required_fields = {"qid", "lang", "split", "top1_score", "answerable", "abstain"}
    for split in ("val", "test"):
        path = Path("results/raw") / f"scores_{split}.jsonl"
        if not path.exists():
            pytest.skip(f"{path} not yet generated — run eval_abstain.py first")
        items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(items) > 0, f"scores_{split}.jsonl is empty"
        for item in items[:10]:
            missing = required_fields - set(item.keys())
            assert not missing, f"Missing fields {missing} in {split} record"
            assert isinstance(item["top1_score"], float)
            assert isinstance(item["abstain"], bool)


# ── abstain.json schema ────────────────────────────────────────────────────────


def test_abstain_json_schema():
    """Validate abstain.json schema per requirements.md 5.5."""
    path = Path("results/raw/abstain.json")
    if not path.exists():
        pytest.skip("abstain.json not yet generated — run eval_abstain.py first")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Rule 6 required fields
    for field in ("experiment_id", "date", "git_commit", "config_hash", "seed", "n_samples", "metrics"):
        assert field in data, f"Missing top-level field: {field}"

    m = data["metrics"]
    assert "auroc" in m
    assert "calibrated_tau" in m
    assert "tau_sweep" in m
    assert isinstance(m["tau_sweep"], list)
    assert len(m["tau_sweep"]) > 0
    # Check 0 <= AUROC <= 1
    assert 0.0 <= m["auroc"] <= 1.0


# ── retrieval_A.json schema ────────────────────────────────────────────────────


def test_retrieval_json_schema():
    """Validate retrieval_A.json schema per Rule 6."""
    path = Path("results/raw/retrieval_A.json")
    if not path.exists():
        pytest.skip("retrieval_A.json not yet generated — run eval_retrieval.py first")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for field in ("experiment_id", "date", "git_commit", "config_hash", "seed", "n_samples", "metrics"):
        assert field in data, f"Missing top-level field: {field}"

    m = data["metrics"]
    assert "E2_raw_only" in m
    assert "E3_raw_plus_translit" in m
    assert "overall" in m["E2_raw_only"]
    assert "MRR" in m["E2_raw_only"]["overall"]


# ── verifier.json schema ───────────────────────────────────────────────────────


def test_verifier_json_schema():
    """Validate verifier.json schema."""
    path = Path("results/raw/verifier.json")
    if not path.exists():
        pytest.skip("verifier.json not yet generated — run eval_verifier.py first")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for field in ("experiment_id", "date", "git_commit", "config_hash", "seed", "n_samples", "metrics"):
        assert field in data

    m = data["metrics"]
    assert "overall_grounded_pass_rate" in m
    assert "overall_corrupted_flag_rate" in m
    # Flag rate should be > 0 (corrupted answers should be detected)
    assert m["overall_corrupted_flag_rate"] > 0.0


# ── LLM call gate integration ──────────────────────────────────────────────────


def test_gate_no_llm_call_when_abstain():
    """Non-Negotiable Rule 4: LLM must NOT be called when trust gate says ABSTAIN."""
    from govdoc.retrieve.trust_gate import evaluate_trust_gate, TrustDecision

    call_count = [0]
    original_chat = None

    # Inject a spy into LLMClient to catch any stray calls
    try:
        from govdoc.generate.llm_client import LLMClient
        original_chat = LLMClient.chat

        def spy_chat(self, *args, **kwargs):
            call_count[0] += 1
            return original_chat(self, *args, **kwargs)

        LLMClient.chat = spy_chat
    except Exception:
        pass  # LLMClient may not be importable in test env

    # Score below tau_lo → ABSTAIN
    decision, _ = evaluate_trust_gate(0.1)
    assert decision == TrustDecision.ABSTAIN
    assert call_count[0] == 0, f"LLM was called {call_count[0]} time(s) on ABSTAIN decision"

    if original_chat is not None:
        LLMClient.chat = original_chat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
