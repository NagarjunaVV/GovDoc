"""Unit tests for M9 Streamlit App and UI components."""

import json
import os
import sys
from pathlib import Path
import pytest

from govdoc.config import resolve_path
from govdoc.runtime.model_manager import ModelManager
from govdoc.runtime.vram_monitor import get_gpu_memory_used_mb, get_gpu_peak_memory_mb
from govdoc.retrieve.trust_gate import TrustDecision, evaluate_trust_gate


def test_app_offline_environment_variables():
    """Verify offline environment variables are strictly enforced (Rule 8)."""
    # Import triggers the environment setting
    import govdoc.app.streamlit_app

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"
    assert os.environ.get("STREAMLIT_BROWSER_GATHER_USAGE_STATS") == "false"


def test_model_manager_sequential_residency_in_app():
    """Verify ModelManager sequential residency handles mode switching for app tabs."""
    mm = ModelManager(initial_mode="qa")
    assert mm.get_active_mode() == "qa"

    # Switch to summarise for Tab 3
    res_summ = mm.set_mode("summarise")
    assert mm.get_active_mode() == "summarise"
    assert res_summ["status"] in ("success", "already_active")

    # Switch to ingest for Tab 1
    res_ingest = mm.set_mode("ingest")
    assert mm.get_active_mode() == "ingest"
    assert res_ingest["status"] in ("success", "already_active")

    # Restore to qa for Tab 2
    res_qa = mm.set_mode("qa")
    assert mm.get_active_mode() == "qa"
    assert res_qa["status"] in ("success", "already_active")


def test_vram_telemetry_reads_mb():
    """Verify VRAM telemetry reports non-negative float MB values."""
    used = get_gpu_memory_used_mb()
    peak = get_gpu_peak_memory_mb()
    assert isinstance(used, float)
    assert isinstance(peak, float)
    assert used >= 0.0
    assert peak >= 0.0


def test_trust_gate_decision_confidence_states():
    """Verify 3-way trust gate decisions map to appropriate confidence bands."""
    # High confidence -> ANSWER
    dec_high, details_high = evaluate_trust_gate(0.85)
    assert dec_high == TrustDecision.ANSWER

    # Mid confidence -> EVIDENCE_ONLY
    dec_mid, details_mid = evaluate_trust_gate(0.55)
    assert dec_mid == TrustDecision.EVIDENCE_ONLY

    # Low confidence -> ABSTAIN
    dec_low, details_low = evaluate_trust_gate(0.30)
    assert dec_low == TrustDecision.ABSTAIN


def test_results_raw_benchmark_files_exist_for_tab4():
    """Verify all benchmark JSON files required by Tab 4 exist in results/raw/."""
    raw_dir = resolve_path("results/raw")
    required_files = [
        "retrieval_A.json",
        "abstain.json",
        "verifier.json",
        "summarization.json",
    ]
    for filename in required_files:
        filepath = raw_dir / filename
        assert filepath.exists(), f"Missing benchmark file for Tab 4: {filepath}"
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            assert "metrics" in data, f"Metrics missing in {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
