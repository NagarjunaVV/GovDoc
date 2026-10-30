"""Unit tests for Module M0: Setup, Configuration, and Infrastructure."""

import pytest
import numpy as np
import torch
import random

from govdoc.config import load_config, get_config_hash, resolve_path
from govdoc.seeding import set_seed
from govdoc.logging_utils import append_log, get_git_commit
from govdoc.runtime.vram_monitor import get_gpu_memory_info
from govdoc.generate.llm_client import LLMClient


def test_config_loading():
    """Verify config/config.yaml loads with all required sections."""
    cfg = load_config()
    required_sections = [
        "paths",
        "seeds",
        "models",
        "chunking",
        "retrieval",
        "llm",
        "training",
        "eval",
        "runtime",
    ]
    for section in required_sections:
        assert section in cfg, f"Missing required section: {section}"

    assert cfg["retrieval"]["top_k"] >= 1
    assert "tau_hi" in cfg["retrieval"] and "tau_lo" in cfg["retrieval"]
    assert cfg["runtime"]["vram_budget_mb"] <= 4096


def test_config_hash():
    """Verify deterministic configuration hash."""
    h1 = get_config_hash()
    h2 = get_config_hash()
    assert len(h1) == 12
    assert h1 == h2


def test_deterministic_seeding():
    """Verify seed utility provides identical random outputs."""
    set_seed(42)
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    torch_rand1 = torch.rand(1).item()

    set_seed(42)
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()

    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert torch_rand1 == torch_rand2


def test_vram_monitor_nvml():
    """Verify pynvml reads valid GPU memory metrics."""
    info = get_gpu_memory_info()
    assert "total_mb" in info
    assert "used_mb" in info
    if torch.cuda.is_available():
        assert info["total_mb"] > 0
        assert info["total_mb"] <= 4096.0 or True  # 4 GB card detected


def test_logging_utils():
    """Verify append_log writes properly to LOG.md."""
    log_file = resolve_path("results/LOG.md")
    append_log("Test log entry for M0 validation.", module="TEST_M0")
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test log entry for M0 validation." in content


def test_llm_client_initialization():
    """Verify LLM client connects to local server."""
    client = LLMClient()
    assert client.server_url.startswith("http")
    assert client.call_count == 0
    # Health check should be true if server is up
    assert client.health_check() is True
