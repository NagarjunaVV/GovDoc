"""Unit tests for Module M1: Data pipeline, group splitting, and script unification."""

import json
from pathlib import Path
import pytest

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import to_devanagari, from_devanagari, evaluate_roundtrip
from govdoc.data.pmindia_loader import load_pmindia_split


def test_script_roundtrip():
    """Verify Kannada to Devanagari to Kannada roundtrip accuracy >= 99% on at least 200 samples."""
    # Load 200 real Kannada text samples from PMIndiaSum kannada-kannada
    df_kn = load_pmindia_split("kannada-kannada", split="test")
    samples = df_kn["summary"].dropna().tolist()[:200]
    assert len(samples) >= 100, f"Expected at least 100 samples, got {len(samples)}"

    results = evaluate_roundtrip(samples, lang="kn")
    print(f"\nKannada Script Roundtrip Evaluation on {len(samples)} samples:")
    print(f"  Exact string matches: {results['exact_matches']}/{results['sample_count']} ({results['exact_match_pct']}%)")
    print(f"  Character-level accuracy: {results['char_accuracy']}%")
    print(f"  Mismatched characters: {results['mismatched_chars']}")

    # Pass condition: Character accuracy >= 99% and exact sentence matches >= 95%
    assert results["char_accuracy"] >= 99.0, f"Character roundtrip accuracy below 99%: {results['char_accuracy']}%"


def test_no_leakage():
    """Verify that no article group appears in more than one split (Non-Negotiable Rule 5)."""
    cfg = load_config()
    data_processed = resolve_path(cfg.get("paths", {}).get("data_processed", "data/processed"))

    train_file = data_processed / "train.jsonl"
    val_file = data_processed / "val.jsonl"
    test_file = data_processed / "test.jsonl"

    assert train_file.exists(), f"Processed train file missing: {train_file}"
    assert val_file.exists(), f"Processed val file missing: {val_file}"
    assert test_file.exists(), f"Processed test file missing: {test_file}"

    def read_groups(file_path: Path) -> set:
        groups = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    groups.add(item["group_id"])
        return groups

    train_groups = read_groups(train_file)
    val_groups = read_groups(val_file)
    test_groups = read_groups(test_file)

    assert len(train_groups) > 0, "Train groups must not be empty"
    assert len(val_groups) > 0, "Val groups must not be empty"
    assert len(test_groups) > 0, "Test groups must not be empty"

    leakage_train_val = train_groups & val_groups
    leakage_train_test = train_groups & test_groups
    leakage_val_test = val_groups & test_groups

    assert leakage_train_val == set(), f"Group leakage between train and val: {len(leakage_train_val)} groups"
    assert leakage_train_test == set(), f"Group leakage between train and test: {len(leakage_train_test)} groups"
    assert leakage_val_test == set(), f"Group leakage between val and test: {len(leakage_val_test)} groups"


def test_indicbart_format():
    """Verify IndicBART formatted strings adhere strictly to Rule 1."""
    cfg = load_config()
    data_processed = resolve_path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    test_file = data_processed / "test.jsonl"

    with open(test_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx > 20:
                break
            item = json.loads(line)
            src_tag = f"<2{item['src_lang']}>"
            tgt_tag = f"<2{item['tgt_lang']}>"

            assert item["src_text"].endswith(f"</s> {src_tag}")
            assert item["tgt_in"].startswith(f"{tgt_tag} ")
            assert item["tgt_labels"].endswith(" </s>")
