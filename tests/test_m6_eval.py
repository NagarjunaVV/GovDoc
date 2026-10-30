"""
Unit tests for Module M6: Evaluation Sets (Benchmark A & Benchmark B).
Validates schema compliance, group isolation (no leakage between index and unanswerable),
and phonetic transliteration.
"""

import json
from pathlib import Path
import pytest

from govdoc.data.build_benchmarks import native_to_roman


def test_native_to_roman_transliteration():
    """Verify Hindi and Kannada native-to-Roman transliteration produces valid Latin text."""
    hi_sample = "भारत सरकार का नया परिपत्र"
    kn_sample = "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಸುತ್ತೋಲೆ"

    hi_roman = native_to_roman(hi_sample, "hi")
    kn_roman = native_to_roman(kn_sample, "kn")

    assert hi_roman and isinstance(hi_roman, str)
    assert kn_roman and isinstance(kn_roman, str)

    # Must be pure ASCII / Latin script
    assert all(ord(c) < 128 for c in hi_roman)
    assert all(ord(c) < 128 for c in kn_roman)
    assert "bhaarata" in hi_roman.lower()
    assert "karnaataka" in kn_roman.lower()


def test_benchmark_A_structure_and_counts():
    """Verify Benchmark A index and query files exist and match specification."""
    index_file = Path("data/eval/benchmark_A_index.jsonl")
    queries_file = Path("data/eval/benchmark_A.jsonl")

    assert index_file.exists(), f"Index file {index_file} missing"
    assert queries_file.exists(), f"Queries file {queries_file} missing"

    # Verify index
    with open(index_file, "r", encoding="utf-8") as f:
        index_docs = [json.loads(line) for line in f if line.strip()]

    assert len(index_docs) == 1000, f"Expected 1000 indexed documents, got {len(index_docs)}"
    for doc in index_docs:
        assert "doc_id" in doc
        assert "text" in doc
        assert len(doc["text"]) > 20

    # Verify queries
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    assert len(queries) == 3000, f"Expected 3000 queries, got {len(queries)}"

    expected_fields = {
        "qid", "lang", "mode", "question", "gold_doc", "gold_answer",
        "answerable", "unanswerable_type"
    }

    modes = set()
    langs = set()
    answerable_count = 0
    unanswerable_count = 0

    for q in queries:
        assert expected_fields.issubset(q.keys())
        modes.add(q["mode"])
        langs.add(q["lang"])
        if q["answerable"]:
            answerable_count += 1
        else:
            unanswerable_count += 1

    assert modes == {"native", "romanised"}
    assert langs == {"en", "hi", "kn"}
    assert answerable_count == 1500
    assert unanswerable_count == 1500


def test_no_leakage_between_indexed_and_unanswerable():
    """Verify Rule 5: Zero leakage between indexed corpus and unanswerable queries."""
    index_file = Path("data/eval/benchmark_A_index.jsonl")
    queries_file = Path("data/eval/benchmark_A.jsonl")

    with open(index_file, "r", encoding="utf-8") as f:
        indexed_doc_ids = {json.loads(line)["doc_id"] for line in f if line.strip()}

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    answerable_docs = set()
    unanswerable_docs = set()

    for q in queries:
        if q["answerable"]:
            answerable_docs.add(q["gold_doc"])
        else:
            unanswerable_docs.add(q["gold_doc"])

    # 1. All answerable target docs must be within the indexed corpus
    assert answerable_docs.issubset(indexed_doc_ids), "Answerable queries reference docs outside index"

    # 2. All unanswerable target docs must NOT exist in the indexed corpus
    leakage = unanswerable_docs.intersection(indexed_doc_ids)
    assert len(leakage) == 0, f"Leakage detected! {len(leakage)} unanswerable docs found in index: {leakage}"


def test_benchmark_B_schema_and_integrity():
    """Verify Benchmark B items match schema from datasets.md section 3."""
    benchmark_b_file = Path("data/eval/benchmark_B.jsonl")
    assert benchmark_b_file.exists(), f"Benchmark B file {benchmark_b_file} missing"

    with open(benchmark_b_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    assert len(items) == 70, f"Expected 70 items in Benchmark B, got {len(items)}"

    required_fields = {
        "qid", "lang", "mode", "question", "gold_doc", "gold_chunk_ids",
        "gold_answer", "answerable", "unanswerable_type"
    }

    for item in items:
        assert required_fields.issubset(item.keys())
        assert isinstance(item["gold_chunk_ids"], list)
        assert item["lang"] in {"en", "hi", "kn"}
        assert item["mode"] in {"native", "romanised"}
        if item["answerable"]:
            assert item["unanswerable_type"] is None
            assert len(item["gold_chunk_ids"]) > 0
        else:
            assert item["unanswerable_type"] in {"near_miss", "off_topic"}
