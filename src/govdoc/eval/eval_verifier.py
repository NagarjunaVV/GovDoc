"""Verifier evaluation on Benchmark B (E6).

For each answerable item in benchmark_B.jsonl:
  - Use gold_answer as the generated answer proxy (since we don't have a real PDF corpus)
  - Also test with a synthetic incorrect answer to validate flag rate
  - Run verify_grounded_answer(answer, evidence_chunks)
  - Record pass/flag rate by language

Saves results/raw/verifier.json per Rule 6 schema.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from govdoc.config import resolve_path
from govdoc.generate.verifier import verify_grounded_answer
from govdoc.logging_utils import append_log, save_experiment_record

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def make_evidence_from_answer(answer: str) -> List[Dict[str, Any]]:
    """Synthesize a single evidence chunk containing the answer text."""
    return [{"text": answer, "chunk_id": "gold_evidence", "page": 1}]


def make_corrupted_answer(answer: str) -> str:
    """Create a verifier-unfriendly answer by appending an invented number/entity.

    Used to validate that the verifier correctly flags unsupported claims.
    """
    return answer.strip() + " Additionally, the total amount was Rs. 99,99,999 Crores as of 2099."


def evaluate_verifier(items: List[Dict]) -> Dict[str, Any]:
    """Run verifier on answerable benchmark B items.

    Two conditions per item:
      1. Grounded: gold_answer is both the 'generated' answer and evidence → should PASS
      2. Corrupted: tampered answer against gold evidence → should FLAG
    """
    lang_pass: Dict[str, int] = defaultdict(int)
    lang_total: Dict[str, int] = defaultdict(int)
    lang_flag_correct: Dict[str, int] = defaultdict(int)  # correctly flagged corrupted
    lang_flag_total: Dict[str, int] = defaultdict(int)

    flagged_examples: List[Dict] = []
    all_records: List[Dict] = []

    for q in items:
        lang = q.get("lang", "en")
        qid = q.get("qid", "")
        answer = q.get("gold_answer", "")
        if not answer:
            continue

        # Evidence derived from the gold answer itself
        evidence = make_evidence_from_answer(answer)

        # ── Condition 1: Grounded (should pass) ──────────────────────────────
        result_grounded = verify_grounded_answer(answer, evidence)
        lang_total[lang] += 1
        if result_grounded["pass"]:
            lang_pass[lang] += 1

        # ── Condition 2: Corrupted (should flag) ─────────────────────────────
        corrupted = make_corrupted_answer(answer)
        result_corrupted = verify_grounded_answer(corrupted, evidence)
        lang_flag_total[lang] += 1
        if not result_corrupted["pass"]:
            lang_flag_correct[lang] += 1
            if len(flagged_examples) < 5:
                flagged_examples.append({
                    "qid": qid,
                    "lang": lang,
                    "corrupted_answer": corrupted[:120] + "...",
                    "missing_tokens": result_corrupted["missing"],
                })

        all_records.append({
            "qid": qid,
            "lang": lang,
            "grounded_pass": result_grounded["pass"],
            "grounded_missing": result_grounded["missing"],
            "corrupted_flagged": not result_corrupted["pass"],
            "corrupted_missing": result_corrupted["missing"],
        })

    # Aggregate
    total_pass = sum(lang_pass.values())
    total_total = sum(lang_total.values())
    total_flag = sum(lang_flag_correct.values())
    total_flag_total = sum(lang_flag_total.values())

    pass_rate = total_pass / max(total_total, 1)
    flag_rate = total_flag / max(total_flag_total, 1)

    by_lang: Dict[str, Any] = {}
    for lang in sorted(set(lang_total)):
        by_lang[lang] = {
            "n": lang_total[lang],
            "grounded_pass_rate": round(lang_pass[lang] / max(lang_total[lang], 1), 4),
            "corrupted_flag_rate": round(lang_flag_correct[lang] / max(lang_flag_total[lang], 1), 4),
        }

    return {
        "n_items": total_total,
        "overall_grounded_pass_rate": round(pass_rate, 4),
        "overall_corrupted_flag_rate": round(flag_rate, 4),
        "by_lang": by_lang,
        "flagged_examples": flagged_examples,
        "all_records": all_records,
    }


def main() -> None:
    benchmark_B_path = resolve_path("data/eval/benchmark_B.jsonl")
    raw_dir = resolve_path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Benchmark B...")
    items = [json.loads(l) for l in benchmark_B_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [i for i in items if i.get("answerable", True)]
    print(f"  Total: {len(items)}  Answerable: {len(answerable)}")

    print("\nRunning verifier evaluation (E6)...")
    results = evaluate_verifier(answerable)

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VERIFIER EVALUATION RESULTS (Benchmark B)")
    print("=" * 65)
    print(f"Total items: {results['n_items']}")
    print(f"Grounded pass rate (should ≈ 1.0): {results['overall_grounded_pass_rate']:.4f}")
    print(f"Corrupted flag rate (should ≈ 1.0): {results['overall_corrupted_flag_rate']:.4f}")
    print("\nBy Language:")
    for lang, m in results["by_lang"].items():
        print(
            f"  {lang:4s}: n={m['n']:3d}  "
            f"Grounded Pass={m['grounded_pass_rate']:.4f}  "
            f"Corrupted Flag={m['corrupted_flag_rate']:.4f}"
        )
    if results["flagged_examples"]:
        print("\nFlagged examples (corrupted answers):")
        for ex in results["flagged_examples"]:
            print(f"  [{ex['qid']}] missing: {ex['missing_tokens']}")
    print("=" * 65)

    # ── Save verifier.json per Rule 6 ────────────────────────────────────────
    metrics = {
        "n_items": results["n_items"],
        "overall_grounded_pass_rate": results["overall_grounded_pass_rate"],
        "overall_corrupted_flag_rate": results["overall_corrupted_flag_rate"],
        "by_lang": results["by_lang"],
        "flagged_examples": results["flagged_examples"],
    }
    save_experiment_record(
        experiment_id="verifier",
        metrics=metrics,
        n_samples=results["n_items"],
        seed=SEED,
        notes="E6: verifier pass/flag rates on Benchmark B (grounded=gold_answer vs evidence; corrupted=tampered answer)",
    )

    # Save detailed records
    records_path = raw_dir / "verifier_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for r in results["all_records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved verifier.json and verifier_records.jsonl → {raw_dir}")

    append_log(
        f"eval_verifier: n={results['n_items']}  "
        f"grounded_pass={results['overall_grounded_pass_rate']:.4f}  "
        f"corrupted_flag={results['overall_corrupted_flag_rate']:.4f}",
        module="M8:VERIFIER",
    )


if __name__ == "__main__":
    main()
