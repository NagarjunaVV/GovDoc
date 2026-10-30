"""Abstention evaluation: score distributions, tau sweep, AUROC, calibration (E4, E5).

Saves:
  results/raw/scores_answerable.jsonl
  results/raw/scores_nearmiss.jsonl
  results/raw/scores_ood.jsonl        (benchmark_B off-topic items)
  results/raw/abstain.json            (full sweep + AUROC + calibrated tau)
  results/raw/scores_val.jsonl
  results/raw/scores_test.jsonl
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from govdoc.config import load_config, resolve_path
from govdoc.ingest.embed_index import BGEM3Embedder
from govdoc.logging_utils import append_log, save_experiment_record
from govdoc.retrieve.query_normalizer import QueryNormalizer
from govdoc.eval.eval_retrieval import get_or_build_doc_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Tau-sweep metrics ──────────────────────────────────────────────────────────


def compute_sweep_metrics(
    answerable_scores: List[float],
    unanswerable_scores: List[float],
    tau_values: List[float],
) -> List[Dict[str, Any]]:
    """Compute FAR, answerable recall, F1, balanced accuracy, precision across tau sweep."""
    n_ans = len(answerable_scores)
    n_unans = len(unanswerable_scores)
    rows = []
    for tau in tau_values:
        TP = sum(1 for s in answerable_scores if s >= tau)
        FN = n_ans - TP
        FP = sum(1 for s in unanswerable_scores if s >= tau)
        TN = n_unans - FP

        recall_ans = TP / max(n_ans, 1)
        far = FN / max(n_ans, 1)         # false abstain rate
        false_accept = FP / max(n_unans, 1)
        precision = TP / max(TP + FP, 1)
        f1 = 2 * precision * recall_ans / max(precision + recall_ans, 1e-9)
        balanced_acc = (recall_ans + (TN / max(n_unans, 1))) / 2.0

        rows.append({
            "tau": round(float(tau), 3),
            "TP": TP,
            "FN": FN,
            "FP": FP,
            "TN": TN,
            "recall_answerable": round(recall_ans, 4),
            "far": round(far, 4),               # false abstain rate
            "false_accept_rate": round(false_accept, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        })
    return rows


def compute_auroc(
    answerable_scores: List[float],
    unanswerable_scores: List[float],
) -> float:
    """Compute AUROC for separating answerable from unanswerable queries."""
    try:
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(answerable_scores) + [0] * len(unanswerable_scores)
        y_score = answerable_scores + unanswerable_scores
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        all_scores = answerable_scores + unanswerable_scores
        y_true = [1] * len(answerable_scores) + [0] * len(unanswerable_scores)
        sorted_idx = np.argsort(all_scores)[::-1]
        tprs, fprs = [0.0], [0.0]
        tp = fp = 0
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        for i in sorted_idx:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
            tprs.append(tp / max(n_pos, 1))
            fprs.append(fp / max(n_neg, 1))
        tprs.append(1.0)
        fprs.append(1.0)
        return float(np.trapz(tprs, fprs))


def calibrate_tau(
    sweep_rows: List[Dict[str, Any]],
    target_far: float = 0.05,
) -> float:
    """Calibrate tau on validation scores (highest tau satisfying FAR <= target_far)."""
    candidates = [r for r in sweep_rows if r["far"] <= target_far]
    if candidates:
        return max(candidates, key=lambda r: r["tau"])["tau"]
    else:
        return min(sweep_rows, key=lambda r: r["far"])["tau"]


# ── Batched query scoring ──────────────────────────────────────────────────────


def score_queries_batched(
    qlist: List[Dict],
    doc_matrix: np.ndarray,
    doc_ids: List[str],
    embedder: BGEM3Embedder,
    normalizer: QueryNormalizer,
    label: str,
    out_file: Path,
    batch_size: int = 32,
) -> List[float]:
    """Score a query list using batched embeddings and matrix multiplication."""
    n = len(qlist)
    print(f"\nScoring {n} {label} queries (batched)...")
    if n == 0:
        return []

    # 1. Normalise all queries and extract views
    query_view_lists: List[List[str]] = []
    unique_views_map: Dict[str, int] = {}
    all_unique_views: List[str] = []

    for q in qlist:
        question = q.get("question", "")
        lang = q.get("lang", "en")
        mode = q.get("mode", "native")
        qmode = "auto"
        if mode == "romanised":
            qmode = "hinglish" if lang == "hi" else "kanglish"

        norm = normalizer.normalize(question, mode=qmode)
        views = norm["views"]
        query_view_lists.append(views)
        for v in views:
            if v not in unique_views_map:
                unique_views_map[v] = len(all_unique_views)
                all_unique_views.append(v)

    # 2. Batch embed unique views
    print(f"  Embedding {len(all_unique_views)} unique views in batches of {batch_size}...")
    t0 = time.time()
    view_matrix = embedder.embed_texts(all_unique_views, batch_size=batch_size)
    print(f"  Views embedded in {time.time() - t0:.1f}s.")

    # 3. Similarity matrix
    sim_matrix = doc_matrix @ view_matrix.T  # (N_docs, N_views)

    # 4. Extract top-1 score and best doc for each query
    scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for idx, q in enumerate(qlist):
        views = query_view_lists[idx]
        view_indices = [unique_views_map[v] for v in views]
        if len(view_indices) == 1:
            doc_scores = sim_matrix[:, view_indices[0]]
        else:
            doc_scores = np.max(sim_matrix[:, view_indices], axis=1)

        best_idx = int(np.argmax(doc_scores))
        top1 = float(doc_scores[best_idx])
        best_doc = doc_ids[best_idx]

        scores.append(top1)
        records.append({
            "qid": q.get("qid", f"q{idx}"),
            "lang": q.get("lang", "en"),
            "mode": q.get("mode", "native"),
            "top1_score": round(top1, 6),
            "best_doc": best_doc,
            "answerable": q.get("answerable", True),
            "unanswerable_type": q.get("unanswerable_type"),
        })

    print(f"  Mean score={float(np.mean(scores)):.4f}  Std={float(np.std(scores)):.4f}")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved → {out_file}")
    return scores


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = load_config()
    batch_size = int(cfg.get("ingest", {}).get("embed_batch_size", 32))

    benchmark_A_path = resolve_path("data/eval/benchmark_A.jsonl")
    benchmark_B_path = resolve_path("data/eval/benchmark_B.jsonl")
    index_path = resolve_path("data/eval/benchmark_A_index.jsonl")
    raw_dir = resolve_path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Benchmark A queries...")
    queries_A = [json.loads(l) for l in benchmark_A_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable_A = [q for q in queries_A if q.get("answerable", True)]
    nearmiss_A = [q for q in queries_A if not q.get("answerable", True) and q.get("unanswerable_type") == "near_miss"]

    print("Loading Benchmark B for OOD queries...")
    queries_B = [json.loads(l) for l in benchmark_B_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ood_B = [q for q in queries_B if not q.get("answerable", True)]

    print(f"  Answerable: {len(answerable_A)}  Near-miss: {len(nearmiss_A)}  OOD: {len(ood_B)}")

    print("\nLoading index documents...")
    docs = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  Index docs: {len(docs)}")

    embedder = BGEM3Embedder(device="cpu")
    normalizer = QueryNormalizer()

    # Load or build doc matrix (cached)
    doc_matrix, doc_ids_list = get_or_build_doc_matrix(docs, embedder, batch_size=batch_size)

    ans_scores_path = raw_dir / "scores_answerable.jsonl"
    nearmiss_scores_path = raw_dir / "scores_nearmiss.jsonl"
    ood_scores_path = raw_dir / "scores_ood.jsonl"

    ans_scores = score_queries_batched(
        answerable_A, doc_matrix, doc_ids_list, embedder, normalizer,
        "answerable", ans_scores_path, batch_size=batch_size,
    )
    nearmiss_scores = score_queries_batched(
        nearmiss_A, doc_matrix, doc_ids_list, embedder, normalizer,
        "near-miss", nearmiss_scores_path, batch_size=batch_size,
    )

    if ood_B:
        ood_scores = score_queries_batched(
            ood_B, doc_matrix, doc_ids_list, embedder, normalizer,
            "OOD (off-topic)", ood_scores_path, batch_size=batch_size,
        )
    else:
        ood_scores = []

    # Val/Test split for calibration (80% val, 20% test)
    n_ans = len(ans_scores)
    n_val = int(n_ans * 0.8)
    val_ans_scores = ans_scores[:n_val]
    test_ans_scores = ans_scores[n_val:]

    n_near = len(nearmiss_scores)
    n_val_near = int(n_near * 0.8)
    val_nearmiss_scores = nearmiss_scores[:n_val_near]
    test_nearmiss_scores = nearmiss_scores[n_val_near:]

    print("\nRunning tau sweep on VALIDATION queries...")
    tau_values = [round(0.20 + 0.05 * i, 2) for i in range(15)]  # 0.20 … 0.90
    sweep_rows = compute_sweep_metrics(val_ans_scores, val_nearmiss_scores, tau_values)

    print("Computing AUROC (answerable vs near-miss)...")
    auroc = compute_auroc(ans_scores, nearmiss_scores)
    print(f"  AUROC = {auroc:.4f}")

    print("Calibrating tau (FAR <= 5% on validation)...")
    calibrated_tau = calibrate_tau(sweep_rows, target_far=0.05)
    print(f"  Calibrated tau = {calibrated_tau:.3f}")

    val_row = min(sweep_rows, key=lambda r: abs(r["tau"] - calibrated_tau))

    print("\nApplying calibrated tau to TEST set...")
    test_sweep = compute_sweep_metrics(test_ans_scores, test_nearmiss_scores, [calibrated_tau])
    test_row = test_sweep[0]

    # Save split scores
    def save_split_scores(scores: List[float], queries: List[Dict], path: Path, split: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for q, s in zip(queries, scores):
                r = {
                    "qid": q.get("qid", ""),
                    "lang": q.get("lang", "en"),
                    "split": split,
                    "top1_score": round(s, 6),
                    "answerable": q.get("answerable", True),
                    "unanswerable_type": q.get("unanswerable_type"),
                    "abstain": s < calibrated_tau,
                }
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_split_scores(val_ans_scores, answerable_A[:n_val], raw_dir / "scores_val.jsonl", "val")
    save_split_scores(test_ans_scores, answerable_A[n_val:], raw_dir / "scores_test.jsonl", "test")

    print("\n" + "=" * 70)
    print("ABSTENTION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nAUROC (answerable vs near-miss): {auroc:.4f}")
    print(f"Calibrated tau (FAR <= 5% on val): {calibrated_tau:.3f}")
    print(f"\nVal metrics at tau={calibrated_tau}:")
    print(f"  Answerable recall = {val_row['recall_answerable']:.4f}  FAR = {val_row['far']:.4f}")
    print(f"  False-accept rate = {val_row['false_accept_rate']:.4f}")
    print(f"  Precision = {val_row['precision']:.4f}  F1 = {val_row['f1']:.4f}")
    print(f"  Balanced accuracy = {val_row['balanced_accuracy']:.4f}")
    print(f"\nTest metrics at tau={calibrated_tau}:")
    print(f"  Answerable recall = {test_row['recall_answerable']:.4f}  FAR = {test_row['far']:.4f}")
    print(f"  False-accept rate = {test_row['false_accept_rate']:.4f}")
    print(f"  Precision = {test_row['precision']:.4f}  F1 = {test_row['f1']:.4f}")
    print(f"  Balanced accuracy = {test_row['balanced_accuracy']:.4f}")

    print("\nTau sweep table (validation):")
    print(f"{'tau':>6}  {'RecallAns':>9}  {'FAR':>5}  {'FAccR':>6}  {'F1':>6}  {'BalAcc':>7}")
    for row in sweep_rows:
        marker = " ← calibrated" if abs(row["tau"] - calibrated_tau) < 0.001 else ""
        print(
            f"{row['tau']:>6.2f}  {row['recall_answerable']:>9.4f}  "
            f"{row['far']:>5.4f}  {row['false_accept_rate']:>6.4f}  "
            f"{row['f1']:>6.4f}  {row['balanced_accuracy']:>7.4f}{marker}"
        )
    print("=" * 70)

    metrics = {
        "auroc": round(auroc, 4),
        "calibrated_tau": calibrated_tau,
        "n_answerable": len(ans_scores),
        "n_nearmiss": len(nearmiss_scores),
        "n_ood": len(ood_scores),
        "val_metrics_at_tau": {
            "recall_answerable": val_row["recall_answerable"],
            "far": val_row["far"],
            "false_accept_rate": val_row["false_accept_rate"],
            "precision": val_row["precision"],
            "f1": val_row["f1"],
            "balanced_accuracy": val_row["balanced_accuracy"],
        },
        "test_metrics_at_tau": {
            "recall_answerable": test_row["recall_answerable"],
            "far": test_row["far"],
            "false_accept_rate": test_row["false_accept_rate"],
            "precision": test_row["precision"],
            "f1": test_row["f1"],
            "balanced_accuracy": test_row["balanced_accuracy"],
        },
        "tau_sweep": sweep_rows,
        "score_stats": {
            "answerable_mean": round(float(np.mean(ans_scores)), 4),
            "answerable_std": round(float(np.std(ans_scores)), 4),
            "nearmiss_mean": round(float(np.mean(nearmiss_scores)), 4),
            "nearmiss_std": round(float(np.std(nearmiss_scores)), 4),
            "ood_mean": round(float(np.mean(ood_scores)), 4) if ood_scores else None,
            "ood_std": round(float(np.std(ood_scores)), 4) if ood_scores else None,
        },
    }

    n_total = len(ans_scores) + len(nearmiss_scores) + len(ood_scores)
    save_experiment_record(
        experiment_id="abstain",
        metrics=metrics,
        n_samples=n_total,
        seed=SEED,
        notes="E4/E5: tau sweep, AUROC, calibrated tau on validation; evaluated on test",
    )
    print(f"\nSaved → {raw_dir / 'abstain.json'}")
    append_log(
        f"eval_abstain: AUROC={auroc:.4f}  tau={calibrated_tau:.3f}  "
        f"FAR={val_row['far']:.4f}  RecallAns={val_row['recall_answerable']:.4f}",
        module="M8:ABSTAIN",
    )


if __name__ == "__main__":
    main()
