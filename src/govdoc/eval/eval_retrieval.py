"""Retrieval evaluation: Recall@K, MRR, transliteration ablation (E2, E3).

Uses in-memory cosine similarity over the benchmark_A_index docs.
Optimized with batch view embedding and index vector caching.
"""

import json
import random
import sys
import time
from collections import defaultdict
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Metric helpers ─────────────────────────────────────────────────────────────


def recall_at_k(ranked_doc_ids: List[str], gold_doc: str, k: int) -> float:
    """Return 1.0 if gold_doc is in the top-k retrieved docs, else 0.0."""
    return float(gold_doc in ranked_doc_ids[:k])


def reciprocal_rank(ranked_doc_ids: List[str], gold_doc: str) -> float:
    """Return 1/(rank) if gold_doc found, else 0.0."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == gold_doc:
            return 1.0 / rank
    return 0.0


def cosine_sim_matrix(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities (dot product of L2-normalised vectors)."""
    q = query_emb.reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return (doc_embs @ q.T).squeeze(-1)


# ── Benchmark & Index loading ──────────────────────────────────────────────────


def load_benchmark_A(
    benchmark_path: Path,
    index_path: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Load queries and index documents."""
    queries = [json.loads(l) for l in benchmark_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return queries, docs


def get_or_build_doc_matrix(
    docs: List[Dict],
    embedder: BGEM3Embedder,
    cache_path: Optional[Path] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """Load cached index doc matrix or embed and save cache."""
    doc_ids = [d["doc_id"] for d in docs]
    if cache_path is None:
        cache_path = resolve_path("data/interim/benchmark_A_index_embs.npy")

    if cache_path.exists():
        print(f"Loading cached doc matrix from {cache_path}...")
        matrix = np.load(cache_path)
        if matrix.shape == (len(docs), 1024):
            print(f"  Loaded cached shape: {matrix.shape}")
            return matrix, doc_ids

    print(f"Embedding {len(docs)} index documents (batch_size={batch_size})...")
    t0 = time.time()
    texts = [d["text"] for d in docs]
    matrix = embedder.embed_texts(texts, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  shape={matrix.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    print(f"  Saved cache → {cache_path}")
    return matrix, doc_ids


# ── Batched retrieval evaluation ───────────────────────────────────────────────


def evaluate_retrieval_batched(
    queries: List[Dict],
    doc_matrix: np.ndarray,
    doc_ids: List[str],
    embedder: BGEM3Embedder,
    normalizer: QueryNormalizer,
    use_translit_view: bool = True,
    top_k_list: List[int] = (1, 3, 5, 10),
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run batched retrieval evaluation for high efficiency."""
    answerable = [q for q in queries if q.get("answerable", False)]
    cfg = load_config()
    translit_enabled = cfg.get("retrieval", {}).get("transliteration_enabled", True)

    total = len(answerable)
    print(f"\nPreparing views for {total} answerable queries (translit_view={use_translit_view})...")

    # 1. Collect views and map each query to its view strings
    query_view_lists: List[List[str]] = []
    unique_views_map: Dict[str, int] = {}
    all_unique_views: List[str] = []

    for q in answerable:
        lang = q.get("lang", "en")
        mode = q.get("mode", "native")
        question = q.get("question", "")

        query_mode = "auto"
        if mode == "romanised":
            query_mode = "hinglish" if lang == "hi" else "kanglish"

        if use_translit_view and translit_enabled:
            norm = normalizer.normalize(question, mode=query_mode)
            views = norm["views"]
        else:
            norm = normalizer.normalize(question, mode=query_mode)
            views = norm["views"][:1]  # raw view only

        query_view_lists.append(views)
        for v in views:
            if v not in unique_views_map:
                unique_views_map[v] = len(all_unique_views)
                all_unique_views.append(v)

    print(f"  Total unique views across all queries: {len(all_unique_views)}")
    print(f"  Embedding unique views in batches of {batch_size}...")
    t0 = time.time()
    view_matrix = embedder.embed_texts(all_unique_views, batch_size=batch_size)
    print(f"  Views embedded in {time.time() - t0:.1f}s. Shape={view_matrix.shape}")

    # 2. Compute full similarity matrix: doc_matrix (N_docs, 1024) @ view_matrix.T (1024, N_views)
    print("  Computing cosine similarity matrix...")
    t0 = time.time()
    sim_matrix = doc_matrix @ view_matrix.T
    print(f"  Matrix multiplication done in {time.time() - t0:.2f}s.")

    # 3. Score each query: take max score over views for each doc, then sort top-k
    per_lang_recall: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_mode_recall: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_rr: List[float] = []
    per_lang_rr: Dict[str, List[float]] = defaultdict(list)
    per_mode_rr: Dict[str, List[float]] = defaultdict(list)
    max_k = max(top_k_list)

    for idx, q in enumerate(answerable):
        lang = q.get("lang", "en")
        mode = q.get("mode", "native")
        gold_doc = q.get("gold_doc", "")
        views = query_view_lists[idx]

        view_indices = [unique_views_map[v] for v in views]
        if len(view_indices) == 1:
            doc_scores = sim_matrix[:, view_indices[0]]
        else:
            doc_scores = np.max(sim_matrix[:, view_indices], axis=1)

        ranked_indices = np.argsort(doc_scores)[::-1][:max_k]
        ranked_docs = [doc_ids[i] for i in ranked_indices]

        rr = reciprocal_rank(ranked_docs, gold_doc)
        all_rr.append(rr)
        per_lang_rr[lang].append(rr)
        per_mode_rr[mode].append(rr)

        for k in top_k_list:
            r = recall_at_k(ranked_docs, gold_doc, k)
            per_lang_recall[lang][k].append(r)
            per_mode_recall[mode][k].append(r)

    def mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    overall: Dict[str, Any] = {
        "MRR": round(mean(all_rr), 4),
        "n": total,
    }
    for k in top_k_list:
        all_k = [v for lang_dict in per_lang_recall.values() for v in lang_dict[k]]
        overall[f"Recall@{k}"] = round(mean(all_k), 4)

    by_lang: Dict[str, Any] = {}
    for lang, k_dict in per_lang_recall.items():
        by_lang[lang] = {f"Recall@{k}": round(mean(v), 4) for k, v in k_dict.items()}
        by_lang[lang]["MRR"] = round(mean(per_lang_rr[lang]), 4)
        by_lang[lang]["n"] = len(per_lang_rr[lang])

    by_mode: Dict[str, Any] = {}
    for mode, k_dict in per_mode_recall.items():
        by_mode[mode] = {f"Recall@{k}": round(mean(v), 4) for k, v in k_dict.items()}
        by_mode[mode]["MRR"] = round(mean(per_mode_rr[mode]), 4)
        by_mode[mode]["n"] = len(per_mode_rr[mode])

    return {
        "overall": overall,
        "by_lang": by_lang,
        "by_mode": by_mode,
    }


def print_table(results_raw: Dict, results_translit: Dict) -> None:
    """Print formatted retrieval results table."""
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)

    ks = [1, 3, 5, 10]

    for cond_label, res in [("Raw view only (E2)", results_raw), ("Raw+Translit view (E3)", results_translit)]:
        print(f"\n{'─'*60}")
        print(f"Condition: {cond_label}")
        print(f"{'─'*60}")
        ov = res["overall"]
        print(f"  Overall: n={ov['n']}  MRR={ov['MRR']:.4f}  " + "  ".join(f"R@{k}={ov.get(f'Recall@{k}',0):.4f}" for k in ks))

        print("\n  By Language:")
        for lang, m in sorted(res["by_lang"].items()):
            print(f"    {lang:4s}: n={m['n']:4d}  MRR={m['MRR']:.4f}  " + "  ".join(f"R@{k}={m.get(f'Recall@{k}',0):.4f}" for k in ks))

        print("\n  By Mode (native vs romanised):")
        for mode, m in sorted(res["by_mode"].items()):
            print(f"    {mode:10s}: n={m['n']:4d}  MRR={m['MRR']:.4f}  " + "  ".join(f"R@{k}={m.get(f'Recall@{k}',0):.4f}" for k in ks))

    print("\n" + "=" * 70)


def main() -> None:
    cfg = load_config()
    top_k_list = [1, 3, 5, 10]
    batch_size = int(cfg.get("ingest", {}).get("embed_batch_size", 32))

    benchmark_path = resolve_path("data/eval/benchmark_A.jsonl")
    index_path = resolve_path("data/eval/benchmark_A_index.jsonl")
    out_path = resolve_path("results/raw/retrieval_A.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark data...")
    queries, docs = load_benchmark_A(benchmark_path, index_path)
    print(f"  Queries: {len(queries)}  Index docs: {len(docs)}")

    embedder = BGEM3Embedder(device="cpu")
    normalizer = QueryNormalizer()

    # Build or load cached doc matrix
    doc_matrix, doc_ids = get_or_build_doc_matrix(docs, embedder, batch_size=batch_size)

    # E2: Raw view only
    print("\n[E2] Evaluating raw-view-only retrieval...")
    t0 = time.time()
    results_raw = evaluate_retrieval_batched(
        queries, doc_matrix, doc_ids, embedder, normalizer,
        use_translit_view=False, top_k_list=top_k_list, batch_size=batch_size,
    )
    e2_time = time.time() - t0
    print(f"  E2 done in {e2_time:.1f}s")

    # E3: Raw + transliteration view
    print("\n[E3] Evaluating raw+transliteration retrieval...")
    t0 = time.time()
    results_translit = evaluate_retrieval_batched(
        queries, doc_matrix, doc_ids, embedder, normalizer,
        use_translit_view=True, top_k_list=top_k_list, batch_size=batch_size,
    )
    e3_time = time.time() - t0
    print(f"  E3 done in {e3_time:.1f}s")

    print_table(results_raw, results_translit)

    metrics = {
        "E2_raw_only": results_raw,
        "E3_raw_plus_translit": results_translit,
        "transliteration_delta": {
            "overall_MRR": round(results_translit["overall"]["MRR"] - results_raw["overall"]["MRR"], 4),
            **{
                f"overall_Recall@{k}": round(
                    results_translit["overall"].get(f"Recall@{k}", 0)
                    - results_raw["overall"].get(f"Recall@{k}", 0),
                    4,
                )
                for k in top_k_list
            },
        },
    }

    n_total = results_raw["overall"]["n"]
    save_experiment_record(
        experiment_id="retrieval_A",
        metrics=metrics,
        n_samples=n_total,
        seed=SEED,
        notes="E2/E3: article-level batched cosine retrieval on Benchmark A (answerable queries only)",
    )
    print(f"\nSaved → {out_path}")
    append_log(
        f"eval_retrieval: E2 MRR={results_raw['overall']['MRR']:.4f}  "
        f"E3 MRR={results_translit['overall']['MRR']:.4f}  n={n_total}",
        module="M8:RETRIEVAL",
    )


if __name__ == "__main__":
    main()
