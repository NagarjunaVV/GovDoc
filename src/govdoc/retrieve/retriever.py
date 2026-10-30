"""Multi-view dense retriever with ChromaDB cosine scoring and trust gate evaluation."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from govdoc.config import load_config
from govdoc.ingest.embed_index import (
    distance_to_cosine,
    BGEM3Embedder,
    get_chroma_collection,
)
from govdoc.retrieve.query_normalizer import QueryNormalizer
from govdoc.retrieve.trust_gate import TrustDecision, evaluate_trust_gate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class MultilingualRetriever:
    """Multi-view dense retriever supporting cross-lingual and Romanised Indic queries."""

    def __init__(
        self,
        collection_name: str = "govdoc_corpus",
        device: str = "cpu",
        top_k: Optional[int] = None,
        embedder: Optional[BGEM3Embedder] = None,
    ):
        cfg = load_config()
        r_cfg = cfg.get("retrieval", {})

        self.collection_name = collection_name
        self.collection = get_chroma_collection(collection_name)
        self.top_k = top_k or int(r_cfg.get("top_k", 10))

        # Query embedding on CPU by default per M4 requirements
        self.embedder = embedder or BGEM3Embedder(device=device)
        self.normalizer = QueryNormalizer()

    def retrieve(
        self,
        query: str,
        mode: str = "auto",
        top_k: Optional[int] = None,
        tau_hi: Optional[float] = None,
        tau_lo: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute multi-view retrieval:
        1. Produce query views (original + transliterated if Hinglish/Kanglish)
        2. Embed all views with BGE-M3
        3. Query ChromaDB with cosine metric space
        4. Fuse scores by taking maximum cosine score over views per chunk
        5. Evaluate trust gate (ANSWER, EVIDENCE_ONLY, ABSTAIN)

        Returns:
            Dict containing query metadata, trust decision, and ranked chunks.
        """
        k = top_k or self.top_k
        norm_result = self.normalizer.normalize(query, mode=mode)
        views = norm_result["views"]

        # Embed all views
        view_embeddings = self.embedder.embed_texts(views)

        # Query ChromaDB collection for each view embedding
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for view_idx, view_text in enumerate(views):
            emb = view_embeddings[view_idx].tolist()
            res = self.collection.query(
                query_embeddings=[emb],
                n_results=min(k, max(self.collection.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )

            ids = res["ids"][0] if res["ids"] else []
            docs = res["documents"][0] if res["documents"] else []
            metas = res["metadatas"][0] if res["metadatas"] else []
            distances = res["distances"][0] if res["distances"] else []

            for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                # Non-Negotiable Rule 3: distance = 1 - cosine
                cosine_sim = distance_to_cosine(float(dist))

                if chunk_id not in chunk_map or cosine_sim > chunk_map[chunk_id]["score"]:
                    chunk_map[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": doc_text,
                        "doc_id": meta.get("doc_id", "unknown"),
                        "page": int(meta.get("page", 1)),
                        "lang_guess": meta.get("lang_guess", "en"),
                        "token_count": int(meta.get("token_count", 0)),
                        "score": round(cosine_sim, 4),
                        "distance": round(float(dist), 4),
                        "best_view": view_text,
                    }

        # Rank all unique chunks by maximum score descending
        ranked_chunks = sorted(
            chunk_map.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:k]

        max_score = ranked_chunks[0]["score"] if ranked_chunks else 0.0
        decision, gate_details = evaluate_trust_gate(max_score, tau_hi=tau_hi, tau_lo=tau_lo)

        return {
            "query": query,
            "mode": norm_result["mode"],
            "detected_script": norm_result["detected_script"],
            "views": views,
            "decision": decision.value,
            "gate_details": gate_details,
            "top_score": max_score,
            "chunks": ranked_chunks,
        }
