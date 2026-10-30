"""Dense batch embedding with BAAI/bge-m3 and ChromaDB vector indexing."""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.logging_utils import append_log
from govdoc.ingest.pdf_extract import extract_pages_from_pdf
from govdoc.ingest.quality_gate import filter_valid_pages
from govdoc.ingest.chunker import DocumentChunker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def distance_to_cosine(distance: float) -> float:
    """
    Convert ChromaDB cosine distance to cosine similarity.

    NON-NEGOTIABLE RULE 3:
    Chroma returns distance = 1 - cosine for cosine space. Convert in ONE function
    and unit-test it.
    Range: distance in [0.0, 2.0] maps to cosine in [1.0, -1.0].
    """
    return float(1.0 - distance)


class BGEM3Embedder:
    """Dense BGE-M3 embedding generator with L2 normalization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        server_url: Optional[str] = None,
    ):
        cfg = load_config()
        self.model_name = model_name or cfg.get("models", {}).get("embeddings", {}).get("name", "BAAI/bge-m3")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.server_url = server_url or cfg.get("llm", {}).get("server_url", "http://127.0.0.1:11434")

        # Check if local Ollama serves bge-m3
        self.use_ollama = False
        try:
            import requests
            resp = requests.get(f"{self.server_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                if any("bge-m3" in m for m in models):
                    self.use_ollama = True
                    print(f"BGEM3Embedder using local Ollama daemon ({self.server_url}) for 'bge-m3'.")
        except Exception:
            self.use_ollama = False

        if not self.use_ollama:
            print(f"Initializing PyTorch BGEM3Embedder on device: {self.device} ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            try:
                self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
            except Exception:
                cfg_model = AutoConfig.from_pretrained(self.model_name)
                self.model = AutoModel.from_config(cfg_model).to(dtype=dtype, device=self.device)

            self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Compute L2-normalized dense embeddings for a list of texts in batches.

        Returns:
            np.ndarray of shape (len(texts), 1024)
        """
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        if self.use_ollama:
            import requests
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                payload = {"model": "bge-m3", "input": batch_texts}
                resp = requests.post(f"{self.server_url}/api/embed", json=payload, timeout=60.0)
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama embedding error: {resp.status_code} - {resp.text}")
                batch_emb = np.array(resp.json()["embeddings"], dtype=np.float32)
                # L2-normalization
                norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                norm_emb = batch_emb / np.maximum(norms, 1e-12)
                all_embeddings.append(norm_emb)
            return np.vstack(all_embeddings)

        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                # Dense embedding from [CLS] representation (token 0)
                cls_rep = outputs.last_hidden_state[:, 0]
                # L2 normalization
                norm_rep = F.normalize(cls_rep, p=2, dim=1)
                all_embeddings.append(norm_rep.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to strictly comply with ChromaDB validation rules:
    - 3-512 characters
    - Allowed characters: [a-zA-Z0-9._-]
    - Must start and end with an alphanumeric character in [a-zA-Z0-9]
    """
    if not name:
        return "govdoc_default"

    # Replace whitespace, colons, slashes, and other invalid characters with underscore
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", str(name))
    # Collapse multiple consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Strip non-alphanumeric chars from start and end
    cleaned = cleaned.strip("._-")

    # If empty or shorter than 3 chars, prefix with doc_ or col_
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}".strip("._-")
        if len(cleaned) < 3:
            cleaned = "col_default"

    # Truncate to 512 chars if needed
    if len(cleaned) > 512:
        cleaned = cleaned[:512].rstrip("._-")

    return cleaned


def get_chroma_collection(
    collection_name: str,
    persist_dir: Optional[Path | str] = None,
) -> chromadb.Collection:
    """
    Get or create a persistent ChromaDB collection configured for cosine distance space.
    Automatically sanitizes the collection name to conform to ChromaDB constraints.
    """
    collection_name = sanitize_collection_name(collection_name)
    cfg = load_config()
    if persist_dir is None:
        raw_persist = cfg.get("paths", {}).get("vector_db", "data/vector_db")
        persist_dir = resolve_path(raw_persist)
    else:
        persist_dir = resolve_path(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Explicit cosine metric space per Non-Negotiable Rule 3
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    persist_dir: Optional[Path | str] = None,
    embedder: Optional[BGEM3Embedder] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Generate embeddings and store chunks with metadata into Chroma collection.

    Returns stats dict with chunk_count, elapsed_time_s, and time_per_1000_chunks.
    """
    if not chunks:
        print("No chunks provided to index.")
        return {"chunks": 0, "elapsed_s": 0.0, "time_per_1000": 0.0}

    embedder = embedder or BGEM3Embedder()
    collection = get_chroma_collection(collection_name, persist_dir=persist_dir)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "page": int(c["page"]),
            "lang_guess": c["lang_guess"],
            "token_count": int(c.get("token_count", 0)),
        }
        for c in chunks
    ]

    print(f"Embedding {len(chunks)} chunks with BGE-M3 (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = embedder.embed_texts(texts, batch_size=batch_size)
    embed_time = time.time() - t0

    # Add to Chroma collection in batches
    chroma_batch_size = 100
    for i in range(0, len(chunks), chroma_batch_size):
        end = min(i + chroma_batch_size, len(chunks))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    total_time = time.time() - t0
    time_per_1000 = (embed_time / max(len(chunks), 1)) * 1000.0

    print(f"Indexing complete:")
    print(f"  Chunks Indexed: {len(chunks)}")
    print(f"  Embedding Time: {embed_time:.2f}s ({time_per_1000:.2f}s per 1000 chunks)")
    print(f"  Collection Count: {collection.count()}")

    return {
        "chunks": len(chunks),
        "embed_time_s": round(embed_time, 2),
        "total_time_s": round(total_time, 2),
        "time_per_1000_chunks": round(time_per_1000, 2),
        "collection_count": collection.count(),
    }


def ingest_pdf(
    pdf_path: Path | str,
    collection_name: str,
    doc_id: Optional[str] = None,
    chunker: Optional[DocumentChunker] = None,
    embedder: Optional[BGEM3Embedder] = None,
) -> Dict[str, Any]:
    """End-to-end ingestion of a single PDF: extract -> quality gate -> chunk -> embed & index."""
    pdf_path = Path(pdf_path)
    print(f"\nProcessing PDF: {pdf_path.name}")

    # 1. Extract pages
    pages = extract_pages_from_pdf(pdf_path, doc_id=doc_id)
    print(f"  Extracted {len(pages)} pages.")

    # 2. Quality gate
    valid_pages, reports = filter_valid_pages(pages)
    passed_count = sum(1 for r in reports if r["passed"])
    print(f"  Quality Gate: {passed_count}/{len(pages)} pages passed.")

    if not valid_pages:
        print("  WARNING: No pages passed the quality gate. Skipping indexing.")
        return {"pages": len(pages), "valid_pages": 0, "chunks": 0}

    # 3. Chunker
    chunker = chunker or DocumentChunker()
    chunks = chunker.chunk_document(valid_pages)
    print(f"  Generated {len(chunks)} sentence-aware chunks.")

    # 4. Embed and Index
    stats = index_chunks(chunks, collection_name=collection_name, embedder=embedder)
    stats["pages"] = len(pages)
    stats["valid_pages"] = len(valid_pages)
    stats["quality_reports"] = reports
    return stats
