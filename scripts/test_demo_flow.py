"""End-to-end demo flow script for M9 acceptance verification.

Exercises the full user flow corresponding to the Streamlit UI tabs:
1. Ingest PDF & check quality gate (Tab 1: Upload and Index)
2. Ask English, Hindi, and Kannada questions (Tab 2: Ask)
3. Ask Off-topic question and verify ABSTAIN with 0 LLM calls (Tab 2: Trust Gate)
4. Generate Executive Digest in Hindi and Kannada (Tab 3: Summarise)
5. Verify VRAM remains under 4.0 GB (Tab 4: System Telemetry)
"""

import sys
import time
from pathlib import Path

# UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from govdoc.config import resolve_path
from govdoc.runtime.model_manager import ModelManager
from govdoc.runtime.vram_monitor import get_gpu_memory_used_mb, get_gpu_peak_memory_mb
from govdoc.ingest.embed_index import ingest_pdf, get_chroma_collection
from govdoc.generate.rag_pipeline import RAGPipeline
from govdoc.retrieve.trust_gate import TrustDecision
from govdoc.summarize.digest import generate_digest
from govdoc.summarize.summarizer import IndicBARTSummarizer
from govdoc.logging_utils import append_log


def run_full_demo():
    print("=" * 70)
    print("GOVDOC-RAG M9 ACCEPTANCE DEMO: FULL WORKFLOW")
    print("=" * 70)

    mm = ModelManager(initial_mode="qa")
    sample_pdf = resolve_path("data/corpus_pdfs/doc1_pib_digital_india_en.pdf")
    doc_id = sample_pdf.stem
    collection_name = f"doc_{doc_id}"

    # ── Step 1: Ingestion & Quality Gate (Tab 1) ─────────────────────────────
    print("\n[Step 1] Ingesting PDF & Quality Gate...")
    t0 = time.time()
    stats = ingest_pdf(sample_pdf, collection_name=collection_name)
    print(f"  Pages: {stats['pages']}, Valid: {stats['valid_pages']}, Chunks: {stats['chunks']}")
    print(f"  Ingest Time: {stats.get('total_time_s', 0.0):.2f}s")
    assert stats["valid_pages"] > 0, "No pages passed quality gate"
    assert stats["chunks"] > 0, "No chunks generated"

    # ── Step 2: Grounded QA - EN, HI, KN (Tab 2) ────────────────────────────
    print("\n[Step 2] Grounded Multilingual Question Answering...")
    mm.set_mode("qa")
    pipeline = RAGPipeline(collection_name=collection_name, llm_client=mm.get_qa_client())

    queries = [
        ("en", "What is the capital outlay sanctioned for Digital India implementation?"),
        ("hi", "डिजिटल इंडिया कार्यक्रम के लिए कुल कितना वित्तीय परिव्यय स्वीकृत किया गया है?"),
        ("kn", "ಡಿಜಿಟಲ್ ಇಂಡಿಯಾ ಯೋಜನೆಗಾಗಿ ಅನುಮೋದಿಸಲಾದ ಒಟ್ಟು ಬಂಡವಾಳ ವೆಚ್ಚ ಎಷ್ಟು?"),
    ]

    for lang, q_text in queries:
        print(f"\n  Query ({lang.upper()}): {q_text}")
        res = pipeline.answer_query(query=q_text, mode=lang, top_k=3)
        print(f"    Decision: {res['decision']} (Score: {res['top_score']:.4f})")
        print(f"    Answer:   {res['answer'][:120]}...")
        print(f"    Verifier: {'PASSED' if res['verifier']['pass'] else 'FLAGGED'}")
        print(f"    LLM Called: {res['llm_called']}")
        assert res["decision"] in (TrustDecision.ANSWER, TrustDecision.EVIDENCE_ONLY), f"Expected ANSWER/EVIDENCE_ONLY, got {res['decision']}"

    # ── Step 3: Off-topic Query Abstention (Tab 2) ───────────────────────────
    print("\n[Step 3] Off-Topic Query Abstention Test (Trust Gate)...")
    off_topic = "Who won the ICC Cricket World Cup in 2023?"
    res_abstain = pipeline.answer_query(query=off_topic, mode="en", top_k=3)
    print(f"  Query:      {off_topic}")
    print(f"  Decision:   {res_abstain['decision']} (Score: {res_abstain['top_score']:.4f})")
    print(f"  Answer:     {res_abstain['answer']}")
    print(f"  LLM Called: {res_abstain['llm_called']}")
    assert res_abstain["decision"] == TrustDecision.ABSTAIN, f"Expected ABSTAIN, got {res_abstain['decision']}"
    assert res_abstain["llm_called"] is False, "LLM was called on ABSTAIN query! Rule 4 violated."

    # ── Step 4: Executive Summarisation - HI & KN (Tab 3) ───────────────────
    print("\n[Step 4] Executive Digest Generation (Hindi & Kannada)...")
    mm.set_mode("summarise")
    summarizer = IndicBARTSummarizer()

    # Get chunks from indexed collection
    col = get_chroma_collection(collection_name)
    sample = col.get(limit=10, include=["documents", "metadatas"])
    chunks = [
        {"text": doc, "page": meta.get("page", 1), "chunk_id": cid}
        for cid, doc, meta in zip(sample["ids"], sample["documents"], sample["metadatas"])
    ]

    for tgt_lang in ["hi", "kn"]:
        print(f"\n  Generating digest in {tgt_lang.upper()}...")
        t0 = time.time()
        digest = generate_digest(
            chunks=chunks,
            src_lang="en",
            tgt_lang=tgt_lang,
            summarizer=summarizer,
            top_k=2,
        )
        print(f"  Digest ({tgt_lang.upper()}): {digest['digest'][:120]}... (in {time.time()-t0:.2f}s)")
        assert len(digest["digest"]) > 0, f"Digest for {tgt_lang} is empty"
        assert len(digest["gists"]) > 0, f"No gists generated for {tgt_lang}"

    # ── Step 5: System Telemetry (Tab 4) ─────────────────────────────────────
    print("\n[Step 5] System Telemetry & VRAM Check...")
    used_vram = get_gpu_memory_used_mb()
    peak_vram = get_gpu_peak_memory_mb()
    print(f"  Current VRAM: {used_vram:.1f} MB")
    print(f"  Peak VRAM:    {peak_vram:.1f} MB (Budget: 4096.0 MB)")
    assert used_vram <= 4096.0, f"VRAM budget exceeded: {used_vram} MB > 4096 MB"

    print("\n" + "=" * 70)
    print("M9 ACCEPTANCE: FULL DEMO FLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    append_log(
        f"M9 Demo Acceptance Flow: Ingestion ({stats['chunks']} chunks), "
        f"QA (en/hi/kn answers verified), Abstain (0 LLM calls), "
        f"Summarisation (hi/kn digests), Peak VRAM: {peak_vram:.1f} MB",
        module="M9:ACCEPTANCE",
    )


if __name__ == "__main__":
    run_full_demo()
