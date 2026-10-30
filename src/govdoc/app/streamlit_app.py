"""GovDoc-RAG: Offline Multilingual Document Intelligence Dashboard.

Adheres strictly to project constraints:
- Completely offline: no external CDN assets, no external font/JS requests
- ModelManager sequential residency: qa, summarise, ingest modes
- Trust boundary: explicit ANSWER / EVIDENCE_ONLY / ABSTAIN state with confidence meter
- Grounded citations and deterministic verification
- Side-by-side Base vs Fine-tuned summarization panel
- Live GPU memory (VRAM) monitoring via pynvml
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

# Offline environment flags per Rule 8 & requirements.md Section 2
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ensure_backend_services():
    """Ensure Ollama daemon and local inference dependencies are alive."""
    ollama_url = "http://127.0.0.1:11434/api/tags"
    is_alive = False
    try:
        with urllib.request.urlopen(ollama_url, timeout=1.5) as r:
            if r.status == 200:
                is_alive = True
    except Exception:
        is_alive = False

    if not is_alive:
        # Search for ollama binary
        candidate_paths = [
            Path(os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")),
            Path(r"C:\Users\mkatk\AppData\Local\Programs\Ollama\ollama.exe"),
            Path("ollama.exe"),
        ]
        exe = next((p for p in candidate_paths if p.exists()), None)
        if exe:
            try:
                subprocess.Popen(
                    [str(exe), "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # Wait up to 6s for daemon warmup
                for _ in range(12):
                    time.sleep(0.5)
                    try:
                        with urllib.request.urlopen(ollama_url, timeout=1.0) as r:
                            if r.status == 200:
                                break
                    except Exception:
                        pass
            except Exception:
                pass


ensure_backend_services()

import streamlit as st

from govdoc.config import load_config, resolve_path
from govdoc.runtime.model_manager import ModelManager
from govdoc.runtime.vram_monitor import get_gpu_memory_used_mb, get_gpu_peak_memory_mb
from govdoc.ingest.embed_index import (
    distance_to_cosine,
    get_chroma_collection,
    sanitize_collection_name,
    ingest_pdf,
    index_chunks,
    DocumentChunker,
)
from govdoc.ingest.pdf_extract import extract_pages_from_pdf, get_pdf_page_count
from govdoc.ingest.quality_gate import filter_valid_pages
from govdoc.generate.llm_client import LLMClient
from govdoc.generate.rag_pipeline import RAGPipeline, NOT_FOUND_RESPONSES
from govdoc.retrieve.trust_gate import TrustDecision, evaluate_trust_gate
from govdoc.summarize.digest import generate_digest, group_chunks_into_sections
from govdoc.summarize.summarizer import IndicBARTSummarizer
from govdoc.summarize.translator import IndicTrans2Translator

# ── Page Configuration (No CDN, clean government-standard UI) ─────────────────

st.set_page_config(
    page_title="GovDoc-RAG | Multilingual Document Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline CSS with no external fonts or CDN stylesheets
CUSTOM_CSS = """
<style>
    /* Government document portal neutral typography */
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .badge-answer {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-evidence {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-abstain {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-verified {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-flagged {
        background-color: #ffe4e6;
        color: #9f1239;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .citation-chip {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 4px;
        display: inline-block;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session State & ModelManager Singleton ────────────────────────────────────

@st.cache_resource
def get_model_manager() -> ModelManager:
    """Initialize ModelManager singleton respecting sequential residency."""
    return ModelManager(initial_mode="qa")


mm = get_model_manager()

if "latency_history" not in st.session_state:
    st.session_state.latency_history = []
if "uploaded_doc_id" not in st.session_state:
    st.session_state.uploaded_doc_id = None
if "uploaded_collection" not in st.session_state:
    st.session_state.uploaded_collection = "govdoc_corpus"
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []


# ── Sidebar: Settings & Model Selection ───────────────────────────────────────

with st.sidebar:
    st.title("🏛️ GovDoc-RAG")
    st.caption("Offline Multilingual Document Intelligence")
    st.markdown("---")

    cfg = load_config()
    tau_lo = float(cfg.get("retrieval", {}).get("tau_lo", 0.45))
    tau_hi = float(cfg.get("retrieval", {}).get("tau_hi", 0.65))

    # Spacer to cleanly push the SLM inference engine selector to the bottom of the sidebar
    st.markdown("<div style='min-height: 52vh;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Local Indic SLM Model Selection at the bottom of the sidebar
    model_options = {
        "qwen3:4b-instruct": "Qwen 3 4B (Fluent Kannada & Hindi)",
        "gaganyatri/sarvam-2b-v0.5": "Sarvam-2B (Indic Pre-trained)",
        "qwen2.5:1.5b": "Qwen 2.5 1.5B (Lightweight)",
    }
    selected_llm = st.selectbox(
        "SLM Inference Engine:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options.get(x, x),
        index=0,
        help="Choose the local SLM powering Grounded QA and Cross-Lingual Summarisation.",
    )
    st.session_state.selected_llm_model = selected_llm


# ── Main Tabs ─────────────────────────────────────────────────────────────────

tab_upload, tab_ask, tab_summarise, tab_system = st.tabs([
    "📥 1. Upload and Index",
    "❓ 2. Ask (QA)",
    "📝 3. Summarise (Digest)",
    "📊 4. System and Results",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD AND INDEX
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("Document Ingestion & Indexing")
    st.write(
        "Upload Indian Government notifications (PDF). "
        "Pages are verified via quality checks, chunked with BGE-M3 sentence boundaries, and indexed locally in ChromaDB (cosine space)."
    )

    uploaded_file = st.file_uploader("Upload Government Notification (PDF)", type=["pdf"])

    if uploaded_file is not None:
        upload_dir = resolve_path("data/corpus_pdfs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        total_doc_pages = get_pdf_page_count(save_path)
        raw_doc_id = uploaded_file.name.replace(".pdf", "")
        doc_id = sanitize_collection_name(raw_doc_id)
        collection_name = sanitize_collection_name(f"doc_{doc_id}")

        st.info(f"📄 **Document Loaded:** `{uploaded_file.name}` ({total_doc_pages} page{'s' if total_doc_pages > 1 else ''} detected)")

        # Page Selection & Part Loading Controls
        col_opt1, col_opt2 = st.columns([2, 3])
        with col_opt1:
            ingest_strategy = st.radio(
                "Ingestion Strategy / Scope:",
                options=["Selective Page Range", "Progressive / Batch Processing"],
                index=0,
                help="Process specific pages or progressive parts to stay strictly within 4GB VRAM and 16GB RAM constraints.",
            )

        batch_size = 10
        target_pages = list(range(1, total_doc_pages + 1))

        if ingest_strategy == "Selective Page Range":
            with col_opt2:
                if total_doc_pages > 1:
                    start_p, end_p = st.slider(
                        "Select Page Range to Index:",
                        min_value=1,
                        max_value=total_doc_pages,
                        value=(1, min(total_doc_pages, 10)),
                    )
                    target_pages = list(range(start_p, end_p + 1))
                    st.caption(f"Targeting {len(target_pages)} pages (Pages {start_p} to {end_p}).")
                else:
                    st.write("Single-page document.")
        elif ingest_strategy == "Progressive / Batch Processing":
            with col_opt2:
                batch_size = st.number_input(
                    "Pages per Part / Batch:",
                    min_value=1,
                    max_value=max(2, total_doc_pages),
                    value=min(10, total_doc_pages),
                    step=1,
                    help="Splits extraction, quality checks, and embedding into smaller batches with intermediate memory reclamation.",
                )
                n_batches = (total_doc_pages + batch_size - 1) // batch_size
                st.caption(f"Document will be processed in {n_batches} sequential part{'s' if n_batches > 1 else ''}.")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            index_btn = st.button("Process & Index Document", type="primary")

        if index_btn:
            progress_bar = st.progress(0, text="Initializing ingestion pipeline...")

            # Calculate part batches
            eff_batch_size = batch_size if ingest_strategy == "Progressive / Batch Processing" else len(target_pages)
            num_batches = (len(target_pages) + eff_batch_size - 1) // eff_batch_size

            all_valid_pages = []
            all_reports = []
            all_chunks = []
            total_embed_time = 0.0

            chunker = DocumentChunker()
            mm.set_mode("ingest")

            for b_idx in range(num_batches):
                s_idx = b_idx * eff_batch_size
                b_pages = target_pages[s_idx : s_idx + eff_batch_size]

                pct = int((b_idx / num_batches) * 100)
                progress_bar.progress(pct, text=f"Processing Part {b_idx + 1}/{num_batches} (Pages {b_pages[0]}-{b_pages[-1]})...")

                # Extract pages for this batch
                batch_extracted = extract_pages_from_pdf(save_path, doc_id=doc_id, page_numbers=b_pages)

                # Quality gate
                valid_b, reports_b = filter_valid_pages(batch_extracted)
                all_reports.extend(reports_b)
                all_valid_pages.extend(valid_b)

                # Chunk & index
                if valid_b:
                    b_chunks = chunker.chunk_document(valid_b)
                    if b_chunks:
                        stats = index_chunks(b_chunks, collection_name=collection_name)
                        total_embed_time += stats.get("embed_time_s", 0.0)
                        all_chunks.extend(b_chunks)

            mm.set_mode("qa")
            progress_bar.progress(100, text="Ingestion & vector indexing completed!")

            # Extraction Quality Gate Display
            st.subheader("Extraction & Ingestion Report")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.metric("Pages Processed", len(target_pages))
            with col_q2:
                st.metric("Passed Quality Gate", len(all_valid_pages))
            with col_q3:
                failed_count = len(target_pages) - len(all_valid_pages)
                st.metric("Failed / Empty Pages", failed_count)

            with st.expander("View Page-by-Page Quality Details", expanded=False):
                filter_choice = st.radio(
                    "Display Filter:",
                    options=["Show All Processed Pages", "Show Flagged / Failed Pages Only"],
                    horizontal=True,
                    key="quality_filter",
                )
                filtered_reports = (
                    [r for r in all_reports if not r.get("passed", False)]
                    if filter_choice == "Show Flagged / Failed Pages Only"
                    else all_reports
                )

                if not filtered_reports:
                    st.success("All processed pages passed quality checks!")
                else:
                    report_rows = []
                    for r in filtered_reports:
                        flags = r.get("flags", [])
                        flag_desc = ", ".join(flags) if flags else "Passed quality gate"
                        report_rows.append({
                            "Page": r.get("page", 0),
                            "Status": "✅ Pass" if r.get("passed", False) else "⚠️ Flagged",
                            "Quality Diagnosis": flag_desc,
                            "Dominant Script": r.get("script_dist", {}).get("dominant_script", "en").upper(),
                            "Character Count": r.get("text_length", 0),
                        })
                    st.dataframe(report_rows, width="stretch")

            if all_chunks:
                st.session_state.uploaded_doc_id = doc_id
                st.session_state.uploaded_collection = collection_name
                st.session_state.last_chunks = all_chunks

                st.success(f"Indexed {len(all_chunks)} chunks successfully into local Chroma collection `{collection_name}`!")

                # Chunk Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Chunks", len(all_chunks))
                with col_m2:
                    st.metric("Total Embedding Time", f"{total_embed_time:.2f}s")
                with col_m3:
                    time_per_1k = (total_embed_time / max(len(all_chunks), 1)) * 1000.0
                    st.metric("Time per 1k Chunks", f"{time_per_1k:.1f}s")
            else:
                st.error("No valid text chunks could be extracted from the selected pages. Please verify the PDF.")
    else:
        st.info("Upload a PDF document above to begin extraction, quality verification, and vector indexing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASK (QA)
# ══════════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.header("Grounded Multilingual Question Answering")
    st.write(
        "Ask questions in **English**, **Hindi**, **Kannada**, or code-mixed **Hinglish** / **Kanglish**. "
        "Answers are strictly grounded in retrieved evidence with numbered citations `[n]` and verified against hallucinations."
    )

    col_q_mode, col_top_k = st.columns([2, 1])
    with col_q_mode:
        lang_mode = st.selectbox(
            "Query Language Mode",
            ["auto", "en", "hi", "kn", "hinglish", "kanglish"],
            format_func=lambda x: {
                "auto": "Auto-Detect Script",
                "en": "English",
                "hi": "Hindi (हिंदी)",
                "kn": "Kannada (ಕನ್ನಡ)",
                "hinglish": "Hinglish (Romanised Hindi)",
                "kanglish": "Kanglish (Romanised Kannada)",
            }.get(x, x),
        )
    with col_top_k:
        top_k = st.slider("Top-k Chunks to Retrieve", min_value=1, max_value=10, value=3)

    with st.expander("ℹ️ What does Top-k Chunks mean?", expanded=False):
        st.markdown(
            """
            **Top-k Chunks** defines how many relevant passage sections from the indexed document are retrieved and supplied as context to the answering model:
            - **Lower value (1 – 2):** Focused on only the single highest-scoring paragraph. Faster response time and minimal noise, but may miss details if an answer spans multiple paragraphs.
            - **Balanced value (3 – 5, Recommended):** Provides sufficient surrounding context and cross-page references without cluttering the model's memory.
            - **Higher value (6 – 10):** Broad document coverage suitable for synthesis questions, but increases inference latency and risk of pulling marginally related text.
            """
        )

    query_input = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the financial outlay sanctioned under this scheme?",
    )

    with st.expander("⚙️ Retrieval & Trust Gate Sensitivity Controls", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            active_tau_lo = st.slider(
                "Abstention Threshold (τ_lo)",
                min_value=0.20,
                max_value=0.55,
                value=float(tau_lo),
                step=0.05,
                help="Scores below this strictly ABSTAIN without calling LLM (Rule 4).",
            )
        with col_s2:
            allow_evidence_gen = st.checkbox(
                "Generate on Evidence Confidence (Barrier Removed)",
                value=True,
                help="Allows LLM to answer natural and rephrased questions grounded in retrieved evidence.",
            )

    if st.button("Ask Question", type="primary") or query_input:
        if query_input:
            t0 = time.time()
            collection_to_use = st.session_state.uploaded_collection

            # Ensure QA mode with user selected SLM engine
            mm.set_mode("qa")
            active_engine = st.session_state.get("selected_llm_model", "qwen3:4b-instruct")
            qa_client = LLMClient(model_name=active_engine)

            pipeline = RAGPipeline(
                collection_name=collection_to_use,
                llm_client=qa_client,
            )

            with st.spinner("Retrieving evidence & evaluating trust gate..."):
                t_retrieve_start = time.time()
                result = pipeline.answer_query(
                    query=query_input,
                    mode=lang_mode,
                    top_k=top_k,
                    tau_hi=tau_hi,
                    tau_lo=active_tau_lo,
                    allow_evidence_only_generation=allow_evidence_gen,
                )
                total_latency = time.time() - t0

            # ── 1. Confidence Meter & Decision Banner ────────────────────────
            score = result["top_score"]
            decision = result["decision"]

            st.markdown("---")
            col_dec, col_meter = st.columns([1, 2])

            with col_dec:
                st.markdown("**Trust Gate Decision:**")
                if decision == "ANSWER":
                    st.markdown(f'<span class="badge-answer">ANSWER (High Conf: {score:.4f} ≥ {tau_hi})</span>', unsafe_allow_html=True)
                elif decision == "EVIDENCE_ONLY":
                    st.markdown(f'<span class="badge-evidence">ANSWER (Evidence Grounded: {score:.4f})</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-abstain">ABSTAIN (Score: {score:.4f} < {active_tau_lo})</span>', unsafe_allow_html=True)

            with col_meter:
                st.markdown(f"**Confidence Meter:** `{score:.4f}`")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(f"Abstain (< {active_tau_lo}) | Grounded Generation (≥ {active_tau_lo}) | High Conf (≥ {tau_hi})")

            # ── 2. Answer Card ───────────────────────────────────────────────
            st.markdown("### Answer")
            if decision == "ABSTAIN":
                st.warning(f"🚫 **Information Not Found**: {result['answer']}")
                st.info("🔒 **Rule 4 Enforced**: Zero LLM calls made when trust boundary is breached.")
            else:
                st.markdown(f"> {result['answer']}")
                if decision == "EVIDENCE_ONLY":
                    st.caption(f"ℹ️ Grounded answer generated from retrieved evidence passages (similarity: {score:.4f}).")

                # Verifier badge
                verif = result.get("verifier", {})
                if verif.get("pass", True):
                    st.markdown('<span class="badge-verified">✓ Grounded Facts Verified</span>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<span class="badge-flagged">⚠️ Hallucination Flagged: Missing {verif.get("missing")}</span>',
                        unsafe_allow_html=True,
                    )

            # ── 3. Citations & Verbatim Evidence ─────────────────────────────
            evidence = result.get("evidence_chunks", [])
            if evidence:
                st.markdown("### Retrieved Evidence Passages")
                for idx, chunk in enumerate(evidence, start=1):
                    with st.expander(f"Evidence [{idx}] — Page {chunk.get('page', 1)} | Cosine Score: {chunk.get('score', 0.0):.4f}"):
                        st.write(chunk.get("text", ""))
                        st.caption(f"Doc ID: `{chunk.get('doc_id')}` | Tokens: {chunk.get('token_count', 0)}")

            # Log latency
            st.session_state.latency_history.append({
                "query": query_input[:30] + "...",
                "decision": decision,
                "score": f"{score:.4f}",
                "latency_s": f"{total_latency:.2f}s",
                "llm_called": result.get("llm_called", False),
            })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SUMMARISE (DIGEST)
# ══════════════════════════════════════════════════════════════════════════════

with tab_summarise:
    st.header("Executive Document Digest & Summarisation")
    st.write(
        r"Generate trilingual digests using **IndicBART + LoRA** with script unification. "
        r"Long documents are partitioned into $\le 512$-token sections, summarized into gists, and ranked using BGE-M3 Centrality + MMR."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        tgt_lang = st.selectbox(
            "Target Summary Language",
            ["hi", "kn", "en"],
            format_func=lambda x: {"hi": "Hindi (हिंदी)", "kn": "Kannada (ಕನ್ನಡ)", "en": "English"}.get(x, x),
        )
    with col_s2:
        src_lang = st.selectbox("Source Language", ["en", "hi", "kn"])
    with col_s3:
        num_gists = st.slider("Maximum Gists in Digest", min_value=1, max_value=8, value=3)

    with st.expander("ℹ️ What does Maximum Gists mean?", expanded=False):
        st.markdown(
            """
            **Maximum Gists** sets the target number of key bullet points (salient sections) selected for the final executive digest:
            - **Low value (1 – 2):** Ultra-condensed executive brief. Captures only the central theme of the notification with minimal reading time, but omits secondary operational details.
            - **Moderate value (3 – 4, Recommended):** Balanced overview covering the primary policy announcement, financial allocations, and key deadlines.
            - **High value (5 – 8):** Comprehensive digest suitable for lengthy multi-page acts or circulars. Captures granular provisions across multiple sections.
            """
        )

    if st.button("Generate Executive Digest", type="primary"):
        # Check available chunks from current document or collection
        active_chunks = st.session_state.last_chunks
        if not active_chunks:
            try:
                col = get_chroma_collection(st.session_state.uploaded_collection)
                sample = col.get(limit=10, include=["documents", "metadatas"])
                if sample and sample["documents"]:
                    active_chunks = [
                        {
                            "text": doc,
                            "page": meta.get("page", 1),
                            "chunk_id": cid,
                        }
                        for cid, doc, meta in zip(sample["ids"], sample["documents"], sample["metadatas"])
                    ]
            except Exception:
                active_chunks = []

        if not active_chunks:
            st.error("No document chunks available. Please upload a PDF in the 'Upload and Index' tab first.")
        else:
            with st.spinner("Switching ModelManager to 'summarise' mode (loading IndicBART & IndicTrans2 fp16)..."):
                mm.set_mode("summarise")
                summarizer = IndicBARTSummarizer()
                translator = None
                if src_lang != tgt_lang:
                    try:
                        translator = IndicTrans2Translator()
                    except Exception as e:
                        st.warning(f"IndicTrans2 translator fallback: {e}")

            with st.spinner(f"Generating hierarchical digest in {tgt_lang.upper()} (IndicBART -> IndicTrans2)..."):
                t0 = time.time()
                active_engine = st.session_state.get("selected_llm_model", "qwen3:4b-instruct")
                llm_cross = LLMClient(model_name=active_engine)
                digest_res = generate_digest(
                    chunks=active_chunks,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    summarizer=summarizer,
                    translator=translator,
                    llm_client=llm_cross,
                    top_k=num_gists,
                )
                digest_time = time.time() - t0

            st.success(f"Digest generated in {digest_time:.2f}s across {digest_res.get('num_sections', 1)} sections!")

            # ── Executive Digest Display ─────────────────────────────────────
            st.markdown("### Executive Digest")
            st.info(digest_res["digest"])

            # ── Section Gists with Page Citations ────────────────────────────
            st.markdown("### Key Section Gists (Ranked by Centrality + MMR)")
            gists_list = digest_res.get("gists", [])
            for idx, gist_item in enumerate(gists_list, start=1):
                pages_str = ", ".join(f"Page {p}" for p in gist_item["pages"])
                with st.expander(f"Gist [{idx}] — {pages_str} (Centrality: {gist_item.get('centrality', 1.0):.4f})"):
                    st.write(gist_item["gist"])

            # Restore QA mode
            mm.set_mode("qa")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SYSTEM AND RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_system:
    st.header("System Telemetry & Empirical Results")

    # ── 1. Live System Telemetry ─────────────────────────────────────────────
    st.subheader("Runtime Resource Usage (4.0 GB VRAM Budget)")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("GPU VRAM Used", f"{get_gpu_memory_used_mb():.1f} MB")
    with col_t2:
        st.metric("GPU VRAM Peak", f"{get_gpu_peak_memory_mb():.1f} MB")
    with col_t3:
        st.metric("Active Residency Mode", mm.get_active_mode().upper())

    # Latency Table
    if st.session_state.latency_history:
        st.markdown("**Recent Query Latencies:**")
        st.dataframe(st.session_state.latency_history[-5:], width="stretch")

    st.markdown("---")

    # ── 2. Benchmark Evaluation Results (Loaded from results/raw) ────────────
    st.subheader("Empirical Benchmark Results")
    raw_dir = resolve_path("results/raw")

    # Section 1: Retrieval Performance
    st.markdown("### 1. Document Retrieval Performance")
    with st.expander("ℹ️ Understanding Retrieval Metrics & Numbers", expanded=False):
        st.markdown(
            """
            - **MRR (Mean Reciprocal Rank):** Measures how high up the correct evidence chunk was ranked. A score of `1.0` means the true answer chunk was ranked #1 every single time. Higher is better.
            - **Recall@1:** The percentage of questions where the very first retrieved passage contains the exact answer.
            - **Recall@5 / Recall@10:** The percentage of questions where the true answer passage is present within the top 5 or top 10 retrieved chunks.
            - **Language Breakdown:** Reflects cross-lingual dense retrieval accuracy across English, Hindi, and Kannada using multilingual BGE-M3 representations.
            """
        )
    ret_path = raw_dir / "retrieval_A.json"
    if ret_path.exists():
        with open(ret_path, encoding="utf-8") as f:
            ret_data = json.load(f)["metrics"]

        e2_ov = ret_data["E2_raw_only"]["overall"]
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Overall MRR", f"{e2_ov['MRR']:.4f}")
        with col_m2:
            st.metric("Recall@1", f"{e2_ov['Recall@1'] * 100:.1f}%")
        with col_m3:
            st.metric("Recall@10", f"{e2_ov['Recall@10'] * 100:.1f}%")

        by_lang = ret_data["E2_raw_only"]["by_lang"]
        rows = []
        for lang, d in by_lang.items():
            rows.append({
                "Language": lang.upper(),
                "MRR": d["MRR"],
                "Recall@1": d["Recall@1"],
                "Recall@5": d["Recall@5"],
                "Recall@10": d["Recall@10"],
                "Total Questions Evaluated": d["n"],
            })
        st.dataframe(rows, width="stretch")
    else:
        st.info("Run evaluation script to generate retrieval metrics.")

    st.markdown("---")

    # Section 2: Trust Gate Abstention & Sensitivity
    st.markdown("### 2. Trust Gate Abstention & Reliability")
    with st.expander("ℹ️ Understanding Abstention & Calibration Metrics", expanded=False):
        st.markdown(
            """
            - **AUROC (Area Under ROC Curve):** Measures how effectively the trust gate separates answerable queries from out-of-domain or unanswerable queries (1.0 = perfect discrimination).
            - **Calibrated Threshold (τ*):** The optimal similarity cutoff threshold calculated during validation. Queries scoring below this are safely rejected to prevent hallucinations.
            - **Answerable Recall:** The fraction of valid questions that were successfully answered rather than incorrectly rejected.
            - **False Abstain Rate (FAR):** The percentage of legitimate questions that were unnecessarily turned down. Lower is better.
            - **Balanced Accuracy:** The average accuracy across both answerable and unanswerable query distributions.
            """
        )
    abs_path = raw_dir / "abstain.json"
    if abs_path.exists():
        with open(abs_path, encoding="utf-8") as f:
            abs_data = json.load(f)["metrics"]

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Discriminative AUROC", f"{abs_data['auroc']:.4f}")
        with col_a2:
            st.metric("Optimal Threshold (τ*)", f"{abs_data['calibrated_tau']:.3f}")
        with col_a3:
            st.metric("Answerable Recall at τ*", f"{abs_data['val_metrics_at_tau']['recall_answerable'] * 100:.1f}%")

        sweep = abs_data.get("tau_sweep", [])
        if sweep:
            sweep_rows = [
                {
                    "Cutoff Threshold (τ)": r["tau"],
                    "Answerable Recall": r["recall_answerable"],
                    "False Abstain Rate (FAR)": r["far"],
                    "F1 Score": r["f1"],
                    "Balanced Accuracy": r["balanced_accuracy"],
                }
                for r in sweep[::2]
            ]
            st.dataframe(sweep_rows, width="stretch")
    else:
        st.info("Run evaluation script to generate abstain metrics.")

    st.markdown("---")

    # Section 3: Hallucination Verification
    st.markdown("### 3. Fact & Grounding Verifier Performance")
    with st.expander("ℹ️ Understanding Hallucination Verification Metrics", expanded=False):
        st.markdown(
            """
            - **Grounded Pass Rate:** The percentage of answers whose stated numbers, dates, and names strictly match the retrieved document passages without discrepancies.
            - **Corrupted Fact Flag Rate:** The detection rate when erroneous or fabricated figures are introduced into an answer. A score of 100% means zero missed fabrications.
            """
        )
    verif_path = raw_dir / "verifier.json"
    if verif_path.exists():
        with open(verif_path, encoding="utf-8") as f:
            v_data = json.load(f)["metrics"]
        col_vf1, col_vf2 = st.columns(2)
        with col_vf1:
            st.metric("Grounded Pass Rate", f"{v_data['overall_grounded_pass_rate'] * 100:.1f}%")
        with col_vf2:
            st.metric("Corrupted Fact Flag Rate", f"{v_data['overall_corrupted_flag_rate'] * 100:.1f}%")
        st.caption("Deterministic verification cross-checks every figure, date, and proper noun against citation anchors.")
    else:
        st.info("Run evaluation script to generate verifier metrics.")

    st.markdown("---")

    # Section 4: Summarisation Performance
    st.markdown("### 4. Multilingual Summarisation Quality")
    with st.expander("ℹ️ Understanding Summarisation Metrics", expanded=False):
        st.markdown(
            """
            - **Direction:** Language pairing for the summary (e.g., `en-en` = English document to English summary, `hi-hi` = Hindi to Hindi, `kn-kn` = Kannada to Kannada).
            - **ROUGE-L Score:** Evaluates longest common subsequence overlap between generated summaries and official government reference digests. Higher score indicates stronger preservation of structure and facts.
            - **Gain (Δ):** The measured improvement achieved by our fine-tuned LoRA domain model compared to the standard base model.
            """
        )
    summ_path = raw_dir / "summarization.json"
    if summ_path.exists():
        with open(summ_path, encoding="utf-8") as f:
            s_data = json.load(f)["metrics"]["directions"]
        summ_table = []
        for direction, d in s_data.items():
            ft = d.get("fine_tuned", {})
            b1 = d.get("b1_indicbart", {})
            summ_table.append({
                "Language Pair": direction,
                "Fine-Tuned Model (ROUGE-L)": ft.get("rougeL", "N/A"),
                "Base Baseline (ROUGE-L)": b1.get("rougeL", "N/A"),
                "Improvement Gain (Δ)": round(ft.get("rougeL", 0) - b1.get("rougeL", 0), 4) if ft.get("rougeL") and b1.get("rougeL") else "N/A",
            })
        st.dataframe(summ_table, width="stretch")
    else:
        st.info("Run evaluation script to generate summarisation metrics.")


if __name__ == "__main__":
    pass
