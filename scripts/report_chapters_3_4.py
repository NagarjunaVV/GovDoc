"""
scripts/report_chapters_3_4.py
Chapters 3 and 4 for GovDoc-RAG RV University Project Report.
"""

import os
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_chapter_3(doc, h):
    add_p = h["add_p"]
    add_chap_heading = h["add_chap_heading"]
    add_sec_heading = h["add_sec_heading"]
    add_subsec_heading = h["add_subsec_heading"]
    add_subsubsec_heading = h["add_subsubsec_heading"]
    add_fig_image = h["add_fig_image"]
    style_academic_table = h["style_academic_table"]

    # ════════════════════════════════════════════════════════════════════════════
    # CHAPTER 3: SYSTEM METHODOLOGY AND ARCHITECTURE
    # ════════════════════════════════════════════════════════════════════════════
    add_chap_heading(3, "SYSTEM METHODOLOGY AND ARCHITECTURE")

    add_sec_heading("3.1", "End-to-End System Architecture")
    add_p(
        "GovDoc-RAG is architected as an edge-native, privacy-preserving document intelligence framework designed specifically to navigate "
        "the linguistic, syntactic, and computational challenges of Indian government notifications. Rather than treating RAG as a monolithic "
        "vector-lookup script, GovDoc-RAG decouples execution into six strictly guarded, modular components organized into two synchronized "
        "functional workflows: (1) The Interactive Cross-Lingual Question Answering and Trust Verification Workflow; and (2) The Parameter-Efficient "
        "Multilingual Summarisation Workflow. Both workflows execute locally under the supervision of the Edge Memory Co-Residency Guard."
    )
    add_p(
        "The overall pipeline operates through two discrete operational phases: an offline preparation phase and an online interactive serving phase. "
        "In the offline phase, raw official PDF notifications are ingested, parsed into spatial layout hierarchies, partitioned into semantic section chunks, "
        "and embedded into a persistent local vector database (ChromaDB) via BAAI/bge-m3. In the online phase, user queries submitted in any supported "
        "language (English, Hindi, Kannada, Hinglish, or Kanglish) are ingested by the Cross-Lingual Query Bridge. The bridge synthesizes dual-view embeddings, "
        "which are evaluated against a calibrated Cosine Trust Gate. Depending on empirical similarity confidence, queries are either rejected via deterministic "
        "abstention, subjected to strict evidence grounding, or routed directly to the Small Language Model (Qwen 3 4B Instruct). Responses undergo entity-level "
        "hallucination verification before display. Concurrently, executive document summarisation is executed through LoRA-adapted IndicBART and IndicTrans2."
    )

    # Embed Figure 1: Architecture
    add_fig_image(
        "results/figs_png/fig1_architecture.png",
        "3.1",
        "End-to-End GovDoc-RAG System Architecture Pipeline showing Offline Ingestion, Cross-Lingual Query Bridging, Calibrated Trust Boundary Gate, Grounded Generation with Hallucination Verification, IndicBART LoRA Summarisation, and Edge Memory Co-Residency Guard.",
        width=Inches(4.8)
    )

    add_sec_heading("3.2", "Layout-Aware Document Ingestion and Chunking")
    add_p(
        "Standard RAG pipelines utilize arbitrary character-count or token-count chunking (e.g., 500 characters with 50-character overlap). On official government "
        "circulars, this naive approach causes catastrophic context fragmentation: statutory tables are severed mid-row, section titles are separated from their "
        "governing legal stipulations, and bracketed page numbers are lost. GovDoc-RAG implements a specialized layout-aware PDF ingestion engine powered by PyMuPDF (fitz)."
    )
    add_p(
        "The layout ingestion engine processes documents through a multi-pass hierarchical parsing algorithm: "
        "(1) Block & Span Font Profiling: For every text span on page p, the engine records font size, font weight (bold/regular), font color, and bounding box coordinates (x0, y0, x1, y1); "
        "(2) Header & Section Boundary Detection: Spans exceeding 1.2x the modal body font size or rendered in bold capital letters are classified as section delimiters (e.g., '1. ELIGIBILITY CRITERIA', 'ORDER NO. FD 05 PES 2024'); "
        "(3) Structural Section Preservation: Text blocks belonging to the same heading are grouped into unified semantic units. When a section exceeds the dense retriever's 512-token optimal window, an adaptive sentence-boundary sliding window (window size = 450 tokens, stride = 100 tokens) is applied; and "
        "(4) Metadata Enrichment: Each indexed chunk c is tagged with rich metadata: {doc_id, page_number, section_title, char_length, language_script}. This metadata is preserved into ChromaDB to enable deterministic [Page X] citation tracing during generation."
    )

    add_sec_heading("3.3", "Cross-Lingual Dual-View Query Bridging")
    add_p(
        "A central impediment to multilingual retrieval in India is the script divide. Official notifications exist in native Devanagari or English scripts, while "
        "a vast demographic of citizens formulate digital queries using Roman keyboards (e.g., typing 'karnataka raita vidyanidhi yojana eligibility' in Kanglish, "
        "or 'pm kisan samman nidhi kist kab aayegi' in Hinglish). Standard multilingual embedding models exhibit severe catastrophic recall degradation when queries "
        "are presented in romanised or code-mixed formats, as their tokenizers fragment informal transliterations into out-of-vocabulary character n-grams."
    )
    add_p(
        "GovDoc-RAG overcomes this barrier through an innovative Dual-View Query Bridge powered by AI4Bharat's IndicXlit neural transliteration engine. When a query "
        "q_raw arrives, the bridge determines whether the query contains Roman script text representing an Indic language. If detected, IndicXlit performs Aksharantar "
        "phonological transliteration to project the Latin string into native Devanagari (q_hi) and native Kannada (q_kn) scripts. The dual-view query representation "
        "v_dual(q) is formulated as the balanced interpolation of dense embeddings generated by BAAI/bge-m3:"
    )
    add_p(
        "    v_dual(q) = 0.5 * e_bge(q_raw) + 0.5 * e_bge(IndicXlit(q_raw))\n\n"
        "where e_bge represents the 1024-dimensional normalized embedding vector. Both representations query the persistent ChromaDB index, and candidate chunks are "
        "re-ranked using reciprocal rank fusion. This guarantees that whether a user types in English, formal Devanagari Hindi, native Kannada, or romanised Hinglish/Kanglish, "
        "the dense search probes the exact underlying semantic manifold."
    )

    # Embed Figure 2: Query Bridge
    add_fig_image(
        "results/figs_png/fig2_query_bridge.png",
        "3.2",
        "Cross-Lingual Query Bridge and Calibrated Trust Gate Decision Logic showing Dual-View Transliteration, BGE-M3 Dense Semantic Retrieval, and Three-Way Threshold Routing.",
        width=Inches(5.0)
    )

    add_sec_heading("3.4", "Calibrated Trust Gate and Abstention Shield")
    add_p(
        "A critical vulnerability of generative AI in public administration is ungrounded hallucination on unanswerable queries. Standard RAG frameworks blindly feed "
        "top-k retrieved chunks to the LLM regardless of semantic relevance, forcing the generator to fabricate plausible answers. In GovDoc-RAG, we implement a "
        "mathematically calibrated Cosine Trust Gate that acts as an immutable firewall prior to Small Language Model invocation."
    )
    add_p(
        "ChromaDB computes cosine distance d_cos(q, c) between normalized query vector e(q) and candidate chunk e(c), defined as d_cos(q, c) = 1 - cos(q, c). "
        "We convert this distance into a standard cosine similarity metric bounded in [-1, 1]:\n\n"
        "    s(q, c) = 1 - 0.5 * d_cos(q, c) = cos(q, c)\n\n"
        "Let s* denote the maximum similarity score across all retrieved chunks for query q:\n\n"
        "    s* = max_{c in C} s(q, c)\n\n"
        "The Trust Gate evaluates s* against a calibrated dual-threshold decision rule:\n\n"
        "    Action(q) = {\n"
        "        Direct Answer Generation via Qwen 3 4B,           if s* >= tau_hi (0.65)\n"
        "        Strict Evidence-Grounded Verification,            if tau_lo <= s* < tau_hi (tau_lo = 0.30)\n"
        "        Deterministic Abstention (Out-of-Domain),         if s* < tau_lo (0.30)\n"
        "    }\n\n"
        "Our empirical calibration on Benchmark A established that setting the critical trust threshold tau* = 0.400 guarantees 100% answerable query recall while "
        "driving the False Accept Rate on out-of-domain queries to 0.0%. When a query falls into the abstention band, the system immediately returns an authoritative, "
        "multilingual refusal: 'This document does not contain verifiable evidence regarding your query. No answer generated to prevent misinformation.'"
    )

    add_sec_heading("3.5", "Grounded Answer Generation and Entity Verification")
    add_p(
        "For queries that clear the trust gate (s* >= tau_lo), the top-k retrieved evidence chunks (k = 3) are compiled into a strict grounding prompt. "
        "The prompt is dispatched to Qwen 3 4B Instruct, running locally in 4-bit quantisation (Q4_K_M GGUF format) via llama.cpp/Ollama. The system prompt instructs "
        "the model with inviolable operational constraints:\n"
        "(a) Synthesize answers exclusively using the provided passage excerpts;\n"
        "(b) Do not extrapolate beyond explicit textual facts;\n"
        "(c) Attach mandatory bracketed citation anchors referencing the exact source page (e.g., '[Page 2]', '[Page 5]') for every asserted claim; and\n"
        "(d) If a specific fact is absent from the provided excerpts, state that it is not specified."
    )
    add_p(
        "To provide foolproof safety, draft responses generated by Qwen 3 4B are not displayed directly to users. Instead, they pass through a deterministic, "
        "post-generation Entity-Level Hallucination Verifier. The verifier executes four parallel checks using specialized regex parsers and named entity rules:\n"
        "1. Date and Temporal Consistency: Extracts all dates (e.g., '15-08-2024', 'August 2024') and validates that each date exists verbatim in the retrieved source chunks;\n"
        "2. Monetary and Numerical Verification: Extracts currency amounts, percentages, penalty figures, and scheme allocations (e.g., 'Rs. 25,000', '18% GST', '5.2 Crore') and verifies their existence in source evidence;\n"
        "3. Proper Entity Matching: Checks department names, act titles, and scheme identifiers; and\n"
        "4. Citation Validity: Verifies that cited page numbers [Page X] correspond to pages actually present in the retrieved evidence set.\n\n"
        "If any numerical or temporal assertion in the generated draft contradicts or is missing from the evidence chunks, the verifier flags the response, rejects the ungrounded assertion, "
        "and appends a conspicuous audit warning for the user."
    )

    add_sec_heading("3.6", "Edge Memory Co-Residency Guard (Sub-4GB VRAM)")
    add_p(
        "Deploying an advanced multilingual RAG system on commodity edge hardware presents a formidable memory bottleneck: modern consumer laptops (such as those "
        "equipped with NVIDIA GeForce RTX 2050 or RTX 3050 GPUs) have only 4.0 GB of dedicated VRAM (4,096 MB). Concurrently loading BGE-M3 (568M params, ~1.85 GB), "
        "Qwen 3 4B (4.0B params, ~2.5 GB), IndicBART with LoRA (~1.2 GB), and IndicTrans2 (~1.1 GB) would require over 6.6 GB of VRAM, triggering immediate Out-of-Memory (OOM) "
        "crashes or catastrophic CUDA paging onto system RAM."
    )
    add_p(
        "GovDoc-RAG solves this hardware barrier by implementing an Edge Memory Co-Residency Guard. The guard operates as a strict mutual-exclusion state machine that enforces "
        "discrete lifecycle transitions across three residency modes:\n"
        "• Mode 1: Ingestion & Indexing Mode: Allocates BGE-M3 dense embedding engine. Peak VRAM: 1,850.0 MB (Margin: +2,246.0 MB);\n"
        "• Mode 2: Question Answering Mode: Unloads heavy batch buffers, loads Qwen 3 4B Instruct via Ollama with 4,096-token context window. Peak VRAM: 3,120.0 MB (Margin: +976.0 MB);\n"
        "• Mode 3: Summarisation Mode: Flushes Qwen 3 4B from active VRAM, executes torch.cuda.empty_cache() and Python garbage collection (gc.collect()), loads IndicBART with LoRA adapter and IndicTrans2. Peak VRAM: 3,219.0 MB (Margin: +877.0 MB).\n\n"
        "Mathematically, total peak GPU utilization across the system is strictly bounded by the maximum mode footprint:\n\n"
        "    M_peak = max(M_ingest, M_qa, M_summ) = max(1850.0, 3120.0, 3219.0) = 3219.0 MB < 4096.0 MB\n\n"
        "This architectural discipline guarantees that GovDoc-RAG operates reliably with a minimum safety headroom of 877.0 MB at all times, completely eliminating GPU memory overflow crashes."
    )

    add_sec_heading("3.7", "Abstractive Summarisation Pipeline with LoRA")
    add_p(
        "While Question Answering resolves specific factual inquiries, senior administrative officers frequently require rapid, holistic executive digests of multi-page "
        "gazette notifications. GovDoc-RAG implements an abstractive summarisation workflow that leverages ai4bharat/IndicBART augmented with Low-Rank Adaptation (LoRA)."
    )
    add_p(
        "Standard pre-trained IndicBART achieves near-zero summarisation scores on official government press releases due to cross-lingual vocabulary fragmentation. We apply "
        "LoRA (rank r=16, scaling factor alpha=32, dropout=0.05) to all self-attention query and value projection matrices (q_proj, v_proj). Trainable parameter updates are "
        "decomposed as W = W0 + Delta W = W0 + (alpha / r) * B * A, where B in R^{d x r} and A in R^{r x k}. Only 1,179,648 parameters are updated out of 244 million total parameters "
        "(0.48% trainable footprint), completely preserving pre-trained multilingual representations while injecting bureaucratic headline synthesis capabilities."
    )
    add_p(
        "For multi-page circulars, GovDoc-RAG applies a hierarchical Map-Reduce summarisation pattern: "
        "(1) Document Partitioning: The document is segmented into layout-delimited logical sections; "
        "(2) Map Stage: IndicBART + LoRA generates concise 25–40 word abstractive section gists in Hindi/English; "
        "(3) Reduce Stage: The section gists are aggregated, re-summarized into an executive notification summary; and "
        "(4) Cross-Lingual Projection: IndicTrans2 translates the final executive digest into the user's selected regional language (e.g., Kannada), delivering fluent, "
        "vernacular executive briefings within 18.04 seconds per document section."
    )

    print("Chapter 3 built.")

def build_chapter_4(doc, h):
    add_p = h["add_p"]
    add_chap_heading = h["add_chap_heading"]
    add_sec_heading = h["add_sec_heading"]
    add_subsec_heading = h["add_subsec_heading"]
    add_subsubsec_heading = h["add_subsubsec_heading"]
    add_fig_image = h["add_fig_image"]
    style_academic_table = h["style_academic_table"]

    # ════════════════════════════════════════════════════════════════════════════
    # CHAPTER 4: IMPLEMENTATION AND EXPERIMENTAL SETUP
    # ════════════════════════════════════════════════════════════════════════════
    add_chap_heading(4, "IMPLEMENTATION AND EXPERIMENTAL SETUP")

    add_sec_heading("4.1", "Datasets and Curated Benchmarks")
    add_p(
        "To rigorously evaluate GovDoc-RAG across summarisation, cross-lingual retrieval, trust abstention, and factual grounding, we curated an extensive multi-source "
        "experimental corpus comprising official Indian government notifications and standardized benchmarks. Table 4.1 details dataset statistics, linguistic pairings, "
        "and evaluation partitions."
    )

    ds_cols = [1.2, 1.8, 0.8, 0.8, 0.8, 1.2]
    ds_hdrs = ["Pair / Subset", "Domain / Source", "Train", "Val", "Test", "Licence"]
    ds_data = [
        ["en-en", "English Notifications (PMIndiaSum)", "3,000", "200", "200", "CC-BY 4.0"],
        ["hi-hi", "Hindi Notifications (PMIndiaSum)", "3,000", "200", "200", "CC-BY 4.0"],
        ["kn-kn", "Kannada Notifications (PMIndiaSum)", "3,000", "200", "200", "CC-BY 4.0"],
        ["en-hi", "Bilingual Cross-Lingual (PIB Gazette)", "3,000", "200", "200", "Gov Data Open"],
        ["en-kn", "Bilingual Cross-Lingual (GoK Circulars)", "3,000", "200", "200", "Gov Data Open"],
        ["Benchmark A", "Cross-Lingual Retrieval Queries (en/hi/kn)", "--", "200", "1,500", "Curated"],
        ["Benchmark B", "Grounded QA Circulars (PIB/GoK)", "--", "--", "40", "Human Verified"],
    ]
    ds_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]
    
    t_ds = doc.add_table(rows=len(ds_data) + 1, cols=6)
    style_academic_table(t_ds, ds_cols, ds_hdrs, ds_data, ds_aligns, caption="Table 4.1: Dataset Statistics across Languages, Domains, and Evaluation Splits.")

    add_p(
        "The primary summarisation dataset is derived from the PMIndiaSum corpus released by Urlana et al. (2024), containing authentic press releases published by the "
        "Press Information Bureau of India. We curated balanced 3,000-example training splits, 200-example validation splits, and 200-example test splits for monolingual "
        "(en-en, hi-hi, kn-kn) and cross-lingual (en-hi, en-kn) pairs. For retrieval benchmarking, we constructed Benchmark A: an exhaustive evaluation suite of 1,500 test "
        "queries evenly divided across English, Hindi (Devanagari), Kannada (native script), Hinglish, and Kanglish. For factual grounding and hallucination evaluation, "
        "we established Benchmark B, consisting of 40 full-length, authentic government gazettes paired with gold-standard factual question-answer pairs and corrupted entity test sets."
    )

    add_sec_heading("4.2", "System Components, Parameter Scales, and Quantisation")
    add_p(
        "GovDoc-RAG integrates heterogeneous open-weights model backbones optimized for edge memory co-residency. Table 4.2 presents complete architectural specifications, "
        "parameter scales, precision formats, and operational hyperparameters for all deployed subsystems."
    )

    comp_cols = [1.3, 1.4, 0.8, 1.1, 2.0]
    comp_hdrs = ["System Role", "Model Backbone", "Params", "Precision / Q", "Key Hyperparameters"]
    comp_data = [
        ["Dense Retriever", "BAAI/bge-m3", "568 M", "FP16 (Ollama)", "Cosine Metric, 1024-dim, Max 512 tokens"],
        ["Transliteration", "IndicXlit", "11 M", "FP32 (CPU)", "Aksharantar Roman -> Native, Beam 1"],
        ["Monolingual Summariser", "ai4bharat/IndicBART", "244 M", "FP16 + LoRA", "r=16, alpha=32, lr=5e-4, Beam 4, no-repeat-3"],
        ["Neural Translator", "IndicTrans2-dist", "212 M", "FP16 (Rotary)", "Beam 4, Script Unification, Max 256 tokens"],
        ["Grounded Answer SLM", "Qwen 3 4B Instruct", "4.0 B", "Q4_K_M (GGUF)", "Context 4096, Temp 0.1, Rep-Penalty 1.1"],
        ["Trust Decision Gate", "Dual-Threshold Cosine", "--", "Deterministic", "tau*=0.400, tau_hi=0.65, tau_lo=0.30"],
        ["Grounding Verifier", "Entity & Numeric Anchor", "--", "Deterministic", "Exact regex date/currency match, [Page X] check"],
    ]
    comp_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT]

    t_comp = doc.add_table(rows=len(comp_data) + 1, cols=5)
    style_academic_table(t_comp, comp_cols, comp_hdrs, comp_data, comp_aligns, caption="Table 4.2: System Components, Model Parameter Scales, Precision, and Hyperparameters.")

    add_sec_heading("4.3", "Engineering Milestones and Workflow (M1–M11)")
    add_p(
        "The project was executed through eleven sequential engineering milestones (M1 through M11), each subjected to strict empirical acceptance gates:\n"
        "• Milestone M1 (Layout Ingestion): Built PyMuPDF layout-aware chunking engine, achieving 100% section-boundary preservation and metadata extraction on test circulars;\n"
        "• Milestone M2 (Dense Indexing): Implemented persistent ChromaDB collection with BGE-M3 embeddings and cosine distance metric conversion;\n"
        "• Milestone M3 (Query Bridge): Integrated IndicXlit Aksharantar transliteration, eliminating zero-shot script vocabulary gaps for Hinglish and Kanglish;\n"
        "• Milestone M4 (Trust Gate): Calibrated dual-threshold decision rules (tau_hi=0.65, tau_lo=0.30, tau*=0.400) on 500 calibration pairs;\n"
        "• Milestone M5 (Grounded Generation): Deployed Qwen 3 4B Instruct via local Ollama GGUF Q4_K_M, enforcing strict [Page X] citation formatting;\n"
        "• Milestone M6 (Hallucination Verifier): Implemented deterministic entity-level fact validator, achieving 100% corruption flag rate on Benchmark B;\n"
        "• Milestone M7 (LoRA Summarisation): Fine-tuned IndicBART on PMIndiaSum using Hugging Face PEFT, training for 5 epochs;\n"
        "• Milestone M8 (Edge Memory Guard): Engineered mutual-exclusion VRAM manager, guaranteeing peak consumption <= 3,219.0 MB on 4GB GPU;\n"
        "• Milestone M9 (Streamlit Interface): Built responsive web application with selective/progressive ingestion and real-time citation highlights;\n"
        "• Milestone M10 (Empirical Evaluation): Executed automated benchmarking scripts across 1,500 queries, logging metrics into results/raw/*; and\n"
        "• Milestone M11 (Figures & Tables): Developed make_figures.py and make_tables.py to regenerate all publication-grade vector graphics and LaTeX/Word tables."
    )

    add_sec_heading("4.4", "Hardware and Experimental Environment")
    add_p(
        "All model training, indexing, benchmarking, and latency profiling were executed locally on a single commodity developer laptop without external cloud servers:\n"
        "• Graphics Processing Unit (GPU): NVIDIA GeForce RTX 2050 / RTX 3050 Laptop GPU with 4.0 GB GDDR6 Dedicated VRAM (Compute Capability 8.6);\n"
        "• Central Processing Unit (CPU): 12th Gen Intel Core i5-12450H (8 Cores, 12 Threads @ 2.50 GHz base, 4.40 GHz boost);\n"
        "• Host Memory (RAM): 16.0 GB DDR4 @ 3200 MHz;\n"
        "• Operating System & Stack: Microsoft Windows 11 Home 64-bit, Python 3.13.0, PyTorch 2.6.0 with CUDA 12.4 runtime, Hugging Face Transformers 4.49.0, PEFT 0.14.0, Ollama 0.5.11, ChromaDB 0.6.3, and PyMuPDF 1.25.3."
    )

    add_sec_heading("4.5", "Training Dynamics of IndicBART LoRA Adapter")
    add_p(
        "The LoRA fine-tuning of ai4bharat/IndicBART was conducted on the 3,000-example PMIndiaSum training split using AdamW optimizer (learning rate = 5e-4, "
        "linear warmup for 500 steps, weight decay = 0.01) with a per-device batch size of 8 and gradient accumulation steps of 2 (effective batch size = 16) "
        "over 5 complete epochs. Figure 4.1 illustrates the training loss and validation loss trajectories."
    )

    # Embed Figure 3: Training Curves
    add_fig_image(
        "results/figs_png/fig3_training_curves.png",
        "4.1",
        "IndicBART LoRA Adapter Training and Validation Loss Curves over 5 Epochs showing Smooth Convergence and Absence of Overfitting.",
        width=Inches(4.8)
    )

    add_p(
        "The training loss drops rapidly from an initial value of 2.84 to 1.42 by the end of Epoch 2, eventually converging to 1.12 at Epoch 5. "
        "The validation loss decreases smoothly from 2.68 to 1.34, plateauing between Epoch 4 and 5 without any divergence or overfitting. "
        "By restricting trainable parameters to 0.48% of the model backbone, LoRA prevents catastrophic forgetting of general Indic linguistic structures "
        "while successfully specializing the self-attention projections toward formal bureaucratic press release syntax."
    )

    print("Chapter 4 built.")
