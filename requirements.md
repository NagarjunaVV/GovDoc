# requirements.md

Companion to `plan.md`. "MUST / SHOULD / MAY" follow the P0 / P1 / P2 tiers in the plan.

---

## 1. Hardware

| Role | Requirement |
|---|---|
| Local runtime (the demo) | Modern x64 or Apple Silicon CPU, **16 GB RAM**, GPU with **4 GB VRAM** (CUDA), 20 GB free disk |
| Training and bulk embedding | Free **Kaggle** (T4 x2 or P100, long sessions) or **Colab** T4. Kaggle preferred for session stability |
| If you do not own a 4 GB GPU | Simulate: `torch.cuda.set_per_process_memory_fraction(4/total_gb)` for PyTorch parts; llama.cpp memory must be read from `nvidia-smi`/pynvml. Say in the paper that the cap was simulated |

## 2. Software

Python 3.10 or 3.11. After the first successful install run `pip freeze > requirements.lock` and treat the lock file as the reproducibility record.

| Area | Packages / tools | Notes |
|---|---|---|
| Core ML | `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `sentencepiece`, `sacrebleu` | IndicBART uses `AutoTokenizer(use_fast=False, keep_accents=True, do_lower_case=False)`; pin the transformers version once it works |
| Quantisation (optional QLoRA ablation) | `bitsandbytes` | Linux/Kaggle only is simplest; skip on Windows if painful |
| Embeddings | `sentence-transformers` or `FlagEmbedding` for `BAAI/bge-m3` | Dense vectors only; normalise |
| Vector DB | `chromadb` (FAISS fallback) | cosine space |
| PDF | `pymupdf` | Keep a page to text map |
| LLM runtime | prebuilt **llama.cpp `llama-server`** (CUDA) or **Ollama** | Call over local HTTP (OpenAI-compatible). Avoid compiling `llama-cpp-python` on Windows |
| Indic NLP | `indic-nlp-library` (script conversion), `ai4bharat-transliteration` (IndicXlit) | Script conversion is mandatory for IndicBART; IndicXlit may need extra dependencies |
| Optional translation | `IndicTransToolkit` + `ai4bharat/indictrans2-en-indic-dist-200M` | Only for P2 baselines; model access may require accepting terms on Hugging Face |
| Evaluation | `rouge-score` (custom tokenizer, see 5.4), `bert-score`, `sacrebleu` (chrF++), `numpy`, `scipy`, `scikit-learn` | |
| UI | `streamlit`, `plotly` or `matplotlib` | No CDN assets |
| Monitoring | `pynvml` (GPU memory), `psutil` | |
| Plotting for the paper | `matplotlib`, `pandas`, `seaborn` optional | |

Offline flags for runtime: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, plus `ANONYMIZED_TELEMETRY=False` for Chroma and `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`.

## 3. Models and VRAM budget

Numbers are estimates; **measure in M10 and report measured values**.

| Model | Role | Size (approx.) | Where it runs at query time |
|---|---|---|---|
| `BAAI/bge-m3` (about 568M parameters) | Multilingual dense retrieval | about 1.1 GB fp16 | CPU at query time; GPU during bulk ingestion |
| Qwen2.5-1.5B-Instruct, Q4_K_M GGUF | Grounded answer generation | about 1.0 GB weights + about 0.1 GB KV at ctx 4096 | GPU via llama-server |
| `ai4bharat/IndicBART` (about 244M) + merged LoRA | Summarisation | about 0.5 GB fp16 + activations | GPU on demand; Qwen unloaded or kept small |
| IndicXlit (about 11M) | Roman to native transliteration | tiny | CPU |
| Optional: Qwen3-1.7B Q4_K_M | Alternative generator (Kannada quality) | about 1.1 GB | swap-in for ablation |

Residency modes (record which mode every reported number used):

| Mode | Resident on GPU | Expected peak |
|---|---|---|
| `qa` | Qwen Q4 (+ CUDA context) | about 1.5-2 GB |
| `summarise` | IndicBART fp16 only | about 1.0-1.5 GB |
| `ingest` | BGE-M3 fp16 only | about 1.5-2 GB |
| `all-resident` (not recommended) | all three + two CUDA contexts | likely above 4 GB; include once to justify the staged policy |

## 4. Functional requirements

| ID | Tier | Requirement |
|---|---|---|
| FR-01 | MUST | Upload one or more PDFs (up to about 100 pages each), extract text with page numbers |
| FR-02 | MUST | Extraction quality gate: warn the user when a PDF yields garbled or empty text |
| FR-03 | MUST | Chunk (about 300 BGE-M3 tokens, 50 overlap) and index locally with metadata |
| FR-04 | MUST | Accept questions in English, Hindi, Kannada; SHOULD accept Hinglish/Kanglish (romanised) |
| FR-05 | MUST | Retrieve top-k chunks; return similarity scores in cosine units |
| FR-06 | MUST | Trust boundary: below threshold the system returns "Information not found" and **does not call the LLM** |
| FR-07 | MUST | Grounded answer with numbered citations and page references; evidence passages shown verbatim |
| FR-08 | SHOULD | Deterministic verifier flags answers containing numbers/entities absent from evidence |
| FR-09 | MUST | Digest summarisation of a whole document to a target language (hi / kn / en) |
| FR-10 | SHOULD | Side-by-side Base vs Fine-tuned summary panel for a chosen section |
| FR-11 | MUST | System tab: live VRAM gauge, per-stage latency, active model residency mode |
| FR-12 | SHOULD | Results tab renders benchmark JSON as charts (summarisation table, Recall@K, threshold sweep) |
| FR-13 | MUST | All features work with the network disabled |
| FR-14 | SHOULD | Threshold slider (default = calibrated value) with a visible explanation of what it changes |

### Dashboard spec (Streamlit)

1. **Upload and Index:** file uploader, extraction-quality report per document, progress bar, chunk count, index size.
2. **Ask:** language mode selector (Auto / English / Hindi / Kannada / Hinglish / Kanglish), question box, answer card (text, citation chips, verifier badge), evidence expanders (page, score), confidence meter (score vs threshold), explicit "Information not found" state.
3. **Summarise:** target language selector, length (number of gists), Generate digest, per-section gists with page refs, Base vs Fine-tuned toggle.
4. **System and Results:** VRAM bar (pynvml), latency table (embed / retrieve / generate / verify), benchmark charts loaded from `results/raw`, "offline check" status.

## 5. Experimental and implementation requirements

### 5.1 Reproducibility
- Seeds fixed (`42` default) and logged; seeds applied to Python, NumPy, PyTorch.
- Every experiment writes a JSON with `{experiment_id, date, git_commit, config_hash, seed, n_samples, metrics, notes}`.
- `results/LOG.md` is append-only: one line per run (what, where, wall time).
- Test sets are frozen after M6 and never used for tuning (thresholds and hyperparameters use validation only).

### 5.2 Fine-tuning recipe (starting values, adjust on validation)

| Setting | Value |
|---|---|
| Base | `ai4bharat/IndicBART` |
| Method | LoRA, r = 32, alpha = 64, dropout 0.05, targets `q_proj,k_proj,v_proj,out_proj,fc1,fc2` |
| Precision | fp16 (T4/P100 have no bf16) |
| Directions | en-en, en-hi, en-kn, hi-hi, kn-kn (multi-task, one model) |
| Data | up to 3,000 pairs per direction; src max 512 tokens, tgt max 64 |
| Optimiser | AdamW, LR 3e-4, 5% warmup, linear decay, label smoothing 0.1 |
| Epochs / batch | 3 epochs, effective batch 32 (grad accumulation) |
| Decoding | beam 4, `no_repeat_ngram_size=3`, `encoder_no_repeat_ngram_size=3`, max_new_tokens 64 |
| Time box | stop at 100 minutes of wall time even if epochs are unfinished; keep the best-val checkpoint |
| Ablations (P2) | full fine-tune; 4-bit QLoRA; LoRA rank 8 vs 32 |

Language tags and format must follow the model card: source `text </s> <2src>`; decoder start token `<2tgt>`; labels `summary </s>`. Kannada text is converted to Devanagari before tokenisation and converted back after decoding.

### 5.3 Retrieval and gate
- Chunk size about 300 tokens (BGE-M3 tokenizer), overlap 50, top-k 10.
- Score = max cosine over query views; document score = max over its chunks.
- Threshold calibration on the **validation** queries: choose tau at the point that maximises balanced accuracy (or meets a target false-accept rate of at most 5%), then freeze it. Report AUROC and the full sweep.
- Three-way decision (SHOULD): `ANSWER`, `EVIDENCE_ONLY`, `ABSTAIN`.

### 5.4 Metrics

| Task | Metrics | Implementation notes |
|---|---|---|
| Summarisation | ROUGE-L F1 (and R-1/R-2), BERTScore F1, chrF++ | **ROUGE must use a Unicode-aware tokenizer** (split on whitespace after NFC normalisation and punctuation stripping, keep Devanagari/Kannada letters and combining marks). Verify with a self-test: identical strings must score 1.0 |
| Significance | Paired bootstrap, 1,000 resamples, 95% CI | on the same test items |
| Retrieval | Recall@1/3/5/10, MRR | doc-level (max-pool chunks) for Benchmark A; chunk-level for Benchmark B |
| Abstention | AUROC, false-accept rate at fixed answerable recall, abstention precision/recall, full threshold sweep | report OOD and near-miss unanswerables separately |
| Answer quality (Benchmark B subset) | Correct / Partially / Incorrect (human), verifier pass rate | at least 40 items, two raters if possible |
| Human summary evaluation (P1) | 1-5 adequacy and fluency, 20 samples per language | Hindi and Kannada speakers on the team |
| Resources | peak VRAM per stage (pynvml), P50/P95 latency per stage, Qwen tokens/s, disk size | 50 queries + 10 summaries |

### 5.5 Result files (all under `results/raw/`)

| File | Content |
|---|---|
| `summarization.json` | per system x direction: n, rouge1/2/L, bertscore_f1, chrf, CI bounds |
| `summ_samples.jsonl` | input snippet, reference, system outputs (for qualitative table) |
| `training_curve.csv` | step, train_loss, val_loss, val_rougeL |
| `retrieval_A.json` | Recall@K, MRR per query language and per condition (with/without transliteration) |
| `retrieval_B.json` | same for the real-PDF benchmark |
| `scores_{answerable,ood,nearmiss}.jsonl` | top-1 score per query (feeds histograms) |
| `abstain.json` | sweep table, AUROC, chosen tau, FAR, recall, abstention precision |
| `verifier.json` | pass/flag rates, flagged examples |
| `profile.json` | per-stage latency percentiles, peak VRAM per mode, tokens/s |
| `vram_trace.jsonl` | timestamped VRAM readings with stage labels |
| `offline_check.json` | pass/fail per feature with network disabled |

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | Peak GPU memory at most 4.0 GB in every stage (measured by `nvidia-smi`/pynvml, not estimated) |
| NFR-02 | No network access at runtime; no telemetry |
| NFR-03 | Provisional latency targets: single question under 10 s P95 on the 4 GB GPU, digest of a 20-page PDF under 3 min. Measure first, then state the real numbers in the paper |
| NFR-04 | Deterministic gate: same query + same index gives the same decision |
| NFR-05 | One command to regenerate every figure and table from `results/raw` |
| NFR-06 | Code is modular: each module has a CLI entry point and at least one test |

## 7. Acceptance tests (run before each gate)

| Test | Pass condition |
|---|---|
| `test_script_roundtrip` | Kannada to Devanagari to Kannada returns the original on at least 99% of 200 samples (note and exclude exceptions such as unsupported characters) |
| `test_rouge_unicode` | identical Hindi/Kannada strings score 1.0; unrelated strings score below 0.2 |
| `test_distance_to_cosine` | Chroma distance converted to cosine equals the direct dot product of normalised vectors within 1e-4 |
| `test_no_leakage` | no article group appears in more than one split |
| `test_gate_abstains` | off-topic query returns `ABSTAIN`; LLM call counter stays at 0 |
| `test_verifier` | answer with an invented amount is flagged; answer with evidence amount passes (also with Devanagari numerals) |
| `test_offline` | full flow runs with the network off |
| `test_vram` | peak VRAM at most 4.0 GB in `qa` and `summarise` modes |
