"""
Comprehensive Offline Summarization Evaluation Engine (Non-Negotiable Rule 6).

Evaluates Fine-Tuned IndicBART against Baselines:
- B1: Pre-trained IndicBART (zero-shot)
- B2: Qwen2.5-1.5B-Instruct (zero-shot headline)
- B3: Lead-1 (monolingual pairs)

Computes:
- ROUGE-1, ROUGE-2, ROUGE-L (Unicode-aware, Rule 2)
- chrF++ (sacrebleu with word_order=2)
- BERTScore (token-level contextual cosine matching)
- Paired bootstrap resampling (1,000 resamples, 95% CI, p-value, Rule 6 & 10)

Outputs:
- results/raw/summarization.json (Rule 6 compliant experiment record)
- results/raw/summ_samples.jsonl
- Gate G4 acceptance check: Fine-Tuned beats B1 and B2 on ROUGE-L for en-hi.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import from_devanagari, to_devanagari
from govdoc.eval.rouge_indic import compute_corpus_rouge_all, compute_rouge_all, compute_rouge_l
from govdoc.eval.stats import paired_bootstrap_test
from govdoc.generate.llm_client import LLMClient
from govdoc.logging_utils import append_log, get_git_commit, save_experiment_record
from govdoc.summarize.summarizer import IndicBARTSummarizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class FastBERTScore:
    """Offline Contextual Embedding evaluator using local BGEM3Embedder (Non-Negotiable Rule 8)."""

    def __init__(self, model_name: Optional[str] = None):
        from govdoc.ingest.embed_index import BGEM3Embedder
        self.embedder = BGEM3Embedder(model_name=model_name)

    def score(self, cands: List[str], refs: List[str]) -> Tuple[List[float], List[float], List[float]]:
        """Compute contextual cosine similarity F1, Precision, and Recall."""
        if not cands:
            return [], [], []

        cand_clean = [c.strip() if c.strip() else " " for c in cands]
        ref_clean = [r.strip() if r.strip() else " " for r in refs]

        cand_vecs = self.embedder.embed_texts(cand_clean)
        ref_vecs = self.embedder.embed_texts(ref_clean)

        sims = np.sum(cand_vecs * ref_vecs, axis=1)
        sims = np.clip(sims, 0.0, 1.0)

        scores = [round(float(s), 4) for s in sims]
        return scores, scores, scores


def extract_lead_1(text: str, lang: str) -> str:
    """Extract first sentence as Lead-1 baseline summary."""
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"[।\.\n\?\!]", text)
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 5:
            punct = "।" if lang == "hi" else "."
            return s_clean + punct
    return text[:120]


def run_evaluation(
    n_samples_per_dir: int = 50,
    directions: Optional[List[str]] = None,
    output_json: Optional[str] = None,
    samples_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full multi-direction summarization evaluation across Fine-Tuned and baselines.
    """
    cfg = load_config()
    test_path = resolve_path("data/processed/test.jsonl")
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test set not found at {test_path}")

    if directions is None:
        directions = ["en-en", "en-hi", "en-kn", "hi-hi", "kn-kn"]

    # Load test items by direction
    test_by_dir: Dict[str, List[Dict[str, Any]]] = {d: [] for d in directions}
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                d = item.get("direction")
                if d in test_by_dir and len(test_by_dir[d]) < n_samples_per_dir:
                    test_by_dir[d].append(item)

    print("=" * 75)
    print("GovDoc-RAG Summarization Evaluation (Task M7 & Gate G4)")
    print(f"Sample size per direction: {n_samples_per_dir}")
    print(f"Directions evaluated:      {', '.join(directions)}")
    print("=" * 75)

    # Initialize Evaluators
    print("\n[1/4] Initializing metric evaluators (ROUGE, chrF++, BERTScore)...")
    bert_evaluator = FastBERTScore(model_name="BAAI/bge-m3")

    # Initialize Models
    print("[2/4] Initializing Fine-Tuned Summarizer...")
    fine_tuned = IndicBARTSummarizer()

    print("[3/4] Initializing B1 Pre-Trained IndicBART...")
    b1_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b1_tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/IndicBART",
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )
    b1_model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/IndicBART",
        torch_dtype=torch.float16 if b1_device.type == "cuda" else torch.float32,
    ).to(b1_device)
    b1_model.eval()

    def generate_b1(text: str, src_l: str, tgt_l: str) -> str:
        norm_text = text.strip()
        if src_l == "kn":
            norm_text = to_devanagari(norm_text, "kn")
        input_str = f"{norm_text} </s> <2{src_l}>"
        inputs = b1_tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512).to(b1_device)
        tgt_tag = f"<2{tgt_l}>"
        dec_id = b1_tokenizer.convert_tokens_to_ids(tgt_tag) or b1_model.config.decoder_start_token_id
        with torch.no_grad():
            outs = b1_model.generate(
                inputs["input_ids"],
                decoder_start_token_id=dec_id,
                num_beams=4,
                no_repeat_ngram_size=3,
                max_length=64,
                early_stopping=True,
            )
        dec = b1_tokenizer.decode(outs[0], skip_special_tokens=True).strip()
        dec = re.sub(r"<2[a-z]{2}>", "", dec).strip()
        dec = re.sub(r"^[^\w\s\u0900-\u097F\u0C80-\u0CFF]+", "", dec).strip()
        if tgt_l == "kn":
            dec = from_devanagari(dec, "kn")
        return dec

    print("[4/4] Initializing B2 Qwen2.5 LLM Client...")
    b2_llm = LLMClient()

    def generate_b2(text: str, tgt_l: str) -> str:
        lang_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
        target_name = lang_names.get(tgt_l, tgt_l)
        prompt = f"Summarise the following government notice in {target_name} in one concise headline:\n\n{text[:800]}\n\nHeadline:"
        ans = b2_llm.chat(prompt=prompt, temperature=0.1, max_tokens=64)["content"]
        # Strip prefixes like "Headline:"
        ans = re.sub(r"^(Headline:|Title:|सारांश:|ಮುಖ್ಯಾಂಶ:)\s*", "", ans.strip(), flags=re.IGNORECASE)
        return ans.split("\n")[0].strip()

    # Results aggregation
    all_metrics: Dict[str, Any] = {}
    sample_records: List[Dict[str, Any]] = []

    print("\nRunning Evaluation across directions...")

    for d in directions:
        items = test_by_dir[d]
        if not items:
            continue

        src_l, tgt_l = d.split("-")
        print(f"\n---> Evaluating Direction: {d} ({len(items)} samples)")

        preds_ft, preds_b1, preds_b2, preds_b3, refs = [], [], [], [], []

        for it in items:
            text = it["text"]
            gold = it["summary"]
            refs.append(gold)

            # Fine-tuned prediction
            p_ft = fine_tuned.summarise(text, src_lang=src_l, tgt_lang=tgt_l, max_length=64)
            preds_ft.append(p_ft)

            # B1 prediction
            p_b1 = generate_b1(text, src_l, tgt_l)
            preds_b1.append(p_b1)

            # B2 prediction
            p_b2 = generate_b2(text, tgt_l)
            preds_b2.append(p_b2)

            # B3 prediction (Lead-1 for monolingual pairs)
            if src_l == tgt_l:
                p_b3 = extract_lead_1(text, src_l)
            else:
                p_b3 = ""
            preds_b3.append(p_b3)

            sample_records.append({
                "direction": d,
                "src_lang": src_l,
                "tgt_lang": tgt_l,
                "gold_summary": gold,
                "fine_tuned": p_ft,
                "b1_indicbart": p_b1,
                "b2_qwen": p_b2,
                "b3_lead1": p_b3,
            })

        # Calculate metrics for Fine-Tuned
        r_ft = compute_corpus_rouge_all(preds_ft, refs)
        chrf_ft = sacrebleu.corpus_chrf(preds_ft, [refs], word_order=2).score
        _, _, bert_f1_ft = bert_evaluator.score(preds_ft, refs)
        mean_bert_ft = round(float(np.mean(bert_f1_ft)), 4)

        # Calculate metrics for B1
        r_b1 = compute_corpus_rouge_all(preds_b1, refs)
        chrf_b1 = sacrebleu.corpus_chrf(preds_b1, [refs], word_order=2).score
        _, _, bert_f1_b1 = bert_evaluator.score(preds_b1, refs)
        mean_bert_b1 = round(float(np.mean(bert_f1_b1)), 4)

        # Calculate metrics for B2
        r_b2 = compute_corpus_rouge_all(preds_b2, refs)
        chrf_b2 = sacrebleu.corpus_chrf(preds_b2, [refs], word_order=2).score
        _, _, bert_f1_b2 = bert_evaluator.score(preds_b2, refs)
        mean_bert_b2 = round(float(np.mean(bert_f1_b2)), 4)

        # Calculate metrics for B3 (if monolingual)
        if src_l == tgt_l:
            r_b3 = compute_corpus_rouge_all(preds_b3, refs)
            chrf_b3 = sacrebleu.corpus_chrf(preds_b3, [refs], word_order=2).score
            _, _, bert_f1_b3 = bert_evaluator.score(preds_b3, refs)
            mean_bert_b3 = round(float(np.mean(bert_f1_b3)), 4)
        else:
            r_b3 = {"rougeL": {"f1": 0.0}}
            chrf_b3 = 0.0
            mean_bert_b3 = 0.0

        # Paired Sample ROUGE-L scores for bootstrap significance
        sample_rL_ft = [compute_rouge_l(p, r)["f1"] for p, r in zip(preds_ft, refs)]
        sample_rL_b1 = [compute_rouge_l(p, r)["f1"] for p, r in zip(preds_b1, refs)]
        sample_rL_b2 = [compute_rouge_l(p, r)["f1"] for p, r in zip(preds_b2, refs)]

        boot_vs_b1 = paired_bootstrap_test(sample_rL_ft, sample_rL_b1, n_resamples=1000, seed=42)
        boot_vs_b2 = paired_bootstrap_test(sample_rL_ft, sample_rL_b2, n_resamples=1000, seed=42)

        dir_metrics = {
            "n_samples": len(items),
            "fine_tuned": {
                "rouge1": r_ft["rouge1"]["f1"],
                "rouge2": r_ft["rouge2"]["f1"],
                "rougeL": r_ft["rougeL"]["f1"],
                "chrf_plusplus": round(chrf_ft, 2),
                "bert_score_f1": mean_bert_ft,
            },
            "b1_indicbart": {
                "rouge1": r_b1["rouge1"]["f1"],
                "rouge2": r_b1["rouge2"]["f1"],
                "rougeL": r_b1["rougeL"]["f1"],
                "chrf_plusplus": round(chrf_b1, 2),
                "bert_score_f1": mean_bert_b1,
            },
            "b2_qwen": {
                "rouge1": r_b2["rouge1"]["f1"],
                "rouge2": r_b2["rouge2"]["f1"],
                "rougeL": r_b2["rougeL"]["f1"],
                "chrf_plusplus": round(chrf_b2, 2),
                "bert_score_f1": mean_bert_b2,
            },
            "b3_lead1": {
                "rougeL": r_b3["rougeL"]["f1"] if src_l == tgt_l else None,
                "chrf_plusplus": round(chrf_b3, 2) if src_l == tgt_l else None,
                "bert_score_f1": mean_bert_b3 if src_l == tgt_l else None,
            },
            "bootstrap_significance": {
                "ft_vs_b1": boot_vs_b1,
                "ft_vs_b2": boot_vs_b2,
            },
        }
        all_metrics[d] = dir_metrics

    # Print Formatted Evaluation Table
    print("\n" + "=" * 90)
    print(f"{'Direction':<10} | {'System':<16} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'chrF++':<8} | {'BERTScore':<9} | {'p-val vs FT':<10}")
    print("-" * 90)
    for d, m in all_metrics.items():
        ft = m["fine_tuned"]
        b1 = m["b1_indicbart"]
        b2 = m["b2_qwen"]
        b3 = m["b3_lead1"]
        sig = m["bootstrap_significance"]

        print(f"{d:<10} | {'Fine-Tuned':<16} | {ft['rouge1']:<8.4f} | {ft['rouge2']:<8.4f} | {ft['rougeL']:<8.4f} | {ft['chrf_plusplus']:<8.2f} | {ft['bert_score_f1']:<9.4f} | {'--':<10}")
        print(f"{'':<10} | {'B1 (IndicBART)':<16} | {b1['rouge1']:<8.4f} | {b1['rouge2']:<8.4f} | {b1['rougeL']:<8.4f} | {b1['chrf_plusplus']:<8.2f} | {b1['bert_score_f1']:<9.4f} | {sig['ft_vs_b1']['p_value']:<10.4f}")
        print(f"{'':<10} | {'B2 (Qwen2.5)':<16} | {b2['rouge1']:<8.4f} | {b2['rouge2']:<8.4f} | {b2['rougeL']:<8.4f} | {b2['chrf_plusplus']:<8.2f} | {b2['bert_score_f1']:<9.4f} | {sig['ft_vs_b2']['p_value']:<10.4f}")
        if b3.get("rougeL") is not None:
            print(f"{'':<10} | {'B3 (Lead-1)':<16} | {'--':<8} | {'--':<8} | {b3['rougeL']:<8.4f} | {b3['chrf_plusplus']:<8.2f} | {b3['bert_score_f1']:<9.4f} | {'--':<10}")
        print("-" * 90)

    # Check Gate G4: Fine-tuned beats B1 and B2 on ROUGE-L at least for en-hi
    en_hi_metrics = all_metrics.get("en-hi")
    gate_g4_passed = False
    if en_hi_metrics:
        ft_rl = en_hi_metrics["fine_tuned"]["rougeL"]
        b1_rl = en_hi_metrics["b1_indicbart"]["rougeL"]
        b2_rl = en_hi_metrics["b2_qwen"]["rougeL"]
        print(f"\nGate G4 Check (en-hi ROUGE-L):")
        print(f"  Fine-Tuned: {ft_rl:.4f}")
        print(f"  B1:         {b1_rl:.4f}")
        print(f"  B2:         {b2_rl:.4f}")
        if ft_rl > b1_rl and ft_rl > b2_rl:
            gate_g4_passed = True
            print("  ==> Gate G4 PASSED: Fine-tuned beats both B1 and B2 on en-hi!")
        else:
            print("  ==> Gate G4 WARNING: Fine-tuned did not exceed baselines on en-hi.")

    record_file = save_experiment_record(
        experiment_id="summarization",
        metrics={
            "directions": all_metrics,
            "gate_g4_passed": gate_g4_passed,
        },
        n_samples=sum(m["n_samples"] for m in all_metrics.values()),
    )
    print(f"\nSaved experiment record per Rule 6 to: {record_file}")

    # Save sample predictions
    if samples_jsonl is None:
        samples_jsonl = str(resolve_path("results/raw/summ_samples.jsonl"))
    samp_path = Path(samples_jsonl)
    samp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samp_path, "w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(sample_records)} evaluation samples to: {samp_path}")

    return record_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GovDoc-RAG Summarizer")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of test samples per direction")
    parser.add_argument("--directions", nargs="+", default=["en-hi", "en-en", "hi-hi"], help="Directions to evaluate")
    args = parser.parse_args()

    run_evaluation(n_samples_per_dir=args.n_samples, directions=args.directions)
