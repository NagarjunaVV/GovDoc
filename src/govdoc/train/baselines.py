"""Evaluation baselines for summarization: B1 (IndicBART zero-shot), B2 (Qwen zero-shot), B3 (Lead-1)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import from_devanagari, to_devanagari
from govdoc.eval.rouge_indic import compute_corpus_rouge_l, compute_rouge_l
from govdoc.generate.llm_client import LLMClient
from govdoc.logging_utils import append_log, save_experiment_record

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def extract_lead_1(text: str, lang: str) -> str:
    """Extract first sentence as Lead-1 summary."""
    text = text.strip()
    if not text:
        return ""
    # Split on full stop, Hindi purna viram (।), or newline
    sentences = re.split(r"[।\.\n\?\!]", text)
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 5:
            # Re-attach terminal punctuation
            punct = "।" if lang == "hi" else "."
            return s_clean + punct
    return text[:100]


def run_lead1_baseline(
    test_items: List[Dict[str, Any]],
    directions: Optional[List[str]] = None,
    n_samples: int = 200,
) -> Dict[str, Any]:
    """Run Lead-1 baseline on monolingual test sets (en-en, hi-hi, kn-kn)."""
    if directions is None:
        directions = ["en-en", "hi-hi", "kn-kn"]

    results = {}
    for direction in directions:
        if direction not in ("en-en", "hi-hi", "kn-kn"):
            continue

        dir_items = [it for it in test_items if it["direction"] == direction][:n_samples]
        if not dir_items:
            continue

        preds, refs = [], []
        records = []
        for it in dir_items:
            pred = extract_lead_1(it["text"], it["src_lang"])
            ref = it["summary"]
            preds.append(pred)
            refs.append(ref)
            records.append({
                "group_id": it["group_id"],
                "direction": direction,
                "prediction": pred,
                "reference": ref,
                "rouge_l": compute_rouge_l(pred, ref)["f1"],
            })

        metrics = compute_corpus_rouge_l(preds, refs)
        results[direction] = {
            "n": len(preds),
            "rouge_l": metrics,
        }

        # Save to results/raw/baseline_lead1_<direction>.jsonl
        out_file = resolve_path(f"results/raw/baseline_lead1_{direction}.jsonl")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def run_qwen_baseline(
    test_items: List[Dict[str, Any]],
    directions: Optional[List[str]] = None,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Run Qwen2.5-1.5B zero-shot baseline via local LLM server."""
    if directions is None:
        directions = ["en-en", "en-hi", "en-kn", "hi-hi", "kn-kn"]

    client = LLMClient()
    if not client.health_check():
        print("WARNING: Local LLM server is not responding. Skipping Qwen baseline.")
        return {}

    lang_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    results = {}

    for direction in directions:
        dir_items = [it for it in test_items if it["direction"] == direction][:n_samples]
        if not dir_items:
            continue

        print(f"Running Qwen baseline on {direction} ({len(dir_items)} samples)...")
        preds, refs = [], []
        records = []
        for it in dir_items:
            tgt_lang_name = lang_names.get(it["tgt_lang"], it["tgt_lang"])
            # Truncate text context for prompt
            prompt = (
                f"Summarise the following government article in {tgt_lang_name} in one single headline or short sentence.\n"
                f"Article: {it['text'][:1200]}\n\nHeadline:"
            )
            try:
                res = client.chat(prompt, max_tokens=64, temperature=0.1)
                pred = res["content"].strip().split("\n")[0]
            except Exception as e:
                pred = f"[ERROR: {e}]"

            ref = it["summary"]
            preds.append(pred)
            refs.append(ref)
            records.append({
                "group_id": it["group_id"],
                "direction": direction,
                "prediction": pred,
                "reference": ref,
                "rouge_l": compute_rouge_l(pred, ref)["f1"],
            })

        metrics = compute_corpus_rouge_l(preds, refs)
        results[direction] = {
            "n": len(preds),
            "rouge_l": metrics,
        }

        out_file = resolve_path(f"results/raw/baseline_qwen_{direction}.jsonl")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def run_indicbart_pretrained_baseline(
    test_items: List[Dict[str, Any]],
    directions: Optional[List[str]] = None,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Run IndicBART pre-trained baseline without fine-tuning."""
    if directions is None:
        directions = ["en-en", "en-hi", "en-kn", "hi-hi", "kn-kn"]

    cfg = load_config()
    model_name = cfg["models"]["summarizer"]["base"]

    print(f"Loading pre-trained IndicBART ({model_name}) for B1 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    results = {}
    for direction in directions:
        dir_items = [it for it in test_items if it["direction"] == direction][:n_samples]
        if not dir_items:
            continue

        print(f"Running IndicBART pre-trained on {direction} ({len(dir_items)} samples)...")
        preds, refs = [], []
        records = []
        for it in dir_items:
            tgt_lang = it["tgt_lang"]
            inputs = tokenizer(
                it["src_text"],
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            inputs.pop("token_type_ids", None)

            # Decoder start token ID for target language tag
            tgt_tag = f"<2{tgt_lang}>"
            decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_tag)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    decoder_start_token_id=decoder_start_token_id,
                    max_new_tokens=64,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    encoder_no_repeat_ngram_size=3,
                )

            decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            # If target language is Kannada, convert from Devanagari back to Kannada
            if tgt_lang == "kn":
                final_pred = from_devanagari(decoded, "kn")
            else:
                final_pred = decoded

            ref = it["summary"]
            preds.append(final_pred)
            refs.append(ref)
            records.append({
                "group_id": it["group_id"],
                "direction": direction,
                "prediction": final_pred,
                "reference": ref,
                "rouge_l": compute_rouge_l(final_pred, ref)["f1"],
            })

        metrics = compute_corpus_rouge_l(preds, refs)
        results[direction] = {
            "n": len(preds),
            "rouge_l": metrics,
        }

        out_file = resolve_path(f"results/raw/baseline_indicbart_{direction}.jsonl")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run summarization baselines (B1, B2, B3).")
    parser.add_argument("--model", type=str, default="lead1", choices=["lead1", "qwen", "indicbart", "all"])
    parser.add_argument("--n", type=int, default=50, help="Number of test samples per direction.")
    parser.add_argument("--direction", type=str, default="all", help="Specific direction or 'all'.")
    args = parser.parse_args()

    cfg = load_config()
    test_path = resolve_path(cfg.get("paths", {}).get("data_processed", "data/processed")) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test set not found at {test_path}. Run group_split.py first.")

    with open(test_path, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f if line.strip()]

    target_dirs = None if args.direction == "all" else [args.direction]

    print("=" * 65)
    print(f"Running Baselines (Model: {args.model}, n={args.n}, Directions: {args.direction})")
    print("=" * 65)

    all_results = {}
    if args.model in ("lead1", "all"):
        print("\n--- [B3] Lead-1 Baseline ---")
        lead1_res = run_lead1_baseline(test_items, directions=target_dirs, n_samples=args.n)
        all_results["lead1"] = lead1_res
        for d, res in lead1_res.items():
            print(f"  {d}: ROUGE-L F1 = {res['rouge_l']['f1']:.4f} (n={res['n']})")

    if args.model in ("qwen", "all"):
        print("\n--- [B2] Qwen2.5-1.5B Zero-Shot Baseline ---")
        qwen_res = run_qwen_baseline(test_items, directions=target_dirs, n_samples=min(args.n, 30))
        all_results["qwen"] = qwen_res
        for d, res in qwen_res.items():
            print(f"  {d}: ROUGE-L F1 = {res['rouge_l']['f1']:.4f} (n={res['n']})")

    if args.model in ("indicbart", "all"):
        print("\n--- [B1] IndicBART Pre-trained Zero-Shot Baseline ---")
        ib_res = run_indicbart_pretrained_baseline(test_items, directions=target_dirs, n_samples=args.n)
        all_results["indicbart"] = ib_res
        for d, res in ib_res.items():
            print(f"  {d}: ROUGE-L F1 = {res['rouge_l']['f1']:.4f} (n={res['n']})")

    append_log(f"Baselines run complete for {args.model}: results saved to results/raw/", module="M2:BASELINES")
    print("\nBaseline runs complete. Outputs stored in results/raw/ and logged to results/LOG.md.")


if __name__ == "__main__":
    main()
