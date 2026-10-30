"""Group-aware dataset splitter and IndicBART formatter (Non-Negotiable Rules 1 & 5)."""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.data.pmindia_loader import CONFIGS, load_all_pmindia
from govdoc.data.script_unify import to_devanagari
from govdoc.logging_utils import append_log

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_group_split(group_id: str) -> str:
    """Deterministically assign a group_id to train, val, or test using stable hashing."""
    h = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"


def build_url_group_mappings(all_raw: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build hi_url -> en_url and kn_url -> en_url mappings from cross-lingual splits."""
    hi_to_en: Dict[str, str] = {}
    kn_to_en: Dict[str, str] = {}

    # From english-hindi
    for split_df in all_raw["english-hindi"].values():
        for _, row in split_df.iterrows():
            en_url = str(row["source_url"]).strip()
            hi_url = str(row["target_url"]).strip()
            if en_url and hi_url:
                hi_to_en[hi_url] = en_url

    # From english-kannada
    for split_df in all_raw["english-kannada"].values():
        for _, row in split_df.iterrows():
            en_url = str(row["source_url"]).strip()
            kn_url = str(row["target_url"]).strip()
            if en_url and kn_url:
                kn_to_en[kn_url] = en_url

    return hi_to_en, kn_to_en


def format_indicbart_fields(
    text: str,
    summary: str,
    src_lang: str,
    tgt_lang: str,
) -> Tuple[str, str, str]:
    """Format input and target fields strictly according to IndicBART specification.
    
    Adheres strictly to Non-Negotiable Rule 1:
    - Input format: "<text> </s> <2src>"
    - Decoder start: "<2tgt>"
    - Labels: "<summary> </s>"
    - Non-Devanagari text (Kannada) converted to Devanagari.
    """
    # Unify scripts to Devanagari if Kannada
    src_unified = to_devanagari(text, src_lang)
    tgt_unified = to_devanagari(summary, tgt_lang)

    # IndicBART language tags
    src_tag = f"<2{src_lang}>"
    tgt_tag = f"<2{tgt_lang}>"

    src_text = f"{src_unified} </s> {src_tag}"
    tgt_in = f"{tgt_tag} {tgt_unified}"
    tgt_labels = f"{tgt_unified} </s>"

    return src_text, tgt_in, tgt_labels


def process_and_split_data(cfg: dict | None = None) -> Dict[str, Any]:
    """Main pipeline to build group splits, format for IndicBART, and output JSONLs."""
    if cfg is None:
        cfg = load_config()

    train_cap = int(cfg.get("training", {}).get("max_train_samples_per_dir", 3000))
    val_cap = int(cfg.get("training", {}).get("max_val_samples_per_dir", 200))
    test_cap = int(cfg.get("training", {}).get("max_test_samples_per_dir", 200))

    data_processed = resolve_path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    data_processed.mkdir(parents=True, exist_ok=True)

    print("Loading all raw PMIndiaSum datasets...")
    all_raw = load_all_pmindia()

    print("Building cross-lingual URL-to-Group mappings...")
    hi_to_en, kn_to_en = build_url_group_mappings(all_raw)
    print(f"  Mapped {len(hi_to_en)} Hindi URLs and {len(kn_to_en)} Kannada URLs to English group IDs.")

    # Direction-specific records pooled by group
    # Structure: split -> direction -> list of records
    pooled_by_split = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    groups_seen = {"train": set(), "val": set(), "test": set()}
    dropped_unmapped = 0

    for config_name, splits_dict in all_raw.items():
        src_lang, tgt_lang = config_name.split("-")
        lang_map = {"english": "en", "hindi": "hi", "kannada": "kn"}
        src = lang_map[src_lang]
        tgt = lang_map[tgt_lang]
        direction = f"{src}-{tgt}"

        # Combine all train/valid/test rows from PMIndia to re-split cleanly by group
        combined_df = pd.concat(splits_dict.values(), ignore_index=True)
        # Deduplicate identical (source_url, target_url, text)
        combined_df = combined_df.drop_duplicates(subset=["source_url", "target_url", "text"])

        for _, row in combined_df.iterrows():
            source_url = str(row["source_url"]).strip()
            target_url = str(row["target_url"]).strip()
            text = str(row["text"]).strip()
            summary = str(row["summary"]).strip()

            # Determine English group_id per datasets.md section 1.1
            if src == "en":
                group_id = source_url
            elif src == "hi" and tgt == "hi":
                group_id = hi_to_en.get(source_url)
            elif src == "kn" and tgt == "kn":
                group_id = kn_to_en.get(source_url)
            else:
                group_id = None

            if not group_id:
                dropped_unmapped += 1
                continue

            split = get_group_split(group_id)
            groups_seen[split].add(group_id)

            src_text, tgt_in, tgt_labels = format_indicbart_fields(text, summary, src, tgt)

            record = {
                "direction": direction,
                "src_lang": src,
                "tgt_lang": tgt,
                "group_id": group_id,
                "text": text,
                "summary": summary,
                "src_text": src_text,
                "tgt_in": tgt_in,
                "tgt_labels": tgt_labels,
            }
            pooled_by_split[split][direction].append(record)

    # Validate NON-NEGOTIABLE RULE 5: Group Leakage Check
    train_groups = groups_seen["train"]
    val_groups = groups_seen["val"]
    test_groups = groups_seen["test"]

    assert train_groups & val_groups == set(), "CRITICAL: Group leakage between train and val!"
    assert train_groups & test_groups == set(), "CRITICAL: Group leakage between train and test!"
    assert val_groups & test_groups == set(), "CRITICAL: Group leakage between val and test!"
    print("Group leakage assertion PASSED: train, val, and test group sets are completely disjoint.")

    # Apply caps and write JSONL files
    final_records = {"train": [], "val": [], "test": []}
    counts_table: Dict[str, Dict[str, int]] = {}

    caps = {"train": train_cap, "val": val_cap, "test": test_cap}

    for split in ["train", "val", "test"]:
        counts_table[split] = {}
        for direction, recs in pooled_by_split[split].items():
            cap = caps[split]
            selected = recs[:cap]
            counts_table[split][direction] = len(selected)
            final_records[split].extend(selected)

        # Write to data/processed/{split}.jsonl
        out_path = data_processed / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in final_records[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(final_records[split])} records to {out_path}")

    # Compute length statistics with IndicBART tokenizer
    print("\nComputing token-length statistics with IndicBART tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["models"]["summarizer"]["base"],
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )

    all_train_samples = final_records["train"]
    src_lengths = []
    tgt_lengths = []

    for item in all_train_samples:
        src_tokens = tokenizer.encode(item["src_text"], add_special_tokens=False)
        tgt_tokens = tokenizer.encode(item["tgt_labels"], add_special_tokens=False)
        src_lengths.append(len(src_tokens))
        tgt_lengths.append(len(tgt_tokens))

    stats_summary = {
        "source": {
            "mean": round(float(np.mean(src_lengths)), 2),
            "median": round(float(np.median(src_lengths)), 2),
            "p90": round(float(np.percentile(src_lengths, 90)), 2),
            "p99": round(float(np.percentile(src_lengths, 99)), 2),
            "max": int(np.max(src_lengths)),
        },
        "target": {
            "mean": round(float(np.mean(tgt_lengths)), 2),
            "median": round(float(np.median(tgt_lengths)), 2),
            "p90": round(float(np.percentile(tgt_lengths, 90)), 2),
            "p99": round(float(np.percentile(tgt_lengths, 99)), 2),
            "max": int(np.max(tgt_lengths)),
        },
        "counts_per_split_direction": counts_table,
        "dropped_unmappable_rows": dropped_unmapped,
        "unique_groups": {
            "train": len(train_groups),
            "val": len(val_groups),
            "test": len(test_groups),
            "total": len(train_groups | val_groups | test_groups),
        },
    }

    # Save stats to results/raw/data_stats.json
    results_raw = resolve_path(cfg.get("paths", {}).get("results_raw", "results/raw"))
    results_raw.mkdir(parents=True, exist_ok=True)
    stats_file = results_raw / "data_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Saved token and split stats to {stats_file}")

    # Generate and save histogram
    figs_dir = resolve_path(cfg.get("paths", {}).get("results_figs", "results/figs"))
    figs_dir.mkdir(parents=True, exist_ok=True)
    hist_path = figs_dir / "token_lengths.png"

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(src_lengths, bins=40, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.axvline(stats_summary["source"]["p90"], color="red", linestyle="--", label=f"P90={stats_summary['source']['p90']:.0f}")
    plt.title("IndicBART Source Token Lengths")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(tgt_lengths, bins=30, color="#2ca02c", edgecolor="black", alpha=0.8)
    plt.axvline(stats_summary["target"]["p90"], color="red", linestyle="--", label=f"P90={stats_summary['target']['p90']:.0f}")
    plt.title("IndicBART Target Token Lengths")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved token length histogram to {hist_path}")

    return stats_summary


def main():
    print("=" * 65)
    print("GovDoc-RAG Data Processing & Group Split Pipeline (Task M1)")
    print("=" * 65)
    stats = process_and_split_data()

    print("\n" + "=" * 65)
    print("Dataset Split Summary (Counts per Direction):")
    counts = stats["counts_per_split_direction"]
    directions = sorted(list(counts["train"].keys()))
    header = f"{'Direction':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8}"
    print(header)
    print("-" * len(header))
    for d in directions:
        tr = counts["train"].get(d, 0)
        va = counts["val"].get(d, 0)
        te = counts["test"].get(d, 0)
        print(f"{d:<12} | {tr:<8} | {va:<8} | {te:<8}")
    print("=" * 65)

    print("\nToken Length Percentiles:")
    print(f"  Source: Mean={stats['source']['mean']}, Median={stats['source']['median']}, P90={stats['source']['p90']}, Max={stats['source']['max']}")
    print(f"  Target: Mean={stats['target']['mean']}, Median={stats['target']['median']}, P90={stats['target']['p90']}, Max={stats['target']['max']}")

    log_msg = f"Task M1 Data Pipeline Complete: {stats['unique_groups']['total']} unique groups across 5 directions. Stats saved to results/raw/data_stats.json."
    append_log(log_msg, module="M1:GROUP_SPLIT")
    print("\nLog appended to results/LOG.md.")


if __name__ == "__main__":
    main()
