"""PMIndiaSum dataset loader with offline local caching."""

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests

from govdoc.config import load_config, resolve_path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONFIGS = [
    "english-english",
    "english-hindi",
    "english-kannada",
    "hindi-hindi",
    "kannada-kannada",
]

BASE_HF_URL = "https://huggingface.co/datasets/PMIndiaData/PMIndiaSum/resolve/main"


def normalize_text(text: str) -> str:
    """Normalize text using Unicode NFC form and normalize spaces."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def download_pmindia_config(
    config_name: str,
    raw_dir: Optional[Path | str] = None,
    splits: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Download and cache raw CSVs for a PMIndiaSum configuration."""
    if splits is None:
        splits = ["train", "valid", "test"]

    if raw_dir is None:
        cfg = load_config()
        base_raw = resolve_path(cfg.get("paths", {}).get("data_raw", "data/raw"))
    else:
        base_raw = resolve_path(raw_dir)

    target_dir = base_raw / "pmindia" / config_name
    target_dir.mkdir(parents=True, exist_ok=True)

    cached_files = {}
    for split in splits:
        dest_file = target_dir / f"{split}.csv"
        cached_files[split] = dest_file
        if dest_file.exists() and dest_file.stat().st_size > 0:
            continue

        url = f"{BASE_HF_URL}/{config_name}/{split}.csv"
        print(f"Downloading {config_name}/{split}.csv...")
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download {url}: status {resp.status_code}"
            )
        temp_file = dest_file.with_suffix(".csv.tmp")
        with open(temp_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        temp_file.replace(dest_file)
        print(f"  Cached {dest_file.name} ({dest_file.stat().st_size / 1024:.1f} KB)")

    return cached_files


def load_pmindia_split(
    config_name: str,
    split: str = "train",
    raw_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load, clean and return a single split DataFrame."""
    files = download_pmindia_config(config_name, raw_dir=raw_dir, splits=[split])
    csv_path = files[split]

    df = pd.read_csv(csv_path)
    required_cols = ["source_url", "target_url", "text", "summary"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from {csv_path}")

    # Drop nulls and empty texts
    df = df.dropna(subset=["text", "summary"]).copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["summary"] = df["summary"].astype(str).map(normalize_text)
    df = df[(df["text"].str.len() > 10) & (df["summary"].str.len() > 3)]

    # Add source/target language tags
    src_lang, tgt_lang = config_name.split("-")
    lang_map = {"english": "en", "hindi": "hi", "kannada": "kn"}
    df["src_lang"] = lang_map.get(src_lang, src_lang)
    df["tgt_lang"] = lang_map.get(tgt_lang, tgt_lang)
    df["direction"] = f"{df['src_lang'].iloc[0]}-{df['tgt_lang'].iloc[0]}"

    return df


def load_all_pmindia(raw_dir: Optional[Path | str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Download and return all 5 configurations across all splits."""
    all_data = {}
    for c in CONFIGS:
        all_data[c] = {}
        for s in ["train", "valid", "test"]:
            all_data[c][s] = load_pmindia_split(c, split=s, raw_dir=raw_dir)
    return all_data


def main():
    parser = argparse.ArgumentParser(description="PMIndiaSum data loader.")
    parser.add_argument("--config", type=str, default="all", choices=["all"] + CONFIGS)
    parser.add_argument("--download-only", action="store_true", help="Download raw CSVs without printing preview.")
    args = parser.parse_args()

    configs_to_process = CONFIGS if args.config == "all" else [args.config]

    print("=" * 65)
    print("PMIndiaSum Configuration & Column Verification")
    print(f"Configurations to process: {configs_to_process}")
    print("=" * 65)

    for cfg_name in configs_to_process:
        print(f"\nProcessing {cfg_name}...")
        files = download_pmindia_config(cfg_name)
        df_train = load_pmindia_split(cfg_name, split="train")
        df_val = load_pmindia_split(cfg_name, split="valid")
        df_test = load_pmindia_split(cfg_name, split="test")

        print(f"  Columns: {list(df_train.columns)}")
        print(f"  Counts: train={len(df_train)}, valid={len(df_val)}, test={len(df_test)}")
        if not args.download_only:
            sample = df_train.iloc[0]
            print(f"  Sample summary: {sample['summary'][:80]}...")


if __name__ == "__main__":
    main()
