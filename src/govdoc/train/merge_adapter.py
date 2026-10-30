"""Merge fine-tuned LoRA adapter into base IndicBART and save fp16 model."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from govdoc.config import load_config, resolve_path
from govdoc.logging_utils import append_log

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def merge_and_save_adapter(
    adapter_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    base_model_name: Optional[str] = None,
) -> Path:
    """Merge LoRA adapter into base model and export merged weights in fp16."""
    cfg = load_config()
    if base_model_name is None:
        base_model_name = cfg.get("training", {}).get("base_model", "ai4bharat/IndicBART")
    if adapter_dir is None:
        adapter_dir = resolve_path("models/indicbart_lora_adapter")
    else:
        adapter_dir = resolve_path(adapter_dir)
    if output_dir is None:
        output_dir = resolve_path("models/indicbart_lora_merged")
    else:
        output_dir = resolve_path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GovDoc-RAG LoRA Adapter Merger")
    print(f"Base Model:  {base_model_name}")
    print(f"Adapter:     {adapter_dir}")
    print(f"Output:      {output_dir}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model in fp16 on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )

    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
    except Exception:
        cfg_ib = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModelForSeq2SeqLM.from_config(cfg_ib)
        if device == "cuda":
            base_model = base_model.half().to(device)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging LoRA weights into base model layers...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Merge completed successfully.")
    append_log(f"Merged LoRA adapter {adapter_dir} into {output_dir}", module="M2:MERGE_ADAPTER")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--adapter", type=str, default="models/indicbart_lora_adapter")
    parser.add_argument("--output", type=str, default="models/indicbart_lora_merged")
    args = parser.parse_args()

    merge_and_save_adapter(adapter_dir=args.adapter, output_dir=args.output)


if __name__ == "__main__":
    main()
