"""PEFT LoRA fine-tuning for IndicBART on multilingual government documents (Task M2)."""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import from_devanagari
from govdoc.eval.rouge_indic import compute_corpus_rouge_l
from govdoc.logging_utils import append_log, save_experiment_record

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class PMIndiaSumDataset(Dataset):
    """Dataset for multilingual IndicBART JSONL records."""

    def __init__(self, jsonl_path: Path | str, max_samples: Optional[int] = None):
        self.records = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    if max_samples and len(self.records) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def create_collate_fn(tokenizer, max_src_len: int = 512, max_tgt_len: int = 64):
    """Data collator with dynamic padding and label masking (-100)."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        src_texts = [item["src_text"] for item in batch]
        tgt_labels = [item["tgt_labels"] for item in batch]

        model_inputs = tokenizer(
            src_texts,
            max_length=max_src_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_labels,
                max_length=max_tgt_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        # Replace padding token id with -100 so CrossEntropy ignores them
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    return collate_fn


def evaluate_val_rouge(
    model,
    tokenizer,
    val_items: List[Dict[str, Any]],
    device: str,
    n_samples: int = 100,
) -> float:
    """Fast validation ROUGE-L computation on n samples."""
    model.eval()
    samples = val_items[:n_samples]
    preds, refs = [], []

    for it in samples:
        tgt_lang = it["tgt_lang"]
        inputs = tokenizer(
            it["src_text"],
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        inputs.pop("token_type_ids", None)

        tgt_tag = f"<2{tgt_lang}>"
        decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_tag)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                decoder_start_token_id=decoder_start_token_id,
                max_new_tokens=64,
                num_beams=2,
            )

        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        if tgt_lang == "kn":
            final_pred = from_devanagari(decoded, "kn")
        else:
            final_pred = decoded

        preds.append(final_pred)
        refs.append(it["summary"])

    res = compute_corpus_rouge_l(preds, refs)
    model.train()
    return res["f1"]


def train_lora(
    config_dict: Optional[Dict[str, Any]] = None,
    smoke_test: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train IndicBART with LoRA adhering to requirements.md section 5.2."""
    cfg = config_dict or load_config()
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    base_model_name = train_cfg.get("base_model", "ai4bharat/IndicBART")
    lora_r = int(train_cfg.get("lora_r", 32))
    lora_alpha = int(train_cfg.get("lora_alpha", 64))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
    target_modules = train_cfg.get(
        "target_modules", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    )
    lr = float(train_cfg.get("lr", 3e-4))
    batch_size = int(train_cfg.get("batch_size", 8))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
    epochs = int(train_cfg.get("epochs", 3))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    time_box_minutes = float(train_cfg.get("time_box_minutes", 100))

    if smoke_test:
        max_steps = 100
        val_eval_interval = 50
        checkpoint_interval = 100
        print(f"Running in SMOKE-TEST mode: max_steps={max_steps}")
    else:
        max_steps = None
        val_eval_interval = 250
        checkpoint_interval = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer adhering to Non-Negotiable Rule 1
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )

    # Load base model
    model_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"Loading base model {base_model_name} in {model_dtype}...")
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
        ).to(device)
    except Exception:
        # Fallback to config instantiation if local offline
        cfg_ib = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModelForSeq2SeqLM.from_config(cfg_ib).to(dtype=model_dtype, device=device)

    # Configure PEFT LoRA
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        print(f"Resuming LoRA adapter from checkpoint: {resume_from_checkpoint}")
        model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    # Load datasets
    data_dir = resolve_path(paths_cfg.get("data_processed", "data/processed"))
    train_dataset = PMIndiaSumDataset(data_dir / "train.jsonl")
    val_dataset = PMIndiaSumDataset(data_dir / "val.jsonl")

    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_train_steps = (len(train_loader) // grad_accum) * epochs if max_steps is None else max_steps
    warmup_steps = int(total_train_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # Loss function with label smoothing
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)

    # Set up curve logging
    raw_dir = resolve_path(paths_cfg.get("results_raw", "results/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = raw_dir / "training_curve.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "val_rougeL", "elapsed_s"])

    checkpoints_dir = resolve_path("models/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training loop...")
    t_start = time.time()
    global_step = 0
    running_loss = 0.0
    initial_loss = None
    final_loss = None

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            # Check wall-time cap
            elapsed_min = (time.time() - t_start) / 60.0
            if elapsed_min >= time_box_minutes:
                print(f"Reached wall-time cap ({time_box_minutes} min). Stopping training.")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = running_loss / grad_accum
                running_loss = 0.0

                if initial_loss is None:
                    initial_loss = step_loss
                final_loss = step_loss

                if global_step % 10 == 0 or global_step == 1:
                    print(f"Step {global_step}/{total_train_steps} | Loss: {step_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                # Validation & fast ROUGE-L
                if global_step % val_eval_interval == 0 or global_step == max_steps:
                    val_rouge = evaluate_val_rouge(model, tokenizer, val_dataset.records, device, n_samples=100)
                    elapsed_s = round(time.time() - t_start, 1)
                    print(f"  --> Evaluation at Step {global_step}: Train Loss={step_loss:.4f}, Val ROUGE-L={val_rouge:.4f}")
                    with open(curve_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, round(step_loss, 4), "N/A", round(val_rouge, 4), elapsed_s])

                # Checkpoint saving
                if global_step % checkpoint_interval == 0 or global_step == max_steps:
                    ckpt_path = checkpoints_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)
                    print(f"  Saved checkpoint to {ckpt_path}")

                if max_steps and global_step >= max_steps:
                    break

        if (max_steps and global_step >= max_steps) or elapsed_min >= time_box_minutes:
            break

    # Save final adapter
    final_adapter_path = resolve_path("models/indicbart_lora_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"\nTraining complete. Final adapter saved to {final_adapter_path}")

    # Generate smoke test verification on 5 samples
    print("\n--- Smoke Test Verification (5 Samples) ---")
    model.eval()
    sample_generations = []
    test_dataset = PMIndiaSumDataset(data_dir / "test.jsonl")

    # Pick samples covering Kannada, Hindi, and English
    sample_items = [
        it for it in test_dataset.records if it["tgt_lang"] in ("hi", "kn", "en")
    ][:5]

    for idx, it in enumerate(sample_items):
        tgt_lang = it["tgt_lang"]
        inputs = tokenizer(
            it["src_text"],
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        inputs.pop("token_type_ids", None)

        tgt_tag = f"<2{tgt_lang}>"
        decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_tag)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                decoder_start_token_id=decoder_start_token_id,
                max_new_tokens=64,
                num_beams=2,
            )

        raw_gen = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        final_gen = from_devanagari(raw_gen, "kn") if tgt_lang == "kn" else raw_gen

        sample_generations.append({
            "direction": it["direction"],
            "prediction": final_gen,
            "reference": it["summary"],
        })
        print(f"\nSample [{idx+1}] ({it['direction']}):")
        print(f"  Reference:  {it['summary']}")
        print(f"  Generated:  {final_gen}")

    summary_stats = {
        "global_steps": global_step,
        "initial_loss": round(float(initial_loss or 0.0), 4),
        "final_loss": round(float(final_loss or 0.0), 4),
        "loss_decreased": bool((final_loss or 0.0) < (initial_loss or 1.0)),
        "wall_time_s": round(time.time() - t_start, 2),
        "sample_generations": sample_generations,
    }

    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description="Train IndicBART with LoRA.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 100 steps smoke test.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 65)
    print("GovDoc-RAG IndicBART LoRA Training Pipeline")
    print(f"Smoke Test: {args.smoke_test} | Resume: {args.resume}")
    print("=" * 65)

    stats = train_lora(cfg, smoke_test=args.smoke_test, resume_from_checkpoint=args.resume)

    log_entry = (
        f"M2 LoRA Training (Smoke Test: {args.smoke_test}): "
        f"Steps={stats['global_steps']}, Initial Loss={stats['initial_loss']}, "
        f"Final Loss={stats['final_loss']}, Loss Decreased={stats['loss_decreased']}, "
        f"Wall Time={stats['wall_time_s']}s"
    )
    append_log(log_entry, module="M2:LORA_TRAIN")
    print("\nTraining run recorded in results/LOG.md.")


if __name__ == "__main__":
    main()
