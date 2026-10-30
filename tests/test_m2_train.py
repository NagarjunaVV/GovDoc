"""Unit tests for Module M2: Training pipeline, baselines, and PEFT LoRA."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config
from govdoc.eval.rouge_indic import compute_rouge_l
from govdoc.train.baselines import extract_lead_1
from govdoc.train.lora_train import create_collate_fn


def test_lead1_extraction():
    """Verify Lead-1 extraction across English, Hindi, and Kannada."""
    en_text = "The Union Cabinet approved the national mission. It will commence soon."
    hi_text = "प्रधानमंत्री ने नई योजना की शुरुआत की। इस योजना से लाखों लोग लाभान्वित होंगे।"
    kn_text = "ಕೇಂದ್ರ ಸಚಿವ ಸಂಪುಟವು ನೂತನ ಯೋಜನೆಗೆ ಅನುಮೋದನೆ ನೀಡಿದೆ. ಇದು ಶೀಘ್ರವೇ ಜಾರಿಯಾಗಲಿದೆ."

    lead_en = extract_lead_1(en_text, "en")
    lead_hi = extract_lead_1(hi_text, "hi")
    lead_kn = extract_lead_1(kn_text, "kn")

    assert "The Union Cabinet approved the national mission" in lead_en
    assert "प्रधानमंत्री ने नई योजना की शुरुआत की" in lead_hi
    assert "ಕೇಂದ್ರ ಸಚಿವ ಸಂಪುಟವು ನೂತನ ಯೋಜನೆಗೆ ಅನುಮೋದನೆ ನೀಡಿದೆ" in lead_kn


def test_lora_configuration():
    """Verify LoRA configuration targets specified projection layers and has < 5% trainable params."""
    cfg = load_config()
    train_cfg = cfg.get("training", {})
    base_name = train_cfg.get("base_model", "ai4bharat/IndicBART")

    model_cfg = AutoConfig.from_pretrained(base_name)
    base_model = AutoModelForSeq2SeqLM.from_config(model_cfg)

    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 32)),
        lora_alpha=int(train_cfg.get("lora_alpha", 64)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=train_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]),
        task_type="SEQ_2_SEQ_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    trainable_pct = (trainable_params / total_params) * 100
    print(f"LoRA Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

    assert trainable_pct < 5.0, f"LoRA trainable parameters should be < 5%, got {trainable_pct}%"
    assert trainable_params > 5_000_000, "Trainable params should cover target projection layers"


def test_data_collator():
    """Verify data collator pads labels and sets padding indices to -100."""
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/IndicBART",
        use_fast=False,
        keep_accents=True,
        do_lower_case=False,
    )
    collate_fn = create_collate_fn(tokenizer, max_src_len=32, max_tgt_len=16)

    dummy_batch = [
        {"src_text": "Short source </s> <2en>", "tgt_labels": "Short target </s>"},
        {"src_text": "A somewhat longer source text for batch testing </s> <2en>", "tgt_labels": "Longer target summary </s>"},
    ]

    batch = collate_fn(dummy_batch)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert -100 in batch["labels"]  # Masked padding in shorter label
