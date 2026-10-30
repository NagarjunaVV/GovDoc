"""
IndicBART fp16 Summarization Engine (Non-Negotiable Technical Rule 1).

Adheres strictly to Non-Negotiable Technical Rule 1:
- Tokenizer: AutoTokenizer(use_fast=False, keep_accents=True, do_lower_case=False)
- Input format: "<text> </s> <2src>"
- Decoder start token: token ID of "<2tgt>"
- Script unification: Non-Devanagari text (Kannada) MUST be converted to Devanagari
  with Indic NLP Library before tokenising, and converted back after decoding.
- Generation: beam 4, no_repeat_ngram_size 3.
"""

from pathlib import Path
import re
from typing import List, Optional, Union
import unicodedata

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import from_devanagari, to_devanagari

# Language mapping
LANG_MAP = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "kn": "kn",
    "kannada": "kn",
}


class IndicBARTSummarizer:
    """Offline Multilingual Summarizer based on ai4bharat/IndicBART + LoRA."""

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
        precision: str = "fp16",
    ):
        cfg = load_config()
        if model_dir is None:
            # Check default candidate paths
            candidates = [
                resolve_path("models/indicbart_merged"),
                resolve_path(cfg.get("models", {}).get("summarizer", {}).get("lora_adapter", "models/indicbart_lora_merged")),
                resolve_path("models/indicbart_lora_adapter"),
                cfg.get("models", {}).get("summarizer", {}).get("base", "ai4bharat/IndicBART"),
            ]
            for cand in candidates:
                cand_path = Path(cand) if isinstance(cand, (str, Path)) else None
                if cand_path and cand_path.exists():
                    model_dir = cand
                    break
            if model_dir is None:
                model_dir = "ai4bharat/IndicBART"

        self.model_path = str(model_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = torch.float16 if (self.device.type == "cuda" and precision == "fp16") else torch.float32

        # Non-Negotiable Technical Rule 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            keep_accents=True,
            do_lower_case=False,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        # Cache language-specific bad token IDs to prevent cross-script leakage
        # (e.g. Hindi/Devanagari leaking into English summaries)
        self._devanagari_token_ids = [
            [tid]
            for token, tid in self.tokenizer.get_vocab().items()
            if any("\u0900" <= c <= "\u097F" for c in token)
        ]

    def _normalize_lang(self, lang: str) -> str:
        lang_lower = lang.strip().lower()
        if lang_lower not in LANG_MAP:
            raise ValueError(f"Unsupported language: {lang}. Must be one of en, hi, kn.")
        return LANG_MAP[lang_lower]

    def summarise(
        self,
        text: str,
        src_lang: str = "en",
        tgt_lang: str = "hi",
        max_length: int = 128,
        min_length: int = 8,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Summarise input text using IndicBART with Rule 1 script unification.

        Args:
            text: Source text to summarise.
            src_lang: Source language ('en', 'hi', 'kn').
            tgt_lang: Target language ('en', 'hi', 'kn').
            max_length: Max generation tokens (default: 128).
            min_length: Min generation tokens (default: 8).
            num_beams: Beam size (default: 4).
            no_repeat_ngram_size: N-gram deduplication size (default: 3).
            length_penalty: Length penalty (default: 1.0).

        Returns:
            Generated summary string in target language script.
        """
        if not text or not text.strip():
            return ""

        src_l = self._normalize_lang(src_lang)
        tgt_l = self._normalize_lang(tgt_lang)

        # Rule 1: Non-Devanagari text (Kannada) MUST be converted to Devanagari before tokenising
        norm_text = text.strip()
        if src_l == "kn":
            norm_text = to_devanagari(norm_text, "kn")

        # Rule 1: Input format "<text> </s> <2src>"
        input_str = f"{norm_text} </s> <2{src_l}>"

        # Tokenize with add_special_tokens=False to prevent double boundary tokens
        inputs = self.tokenizer(
            input_str,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Rule 1: decoder start token = id of "<2tgt>"
        tgt_tag = f"<2{tgt_l}>"
        bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")
        lang_id = self.tokenizer._convert_token_to_id_with_added_voc(tgt_tag)
        decoder_start_token_id = lang_id if lang_id is not None else (self.model.config.decoder_start_token_id or eos_id)

        # If target language is English, forbid Devanagari tokens during generation
        # to prevent vocabulary drift/code-mixing
        bad_words_ids = self._devanagari_token_ids if tgt_l == "en" else None

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                bad_words_ids=bad_words_ids,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                early_stopping=True,
            )

        # Decode tokens
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Clean stray tag artifacts, CJK ideogram leaks from mBART vocab (e.g. 達, 留), and leading symbols
        decoded = re.sub(r"<2[a-z]{2}>", "", decoded).strip()
        decoded = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", "", decoded).strip()
        decoded = re.sub(r"^[\s\W_]+", "", decoded).strip()

        # Rule 1: Converted back to target script after decoding (Kannada)
        if tgt_l == "kn":
            decoded = from_devanagari(decoded, "kn")

        return decoded

    def summarise_batch(
        self,
        texts: List[str],
        src_lang: str = "en",
        tgt_lang: str = "hi",
        max_length: int = 64,
        min_length: int = 8,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> List[str]:
        """Summarise a batch of text segments concurrently on GPU."""
        if not texts:
            return []

        src_l = self._normalize_lang(src_lang)
        tgt_l = self._normalize_lang(tgt_lang)

        # Pre-process inputs
        formatted_inputs = []
        for t in texts:
            norm_text = t.strip()
            if src_l == "kn":
                norm_text = to_devanagari(norm_text, "kn")
            formatted_inputs.append(f"{norm_text} </s> <2{src_l}>")

        inputs = self.tokenizer(
            formatted_inputs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        tgt_tag = f"<2{tgt_l}>"
        bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")
        lang_id = self.tokenizer._convert_token_to_id_with_added_voc(tgt_tag)
        decoder_start_token_id = lang_id if lang_id is not None else (self.model.config.decoder_start_token_id or eos_id)

        bad_words_ids = self._devanagari_token_ids if tgt_l == "en" else None

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                bad_words_ids=bad_words_ids,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                early_stopping=True,
            )

        results = []
        for out in outputs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True).strip()
            decoded = re.sub(r"<2[a-z]{2}>", "", decoded).strip()
            decoded = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", "", decoded).strip()
            decoded = re.sub(r"^[\s\W_]+", "", decoded).strip()
            if tgt_l == "kn":
                decoded = from_devanagari(decoded, "kn")
            results.append(decoded)

        return results

    summarize_batch = summarise_batch

    # Alias for spelling compatibility (summarise / summarize)
    summarize = summarise
