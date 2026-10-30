"""
IndicTrans2 Translation Engine for Indic Languages (Hindi, Kannada, etc.).

Uses ai4bharat / prajdabre rotary-indictrans2-en-indic-dist-200M for high-fidelity translation
from English to Indic languages (hin_Deva, kan_Knda).
Incorporates Indic NLP Library script conversion for Kannada.
"""

from pathlib import Path
import re
from typing import List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.data.script_unify import from_devanagari

LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "kn": "kan_Knda",
    "kannada": "kan_Knda",
}


class IndicTrans2Translator:
    """Offline translator based on IndicTrans2 distilled 200M model."""

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
        precision: str = "fp16",
    ):
        if model_dir is None:
            model_dir = resolve_path("models/indictrans2_en_indic")
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"IndicTrans2 model directory not found at {self.model_dir}. "
                f"Please download rotary-indictrans2-en-indic-dist-200M to models/indictrans2_en_indic."
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch_dtype = torch.float16 if (precision == "fp16" and self.device == "cuda") else torch.float32

        # Load Tokenizer & Model with trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def translate(
        self,
        text: str,
        src_lang: str = "en",
        tgt_lang: str = "hi",
        max_length: int = 256,
        num_beams: int = 4,
    ) -> str:
        """Translate a single text string from English to Hindi or Kannada."""
        if not text or not text.strip():
            return ""

        src_l = src_lang.strip().lower()
        tgt_l = tgt_lang.strip().lower()

        if src_l == tgt_l:
            return text

        src_tag = LANGUAGE_CODE_MAP.get(src_l, "eng_Latn")
        tgt_tag = LANGUAGE_CODE_MAP.get(tgt_l, "hin_Deva")

        # IndicTrans2 input format: "<src_lang> <tgt_lang> <text>"
        prompt = f"{src_tag} {tgt_tag} {text.strip()}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # If target is Kannada, decode from Devanagari representation back to Kannada script
        if tgt_l in ("kn", "kannada"):
            decoded = from_devanagari(decoded, "kn")

        # Clean stray artifacts
        decoded = re.sub(r"\s+", " ", decoded).strip()
        return decoded

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = "en",
        tgt_lang: str = "hi",
        max_length: int = 256,
        num_beams: int = 4,
    ) -> List[str]:
        """Translate a batch of texts."""
        return [
            self.translate(t, src_lang=src_lang, tgt_lang=tgt_lang, max_length=max_length, num_beams=num_beams)
            for t in texts
        ]
