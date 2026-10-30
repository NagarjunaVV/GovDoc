"""Query normalizer with script detection, mode override, and multilingual multi-view generation."""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from govdoc.config import load_config
from govdoc.ingest.quality_gate import RE_LATIN, RE_DEVANAGARI, RE_KANNADA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Common Hinglish stopwords and keywords
HINGLISH_KEYWORDS = {
    "kya", "kaise", "kab", "kaha", "yojana", "sarkar", "sarkari", "suchna",
    "shiksha", "swasthya", "pension", "aavedan", "patra", "niyam", "adhikari",
    "pradhan", "mantri", "seva", "yojna", "kare", "hoga", "hai", "mein", "liye"
}

# Common Kanglish stopwords and keywords
KANGLISH_KEYWORDS = {
    "yenu", "hege", "yaava", "ellide", "yojane", "sarkara", "sarkari",
    "adhisuchane", "shikshana", "aarogya", "neeru", "hanchike", "samiti",
    "grama", "panchayati", "bagge", "vivara", "maaduvudu"
}

# Phonetic mapping dictionaries for rule-based roman->native transliteration
ROMAN_TO_DEVANAGARI = [
    ("shree", "श्री"), ("shri", "श्री"), ("ksh", "क्ष"), ("gy", "ज्ञ"),
    ("sh", "श"), ("shh", "ष"), ("ch", "च"), ("chh", "छ"), ("th", "थ"),
    ("thh", "ठ"), ("dh", "ध"), ("dhh", "ढ"), ("kh", "ख"), ("gh", "घ"),
    ("bh", "भ"), ("ph", "फ"), ("jh", "झ"),
    ("aa", "आ"), ("ee", "ई"), ("oo", "ऊ"), ("ai", "ऐ"), ("au", "औ"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
    ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "त"),
    ("d", "द"), ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"),
    ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"),
    ("s", "स"), ("h", "ह")
]

ROMAN_TO_KANNADA = [
    ("shree", "ಶ್ರೀ"), ("shri", "ಶ್ರೀ"), ("ksh", "ಕ್ಷ"), ("jn", "ಜ್ಞ"),
    ("sh", "ಶ"), ("shh", "ಷ"), ("ch", "ಚ"), ("chh", "ಛ"), ("th", "ಥ"),
    ("thh", "ಠ"), ("dh", "ಧ"), ("dhh", "ಢ"), ("kh", "ಖ"), ("gh", "ಘ"),
    ("bh", "ಭ"), ("ph", "ಫ"), ("jh", "ಝ"),
    ("aa", "ಆ"), ("ee", "ಈ"), ("oo", "ಊ"), ("ai", "ಐ"), ("au", "ಔ"),
    ("a", "ಅ"), ("i", "ಇ"), ("u", "ಉ"), ("e", "ಎ"), ("o", "ಒ"),
    ("k", "ಕ"), ("g", "ಗ"), ("c", "ಕ"), ("j", "ಜ"), ("t", "ತ"),
    ("d", "ದ"), ("n", "ನ"), ("p", "ಪ"), ("b", "ಬ"), ("m", "ಮ"),
    ("y", "ಯ"), ("r", "ರ"), ("l", "ಲ"), ("v", "ವ"), ("w", "ವ"),
    ("s", "ಸ"), ("h", "ಹ")
]


def detect_query_script(query: str) -> str:
    """
    Detect dominant script in query:
    Returns 'hi' (Devanagari), 'kn' (Kannada), or 'latin'.
    """
    latin_count = len(RE_LATIN.findall(query))
    deva_count = len(RE_DEVANAGARI.findall(query))
    kan_count = len(RE_KANNADA.findall(query))

    if deva_count > 0 and deva_count >= latin_count and deva_count >= kan_count:
        return "hi"
    if kan_count > 0 and kan_count >= latin_count and kan_count >= deva_count:
        return "kn"
    return "latin"


def detect_query_mode(query: str) -> str:
    """
    Detect language/mode from query text:
    'en', 'hi', 'kn', 'hinglish', 'kanglish'.
    """
    script = detect_query_script(query)
    if script == "hi":
        return "hi"
    if script == "kn":
        return "kn"

    # In Latin script: check for Hinglish / Kanglish vocabulary
    words = set(re.findall(r"\b\w+\b", query.lower()))
    hinglish_overlap = len(words & HINGLISH_KEYWORDS)
    kanglish_overlap = len(words & KANGLISH_KEYWORDS)

    if hinglish_overlap > 0 and hinglish_overlap >= kanglish_overlap:
        return "hinglish"
    if kanglish_overlap > 0:
        return "kanglish"
    return "en"


def phonetic_transliterate(text: str, target_lang: str) -> str:
    """Deterministic phonetic mapping from Romanised text to native Indic script."""
    rules = ROMAN_TO_DEVANAGARI if target_lang == "hi" else ROMAN_TO_KANNADA
    words = text.split()
    out_words = []

    for word in words:
        w_lower = word.lower()
        res = ""
        i = 0
        while i < len(w_lower):
            matched = False
            for roman_pat, native_char in rules:
                if w_lower[i : i + len(roman_pat)] == roman_pat:
                    res += native_char
                    i += len(roman_pat)
                    matched = True
                    break
            if not matched:
                res += w_lower[i]
                i += 1
        out_words.append(res)

    return " ".join(out_words)


class QueryNormalizer:
    """Query script normalizer with multi-view query generation."""

    def __init__(self):
        cfg = load_config()
        t_cfg = cfg.get("transliteration", {})
        self.transliteration_enabled = t_cfg.get("enabled", True)
        self.engine = None

        if self.transliteration_enabled:
            try:
                # Attempt to load IndicXlit if package is present
                from ai4bharat.transliteration import XlitEngine
                self.engine = XlitEngine(["hi", "kn"], beam_width=4, rescore=False)
                print("QueryNormalizer: IndicXlit engine loaded successfully.")
            except Exception:
                # Fallback to deterministic rule-based transliterator
                self.engine = None

    def transliterate_to_native(self, text: str, lang: str) -> str:
        """Transliterate Roman text to native Indic script."""
        if not self.transliteration_enabled:
            return text

        if self.engine is not None:
            try:
                out = self.engine.translit_sentence(text, lang_code=lang)
                return out.strip()
            except Exception:
                pass

        return phonetic_transliterate(text, lang)

    def normalize(
        self,
        query: str,
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Normalize query and produce multi-view representation.

        Args:
            query: Input user query.
            mode: Mode override in {'auto', 'en', 'hi', 'kn', 'hinglish', 'kanglish'}.

        Returns:
            Dict containing:
            {
                "query": str,
                "detected_script": str,
                "mode": str,
                "views": List[str],  # Primary and native views
            }
        """
        clean_q = query.strip()
        script = detect_query_script(clean_q)

        # Resolve mode
        if mode == "auto":
            resolved_mode = detect_query_mode(clean_q)
        else:
            resolved_mode = mode.lower()

        views = [clean_q]

        # For Latin-script Indic modes (Hinglish/Kanglish), produce native view
        if resolved_mode == "hinglish" and self.transliteration_enabled:
            native_view = self.transliterate_to_native(clean_q, "hi")
            if native_view and native_view != clean_q:
                views.append(native_view)

        elif resolved_mode == "kanglish" and self.transliteration_enabled:
            native_view = self.transliterate_to_native(clean_q, "kn")
            if native_view and native_view != clean_q:
                views.append(native_view)

        return {
            "query": clean_q,
            "detected_script": script,
            "mode": resolved_mode,
            "views": views,
        }
