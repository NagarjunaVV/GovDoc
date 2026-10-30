"""Script unification module for multilingual Indic processing (Non-Negotiable Rule 1)."""

import argparse
import sys
import unicodedata
from typing import Dict, List, Set, Tuple

# Configure stdout for Unicode display on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
except ImportError:
    UnicodeIndicTransliterator = None


# Known characters in Kannada that may have one-way mappings in Devanagari
KNOWN_EXCEPTIONS: Set[str] = {
    "\u0cde",  # KANNADA SIGN JIHVAMULIYA
    "\u0cdf",  # KANNADA SIGN UPADHMANIYA
    "\u0cb4",  # KANNADA LETTER LLLA
}


def to_devanagari(text: str, lang: str) -> str:
    """Convert non-Devanagari Indic text (specifically Kannada) to Devanagari.
    
    Adheres strictly to Non-Negotiable Technical Rule 1:
    Non-Devanagari text (Kannada) MUST be converted to Devanagari with Indic NLP
    Library before tokenising.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if lang in ("kn", "kannada"):
        if UnicodeIndicTransliterator is None:
            raise RuntimeError("indic-nlp-library is required for Kannada script unification.")
        return UnicodeIndicTransliterator.transliterate(text, "kn", "hi")
    return text


def from_devanagari(text: str, lang: str) -> str:
    """Convert Devanagari text back to target language script (e.g. Kannada).
    
    Adheres strictly to Non-Negotiable Technical Rule 1:
    Converted back to native script after decoding.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if lang in ("kn", "kannada"):
        if UnicodeIndicTransliterator is None:
            raise RuntimeError("indic-nlp-library is required for Kannada script unification.")
        return UnicodeIndicTransliterator.transliterate(text, "hi", "kn")
    return text


def evaluate_roundtrip(samples: List[str], lang: str = "kn") -> Dict[str, any]:
    """Test round-trip fidelity on a collection of sample texts.
    
    Returns:
      dict with:
        - sample_count: total samples tested
        - exact_matches: samples with 100% string equality
        - exact_match_pct: percentage of exact matches
        - char_accuracy: overall character-level agreement percentage
        - mismatched_chars: list of characters that differed in round-trip
    """
    exact_matches = 0
    total_chars = 0
    matching_chars = 0
    mismatched_chars: Dict[str, int] = {}

    for sample in samples:
        sample_norm = unicodedata.normalize("NFC", sample)
        dev = to_devanagari(sample_norm, lang)
        rt = from_devanagari(dev, lang)

        if sample_norm == rt:
            exact_matches += 1
            matching_chars += len(sample_norm)
        else:
            # Character-by-character alignment
            for c_orig, c_rt in zip(sample_norm, rt):
                if c_orig == c_rt:
                    matching_chars += 1
                else:
                    mismatched_chars[c_orig] = mismatched_chars.get(c_orig, 0) + 1

            if len(sample_norm) != len(rt):
                # Account for length differences
                diff_len = abs(len(sample_norm) - len(rt))
                mismatched_chars["<len_diff>"] = mismatched_chars.get("<len_diff>", 0) + diff_len

        total_chars += len(sample_norm)

    char_acc = (matching_chars / total_chars * 100.0) if total_chars > 0 else 100.0
    exact_pct = (exact_matches / len(samples) * 100.0) if samples else 100.0

    return {
        "sample_count": len(samples),
        "exact_matches": exact_matches,
        "exact_match_pct": round(exact_pct, 2),
        "char_accuracy": round(char_acc, 2),
        "mismatched_chars": mismatched_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="Test script unification roundtrip.")
    parser.add_argument(
        "--text",
        type=str,
        default="ಭಾರತ ಸರ್ಕಾರದ ಅಧಿಕೃತ ಪ್ರಕಟಣೆ: ನವದೆಹಲಿ ಮತ್ತು ಬೆಂಗಳೂರು",
        help="Test Kannada sentence.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Script Unification Verification (Non-Negotiable Rule 1)")
    print(f"Original (kn):   {args.text}")
    deva = to_devanagari(args.text, "kn")
    print(f"Devanagari (hi): {deva}")
    rt = from_devanagari(deva, "kn")
    print(f"Roundtrip (kn):  {rt}")
    print(f"Exact Match:     {args.text == rt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
