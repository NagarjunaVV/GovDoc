"""Per-page quality checks for extracted PDF text.

Performs checks for:
1. Empty or near-empty text.
2. Fraction of expected-script letters (Devanagari, Kannada, Latin).
3. Legacy-font garbage heuristics (Kruti Dev / Shree Lipi ASCII artifacts, symbol density).
Does not crash on bad pages; returns a structured quality report.
"""

import re
import sys
import unicodedata
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Unicode ranges for expected scripts in GovDoc
RE_LATIN = re.compile(r"[a-zA-Z]")
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
RE_KANNADA = re.compile(r"[\u0C80-\u0CFF]")

# Common legacy-font garbage patterns (e.g., Kruti Dev / Shree Lipi mapped onto ASCII symbols)
# Kruti Dev frequently produces patterns with embedded symbols like ;, [, ], {, }, ~, |, ^, \
RE_LEGACY_SYMBOLS = re.compile(r"[;\[\]\{\}\\\^~|`]")
RE_CONSECUTIVE_CONSONANTS_ASCII = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{6,}")


def analyze_script_distribution(text: str) -> Dict[str, Any]:
    """Calculate character and script distribution across Devanagari, Kannada, Latin, and symbols."""
    latin_count = len(RE_LATIN.findall(text))
    deva_count = len(RE_DEVANAGARI.findall(text))
    kan_count = len(RE_KANNADA.findall(text))
    total_letters = latin_count + deva_count + kan_count

    digits = sum(1 for c in text if c.isdigit())
    whitespace = sum(1 for c in text if c.isspace())
    total_chars = len(text)
    other_chars = total_chars - (total_letters + digits + whitespace)

    expected_ratio = (total_letters / max(total_chars - whitespace, 1)) if total_chars > 0 else 0.0

    dominant_script = "none"
    if total_letters > 0:
        counts = {"en": latin_count, "hi": deva_count, "kn": kan_count}
        dominant_script = max(counts, key=counts.get)

    return {
        "latin": latin_count,
        "devanagari": deva_count,
        "kannada": kan_count,
        "total_letters": total_letters,
        "digits": digits,
        "other_chars": other_chars,
        "total_chars": total_chars,
        "expected_script_ratio": round(expected_ratio, 4),
        "dominant_script": dominant_script,
    }


def detect_legacy_font_garbage(text: str) -> Tuple[bool, str]:
    """
    Heuristic check to detect legacy Indian font garbage (e.g. Kruti Dev / Shree Lipi
    text pasted as ASCII without Unicode encoding).
    """
    if len(text.strip()) < 30:
        return False, ""

    # Check 1: High density of ASCII punctuation / brackets inside words
    legacy_sym_count = len(RE_LEGACY_SYMBOLS.findall(text))
    sym_ratio = legacy_sym_count / max(len(text), 1)
    if legacy_sym_count > 10 and sym_ratio > 0.04:
        return True, f"High legacy symbol density ({sym_ratio:.2%})"

    # Check 2: Unnatural 6+ consecutive ASCII consonants with no vowels (frequent in unmapped font glyphs)
    consonant_matches = RE_CONSECUTIVE_CONSONANTS_ASCII.findall(text)
    if len(consonant_matches) > 5:
        return True, f"Unnatural consonant clusters ({len(consonant_matches)} found)"

    # Check 3: Private Use Area (PUA) characters
    pua_count = sum(1 for ch in text if "\uE000" <= ch <= "\uF8FF")
    if pua_count > 5:
        return True, f"PUA unmapped glyphs ({pua_count} characters)"

    return False, ""


def check_page_quality(page_dict: Dict[str, Any], min_chars: int = 15) -> Dict[str, Any]:
    """
    Evaluate extraction quality for a single page.
    Never crashes; captures error conditions as flags in the quality report.

    Args:
        page_dict: Dict with at least 'page' and 'text'.
        min_chars: Minimum non-whitespace characters to pass.

    Returns:
        Structured quality report:
        {
            "doc_id": str,
            "page": int,
            "passed": bool,
            "flags": List[str],
            "script_dist": Dict[str, Any],
            "text_length": int,
        }
    """
    try:
        text = page_dict.get("text", "")
        doc_id = page_dict.get("doc_id", "unknown")
        page_num = page_dict.get("page", 0)

        flags: List[str] = []
        stripped = text.strip()

        # Check 1: Empty or near-empty text
        if len(stripped) < min_chars:
            flags.append("EMPTY_OR_SPARSE_TEXT")

        # Check 2: Script distribution
        dist = analyze_script_distribution(text)
        if dist["total_chars"] >= min_chars and dist["expected_script_ratio"] < 0.25:
            flags.append("LOW_EXPECTED_SCRIPT_RATIO")

        # Check 3: Legacy font garbage
        is_garbage, reason = detect_legacy_font_garbage(text)
        if is_garbage:
            flags.append(f"LEGACY_FONT_GARBAGE: {reason}")

        passed = len(flags) == 0

        return {
            "doc_id": doc_id,
            "page": page_num,
            "passed": passed,
            "flags": flags,
            "script_dist": dist,
            "text_length": len(text),
        }

    except Exception as e:
        # Non-crashing resilience
        return {
            "doc_id": page_dict.get("doc_id", "unknown"),
            "page": page_dict.get("page", 0),
            "passed": False,
            "flags": [f"EXCEPTION_DURING_CHECK: {str(e)}"],
            "script_dist": {},
            "text_length": 0,
        }


def filter_valid_pages(
    pages: List[Dict[str, Any]],
    min_chars: int = 15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter extracted pages into accepted pages and a list of quality reports.

    Returns:
        valid_pages: List of original page dicts that passed the quality gate.
        reports: List of quality reports for all pages.
    """
    valid_pages = []
    reports = []

    for p in pages:
        rep = check_page_quality(p, min_chars=min_chars)
        reports.append(rep)
        if rep["passed"]:
            # Attach detected dominant script to page metadata
            p_augmented = dict(p)
            p_augmented["lang_guess"] = rep["script_dist"].get("dominant_script", "en")
            valid_pages.append(p_augmented)

    return valid_pages, reports
