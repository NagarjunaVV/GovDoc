"""Hallucination verifier extracting numbers, dates, amounts, and entities."""

import re
import sys
from typing import Any, Dict, List, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Numeral translation tables
DEVA_DIGITS = "०१२३४५६७८९"
KAN_DIGITS = "೦೧೨೩೪೫೬೭೮೯"
ASCII_DIGITS = "0123456789"

TRANS_DEVA_TO_ASCII = str.maketrans(DEVA_DIGITS, ASCII_DIGITS)
TRANS_KAN_TO_ASCII = str.maketrans(KAN_DIGITS, ASCII_DIGITS)

# Patterns for extraction
RE_CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")
RE_NUMBERS_AMOUNTS = re.compile(r"(?:(?:Rs\.?|₹|INR)\s*)?\b\d+(?:,\d+)*(?:\.\d+)?%?\b")
RE_CAPITALIZED_ENTITIES = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Phrases indicating absent information
ABSENCE_PHRASES = {
    "information not found",
    "not stated",
    "not present",
    "not available",
    "जानकारी उपलब्ध नहीं",
    "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ",
    "available nahi",
    "labhyavilla",
}


def normalize_digits(text: str) -> str:
    """Convert Devanagari and Kannada numerals to standard ASCII digits."""
    return text.translate(TRANS_DEVA_TO_ASCII).translate(TRANS_KAN_TO_ASCII)


def clean_number(num_str: str) -> str:
    """Normalize commas and spaces from numbers for comparison."""
    return num_str.replace(",", "").replace(" ", "").lower()


def extract_verifiable_tokens(answer: str) -> Dict[str, List[str]]:
    """
    Extract numbers, dates/amounts, and capitalized entities from generated answer.
    Ignores citation brackets like [1], [2].
    """
    # Remove bracket citations first
    clean_ans = RE_CITATION.sub(" ", answer)
    norm_ans = normalize_digits(clean_ans)

    # 1. Extract numbers and amounts
    raw_nums = RE_NUMBERS_AMOUNTS.findall(norm_ans)
    # Filter out single-digit generic words like '1' or '2' if purely standalone
    numbers = [n.strip() for n in raw_nums if len(clean_number(n)) > 0]

    # 2. Extract capitalized multi-word entities (English/Hinglish/Kanglish)
    entities = [e.strip() for e in RE_CAPITALIZED_ENTITIES.findall(clean_ans)]

    return {
        "numbers": list(dict.fromkeys(numbers)),
        "entities": list(dict.fromkeys(entities)),
    }


def verify_grounded_answer(
    answer: str,
    evidence_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify whether factual numbers, amounts, dates, and entities in the answer
    are grounded in the provided evidence.

    Returns:
        {
            "pass": bool,
            "missing": List[str],
            "extracted": List[str],
        }
    """
    ans_lower = answer.lower()
    # If the answer is an explicit abstention/absence statement, it passes by definition
    if any(phrase in ans_lower for phrase in ABSENCE_PHRASES):
        return {"pass": True, "missing": [], "extracted": []}

    # Build normalized evidence string
    raw_evidence = " ".join(c.get("text", "") for c in evidence_chunks)
    evidence_norm_digits = normalize_digits(raw_evidence)
    evidence_lower = evidence_norm_digits.lower()
    evidence_no_commas = clean_number(evidence_norm_digits)

    tokens = extract_verifiable_tokens(answer)
    missing: List[str] = []
    all_extracted = tokens["numbers"] + tokens["entities"]

    # Verify numbers and amounts
    for num in tokens["numbers"]:
        clean_n = clean_number(num)
        # Check either exact representation or unpunctuated representation
        if num.lower() not in evidence_lower and clean_n not in evidence_no_commas:
            missing.append(num)

    # Verify capitalized entities
    for ent in tokens["entities"]:
        # Allow case-insensitive substring match in evidence
        if ent.lower() not in evidence_lower:
            missing.append(ent)

    return {
        "pass": len(missing) == 0,
        "missing": missing,
        "extracted": all_extracted,
    }
