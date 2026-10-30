"""
Unicode-aware ROUGE evaluation module for Hindi, Kannada, and English (Non-Negotiable Technical Rule 2).

Standard rouge_score drops non-ASCII characters, producing corrupted zero scores for Indic scripts.
This module implements Unicode-aware tokenization, ROUGE-1, ROUGE-2, and ROUGE-L with strict
self-test identity validation (identical prediction and reference yields precision=1.0, recall=1.0, f1=1.0).
"""

from collections import Counter
import re
from typing import Any, Dict, List, Tuple
import unicodedata


def tokenize_indic(text: str) -> List[str]:
    """
    Unicode-aware tokenizer for Indic scripts (Hindi, Kannada) and English.
    
    Adheres strictly to Non-Negotiable Technical Rule 2:
    - Normalizes Unicode to NFC form.
    - Preserves Devanagari (\u0900-\u097F) and Kannada (\u0C80-\u0CFF) letters,
      matras, viramas, numerals, and Latin alphanumeric characters.
    - Strips punctuation and symbols.
    """
    if not isinstance(text, str) or not text:
        return []

    text = unicodedata.normalize("NFC", text.strip())

    # Replace punctuation characters with space, keeping word characters and combining marks
    # Regex keeps alphanumeric (Latin + Indic letters + digits) and combining marks (\u0300-\u036F, \u0900-\u097F, \u0C80-\u0CFF)
    cleaned = re.sub(r"[^\w\s\u0900-\u097F\u0C80-\u0CFF]", " ", text)
    tokens = [t for t in cleaned.split() if t]
    return tokens


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract character or word n-grams from a token list."""
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_rouge(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    Compute Unicode-aware ROUGE-N (n=1, 2) precision, recall, and F1.
    """
    pred_tokens = tokenize_indic(prediction)
    ref_tokens = tokenize_indic(reference)

    if not pred_tokens or not ref_tokens:
        score = 1.0 if (not pred_tokens and not ref_tokens) else 0.0
        return {"precision": score, "recall": score, "f1": score}

    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    if not pred_ngrams or not ref_ngrams:
        score = 1.0 if (not pred_ngrams and not ref_ngrams) else 0.0
        return {"precision": score, "recall": score, "f1": score}

    pred_counter = Counter(pred_ngrams)
    ref_counter = Counter(ref_ngrams)

    # Intersection count
    overlap = sum((pred_counter & ref_counter).values())

    precision = overlap / len(pred_ngrams)
    recall = overlap / len(ref_ngrams)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute Unicode-aware ROUGE-L precision, recall, and F1 based on LCS.
    
    Self-test property: identical strings always return precision=1.0, recall=1.0, f1=1.0.
    """
    pred_tokens = tokenize_indic(prediction)
    ref_tokens = tokenize_indic(reference)

    if not pred_tokens or not ref_tokens:
        score = 1.0 if (not pred_tokens and not ref_tokens) else 0.0
        return {"precision": score, "recall": score, "f1": score}

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_rouge_all(prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for a prediction/reference pair."""
    return {
        "rouge1": compute_ngram_rouge(prediction, reference, n=1),
        "rouge2": compute_ngram_rouge(prediction, reference, n=2),
        "rougeL": compute_rouge_l(prediction, reference),
    }


def compute_corpus_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute mean ROUGE-L across a corpus."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p_total, r_total, f1_total = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        res = compute_rouge_l(pred, ref)
        p_total += res["precision"]
        r_total += res["recall"]
        f1_total += res["f1"]

    n = len(predictions)
    return {
        "precision": round(p_total / n, 4),
        "recall": round(r_total / n, 4),
        "f1": round(f1_total / n, 4),
    }


def compute_corpus_rouge_all(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute mean ROUGE-1, ROUGE-2, and ROUGE-L across a corpus."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        zero_score = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"rouge1": zero_score, "rouge2": zero_score, "rougeL": zero_score}

    totals = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    n = len(predictions)
    for pred, ref in zip(predictions, references):
        pair_scores = compute_rouge_all(pred, ref)
        for metric, scores in pair_scores.items():
            for key in ["precision", "recall", "f1"]:
                totals[metric][key] += scores[key]

    return {
        metric: {k: round(v / n, 4) for k, v in scores.items()}
        for metric, scores in totals.items()
    }


def self_test_rouge() -> bool:
    """Self-test: identical strings must yield 1.0 for all ROUGE metrics across Hindi, Kannada, English."""
    samples = [
        ("Prime Minister Narendra Modi launched the initiative today.", "en"),
        ("प्रधानमंत्री नरेंद्र मोदी ने आज इस पहल की शुरुआत की।", "hi"),
        ("ಪ್ರಧಾನ ಮಂತ್ರಿ ನರೇಂದ್ರ ಮೋದಿ ಅವರು ಇಂದು ಈ ಉಪಕ್ರಮವನ್ನು ಪ್ರಾರಂಭಿಸಿದರು.", "kn"),
    ]
    for text, lang in samples:
        res = compute_rouge_all(text, text)
        for metric in ["rouge1", "rouge2", "rougeL"]:
            if res[metric]["f1"] != 1.0 or res[metric]["precision"] != 1.0 or res[metric]["recall"] != 1.0:
                raise AssertionError(f"ROUGE self-test failed for {lang}: {res}")
    return True


if __name__ == "__main__":
    self_test_rouge()
    print("ROUGE Unicode-aware self-test passed successfully (all identical scores == 1.0)!")
