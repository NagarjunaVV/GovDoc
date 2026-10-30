"""
Statistical significance and bootstrap resampling module (plan.md section 4, M7).

Computes paired bootstrap resampling (default 1,000 resamples, 95% CI) between the fine-tuned
summarizer and baselines, adhering to Non-Negotiable Technical Rules 6 and 10 (fixed seeds).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from govdoc.config import load_config


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    alternative: str = "greater",
) -> Dict[str, Any]:
    """
    Perform paired bootstrap hypothesis test and confidence interval estimation.

    Args:
        scores_a: Sample scores for System A (e.g., Fine-tuned model).
        scores_b: Sample scores for System B (e.g., Baseline B1 or B2).
        n_resamples: Number of bootstrap resamples (default: 1000).
        confidence_level: Confidence level for CI (default: 0.95).
        seed: Random seed for reproducibility (default: 42).
        alternative: "greater" (A > B), "two-sided", or "less".

    Returns:
        Dict containing mean_a, mean_b, delta, ci_lower, ci_upper, p_value, and significant.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Scores must have equal length: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) == 0:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "delta": 0.0,
            "ci_95": [0.0, 0.0],
            "p_value": 1.0,
            "significant": False,
        }

    arr_a = np.array(scores_a, dtype=np.float64)
    arr_b = np.array(scores_b, dtype=np.float64)
    n = len(arr_a)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    observed_delta = mean_a - mean_b

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)

    # Paired resampling: resample sample indices
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        sample_a = arr_a[indices]
        sample_b = arr_b[indices]
        diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    # Calculate confidence interval for the delta
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(diffs, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))

    # Compute p-value under null hypothesis (H0: mean(delta) <= 0)
    # Centered diffs for null distribution
    null_diffs = diffs - observed_delta
    if alternative == "greater":
        # Fraction of null replicates >= observed_delta
        p_val = float(np.mean(null_diffs >= observed_delta))
    elif alternative == "less":
        p_val = float(np.mean(null_diffs <= observed_delta))
    else:  # two-sided
        p_val = float(np.mean(np.abs(null_diffs) >= np.abs(observed_delta)))

    # Fallback empirical bound: fraction where diff <= 0
    if alternative == "greater" and observed_delta > 0:
        emp_p = float(np.mean(diffs <= 0.0))
        p_val = min(p_val, emp_p)

    significant = bool(p_val < (1.0 - confidence_level))

    return {
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(observed_delta, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(p_val, 4),
        "significant": significant,
    }


def bootstrap_ci_single(
    scores: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval for a single series."""
    if not scores:
        return 0.0, 0.0, 0.0

    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    return round(mean_val, 4), round(ci_low, 4), round(ci_high, 4)
