"""Trust gate decision module based on retrieval confidence thresholds tau_hi and tau_lo."""

from enum import Enum
import sys
from typing import Any, Dict, Optional, Tuple

from govdoc.config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class TrustDecision(str, Enum):
    ANSWER = "ANSWER"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    ABSTAIN = "ABSTAIN"


def evaluate_trust_gate(
    score: float,
    tau_hi: Optional[float] = None,
    tau_lo: Optional[float] = None,
) -> Tuple[TrustDecision, Dict[str, Any]]:
    """
    Pure function to evaluate retrieval score against confidence thresholds.

    Decisions:
    - ANSWER: score >= tau_hi (LLM generation permitted)
    - EVIDENCE_ONLY: tau_lo <= score < tau_hi (Show retrieved evidence, but do not hallucinate answer)
    - ABSTAIN: score < tau_lo (No relevant information; LLM call forbidden per Rule 4)

    Args:
        score: Highest cosine similarity score among retrieved chunks.
        tau_hi: High threshold (default from config/config.yaml: 0.65)
        tau_lo: Low threshold (default from config/config.yaml: 0.45)

    Returns:
        Tuple of (TrustDecision, details_dict)
    """
    if tau_hi is None or tau_lo is None:
        cfg = load_config()
        r_cfg = cfg.get("retrieval", {})
        if tau_hi is None:
            tau_hi = float(r_cfg.get("tau_hi", 0.65))
        if tau_lo is None:
            tau_lo = float(r_cfg.get("tau_lo", 0.45))

    score_val = float(score)

    if score_val >= tau_hi:
        decision = TrustDecision.ANSWER
    elif score_val >= tau_lo:
        decision = TrustDecision.EVIDENCE_ONLY
    else:
        decision = TrustDecision.ABSTAIN

    details = {
        "score": round(score_val, 4),
        "tau_hi": tau_hi,
        "tau_lo": tau_lo,
        "decision": decision.value,
        "allow_llm_generation": decision == TrustDecision.ANSWER,
    }

    return decision, details
