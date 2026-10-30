"""End-to-end grounded RAG answering pipeline combining retrieval, trust gate, and LLM generation."""

import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from govdoc.config import load_config
from govdoc.retrieve.retriever import MultilingualRetriever
from govdoc.retrieve.trust_gate import TrustDecision
from govdoc.generate.llm_client import LLMClient
from govdoc.generate.prompts import build_grounded_prompt, SYSTEM_PROMPT
from govdoc.generate.verifier import verify_grounded_answer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Absence responses by language mode
NOT_FOUND_RESPONSES = {
    "en": "Information not found in the provided document.",
    "hi": "प्रदत्त दस्तावेज़ में यह जानकारी उपलब्ध नहीं है।",
    "kn": "ಒದಗಿಸಲಾದ ದಾಖಲೆಯಲ್ಲಿ ಈ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.",
    "hinglish": "Provided document mein yeh information available nahi hai.",
    "kanglish": "Provided document nalli ee mahiti labhyavilla.",
}


class RAGPipeline:
    """End-to-end grounded RAG answering system for multilingual government notifications."""

    def __init__(
        self,
        collection_name: str = "govdoc_corpus",
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[MultilingualRetriever] = None,
    ):
        self.cfg = load_config()
        self.collection_name = collection_name
        self.retriever = retriever or MultilingualRetriever(collection_name=collection_name)
        self.llm_client = llm_client or LLMClient(self.cfg)

    def answer_query(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 3,
        tau_hi: Optional[float] = None,
        tau_lo: Optional[float] = None,
        allow_evidence_only_generation: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Retrieve top chunks and evaluate trust gate
        2. If ABSTAIN: strictly bypass LLM per Non-Negotiable Rule 4
        3. If EVIDENCE_ONLY: generate grounded answer from evidence chunks when
           allow_evidence_only_generation=True (barrier removed for rephrased questions)
        4. If ANSWER: construct grounded prompt, call LLM, and verify facts

        Returns:
            Structured answer dict with answer, citations, decision, and verifier report.
        """
        # 1. Retrieval & Trust Gate
        retrieval_res = self.retriever.retrieve(
            query=query,
            mode=mode,
            top_k=top_k,
            tau_hi=tau_hi,
            tau_lo=tau_lo,
        )
        decision = retrieval_res["decision"]
        chunks = retrieval_res["chunks"][:top_k]
        resolved_mode = retrieval_res["mode"]

        # Rule 4: Never call the LLM when trust gate says ABSTAIN
        if decision == TrustDecision.ABSTAIN:
            absent_msg = NOT_FOUND_RESPONSES.get(resolved_mode, NOT_FOUND_RESPONSES["en"])
            return {
                "query": query,
                "answer": absent_msg,
                "decision": decision,
                "top_score": retrieval_res["top_score"],
                "citations": [],
                "verifier": {"pass": True, "missing": [], "extracted": []},
                "llm_called": False,
                "evidence_chunks": [],
            }

        # Handle EVIDENCE_ONLY when generation is explicitly disabled
        if decision == TrustDecision.EVIDENCE_ONLY and not allow_evidence_only_generation:
            evidence_citations = [
                {"doc_id": c["doc_id"], "page": c["page"], "score": c["score"]}
                for c in chunks
            ]
            return {
                "query": query,
                "answer": (
                    f"Evidence retrieved with moderate confidence (Score: {retrieval_res['top_score']:.2f}). "
                    f"Direct LLM generation bypassed to prevent speculation. Consult cited sections."
                ),
                "decision": decision,
                "top_score": retrieval_res["top_score"],
                "citations": evidence_citations,
                "verifier": {"pass": True, "missing": [], "extracted": []},
                "llm_called": False,
                "evidence_chunks": chunks,
            }

        # Evidence is present (ANSWER or EVIDENCE_ONLY with barrier removed):
        # Proceed to Grounded LLM Generation
        prompt = build_grounded_prompt(query=query, evidence_chunks=chunks, mode=resolved_mode)
        llm_res = self.llm_client.chat(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=256,
        )
        raw_answer = llm_res["content"].strip()

        # Map citation brackets [1], [2] to actual [Doc, Page]
        citations: List[Dict[str, Any]] = []
        for idx, c in enumerate(chunks):
            bracket = f"[{idx + 1}]"
            if bracket in raw_answer:
                citations.append({
                    "citation_id": idx + 1,
                    "doc_id": c["doc_id"],
                    "page": c["page"],
                    "score": c["score"],
                })

        # If LLM didn't insert explicit brackets, associate top chunk citation
        if not citations and chunks:
            citations.append({
                "citation_id": 1,
                "doc_id": chunks[0]["doc_id"],
                "page": chunks[0]["page"],
                "score": chunks[0]["score"],
            })

        # Run verification check
        verif_report = verify_grounded_answer(raw_answer, chunks)

        return {
            "query": query,
            "answer": raw_answer,
            "decision": decision,
            "top_score": retrieval_res["top_score"],
            "citations": citations,
            "verifier": verif_report,
            "llm_called": True,
            "evidence_chunks": chunks,
        }
