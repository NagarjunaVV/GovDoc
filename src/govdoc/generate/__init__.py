"""GovDoc-RAG generation package."""

from govdoc.generate.llm_client import LLMClient
from govdoc.generate.prompts import build_grounded_prompt, SYSTEM_PROMPT, TEMPLATES
from govdoc.generate.verifier import verify_grounded_answer, normalize_digits
from govdoc.generate.rag_pipeline import RAGPipeline

__all__ = [
    "LLMClient",
    "build_grounded_prompt",
    "SYSTEM_PROMPT",
    "TEMPLATES",
    "verify_grounded_answer",
    "normalize_digits",
    "RAGPipeline",
]
