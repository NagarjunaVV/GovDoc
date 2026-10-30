"""Grounded multi-lingual prompt templates adhering to citation and factual constraints."""

import sys
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SYSTEM_PROMPT = (
    "You are a strict, factual assistant for Indian government documents. "
    "You answer questions exclusively using the provided numbered evidence chunks. "
    "Every statement must be grounded with citations [n]. "
    "If the answer cannot be found in the evidence, state that the information is absent. "
    "Do not extrapolate or invent facts."
)

# 1. English prompt template
PROMPT_TEMPLATE_EN = """Instructions:
Answer the question below using ONLY the provided numbered evidence.
Every factual statement, amount, date, or organization must cite its source chunk using [n] (e.g., [1], [2]).
If the information is not present in the provided evidence, reply strictly: "Information not found in the provided document."
Answer in English concisely and truthfully.

Evidence:
{evidence_block}

Question: {query}

Grounded Answer:"""

# 2. Hindi prompt template
PROMPT_TEMPLATE_HI = """निर्देश:
नीचे दिए गए प्रश्न का उत्तर केवल दिए गए क्रमांकित साक्ष्यों के आधार पर दें।
प्रत्येक तथ्य, तिथि, संख्या अथवा संस्था के नाम के साथ स्रोत संख्या का संदर्भ [n] (जैसे [1], [2]) अवश्य लिखें।
यदि प्रश्न का उत्तर दिए गए साक्ष्य में उपलब्ध नहीं है, तो केवल यही लिखें: "प्रदत्त दस्तावेज़ में यह जानकारी उपलब्ध नहीं है।"
हिंदी में संक्षिप्त और सत्यनिष्ठ उत्तर दें।

साक्ष्य:
{evidence_block}

प्रश्न: {query}

तथ्यपरक उत्तर:"""

# 3. Kannada prompt template
PROMPT_TEMPLATE_KN = """ಸೂಚನೆಗಳು:
ಕೆಳಗಿನ ಪ್ರಶ್ನೆಗೆ ಒದಗಿಸಲಾದ ಸಂಖ್ಯೆಯುಳ್ಳ ಆಧಾರಗಳಿಂದ (Evidence) ಮಾತ್ರ ಉತ್ತರಿಸಿ.
ಪ್ರತಿಯೊಂದು ಸತ್ಯಾಂಶ, ದಿನಾಂಕ, ಮೊತ್ತ ಅಥವಾ ಇಲಾಖೆಯ ಹೆಸರಿನೊಂದಿಗೆ ಆಧಾರದ ಸಂಖ್ಯೆಯನ್ನು [n] (ಉದಾಹರಣೆಗೆ [1], [2]) ಎಂದು ನಮೂದಿಸಬೇಕು.
ಒದಗಿಸಲಾದ ದಾಖಲೆಯಲ್ಲಿ ಉತ್ತರ ಲಭ್ಯವಿಲ್ಲದಿದ್ದರೆ, ಕಡ್ಡಾಯವಾಗಿ "ಒದಗಿಸಲಾದ ದಾಖಲೆಯಲ್ಲಿ ಈ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ." ಎಂದು ಮಾತ್ರ ಉತ್ತರಿಸಿ.
ಕನ್ನಡದಲ್ಲಿ ನಿಖರವಾಗಿ ಉತ್ತರಿಸಿ.

ಆಧಾರಗಳು (Evidence):
{evidence_block}

ಪ್ರಶ್ನೆ: {query}

ನಿಖರ ಉತ್ತರ:"""

# 4. Hinglish prompt template
PROMPT_TEMPLATE_HINGLISH = """Instructions:
Neeche diye gaye question ka answer SIRF provide kiye gaye numbered evidence [n] ke aadhar par dein.
Har fact, date, amount aur organization ke saath reference [n] (jaise [1], [2]) zaroor cite karein.
Agar information provided evidence mein nahi hai, toh strictly yahi likhein: "Provided document mein yeh information available nahi hai."
Hinglish (Hindi written in Roman script) mein clear aur truthful answer dein.

Evidence:
{evidence_block}

Question: {query}

Grounded Answer:"""

# 5. Kanglish prompt template
PROMPT_TEMPLATE_KANGLISH = """Instructions:
Kelagina question ge ONLY provide maadida numbered evidence [n] inda maatra answer maadi.
Pratiyondu fact, date, amount matthu organization jothe source reference [n] (eg: [1], [2]) kaddayavagi cite maadi.
Provided evidence nalli information illa andre, strictly "Provided document nalli ee mahiti labhyavilla." anta maatra reply maadi.
Kanglish (Kannada written in Roman script) nalli factual answer maadi.

Evidence:
{evidence_block}

Question: {query}

Grounded Answer:"""

TEMPLATES: Dict[str, str] = {
    "en": PROMPT_TEMPLATE_EN,
    "hi": PROMPT_TEMPLATE_HI,
    "kn": PROMPT_TEMPLATE_KN,
    "hinglish": PROMPT_TEMPLATE_HINGLISH,
    "kanglish": PROMPT_TEMPLATE_KANGLISH,
}


def format_evidence_block(chunks: List[Dict[str, Any]]) -> str:
    """Format evidence chunks into numbered blocks with document and page metadata."""
    if not chunks:
        return "[No evidence provided]"

    blocks = []
    for idx, c in enumerate(chunks):
        doc_id = c.get("doc_id", "doc")
        page = c.get("page", 1)
        text = c.get("text", "").strip()
        blocks.append(f"[{idx + 1}] Document: {doc_id}, Page: {page}\n{text}")

    return "\n\n".join(blocks)


def build_grounded_prompt(
    query: str,
    evidence_chunks: List[Dict[str, Any]],
    mode: str = "en",
) -> str:
    """
    Construct grounded prompt for the specified language mode.

    Args:
        query: User query string.
        evidence_chunks: Ranked retrieved chunks.
        mode: Target language mode in {'en', 'hi', 'kn', 'hinglish', 'kanglish'}.

    Returns:
        Formatted prompt string.
    """
    mode_key = mode.lower() if mode.lower() in TEMPLATES else "en"
    template = TEMPLATES[mode_key]
    evidence_block = format_evidence_block(evidence_chunks)

    return template.format(
        evidence_block=evidence_block,
        query=query.strip(),
    )
