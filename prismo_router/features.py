"""Request feature extraction for PrismoRouter."""

from __future__ import annotations

import re

from .models import RequestFeatures
from .safety import detect_jailbreak

_CODE_PATTERNS = re.compile(
    r"\b(code|function|implement|debug|refactor|class |def |import |return |"
    r"algorithm|database|query|script|python|javascript|typescript|java|"
    r"golang|rust|sql)\b",
    re.IGNORECASE,
)
_MATH_PATTERNS = re.compile(r"\b(prove|calculate|derive|equation|theorem|integral|probability)\b", re.IGNORECASE)
_PLANNING_PATTERNS = re.compile(r"\b(design|architect|plan|strategy|roadmap|system design|trade-?offs)\b", re.IGNORECASE)
_SUMMARIZE_PATTERNS = re.compile(r"\b(summarize|summary|tldr|tl;dr|key points|main points)\b", re.IGNORECASE)
_CREATIVE_PATTERNS = re.compile(r"\b(write a|poem|story|creative|haiku|fiction|narrative)\b", re.IGNORECASE)
_FACTUAL_PATTERNS = re.compile(r"\b(what is|who is|define|when did|where is|how many)\b", re.IGNORECASE)
_ANALYTICAL = re.compile(r"\b(compare|analyze|explain|evaluate|assess|why does|why is)\b", re.IGNORECASE)
_CONSTRAINTS = re.compile(r"\b(must|should|require|ensure|always|never)\b", re.IGNORECASE)
_JSON_BLOCK = re.compile(r"\{[\s\S]*?:[\s\S]*?\}")
_LIST = re.compile(r"^\s*(?:[-*]|\d+[.)])\s", re.MULTILINE)
_CODE_FENCE = re.compile(r"```")

LEGAL_TERMS = frozenset(["legal", "lawyer", "attorney", "lawsuit", "contract", "liability", "litigation"])
MEDICAL_TERMS = frozenset(["medical", "medication", "diagnosis", "treatment", "prescription", "symptoms", "patient", "healthcare", "clinical"])
FINANCIAL_TERMS = frozenset(["financial", "investment", "securities", "fiduciary", "tax"])


def estimate_tokens(text: str) -> int:
    """Fast local token estimate. Avoids provider-specific tokenizers."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35))


def extract_features(prompt: str) -> RequestFeatures:
    """Extract deterministic routing features from a prompt."""
    text = prompt or ""
    lower = text.lower()
    jailbreak = detect_jailbreak(text)

    if _CODE_PATTERNS.search(text) or _CODE_FENCE.search(text):
        task = "code"
    elif _MATH_PATTERNS.search(text):
        task = "math"
    elif _PLANNING_PATTERNS.search(text):
        task = "planning"
    elif _SUMMARIZE_PATTERNS.search(text):
        task = "summarize"
    elif _CREATIVE_PATTERNS.search(text):
        task = "creative"
    elif _ANALYTICAL.search(text):
        task = "reasoning"
    else:
        task = "chat"

    if _FACTUAL_PATTERNS.search(text):
        question_type = "factual"
    elif _ANALYTICAL.search(text):
        question_type = "analytical"
    else:
        question_type = "unknown"

    domain = "general"
    risk_keywords: list[str] = []
    for label, terms in (
        ("legal", LEGAL_TERMS),
        ("medical", MEDICAL_TERMS),
        ("financial", FINANCIAL_TERMS),
    ):
        found = sorted(term for term in terms if term in lower)
        if found:
            domain = label
            risk_keywords.extend(found)
            break

    return RequestFeatures(
        prompt=text,
        token_estimate=estimate_tokens(text),
        detected_task=task,
        question_type=question_type,
        detected_domain=domain,
        risk_keywords=tuple(risk_keywords),
        has_code=bool(_CODE_PATTERNS.search(text) or _CODE_FENCE.search(text)),
        has_json=bool(_JSON_BLOCK.search(text)),
        has_list=bool(_LIST.search(text)),
        constraint_count=len(_CONSTRAINTS.findall(text)),
        jailbreak_detected=jailbreak.detected,
        jailbreak_confidence=jailbreak.confidence,
        jailbreak_category=jailbreak.category,
    )

