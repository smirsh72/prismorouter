"""
Shared Utility — Request Feature Extractor.

Portions of this feature extraction approach are adapted from vLLM Semantic
Router.

Extracts structured features from a UnifiedRequest for the heuristic scorer.
Must produce identical output for streaming and non-streaming requests.

This is pure extraction logic — no scoring, no routing decisions, no DB calls.
"""
import re
try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional


# =============================================================================
# DETECTION PATTERNS (compiled once at import time)
# =============================================================================

# Task type signals — scored by the LAST user message only
_FACTUAL_PATTERNS = re.compile(
    r"\b(what is|who is|define|when did|where is|how many)\b", re.IGNORECASE
)
_TRANSLATION_PATTERNS = re.compile(
    r"\b(translate|reformat|convert|format as)\b", re.IGNORECASE
)
_SUMMARIZE_PATTERNS = re.compile(
    r"\b(summarize|summary|tldr|tl;dr|key points|main points)\b", re.IGNORECASE
)
_CREATIVE_PATTERNS = re.compile(
    r"\b(write a|poem|story|creative|haiku|fiction|narrative)\b", re.IGNORECASE
)
_CODE_PATTERNS = re.compile(
    r"\b(code|function|implement|debug|refactor|class |def |import |return |algorithm|data structure|API|endpoint|server|database|query|script|program|module|library|package|compile|runtime|syntax|variable|array|loop|recursion|binary tree|linked list|hash|sort|search|insert|delete|rotate|traverse|python|javascript|typescript|java|golang|rust|sql)\b", re.IGNORECASE
)
_CODE_FENCE = re.compile(r"```")
_REASONING_PATTERNS = re.compile(
    r"\b(compare|analyze|explain why|step by step|pros and cons|evaluate|assess)\b", re.IGNORECASE
)
_MATH_PATTERNS = re.compile(
    r"\b(prove|calculate|derive|equation|theorem|integral|probability)\b", re.IGNORECASE
)
_MATH_OPERATORS = re.compile(r"[+\-*/=<>]{2,}|\\frac|\\sum|\\int|\d+\s*[+\-*/]\s*\d+")
_PLANNING_PATTERNS = re.compile(
    r"\b(design|architect|plan|strategy|roadmap|system design|trade-?offs|draft|outline|structure|framework|proposal)\b", re.IGNORECASE
)

# Domain risk — Layer 1 keyword sets (single source of truth for domain classification)
LEGAL_TERMS = frozenset(["legal", "lawyer", "attorney", "lawsuit", "contract", "liability", "litigation"])
MEDICAL_TERMS = frozenset(["medical", "medication", "diagnosis", "treatment", "prescription", "symptoms", "patient", "healthcare", "clinical"])
FINANCIAL_TERMS = frozenset(["financial", "investment", "securities", "fiduciary", "tax"])

# Regex built from the sets above — no duplication
_DOMAIN_KEYWORDS = re.compile(
    r"\b(" + "|".join(LEGAL_TERMS | MEDICAL_TERMS | FINANCIAL_TERMS) + r")\b",
    re.IGNORECASE,
)

# Domain risk — Layer 2 action verbs
_HIGH_RISK_VERBS = re.compile(
    r"\b(terminate|liable|enforceable|binding|indemnify|prosecute|malpractice|fiduciary|negligence|statutory)\b",
    re.IGNORECASE,
)
_MEDIUM_RISK_VERBS = re.compile(
    r"\b(compliant|obligated|penalties|sue|audit|regulated|hipaa|gdpr|sox|pci)\b",
    re.IGNORECASE,
)

# Domain risk — Layer 3 structural signals
_ADVICE_PATTERNS = re.compile(
    r"\b(should i|do you recommend|advise|what should|is it safe to)\b", re.IGNORECASE
)
_PROPER_NOUN_LEGAL_FIN = re.compile(
    r"\b[A-Z][a-z]+ (?:v\.|vs\.?|Inc\.|Corp\.|LLC|Ltd\.|Act|Code|§)\b"
)

# Structural complexity
_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(must|should|require|ensure|always|never)\b", re.IGNORECASE
)
_JSON_BLOCK = re.compile(r"\{[\s\S]*?:[\s\S]*?\}")
_NUMBERED_LIST = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_BULLET_LIST = re.compile(r"^\s*[-*]\s", re.MULTILINE)

# Question type detection
_HOW_TO = re.compile(r"\b(how to|how do|how can|how should)\b", re.IGNORECASE)
_OPINION = re.compile(r"\b(do you think|opinion|your view|your take)\b", re.IGNORECASE)
_ANALYTICAL = re.compile(r"\b(why does|why is|why do|explain|reason|cause)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RequestFeatures:
    """
    Extracted features from a single request.
    Immutable value object — scorer reads this, never mutates it.
    """
    # Text signals
    last_user_message: str
    system_prompt_length: int
    conversation_turns: int
    total_token_estimate: int

    # Structural signals
    has_code_fences: bool
    has_json_blocks: bool
    has_numbered_list: bool
    has_bullet_list: bool
    constraint_keyword_count: int

    # Classification signals
    question_type: str  # factual, how-to, opinion, creative, analytical, unknown
    detected_domain: str  # legal, medical, financial, technical, general
    detected_task: str  # chat, summarize, extract, code, math, planning, creative
    risk_keywords_found: List[str] = field(default_factory=list)

    # Language detection
    detected_language: str = "en"  # ISO 639-1 code
    language_confidence: float = 1.0

    # Jailbreak/prompt injection detection
    jailbreak_detected: bool = False
    jailbreak_confidence: float = 0.0
    jailbreak_category: Optional[str] = None


def _estimate_tokens_fallback(text: str) -> int:
    """Heuristic fallback when tokenizer metadata is unavailable."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@lru_cache(maxsize=16)
def _get_tiktoken_encoding(model: str):
    """Resolve and cache tokenizer metadata so only the first call pays init cost."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    except Exception:
        return None


def extract_features(
    messages: Optional[list] = None,
    input_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o",
) -> RequestFeatures:
    """
    Extract features from a request.

    Args:
        messages: Chat Completions API messages (list of dicts with role/content).
        input_text: Responses API input string.
        system_prompt: Explicit system prompt (extracted from messages if not provided).
        model: Model name for tiktoken encoding selection.

    Returns:
        RequestFeatures with all fields populated.
    """
    # --- Resolve last user message, system prompt, and turn count ---
    last_user_message = ""
    sys_prompt = system_prompt or ""
    conversation_turns = 0

    if messages:
        for msg in messages:
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            content = str(content) if content else ""

            if role == "system" and not system_prompt:
                sys_prompt = content
            elif role == "user":
                last_user_message = content
                conversation_turns += 1
    elif input_text:
        last_user_message = str(input_text)
        conversation_turns = 1

    # --- Token estimate ---
    encoding = _get_tiktoken_encoding(model)

    full_text = last_user_message
    if messages:
        full_text = " ".join(
            str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in messages
        )
    if not full_text:
        total_token_estimate = 0
    elif encoding:
        total_token_estimate = len(encoding.encode(full_text))
    else:
        total_token_estimate = _estimate_tokens_fallback(full_text)

    # --- Structural signals (from last user message) ---
    text = last_user_message
    has_code_fences = bool(_CODE_FENCE.search(text))
    has_json_blocks = bool(_JSON_BLOCK.search(text))
    has_numbered_list = bool(_NUMBERED_LIST.search(text))
    has_bullet_list = bool(_BULLET_LIST.search(text))
    constraint_keyword_count = len(_CONSTRAINT_KEYWORDS.findall(text))

    # --- Question type ---
    question_type = _detect_question_type(text)

    # --- Domain detection ---
    detected_domain, risk_keywords = _detect_domain(text)

    # --- Task detection ---
    detected_task = _detect_task(text, has_code_fences)

    # --- Language detection ---
    detected_language, language_confidence = _detect_language(text)

    # --- Jailbreak detection ---
    jailbreak_detected = False
    jailbreak_confidence = 0.0
    jailbreak_category = None
    try:
        from .config import settings as _cfg
        if _cfg.ROUTING_JAILBREAK_DETECTION:
            from .jailbreak_detector import detect_jailbreak
            jb = detect_jailbreak(text)
            jailbreak_detected = jb.detected
            jailbreak_confidence = jb.confidence
            jailbreak_category = jb.category
    except Exception:
        pass

    return RequestFeatures(
        last_user_message=last_user_message,
        system_prompt_length=len(sys_prompt),
        conversation_turns=conversation_turns,
        total_token_estimate=total_token_estimate,
        has_code_fences=has_code_fences,
        has_json_blocks=has_json_blocks,
        has_numbered_list=has_numbered_list,
        has_bullet_list=has_bullet_list,
        constraint_keyword_count=constraint_keyword_count,
        question_type=question_type,
        detected_domain=detected_domain,
        detected_task=detected_task,
        risk_keywords_found=risk_keywords,
        detected_language=detected_language,
        language_confidence=language_confidence,
        jailbreak_detected=jailbreak_detected,
        jailbreak_confidence=jailbreak_confidence,
        jailbreak_category=jailbreak_category,
    )


# =============================================================================
# INTERNAL DETECTORS
# =============================================================================

def _detect_question_type(text: str) -> str:
    """Classify the question type of the last user message."""
    if _FACTUAL_PATTERNS.search(text):
        return "factual"
    if _HOW_TO.search(text):
        return "how-to"
    if _OPINION.search(text):
        return "opinion"
    if _CREATIVE_PATTERNS.search(text):
        return "creative"
    if _ANALYTICAL.search(text):
        return "analytical"
    return "unknown"


def _detect_domain(text: str) -> tuple:
    """
    Detect domain and collect matched risk keywords.
    Returns (domain_label, list_of_matched_keywords).
    """
    risk_keywords: List[str] = []

    # Layer 1 — domain keywords
    for m in _DOMAIN_KEYWORDS.finditer(text):
        risk_keywords.append(m.group(0).lower())

    # Layer 2 — action verbs
    for m in _HIGH_RISK_VERBS.finditer(text):
        risk_keywords.append(m.group(0).lower())
    for m in _MEDIUM_RISK_VERBS.finditer(text):
        risk_keywords.append(m.group(0).lower())

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in risk_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    # Determine domain label from first matched keyword
    domain = "general"
    for kw in unique_keywords:
        if kw in LEGAL_TERMS:
            domain = "legal"
            break
        if kw in MEDICAL_TERMS:
            domain = "medical"
            break
        if kw in FINANCIAL_TERMS:
            domain = "financial"
            break

    # If no Layer 1 match but has code signals, mark technical
    if domain == "general" and (_CODE_PATTERNS.search(text) or _CODE_FENCE.search(text)):
        domain = "technical"

    return domain, unique_keywords


def _detect_language(text: str) -> tuple:
    """
    Detect the language of the input text.
    Returns (iso_code, confidence).

    Uses langdetect when available, defaults to "en" with 1.0 confidence.
    Gated by ROUTING_LANGUAGE_DETECTION feature flag.
    """
    if not text or len(text.strip()) < 10:
        return "en", 1.0

    try:
        from .config import settings
        if not settings.ROUTING_LANGUAGE_DETECTION:
            return "en", 1.0
    except Exception:
        return "en", 1.0

    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            top = results[0]
            return str(top.lang), float(top.prob)
    except ImportError:
        pass
    except Exception:
        pass

    return "en", 1.0


def prewarm_feature_extraction() -> None:
    """
    Move one-time tokenizer/language/jailbreak initialization out of the first
    live request path.
    """
    sample_text = "Hello there, please say hi in three words."
    sample_model = "gpt-4o-mini"

    _get_tiktoken_encoding(sample_model)
    _detect_language(sample_text)

    try:
        from .config import settings as _cfg
        if _cfg.ROUTING_JAILBREAK_DETECTION:
            from .jailbreak_detector import detect_jailbreak
            detect_jailbreak(sample_text)
    except Exception:
        pass


def _detect_task(text: str, has_code_fences: bool) -> str:
    """
    Detect the primary task type from the last user message.
    Returns the highest-signal match (not cumulative).
    """
    if _PLANNING_PATTERNS.search(text):
        return "planning"
    if _MATH_PATTERNS.search(text) or _MATH_OPERATORS.search(text):
        return "math"
    if _REASONING_PATTERNS.search(text):
        return "extract"  # "analysis/extraction" bucket
    if _CODE_PATTERNS.search(text) or has_code_fences:
        return "code"
    if _CREATIVE_PATTERNS.search(text):
        return "creative"
    if _SUMMARIZE_PATTERNS.search(text):
        return "summarize"
    return "chat"
