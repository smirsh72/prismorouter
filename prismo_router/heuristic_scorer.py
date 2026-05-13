"""
Infrastructure/Strategy Layer — Heuristic Scorer (Layer 1 Router).

Portions of this heuristic scoring approach are adapted from vLLM Semantic
Router.

Fully deterministic. No ML, no embeddings, no DB calls.
Takes RequestFeatures, returns a RoutingProfile + HeuristicScores.

Scoring dimensions (Section 5.1):
  - Task Type:              35%
  - Domain Risk:            30%
  - Structural Complexity:  25%
  - Token Context Signal:   10%

Final score maps to a minimum tier floor via configurable thresholds.
Supports quality-preference threshold shifts and hysteresis.
"""
import re
import logging
from typing import Optional, Literal

from .feature_extractor import (
    RequestFeatures,
    LEGAL_TERMS,
    MEDICAL_TERMS,
    FINANCIAL_TERMS,
)
from .routing_profile import RoutingProfile, HeuristicScores

logger = logging.getLogger(__name__)


# =============================================================================
# WEIGHTS (Section 5.1)
# =============================================================================
WEIGHT_TASK_TYPE = 0.35
WEIGHT_DOMAIN_RISK = 0.30
WEIGHT_STRUCTURE = 0.25
WEIGHT_TOKEN_SIGNAL = 0.10

# =============================================================================
# TIER BOUNDARY THRESHOLDS — default (medium / auto)
# =============================================================================
DEFAULT_BOUNDARIES = {
    "nano_ceiling": 0.25,
    "mini_ceiling": 0.50,
    "base_ceiling": 0.75,
}

# Quality preference shifts (Section 5.4)
QUALITY_BOUNDARIES = {
    "low": {"nano_ceiling": 0.30, "mini_ceiling": 0.55, "base_ceiling": 0.80},
    "medium": DEFAULT_BOUNDARIES,
    "auto": DEFAULT_BOUNDARIES,
    "high": {"nano_ceiling": 0.15, "mini_ceiling": 0.40, "base_ceiling": 0.65},
}

# Task complexity floors: prevent high-complexity tasks from being
# dragged to mini/nano by zero domain_risk (which wastes 30% weight).
# Floor values correspond to tier boundaries so the task always reaches
# at least the appropriate tier.
# Only applied when the query has non-trivial substance (length >= 50 or
# structure_score >= 0.25) to avoid escalating trivial queries like "2+2?".
_TASK_COMPLEXITY_FLOORS = {
    "math": 0.52,       # → base tier (above mini_ceiling 0.50)
    "planning": 0.52,   # → base tier
    "code": 0.52,       # → base tier (code generation needs capability)
    "reasoning": 0.45,  # → upper mini, bumps to base near boundary
}
_FLOOR_MIN_TEXT_LEN = 50
_FLOOR_MIN_STRUCTURE = 0.25

# Hysteresis dead zone (Section 5.3): +/- 0.03 around each boundary
HYSTERESIS_BAND = 0.03

# Near-boundary logging threshold (Section 5.3): +/- 0.05
NEAR_BOUNDARY_BAND = 0.05


# =============================================================================
# TASK TYPE SCORING (Section 5.1 — 35% weight)
# =============================================================================
# Scored on last user message only. Take the MAX matched signal.

_TASK_SCORES = {
    "factual_lookup": 0.1,
    "translation": 0.2,
    "summarize": 0.3,
    "creative": 0.3,
    "chat": 0.4,
    "code": 0.6,
    "reasoning": 0.7,
    "math": 0.8,
    "planning": 0.8,
}

# System prompt bonus
_SYSTEM_PROMPT_BONUS_THRESHOLD = 500
_SYSTEM_PROMPT_BONUS = 0.15
_GENERATIVE_TASKS = {"chat", "planning", "creative", "code", "summarize"}


def _score_task_type(features: RequestFeatures, domain_risk_score: float = 0.0) -> float:
    """
    Score based on detected task type + system prompt bonus. Capped at 1.0.

    Domain-generative bonus: when domain risk is elevated (≥ 0.7) and the
    task is generative (planning, creative, code, chat) rather than a pure
    factual lookup, apply a +0.15 bonus. Generating legal contracts or
    medical summaries requires more capability than asking a factual question.
    """
    task_map = {
        "chat": "chat",
        "summarize": "summarize",
        "creative": "creative",
        "code": "code",
        "extract": "reasoning",  # analysis/extraction mapped to reasoning
        "math": "math",
        "planning": "planning",
    }

    # Also check question_type for factual/translation signals
    question_map = {
        "factual": "factual_lookup",
    }

    # Start with the detected_task score
    task_key = task_map.get(features.detected_task, "chat")
    score = _TASK_SCORES.get(task_key, 0.4)

    # Check if question_type gives a lower score (take max, but factual can pull down)
    q_key = question_map.get(features.question_type)
    if q_key:
        q_score = _TASK_SCORES.get(q_key, 0.4)
        # For factual lookups, use the lower score only if no higher task was detected
        if features.detected_task == "chat":
            score = q_score

    # Domain-generative bonus: generating high-risk domain content needs more capability
    # than asking a factual question about the domain.
    if domain_risk_score >= 0.7 and features.detected_task in _GENERATIVE_TASKS:
        if features.question_type != "factual":
            score += 0.15

    # System prompt > 500 chars bonus (additive)
    if features.system_prompt_length > _SYSTEM_PROMPT_BONUS_THRESHOLD:
        score += _SYSTEM_PROMPT_BONUS

    return min(score, 1.0)


# =============================================================================
# DOMAIN RISK SCORING (Section 5.1 — 30% weight)
# =============================================================================
# Three layers. Take MAX across all layers, capped at 1.0.

# Layer 2 high-risk verbs (for scoring — detection is in feature_extractor)
_HIGH_RISK_VERB_SET = frozenset([
    "terminate", "liable", "enforceable", "binding", "indemnify",
    "prosecute", "malpractice", "fiduciary", "negligence", "statutory",
])
_MEDIUM_RISK_VERB_SET = frozenset([
    "compliant", "obligated", "penalties", "sue", "audit",
    "regulated", "hipaa", "gdpr", "sox", "pci",
])

# Layer 1 domain keywords — derived from feature_extractor's single source of truth
_DOMAIN_KEYWORD_SET = LEGAL_TERMS | MEDICAL_TERMS | FINANCIAL_TERMS

# Layer 3 advice pattern (for additive bonus)
_ADVICE_RE = re.compile(
    r"\b(should i|do you recommend|advise|what should|is it safe to)\b", re.IGNORECASE
)
_PROPER_NOUN_LEGAL_FIN_RE = re.compile(
    r"\b[A-Z][a-z]+ (?:v\.|vs\.?|Inc\.|Corp\.|LLC|Ltd\.|Act|Code|§)\b"
)


def _score_domain_risk(features: RequestFeatures) -> float:
    """Score domain risk across three layers. Capped at 1.0."""
    score = 0.0

    risk_set = set(features.risk_keywords_found)

    # Layer 1 — domain keywords → 0.8
    if risk_set & _DOMAIN_KEYWORD_SET:
        score = max(score, 0.8)

    # Layer 2 — action verbs
    if risk_set & _HIGH_RISK_VERB_SET:
        score = max(score, 0.9)
    elif risk_set & _MEDIUM_RISK_VERB_SET:
        score = max(score, 0.6)

    # Layer 3 — structural risk signals (additive)
    text = features.last_user_message
    # Advice on high-risk topic
    if score >= 0.6 and _ADVICE_RE.search(text):
        score += 0.2
    # Proper nouns in legal/financial context
    if score >= 0.6 and _PROPER_NOUN_LEGAL_FIN_RE.search(text):
        score += 0.1

    return min(score, 1.0)


# =============================================================================
# STRUCTURAL COMPLEXITY SCORING (Section 5.1 — 25% weight)
# =============================================================================
# Take MAX matched signal, then add conversation-depth bonus. Capped at 1.0.

_CONVERSATION_DEPTH_THRESHOLD = 10
_CONVERSATION_DEPTH_BONUS = 0.15


def _score_structure(features: RequestFeatures) -> float:
    """Score structural complexity of the request."""
    text = features.last_user_message
    text_len = len(text)
    score = 0.0

    # Long detailed specification (> 500 chars)
    if text_len > 500:
        score = max(score, 0.8)
    # Multi-part / nested structure
    elif features.has_code_fences or features.has_json_blocks:
        score = max(score, 0.7)
    elif features.has_numbered_list:
        score = max(score, 0.7)
    # Multiple explicit constraints
    elif features.constraint_keyword_count > 2 or features.has_bullet_list:
        score = max(score, 0.6)
    # Single question with some context (50-200 chars)
    elif text_len >= 50:
        score = max(score, 0.3)
    # Single short question
    else:
        score = max(score, 0.1)

    # Conversation depth bonus
    if features.conversation_turns > _CONVERSATION_DEPTH_THRESHOLD:
        score += _CONVERSATION_DEPTH_BONUS

    return min(score, 1.0)


# =============================================================================
# TOKEN CONTEXT SIGNAL (Section 5.1 — 10% weight)
# =============================================================================
# Simple token-count buckets. Intentionally low-weighted.

def _score_token_signal(features: RequestFeatures) -> float:
    """Score based on total token estimate."""
    tokens = features.total_token_estimate
    if tokens > 2000:
        return 0.9
    if tokens > 500:
        return 0.7
    if tokens > 200:
        return 0.5
    if tokens > 50:
        return 0.3
    return 0.1


# =============================================================================
# TIER MAPPING + HYSTERESIS (Sections 5.1, 5.3)
# =============================================================================

def _map_score_to_tier(
    score: float,
    quality: str = "auto",
    previous_tier: Optional[str] = None,
) -> tuple:
    """
    Map final score to a minimum tier floor.

    Returns:
        (tier, near_boundary, hysteresis_applied)
    """
    boundaries = QUALITY_BOUNDARIES.get(quality, DEFAULT_BOUNDARIES)
    nano_ceil = boundaries["nano_ceiling"]
    mini_ceil = boundaries["mini_ceiling"]
    base_ceil = boundaries["base_ceiling"]

    # Determine raw tier
    if score < nano_ceil:
        raw_tier = "nano"
    elif score < mini_ceil:
        raw_tier = "mini"
    elif score < base_ceil:
        raw_tier = "base"
    else:
        raw_tier = "premium"

    # Check near-boundary (within +/- 0.05 of any threshold)
    near_boundary = any(
        abs(score - b) <= NEAR_BOUNDARY_BAND
        for b in (nano_ceil, mini_ceil, base_ceil)
    )

    # Hysteresis: if score is in dead zone (+/- 0.03) around a boundary,
    # keep previous tier if available, otherwise round up
    hysteresis_applied = False
    in_dead_zone = any(
        abs(score - b) <= HYSTERESIS_BAND
        for b in (nano_ceil, mini_ceil, base_ceil)
    )

    if in_dead_zone:
        if previous_tier:
            raw_tier = previous_tier
            hysteresis_applied = True
        else:
            # No history — round up (conservative)
            if abs(score - nano_ceil) <= HYSTERESIS_BAND:
                raw_tier = "mini"
            elif abs(score - mini_ceil) <= HYSTERESIS_BAND:
                raw_tier = "base"
            elif abs(score - base_ceil) <= HYSTERESIS_BAND:
                raw_tier = "premium"
            hysteresis_applied = True

    return raw_tier, near_boundary, hysteresis_applied


# =============================================================================
# TASK → ROUTING PROFILE MAPPING
# =============================================================================

_TASK_TO_REASONING_DEPTH = {
    "chat": "low",
    "summarize": "low",
    "creative": "medium",
    "code": "medium",
    "extract": "medium",
    "math": "high",
    "planning": "high",
}

_TASK_TO_RELIABILITY = {
    "chat": "low",
    "summarize": "low",
    "creative": "low",
    "code": "medium",
    "extract": "medium",
    "math": "high",
    "planning": "high",
}

_ELEVATED_RISK_THRESHOLD = 0.6
_HIGH_RISK_ESCALATION_THRESHOLD = 0.85
_LOW_TIERS = ("nano", "mini")


def _compute_escalation_signal(
    near_boundary: bool,
    risk_level: Literal["low", "elevated"],
    risk_score: float,
    selected_tier: str,
    final_score: float,
) -> tuple[bool, Optional[str]]:
    """Determine whether a routing decision should be escalated for safety."""
    if near_boundary and risk_level == "elevated":
        reason = f"near_boundary_elevated_risk(score={final_score:.3f},tier={selected_tier})"
        return True, reason

    if risk_score >= _HIGH_RISK_ESCALATION_THRESHOLD and selected_tier in _LOW_TIERS:
        reason = f"high_risk_domain_low_tier(risk={risk_score:.2f},tier={selected_tier})"
        return True, reason

    return False, None


# =============================================================================
# PUBLIC API
# =============================================================================

def score_request(
    features: RequestFeatures,
    quality: str = "auto",
    previous_tier: Optional[str] = None,
    override_reason: Optional[str] = None,
    image_count: int = 0,
) -> tuple:
    """
    Score a request and produce a RoutingProfile + HeuristicScores.

    Args:
        features: Extracted request features.
        quality: User's quality preference (auto/high/medium/low).
        previous_tier: Tier from previous request in same session (for hysteresis).
        override_reason: If set, skip scoring and use this reason (e.g. "optimize_disabled").

    Returns:
        (RoutingProfile, HeuristicScores)
    """
    # --- Hard override: skip scoring entirely ---
    if override_reason:
        profile = RoutingProfile(
            task_type=features.detected_task,
            min_tier="base",
            confidence=1.0,
            explanation=f"override:{override_reason}",
        )
        scores = HeuristicScores(
            selected_tier="base",
            override_reason=override_reason,
        )
        return profile, scores

    # --- Lazy signal evaluation ---
    # Skip expensive scoring for trivially simple queries.
    # If the query is very short, has no structural complexity, no risk
    # keywords, and is a simple task — fast-path to nano/mini.
    try:
        from .config import settings as _settings
        _lazy_eval = _settings.ROUTING_LAZY_EVAL
    except Exception:
        _lazy_eval = False

    if _lazy_eval:
        text_len = len(features.last_user_message)
        is_trivial = (
            text_len < 50
            and features.total_token_estimate < 100
            and not features.risk_keywords_found
            and not features.has_code_fences
            and not features.has_json_blocks
            and features.detected_task in ("chat",)
            and features.question_type in ("factual", "unknown")
        )
        if is_trivial:
            tier, near_boundary, hysteresis_applied = _map_score_to_tier(
                0.10, quality, previous_tier
            )
            profile = RoutingProfile(
                task_type=features.detected_task,
                min_tier=tier,
                confidence=0.95,
                explanation="lazy_eval:trivial_query",
            )
            scores = HeuristicScores(
                task_type_score=0.1,
                domain_risk_score=0.0,
                structure_score=0.1,
                token_signal_score=0.1,
                final_score=0.10,
                selected_tier=tier,
                near_boundary=near_boundary,
                hysteresis_applied=hysteresis_applied,
            )
            logger.debug(f"Lazy eval fast-path: trivial query → {tier}")
            return profile, scores

        # Also fast-path obviously complex queries (long + code + planning/math)
        is_obviously_complex = (
            text_len > 500
            and features.total_token_estimate > 500
            and features.detected_task in ("math", "planning", "code")
        )
        if is_obviously_complex:
            tier, near_boundary, hysteresis_applied = _map_score_to_tier(
                0.80, quality, previous_tier
            )
            profile = RoutingProfile(
                task_type=features.detected_task,
                reasoning_depth="high",
                reliability_need="high" if features.detected_task == "math" else "medium",
                min_tier=tier,
                confidence=0.90,
                explanation="lazy_eval:obviously_complex",
            )
            scores = HeuristicScores(
                task_type_score=0.8,
                domain_risk_score=0.0,
                structure_score=0.8,
                token_signal_score=0.7,
                final_score=0.80,
                selected_tier=tier,
                near_boundary=near_boundary,
                hysteresis_applied=hysteresis_applied,
            )
            logger.debug(f"Lazy eval fast-path: obviously complex → {tier}")
            return profile, scores

    # --- Score each dimension ---
    # Domain risk scored first so task scorer can use it for the generative bonus
    risk_score = _score_domain_risk(features)
    task_score = _score_task_type(features, domain_risk_score=risk_score)
    structure_score = _score_structure(features)
    token_score = _score_token_signal(features)

    # --- Contrastive complexity override — dual-signal ---
    # Uses combined text + visual difficulty signal when images are present.
    # Falls back to text-only signal, then to regex-based structure scoring.
    try:
        from .config import settings
        if settings.ROUTING_CONTRASTIVE_COMPLEXITY:
            from .complexity_detector import (
                compute_combined_difficulty,
                difficulty_to_score,
            )
            difficulty_signal = compute_combined_difficulty(
                features.last_user_message,
                image_count=image_count,
            )
            contrastive_score = difficulty_to_score(difficulty_signal)
            if contrastive_score is not None:
                # Blend: 70% contrastive (semantic), 30% structural (regex)
                structure_score = contrastive_score * 0.7 + structure_score * 0.3
    except Exception:
        pass  # Contrastive detection is additive, never blocks routing

    # --- Jailbreak escalation ---
    # If prompt injection is detected, route to a more capable model
    # (premium/base models have better safety alignment).
    jailbreak_boost = 0.0
    if features.jailbreak_detected and features.jailbreak_confidence >= 0.75:
        jailbreak_boost = 0.25  # Push firmly toward premium tier
        logger.warning(
            f"Jailbreak escalation: category={features.jailbreak_category}, "
            f"confidence={features.jailbreak_confidence:.2f}, boost={jailbreak_boost}"
        )

    # --- Language boost for non-English ---
    # Smaller models are significantly worse at non-English. Boost the
    # score to push toward more capable models for non-English queries.
    language_boost = 0.0
    if (features.detected_language != "en"
            and features.language_confidence >= 0.5):
        language_boost = 0.10  # ~1 tier bump worth of score

    # --- Final weighted score ---
    final_score = (
        WEIGHT_TASK_TYPE * task_score
        + WEIGHT_DOMAIN_RISK * risk_score
        + WEIGHT_STRUCTURE * structure_score
        + WEIGHT_TOKEN_SIGNAL * token_score
        + language_boost
        + jailbreak_boost
    )

    # --- Task complexity floor ---
    # High-complexity tasks (math, planning, reasoning, code) must not be
    # dragged below their natural tier by zero domain_risk (30% weight is
    # wasted on most non-risk queries, capping the max possible score at 0.70).
    # Gated on minimum query substance to avoid escalating trivial queries.
    task_floor = _TASK_COMPLEXITY_FLOORS.get(features.detected_task, 0.0)
    has_substance = (
        len(features.last_user_message) >= _FLOOR_MIN_TEXT_LEN
        or structure_score >= _FLOOR_MIN_STRUCTURE
    )
    if final_score < task_floor and has_substance:
        logger.info(
            f"Task complexity floor: {features.detected_task} "
            f"score {final_score:.3f} → {task_floor:.3f}"
        )
        final_score = task_floor

    final_score = round(min(final_score, 1.0), 4)

    # --- Map to tier ---
    selected_tier, near_boundary, hysteresis_applied = _map_score_to_tier(
        final_score, quality, previous_tier
    )

    # --- Build explanation ---
    signals = []
    if task_score >= 0.6:
        signals.append(f"task={features.detected_task}({task_score:.2f})")
    if risk_score >= 0.6:
        signals.append(f"risk={features.detected_domain}({risk_score:.2f})")
    if structure_score >= 0.6:
        signals.append(f"structure({structure_score:.2f})")
    if not signals:
        signals.append(f"score={final_score:.2f}")
    explanation = ", ".join(signals)

    # --- Determine risk level ---
    risk_level: Literal["low", "elevated"] = (
        "elevated" if risk_score >= _ELEVATED_RISK_THRESHOLD else "low"
    )

    # --- Determine structure need ---
    if features.has_json_blocks:
        structure_need: Literal["freeform", "strict_json", "schema"] = "strict_json"
    else:
        structure_need = "freeform"

    # --- Confidence: lower when near boundary or risk is ambiguous ---
    confidence = 1.0
    if near_boundary:
        confidence -= 0.2
    if 0.3 < risk_score < 0.7:
        confidence -= 0.15
    confidence = round(max(confidence, 0.0), 2)

    # --- Assemble profile ---
    profile = RoutingProfile(
        task_type=features.detected_task,
        reasoning_depth=_TASK_TO_REASONING_DEPTH.get(features.detected_task, "low"),
        reliability_need=_TASK_TO_RELIABILITY.get(features.detected_task, "low"),
        structure_need=structure_need,
        risk_level=risk_level,
        min_tier=selected_tier,
        confidence=confidence,
        explanation=explanation,
    )

    would_escalate, escalation_reason = _compute_escalation_signal(
        near_boundary=near_boundary,
        risk_level=risk_level,
        risk_score=risk_score,
        selected_tier=selected_tier,
        final_score=final_score,
    )

    scores = HeuristicScores(
        task_type_score=round(task_score, 4),
        domain_risk_score=round(risk_score, 4),
        structure_score=round(structure_score, 4),
        token_signal_score=round(token_score, 4),
        final_score=final_score,
        selected_tier=selected_tier,
        near_boundary=near_boundary,
        hysteresis_applied=hysteresis_applied,
        override_reason=override_reason,
        would_escalate=would_escalate,
        escalation_reason=escalation_reason,
    )

    if near_boundary:
        logger.info(
            f"Near-boundary: score={final_score}, tier={selected_tier}, "
            f"hysteresis={hysteresis_applied}, quality={quality}"
        )

    return profile, scores
