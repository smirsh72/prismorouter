"""Heuristic request scoring."""

from __future__ import annotations

from .models import RequestFeatures, RoutingProfile, Tier

TASK_SCORES = {
    "chat": 0.35,
    "summarize": 0.30,
    "creative": 0.35,
    "code": 0.60,
    "reasoning": 0.70,
    "math": 0.80,
    "planning": 0.80,
}


def _tier_for_score(score: float) -> Tier:
    if score <= 0.25:
        return "nano"
    if score <= 0.50:
        return "mini"
    if score <= 0.75:
        return "base"
    return "premium"


def score_request(features: RequestFeatures) -> RoutingProfile:
    """Convert request features into a minimum model tier."""
    task_score = TASK_SCORES.get(features.detected_task, 0.35)
    domain_score = 0.0 if features.detected_domain == "general" else 0.70
    structure_score = min(
        1.0,
        (0.20 if features.has_code else 0.0)
        + (0.15 if features.has_json else 0.0)
        + (0.10 if features.has_list else 0.0)
        + min(0.25, features.constraint_count * 0.05),
    )
    token_score = min(1.0, features.token_estimate / 2500.0)
    safety_score = features.jailbreak_confidence if features.jailbreak_detected else 0.0

    final = max(
        safety_score,
        task_score * 0.35 + domain_score * 0.30 + structure_score * 0.25 + token_score * 0.10,
    )

    tier = _tier_for_score(final)
    if features.detected_task in {"code", "math", "planning"} and tier in {"nano", "mini"}:
        tier = "base"
    if features.jailbreak_detected:
        tier = "base"
    elif domain_score >= 0.70:
        tier = "base" if tier in {"nano", "mini"} else tier

    risk_level = "elevated" if domain_score >= 0.70 or features.jailbreak_detected else "low"
    confidence = max(0.35, 1.0 - abs(final - 0.50) * 0.4)

    return RoutingProfile(
        task_type=features.detected_task,
        risk_level=risk_level,
        min_tier=tier,
        confidence=round(confidence, 3),
        explanation=f"{features.detected_task} task with {risk_level} risk",
        scores={
            "task": round(task_score, 3),
            "domain": round(domain_score, 3),
            "structure": round(structure_score, 3),
            "tokens": round(token_score, 3),
            "safety": round(safety_score, 3),
            "final": round(final, 3),
        },
    )
