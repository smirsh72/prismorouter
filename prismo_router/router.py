"""Standalone cost-aware model routing."""

from __future__ import annotations

from collections.abc import Sequence

from .elo import EloStore
from .features import extract_features
from .heuristic import score_request
from .models import ModelCandidate, RouteOptions, RoutingDecision, TIER_RANK

QUALITY_TIERS = {
    "low": {"nano", "mini", "base"},
    "medium": {"mini", "base"},
    "high": {"mini", "base", "premium"},
    "auto": {"nano", "mini", "base", "premium"},
}


def _candidate_score(
    candidate: ModelCandidate,
    max_input_price: float,
    options: RouteOptions,
    elo_bonus: float = 0.0,
) -> float:
    reliability = max(0.01, 1.0 - candidate.error_rate_7d - candidate.timeout_rate_7d)
    quality = reliability * 0.50 + candidate.schema_pass_rate_7d * 0.25 + candidate.tool_success_rate_7d * 0.25
    normalized_cost = candidate.input_price / max_input_price if max_input_price > 0 else 0.0
    cost_bonus = (1.0 - normalized_cost) * options.cost_weight * options.cost_scaling_factor
    score = quality * (1.0 + cost_bonus) + elo_bonus

    if options.use_latency and candidate.avg_ttft_ms and candidate.avg_tpot_ms:
        estimated_ms = candidate.avg_ttft_ms + candidate.avg_tpot_ms * options.expected_output_tokens
        score -= min(0.05, max(0.0, (candidate.avg_ttft_ms - 300) / 10000.0))
        score -= min(0.05, max(0.0, (estimated_ms - 1000) / 40000.0))
    return score


def _filter_candidates(
    candidates: Sequence[ModelCandidate],
    requested: ModelCandidate | None,
    profile_min_tier: str,
    options: RouteOptions,
) -> list[ModelCandidate]:
    allowed_tiers = QUALITY_TIERS.get(options.quality, QUALITY_TIERS["auto"])
    min_rank = TIER_RANK[profile_min_tier]
    filtered = []
    for candidate in candidates:
        if candidate.tier not in allowed_tiers:
            continue
        if TIER_RANK[candidate.tier] < min_rank:
            continue
        if options.require_tools and not candidate.supports_tools:
            continue
        if options.require_vision and not candidate.supports_vision:
            continue
        if options.require_json and not candidate.supports_json:
            continue
        if options.min_context_window and candidate.context_window and candidate.context_window < options.min_context_window:
            continue
        if requested and options.scope == "provider" and candidate.provider != requested.provider:
            continue
        if requested and options.scope == "family" and candidate.family != requested.family:
            continue
        filtered.append(candidate)
    return filtered


def route_request(
    prompt: str,
    candidates: Sequence[ModelCandidate],
    requested_model: str | None = None,
    options: RouteOptions | None = None,
    elo_store: EloStore | None = None,
) -> RoutingDecision:
    """Select the best model candidate for a prompt."""
    if not candidates:
        raise ValueError("candidates cannot be empty")

    options = options or RouteOptions()
    requested = next((candidate for candidate in candidates if candidate.name == requested_model), None)
    features = extract_features(prompt)
    profile = score_request(features)

    if not options.optimize and requested:
        return RoutingDecision(
            selected_model=requested.name,
            requested_model=requested_model,
            optimized=False,
            confidence=1.0,
            reason="optimization disabled",
            profile=profile,
            features=features,
        )

    filtered = _filter_candidates(candidates, requested, profile.min_tier, options)
    if not filtered:
        fallback = requested or candidates[0]
        return RoutingDecision(
            selected_model=fallback.name,
            requested_model=requested_model,
            optimized=False,
            confidence=profile.confidence,
            reason="no candidate satisfied routing constraints",
            profile=profile,
            features=features,
        )

    max_price = max(candidate.input_price for candidate in filtered)
    scored = {}
    for candidate in filtered:
        elo_bonus = elo_store.normalized_bonus(candidate.name, profile.task_type) if elo_store else 0.0
        scored[candidate.name] = _candidate_score(candidate, max_price, options, elo_bonus)

    selected = max(filtered, key=lambda candidate: scored[candidate.name])
    return RoutingDecision(
        selected_model=selected.name,
        requested_model=requested_model,
        optimized=selected.name != requested_model,
        confidence=profile.confidence,
        reason=f"{profile.explanation}; selected {selected.tier} tier by cost-adjusted score",
        profile=profile,
        features=features,
        scores={name: round(score, 4) for name, score in scored.items()},
    )

