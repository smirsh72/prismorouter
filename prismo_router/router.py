"""Public standalone wrapper around the production-derived routing strategy."""

from __future__ import annotations

from collections.abc import Sequence

from .feature_extractor import extract_features
from .heuristic_scorer import score_request
from .models import ModelCandidate, RouteOptions, RoutingDecision
from .repository import InMemoryModelRepository
from .routing_strategy import CostOptimizedStrategy


def route_request(
    prompt: str,
    candidates: Sequence[ModelCandidate],
    requested_model: str | None = None,
    options: RouteOptions | None = None,
) -> RoutingDecision:
    """Select a model using the same strategy shape as Prismo's hosted router."""
    if not candidates:
        raise ValueError("candidates cannot be empty")

    options = options or RouteOptions()
    requested_model = requested_model or candidates[0].name

    features = extract_features(input_text=prompt, model=requested_model)
    profile, scores = score_request(features, quality=options.quality)

    repo = InMemoryModelRepository(candidates)
    strategy = CostOptimizedStrategy()
    selected = strategy.select_model(
        input_tokens=features.total_token_estimate,
        requested_model=requested_model,
        db_session=repo,
        optimize=options.optimize,
        quality=options.quality,
        scope=options.scope,
        require_tools=options.require_tools,
        require_vision=options.require_vision,
        require_json=options.require_json,
        min_context_window=options.min_context_window,
        min_output_tokens=0,
        min_tier=profile.min_tier,
        confidence=profile.confidence,
        risk_level=profile.risk_level,
        task_type=profile.task_type,
        query_text=prompt,
    )

    return RoutingDecision(
        selected_model=selected,
        requested_model=requested_model,
        optimized=selected != requested_model,
        confidence=profile.confidence,
        reason=profile.explanation,
        profile=profile,
        features=features,
        scores={
            "task_type": scores.task_type_score,
            "domain_risk": scores.domain_risk_score,
            "structure": scores.structure_score,
            "token_signal": scores.token_signal_score,
            "final": scores.final_score,
        },
    )
