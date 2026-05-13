"""Core value objects for PrismoRouter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["nano", "mini", "base", "premium"]
Quality = Literal["low", "medium", "high", "auto"]
Scope = Literal["family", "provider", "global"]

TIER_ORDER: list[Tier] = ["nano", "mini", "base", "premium"]
TIER_RANK = {tier: idx for idx, tier in enumerate(TIER_ORDER)}


@dataclass(frozen=True)
class ModelCandidate:
    """A model that PrismoRouter may select."""

    name: str
    provider: str
    family: str
    tier: Tier
    input_price: float
    output_price: float = 0.0
    context_window: int = 0
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json: bool = False
    error_rate_7d: float = 0.0
    timeout_rate_7d: float = 0.0
    schema_pass_rate_7d: float = 1.0
    tool_success_rate_7d: float = 1.0
    avg_ttft_ms: float | None = None
    avg_tpot_ms: float | None = None
    p95_latency_ms: int | None = None
    max_output: int = 0
    id: int | None = None

    @property
    def model_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class RouteOptions:
    """Controls model selection behavior."""

    quality: Quality = "auto"
    scope: Scope = "provider"
    optimize: bool = True
    require_tools: bool = False
    require_vision: bool = False
    require_json: bool = False
    min_context_window: int = 0
    expected_output_tokens: int = 200
    use_latency: bool = True
    cost_weight: float = 0.4
    cost_scaling_factor: float = 1.5


@dataclass(frozen=True)
class RoutingDecision:
    """Final model selection result."""

    selected_model: str
    requested_model: str | None
    optimized: bool
    confidence: float
    reason: str
    profile: Any
    features: Any
    scores: dict[str, float] = field(default_factory=dict)
