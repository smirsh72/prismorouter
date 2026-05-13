"""
Domain Layer — Routing Profile.

The routing profile is the output of the router intelligence layer.
It describes WHAT a request needs (task type, risk level, structure needs),
not WHICH model to use. The deterministic picker consumes this to select a model.

This is a pure value object with no side effects or DB dependencies.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class RoutingProfile(BaseModel):
    """
    Router output — describes request requirements for the picker.

    Fields map directly to the blueprint spec (Section 4.2).
    """
    task_type: Literal[
        "chat", "summarize", "extract", "code",
        "math", "planning", "creative"
    ] = "chat"

    reasoning_depth: Literal["low", "medium", "high"] = "low"

    reliability_need: Literal["low", "medium", "high"] = "low"

    structure_need: Literal["freeform", "strict_json", "schema"] = "freeform"

    risk_level: Literal["low", "elevated"] = "low"

    min_tier: Literal["nano", "mini", "base", "premium"] = "nano"

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    explanation: str = ""


class HeuristicScores(BaseModel):
    """
    Breakdown of the four scoring dimensions (Section 5.1).

    Stored alongside the routing profile for telemetry and debugging.
    """
    task_type_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    structure_score: float = Field(default=0.0, ge=0.0, le=1.0)
    token_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Tier decision metadata
    selected_tier: Literal["nano", "mini", "base", "premium"] = "nano"
    near_boundary: bool = False
    hysteresis_applied: bool = False
    override_reason: Optional[str] = None

    # Shadow-mode escalation (logged but not live yet)
    would_escalate: bool = False
    escalation_reason: Optional[str] = None
