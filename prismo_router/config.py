"""Standalone routing settings.

These defaults mirror Prismo's hosted router flags, without importing the
hosted SaaS settings module.
"""

from dataclasses import dataclass


@dataclass
class RouterSettings:
    ROUTING_QUALITY_COST_SCORING: bool = True
    ROUTING_ELO_ENABLED: bool = True
    ROUTING_CONTRASTIVE_COMPLEXITY: bool = True
    ROUTING_LAZY_EVAL: bool = True
    ROUTING_LANGUAGE_DETECTION: bool = True
    ROUTING_JAILBREAK_DETECTION: bool = True
    ROUTING_LATENCY_AWARE: bool = True
    ROUTING_COST_WEIGHT: float = 0.4
    ROUTING_COST_SCALING_FACTOR: float = 1.5
    ROUTING_HYBRID_SCORING: bool = True
    ROUTING_HYBRID_TIMEOUT_MS: int = 900


settings = RouterSettings()
