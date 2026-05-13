"""In-memory model repository used by the standalone router."""

from __future__ import annotations

from collections.abc import Sequence

from .models import ModelCandidate


class InMemoryModelRepository:
    """Repository adapter matching the hosted router's candidate lookup API."""

    def __init__(self, candidates: Sequence[ModelCandidate]):
        self.candidates = list(candidates)

    def get_model_metadata(self, model_name: str) -> dict | None:
        candidate = self._find_by_name(model_name)
        if not candidate:
            return None
        return {
            "provider": candidate.provider,
            "family": candidate.family,
            "tier": candidate.tier,
        }

    def find_candidates(
        self,
        scope: str,
        family: str | None = None,
        provider: str | None = None,
        allowed_tiers: list[str] | None = None,
        allowed_model_ids: list[str] | None = None,
    ) -> list[ModelCandidate]:
        allowed_tiers = allowed_tiers or []
        allowed_ids = set(allowed_model_ids or [])
        results = []
        for candidate in self.candidates:
            if allowed_tiers and candidate.tier not in allowed_tiers:
                continue
            if allowed_ids and str(candidate.id or candidate.name) not in allowed_ids:
                continue
            if scope == "family" and family and candidate.family != family:
                continue
            if scope in {"family", "provider"} and provider and candidate.provider != provider:
                continue
            results.append(candidate)
        return sorted(results, key=lambda c: (c.input_price, c.output_price, c.name))

    def find_best_model(
        self,
        family: str,
        tier: str,
        provider: str,
        allowed_model_ids: list[str] | None = None,
    ) -> str | None:
        candidates = self.find_candidates(
            scope="family",
            family=family,
            provider=provider,
            allowed_tiers=[tier],
            allowed_model_ids=allowed_model_ids,
        )
        return candidates[0].name if candidates else None

    def _find_by_name(self, model_name: str) -> ModelCandidate | None:
        return next((candidate for candidate in self.candidates if candidate.name == model_name), None)
