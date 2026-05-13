"""
Infrastructure/Strategy Layer - Provider-Agnostic Model Routing.

Portions of this routing and cost-aware scoring approach are adapted from vLLM
Semantic Router and modified for PrismoRouter's standalone Python API.

ARCHITECTURE: Clean DB-Driven Routing
=====================================
1. REQUEST: User sends request with model preference (e.g., "gpt-4o", "claude-3-5-sonnet")
2. METADATA LOOKUP: Get model metadata (provider, family, tier)
3. ROUTING SCOPE: Based on user settings, scope routing to:
   - "family": Same model family (gpt-4o → gpt-4o-mini)
   - "provider": Same provider (OpenAI → any OpenAI model)
   - "global": Any enabled provider (future)
4. QUALITY TIER: Map user preference to allowed tiers:
   - "cheap": nano, mini
   - "balanced": mini, base
   - "best": base, premium
   - "auto": all tiers
5. SELECTION: Pick cheapest model satisfying constraints.

Model metadata comes from the candidate repository passed to the router.
"""
from abc import ABC, abstractmethod
import re
import time
from typing import Optional, List
from .model_normalizer import normalize_model
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MIN_QUALITY MAPPING: User-facing quality levels → routing behavior
# =============================================================================
# high: Stay within EXACT same model family only
#       - gpt-4o → gpt-4o-mini (same family, different tier)
#       - claude-3.5-sonnet → claude-3.5-sonnet only (no sonnet→haiku, they're different families)
#       - If no cheaper variant exists in family, stay on requested model
# medium: Stay within same family but allow all tiers (more flexible tier selection)
#       - gpt-4.1 → gpt-4.1-mini allowed (same family, cheaper tier)
#       - gpt-4.1 → gpt-5.2 NOT allowed (different family)
# low: Allow fallback to legacy/cheapest models within same provider
# auto: Smart routing based on request complexity (default, same as high)

MIN_QUALITY_CONFIG = {
    "high": {
        "scope": "family",           # Only route within EXACT same family
        "allowed_tiers": ["mini", "base", "premium"],  # No nano for high quality
        "strict_family": True,       # If no match in family, use requested model
    },
    "medium": {
        "scope": "provider",         # Allow different families within same provider
        "allowed_tiers": ["mini", "base"],  # Modern models only (no nano, no premium)
        "strict_family": False,
    },
    "low": {
        "scope": "family",           # Stay within same family (don't jump to newer generations)
        "allowed_tiers": ["nano", "mini", "base"],  # Include cheapest tiers
        "strict_family": True,       # If no match in family, use requested model
    },
    "auto": {
        "scope": "provider",         # Route across all families within same provider
        "allowed_tiers": ["nano", "mini", "base", "premium"],  # All tiers
        "strict_family": False,      # Allow cross-family routing for best cost/quality
    },
}

# Tier ordering for fallback selection
TIER_ORDER = ["nano", "mini", "base", "premium"]


def get_allowed_tiers(quality: str) -> List[str]:
    """
    Return the allowed tiers for a given quality preference.

    Supports both current names (high/medium/low/auto) and legacy aliases
    (cheap/balanced/best) for backwards compatibility.
    """
    _QUALITY_TO_TIERS = {
        # Current names (from MIN_QUALITY_CONFIG)
        "high": ["mini", "base", "premium"],
        "medium": ["mini", "base"],
        "low": ["nano", "mini", "base"],
        "auto": ["nano", "mini", "base", "premium"],
        # Legacy aliases
        "cheap": ["nano", "mini"],
        "balanced": ["mini", "base"],
        "best": ["base", "premium"],
    }
    return _QUALITY_TO_TIERS.get(quality, _QUALITY_TO_TIERS["auto"])


def get_tier_for_tokens(input_tokens: int) -> str:
    """
    Suggest a tier based on input token count.
    This is a heuristic for simple requests.
    
    - < 100 tokens: nano (very simple)
    - < 400 tokens: mini (simple-moderate)
    - >= 400 tokens: base (full capability needed)
    """
    if input_tokens < 100:
        return "nano"
    elif input_tokens < 400:
        return "mini"
    else:
        return "base"


def _select_target_tier(suggested_tier: str, allowed_tiers: List[str]) -> str:
    """Select target tier: use suggested if allowed, otherwise cheapest allowed."""
    if suggested_tier in allowed_tiers:
        return suggested_tier
    return next((t for t in TIER_ORDER if t in allowed_tiers), "base")


def _apply_confidence_bumping(tier: str, confidence: float, risk_level: str) -> str:
    """
    Confidence-based tier bumping (Blueprint Section 4.3).

    - If confidence < 0.6 and risk_level != "low" → bump up one level
    - If confidence < 0.4 → bump up one level regardless
    - Never bump beyond premium
    """
    if confidence >= 0.6:
        return tier

    should_bump = (
        (confidence < 0.4) or
        (confidence < 0.6 and risk_level != "low")
    )

    if not should_bump:
        return tier

    idx = TIER_ORDER.index(tier) if tier in TIER_ORDER else 0
    bumped_idx = min(idx + 1, len(TIER_ORDER) - 1)
    bumped = TIER_ORDER[bumped_idx]
    if bumped != tier:
        logger.info(f"Confidence bump: {tier} → {bumped} (confidence={confidence}, risk={risk_level})")
    return bumped


def _filter_by_context_window(candidates: list, min_context_window: int) -> list:
    """Filter candidates by context window, allowing unknown (0) values."""
    if min_context_window <= 0:
        return candidates
    return [c for c in candidates if c.context_window == 0 or c.context_window >= min_context_window]


# =============================================================================
# ROUTING STRATEGIES
# =============================================================================

class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""
    
    @abstractmethod
    def select_model(
        self,
        input_tokens: int,
        requested_model: str,
        db_session,
        optimize: bool = True,
        quality: str = "auto",
        scope: str = None,
        require_tools: bool = False,
        require_vision: bool = False,
        require_json: bool = False,
        min_context_window: int = 0,
        min_output_tokens: int = 0,
        min_tier: str = None,
        max_tier_override: str = None,
        confidence: float = 1.0,
        risk_level: str = "low",
        allowed_model_ids: List[str] = None,
        task_type: str = "chat",
        query_text: str = "",
    ) -> str:
        """
        Select the best model for the request.

        Args:
            input_tokens: Estimated input token count
            requested_model: User's requested model
            db_session: Database session
            optimize: Whether to optimize (if False, use exact model)
            quality: Quality preference (high, medium, low, auto)
            scope: Override routing scope (family, provider, global)
            require_tools: Request requires function calling support
            require_vision: Request requires image input support
            require_json: Request requires JSON mode output
            min_context_window: Minimum context window required
            min_output_tokens: Minimum output tokens required (new)
            min_tier: Minimum tier from heuristic scorer
            max_tier_override: FinOps-imposed ceiling on tier (cannot exceed this)
            confidence: Router confidence (0.0-1.0)
            risk_level: "low" or "elevated"
            allowed_model_ids: Optional list of allowed model IDs (user access)
            task_type: Detected task type for Elo rating lookup

        Returns:
            Selected model name
        """
        pass


def _compute_candidate_score(
    candidate,
    max_input_price: float,
    cost_weight: float = 0.4,
    cost_scaling_factor: float = 1.5,
    elo_bonus: float = 0.0,
    use_latency: bool = False,
) -> float:
    """
    Score a candidate model using quality-adjusted cost optimization.

    Portions of this scoring approach are adapted from vLLM Semantic Router.

    Formula:
        costBonus = (1 - normalizedCost) * costWeight * costScalingFactor
        score = quality * (1 + costBonus) + elo_bonus - latency_penalty

    Quality is derived from DB reliability metrics (error_rate, timeout_rate,
    schema_pass_rate, tool_success_rate). Cost is normalized against the most
    expensive candidate in the set. The costScalingFactor amplifies the cost
    advantage so cheaper models get a stronger multiplicative bonus.

    elo_bonus is a feedback-driven adjustment from the Elo rating system.
    latency_penalty uses real TPOT/TTFT data when ROUTING_LATENCY_AWARE is on.
    """
    # Quality: combine reliability metrics (all 0-1 scale, higher = better)
    error_rate = getattr(candidate, "error_rate_7d", 0.0) or 0.0
    timeout_rate = getattr(candidate, "timeout_rate_7d", 0.0) or 0.0
    schema_pass = getattr(candidate, "schema_pass_rate_7d", 1.0) or 1.0
    tool_success = getattr(candidate, "tool_success_rate_7d", 1.0) or 1.0

    reliability = 1.0 - error_rate - timeout_rate
    quality = max(0.01, (reliability * 0.5 + schema_pass * 0.25 + tool_success * 0.25))

    # Cost: normalize to 0-1 range against max price in candidate set
    input_price = getattr(candidate, "input_price", 0.0) or 0.0
    if max_input_price > 0:
        normalized_cost = input_price / max_input_price
    else:
        normalized_cost = 0.0

    # Cheaper models get a multiplicative bonus amplified by scaling factor.
    # Adapted from vLLM Semantic Router's cost-aware scoring approach.
    cost_bonus = (1.0 - normalized_cost) * cost_weight * cost_scaling_factor
    score = quality * (1.0 + cost_bonus) + elo_bonus

    # Latency penalty: penalize slow models using real TPOT/TTFT data.
    if use_latency:
        # Prefer real TPOT/TTFT data over static p95
        avg_ttft = getattr(candidate, "avg_ttft_ms", None)
        avg_tpot = getattr(candidate, "avg_tpot_ms", None)

        if avg_ttft and avg_tpot and avg_ttft > 0 and avg_tpot > 0:
            # Estimated total latency for a typical 200-token response
            estimated_ms = avg_ttft + (avg_tpot * 200)
            # TTFT penalty: users notice first-token delay most
            ttft_penalty = min(0.05, max(0.0, (avg_ttft - 300) / 10000.0))
            # Total latency penalty
            total_penalty = min(0.05, max(0.0, (estimated_ms - 1000) / 40000.0))
            score -= (ttft_penalty + total_penalty)
        else:
            # Fallback to static p95_latency_ms
            p95_ms = getattr(candidate, "p95_latency_ms", None)
            if p95_ms and p95_ms > 0:
                latency_penalty = min(0.1, max(0.0, (p95_ms - 500) / 45000.0))
                score -= latency_penalty

    return score


class CostOptimizedStrategy(RoutingStrategy):
    """
    Provider-Agnostic Cost Optimization Strategy.

    Uses DB columns for routing - no string parsing.
    Supports routing within family, provider, or globally.
    Implements min_quality (high/medium/low) for user control.
    Implements capability gating (tools, vision, JSON, context window).
    Ranks candidates by quality-adjusted cost score (not just cheapest).
    When ROUTING_HYBRID_SCORING is enabled, uses the 4-scorer ensemble.
    """

    def __init__(self):
        self._last_timing_breakdown_ms: dict[str, float] = {}

    def _reset_timing_breakdown(self) -> None:
        self._last_timing_breakdown_ms = {}

    def _bump_timing(self, name: str, duration_ms: float) -> None:
        self._last_timing_breakdown_ms[name] = (
            self._last_timing_breakdown_ms.get(name, 0.0) + duration_ms
        )

    def get_last_timing_breakdown_ms(self) -> dict[str, float]:
        return dict(self._last_timing_breakdown_ms)

    @staticmethod
    def _is_version_pinned_model(model_name: str) -> bool:
        """
        True when model id is pinned to a concrete release suffix like YYYYMMDD.
        Example: claude-haiku-4-5-20251001
        """
        if not model_name:
            return False
        return bool(re.search(r"-\d{8}$", model_name))

    def _prefer_pinned_anthropic_models(self, candidates: List) -> List:
        """
        Anthropic catalogs often include both aliases and version-pinned IDs.
        Prefer pinned IDs when both exist in the same family+tier bucket to avoid
        alias-related `model not found` failures (e.g., dotted aliases).
        """
        if not candidates:
            return candidates

        pinned_buckets = {
            (getattr(c, "family", ""), getattr(c, "tier", ""))
            for c in candidates
            if getattr(c, "provider", "") == "anthropic"
            and self._is_version_pinned_model(getattr(c, "model_name", ""))
        }
        if not pinned_buckets:
            return candidates

        filtered = []
        for candidate in candidates:
            if getattr(candidate, "provider", "") != "anthropic":
                filtered.append(candidate)
                continue

            bucket = (getattr(candidate, "family", ""), getattr(candidate, "tier", ""))
            if bucket in pinned_buckets and not self._is_version_pinned_model(getattr(candidate, "model_name", "")):
                continue
            filtered.append(candidate)

        return filtered or candidates

    @staticmethod
    def _is_text_incompatible_candidate(candidate) -> bool:
        """
        Exclude audio-specialized models from normal text chat routing.

        This routing path serves text chat completions; selecting audio-only
        models (e.g., gpt-audio-*) can trigger provider-side 400s for plain
        text payloads.
        """
        model_name = (getattr(candidate, "model_name", "") or "").lower()
        provider = (getattr(candidate, "provider", "") or "").lower()

        if provider != "openai":
            return False

        is_audio_specialized = (
            "gpt-audio" in model_name
            or "audio-preview" in model_name
            or "audio-mini" in model_name
            or "whisper" in model_name
            or model_name.startswith("tts")
        )

        # Legacy completion/embedding families are not valid modern chat targets
        # and frequently 404/deprecated at provider execution time.
        is_legacy_text_model = (
            model_name.startswith("text-")
            or model_name.startswith("ada")
            or model_name.startswith("babbage")
            or model_name.startswith("curie")
            or model_name.startswith("davinci")
            or model_name.startswith("gpt-3.5")
            or model_name.startswith("gpt-35")
            or model_name.endswith("-0301")
            or model_name.endswith("-0314")
            or model_name.endswith("-0613")
            or model_name == "gpt-5-chat"
        )

        return is_audio_specialized or is_legacy_text_model

    def select_model(
        self,
        input_tokens: int,
        requested_model: str,
        db_session,
        optimize: bool = True,
        quality: str = "auto",
        scope: str = None,
        require_tools: bool = False,
        require_vision: bool = False,
        require_json: bool = False,
        min_context_window: int = 0,
        min_output_tokens: int = 0,
        min_tier: str = None,
        max_tier_override: str = None,
        confidence: float = 1.0,
        risk_level: str = "low",
        allowed_model_ids: List[str] = None,
        task_type: str = "chat",
        query_text: str = "",
    ) -> str:
        self._reset_timing_breakdown()

        # If optimize=False, use exact requested model
        if not optimize:
            logger.debug(f"Optimization disabled, using requested: {requested_model}")
            return requested_model

        repo = db_session

        # 1. Look up requested model in DB to get metadata
        requested_meta_started = time.perf_counter()
        model_meta = repo.get_model_metadata(requested_model)
        self._bump_timing("requested_meta", (time.perf_counter() - requested_meta_started) * 1000.0)

        if model_meta:
            provider = model_meta["provider"]
            family = model_meta["family"]
        else:
            # Model not in DB - use normalizer as fallback
            logger.warning(f"Model {requested_model} not in DB, using normalizer")
            normalized = normalize_model(requested_model)
            provider = normalized["provider"]
            family = normalized["family"]

        # 2. Get routing config from min_quality setting
        quality_config = MIN_QUALITY_CONFIG.get(quality, MIN_QUALITY_CONFIG["auto"])
        effective_scope = scope if scope else quality_config["scope"]
        allowed_tiers = quality_config["allowed_tiers"]
        strict_family = quality_config.get("strict_family", False)

        # FinOps Max Tier Override
        # If set, filter allowed_tiers to exclude anything strictly higher than max_tier
        if max_tier_override:
            # TIER_ORDER = ["nano", "mini", "base", "premium"]
            # Logic: keep tiers <= max_tier
            try:
                max_idx = TIER_ORDER.index(max_tier_override)
                valid_tiers_finops = set(TIER_ORDER[:max_idx+1])
                # Intersect with quality allowed_tiers
                allowed_tiers = [t for t in allowed_tiers if t in valid_tiers_finops]
                logger.info(f"FinOps max_tier={max_tier_override} restricted allowed_tiers to: {allowed_tiers}")
                
                # If no tiers left, we might have a problem.
                # Usually shouldn't happen unless max_tier is 'nano' and quality is 'best'.
                if not allowed_tiers:
                    logger.warning(f"FinOps restriction left no allowed tiers. Fallback to max_tier {max_tier_override}")
                    allowed_tiers = [max_tier_override]
            except ValueError:
                logger.warning(f"Invalid max_tier_override: {max_tier_override}, ignoring.")

        # 3. Determine target tier — heuristic scorer or token fallback
        if min_tier:
            suggested_tier = _apply_confidence_bumping(min_tier, confidence, risk_level)
        else:
            suggested_tier = get_tier_for_tokens(input_tokens)
        target_tier = _select_target_tier(suggested_tier, allowed_tiers)

        # Capability floor: tool/json/vision constrained requests should not
        # route to nano-tier models even when the text itself is trivial.
        if (require_tools or require_json or require_vision) and target_tier == "nano":
            fallback_tier = next((t for t in ("mini", "base", "premium") if t in allowed_tiers), target_tier)
            if fallback_tier != target_tier:
                logger.info(
                    "Capability floor applied: tier %s -> %s (tools=%s, json=%s, vision=%s)",
                    target_tier,
                    fallback_tier,
                    require_tools,
                    require_json,
                    require_vision,
                )
                target_tier = fallback_tier

        # 4. Find best model based on scope with capability gating
        best_model = self._find_model_in_scope(
            repo=repo,
            scope=effective_scope,
            family=family,
            provider=provider,
            target_tier=target_tier,
            allowed_tiers=allowed_tiers,
            require_tools=require_tools,
            require_vision=require_vision,
            require_json=require_json,
            min_context_window=min_context_window,
            min_output_tokens=min_output_tokens,
            allowed_model_ids=allowed_model_ids,
            db_session=db_session,
            task_type=task_type,
            query_text=query_text,
            heuristic_confidence=confidence,
        )

        if best_model:
            # Only route if we found a DIFFERENT model that's cheaper
            # If best_model == requested_model, no routing needed
            if best_model != requested_model:
                logger.info(f"Routing: {requested_model} → {best_model} (scope: {effective_scope}, tier: {target_tier})")
            return best_model

        # No match found in scope
        if strict_family:
            # For high/auto quality: if no cheaper model in family, stay on requested
            logger.debug(f"No routing candidate in family {family}, using requested: {requested_model}")
            return requested_model
        else:
            # For medium/low: this shouldn't happen often, but fallback to requested
            logger.warning(f"No routing candidate found in provider {provider}, using requested: {requested_model}")
            return requested_model
    
    def _find_model_in_scope(
        self,
        repo,
        scope: str,
        family: str,
        provider: str,
        target_tier: str,
        allowed_tiers: List[str],
        require_tools: bool,
        require_vision: bool,
        require_json: bool,
        min_context_window: int,
        min_output_tokens: int,
        allowed_model_ids: List[str] = None,
        db_session = None,
        task_type: str = "chat",
        query_text: str = "",
        heuristic_confidence: float = 1.0,
    ) -> Optional[str]:
        """
        Find the best model within the specified scope with capability gating.
        Tries target tier first, then falls back through allowed tiers.
        Only tries tiers that are in allowed_tiers.
        """
        from .config import settings

        routing_started = time.perf_counter()
        routing_timeout_ms = max(0, int(getattr(settings, "ROUTING_HYBRID_TIMEOUT_MS", 900) or 0))

        # Only try tiers that are allowed - target_tier first if allowed, then others in order
        if target_tier in allowed_tiers:
            tiers_to_try = [target_tier] + [t for t in TIER_ORDER if t in allowed_tiers and t != target_tier]
        else:
            tiers_to_try = [t for t in TIER_ORDER if t in allowed_tiers]
        
        for tier in tiers_to_try:
            # Get candidates for this tier
            logger.info(f"Checking tier {tier} in scope {scope} (Allowed: {allowed_model_ids})")
            fetch_started = time.perf_counter()
            if scope == "family":
                candidates = repo.find_candidates(
                    scope="family",
                    family=family,
                    provider=provider,
                    allowed_tiers=[tier],
                    allowed_model_ids=allowed_model_ids
                )
            else:
                candidates = repo.find_candidates(
                    scope=scope,
                    provider=provider if scope == "provider" else None,
                    allowed_tiers=[tier],
                    allowed_model_ids=allowed_model_ids
                )
            self._bump_timing("candidate_fetch", (time.perf_counter() - fetch_started) * 1000.0)
            
            logger.info(f"Found {len(candidates)} candidates for tier {tier}")
            
            # Apply Hybrid Filtering (DB + Fallback + Modalities)
            filter_started = time.perf_counter()
            valid_candidates = []
            for candidate in candidates:
                # Exclude audio-specialized models for text chat requests.
                if self._is_text_incompatible_candidate(candidate):
                    continue

                # 1. Check Context Window (Trusted DB)
                if min_context_window > 0 and candidate.context_window > 0:
                    if candidate.context_window < min_context_window:
                        continue

                # 2. Check Max Output Tokens (Trusted DB)
                if min_output_tokens > 0 and candidate.max_output > 0:
                    if candidate.max_output < min_output_tokens:
                        continue

                # 3. Check Modalities (Trusted DB)
                if require_vision:
                    # Check if 'image' is in input_modalities (if available) or rely on supports_vision flag
                    # Since ModelPricing might not have input_modalities column yet, we use supports_vision
                    # which is synced from MCR input_modalities.
                    if not candidate.supports_vision:
                        # Fallback: check normalizer if DB is empty/default
                        from .model_normalizer import _supports_vision
                        if not _supports_vision(candidate.model_name, candidate.provider):
                            continue

                # 4. Check Tools (Untrusted DB -> Use Normalizer Fallback)
                if require_tools:
                    # Current pipeline uses OpenAI tool-calling via chat completions.
                    # gpt-5* models execute through Responses API in this codebase,
                    # where tool-call plumbing is not yet implemented end-to-end.
                    if candidate.provider == "openai" and "gpt-5" in candidate.model_name.lower():
                        continue
                    from .model_normalizer import _supports_tools
                    if not _supports_tools(candidate.model_name, candidate.provider):
                        continue

                # 5. Check JSON (Untrusted DB -> Use Normalizer Fallback)
                if require_json:
                    # Keep JSON-mode requests on chat-completions-compatible models
                    # until Responses API structured-output is fully wired.
                    if candidate.provider == "openai" and "gpt-5" in candidate.model_name.lower():
                        continue
                    from .model_normalizer import _supports_json
                    if not _supports_json(candidate.model_name, candidate.provider):
                        continue
                
                valid_candidates.append(candidate)
            self._bump_timing("candidate_filter", (time.perf_counter() - filter_started) * 1000.0)
            
            if valid_candidates:
                valid_candidates = self._prefer_pinned_anthropic_models(valid_candidates)
                if len(valid_candidates) == 1:
                    return valid_candidates[0].model_name

                use_quality_scoring = settings.ROUTING_QUALITY_COST_SCORING
                use_elo = settings.ROUTING_ELO_ENABLED
                use_latency = settings.ROUTING_LATENCY_AWARE
                cost_weight = settings.ROUTING_COST_WEIGHT
                cost_scaling = settings.ROUTING_COST_SCALING_FACTOR
                use_hybrid = getattr(settings, "ROUTING_HYBRID_SCORING", False)

                if not use_quality_scoring:
                    # Legacy behavior: cheapest candidate wins
                    return valid_candidates[0].model_name

                # ── Hybrid Ensemble Scoring (4-scorer algorithm) ──
                if use_hybrid and db_session:
                    try:
                        if routing_timeout_ms > 0:
                            elapsed_ms = int((time.perf_counter() - routing_started) * 1000)
                            if elapsed_ms >= routing_timeout_ms:
                                logger.warning(
                                    "Routing timeout budget exceeded before hybrid scoring (%dms >= %dms). "
                                    "Using cheapest candidate.",
                                    elapsed_ms,
                                    routing_timeout_ms,
                                )
                                return valid_candidates[0].model_name
                            remaining_ms = max(50, routing_timeout_ms - elapsed_ms)
                        else:
                            remaining_ms = None

                        from .hybrid_scorer import rank_candidates
                        hybrid_timings: dict[str, float] = {}
                        hybrid_started = time.perf_counter()
                        ranked = rank_candidates(
                            valid_candidates,
                            db_session=db_session,
                            task_type=task_type,
                            suggested_tier=target_tier,
                            heuristic_confidence=heuristic_confidence,
                            cost_weight=cost_weight,
                            cost_scaling_factor=cost_scaling,
                            use_elo=use_elo,
                            use_thompson=True,
                            timeout_ms=remaining_ms,
                            timings_out=hybrid_timings,
                        )
                        self._bump_timing("hybrid_rank", (time.perf_counter() - hybrid_started) * 1000.0)
                        for key, value in hybrid_timings.items():
                            self._bump_timing(f"hybrid_{key}", value)
                        if ranked:
                            best = ranked[0]
                            if best.model_name != valid_candidates[0].model_name:
                                logger.info(
                                    f"Hybrid scoring chose {best.model_name} over cheapest "
                                    f"{valid_candidates[0].model_name} "
                                    f"(final={best.final:.3f}, H={best.heuristic:.2f}, "
                                    f"Elo={best.elo:.2f}, "
                                    f"CQ={best.cost_quality:.2f}, TS={best.thompson:.2f})"
                                )
                            return best.model_name
                    except Exception as hybrid_err:
                        logger.warning(f"Hybrid scoring failed, falling back: {hybrid_err}")

                # ── Fallback: Original cost-quality scoring ──
                max_price = max(
                    (getattr(c, "input_price", 0.0) or 0.0) for c in valid_candidates
                )
                fallback_started = time.perf_counter()
                elo_bonuses = {}
                if use_elo and db_session:
                    try:
                        from .elo_rating import (
                            get_model_elo_scores_bulk,
                        )

                        bulk_scores = get_model_elo_scores_bulk(
                            db_session,
                            [c.model_name for c in valid_candidates],
                            task_type,
                        )
                        for candidate in valid_candidates:
                            normalized_score, confidence = bulk_scores.get(
                                candidate.model_name,
                                (0.5, 0.0),
                            )
                            if confidence < 0.3:
                                elo_bonuses[candidate.model_name] = 0.0
                                continue
                            elo_bonuses[candidate.model_name] = (
                                (normalized_score - 0.5) * 2.0 * 0.15 * confidence
                            )
                    except Exception:
                        pass
                best = max(
                    valid_candidates,
                    key=lambda c: _compute_candidate_score(
                        c, max_price,
                        cost_weight=cost_weight,
                        cost_scaling_factor=cost_scaling,
                        elo_bonus=elo_bonuses.get(c.model_name, 0.0),
                        use_latency=use_latency,
                    ),
                )
                self._bump_timing("fallback_rank", (time.perf_counter() - fallback_started) * 1000.0)
                if best.model_name != valid_candidates[0].model_name:
                    logger.info(
                        f"Quality-cost ranking chose {best.model_name} over cheapest "
                        f"{valid_candidates[0].model_name} "
                        f"(elo_bonus={elo_bonuses.get(best.model_name, 0.0):.3f})"
                    )
                return best.model_name

        return None


class PerformanceStrategy(RoutingStrategy):
    """
    Performance Strategy - Always uses the requested model.
    No cost optimization, maximum quality/capability.
    """

    def select_model(
        self,
        input_tokens: int,
        requested_model: str,
        db_session,
        optimize: bool = True,
        quality: str = "auto",
        scope: str = None,
        require_tools: bool = False,
        require_vision: bool = False,
        require_json: bool = False,
        min_context_window: int = 0,
        min_output_tokens: int = 0,
        min_tier: str = None,
        max_tier_override: str = None,
        confidence: float = 1.0,
        risk_level: str = "low",
        allowed_model_ids: List[str] = None,
        task_type: str = "chat",
        query_text: str = "",
    ) -> str:
        # Always use requested model - no routing
        return requested_model


class BalancedStrategy(RoutingStrategy):
    """
    Balanced Strategy - Moderate optimization.
    Only routes to same-tier alternatives, never downgrades capability.
    """

    def select_model(
        self,
        input_tokens: int,
        requested_model: str,
        db_session,
        optimize: bool = True,
        quality: str = "balanced",
        scope: str = None,
        require_tools: bool = False,
        require_vision: bool = False,
        require_json: bool = False,
        min_context_window: int = 0,
        min_output_tokens: int = 0,
        min_tier: str = None,
        max_tier_override: str = None,
        confidence: float = 1.0,
        risk_level: str = "low",
        allowed_model_ids: List[str] = None,
        task_type: str = "chat",
        query_text: str = "",
    ) -> str:
        if not optimize:
            return requested_model
        
        repo = db_session
        
        # Get model metadata
        model_meta = repo.get_model_metadata(requested_model)
        
        if not model_meta:
            return requested_model
        
        # Only route within same tier (no downgrade)
        # Note: This simple implementation doesn't check allowed_model_ids or min_output_tokens
        # It's recommended to use CostOptimizedStrategy for all production routing.
        current_tier = model_meta["tier"]
        family = model_meta["family"]
        provider = model_meta["provider"]
        
        # Pass allowed_model_ids if we want to filter candidates
        best_model = repo.find_best_model(
            family=family,
            tier=current_tier,
            provider=provider,
            allowed_model_ids=allowed_model_ids
        )
        
        return best_model if best_model else requested_model
