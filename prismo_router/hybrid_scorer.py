"""
Infrastructure/Strategy Layer — Hybrid Ensemble Scorer.

Portions of this ensemble scoring approach are adapted from vLLM Semantic Router.

Combines 4 sub-scorers:

  A. Heuristic Score (0.40) — 4-dimension weighted scorer
     "How complex is this query?"

  B. Elo Score (0.25) — pairwise Bradley-Terry per model/task-type
     "Which model has been performing best on this type of task?"

  C. Cost-Quality Score (0.20) — reliability metrics × cost efficiency
     "What's the best quality-per-dollar?"

  D. Thompson Sampling (0.15) — Beta distribution exploration/exploitation
     "Should we try an uncertain model to gather data?"

All scores normalized to 0-1 before combining.
Confidence = agreement ratio of sub-scorers (high agreement = high confidence).

NOTE: RouterDC (semantic-embedding cosine-similarity scorer) was removed
because sentence-transformers was dropped from the dependencies to fix OOM
on small cloud instances. Heuristic scoring already covers task classification
via keyword rules, and Elo provides grounded pairwise quality signal once
traffic accumulates.
"""
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from typing import Any as Session

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ENSEMBLE WEIGHTS (4-scorer, sum to 1.0)
# ─────────────────────────────────────────────────────────────────────
W_HEURISTIC = 0.40
W_ELO = 0.25
W_COST_QUALITY = 0.20
W_THOMPSON = 0.15


@dataclass
class CandidateScore:
    """Detailed scoring breakdown for one candidate model."""
    model_name: str
    heuristic: float = 0.5
    elo: float = 0.5
    cost_quality: float = 0.5
    thompson: float = 0.5
    final: float = 0.5
    confidence: float = 0.5
    breakdown: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# SUB-SCORER A: Heuristic Fit (0.40)
# ─────────────────────────────────────────────────────────────────────

# How well does the candidate tier match what the heuristic scorer suggested?
_TIER_RANK = {"nano": 0, "mini": 1, "base": 2, "premium": 3}


def _score_heuristic_fit(
    candidate_tier: str,
    suggested_tier: str,
    heuristic_confidence: float,
) -> float:
    """
    Score how well a candidate's tier matches the heuristic suggestion.

    Exact match = 1.0, one tier higher = 0.8 (safe overshoot),
    one tier lower = 0.4 (risky undershoot), two+ tiers off = scales down.

    Weighted by heuristic confidence — when the scorer is uncertain,
    tier mismatch penalty is softened.
    """
    c_rank = _TIER_RANK.get(candidate_tier, 1)
    s_rank = _TIER_RANK.get(suggested_tier, 1)
    diff = c_rank - s_rank  # positive = candidate is higher tier

    if diff == 0:
        raw = 1.0      # exact match
    elif diff == 1:
        raw = 0.8       # one tier above (safe, slightly wasteful)
    elif diff == -1:
        raw = 0.4       # one tier below (risky)
    elif diff >= 2:
        raw = 0.6       # two+ tiers above (wasteful but safe)
    else:
        raw = 0.15      # two+ tiers below (dangerous)

    # When confidence is low, soften the penalty (pull toward 0.5)
    return raw * heuristic_confidence + 0.5 * (1.0 - heuristic_confidence)


# ─────────────────────────────────────────────────────────────────────
# SUB-SCORER B: Elo Rating (0.25)
# ─────────────────────────────────────────────────────────────────────

def _score_elo(
    db_session: Session,
    candidate_model: str,
    task_type: str,
    stats_cache: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Get Elo score for a candidate model on a specific task type.

    Returns (score, confidence):
      - score: 0-1 normalized Elo rating (0.5 = neutral/no data)
      - confidence: 0-1 how reliable this score is (based on comparisons)
    """
    if stats_cache is not None:
        return stats_cache.get(candidate_model, (0.5, 0.0))

    try:
        from .elo_rating import get_model_elo_score
        score, confidence = get_model_elo_score(db_session, candidate_model, task_type)
        return score, confidence
    except Exception:
        return 0.5, 0.0


# ─────────────────────────────────────────────────────────────────────
# SUB-SCORER C: Cost-Quality (0.20)
# ─────────────────────────────────────────────────────────────────────

def _score_cost_quality(
    candidate,
    max_input_price: float,
    cost_weight: float = 0.4,
    cost_scaling_factor: float = 1.5,
) -> float:
    """
    Quality-adjusted cost efficiency score.

    quality = reliability * 0.5 + schema_pass * 0.25 + tool_success * 0.25
    cost_efficiency = 1 - normalized_cost
    score = quality * 0.6 + cost_efficiency * 0.4

    Normalized to 0-1. Higher = better quality for the price.
    """
    # Quality from DB metrics
    error_rate = getattr(candidate, "error_rate_7d", 0.0) or 0.0
    timeout_rate = getattr(candidate, "timeout_rate_7d", 0.0) or 0.0
    schema_pass = getattr(candidate, "schema_pass_rate_7d", 1.0) or 1.0
    tool_success = getattr(candidate, "tool_success_rate_7d", 1.0) or 1.0

    reliability = max(0.0, 1.0 - error_rate - timeout_rate)
    quality = reliability * 0.5 + schema_pass * 0.25 + tool_success * 0.25

    # Cost efficiency (cheaper = higher score)
    input_price = getattr(candidate, "input_price", 0.0) or 0.0
    if max_input_price > 0:
        cost_efficiency = 1.0 - (input_price / max_input_price)
    else:
        cost_efficiency = 1.0

    # Latency bonus/penalty (fast models score higher)
    latency_score = 1.0
    avg_ttft = getattr(candidate, "avg_ttft_ms", None)
    avg_tpot = getattr(candidate, "avg_tpot_ms", None)
    if avg_ttft and avg_tpot and avg_ttft > 0 and avg_tpot > 0:
        estimated_ms = avg_ttft + (avg_tpot * 200)
        # 0-500ms = 1.0, 500-3000ms scales down to 0.5, 3000+ = 0.3
        if estimated_ms <= 500:
            latency_score = 1.0
        elif estimated_ms <= 3000:
            latency_score = 1.0 - 0.5 * ((estimated_ms - 500) / 2500)
        else:
            latency_score = 0.3
    else:
        p95 = getattr(candidate, "p95_latency_ms", None)
        if p95 and p95 > 0:
            if p95 <= 500:
                latency_score = 1.0
            elif p95 <= 3000:
                latency_score = 1.0 - 0.5 * ((p95 - 500) / 2500)
            else:
                latency_score = 0.3

    # Combine: quality most important, then cost, then latency
    score = quality * 0.50 + cost_efficiency * 0.30 + latency_score * 0.20
    return min(1.0, max(0.0, score))


# ─────────────────────────────────────────────────────────────────────
# SUB-SCORER D: Thompson Sampling (0.15)
# ─────────────────────────────────────────────────────────────────────

def _score_thompson_with_confidence(
    db_session: Session,
    candidate_model: str,
    task_type: str,
    stats_cache: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Tuple[float, float]:
    """
    Thompson Sampling: sample from Beta(alpha, beta) distribution.

    alpha = wins + 1 (prior)
    beta  = losses + 1 (prior)

    New models with no data: Beta(1, 1) = uniform → random 0-1 score.
    Models with history: distribution narrows around true win rate.

    This provides natural exploration/exploitation:
    - Uncertain models get occasional high samples → tried more often
    - Models with strong track records get consistently high samples
    - Bad models get consistently low samples → avoided
    """
    if stats_cache is not None:
        cached = stats_cache.get(candidate_model)
        if cached is None:
            return 0.5, 0.0

        wins, losses, comparisons = cached
        alpha = wins + 1.0
        beta_param = losses + 1.0
        sample = random.betavariate(alpha, beta_param)
        confidence = min(1.0, comparisons / 20.0)
        return sample, confidence

    try:
        from .elo_rating import ModelEloRating
        rating = (
            db_session.query(ModelEloRating)
            .filter_by(model_name=candidate_model, task_type=task_type)
            .first()
        )

        if rating is None:
            # Cold start: avoid injecting random noise into routing.
            return 0.5, 0.0
        else:
            comparisons = int((rating.comparisons or 0))
            if comparisons <= 0:
                return 0.5, 0.0

            alpha = (rating.wins or 0) + 1.0
            beta_param = (rating.losses or 0) + 1.0

        # Sample from Beta distribution
        sample = random.betavariate(alpha, beta_param)
        confidence = min(1.0, comparisons / 20.0)
        return sample, confidence

    except Exception:
        # Fallback: neutral with zero confidence (no influence on final ranking)
        return 0.5, 0.0


# ─────────────────────────────────────────────────────────────────────
# ENSEMBLE COMBINER
# ─────────────────────────────────────────────────────────────────────

def score_candidate(
    candidate,
    *,
    db_session: Session,
    task_type: str,
    suggested_tier: str,
    heuristic_confidence: float,
    max_input_price: float,
    cost_weight: float = 0.4,
    cost_scaling_factor: float = 1.5,
    use_elo: bool = True,
    use_thompson: bool = True,
    elo_stats_cache: Optional[Dict[str, Tuple[float, float]]] = None,
    thompson_stats_cache: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> CandidateScore:
    """
    Score a single candidate model using the 4-scorer ensemble.

    All sub-scores are normalized to 0-1 before combining.
    Weights are adaptive: Elo weight scales with confidence,
    redistributing to Cost-Quality when Elo data is sparse.

    Returns CandidateScore with full breakdown.
    """
    model_name = candidate.model_name
    candidate_tier = getattr(candidate, "tier", "base") or "base"

    # A. Heuristic Fit (always runs)
    h_score = _score_heuristic_fit(candidate_tier, suggested_tier, heuristic_confidence)

    # B. Elo (gated, with confidence)
    if use_elo:
        elo_score, elo_confidence = _score_elo(
            db_session,
            model_name,
            task_type,
            stats_cache=elo_stats_cache,
        )
    else:
        elo_score, elo_confidence = 0.5, 0.0

    # C. Cost-Quality (always runs)
    cq_score = _score_cost_quality(candidate, max_input_price, cost_weight, cost_scaling_factor)

    # D. Thompson Sampling (gated)
    if use_thompson:
        ts_score, thompson_confidence = _score_thompson_with_confidence(
            db_session,
            model_name,
            task_type,
            stats_cache=thompson_stats_cache,
        )
    else:
        ts_score, thompson_confidence = 0.5, 0.0

    # --- Adaptive weighting ---
    # When data-driven scorers have low confidence, redistribute weight to Cost-Quality.
    # This stabilizes cold-start behavior and avoids neutral-noise influence.
    effective_elo_weight = W_ELO * elo_confidence
    effective_thompson_weight = W_THOMPSON * thompson_confidence
    redistributed = (
        W_ELO * (1.0 - elo_confidence)
        + W_THOMPSON * (1.0 - thompson_confidence)
    )

    w_h = W_HEURISTIC
    w_elo = effective_elo_weight
    w_cq = W_COST_QUALITY + redistributed
    w_ts = effective_thompson_weight

    # Normalize weights to sum to 1.0
    total_w = w_h + w_elo + w_cq + w_ts
    w_h /= total_w
    w_elo /= total_w
    w_cq /= total_w
    w_ts /= total_w

    # --- Final weighted score ---
    final = (
        w_h * h_score
        + w_elo * elo_score
        + w_cq * cq_score
        + w_ts * ts_score
    )

    # --- Confidence: agreement ratio of sub-scorers ---
    # If all scorers agree (all high or all low), confidence is high.
    # If they disagree, confidence drops.
    scores_list = [h_score, elo_score, cq_score, ts_score]
    mean = sum(scores_list) / len(scores_list)
    variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
    # Max variance for 4 scores in [0,1] is 0.25. Invert to confidence.
    agreement_confidence = max(0.0, 1.0 - (variance / 0.25))

    return CandidateScore(
        model_name=model_name,
        heuristic=round(h_score, 4),
        elo=round(elo_score, 4),
        cost_quality=round(cq_score, 4),
        thompson=round(ts_score, 4),
        final=round(final, 4),
        confidence=round(agreement_confidence, 4),
        breakdown={
            "w_heuristic": round(w_h, 3),
            "w_elo": round(w_elo, 3),
            "w_cost_quality": round(w_cq, 3),
            "w_thompson": round(w_ts, 3),
            "elo_confidence": round(elo_confidence, 3),
            "thompson_confidence": round(thompson_confidence, 3),
        },
    )


def rank_candidates(
    candidates: list,
    *,
    db_session: Session,
    task_type: str,
    suggested_tier: str,
    heuristic_confidence: float,
    timeout_ms: Optional[int] = None,
    cost_weight: float = 0.4,
    cost_scaling_factor: float = 1.5,
    use_elo: bool = True,
    use_thompson: bool = True,
    timings_out: Optional[Dict[str, float]] = None,
) -> List[CandidateScore]:
    """
    Rank all candidate models using the hybrid ensemble.

    Returns list of CandidateScore sorted by final score (highest first).
    """
    if not candidates:
        return []

    max_price = max(
        (getattr(c, "input_price", 0.0) or 0.0) for c in candidates
    )

    elo_stats_cache: Optional[Dict[str, Tuple[float, float]]] = None
    thompson_stats_cache: Optional[Dict[str, Tuple[float, float, float]]] = None
    model_names = [getattr(c, "model_name", None) for c in candidates]
    if (use_elo or use_thompson) and db_session:
        try:
            stats_started = time.perf_counter()
            from .elo_rating import (
                get_model_rating_stats_bulk,
            )

            bulk_stats = get_model_rating_stats_bulk(db_session, model_names, task_type)
            if use_elo:
                elo_stats_cache = {
                    name: (values[0], values[1])
                    for name, values in bulk_stats.items()
                }
            if use_thompson:
                thompson_stats_cache = {
                    name: (values[2], values[3], values[4])
                    for name, values in bulk_stats.items()
                    if values[4] > 0
                }
            if timings_out is not None:
                timings_out["stats_query"] = (time.perf_counter() - stats_started) * 1000.0
        except Exception:
            elo_stats_cache = None
            thompson_stats_cache = None

    start = time.perf_counter()
    scored = []
    score_loop_started = time.perf_counter()

    for candidate in candidates:
        if timeout_ms and timeout_ms > 0:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms >= timeout_ms:
                logger.warning(
                    "Hybrid scoring timeout budget exceeded (%dms). Falling back to cheaper scorer.",
                    timeout_ms,
                )
                return []

        cs = score_candidate(
            candidate,
            db_session=db_session,
            task_type=task_type,
            suggested_tier=suggested_tier,
            heuristic_confidence=heuristic_confidence,
            max_input_price=max_price,
            cost_weight=cost_weight,
            cost_scaling_factor=cost_scaling_factor,
            use_elo=use_elo,
            use_thompson=use_thompson,
            elo_stats_cache=elo_stats_cache,
            thompson_stats_cache=thompson_stats_cache,
        )
        scored.append(cs)
    if timings_out is not None:
        timings_out["score_loop"] = (time.perf_counter() - score_loop_started) * 1000.0

    scored.sort(key=lambda s: s.final, reverse=True)

    if len(scored) > 1:
        logger.info(
            f"Hybrid ranking: {scored[0].model_name}({scored[0].final:.3f}) > "
            f"{scored[1].model_name}({scored[1].final:.3f}) | "
            f"H={scored[0].heuristic:.2f} "
            f"Elo={scored[0].elo:.2f} CQ={scored[0].cost_quality:.2f} "
            f"TS={scored[0].thompson:.2f}"
        )

    return scored
