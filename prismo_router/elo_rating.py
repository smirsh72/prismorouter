"""Standalone Elo rating system.

Portions of this pairwise Elo implementation are adapted from vLLM Semantic
Router and modified to use in-memory storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

BASE_RATING = 1500.0
K_FACTOR = 32.0
MIN_COMPARISONS = 5
SELF_PLAY_DAMPEN = 0.1
COST_WEIGHT = 0.3
COST_SCALING_FACTOR = 1.5


@dataclass
class ModelEloRating:
    model_name: str
    task_type: str
    rating: float = BASE_RATING
    comparisons: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0


@dataclass
class InMemoryEloStore:
    ratings: Dict[tuple[str, str], ModelEloRating] = field(default_factory=dict)

    def get_or_create(self, model_name: str, task_type: str) -> ModelEloRating:
        key = (model_name, task_type)
        if key not in self.ratings:
            self.ratings[key] = ModelEloRating(model_name=model_name, task_type=task_type)
        return self.ratings[key]


DEFAULT_ELO_STORE = InMemoryEloStore()


def _resolve_store(db_session=None) -> InMemoryEloStore:
    return db_session if isinstance(db_session, InMemoryEloStore) else DEFAULT_ELO_STORE


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_confidence(comparisons: int) -> float:
    if comparisons <= 0:
        return 0.0
    return min(1.0, comparisons / max(1, MIN_COMPARISONS))


def _normalize_rating(row: ModelEloRating) -> float:
    return max(0.0, min(1.0, (row.rating - 1100.0) / 800.0))


def get_or_create_rating(db_session, model_name: str, task_type: str) -> ModelEloRating:
    return _resolve_store(db_session).get_or_create(model_name, task_type)


def record_pairwise(
    db_session,
    winner_model: str,
    loser_model: str,
    task_type: str,
    draw: bool = False,
) -> Tuple[ModelEloRating, ModelEloRating]:
    winner_obj = get_or_create_rating(db_session, winner_model, task_type)
    loser_obj = get_or_create_rating(db_session, loser_model, task_type)

    e_winner = _expected_score(winner_obj.rating, loser_obj.rating)
    e_loser = _expected_score(loser_obj.rating, winner_obj.rating)
    actual_winner = 0.5 if draw else 1.0
    actual_loser = 0.5 if draw else 0.0

    winner_obj.rating += K_FACTOR * (actual_winner - e_winner)
    loser_obj.rating += K_FACTOR * (actual_loser - e_loser)
    winner_obj.comparisons += 1
    loser_obj.comparisons += 1
    if draw:
        winner_obj.draws += 1
        loser_obj.draws += 1
    else:
        winner_obj.wins += 1
        loser_obj.losses += 1
    return winner_obj, loser_obj


def record_feedback(db_session, model_name: str, task_type: str, outcome: int) -> ModelEloRating:
    if outcome == 0:
        return get_or_create_rating(db_session, model_name, task_type)
    rating_obj = get_or_create_rating(db_session, model_name, task_type)
    actual_score = 1.0 if outcome > 0 else 0.0
    expected = _expected_score(rating_obj.rating, BASE_RATING)
    rating_obj.rating += (K_FACTOR * SELF_PLAY_DAMPEN) * (actual_score - expected)
    rating_obj.comparisons += 1
    if outcome > 0:
        rating_obj.wins += 1
    else:
        rating_obj.losses += 1
    return rating_obj


def get_model_elo_score(db_session, model_name: str, task_type: str) -> tuple[float, float]:
    row = get_or_create_rating(db_session, model_name, task_type)
    return _normalize_rating(row), _elo_confidence(row.comparisons)


def get_model_elo_scores_bulk(
    db_session,
    model_names: Iterable[str],
    task_type: str,
) -> dict[str, tuple[float, float]]:
    return {
        name: get_model_elo_score(db_session, name, task_type)
        for name in model_names
    }


def get_model_rating_stats_bulk(
    db_session,
    model_names: Iterable[str],
    task_type: str,
) -> dict[str, tuple[float, float, float, float, float]]:
    """Return normalized rating, confidence, wins, losses, comparisons."""
    store = _resolve_store(db_session)
    stats = {}
    for name in model_names:
        row = store.get_or_create(name, task_type)
        stats[name] = (
            _normalize_rating(row),
            _elo_confidence(row.comparisons),
            float(row.wins or 0),
            float(row.losses or 0),
            float(row.comparisons or 0),
        )
    return stats


def get_cost_adjusted_elo_score(
    db_session,
    model_name: str,
    task_type: str,
    normalized_cost: float,
    cost_weight: float = COST_WEIGHT,
    cost_scaling_factor: float = COST_SCALING_FACTOR,
) -> float:
    normalized_score, confidence = get_model_elo_score(db_session, model_name, task_type)
    base_score = normalized_score if confidence >= 0.1 else 0.5
    cost_bonus = (1.0 - normalized_cost) * cost_weight * cost_scaling_factor
    return base_score * (1.0 + cost_bonus)


def init_rating_from_config_score(
    db_session,
    model_name: str,
    task_type: str,
    config_score: float,
) -> ModelEloRating:
    rating_obj = get_or_create_rating(db_session, model_name, task_type)
    if rating_obj.comparisons == 0:
        rating_obj.rating = 1000.0 + (config_score * 1000.0)
    return rating_obj
