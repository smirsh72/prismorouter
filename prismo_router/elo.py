"""In-memory Elo feedback helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_RATING = 1500.0
K_FACTOR = 32.0
SELF_PLAY_DAMPEN = 0.1


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloStore:
    """Small in-memory Elo store for local routing experiments."""

    ratings: dict[tuple[str, str], float] = field(default_factory=dict)

    def get(self, model: str, task_type: str = "chat") -> float:
        return self.ratings.get((model, task_type), BASE_RATING)

    def normalized_bonus(self, model: str, task_type: str = "chat") -> float:
        rating = self.get(model, task_type)
        return max(-0.15, min(0.15, (rating - BASE_RATING) / 2000.0))

    def record_pairwise(self, winner: str, loser: str, task_type: str = "chat") -> None:
        winner_key = (winner, task_type)
        loser_key = (loser, task_type)
        winner_rating = self.get(winner, task_type)
        loser_rating = self.get(loser, task_type)
        winner_expected = expected_score(winner_rating, loser_rating)
        loser_expected = expected_score(loser_rating, winner_rating)
        self.ratings[winner_key] = winner_rating + K_FACTOR * (1.0 - winner_expected)
        self.ratings[loser_key] = loser_rating + K_FACTOR * (0.0 - loser_expected)

    def record_feedback(self, model: str, positive: bool, task_type: str = "chat") -> None:
        key = (model, task_type)
        rating = self.get(model, task_type)
        actual = 1.0 if positive else 0.0
        expected = expected_score(rating, BASE_RATING)
        self.ratings[key] = rating + (K_FACTOR * SELF_PLAY_DAMPEN) * (actual - expected)

