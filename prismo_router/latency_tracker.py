"""Standalone TPOT/TTFT latency tracker using Prismo's EMA formula."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

EMA_ALPHA_INITIAL = 0.3
EMA_ALPHA_STABLE = 0.05
EMA_WARMUP_SAMPLES = 20


@dataclass
class LatencyRecord:
    avg_ttft_ms: float | None = None
    avg_tpot_ms: float | None = None
    p95_latency_ms: int | None = None
    latency_sample_count: int = 0


class InMemoryLatencyStore:
    def __init__(self):
        self.records: dict[str, LatencyRecord] = {}

    def get_or_create(self, model_name: str) -> LatencyRecord:
        if model_name not in self.records:
            self.records[model_name] = LatencyRecord()
        return self.records[model_name]

    def get(self, model_name: str) -> LatencyRecord | None:
        return self.records.get(model_name)


DEFAULT_LATENCY_STORE = InMemoryLatencyStore()


def _resolve_store(db_session=None) -> InMemoryLatencyStore:
    return db_session if isinstance(db_session, InMemoryLatencyStore) else DEFAULT_LATENCY_STORE


def _ema_alpha(sample_count: int) -> float:
    if sample_count <= 0:
        return EMA_ALPHA_INITIAL
    if sample_count >= EMA_WARMUP_SAMPLES:
        return EMA_ALPHA_STABLE
    t = sample_count / EMA_WARMUP_SAMPLES
    return EMA_ALPHA_INITIAL + t * (EMA_ALPHA_STABLE - EMA_ALPHA_INITIAL)


def record_latency(
    db_session,
    model_name: str,
    ttft_ms: Optional[float] = None,
    tpot_ms: Optional[float] = None,
    total_latency_ms: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    pricing = _resolve_store(db_session).get_or_create(model_name)
    sample_count = pricing.latency_sample_count or 0
    alpha = _ema_alpha(sample_count)

    if tpot_ms is None and total_latency_ms and completion_tokens and completion_tokens > 0:
        if ttft_ms is not None:
            generation_time = max(0, total_latency_ms - ttft_ms)
            tpot_ms = generation_time / completion_tokens
        else:
            tpot_ms = total_latency_ms / completion_tokens

    if ttft_ms is not None and ttft_ms > 0:
        old_ttft = pricing.avg_ttft_ms
        pricing.avg_ttft_ms = ttft_ms if old_ttft is None or sample_count == 0 else alpha * ttft_ms + (1 - alpha) * old_ttft

    if tpot_ms is not None and tpot_ms > 0:
        old_tpot = pricing.avg_tpot_ms
        pricing.avg_tpot_ms = tpot_ms if old_tpot is None or sample_count == 0 else alpha * tpot_ms + (1 - alpha) * old_tpot

    if total_latency_ms is not None and total_latency_ms > 0:
        old_p95 = pricing.p95_latency_ms
        if old_p95 is None or sample_count == 0:
            pricing.p95_latency_ms = total_latency_ms
        else:
            p95_alpha = alpha * 0.5
            if total_latency_ms > old_p95:
                pricing.p95_latency_ms = int(alpha * total_latency_ms + (1 - alpha) * old_p95)
            else:
                pricing.p95_latency_ms = int(p95_alpha * total_latency_ms + (1 - p95_alpha) * old_p95)

    pricing.latency_sample_count = sample_count + 1


def get_estimated_latency(
    db_session,
    model_name: str,
    expected_output_tokens: int = 200,
) -> Optional[float]:
    pricing = _resolve_store(db_session).get(model_name)
    if not pricing:
        return None
    if pricing.avg_ttft_ms is not None and pricing.avg_tpot_ms is not None:
        return pricing.avg_ttft_ms + (pricing.avg_tpot_ms * expected_output_tokens)
    if pricing.p95_latency_ms:
        return float(pricing.p95_latency_ms)
    return None
