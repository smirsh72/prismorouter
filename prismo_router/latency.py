"""In-memory latency tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyStats:
    avg_ttft_ms: float | None = None
    avg_tpot_ms: float | None = None
    samples: int = 0

    def record(self, ttft_ms: float | None = None, tpot_ms: float | None = None) -> None:
        alpha = 0.3 if self.samples < 20 else 0.05
        if ttft_ms is not None and ttft_ms > 0:
            self.avg_ttft_ms = ttft_ms if self.avg_ttft_ms is None else alpha * ttft_ms + (1 - alpha) * self.avg_ttft_ms
        if tpot_ms is not None and tpot_ms > 0:
            self.avg_tpot_ms = tpot_ms if self.avg_tpot_ms is None else alpha * tpot_ms + (1 - alpha) * self.avg_tpot_ms
        self.samples += 1

    def estimate_total_ms(self, expected_output_tokens: int = 200) -> float | None:
        if self.avg_ttft_ms is None or self.avg_tpot_ms is None:
            return None
        return self.avg_ttft_ms + self.avg_tpot_ms * expected_output_tokens

