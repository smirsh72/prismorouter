"""Prompt-injection and jailbreak detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JailbreakResult:
    detected: bool
    confidence: float
    category: str | None = None
    matched_text: str | None = None


_PATTERNS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (
        re.compile(
            r"\b(ignore|forget|disregard|override) (?:all |any )?(?:previous|prior|above|earlier|safety) "
            r"(?:instructions|rules|prompts|guidelines)\b",
            re.IGNORECASE,
        ),
        "role_override",
        0.85,
    ),
    (
        re.compile(r"(?:</?(?:system|instruction|prompt|rules?)>|<\|im_end\|>|\[/?(?:SYSTEM|INST|SYS)\])", re.IGNORECASE),
        "delimiter",
        0.90,
    ),
    (
        re.compile(r"\b(?:base64|hex|rot13|binary)\s*(?:decode|encode|translate|convert)\b", re.IGNORECASE),
        "encoding",
        0.75,
    ),
    (
        re.compile(r"\b(?:repeat|show|reveal|display|print|output|tell me)\s+(?:your|the)\s+(?:system|initial|original|full)\s+(?:prompt|instructions|message|rules)\b", re.IGNORECASE),
        "extraction",
        0.80,
    ),
)


def detect_jailbreak(text: str) -> JailbreakResult:
    """Return the highest-confidence prompt-injection match."""
    matches: list[JailbreakResult] = []
    for pattern, category, confidence in _PATTERNS:
        match = pattern.search(text or "")
        if match:
            matches.append(JailbreakResult(True, confidence, category, match.group(0)[:80]))
    if not matches:
        return JailbreakResult(False, 0.0)
    return max(matches, key=lambda item: item.confidence)

