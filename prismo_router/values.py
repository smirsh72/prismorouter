"""
Domain Layer (Value Objects).
Defines immutable, self-validating value objects for the Chat Domain.
This prevents "Primitive Obsession" (using raw strings/ints for complex concepts).

ModelName accepts ANY non-empty model string (no allowlist).
Family detection uses prefix/substring matching.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelName:
    """
    Value Object representing a specific LLM Model Name.
    Accepts any non-empty model string - no validation against an allowlist.
    Immutable (frozen=True).
    """
    value: str

    def __post_init__(self):
        # Normalize first: strip whitespace and lowercase
        normalized = self.value.strip().lower() if self.value else ""
        if not normalized:
            raise ValueError("Model name cannot be empty")
        object.__setattr__(self, 'value', normalized)

    @property
    def provider(self) -> str:
        """Determines the provider based on the model name convention."""
        import re
        if "gpt" in self.value or "davinci" in self.value or "codex" in self.value or re.match(r"o\d+-", self.value):
            return "openai"
        if "claude" in self.value:
            return "anthropic"
        if "gemini" in self.value:
            return "google"
        if "llama" in self.value or "mixtral" in self.value:
            return "local"
        return "openai"  # Default fallback

    # =========================================================================
    # Family Detection (for routing decisions)
    # =========================================================================
    
    def is_gpt5_family(self) -> bool:
        """Checks if model belongs to the GPT-5 generation."""
        return "gpt-5" in self.value

    def is_gpt41_family(self) -> bool:
        """Checks if model belongs to the GPT-4.1 family."""
        return "gpt-4.1" in self.value

    def is_gpt4o_family(self) -> bool:
        """Checks if model belongs to the GPT-4o family."""
        return "gpt-4o" in self.value and "gpt-4.1" not in self.value

    def is_gpt4_turbo_family(self) -> bool:
        """Checks if model belongs to the GPT-4 Turbo family."""
        return "gpt-4-turbo" in self.value

    def is_gpt4_family(self) -> bool:
        """Checks if model belongs to any GPT-4 generation."""
        return "gpt-4" in self.value

    def is_gpt35_family(self) -> bool:
        """Checks if model belongs to the GPT-3.5 family."""
        return "gpt-3.5" in self.value

    def is_realtime(self) -> bool:
        """Checks if this is a realtime model."""
        return "realtime" in self.value

    def is_audio(self) -> bool:
        """Checks if this is an audio model."""
        return "audio" in self.value or "whisper" in self.value or "tts" in self.value

    def canonical_family(self) -> str:
        """
        Returns the canonical family name for routing decisions.
        Used to group models for fallback logic.
        """
        # 1. GPT-5 Family
        if self.is_gpt5_family():
            return "gpt-5"

        # 2. GPT-4.1 Family (check before generic gpt-4)
        if self.is_gpt41_family():
            return "gpt-4.1"

        # 3. GPT-4o (Omni) Family
        if self.is_gpt4o_family():
            return "gpt-4o"

        # 4. GPT-4 Turbo Family (check before generic gpt-4)
        if self.is_gpt4_turbo_family():
            return "gpt-4-turbo"

        # 5. GPT-4 Family (Classic)
        if self.is_gpt4_family():
            return "gpt-4"
            
        if self.is_gpt35_family():
            return "gpt-3.5"
        if "claude" in self.value:
            return "claude"
        if "gemini" in self.value:
            return "gemini"
            
        # Default Fallback
        return "gpt-4"

    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class TokenUsage:
    """
    Value Object representing token consumption.
    Enforces non-negative constraints.
    Immutable (frozen=True).
    """
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self):
        if self.prompt_tokens < 0:
            raise ValueError(f"Prompt tokens cannot be negative: {self.prompt_tokens}")
        if self.completion_tokens < 0:
            raise ValueError(f"Completion tokens cannot be negative: {self.completion_tokens}")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def calculate_cost(self, input_price_per_1k: float, output_price_per_1k: float) -> float:
        """Calculates the cost for this usage."""
        return (
            (self.prompt_tokens / 1000.0) * input_price_per_1k +
            (self.completion_tokens / 1000.0) * output_price_per_1k
        )
