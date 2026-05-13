"""
Model Normalizer - SINGLE source of provider-specific naming heuristics.

This is the ONLY place where we parse model names to determine:
- provider (openai, anthropic, google)
- family (gpt-4o, claude-3, etc.)
- tier (nano, mini, base, premium)
- capabilities (tools, vision, json)

Called during pricing ingestion, NOT during routing.
Routing uses DB columns directly.
"""
import re
from typing import TypedDict, Optional


class NormalizedModel(TypedDict):
    provider: str
    family: str
    tier: str
    supports_tools: bool
    supports_vision: bool
    supports_json: bool
    max_output: Optional[int]


def normalize_model(model_id: str) -> NormalizedModel:
    """
    Normalize a model ID into structured metadata.
    
    This function contains ALL provider-specific naming heuristics.
    Keep them here and nowhere else.
    
    Args:
        model_id: The raw model ID (e.g., "gpt-4o-mini-2024-07-18", "claude-3-5-sonnet-20241022")
        
    Returns:
        NormalizedModel with provider, family, tier, and capabilities
    """
    model_lower = model_id.lower().strip()
    
    # === PROVIDER DETECTION ===
    provider = _detect_provider(model_lower)
    
    # === FAMILY DETECTION ===
    family = _detect_family(model_lower, provider)
    
    # === TIER DETECTION ===
    tier = _detect_tier(model_lower, provider)
    
    # === CAPABILITY DETECTION ===
    supports_tools = _supports_tools(model_lower, provider)
    supports_vision = _supports_vision(model_lower, provider)
    supports_json = _supports_json(model_lower, provider)
    max_output = _get_max_output(model_lower, provider)
    
    return NormalizedModel(
        provider=provider,
        family=family,
        tier=tier,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_json=supports_json,
        max_output=max_output
    )


def _detect_provider(model: str) -> str:
    """Detect the provider from model name."""
    # Check for O-series explicitly (o1, o3, o4)
    if any(x in model for x in ["gpt", "codex", "davinci", "babbage", "ada", "whisper", "tts", "dall-e"]):
        return "openai"
    # O-series prefixes (with dash or standalone)
    if "o1" in model or "o3" in model or "o4" in model:
        return "openai"
        
    if "claude" in model:
        return "anthropic"
    if "gemini" in model:
        return "google"
    if any(x in model for x in ["llama", "mixtral", "mistral"]):
        return "meta"
    return "openai"  # Default fallback


def _detect_family(model: str, provider: str) -> str:
    """Detect the model family for routing grouping."""
    if provider == "openai":
        # Embeddings
        if "embedding" in model:
            return "embedding"

        # GPT-4o family (includes mini, nano variants)
        if "gpt-4o" in model:
            return "gpt-4o"
        # GPT-5.2 family (check before 5.1 and 5)
        if "gpt-5.2" in model:
            return "gpt-5.2"
        # GPT-5.1 / codex family (check before generic gpt-5)
        if "gpt-5.1" in model or "5.1-codex" in model:
            return "gpt-5.1"
        # GPT-5 base family
        if "gpt-5" in model:
            return "gpt-5"
        # GPT-4.1 family
        if "gpt-4.1" in model:
            return "gpt-4.1"
        # GPT-4 Turbo
        if "gpt-4-turbo" in model:
            return "gpt-4-turbo"
        # GPT-4 classic
        if "gpt-4" in model:
            return "gpt-4"
        # GPT-3.5
        if "gpt-3.5" in model:
            return "gpt-3.5"
        
        # o1 reasoning models
        # Match "o1" (exact), "o1-" (prefix), or "/o1" (path end)
        if "o1" in model:
            # Avoid matching "pro1" or similar
            if model == "o1" or "o1-" in model or "/o1" in model or "o1-preview" in model:
                return "o1"
        
        # o3 reasoning models
        if "o3" in model:
            if model == "o3" or "o3-" in model or "/o3" in model:
                return "o3"
        
        # o4 reasoning models
        if "o4" in model:
            if model == "o4" or "o4-" in model or "/o4" in model:
                return "o4"
                
        # Regex fallback: extract version from unknown GPT/o-series names
        # e.g. "gpt-5.4-turbo" → "gpt-5.4",  "o5-mini" → "o5"
        gpt_match = re.search(r"gpt-(\d+(?:\.\d+)?)", model)
        if gpt_match:
            return f"gpt-{gpt_match.group(1)}"
        o_match = re.search(r"\bo(\d+)-", model)
        if o_match:
            return f"o{o_match.group(1)}"
        return "gpt-4"  # Last-resort fallback
    
    elif provider == "anthropic":
        # Claude family detection - group by GENERATION so Sonnet/Haiku/Opus can route together
        # Haiku=mini, Sonnet=base, Opus=premium are tiers within the same generation family

        # Claude 4.6 generation (newest)
        if any(x in model for x in ["4-6", "4.6"]):
            return "claude-4.6"

        # Claude 4.5 generation - check before 4.1 and 4.0
        # Includes: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5
        if any(x in model for x in ["4-5", "4.5"]):
            return "claude-4.5"

        # Claude 4.1 generation (check before generic claude-4 catch-all)
        # Includes: claude-opus-4-1
        if "-4-1-" in model or model.endswith("-4-1"):
            return "claude-4.1"

        # Claude 4 generation
        # Includes: claude-opus-4, claude-sonnet-4, claude-4-*
        if "claude-opus-4" in model or "claude-sonnet-4" in model or "claude-4-" in model:
            return "claude-4"
        
        # Claude 3.7 generation
        if "3-7" in model or "3.7" in model:
            return "claude-3.7"
        
        # Claude 3.5 generation
        # Includes: claude-3-5-sonnet, claude-3-5-haiku, claude-3.5-*
        if "3-5" in model or "3.5" in model:
            return "claude-3.5"
        
        # Claude 3 generation
        # Includes: claude-3-opus, claude-3-sonnet, claude-3-haiku
        if "claude-3-" in model or "claude-3" in model:
            return "claude-3"
        
        # Claude 2
        if "claude-2" in model:
            return "claude-2"
        
        # Regex fallback: extract version from unknown Claude names
        # e.g. "claude-opus-5-0-20270101" → "claude-5.0"
        # e.g. "claude-5-20270101" → "claude-5"
        ver_match = re.search(r"claude(?:-\w+)?-(\d+)-(\d+)", model)
        if ver_match:
            return f"claude-{ver_match.group(1)}.{ver_match.group(2)}"
        major_match = re.search(r"claude-(\d+)", model)
        if major_match:
            return f"claude-{major_match.group(1)}"
        return "claude-unknown"
    
    elif provider == "google":
        if "gemini-2" in model:
            return "gemini-2"
        if "gemini-1.5" in model:
            return "gemini-1.5"
        if "gemini-1" in model or "gemini-pro" in model:
            return "gemini-1"
        return "gemini-1.5"  # Fallback
    
    return "unknown"


def _detect_tier(model: str, provider: str) -> str:
    """
    Detect the tier for cost-based routing.
    
    Tiers:
    - nano: Cheapest, simplest tasks
    - mini: Cheap, simple-moderate tasks
    - base: Standard capability
    - premium: Most capable, expensive
    """
    if provider == "openai":
        # Embeddings
        if "embedding" in model:
            return "nano"

        # Explicit tier markers in name
        # Exception: "gpt-4-mini" or similar
        if "nano" in model:
            return "nano"
        if "mini" in model:
            return "mini"
        
        # O-series base models are Premium (reasoning/expensive)
        if "o1" in model or "o3" in model or "o4" in model:
            # We already returned "mini" above if present
            # So anything left is Base or Pro → map to Premium
            return "premium"
            
        # GPT-4 turbo and base are standard (Base)
        # GPT-5 base is also Base unless specified
        return "base"
    
    elif provider == "anthropic":
        # Claude tier mapping:
        # - Haiku = fast/cheap → mini
        # - Sonnet = balanced → base  
        # - Opus = most capable → premium
        # Note: claude-opus-4 and claude-sonnet-4 use new naming convention
        if "haiku" in model:
            return "mini"
        if "opus" in model:
            return "premium"
        if "sonnet" in model:
            return "base"
        return "base"  # Default for unknown Claude
    
    elif provider == "google":
        if "flash" in model:
            return "mini"
        if "pro" in model:
            return "base"
        if "ultra" in model:
            return "premium"
        return "base"
    
    return "base"  # Safe default


# Claude models with full capability support (tools, vision, JSON)
_CLAUDE_CAPABLE_PATTERNS = ["claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku"]


def _is_capable_claude(model: str) -> bool:
    """Check if Claude model supports modern capabilities."""
    return any(p in model for p in _CLAUDE_CAPABLE_PATTERNS)


def _supports_tools(model: str, provider: str) -> bool:
    """Check if model supports function calling/tools."""
    if provider == "openai":
        # Broad capability check for OpenAI families
        if any(x in model for x in ["gpt-4", "gpt-5", "o1", "o3", "gpt-3.5-turbo"]):
            return True
    elif provider == "anthropic":
        return _is_capable_claude(model)
    elif provider == "google":
        return "gemini" in model
    return False


def _supports_vision(model: str, provider: str) -> bool:
    """Check if model supports image input."""
    if provider == "openai":
        # GPT-4o, Turbo, and new models support vision
        if any(x in model for x in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5", "gpt-4.1"]):
            return True
    elif provider == "anthropic":
        return _is_capable_claude(model)
    elif provider == "google":
        return "gemini" in model and "flash" not in model
    return False


def _supports_json(model: str, provider: str) -> bool:
    """Check if model supports JSON mode output."""
    if provider == "openai":
        if any(x in model for x in ["gpt-4", "gpt-5", "gpt-3.5-turbo", "o1", "o3"]):
            return True
    elif provider == "anthropic":
        return _is_capable_claude(model)
    elif provider == "google":
        return "gemini" in model
    return False


def _get_max_output(model: str, provider: str) -> Optional[int]:
    """Get max output tokens for the model."""
    if provider == "openai":
        # Specific Batch / Audio handling if needed
        # But generally they inherit from base
        
        # O-series
        if "o1" in model:
            if "mini" in model:
                return 65536
            return 32768
        if "o3" in model:
            return 100000
            
        # GPT-5 series
        if "gpt-5" in model:
            return 128000
            
        # GPT-4 series
        if "gpt-4o" in model:
            return 16384 # OpenRouter reports 16k output for 4o usually
        if "gpt-4-turbo" in model:
            return 4096
        if "gpt-4.1" in model:
            return 32768 # Estimated
        if "gpt-4" in model: # Classic
            return 4096
            
        # GPT-3.5
        if "gpt-3.5" in model:
            return 4096
            
    elif provider == "anthropic":
        # Opus 4.6 has 128K output; Sonnet/Haiku 4.6 and 4.5 have 64K
        if "opus-4-6" in model:
            return 131072
        if any(x in model for x in ["4-6", "4.6", "4-5", "4.5"]):
            return 65536
        # Claude 4.1 Opus has 32K; Claude 4 Sonnet has 64K, Opus 32K
        if "opus-4-1" in model:
            return 32768
        if "opus-4" in model:
            return 32768
        if "sonnet-4" in model:
            return 65536
        # Claude 3.7 has 128K extended thinking output
        if "3-7" in model or "3.7" in model:
            return 131072
        # Claude 3.5 and older Claude 3 have 8192
        if any(x in model for x in ["claude-3-5", "claude-3.5"]):
            return 8192
        if "claude-3" in model:
            return 4096
    elif provider == "google":
        if "gemini" in model:
            return 8192
    return None


# === HELPER: Batch normalize multiple models ===

def normalize_models(model_ids: list[str]) -> dict[str, NormalizedModel]:
    """Normalize a batch of model IDs."""
    return {model_id: normalize_model(model_id) for model_id in model_ids}
