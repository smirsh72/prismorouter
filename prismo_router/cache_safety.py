"""
Cache Safety Utilities.

Determines when semantic caching should be skipped to prevent
returning incorrect cached responses for time-sensitive, random,
or unique content requests.
"""
import re
from typing import List

# Patterns that indicate the response should NOT be cached
# These queries expect unique/fresh responses each time
NEVER_CACHE_PATTERNS: List[str] = [
    # Randomness/Uniqueness
    "random",
    "unique", 
    "generate new",
    "create new",
    "different each time",
    "vary",
    "shuffle",
    
    # Time-sensitive
    "current date",
    "current time",
    "today",
    "now",
    "latest",
    "recent",
    "live",
    "real-time",
    "up to date",
    "as of",
    
    # Creative/Variable output
    "creative",
    "brainstorm",
    "come up with",
    "suggest alternatives",
    "give me options",
    
    # Explicit freshness requests
    "fresh",
    "new response",
    "don't use cache",
    "no cache",
]

# Models that should NEVER be cached (reasoning models, etc.)
NEVER_CACHE_MODELS: List[str] = [
    "o1",
    "o1-preview", 
    "o1-mini",
    "o3",
    "o3-mini",
    "gpt-5-reasoning",
]

# Compiled regex for efficiency
_PATTERN_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in NEVER_CACHE_PATTERNS) + r')\b',
    re.IGNORECASE
)


def should_skip_cache(text: str, model: str = None) -> bool:
    """
    Determine if caching should be skipped for this request.
    
    Args:
        text: The input text/prompt
        model: The model being used (optional)
        
    Returns:
        True if cache should be skipped, False otherwise
    """
    # Check model exclusions
    if model:
        model_lower = model.lower()
        for excluded in NEVER_CACHE_MODELS:
            if excluded in model_lower:
                return True
    
    # Check text patterns
    if _PATTERN_REGEX.search(text):
        return True
    
    return False


def get_skip_reason(text: str, model: str = None) -> str:
    """
    Get the reason why cache was skipped (for logging).
    
    Returns empty string if cache should NOT be skipped.
    """
    if model:
        model_lower = model.lower()
        for excluded in NEVER_CACHE_MODELS:
            if excluded in model_lower:
                return f"model_excluded:{excluded}"
    
    match = _PATTERN_REGEX.search(text)
    if match:
        return f"pattern_matched:{match.group(0)}"
    
    return ""
