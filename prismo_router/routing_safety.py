"""
Routing Safety Utilities.

Determines when model downgrading should be prevented for
high-stakes or sensitive queries that require maximum quality.
"""
import re
from typing import List, Tuple

# Patterns that indicate HIGH-STAKES content - never downgrade
HIGH_STAKES_PATTERNS: List[str] = [
    # Legal
    "legal",
    "lawyer",
    "attorney",
    "lawsuit",
    "contract",
    "liability",
    "compliance",
    "regulation",
    "court",
    "litigation",
    
    # Medical
    "medical",
    "diagnosis",
    "treatment",
    "medication",
    "prescription",
    "symptoms",
    "doctor",
    "patient",
    "healthcare",
    "clinical",
    "drug interaction",
    "contraindication",
    "drug dosage",
    "overdose",
    "side effect",
    "adverse reaction",
    "pharmaceutical",
    "pathology",
    "prognosis",
    "surgical",
    
    # Financial
    "financial advice",
    "investment",
    "tax",
    "audit",
    "fiduciary",
    "securities",
    "trading",
    
    # Safety-critical
    "safety critical",
    "life or death",
    "emergency",
    "urgent medical",
    
    # Explicit quality requests
    "best model",
    "highest quality",
    "most accurate",
    "don't downgrade",
    "use gpt-4",
    "use gpt-5",
    "premium model",
]

# Compiled regex for efficiency.
# Uses \b only at the START to allow plural/inflected suffixes
# (e.g. "drug interactions" matches "drug interaction", "medications" matches "medication").
# For safety detection, false positives (blocked routing) are always safer than false negatives.
_HIGH_STAKES_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in HIGH_STAKES_PATTERNS) + r')',
    re.IGNORECASE
)


def should_prevent_downgrade(text: str) -> Tuple[bool, str]:
    """
    Determine if model downgrading should be prevented for this request.
    
    Args:
        text: The input text/prompt
        
    Returns:
        Tuple of (should_prevent: bool, reason: str)
    """
    match = _HIGH_STAKES_REGEX.search(text)
    if match:
        return True, f"high_stakes_pattern:{match.group(0)}"
    
    return False, ""


def is_high_stakes_request(text: str) -> bool:
    """
    Simple boolean check for high-stakes content.
    """
    return _HIGH_STAKES_REGEX.search(text) is not None
