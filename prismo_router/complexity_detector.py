"""
Infrastructure/Strategy Layer — Contrastive Complexity Detection (Dual-Signal).

Portions of this contrastive complexity detection approach are adapted from
vLLM Semantic Router.

Contrastive complexity classifier that determines query difficulty using two independent signals:

  1. Semantic signal (d_sem): Text embedding similarity against hard/easy exemplars
  2. Visual signal (d_vis): Image-aware complexity from multimodal request metadata

Combined: difficulty = max(abs(d_vis), abs(d_sem)) with sign from the dominant signal.
This ensures multimodal queries (image analysis, OCR, visual reasoning) are routed
to capable models even if the text prompt alone seems simple.

Uses sentence-transformers (all-MiniLM-L6-v2) for lightweight text embeddings.
Visual signal uses request metadata heuristics (image count, analysis keywords).
Falls back gracefully when the embedding model is not available.
"""
import logging
import os
import importlib.util
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model (singleton)
_model = None
_model_load_attempted = False
_sentence_transformers_available: Optional[bool] = None

# Precomputed exemplar embeddings (computed once at first use)
_hard_embeddings: Optional[np.ndarray] = None
_easy_embeddings: Optional[np.ndarray] = None

# Visual complexity exemplar embeddings (text descriptions of hard/easy visual tasks)
_hard_visual_embeddings: Optional[np.ndarray] = None
_easy_visual_embeddings: Optional[np.ndarray] = None

# ─────────────────────────────────────────────────────────────────────
# EXEMPLAR QUERIES — SEMANTIC (TEXT)
# ─────────────────────────────────────────────────────────────────────

HARD_EXEMPLARS = [
    # Complex reasoning
    "Analyze the trade-offs between microservices and monolithic architecture for a fintech startup processing 10M transactions per day",
    "Compare the legal implications of GDPR vs CCPA for a SaaS company with EU and US customers",
    "Explain the mathematical proof behind the transformer attention mechanism including the softmax gradient",
    "Design a distributed consensus algorithm that handles Byzantine faults in a 5-node cluster",
    "Write a comprehensive security audit plan for a healthcare application handling PHI data",
    # Complex code
    "Implement a lock-free concurrent hash map in Rust with support for resize operations",
    "Build a custom query optimizer for a SQL engine that handles joins across 4+ tables with index hints",
    "Write a CUDA kernel for batched matrix multiplication with shared memory tiling",
    "Implement a garbage collector using the tri-color mark-and-sweep algorithm in C",
    "Design and implement a rate limiter using the token bucket algorithm with Redis and Lua scripting",
    # Multi-step planning
    "Create a detailed migration plan to move a 500GB PostgreSQL database from on-premise to AWS Aurora with zero downtime",
    "Design the system architecture for a real-time multiplayer game server handling 100k concurrent connections",
    "Plan the data pipeline architecture for ingesting, processing, and serving ML features at 1M events per second",
    # Domain-specific hard
    "Calculate the Black-Scholes option pricing for a European call with dividend yield adjustments",
    "Explain the pharmacokinetics of warfarin-amiodarone interaction and dosing implications",
    "Draft a software licensing agreement that covers SaaS, on-premise, and hybrid deployment models",
    # Creative but complex
    "Write a technical blog post explaining how RLHF works, comparing PPO vs DPO approaches with code examples",
    "Create a detailed API design document for a payment processing system following OpenAPI 3.1 spec",
]

EASY_EXEMPLARS = [
    # Simple factual
    "What is Python?",
    "Who created JavaScript?",
    "What does HTTP stand for?",
    "How many bits in a byte?",
    "What is the capital of France?",
    # Simple code
    "Write a hello world program in Python",
    "How do I create a list in JavaScript?",
    "What is the syntax for a for loop in Java?",
    "Fix this typo in my variable name",
    "How do I print to console in Go?",
    # Simple tasks
    "Translate 'hello' to Spanish",
    "Convert this temperature from Celsius to Fahrenheit: 100",
    "What's 2 + 2?",
    "Summarize this sentence in one word",
    "Reformat this date from MM/DD/YYYY to YYYY-MM-DD",
    # Casual chat
    "Hi, how are you?",
    "Tell me a joke",
    "What should I have for lunch?",
    "Thanks for your help!",
    "Good morning",
]

# ─────────────────────────────────────────────────────────────────────
# EXEMPLAR QUERIES — VISUAL (multimodal task descriptions)
# Text descriptions of complex vs simple visual tasks, used to gauge
# whether an image-containing request needs advanced vision reasoning.
# ─────────────────────────────────────────────────────────────────────

HARD_VISUAL_EXEMPLARS = [
    "Analyze this medical X-ray image and identify any abnormalities",
    "Extract all text from this handwritten document using OCR",
    "Compare these two architectural blueprints and list the differences",
    "Identify all objects in this satellite image and their spatial relationships",
    "Read this complex chart and explain the trends with statistical analysis",
    "Debug this code screenshot and identify the syntax errors",
    "Analyze this circuit diagram and explain the signal flow",
    "Extract the data from this multi-page financial statement table",
    "Describe the UI/UX issues in this application screenshot with improvement suggestions",
    "Analyze this scientific graph showing experimental results and interpret the findings",
]

EASY_VISUAL_EXEMPLARS = [
    "What is in this image?",
    "Describe this photo",
    "What color is the object in the picture?",
    "Is there a cat in this image?",
    "How many people are in this photo?",
    "What does this logo look like?",
    "Read the text on this sign",
    "What food is shown in this picture?",
]


def _load_model():
    """Lazy-load the sentence-transformers model."""
    global _model, _model_load_attempted, _sentence_transformers_available
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    if os.getenv("PRISMO_DISABLE_EMBEDDINGS", "false").lower() == "true":
        logger.info("Contrastive complexity disabled via PRISMO_DISABLE_EMBEDDINGS")
        return None

    # Keep local embedding classifiers opt-in; otherwise the first request
    # after deploy pays model load time in the routing path.
    if os.getenv("PRISMO_ENABLE_LOCAL_EMBEDDINGS", "false").lower() != "true":
        logger.info("Contrastive complexity disabled: PRISMO_ENABLE_LOCAL_EMBEDDINGS not enabled")
        return None

    if _sentence_transformers_available is None:
        _sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
    if not _sentence_transformers_available:
        logger.info("Contrastive complexity disabled: sentence-transformers not installed")
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("PRISMO_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        return _model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Contrastive complexity detection disabled. "
            "Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        return None


def _ensure_exemplar_embeddings():
    """Precompute exemplar embeddings on first use (both text and visual)."""
    global _hard_embeddings, _easy_embeddings
    global _hard_visual_embeddings, _easy_visual_embeddings

    if _hard_embeddings is not None:
        return True

    model = _load_model()
    if model is None:
        return False

    try:
        # Text exemplars
        _hard_embeddings = model.encode(HARD_EXEMPLARS, normalize_embeddings=True)
        _easy_embeddings = model.encode(EASY_EXEMPLARS, normalize_embeddings=True)

        # Visual task exemplars (text descriptions of visual tasks)
        _hard_visual_embeddings = model.encode(HARD_VISUAL_EXEMPLARS, normalize_embeddings=True)
        _easy_visual_embeddings = model.encode(EASY_VISUAL_EXEMPLARS, normalize_embeddings=True)

        logger.info(
            f"Precomputed exemplar embeddings: "
            f"{len(HARD_EXEMPLARS)} hard text, {len(EASY_EXEMPLARS)} easy text, "
            f"{len(HARD_VISUAL_EXEMPLARS)} hard visual, {len(EASY_VISUAL_EXEMPLARS)} easy visual"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to compute exemplar embeddings: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC SIGNAL (d_sem) — Text embedding contrastive comparison
# ─────────────────────────────────────────────────────────────────────

def compute_difficulty_signal(text: str) -> Optional[float]:
    """
    Compute semantic (text) difficulty signal for a query.

    Returns:
        Float in range roughly [-1.0, 1.0]:
        - Positive values -> hard query
        - Negative values -> easy query
        - Near zero -> ambiguous/medium
        - None -> embeddings unavailable
    """
    if not _ensure_exemplar_embeddings():
        return None

    model = _load_model()
    if model is None:
        return None

    try:
        query_embedding = model.encode([text], normalize_embeddings=True)

        # Cosine similarity (dot product since embeddings are normalized)
        hard_sims = np.dot(query_embedding, _hard_embeddings.T).flatten()
        easy_sims = np.dot(query_embedding, _easy_embeddings.T).flatten()

        difficulty = float(np.max(hard_sims) - np.max(easy_sims))
        return difficulty
    except Exception as e:
        logger.warning(f"Semantic difficulty signal computation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# VISUAL SIGNAL (d_vis) — Image-aware complexity
# ─────────────────────────────────────────────────────────────────────

def compute_visual_difficulty_signal(
    text: str,
    image_count: int = 0,
) -> Optional[float]:
    """
    Compute visual difficulty signal for a multimodal request.

    Uses text descriptions of visual tasks as proxy embeddings (since we
    can't embed actual images in the routing path without CLIP). The text
    accompanying images is compared against visual task exemplars.

    Also applies heuristic boosts:
      - Multiple images → +0.05 per additional image (complex comparison tasks)
      - Image count > 3 → additional +0.05 (batch processing)

    Returns:
        Float in range roughly [-1.0, 1.0], or None if no images / unavailable.
    """
    if image_count <= 0:
        return None

    if not _ensure_exemplar_embeddings():
        return None

    model = _load_model()
    if model is None:
        return None

    try:
        query_embedding = model.encode([text], normalize_embeddings=True)

        hard_sims = np.dot(query_embedding, _hard_visual_embeddings.T).flatten()
        easy_sims = np.dot(query_embedding, _easy_visual_embeddings.T).flatten()

        visual_difficulty = float(np.max(hard_sims) - np.max(easy_sims))

        # Heuristic boosts for multi-image requests
        if image_count > 1:
            visual_difficulty += 0.05 * (image_count - 1)
        if image_count > 3:
            visual_difficulty += 0.05

        return visual_difficulty
    except Exception as e:
        logger.warning(f"Visual difficulty signal computation failed: {e}")
        return None


def count_images_in_messages(messages: Optional[list]) -> int:
    """
    Count the number of image content parts in a messages array.
    Handles both OpenAI Chat Completions format and Responses API.
    """
    if not messages:
        return 0

    count = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url" or part.get("type") == "image":
                        count += 1
    return count


# ─────────────────────────────────────────────────────────────────────
# DUAL-SIGNAL COMBINATION
# ─────────────────────────────────────────────────────────────────────

def compute_combined_difficulty(
    text: str,
    image_count: int = 0,
) -> Optional[float]:
    """
    Compute dual-signal difficulty: max(abs(d_vis), abs(d_sem)).

    Takes the stronger signal between text and visual complexity,
    preserving the sign of the dominant signal. This ensures that a
    simple text prompt paired with complex image analysis still routes
    to a capable model.

    Returns:
        Combined difficulty signal, or None if embeddings unavailable.
    """
    d_sem = compute_difficulty_signal(text)
    if d_sem is None:
        return None

    if image_count <= 0:
        return d_sem

    d_vis = compute_visual_difficulty_signal(text, image_count)
    if d_vis is None:
        return d_sem

    # Take the stronger signal
    if abs(d_vis) > abs(d_sem):
        return d_vis
    return d_sem


# ─────────────────────────────────────────────────────────────────────
# SCORE MAPPING
# ─────────────────────────────────────────────────────────────────────

def difficulty_to_score(difficulty_signal: Optional[float]) -> Optional[float]:
    """
    Map contrastive difficulty signal to a 0.0-1.0 complexity score.

    Mapping:
        difficulty <= -0.15 -> 0.1 (easy)
        difficulty in [-0.15, -0.05] -> 0.1-0.3 (simple)
        difficulty in [-0.05, 0.05] -> 0.3-0.7 (medium)
        difficulty in [0.05, 0.15] -> 0.7-0.9 (complex)
        difficulty >= 0.15 -> 0.9 (hard)
    """
    if difficulty_signal is None:
        return None

    if difficulty_signal <= -0.15:
        return 0.1
    elif difficulty_signal <= -0.05:
        t = (difficulty_signal + 0.15) / 0.10
        return 0.1 + t * 0.2
    elif difficulty_signal <= 0.05:
        t = (difficulty_signal + 0.05) / 0.10
        return 0.3 + t * 0.4
    elif difficulty_signal <= 0.15:
        t = (difficulty_signal - 0.05) / 0.10
        return 0.7 + t * 0.2
    else:
        return 0.9
