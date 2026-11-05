"""Prompt enhancement logic (connection-based).

Unlike a target-specific enhancer, this keeps the behaviour user-controlled:
the *what* comes from a free-form ``system_prompt`` supplied by the node, and
this module only owns the mechanical parts - the JSON response contract, a
lightweight quality gate (word floor + refusal/echo detection) and a retry
loop so a short first draft is not returned.

ComfyUI-free at module scope (interrupt check is lazy) for testability.
"""

from __future__ import annotations

import json
import logging
import random
from typing import NamedTuple

try:
    from .client import chat_completion_interruptible
    from .server import LLMServer, _print_safe, check_interrupt
except ImportError:
    from client import chat_completion_interruptible
    from server import LLMServer, _print_safe, check_interrupt

logger = logging.getLogger(__name__)

_LOG_TAG = "LlamaServerHelpers"

# Phrases that indicate the model refused or deflected instead of enhancing.
REFUSAL_PATTERNS = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "i apologize",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "as a language model",
    "i'm designed to",
    "i must decline",
    "i have to decline",
    "not appropriate",
    "against my guidelines",
    "goes against my",
    "i cannot assist",
    "unable to help with",
)


class EnhanceResult(NamedTuple):
    """Outcome of one enhance run: the enhanced prompt and, when requested, the
    negative prompt from the same turn (None when the turn failed)."""

    positive: str | None
    negative: str | None


def _contains_refusal(result: str) -> bool:
    result_lower = result.lower()
    return any(pattern in result_lower for pattern in REFUSAL_PATTERNS)


def is_good_prompt(result: str, original: str, min_words: int = 25) -> bool:
    """Validate an enhanced prompt: enough words, not a refusal, not an echo."""
    if not result:
        return False
    if len(result.split()) < min_words:
        return False
    if _contains_refusal(result):
        return False
    if result.strip() == original.strip():
        return False
    # Jaccard word overlap > 0.85 means it is basically the original echoed back.
    # Measuring the union (not just retention) so a faithful expansion that keeps
    # every original word still scores low and passes.
    orig_words = set(original.lower().split())
    result_words = set(result.lower().split())
    if orig_words and result_words:
        overlap = len(orig_words & result_words) / len(orig_words | result_words)
        if overlap > 0.85:
            return False
    return True


def _json_loads_lenient(text: str) -> object | None:
    """Parse JSON from a model response, tolerating markdown code fences."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate[:4].lower() == "json":
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None


def _enhance_json_instruction(generate_negative: bool) -> str:
    """Build the single-turn JSON response contract appended to the system prompt."""
    lines = [
        "",
        "",
        "Respond with a single JSON object and nothing else, with exactly these keys:",
        "{",
    ]
    if generate_negative:
        lines.append('  "positive": "<enhanced prompt>",')
        lines.append('  "negative": "<negative prompt>"')
    else:
        lines.append('  "positive": "<enhanced prompt>"')
    lines.append("}")
    lines.append(
        '- "positive" is the enhanced prompt text itself - plain text, no markdown, no labels, no code fences.'
    )
    if generate_negative:
        lines.extend(
            [
                '- "negative" is a short list of traits the generation model should avoid in the output, '
                "derived from the prompt, or an empty string when nothing applies.",
                "- Phrase each negative item as an undesirable trait; do not negate the positive wording, "
                "no explanations, no labels, no markdown.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_enhancement_response(
    result: str | None, generate_negative: bool
) -> tuple[str | None, str | None]:
    """Split a JSON enhancement response into (positive, negative).

    The negative member is only read when it was requested; a refusal in either
    member is discarded (refusals are never returned). Returns (None, None) for a
    missing or unparseable response.
    """
    if not result:
        return None, None
    payload = _json_loads_lenient(result)
    if not isinstance(payload, dict):
        return None, None
    positive = payload.get("positive")
    positive = positive.strip() if isinstance(positive, str) else None
    negative = None
    if generate_negative:
        raw_negative = payload.get("negative")
        raw_negative = raw_negative.strip() if isinstance(raw_negative, str) else None
        if raw_negative and not _contains_refusal(raw_negative):
            negative = raw_negative
    return positive or None, negative


def _resolve_base_seed(seed: int) -> int:
    """A negative seed means a random 32-bit base per run."""
    return (seed if seed >= 0 else random.getrandbits(32)) % 2**32


def enhance_prompt(
    server: LLMServer,
    user_prompt: str,
    system_prompt: str,
    *,
    seed: int = 0,
    temperature: float | None = None,
    max_retries: int = 3,
    min_words: int = 25,
    images=None,
    generate_negative: bool = False,
) -> EnhanceResult:
    """Enhance a prompt using the given server connection.

    The behaviour is driven by ``system_prompt`` (supplied by the node); this
    function appends the JSON response contract, retries until the quality gate
    passes, and falls back to the longest clean candidate when no attempt passes.

    Returns an :class:`EnhanceResult`; on total failure both members are None.
    """
    check_interrupt(server)
    base_seed = _resolve_base_seed(seed)
    effective_system = (system_prompt or "").rstrip()
    if effective_system:
        effective_system += "\n"
    effective_system += _enhance_json_instruction(generate_negative)

    best: EnhanceResult | None = None
    for attempt in range(1, max_retries + 1):
        check_interrupt(server)
        attempt_seed = (base_seed + attempt - 1) % 2**32
        _print_safe(
            f"  [{_LOG_TAG}] Enhance attempt {attempt}/{max_retries} (seed {attempt_seed})..."
        )
        result = chat_completion_interruptible(
            server,
            effective_system,
            user_prompt,
            attempt_seed,
            images=images,
            temperature=temperature,
        )
        positive, negative = _parse_enhancement_response(result, generate_negative)
        if positive and is_good_prompt(positive, user_prompt, min_words):
            _print_safe(
                f"  [{_LOG_TAG}] Enhance accepted on attempt {attempt} ({len(positive)} chars)"
            )
            return EnhanceResult(positive, negative if generate_negative else None)
        if positive and not _contains_refusal(positive):
            candidate = EnhanceResult(positive, negative if generate_negative else None)
            if best is None or len(positive) > len(best.positive):
                best = candidate
        else:
            _print_safe(
                f"  [{_LOG_TAG}] Enhance attempt {attempt} rejected (refusal/empty/unparseable)"
            )

    if best is not None:
        _print_safe(
            f"  [{_LOG_TAG}] No attempt passed the quality gate; returning best candidate ({len(best.positive)} chars)"
        )
        return best
    logger.error("[%s] All %d enhance attempts failed", _LOG_TAG, max_retries)
    return EnhanceResult(None, None)
