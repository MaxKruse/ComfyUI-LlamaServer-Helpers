"""Rules-checker logic (connection-based).

Sends a free-form set of RULES and a TEXT and/or image(s) to the model and
asks for a strict JSON verdict - whether the input complies with every rule,
and why not. The node layer turns that verdict into a passthrough plus a
reported judgement; aborting the workflow is the Close node's job, not the
checker's.

ComfyUI-free at module scope (interrupt check is lazy) for testability.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

try:
    from .client import chat_completion_interruptible
    from .enhance import _json_loads_lenient
    from .server import LLMServer, _print_safe, check_interrupt
except ImportError:
    from client import chat_completion_interruptible
    from enhance import _json_loads_lenient
    from server import LLMServer, _print_safe, check_interrupt

logger = logging.getLogger(__name__)

_LOG_TAG = "LlamaServerHelpers"


class CheckResult(NamedTuple):
    """Verdict from the rules checker.

    ``ok`` is False when the model never produced a parseable verdict (server
    error, empty response, malformed JSON on every attempt) - that is an
    infrastructure failure, distinct from the model judging the text non-compliant.
    """

    passed: bool
    reason: str
    ok: bool = True


def _judge_system_prompt(rules: str) -> str:
    """System prompt: the user's rules plus the strict JSON verdict contract."""
    return (
        "You are a strict rules checker. You are given a set of RULES and a TEXT. "
        "Decide whether the TEXT complies with ALL of the rules.\n"
        "\n"
        f"RULES:\n{rules.strip()}\n"
        "\n"
        "Respond with a single JSON object and nothing else, with exactly these keys:\n"
        "{\n"
        '  "pass": true,\n'
        '  "reason": ""\n'
        "}\n"
        '- "pass" is true only if the TEXT satisfies every rule; otherwise false.\n'
        '- "reason" is an empty string when pass is true. When pass is false, it is a '
        "short, specific explanation of which rule(s) the TEXT violates and why - "
        "plain text, no markdown, no labels."
    )


def _judge_image_system_prompt(rules: str) -> str:
    """System prompt for the vision path: the user's rules plus the strict JSON
    verdict contract, applied to image(s) instead of text."""
    return (
        "You are a strict rules checker. You are given a set of RULES and one or more IMAGES. "
        "Decide whether the IMAGES comply with ALL of the rules.\n"
        "\n"
        f"RULES:\n{rules.strip()}\n"
        "\n"
        "Respond with a single JSON object and nothing else, with exactly these keys:\n"
        "{\n"
        '  "pass": true,\n'
        '  "reason": ""\n'
        "}\n"
        '- "pass" is true only if the IMAGES satisfy every rule; otherwise false.\n'
        '- "reason" is an empty string when pass is true. When pass is false, it is a '
        "short, specific explanation of which rule(s) the image(s) violate and what you see "
        "in the image(s) that violates them - plain text, no markdown, no labels."
    )


def _coerce_bool(value) -> bool | None:
    """Coerce a loose JSON value to a real bool, or None when it is not one."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _parse_check_response(result: str | None) -> CheckResult:
    """Parse a JSON verdict into a CheckResult. ok=False when not parseable."""
    if not result:
        return CheckResult(False, "", ok=False)
    payload = _json_loads_lenient(result)
    if not isinstance(payload, dict) or "pass" not in payload:
        return CheckResult(False, "", ok=False)
    passed = _coerce_bool(payload.get("pass"))
    if passed is None:
        return CheckResult(False, "", ok=False)
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    if passed:
        reason = ""
    return CheckResult(passed=passed, reason=reason, ok=True)


def check_rules(
    server: LLMServer,
    text: str,
    rules: str,
    *,
    seed: int = 0,
    temperature: float | None = None,
    max_retries: int = 3,
    images=None,
) -> CheckResult:
    """Judge ``text`` and/or ``images`` against ``rules`` using the server.

    When ``images`` (a list of ComfyUI IMAGE tensors) is given, a vision system
    prompt is used and the images are sent to a multimodal model; any ``text``
    then travels with the request as related context (e.g. a caption to check
    against the image). Retries (with a fresh seed) while the model returns a
    non-parseable verdict. Returns a :class:`CheckResult` with ``ok=False`` only
    when no attempt produced a parseable verdict.
    """
    has_text = bool(text and text.strip())
    has_images = bool(images)
    if not has_text and not has_images:
        return CheckResult(True, "", ok=True)
    check_interrupt()

    import random

    base_seed = (seed if seed >= 0 else random.getrandbits(32)) % 2**32
    if has_images:
        system_prompt = _judge_image_system_prompt(rules)
        user_message = "Check the attached image(s) against the rules above."
        if has_text:
            user_message += (
                "\n\nRelated text (part of what is being checked):\n" + text.strip()
            )
    else:
        system_prompt = _judge_system_prompt(rules)
        user_message = f"Check this TEXT:\n\n{text.strip()}"

    for attempt in range(1, max_retries + 1):
        check_interrupt()
        attempt_seed = (base_seed + attempt - 1) % 2**32
        _print_safe(
            f"  [{_LOG_TAG}] Rules check attempt {attempt}/{max_retries} (seed {attempt_seed})..."
        )
        result = chat_completion_interruptible(
            server,
            system_prompt,
            user_message,
            attempt_seed,
            images=images,
            temperature=temperature,
        )
        verdict = _parse_check_response(result)
        if verdict.ok:
            if verdict.passed:
                _print_safe(f"  [{_LOG_TAG}] Rules check PASSED on attempt {attempt}")
            else:
                _print_safe(
                    f"  [{_LOG_TAG}] Rules check FAILED on attempt {attempt}: {verdict.reason[:160]}"
                )
            return verdict
        _print_safe(
            f"  [{_LOG_TAG}] Rules check attempt {attempt} returned no valid verdict"
        )

    logger.error(
        "[%s] Rules check produced no valid verdict after %d attempts",
        _LOG_TAG,
        max_retries,
    )
    return CheckResult(
        False, "the rules checker did not return a valid verdict", ok=False
    )
