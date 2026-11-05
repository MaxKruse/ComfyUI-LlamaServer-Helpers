"""HTTP client that talks to a running llama-server via the OpenAI-compatible API.

Every function takes the :class:`~server.LLMServer` connection and uses its
``base_url`` + ``model_name``. No server lifecycle lives here - that is
:mod:`server`'s job. This module stays ComfyUI-free at module scope (the
interrupt check is lazy) so it can be unit-tested without the runtime.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import socket
from urllib import error, request

try:
    from .server import LLMServer, _print_safe, check_interrupt
except ImportError:
    from server import LLMServer, _print_safe, check_interrupt

logger = logging.getLogger(__name__)

# Per-request timeout. Generous because a long generation can take a while.
_REQUEST_TIMEOUT = 300

_LOG_TAG = "LlamaServerHelpers"


def _log_timings(result: dict):
    """Print llama-server inference speed and context usage from a response."""
    timings = result.get("timings") or {}
    if not timings:
        return
    parts = []
    if timings.get("prompt_per_second") is not None:
        parts.append(
            f"prompt {timings.get('prompt_n', 0)} tok @ {timings['prompt_per_second']:.1f} tok/s"
        )
    if timings.get("predicted_per_second") is not None:
        parts.append(
            f"generated {timings.get('predicted_n', 0)} tok @ {timings['predicted_per_second']:.1f} tok/s"
        )
    ctx_total = (
        (timings.get("prompt_n", 0) or 0)
        + (timings.get("cache_n", 0) or 0)
        + (timings.get("predicted_n", 0) or 0)
    )
    if ctx_total:
        parts.append(f"ctx used {ctx_total} tok")
    if parts:
        _print_safe(f"  [{_LOG_TAG}] llama-server: {' | '.join(parts)}")


def _is_context_error(body: str) -> bool:
    """Whether a server error body indicates the context window was exceeded."""
    lowered = body.lower()
    return "out of context" in lowered or ("exceed" in lowered and "context" in lowered)


def _tensor_to_base64_jpeg(image_tensor) -> str:
    """Convert a ComfyUI IMAGE tensor to a base64-encoded JPEG string.

    Accepts a single image [H, W, 3] in [0, 1] or a batch [N, H, W, 3]
    (the first image is used).
    """
    from PIL import Image

    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    np_img = image_tensor.clamp(0, 1).mul(255).byte().cpu().contiguous().numpy()
    pil_img = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _images_to_base64_jpegs(images) -> list[str]:
    """Encode a list of ComfyUI IMAGE tensors as base64 JPEGs."""
    if not images:
        return []
    return [_tensor_to_base64_jpeg(img) for img in images]


def build_user_content(user_prompt: str, images) -> str | list:
    """Build the user message content: plain text, or text + image parts.

    When images are provided each is labelled <Picture N> so the model can
    reference it. Returns a plain string when there are no images.
    """
    if not images:
        return user_prompt
    image_b64s = _images_to_base64_jpegs(images)
    content = [{"type": "text", "text": user_prompt}]
    for index, b64 in enumerate(image_b64s, start=1):
        content.append(
            {"type": "text", "text": f"Reference image {index} (<Picture {index}>):"}
        )
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    return content


def _payload(
    server: LLMServer,
    system_prompt: str,
    user_content,
    seed: int,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Assemble the /v1/chat/completions payload for one turn."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": server.model_name,
        "seed": seed,
        "messages": messages,
    }
    # Sampling params are only sent when the caller asks for them; otherwise
    # the server uses its own defaults / command-line flags.
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def chat_completion(
    server: LLMServer,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    *,
    images=None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Send one chat turn. Returns the assistant content, or None on failure.

    *images* is an optional list of ComfyUI IMAGE tensors; each is JPEG-encoded
    and sent alongside the text so a multimodal model can see it. Sampling
    params (``temperature``, ``max_tokens``) are sent only when not None.
    """
    content = build_user_content(user_prompt, images)
    payload = _payload(
        server,
        system_prompt,
        content,
        seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{server.base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            _log_timings(result)
            message = result["choices"][0]["message"]
            # Vision-capable llama-server returns a list of content parts for
            # multimodal models; take the text parts and join them.
            c = message.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c.strip() if c else None
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        _print_safe(f"  [{_LOG_TAG}] HTTP {exc.code} from server: {body[:500]}")
        logger.error("Chat completion failed: HTTP %s - %s", exc.code, body[:500])
        if _is_context_error(body):
            _print_safe(
                f"  [{_LOG_TAG}] OUT OF CONTEXT: system prompt + images + text exceed the context window. "
                f"Raise ctx_size on the Load node, reduce reference images, or shorten the prompt."
            )
        return None
    except (
        error.URLError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        socket.timeout,
    ) as exc:
        logger.error("Chat completion failed: %s", exc)
        return None


def chat_completion_interruptible(
    server: LLMServer,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    *,
    images=None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Run :func:`chat_completion` in a worker thread so a long generation can be
    interrupted by the user (ComfyUI interrupt) instead of blocking the queue.
    """
    box: dict = {}

    def worker():
        try:
            box["value"] = chat_completion(
                server,
                system_prompt,
                user_prompt,
                seed,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            box["value"] = None

    import threading

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=0.5)
        check_interrupt()
    t.join()
    return box.get("value")
