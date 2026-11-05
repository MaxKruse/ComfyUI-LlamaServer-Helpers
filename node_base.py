"""Shared building blocks for the node layer.

Defines the custom socket type used to thread the :class:`~server.LLMServer`
connection from the Load node through every consumer node, plus the small path
validators and image-collector helpers the nodes share.

This module imports ``comfy_api`` (a ComfyUI dependency) and so is only
importable inside the ComfyUI runtime (or with it on the path in tests) - the
pure-logic modules (server/client/enhance/judge) deliberately stay free of it.
"""

from __future__ import annotations

import os
from pathlib import Path

from comfy_api.latest import io

# The socket type that carries the LLMServer connection between nodes. Every
# consumer node takes this as an input and re-emits it as an output, so a server
# can be threaded through a chain (Load -> Enhance -> Rules -> Close). Threading
# the connection is what makes Close run *after* the consumers; Close and the
# image gate are additionally declared output nodes (is_output_node=True) so the
# executor actually schedules them - ComfyUI only runs output nodes and their
# upstream dependencies, and a connection-only sink with no outputs would
# otherwise never be reached.
LLMServerType = io.Custom("LLMSERVER")


def resolve_server_path(raw: str, fallback: str = "llama-server") -> str:
    """Resolve the llama-server binary path.

    A bare command name (no path separators) is returned as-is so it is looked
    up on PATH. Otherwise it must be an absolute path. Raises with a clear
    message on a relative path.
    """
    path = (raw or "").strip()
    if not path:
        return fallback
    if "/" not in path and "\\" not in path:
        return path
    if not os.path.isabs(path):
        raise ValueError(f"llama_server_path must be an absolute path, got: {path!r}")
    return path


def require_model_path(raw: str) -> str:
    """Require an absolute, existing GGUF path. Returns the resolved path."""
    path = (raw or "").strip()
    if not path:
        raise ValueError(
            "model_path is required: set the absolute path to your .gguf model file."
        )
    if not os.path.isabs(path):
        raise ValueError(f"model_path must be an absolute path, got: {path!r}")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"model_path does not exist: {path}")
    return str(resolved)


def optional_absolute_path(raw: str, name: str) -> str:
    """Validate an optional absolute path; returns "" when empty."""
    path = (raw or "").strip()
    if not path:
        return ""
    if not os.path.isabs(path):
        raise ValueError(f"{name} must be an absolute path, got: {path!r}")
    return path


def collect_images(ref_images: dict | None) -> list:
    """Collect connected reference images from an Autogrow dict, in order.

    The dict keys are the prefixed socket ids (e.g. ``ref_image_0``); sorting
    them gives a stable order. Unconnected (None) entries are skipped.
    """
    if not ref_images:
        return []
    images = []
    for name in sorted(ref_images.keys()):
        img = ref_images[name]
        if img is not None:
            images.append(img)
    return images
