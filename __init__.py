"""ComfyUI-LlamaServer-Helpers.

Loads a shared ``llama-server`` (llama.cpp) with the Load node and exposes the
connection to a set of LLM consumer nodes (enhancer, rules checker, generic
prompt). Importing the node layer is wrapped so that a failure here can never
prevent ComfyUI itself from starting: on any error the node mappings fall back
to empty and the exception is logged.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    logger.exception(
        "ComfyUI-LlamaServer-Helpers failed to load its nodes; continuing without them."
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
