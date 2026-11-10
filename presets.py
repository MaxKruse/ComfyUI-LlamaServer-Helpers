"""Per-model prompt-dialect system prompts for the model-specific enhancer nodes.

Each generation model speaks its own prompt dialect (FLUX wants flowing prose,
Z-Image ignores negative prompts, MiniMax H3 wants a sectioned audiovisual
script, ...). Each spec here bakes one model's
documented dialect into the system prompt of its enhancer node, plus the
per-model quality-gate and negative-prompt defaults that dialect implies.

The ``presets/`` directory next to this module is the single source of truth
for all dialect text: every spec loads its system prompt from the ``.txt``
file named in its ``preset_file`` field, at import time. Six of the thirteen
files are mirrored verbatim from the sibling ComfyUI-PromptEnhancer package -
keep those in sync with that package when its presets change. The rest are
local dialects: five compact guides for models the package does not cover,
the locally maintained MOSS-Image guide, and the MiniMax H3-Image dialect
adapted from MiniMax's released H3-Image-PE prompt assets.

The module needs only the standard library (``pathlib``) and stays testable
outside the ComfyUI runtime. The node factory in ``model_nodes.py`` builds one
node class per spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _load_preset(filename: str) -> str:
    """Load a dialect guide from the presets/ directory."""
    return (PRESETS_DIR / filename).read_text(encoding="utf-8").strip()


class EnhancerSpec(NamedTuple):
    """Everything the node factory needs to build one model-specific enhancer."""

    node_id: str
    display_name: str
    description: str
    system_prompt: str
    category: str
    preset_file: str
    default_min_words: int = 25
    default_generate_negative: bool = False


FLUX_SYSTEM = _load_preset("flux-t2i.txt")
KREA2_SYSTEM = _load_preset("krea2-t2i.txt")
ZIMAGE_SYSTEM = _load_preset("zimage-t2i.txt")
QWEN_SYSTEM = _load_preset("qwenimage-t2i.txt")
MINIMAX_H3_SYSTEM = _load_preset("minimax-h3-t2v.txt")
WAN22_SYSTEM = _load_preset("wan2.2-t2v.txt")

# Locally maintained (no longer a verbatim mirror of the source package).
MOSS_IMAGE_SYSTEM = _load_preset("moss-image-sft-t2i.txt")

# Adapted from the released MiniMax-H3-Image-PE T2I prompt assets (MiniMaxAI/
# MiniMax-H3-Image-PE: prompt/t2i system + task prompts): the user query is
# the following message instead of the "input_text" placeholder, the PE's
# two-key JSON output contract becomes this node's single-caption contract,
# and the PE's aspect-ratio policy is dropped (the canvas is sized in the
# ComfyUI workflow). Adds reference-image guidance for the model's editing /
# reference-input use and explicit NSFW permission. The H3-Image model
# rejects negative prompts, so none is generated. Locally maintained.
MINIMAX_H3_IMAGE_SYSTEM = _load_preset("minimax-h3-image-t2i.txt")

# Mirrored verbatim from ../ComfyUI-PromptEnhancer/presets/.
QWEN_IMAGE_EDIT_SYSTEM = _load_preset("qwenimage2.1-i2i.txt")
LTX23_SYSTEM = _load_preset("ltx2.3-10eros-i2v.txt")
LTX25_SYSTEM = _load_preset("ltx2.5-i2v.txt")
MINIMAX_H3_R2V_SYSTEM = _load_preset("minimax-h3-r2v.txt")
MINIMAX_H3_EROS_MAX_SYSTEM = _load_preset("minimax-h3-eros-max-r2v.txt")

ENHANCER_SPECS: list[EnhancerSpec] = [
    EnhancerSpec(
        node_id="FluxPromptEnhancer",
        display_name="Flux Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for FLUX models: natural-language prose, no tag lists or SD weight syntax.",
        system_prompt=FLUX_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="flux-t2i.txt",
    ),
    EnhancerSpec(
        node_id="Krea2PromptEnhancer",
        display_name="Krea 2 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Krea 2: one 200-300 word natural-language paragraph, subject and action first, medium last, exclusions steered with positives.",
        system_prompt=KREA2_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="krea2-t2i.txt",
        # The dialect targets 200-300 words; the floor rejects lazy short
        # expansions without failing decent-length ones.
        default_min_words=100,
    ),
    EnhancerSpec(
        node_id="ZImagePromptEnhancer",
        display_name="Z-Image Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Z-Image Turbo: one long structured brief, exclusions as positive constraints.",
        system_prompt=ZIMAGE_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="zimage-t2i.txt",
        default_min_words=40,
    ),
    EnhancerSpec(
        node_id="QwenImagePromptEnhancer",
        display_name="Qwen-Image Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Qwen-Image: structured prose blocks, quoted text anchored to a surface.",
        system_prompt=QWEN_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="qwenimage-t2i.txt",
    ),
    EnhancerSpec(
        node_id="MiniMaxH3PromptEnhancer",
        display_name="MiniMax H3 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for the MiniMax H3 video model: a full audiovisual script with the three official sections.",
        system_prompt=MINIMAX_H3_SYSTEM,
        category="LlamaServer/Video",
        preset_file="minimax-h3-t2v.txt",
        default_min_words=50,
    ),
    EnhancerSpec(
        node_id="Wan22PromptEnhancer",
        display_name="Wan 2.2 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for the Wan 2.2 video model: subject, scene, one dominant action, explicit camera behaviour.",
        system_prompt=WAN22_SYSTEM,
        category="LlamaServer/Video",
        preset_file="wan2.2-t2v.txt",
        default_min_words=40,
    ),
    EnhancerSpec(
        node_id="MossImagePromptEnhancer",
        display_name="MOSS-Image SFT Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for MOSS-Image SFT: observer-style prose caption that commits the whole frame; Layout JSON only with placement structure.",
        system_prompt=MOSS_IMAGE_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="moss-image-sft-t2i.txt",
        default_min_words=40,
    ),
    EnhancerSpec(
        node_id="QwenImageEditPromptEnhancer",
        display_name="Qwen-Image 2.1 Edit Prompt Enhancer (LLM Server)",
        description="Enhance an image-edit instruction for Qwen-Image 2.1 (i2i): precise, unambiguous editing directive anchored on the input image.",
        system_prompt=QWEN_IMAGE_EDIT_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="qwenimage2.1-i2i.txt",
    ),
    EnhancerSpec(
        node_id="MiniMaxH3ImagePromptEnhancer",
        display_name="MiniMax H3-Image Prompt Enhancer (LLM Server)",
        description=(
            "Enhance a prompt for MiniMax H3-Image (text-to-image and image "
            "editing; feed reference images on ref_images for the editing use): "
            "one detailed English caption, per the official H3-Image-PE dialect. "
            "Uncensored - NSFW content is described as requested. No negative "
            "prompt - the model rejects them."
        ),
        system_prompt=MINIMAX_H3_IMAGE_SYSTEM,
        category="LlamaServer/LLM",
        preset_file="minimax-h3-image-t2i.txt",
        # The dialect targets inspectable content, not word count (its own
        # example caption is ~25 words), so the default 25-word floor only
        # rejects lazy under-expansions.
    ),
    EnhancerSpec(
        node_id="Ltx23PromptEnhancer",
        display_name="LTX 2.3 10Eros Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for LTX 2.3 10Eros (i2v): one dense cinematic paragraph, 60-200 words, with audio and camera interwoven.",
        system_prompt=LTX23_SYSTEM,
        category="LlamaServer/Video",
        preset_file="ltx2.3-10eros-i2v.txt",
        default_min_words=60,
    ),
    EnhancerSpec(
        node_id="Ltx25PromptEnhancer",
        display_name="LTX 2.5 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for LTX 2.5 (i2v): one continuous take, prompt the delta not the image, native audio always written.",
        system_prompt=LTX25_SYSTEM,
        category="LlamaServer/Video",
        preset_file="ltx2.5-i2v.txt",
        default_min_words=40,
    ),
    EnhancerSpec(
        node_id="MiniMaxH3R2VPromptEnhancer",
        display_name="MiniMax H3 R2V Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for MiniMax H3 reference-to-video: six-section full-reference rewrite with <Subject>/<Picture>/<Video>/<Audio> labels.",
        system_prompt=MINIMAX_H3_R2V_SYSTEM,
        category="LlamaServer/Video",
        preset_file="minimax-h3-r2v.txt",
        default_min_words=50,
    ),
    EnhancerSpec(
        node_id="MiniMaxH3ErosMaxPromptEnhancer",
        display_name="MiniMax H3 Eros Max Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for MiniMax H3 Eros Max (beta5) ref2va/t2va: full-reference rewrite in literal-temporal style, NSFW focus.",
        system_prompt=MINIMAX_H3_EROS_MAX_SYSTEM,
        category="LlamaServer/Video",
        preset_file="minimax-h3-eros-max-r2v.txt",
        default_min_words=50,
    ),
]
