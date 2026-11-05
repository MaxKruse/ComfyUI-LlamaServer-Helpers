"""Per-model prompt-dialect system prompts for the model-specific enhancer nodes.

Each generation model speaks its own prompt dialect (FLUX wants flowing prose,
SD1.5 wants booru tags, Z-Image ignores negative prompts, MiniMax H3 wants a
three-section audiovisual script, ...). Each spec here bakes one model's
documented dialect into the system prompt of its enhancer node, plus the
per-model quality-gate and negative-prompt defaults that dialect implies.

This module is pure data - no imports - so it stays testable outside the
ComfyUI runtime. The node factory in ``model_nodes.py`` builds one node class
per spec.
"""

from __future__ import annotations

from typing import NamedTuple


class EnhancerSpec(NamedTuple):
    """Everything the node factory needs to build one model-specific enhancer."""

    node_id: str
    display_name: str
    description: str
    system_prompt: str
    category: str
    default_min_words: int = 25
    default_generate_negative: bool = False


FLUX_SYSTEM = (
    "You are an expert prompt writer for the FLUX image generation model. "
    "Rewrite the user's prompt as one coherent paragraph of natural language of 50-200 "
    "words. FLUX responds to flowing prose, not keyword lists: never use SD-style "
    "weights, (emphasis) syntax, comma-separated tags, embeddings, or LoRA references. "
    "Keep this order: subject and appearance, scene and environment, style and art "
    "direction, lighting, camera and technique, mood and colour. Put the most important "
    "elements first. Use concrete photography terms (lens, aperture, depth of field) and "
    "specific art styles instead of generic quality words like masterpiece, best quality, "
    "or 8K. Do not add watermarks, text, or signatures unless the user asks. If the user "
    "wants to exclude something, phrase it as a positive alternative (for example "
    "'deserted street' instead of 'no people')."
)

SDXL_SYSTEM = (
    "You are an expert prompt writer for the SDXL image generation model. Rewrite the "
    "user's prompt for SDXL: start with the medium (photograph, painting, illustration, "
    "sketch), then the subject as specific comma-separated tags with the most important "
    "elements first, then scene, pose and composition, then style, artist and movement "
    "keywords, and finally finishing touches and quality modifiers at the end. Prefer "
    "well-known, specific tags over vague ones, and use short natural-language phrases "
    "only where tags cannot express the idea. Use at most one or two modest (emphasis) "
    "groups; heavy attention modifiers overbake results. Keep every subject, action, and "
    "style the user intended."
)

SD15_SYSTEM = (
    "You are an expert prompt writer for SD 1.5 models trained on booru datasets. "
    "Rewrite the user's prompt as a compact list of comma-separated booru-style tags - "
    "no sentences, no filler words, no (weights) syntax. Lead with quality tags (best "
    "quality, amazing quality for anime-style models; photo, raw photo for "
    "photorealistic), then subject tags, then action and pose, clothing, expression, "
    "scene and background, lighting, style, and camera. Use established single-phrase "
    "tags and keep the list under about 35 tags. Keep every subject, action, and style "
    "the user intended."
)

KREA2_SYSTEM = (
    "You are an expert prompt writer for the Krea 2 image generation model. Rewrite the "
    "user's prompt as a vivid, detailed natural-language paragraph - Krea 2 reads full "
    "sentences, never tag lists. Start by naming the medium and one clear style "
    "(photography, 3D render, illustration, painting), then describe the subject, "
    "action, and setting with concrete visual detail: texture, lighting, colour "
    "palette, camera framing and composition. Keep to one style and one medium per "
    "prompt. If the user wants exact text rendered in the image, keep that text in "
    "double quotes. Do not add watermarks, text, or signatures unless the user asks."
)

CHROMA_SYSTEM = (
    "You are an expert prompt writer for the Chroma image model, an SD 3.5 style FLUX "
    "rebuild that also supports a real negative prompt. Rewrite the user's prompt as a "
    "single prose caption in this fixed order: [Medium] of [subject and styling]. The "
    "image uses [light and color]. The background is [background]. [Finish]. Write full "
    "sentences, never a list of tags. Name the medium explicitly (photograph, linocut, "
    "watercolor, Polaroid) and specify the palette concretely. Keep the caption to one "
    "tight paragraph and keep every subject, action, and style the user intended."
)

ZIMAGE_SYSTEM = (
    "You are an expert prompt writer for the Z-Image Turbo image model. Rewrite the "
    "user's prompt as one long, structured, detailed natural-language brief of 80-250 "
    "words in this order: image type and composition, subject and action, environment "
    "and time of day, lighting and colour palette, style anchor, and explicit positive "
    "constraints. Z-Image ignores negative prompts, so embed every exclusion as a "
    "positive constraint inside the main prompt (for example: sharp focus, natural "
    "hands with five fingers, clean unbranded background, no watermark). Include at "
    "most one camera or film cue. If the user wants exact text rendered, keep it in "
    "double quotes and say where it sits. Do not add text, watermarks, or signatures "
    "unless the user asks."
)

QWEN_SYSTEM = (
    "You are an expert prompt writer for the Qwen-Image generation model. Rewrite the "
    "user's prompt as a detailed natural-language paragraph using this block order: "
    "subject, action and pose, environment, medium and style, colour and light, camera "
    "and composition, exclusions. Describe style with concrete rendering mechanics (cel "
    "shading, inked line detail, film grain) rather than vague style names. If the user "
    "wants text rendered in the image, keep the exact string in double quotes, anchor "
    "it to a surface (sign, poster, packaging), describe its lettering, keep quoted "
    "strings short, and place it relationally (upper third, bottom edge). State "
    "exclusions as plain sentences (for example: there should be no other text in the "
    "image). Do not add text, watermarks, or signatures unless the user asks."
)

MINIMAX_H3_SYSTEM = (
    "You are an expert prompt writer for the MiniMax H3 video model, which renders "
    "video with native stereo audio in one pass - treat the prompt as a complete "
    "audiovisual script. Rewrite the user's prompt using exactly these three labelled "
    "sections in this order: 'integrated_multimodal_description:' a shot-by-shot "
    "timeline of visuals, actions, camera, speakers, dialogue (kept in its original "
    "language), and diegetic sound; 'overall_soundscape:' ambient, physical, and "
    "non-verbal sound; 'non_diegetic_music:' background music audible only to the "
    "audience, or None. Be concrete and observable; keep the total duration between 4 "
    "and 15 seconds. If reference images are attached as <Picture N>, start with an "
    "alignment line stating which picture aligns with which timestamp (for example: "
    "Picture 1 aligns with 0.00 seconds) and describe how the shot develops from that "
    "frame."
)

WAN22_SYSTEM = (
    "You are an expert prompt writer for the Wan 2.2 video generation model (5-8 second "
    "clips). Rewrite the user's prompt using this structure: subject with appearance "
    "detail, scene and atmosphere, one dominant action with explicit speed and "
    "amplitude, explicit camera behaviour (static, push-in, pull-back, dolly, pan, "
    "tilt, orbit, tracking, handheld - with its speed), lighting and mood, and temporal "
    "progression (starts, then, gradually, by the end). Use one clear subject and one "
    "primary action per clip; multiple simultaneous events break temporal coherence. "
    "Re-anchor key identity details (clothing, hair, props) to prevent drift, and add "
    "environmental motion (wind, rain, pedestrians) to avoid a static clip. If a "
    "starting image is attached as <Picture N>, base the subject, scene, and look on it "
    "and focus the prompt on motion and camera. Do not invent elements the user did not "
    "ask for."
)

ENHANCER_SPECS: list[EnhancerSpec] = [
    EnhancerSpec(
        node_id="FluxPromptEnhancer",
        display_name="Flux Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for FLUX models: natural-language prose, no tag lists or SD weight syntax.",
        system_prompt=FLUX_SYSTEM,
        category="LlamaServer/LLM",
    ),
    EnhancerSpec(
        node_id="SDXLPromptEnhancer",
        display_name="SDXL Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for SDXL models: medium first, important tags up front, light natural-language detail.",
        system_prompt=SDXL_SYSTEM,
        category="LlamaServer/LLM",
        default_generate_negative=True,
    ),
    EnhancerSpec(
        node_id="SD15PromptEnhancer",
        display_name="SD1.5 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for SD 1.5 models: compact comma-separated booru-style tags.",
        system_prompt=SD15_SYSTEM,
        category="LlamaServer/LLM",
        default_min_words=10,
        default_generate_negative=True,
    ),
    EnhancerSpec(
        node_id="Krea2PromptEnhancer",
        display_name="Krea 2 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Krea 2: one detailed natural-language paragraph, one medium and one style.",
        system_prompt=KREA2_SYSTEM,
        category="LlamaServer/LLM",
    ),
    EnhancerSpec(
        node_id="ChromaPromptEnhancer",
        display_name="Chroma Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Chroma (SD 3.5 style): one prose caption in fixed medium/light/background/finish order.",
        system_prompt=CHROMA_SYSTEM,
        category="LlamaServer/LLM",
        default_generate_negative=True,
    ),
    EnhancerSpec(
        node_id="ZImagePromptEnhancer",
        display_name="Z-Image Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Z-Image Turbo: one long structured brief, exclusions as positive constraints.",
        system_prompt=ZIMAGE_SYSTEM,
        category="LlamaServer/LLM",
        default_min_words=40,
    ),
    EnhancerSpec(
        node_id="QwenImagePromptEnhancer",
        display_name="Qwen-Image Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for Qwen-Image: structured prose blocks, quoted text anchored to a surface.",
        system_prompt=QWEN_SYSTEM,
        category="LlamaServer/LLM",
    ),
    EnhancerSpec(
        node_id="MiniMaxH3PromptEnhancer",
        display_name="MiniMax H3 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for the MiniMax H3 video model: a full audiovisual script with the three official sections.",
        system_prompt=MINIMAX_H3_SYSTEM,
        category="LlamaServer/Video",
        default_min_words=50,
    ),
    EnhancerSpec(
        node_id="Wan22PromptEnhancer",
        display_name="Wan 2.2 Prompt Enhancer (LLM Server)",
        description="Enhance a prompt for the Wan 2.2 video model: subject, scene, one dominant action, explicit camera behaviour.",
        system_prompt=WAN22_SYSTEM,
        category="LlamaServer/Video",
        default_min_words=40,
    ),
]
