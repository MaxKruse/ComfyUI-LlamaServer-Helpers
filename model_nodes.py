"""Model-specific enhancer nodes and the image rules checker node.

Every generation model wants its prompt in a different dialect, so instead of
one generic enhancer with a one-size-fits-all system prompt, the factory below
builds one node class per :class:`~presets.EnhancerSpec` - same input/output
shape as :class:`nodes.PromptEnhancer`, but the system prompt is pre-filled
with the model's documented dialect (and the quality-gate / negative-prompt
defaults match that dialect). Users can still edit the system prompt on any of
them.

Also defines :class:`ImageRulesChecker`, the vision counterpart of
:class:`nodes.RulesChecker`: it judges generated image(s) against free-form
rules with a multimodal model and gates the queue on the verdict.

This module imports ``comfy_api`` and is only importable inside the ComfyUI
runtime (or with it on the path in tests).
"""

from __future__ import annotations

from comfy_api.latest import io

try:
    from .enhance import enhance_prompt
    from .judge import check_rules
    from .node_base import LLMServerType, collect_images
    from .presets import ENHANCER_SPECS, EnhancerSpec
except ImportError:
    from enhance import enhance_prompt
    from judge import check_rules
    from node_base import LLMServerType, collect_images
    from presets import ENHANCER_SPECS, EnhancerSpec


def _build_enhancer_node(spec: EnhancerSpec) -> type[io.ComfyNode]:
    """Build one enhancer node class for a model dialect spec.

    The class is structurally identical to :class:`nodes.PromptEnhancer`; only
    the schema defaults (system prompt, min_words, negative-prompt default,
    display/description, category) come from the spec.
    """

    def define_schema(_cls) -> io.Schema:
        return io.Schema(
            node_id=spec.node_id,
            display_name=spec.display_name,
            category=spec.category,
            description=spec.description,
            inputs=[
                LLMServerType.Input(
                    "llm_server", tooltip="The server connection from Load LLM Server."
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="A girl in a cyberpunk city...",
                    tooltip="The prompt to enhance. Leave empty to bypass and pass it through unchanged.",
                ),
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    default=spec.system_prompt,
                    multiline=True,
                    tooltip="Enhancement instructions, pre-filled with this model's documented prompt dialect. Edit to fine-tune it.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=-1,
                    max=2**31,
                    step=1,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                    tooltip="Sampling seed. A negative value picks a random seed per run.",
                ),
                io.Float.Input(
                    "temperature",
                    optional=True,
                    default=None,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sampling temperature. Leave unconnected to use the server's default.",
                ),
                io.Int.Input(
                    "max_retries",
                    default=3,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="How many attempts to make before accepting a result, if the quality gate rejects a draft.",
                ),
                io.Int.Input(
                    "min_words",
                    default=spec.default_min_words,
                    min=1,
                    max=500,
                    step=1,
                    tooltip="Minimum word count for an enhanced prompt to be accepted by the quality gate.",
                ),
                io.Boolean.Input(
                    "generate_negative_prompt",
                    optional=True,
                    default=spec.default_generate_negative,
                    tooltip="Also ask the model for a negative prompt of traits to avoid.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                    tooltip="Optional reference images (labelled <Picture N>) for a multimodal model.",
                ),
            ],
            outputs=[
                io.String.Output("enhanced_prompt", tooltip="The enhanced prompt."),
                io.String.Output(
                    "negative_prompt",
                    tooltip="The negative prompt (empty if not generated).",
                ),
                LLMServerType.Output(
                    "llm_server",
                    tooltip="The connection, passed through to the next node.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    def execute(_cls, **inputs):
        server = inputs["llm_server"]
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return io.NodeOutput(prompt, "", server)

        ref_images = collect_images(inputs.get("ref_images"))
        result = enhance_prompt(
            server,
            prompt,
            (inputs.get("system_prompt") or spec.system_prompt).strip(),
            seed=int(inputs.get("seed", 0)),
            temperature=inputs.get("temperature"),
            max_retries=int(inputs.get("max_retries", 3)),
            min_words=int(inputs.get("min_words", spec.default_min_words)),
            images=ref_images or None,
            generate_negative=bool(
                inputs.get("generate_negative_prompt", spec.default_generate_negative)
            ),
        )
        if result.positive:
            return io.NodeOutput(result.positive, result.negative or "", server)
        # Degraded: the model could not produce an acceptable enhancement.
        # Pass the original prompt through so the workflow still completes.
        return io.NodeOutput(prompt, "", server)

    return type(
        spec.node_id,
        (io.ComfyNode,),
        {
            "__doc__": (
                f"Enhance a prompt for the model behind {spec.display_name} using the "
                "LLM. The system prompt is pre-filled with this model's documented "
                "prompt dialect; on a failed enhancement the original prompt passes "
                "through unchanged."
            ),
            "define_schema": classmethod(define_schema),
            "execute": classmethod(execute),
        },
    )


# One node class per dialect spec, keyed by node id.
MODEL_ENHANCER_NODES = {
    spec.node_id: _build_enhancer_node(spec) for spec in ENHANCER_SPECS
}


class ImageRulesChecker(io.ComfyNode):
    """Judge generated image(s) against free-form rules using a multimodal LLM.

    The vision counterpart of :class:`nodes.RulesChecker`. You write the rules in
    plain language ("the character has no watermark", "exactly one person"); the
    model looks at the image(s) and decides whether they comply. Wire the
    VAE-decoded image(s) in and the gate runs before the image is saved. On a
    failure the node either aborts the workflow (default) or reports the model's
    judgement while the workflow continues. Load the server with an ``mmproj``
    path (and a vision-capable model) for this node to work.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageRulesChecker",
            display_name="Image Rules Checker (LLM Server)",
            category="LlamaServer/LLM",
            # An output node so the gate is actually scheduled: it is a pure
            # gate whose outputs (judgement, passthrough connection) are often
            # left unconnected, and the executor only runs output nodes (or
            # their upstream dependencies) - without this an unconnected gate
            # would silently never run.
            is_output_node=True,
            description="Check generated image(s) against free-form rules with a multimodal LLM. Abort the workflow on failure, or report the judgement.",
            inputs=[
                LLMServerType.Input(
                    "llm_server", tooltip="The server connection from Load LLM Server."
                ),
                io.String.Input(
                    "rules",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="The image must show exactly one person, with no text or watermark.",
                    tooltip="Free-form rules the image(s) must satisfy, in plain language. Leave empty to bypass.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=9,
                    ),
                    tooltip="The generated image(s) to check. Leave all unconnected to bypass.",
                ),
                io.Boolean.Input(
                    "abort_on_fail",
                    optional=True,
                    default=True,
                    tooltip="When true, a failed check aborts the workflow. When false, the workflow continues and the judgement is reported on the 'judgement' output.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=-1,
                    max=2**31,
                    step=1,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                    tooltip="Sampling seed. A negative value picks a random seed per run.",
                ),
                io.Float.Input(
                    "temperature",
                    optional=True,
                    default=None,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sampling temperature. Leave unconnected to use the server's default.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "judgement",
                    tooltip="Empty on pass (and on bypass). The model's reason when the check fails and abort_on_fail is false.",
                ),
                LLMServerType.Output(
                    "llm_server",
                    tooltip="The connection, passed through to the next node.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, **inputs):
        server = inputs["llm_server"]
        rules = (inputs.get("rules") or "").strip()
        images = collect_images(inputs.get("images"))
        if not rules or not images:
            return io.NodeOutput("", server)

        verdict = check_rules(
            server,
            "",
            rules,
            seed=int(inputs.get("seed", 0)),
            temperature=inputs.get("temperature"),
            images=images,
        )
        if not verdict.ok:
            raise RuntimeError(
                f"Image rules check could not obtain a verdict from the LLM: {verdict.reason}. "
                "Make sure the server was loaded with a vision model and an mmproj path."
            )
        if verdict.passed:
            return io.NodeOutput("", server)
        if bool(inputs.get("abort_on_fail", True)):
            raise RuntimeError(f"Image rules check failed: {verdict.reason}")
        return io.NodeOutput(verdict.reason, server)

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        # Always re-run, never serve a cached no-op: the gate must evaluate on
        # every execution (NaN never equals itself, so the output cache always
        # misses).
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ImageRulesChecker": ImageRulesChecker,
    **MODEL_ENHANCER_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRulesChecker": "Image Rules Checker (LLM Server)",
    **{spec.node_id: spec.display_name for spec in ENHANCER_SPECS},
}
