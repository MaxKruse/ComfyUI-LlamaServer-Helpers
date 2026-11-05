"""ComfyUI node definitions for the shared llama-server helper set.

Five nodes built around a single shared model connection:

- :class:`LoadLLMServer`   - start (or reuse a warm) llama-server, emit the connection.
- :class:`CloseLLMServer`  - tear a connection down to release VRAM; abort on a failed check.
- :class:`LLMPrompt`       - generic system + user prompt -> response building block.
- :class:`PromptEnhancer`  - enhance a prompt, with optional negative prompt.
- :class:`RulesChecker`    - judge free-form text against free-form rules.

Every LLM consumer takes the ``LLMSERVER`` connection as an input and re-emits
it as an output, so a server is threaded through a workflow (Load -> consumer(s)
-> Close). Threading the connection is what orders Close after the consumers;
Close is additionally declared an output node so the executor actually schedules
it (ComfyUI only runs output nodes and their upstream dependencies).
"""

from __future__ import annotations

from comfy_api.latest import io

try:
    from .client import chat_completion_interruptible
    from .enhance import enhance_prompt
    from .judge import check_rules
    from .node_base import (
        LLMServerType,
        collect_images,
        optional_absolute_path,
        require_model_path,
        resolve_server_path,
    )
    from .server import close_server, get_or_start_server, server_generation
except ImportError:
    from client import chat_completion_interruptible
    from enhance import enhance_prompt
    from judge import check_rules
    from node_base import (
        LLMServerType,
        collect_images,
        optional_absolute_path,
        require_model_path,
        resolve_server_path,
    )
    from server import close_server, get_or_start_server, server_generation


class LoadLLMServer(io.ComfyNode):
    """Start a llama-server for a model and hand out the connection.

    Reuses a warm server with the same configuration when one is already
    running for this session, so repeated queue items do not pay the cold-start
    cost. Connect the emitted ``llm_server`` to any LLM consumer node.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadLLMServer",
            display_name="Load LLM Server",
            category="LlamaServer",
            description="Start (or reuse a warm) llama-server and emit the connection to use with the LLM helper nodes.",
            inputs=[
                io.String.Input(
                    "model_path",
                    default="",
                    placeholder="C:/path/to/your/model.gguf",
                    tooltip="Absolute path to the .gguf model file. Required.",
                ),
                io.String.Input(
                    "llama_server_path",
                    default="llama-server",
                    tooltip="The llama-server executable: a bare command looked up on PATH, or an absolute path to the binary.",
                ),
                io.Int.Input(
                    "ctx_size",
                    default=20480,
                    min=2048,
                    max=262144,
                    step=1024,
                    tooltip="Context window in tokens. The system prompt + images + prompt + generation must all fit here.",
                ),
                io.String.Input(
                    "mmproj_path",
                    optional=True,
                    default="",
                    placeholder="C:/path/to/mmproj-model-f16.gguf",
                    tooltip="Optional vision projector (.gguf) path. Enables reference-image support for multimodal models.",
                ),
                io.String.Input(
                    "extra_server_args",
                    optional=True,
                    default="",
                    placeholder="--temp 0.7 --parallel 1 --flash-attn on",
                    tooltip="Extra command-line flags appended to the llama-server invocation. Quoted arguments are supported.",
                ),
                io.Int.Input(
                    "startup_timeout",
                    optional=True,
                    default=120,
                    min=10,
                    max=900,
                    step=10,
                    tooltip="Seconds to wait for a cold server to load the model and become healthy.",
                ),
            ],
            outputs=[
                LLMServerType.Output(
                    "llm_server",
                    tooltip="The server connection. Feed this into the LLM helper nodes.",
                )
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, **inputs):
        model_path = require_model_path(inputs["model_path"])
        server_path = resolve_server_path(
            inputs.get("llama_server_path") or "llama-server"
        )
        mmproj_path = optional_absolute_path(
            inputs.get("mmproj_path") or "", "mmproj_path"
        )
        extra_flags = (inputs.get("extra_server_args") or "").strip()
        startup_timeout = int(inputs.get("startup_timeout") or 120)
        ctx_size = int(inputs["ctx_size"])

        server = get_or_start_server(
            server_path,
            model_path,
            ctx_size,
            mmproj_path=mmproj_path,
            extra_flags=extra_flags,
            startup_timeout=startup_timeout,
        )
        if server is None:
            raise RuntimeError(
                "Failed to start llama-server. Check the model path, that the "
                "llama-server binary is correct, and the ComfyUI logs for the captured "
                "server output. Lower ctx_size if the model cannot fit."
            )
        return io.NodeOutput(server)

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        # Cache key tied to the server start generation: when a llama-server is
        # (re)started this changes, so ComfyUI's output cache misses and Load
        # re-runs - guaranteeing a fresh, live server is emitted, never a stale
        # (closed) one cached from an earlier prompt.
        return server_generation()


class CloseLLMServer(io.ComfyNode):
    """Stop a llama-server connection to free its VRAM, and abort on a failed check.

    Place this after the last LLM node that uses the connection - typically right
    before a sampling node - so the model's VRAM is released before the sampler
    needs it. Connect the ``llm_server`` output of the last consumer node here.

    Optionally wire a Rules Checker / Image Rules Checker ``judgement`` into the
    ``judgement`` input. The server is always closed first (releasing its VRAM);
    if the verdict is a failure (a non-empty judgement) the workflow then aborts
    with that message. The checker nodes themselves never abort - the close does
    - so a failed check never leaves the server holding VRAM.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CloseLLMServer",
            display_name="Close LLM Server",
            category="LlamaServer",
            # An output node so ComfyUI actually schedules it: the executor only
            # runs nodes that are output nodes or upstream of one, and Close has
            # no outputs of its own, so without this it would never run.
            is_output_node=True,
            description="Stop the llama-server process and release its VRAM. Put this after the last LLM node that uses the connection. Optionally wire in a checker's judgement to abort the workflow after the server is closed.",
            inputs=[
                LLMServerType.Input(
                    "llm_server",
                    tooltip="The connection to close. Connect this to the llm_server output of your last LLM node (e.g. the enhancer), not to the Load node - the connection is what makes Close wait for that node to finish first.",
                ),
                io.String.Input(
                    "judgement",
                    optional=True,
                    default="",
                    multiline=True,
                    tooltip="Optional: the judgement from a Rules Checker / Image Rules Checker. The server is closed first; when this is non-empty the workflow then aborts with that message. Leave unconnected to just close the server.",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, **inputs):
        judgement = (inputs.get("judgement") or "").strip()
        # Close the server first so its VRAM is released even when we then abort
        # on a failed check - the checker no longer aborts, the close does.
        close_server(inputs["llm_server"])
        if judgement:
            raise RuntimeError(judgement)
        return io.NodeOutput()

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        # Always re-run, never serve a cached no-op: closing the server is a
        # side effect that must happen on every execution (NaN never equals
        # itself, so the output cache always misses).
        return float("NaN")


class LLMPrompt(io.ComfyNode):
    """Generic LLM building block: send a system + user prompt, get the response.

    The most flexible of the set - use it for any text-to-text task (summarize,
    rewrite, extract, classify, translate, ...). The ``system_prompt`` and
    ``user_prompt`` are free-form. On a failed LLM call the node raises rather
    than returning an empty string, so a server problem is never silent.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LLMPrompt",
            display_name="LLM Prompt",
            category="LlamaServer/LLM",
            description="Send a system and a user prompt to the LLM and return its response.",
            inputs=[
                LLMServerType.Input(
                    "llm_server", tooltip="The server connection from Load LLM Server."
                ),
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Optional instructions for the model's behaviour (role, format, constraints).",
                ),
                io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="Ask the model something...",
                    tooltip="The prompt to send. Leave empty to bypass and return an empty response.",
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
                    "max_tokens",
                    optional=True,
                    default=0,
                    min=0,
                    max=8192,
                    step=64,
                    tooltip="Maximum tokens to generate. 0 lets the server decide.",
                ),
            ],
            outputs=[
                io.String.Output("response", tooltip="The model's response text."),
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
        user_prompt = inputs.get("user_prompt") or ""
        if not user_prompt.strip():
            return io.NodeOutput("", server)

        response = chat_completion_interruptible(
            server,
            inputs.get("system_prompt") or "",
            user_prompt.strip(),
            int(inputs.get("seed", 0)),
            temperature=inputs.get("temperature"),
            max_tokens=int(inputs.get("max_tokens") or 0) or None,
        )
        if response is None:
            raise RuntimeError(
                "LLM Prompt failed: the server returned no response. Check the ComfyUI logs."
            )
        return io.NodeOutput(response, server)


# Default behaviour for the enhancer. The user can fully override this with their
# own free-form system prompt; the hard-baked target/preset machinery is gone.
DEFAULT_ENHANCE_SYSTEM = (
    "You are an expert at writing prompts for image and video generation models. "
    "Enhance the user's prompt: keep every intended subject, action, style and detail, "
    "and expand it with concrete, generation-useful specifics (lighting, composition, "
    "camera, texture, mood, colour palette) so the result is unambiguous and high quality. "
    "Do not invent contradictory subjects, and do not add watermarks, text, or signatures "
    "unless the user asks for them."
)


class PromptEnhancer(io.ComfyNode):
    """Enhance a prompt for image/video generation using the LLM.

    The enhancement behaviour is driven by the free-form ``system_prompt`` (a
    sensible default is provided); there is no target-model preset system. On a
    failed enhancement the node degrades to passing the original prompt through
    unchanged so a model hiccup never kills the workflow.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PromptEnhancer",
            display_name="Prompt Enhancer (LLM Server)",
            category="LlamaServer/LLM",
            description="Enhance a prompt using the LLM, optionally producing a negative prompt too.",
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
                    default=DEFAULT_ENHANCE_SYSTEM,
                    multiline=True,
                    tooltip="Free-form instructions controlling how the prompt is enhanced. A generic image/video enhancement default is provided.",
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
                    default=25,
                    min=1,
                    max=500,
                    step=1,
                    tooltip="Minimum word count for an enhanced prompt to be accepted by the quality gate.",
                ),
                io.Boolean.Input(
                    "generate_negative_prompt",
                    optional=True,
                    default=True,
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

    @classmethod
    def execute(cls, **inputs):
        server = inputs["llm_server"]
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return io.NodeOutput(prompt, "", server)

        ref_images = collect_images(inputs.get("ref_images"))
        result = enhance_prompt(
            server,
            prompt,
            (inputs.get("system_prompt") or DEFAULT_ENHANCE_SYSTEM).strip(),
            seed=int(inputs.get("seed", 0)),
            temperature=inputs.get("temperature"),
            max_retries=int(inputs.get("max_retries", 3)),
            min_words=int(inputs.get("min_words", 25)),
            images=ref_images or None,
            generate_negative=bool(inputs.get("generate_negative_prompt", True)),
        )
        if result.positive:
            return io.NodeOutput(result.positive, result.negative or "", server)
        # Degraded: the model could not produce an acceptable enhancement.
        # Pass the original prompt through so the workflow still completes.
        return io.NodeOutput(prompt, "", server)


class RulesChecker(io.ComfyNode):
    """Judge free-form text against free-form rules using the LLM.

    You write the rules in plain language; the model decides whether the text
    complies with all of them. On a pass the text is passed through unchanged
    and the ``judgement`` output is empty. On a failure the text still passes
    through and the model's reason is reported on the ``judgement`` output.

    This node never aborts the workflow - wire the ``judgement`` output into the
    ``judgement`` input of the Close LLM Server node to abort (after the server
    is closed and its VRAM is released).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RulesChecker",
            display_name="Rules Checker (LLM Server)",
            category="LlamaServer/LLM",
            description="Check free-form text against free-form rules with the LLM and report the verdict on the judgement output. Wire the judgement into Close LLM Server to abort on failure.",
            inputs=[
                LLMServerType.Input(
                    "llm_server", tooltip="The server connection from Load LLM Server."
                ),
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="The text to check (e.g. an enhanced prompt)...",
                    tooltip="The text to validate. Leave empty to bypass and pass it through unchanged.",
                ),
                io.String.Input(
                    "rules",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="The enhanced prompt must describe the same character as the original, and must not add new people.",
                    tooltip="Free-form rules the text must satisfy, in plain language. Leave empty to bypass.",
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
                    "text_out",
                    tooltip="The input text, always passed through unchanged.",
                ),
                io.String.Output(
                    "judgement",
                    tooltip="Empty on pass (and on bypass). The failure message when the check fails; wire this into the 'judgement' input of Close LLM Server to abort the workflow after the server is closed.",
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
        text = inputs.get("text") or ""
        rules = (inputs.get("rules") or "").strip()
        if not text.strip() or not rules:
            return io.NodeOutput(text, "", server)

        verdict = check_rules(
            server,
            text,
            rules,
            seed=int(inputs.get("seed", 0)),
            temperature=inputs.get("temperature"),
        )
        if not verdict.ok:
            # Infrastructure failure: no usable verdict. Report it as a
            # non-empty judgement (not a pass) so the Close node closes the
            # server and aborts, rather than passing a check we could not run.
            return io.NodeOutput(
                text,
                f"Rules check error: could not obtain a verdict from the LLM: {verdict.reason}",
                server,
            )
        if verdict.passed:
            return io.NodeOutput(text, "", server)
        # Never abort here - the Close node does, after the server is closed.
        return io.NodeOutput(text, f"Rules check failed: {verdict.reason}", server)


NODE_CLASS_MAPPINGS = {
    "LoadLLMServer": LoadLLMServer,
    "CloseLLMServer": CloseLLMServer,
    "LLMPrompt": LLMPrompt,
    "PromptEnhancer": PromptEnhancer,
    "RulesChecker": RulesChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLLMServer": "Load LLM Server",
    "CloseLLMServer": "Close LLM Server",
    "LLMPrompt": "LLM Prompt",
    "PromptEnhancer": "Prompt Enhancer (LLM Server)",
    "RulesChecker": "Rules Checker (LLM Server)",
}
