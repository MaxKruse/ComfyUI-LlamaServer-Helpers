# ComfyUI LlamaServer Helpers

Run a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` from inside ComfyUI and put it to work on your prompts - no separate terminal, and no cold start on every queue item.

The nodes share **one** server connection per model: start it once with **Load LLM Server**, chain any number of LLM nodes on it, then stop it with **Close LLM Server** to release VRAM before sampling.

## What it does

- Starts and manages a `llama-server` process for you (model, context size, and any extra flags).
- Reuses a **warm** server across queue items, so only the first run pays the model-load cost.
- Gives you ready-made nodes for common LLM tasks plus a generic one:
  - enhancing a generation prompt (with an optional negative prompt),
  - checking free-form text against free-form rules,
  - checking generated image(s) against free-form rules with a vision model,
  - and a plain system + user prompt in, response out.
- Includes a prompt enhancer per target model (FLUX, SDXL, SD 1.5, Krea 2, Chroma, Z-Image, Qwen-Image, Qwen-Image 2.1 Edit, MOSS-Image SFT, MiniMax H3, MiniMax H3 R2V, MiniMax H3 Eros Max, Wan 2.2, LTX 2.3 10Eros, LTX 2.5), each pre-filled with that model's documented prompting dialect.

## What you need

- A `llama-server` executable from llama.cpp - either on your `PATH` (then the Load node's server path can stay `llama-server`) or an absolute path to the binary.
- A GGUF model file (`.gguf`), and optionally a vision projector (`.gguf`) if you want to feed reference images to a multimodal model.
- Nothing to `pip install`: the nodes use only the Python standard library, on top of what ComfyUI already provides.

## The nodes

| Node | What it does |
|---|---|
| **Load LLM Server** | Starts (or reuses a warm) `llama-server` for a model + context size + flags, and emits the `llm_server` connection. |
| **Close LLM Server** | Stops the server and frees its VRAM. Put it after the last LLM node. |
| **LLM Prompt** | Generic building block: send a system + a user prompt, get the response back. |
| **Prompt Enhancer (LLM Server)** | Rewrites and expands a generation prompt, optionally producing a negative prompt too. |
| **Flux / SDXL / SD1.5 / Krea 2 / Chroma / Z-Image / Qwen-Image / Qwen-Image 2.1 Edit / MOSS-Image SFT Prompt Enhancer (LLM Server)** | One enhancer per target model: same as the Prompt Enhancer, but the system prompt is pre-filled with that model's documented prompt dialect (prose for FLUX, booru tags for SD 1.5, and so on) and the quality-gate defaults match it. Use the model-matched node when you know which model the prompt feeds. The Qwen-Image 2.1 Edit node is the image-editing (i2i) variant: feed it the image to edit on `ref_images` and write the edit you want. |
| **MiniMax H3 / MiniMax H3 R2V / MiniMax H3 Eros Max / Wan 2.2 / LTX 2.3 10Eros / LTX 2.5 Prompt Enhancer (LLM Server)** | The video-model variants, under the `LlamaServer/Video` category: they produce audiovisual scripts (MiniMax H3), full reference-to-video rewrites (MiniMax H3 R2V / Eros Max), and motion-and-camera scripts (Wan 2.2, LTX) instead of image prompts. The i2v models (LTX, Wan 2.2) take the source image on `ref_images`; the R2V models take any reference images there too. |
| **Rules Checker (LLM Server)** | Checks free-form text against free-form rules; aborts the workflow on failure, or reports the model's judgement. |
| **Image Rules Checker (LLM Server)** | Checks generated image(s) against free-form rules using a vision model (for example: "no watermark", "exactly one person"); aborts the workflow on failure, or reports the model's judgement. Requires the server to be loaded with a vision model and an `mmproj` path. |

Every LLM node takes the `llm_server` connection in and passes it straight back out, so a server threads through the workflow: `Load → LLM node(s) → Close`. The Close node is therefore guaranteed to run after the nodes that used the server.

## How to run

1. Put this folder in `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add a **Load LLM Server** node and set your model path and context size.
4. Wire its `llm_server` output into the LLM node(s) you want, then wire the last one's `llm_server` output into **Close LLM Server**.

## Limitations

- One server is kept per distinct model configuration (model, context size, projector, flags). Loading a *different* configuration starts a *second* server, and each holds its own VRAM.
- The Enhancer and Rules Checker ask the **model** to make the call, so the quality of the result is limited by the model you load. Small local models can give weak enhancements or unreliable verdicts.
- If the Enhancer can't get an acceptable result after a few retries, it falls back to passing your original prompt through unchanged so a model hiccup doesn't kill the queue.
- The Rules Checker aborts (or reports) based on the model's judgement; a "pass" is the model's best guess, not a guarantee.
- The Image Rules Checker needs a **vision-capable** model plus an `mmproj` path on the Load node; a text-only model cannot look at the images. Like the text checker, its verdict is the model's best guess - for hard guarantees, keep the rules specific and observable.
- The per-model enhancers encode each model's *documented* dialect; model generations still vary, and a model the list does not cover should use the generic Prompt Enhancer with your own system prompt.

## Troubleshooting

- **"Failed to start llama-server" / a red Load node** - open the ComfyUI console: the server's own output is captured and logged there. The usual causes are a wrong model path, a missing or incompatible `llama-server` binary, or a context size the model can't fit (lower `ctx_size`).
- **"model_path must be an absolute path"** - use a full path such as `C:/models/your-model.gguf`, not a relative one.
- **Slow first run** - expected; the first queue item loads the model and later items reuse the warm server.
- **Still stuck?** Get help on the ComfyUI GitHub issues or subreddit with your model name, context size, and the console log.
