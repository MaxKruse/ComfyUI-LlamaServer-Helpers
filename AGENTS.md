# ComfyUI LlamaServer Helpers - Agent Guide

Technical documentation for agents working on this custom node. The user-facing
view (what/limits/run/help) is in `README.md`; this file is where the design
lives.

## What this is

A ComfyUI custom node that starts a local llama.cpp `llama-server` and exposes
a set of nodes that all share **one** connection to that model:

- `LoadLLMServer` - start (or reuse a warm) server, emit the connection.
- `CloseLLMServer` - stop the server, free VRAM.
- `LLMPrompt` - generic system + user prompt -> response.
- `PromptEnhancer` - enhance a prompt (+ optional negative prompt).
- `RulesChecker` - judge free-form text against free-form rules.
- `ImageRulesChecker` - judge generated image(s) against free-form rules
  (vision model + mmproj required).
- One enhancer per target model, built by a factory from data-driven specs:
  `FluxPromptEnhancer`, `Krea2PromptEnhancer`, `ZImagePromptEnhancer`,
  `QwenImagePromptEnhancer`, `QwenImageEditPromptEnhancer`
  (Qwen-Image 2.1 i2i), `MossImagePromptEnhancer`, `MiniMaxH3PromptEnhancer`,
  `MiniMaxH3R2VPromptEnhancer`, `MiniMaxH3ErosMaxPromptEnhancer`,
  `Wan22PromptEnhancer`, `Ltx23PromptEnhancer` (LTX 2.3 10Eros),
  `Ltx25PromptEnhancer`
  (same shape as `PromptEnhancer`, but with the model's documented prompt
  dialect baked into the system-prompt default).

The core UX win over a cold-start-per-run design: the server is started once and
kept **warm** for the ComfyUI session, so repeated queue items reuse it instead
of re-loading the model each time.

## Layering

Two layers, split deliberately:

- **Pure-logic modules** (`server.py`, `client.py`, `enhance.py`, `judge.py`)
  import **no** `comfy_api`. They only need the Python standard library, plus
  lazy, `try/except ImportError` imports of `comfy.model_management` inside
  `check_interrupt()` / `free_gpu_memory()` (no-op outside ComfyUI). This keeps
  them unit-testable without the ComfyUI runtime on the path.
- **Pure-data module** (`presets.py`): the `EnhancerSpec` NamedTuple plus
  `ENHANCER_SPECS`, the per-model dialect system prompts and quality-gate
  defaults for the per-model enhancers. Imports only `pathlib`: the six
  mirrored dialect guides are loaded from the `presets/` directory at import
  time. Keep it stdlib-only.
- **Comfy-dependent modules** (`node_base.py`, `nodes.py`, `model_nodes.py`)
  import `comfy_api.latest.io` at module scope and are only importable inside
  the ComfyUI runtime (or with it on the path in tests).

Do not add `comfy_api` imports to the pure-logic/data modules, and do not add
a new third-party dependency (the node is stdlib-only by design).

## File layout

| File | Role |
|---|---|
| `__init__.py` | Merges the mappings from `.nodes` and `.model_nodes` into `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` inside a `try/except` guard (empty on failure so the package never breaks startup). Sets `WEB_DIRECTORY = None`. |
| `nodes.py` | The five base `io.ComfyNode` classes (Load/Close/LLMPrompt/PromptEnhancer/RulesChecker), the base mappings, and `DEFAULT_ENHANCE_SYSTEM`. |
| `presets.py` | Data: `EnhancerSpec` NamedTuple + `ENHANCER_SPECS` (one entry per target model: dialect system prompt, category, min-words and negative-prompt defaults). Compact dialects are inline constants; six full dialect guides are loaded from `presets/` at import time (`_load_preset`). |
| `presets/` | Six `.txt` dialect guides mirrored verbatim from `../ComfyUI-PromptEnhancer/presets/` (moss-image-sft-t2i, qwenimage2.1-i2i, ltx2.3-10eros-i2v, ltx2.5-i2v, minimax-h3-r2v, minimax-h3-eros-max-r2v). Keep them in sync with that package when its presets change. |
| `model_nodes.py` | The `_build_enhancer_node(spec)` factory (one `io.ComfyNode` subclass per spec, keyed in `MODEL_ENHANCER_NODES`), the `ImageRulesChecker` node, and the model mappings. |
| `node_base.py` | `LLMServerType = io.Custom("LLMSERVER")` plus the shared path validators (`resolve_server_path`, `require_model_path`, `optional_absolute_path`) and `collect_images`. |
| `server.py` | `LLMServer` dataclass; process lifecycle (`find_free_port`, `build_command`, `start_server`, `wait_for_server`, `discover_model_name`, `_kill_process_tree`, `kill_server` - the latter kills the whole process tree by PID on Windows); the warm cache (`get_or_start_server`, `close_server`, `close_all_servers`, `server_generation`, `atexit`). |
| `client.py` | HTTP chat client: `chat_completion`, `chat_completion_interruptible`, `build_user_content`, `_payload`, image-to-base64-JPEG, context-error detection, timing log. |
| `enhance.py` | `enhance_prompt` + the quality gate (`is_good_prompt`: min words, refusal patterns, Jaccard echo) + lenient JSON parsing. |
| `judge.py` | `check_rules` (text and/or images via the `images=` kwarg; vision path uses `_judge_image_system_prompt`) + `CheckResult` + verdict parsing. |
| `tests/` | `conftest.py` (path setup + `FakeProc` + fake-server fixtures) and one test file per logic module + the node layer (`test_nodes.py` for the base nodes, `test_model_nodes.py` for the factory-built enhancers, `ImageRulesChecker`, and the preset data). |

## The meta-object: `LLMServer`

A `dataclass` in `server.py` - the "meta-object" handed from `LoadLLMServer` to
every consumer. Fields (in order): `base_url`, `model_name`, `proc`, `port`,
`model_path`, `ctx_size`, `server_path=""`, `mmproj_path=""`, `extra_flags=""`,
`started_at` (monotonic), and a private `_closed` (`compare=False`).

- `config_key` property = `(server_path, model_path, ctx_size, mmproj_path,
  extra_flags)` - the identity of a "same configuration".
- `is_alive()` = `not _closed and proc is not None and proc.poll() is None`.
- `close()` is idempotent -> `kill_server(proc)`.

## Warm-cache state & lifecycle

Module-level `_warm_servers: dict[tuple, LLMServer]` guarded by `_warm_lock`
(lives for the ComfyUI session).

- `get_or_start_server(server_path, model_path, ctx_size, mmproj_path="",
  extra_flags="", startup_timeout=120)`: if a live server with the same
  `config_key` exists, reuse it (no VRAM churn). Otherwise `free_gpu_memory()`,
  `find_free_port()`, `start_server(...)`; on failure return `None`. On a
  successful cold start `_server_start_generation` is bumped (the `LoadLLMServer`
  cache fingerprint - see Gotchas).
- `close_server(server)`: pop the matching cache entry when the stored server
  is this object **or a different wrapper around the same process** (`_same_process`:
  same `proc.pid`), then `server.close()`. `close_server(None)` is a no-op.
- `close_all_servers()` + `atexit.register(close_all_servers)` are the safety
  net so a server is never left running when the process exits.

Invariants: a server is only ever started on a **cold** path (warm reuse never
calls `free_gpu_memory`); `close_server` must match on process identity (same
object, or a different wrapper with the same `proc.pid`), never just
`config_key`, so a *newer* server for the same config is never torn down by
mistake.

## The nodes: inputs, outputs, failure semantics

All use V3 `comfy_api.latest.io` schemas. Every LLM consumer takes
`LLMServerType.Input("llm_server")` and re-emits `LLMServerType.Output(
"llm_server")`, so a server threads `Load -> consumer(s) -> Close`. Threading
the connection is what orders Close *after* the consumers; Close (and the image
gate) are additionally declared output nodes so the executor actually schedules
them - ComfyUI only runs output nodes and their upstream dependencies, and a
connection-only sink with no outputs would otherwise never run. `LoadLLMServer`
returns the start-generation as its `fingerprint_inputs` so its output cache
never hands out a stale (closed) connection.

- **LoadLLMServer** - in: `model_path` (required, absolute, existing),
  `llama_server_path` (default `llama-server`; bare name on PATH or absolute),
  `ctx_size` (default 20480), `mmproj_path` (optional), `extra_server_args`
  (optional), `startup_timeout` (optional, default 120). out: `llm_server`.
  Raises `ValueError` on a bad path, `RuntimeError` if the server fails to start.
- **CloseLLMServer** - in: `llm_server`, `judgement` (optional string). out:
  none. An output node
  (`is_output_node=True`) with a NaN cache fingerprint (`fingerprint_inputs`
  returns `float("NaN")`): the executor only schedules output nodes (or their
  upstream dependencies), so without the flag a no-output sink is never reached
  (and right-click -> Run fails with "Prompt has no outputs"); the NaN
  fingerprint forces a re-run every time because closing is a side effect, never
  a cached no-op. Closes the server **first** (releasing VRAM), then, if
  `judgement` is non-empty (a failed checker verdict), **raises**
  `RuntimeError(judgement)` - this is where the rules checkers' abort lives, so
  a failed check never leaves the server holding VRAM.
- **LLMPrompt** - in: `llm_server`, `system_prompt` (optional), `user_prompt`,
  `seed`, `temperature` (optional), `max_tokens` (optional). out: `response`,
  `llm_server`. Empty `user_prompt` bypasses (returns `""`). A `None` response
  **raises** - a direct ask failing is never silent.
- **PromptEnhancer** - in: `llm_server`, `prompt`, `system_prompt` (default
  `DEFAULT_ENHANCE_SYSTEM`), `seed`, `temperature`, `max_retries` (default 3),
  `min_words` (default 25), `generate_negative_prompt` (default True),
  `ref_images` (Autogrow, 0-9). out: `enhanced_prompt`, `negative_prompt`,
  `llm_server`. **Degrades** to passing the original prompt through when the
  model can't produce an acceptable result, so a hiccup never kills the queue.
- **Per-model prompt enhancers** (`FluxPromptEnhancer`, `Krea2PromptEnhancer`,
  `ZImagePromptEnhancer`, `QwenImagePromptEnhancer`,
  `QwenImageEditPromptEnhancer` (Qwen-Image 2.1 i2i),
  `MossImagePromptEnhancer` under `LlamaServer/LLM`;
  `MiniMaxH3PromptEnhancer`, `MiniMaxH3R2VPromptEnhancer`,
  `MiniMaxH3ErosMaxPromptEnhancer`, `Wan22PromptEnhancer`,
  `Ltx23PromptEnhancer` (LTX 2.3 10Eros), `Ltx25PromptEnhancer` under
  `LlamaServer/Video`) -
  built by `_build_enhancer_node(spec)` in `model_nodes.py` from
  `presets.ENHANCER_SPECS` (one spec per model: node_id, display_name,
  description, dialect system prompt, category, `default_min_words`,
  `default_generate_negative`). Same inputs/outputs as `PromptEnhancer`, but
  `system_prompt` defaults to the model's dialect (still user-editable), and
  `min_words` / `generate_negative_prompt` default from the spec (Z-Image: 40;
  MOSS-Image: 40; LTX 2.3: 60; LTX 2.5: 40; MiniMax H3 / R2V / Eros Max: 50;
  no per-model enhancer defaults `generate_negative_prompt` on - only the
  generic `PromptEnhancer` does). The six mirrored models' dialects (Qwen-Image 2.1 Edit, MOSS-Image,
  both LTX, both MiniMax R2V variants) are the full dialect guides mirrored
  verbatim from `../ComfyUI-PromptEnhancer`, loaded from `presets/*.txt` at
  import time - keep the files in sync with that package. Same **degrade**
  semantics as `PromptEnhancer`.
  `MODEL_ENHANCER_NODES` maps node_id -> built class. The classes are built via
  `type()` (dynamic class creation); they are ordinary `io.ComfyNode`
  subclasses, nothing else is dynamic.
- **RulesChecker** - in: `llm_server`, `text`, `rules`, `seed`,
  `temperature`. out: `text_out`, `judgement`, `llm_server`.
  Empty text OR empty rules bypass (passes text through). The node **never
  aborts**: `ok=False` (no parseable verdict - an infra failure) reports
  `"Rules check error: ..."` on `judgement` (fail-closed, not a pass); `fail`
  reports `"Rules check failed: <reason>"`; `pass` reports `""`. Wire
  `judgement` into `CloseLLMServer` to abort after the server closes.
- **ImageRulesChecker** - in: `llm_server`, `rules` (optional), `images`
  (Autogrow, 0-9, VAE-decoded tensors), `seed`,
  `temperature`. out: `judgement`, `llm_server` (no image passthrough - it is a
  pure gate; fan the image to the save/next node separately). An output node
  (`is_output_node=True`) with a NaN cache fingerprint, so an unconnected gate
  is still scheduled and always re-runs (a cached no-op gate would silently
  pass). No images OR
  empty rules bypass (returns `("", server)`). The node **never aborts**:
  `ok=False` (no parseable verdict - an infra failure) reports
  `"Image rules check error: ..."` (mentions the vision/mmproj requirement,
  fail-closed, not a pass); `fail` reports
  `"Image rules check failed: <reason>"`; `pass` reports `""`. Wire `judgement`
  into `CloseLLMServer` to abort after the server closes. Calls
  `check_rules(server, "", rules, ..., images=...)`.

The Enhancer/Rules quality-gate and verdict logic lives in `enhance.py` /
`judge.py`, not in the nodes - the node just maps inputs and maps the result to
outputs.

## Conventions

- **Dual import pattern** in `client.py` / `enhance.py` / `judge.py` /
  `nodes.py` / `model_nodes.py`: `try: from .x import ... except ImportError:
  from x import ...`. This lets the logic be unit-tested directly (top-level
  import) while still working when ComfyUI loads the hyphenated package
  (relative import). Keep it if you add cross-module imports in those files.
  `server.py` and `node_base.py` use plain imports (server has no relatives;
  node_base only needs `comfy_api`); `presets.py` imports only `pathlib`
  (to load the mirrored `.txt` dialect guides from `presets/`).
- **Failure semantics are per-node and intentional** (see above). Do not make
  `LLMPrompt` silent on failure, or make `PromptEnhancer` (or any per-model
  enhancer) raise. `RulesChecker` / `ImageRulesChecker` never raise on a
  verdict - they report it on `judgement` (fail-closed on `ok=False`);
  aborting is the Close node's job, after the server is closed.
- Keep files under ~500 lines and functions under ~50 lines; keep comments
  sparse (only the "why").
- No internet access except the local `llama-server` on `127.0.0.1`; the only
  outbound HTTP is `request.urlopen` to `http://127.0.0.1:<port>`. No telemetry,
  no tracking, no external model downloads.

## Gotchas / non-obvious constraints

- **Close and the image gate MUST stay output nodes with a NaN fingerprint.**
  ComfyUI's executor (`execution.py`) seeds the run from nodes where
  `OUTPUT_NODE is True`, then walks *backward* through inputs only; a node whose
  outputs are all unconsumed and that is not an output node is pruned and never
  scheduled (right-click -> Run reports "Prompt has no outputs"). `CloseLLMServer`
  and `ImageRulesChecker` have no useful outputs, so both are
  `is_output_node=True`. Keep their `fingerprint_inputs` returning
  `float("NaN")`: closing / gating are side effects and must re-run every time -
  the default `RAM_PRESSURE` output cache is persistent across queue items, so a
  cached no-op would silently skip the side effect. Do not "simplify" them away.
- **`LoadLLMServer.fingerprint_inputs` returns the start-generation, not NaN.**
  `server.get_or_start_server` bumps `_server_start_generation` on every cold
  start and `LoadLLMServer.fingerprint_inputs` returns it, so Load's output-cache
  key changes exactly when a server (re)starts -> a cache miss -> a fresh live
  `LLMServer` is emitted (a cached closed one would otherwise be handed out on
  run 2). Do NOT make it return `float("NaN")`: a NaN in a node's signature
  propagates into every transitive descendant's signature (including the
  expensive sampler), making the whole downstream tree permanently non-cacheable.
  The generation is global (not per-config) on purpose; a bump may re-run a
  sibling Load node, but that is a cheap warm-cache lookup.
- **Hyphenated package name + relative imports.** The dir is
  `ComfyUI-LlamaServer-Helpers` (not a valid identifier). ComfyUI loads it via
  `app/governance.py:pack_module_spec` -> `spec_from_file_location(.../
  __init__.py)`, which **does** make `from .nodes import ...` work (verified
  against the working `ComfyUI-PromptEnhancer`, which uses the identical
  `__init__.py` guard). Do not "simplify" the `__init__.py` to a bare import.
- **Name collision with ComfyUI root.** `C:\Dev\ComfyUI` also has `server.py`
  and `nodes.py`. `tests/conftest.py` inserts the **package root** at
  `sys.path[0]` (before ComfyUI root) so `import server` / `import nodes` in the
  tests resolve to *this* package. If tests suddenly import ComfyUI's modules,
  check the conftest path order.
- **`io.NodeOutput` is tuple-like but not `==` to a tuple.** Indexing (`out[0]`)
  and `tuple(out)` work, but `out == (a, b)` does not. In tests, compare with
  `tuple(out) == (...)`.
- **Windows path semantics.** `os.path.isabs("/x")` is False on Windows; the
  path validators (and their tests) must use drive-letter / `tempfile`-based
  absolute paths, not a leading-slash one.
- **No em/en dashes** in code/comments/docs - use single hyphens.
- Never kill a process by image name, never write to `nul`, and never modify
  `C:\Dev\ComfyUI\.venv`.
- **`kill_server` kills the process tree, by our PID, never by image name.**
  On Windows it runs `taskkill /F /T /PID <pid>` (POSIX falls back to
  terminate -> SIGKILL). This is required because `llama_server_path` may be a
  wrapper (.bat/.cmd/shim) or the llama-server router mode, which spawn the
  real model server as a *child*: a plain `proc.terminate()` would kill only
  the wrapper and orphan the model server (and its VRAM). The same binary
  really does have a router mode (`--models-preset ... --port ...`) that
  spawns the model server as a child process. Never use
  `taskkill /IM llama-server.exe`: this machine also runs the agent's own
  llama-server on port 8080, and by-image kills would take it down.

## Testing

Run from the ComfyUI root (this is a `uv` project - **never** bare
`python`/`pip`; every `uv run` uses `--no-sync`; pytest is pulled in
ephemerally and is NOT installed into `.venv`):

```
uv run --no-sync --with pytest pytest custom_nodes/ComfyUI-LlamaServer-Helpers/tests/ -q
```

Mocking conventions:
- Node logic is mocked by `monkeypatch.setattr(nodes, "<fn>", ...)` where `<fn>`
  is one of `chat_completion_interruptible`, `enhance_prompt`, `check_rules`,
  `close_server`, `get_or_start_server`. The `model_nodes` module is mocked the
  same way (its `enhance_prompt` / `check_rules` are its own module-level
  names, so patch on `model_nodes`, not on `nodes`); the per-model enhancer
  classes are pulled from `model_nodes.MODEL_ENHANCER_NODES[node_id]`.
- HTTP is mocked by `monkeypatch.setattr(<mod>.request, "urlopen", ...)`
  returning a tiny context-manager (`.status`, `.read()`).
- The warm cache is tested by `monkeypatch.setattr(server, "start_server" /
  "find_free_port" / "free_gpu_memory", fake)` returning a fake `LLMServer`
  backed by `conftest.FakeProc`. `fake_server` / `make_fake_server` fixtures
  clear `_warm_servers` around each test.
- An **autouse** `fake_tree_kill` fixture (conftest) replaces
  `server._kill_process_tree` in every test - `FakeProc` PIDs are arbitrary
  and could collide with live host processes, so the real `taskkill` must
  never run in tests. It yields `(calls, real_fn)`; the one real-process
  integration test (`test_kill_process_tree_kills_wrapper_children`,
  Windows-only) restores `real_fn` via monkeypatch.

## Lint / format

```
uvx ruff check custom_nodes/ComfyUI-LlamaServer-Helpers/     # fix: add --fix
uvx ruff format custom_nodes/ComfyUI-LlamaServer-Helpers/
```

Ruff rules are inherited from the ComfyUI root `pyproject.toml` (`N805`, `S307`,
`S102`, `E`, `T`, `W`, `F`; `E501`/`E722`/`E731`/`E712`/`E402`/`E741` ignored).
`_print_safe` in `server.py` carries `# noqa: T201` on its two `print` calls on
purpose - that function is the node's user-facing console output.
