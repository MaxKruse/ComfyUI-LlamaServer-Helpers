"""llama-server lifecycle and the shared connection object.

Starts a single ``llama-server`` (llama.cpp) process on demand, keeps it warm
across prompt executions, and hands out an :class:`LLMServer` connection object
that the consumer nodes (enhancer, rules checker, generic prompt) use to talk
to the model. A Close node - or ComfyUI shutdown - tears the process down.

This module is deliberately free of any ComfyUI import at module scope so it
can be imported and unit-tested without the ComfyUI runtime present. The only
runtime-specific call (freeing GPU memory before a cold start) is lazy and a
no-op when ``comfy`` is not importable.
"""

from __future__ import annotations

import atexit
import gc
import json
import logging
import random
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from urllib import error, request

logger = logging.getLogger(__name__)

# How long to wait for a cold server to become healthy (full GGUF load).
_SERVER_STARTUP_TIMEOUT = 120

# Timeout for a single HTTP round-trip (health / model discovery).
_HTTP_TIMEOUT = 5

# Short timeout for the tight /health poll loop: a not-yet-bound port would
# otherwise block each poll for ~2s (connection-refused delay).
_HEALTH_POLL_TIMEOUT = 1

# Ephemeral port range to probe for a free port.
_PORT_RANGE = range(49152, 65536)

_LOG_TAG = "LlamaServerHelpers"

_print_lock = Lock()


def _print_safe(msg: str):
    """Thread-safe print that survives non-UTF-8 console codepages (e.g. cp1252)."""
    with _print_lock:
        try:
            print(msg)  # noqa: T201
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(msg.encode(enc, "replace").decode(enc, "replace"))  # noqa: T201


def find_free_port() -> int:
    """Find a free TCP port by probing random ports in the ephemeral range."""
    candidates = random.sample(list(_PORT_RANGE), min(100, len(_PORT_RANGE)))
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    # Fallback: let the OS pick a free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_command(
    server_path: str,
    model_path: str,
    port: int,
    *,
    ctx_size: int = 10240,
    mmproj_path: str = "",
    extra_flags: str = "",
) -> list[str]:
    """Build the full llama-server command line.

    ``extra_flags`` is split with ``shlex`` so quoted arguments (e.g. paths with
    spaces) survive. User flags are appended last so they can override the
    defaults above when the user wants to.
    """
    import shlex

    user_flags = shlex.split(extra_flags) if extra_flags else []
    cmd = [
        server_path,
        "-m",
        model_path,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--no-ui",
        "--no-warmup",
        "-c",
        str(ctx_size),
    ]
    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])
    cmd.extend(user_flags)
    return cmd


def check_interrupt():
    """Raise if ComfyUI requested an interrupt. No-op outside ComfyUI (tests)."""
    try:
        import comfy.model_management
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def free_gpu_memory():
    """Unload all ComfyUI models and free VRAM so a cold llama-server can load.

    Called only on a cold start, not when reusing a warm server. No-op (except
    a best-effort torch cache clear) when ComfyUI is not importable.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        model_management = None

    try:
        import torch
    except ImportError:
        torch = None

    if model_management is not None:
        _print_safe(f"  [{_LOG_TAG}] Freeing GPU memory for llama-server...")
        model_management.unload_all_models()
        model_management.soft_empty_cache(force=True)
        gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    if model_management is not None:
        _print_safe(f"  [{_LOG_TAG}] GPU memory freed.")


@dataclass
class LLMServer:
    """Connection to a running llama-server.

    This is the "meta-object" handed from the Load node to every consumer node.
    It carries everything needed to talk to the model (``base_url`` +
    ``model_name``) plus the handle used to tear the process down later.
    """

    base_url: str
    model_name: str
    proc: subprocess.Popen
    port: int
    model_path: str
    ctx_size: int
    server_path: str = ""
    mmproj_path: str = ""
    extra_flags: str = ""
    started_at: float = field(default_factory=time.monotonic)
    _closed: bool = field(default=False, repr=False, compare=False)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def config_key(self) -> tuple:
        """Stable identity for the warm cache: the exact server configuration."""
        return (
            self.server_path,
            self.model_path,
            self.ctx_size,
            self.mmproj_path,
            self.extra_flags,
        )

    def is_alive(self) -> bool:
        """True while the underlying process is still running and not closed."""
        return not self._closed and self.proc is not None and self.proc.poll() is None

    def close(self):
        """Stop the server process. Idempotent."""
        if self._closed:
            return
        self._closed = True
        kill_server(self.proc)


def wait_for_server(
    base_url: str, proc, timeout: int = _SERVER_STARTUP_TIMEOUT
) -> bool:
    """Poll /health until healthy, the process exits, or the timeout is reached.

    Raises ComfyUI's interrupt exception if an interrupt is requested while waiting.
    """
    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        check_interrupt()
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with request.urlopen(url, timeout=_HEALTH_POLL_TIMEOUT) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def discover_model_name(base_url: str) -> str:
    """Fetch the loaded model id from /v1/models. Falls back to "llama"."""
    try:
        with request.urlopen(f"{base_url}/v1/models", timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            for m in models:
                status = m.get("status", {})
                if isinstance(status, dict) and status.get("value") == "loaded":
                    return m.get("id", "llama")
            if models:
                return models[0].get("id", "llama")
    except (error.URLError, json.JSONDecodeError, KeyError, IndexError):
        pass
    return "llama"


def _kill_process_tree(pid: int) -> bool:
    """Force-kill a process and every process it spawned (Windows only).

    ``llama_server_path`` may be a wrapper (.bat/.cmd/shim, or the
    llama-server router mode) that launches the real model server as a child;
    killing only the direct child would orphan the actual llama-server and its
    VRAM. ``taskkill /T`` walks the tree rooted at the PID we spawned - never
    by image name, so unrelated llama-server processes are untouched.
    Returns True when the tree kill was attempted, False to fall back to the
    plain terminate/kill path.
    """
    if sys.platform != "win32":
        return False
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        return True
    except Exception:
        return False


def kill_server(proc: subprocess.Popen | None):
    """Terminate the server process (and any children it spawned) and wait."""
    if proc is None:
        return
    pid = proc.pid
    _print_safe(f"  [{_LOG_TAG}] Stopping llama-server (pid={pid})...")
    try:
        if _kill_process_tree(pid):
            # taskkill already force-killed the tree; just reap our child.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "llama-server (pid=%s) did not terminate, sending SIGKILL", pid
                )
                proc.kill()
                proc.wait(timeout=3)
    except Exception:
        logger.exception("Error stopping llama-server")
    finally:
        _print_safe(f"  [{_LOG_TAG}] llama-server (pid={pid}) stopped.")


def _drain_stream(stream, tail: deque):
    """Read a subprocess text stream line-by-line into a bounded tail deque."""
    try:
        for line in iter(stream.readline, ""):
            tail.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _log_server_error(tail: deque):
    """Log the captured server output tail to help diagnose startup failures."""
    lines = list(tail)
    if not lines:
        return
    snippet = "\n".join(lines[-40:])
    logger.error("[%s] llama-server output (last lines):\n%s", _LOG_TAG, snippet)
    _print_safe(f"  [{_LOG_TAG}] llama-server output (last lines):\n{snippet}")


def start_server(
    server_path: str,
    model_path: str,
    port: int,
    ctx_size: int,
    mmproj_path: str = "",
    extra_flags: str = "",
    startup_timeout: int = _SERVER_STARTUP_TIMEOUT,
) -> LLMServer | None:
    """Start llama-server, wait until healthy, and return a live LLMServer.

    Returns None when the server could not be started (binary missing, process
    exited early, or not healthy in time). Captures server output so early
    failures are logged with diagnostics. Raises the interrupt exception if an
    interrupt is requested while waiting (after cleaning up the process).
    """
    cmd = build_command(
        server_path,
        model_path,
        port,
        ctx_size=ctx_size,
        mmproj_path=mmproj_path,
        extra_flags=extra_flags,
    )
    _print_safe(
        f"  [{_LOG_TAG}] Starting llama-server (model: {Path(model_path).name}, port: {port})..."
    )
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        _print_safe(f"  [{_LOG_TAG}] Started llama-server (pid={proc.pid})")
    except FileNotFoundError:
        logger.error("[%s] llama-server not found at: %s", _LOG_TAG, server_path)
        return None
    except Exception:
        logger.exception("[%s] Failed to start llama-server", _LOG_TAG)
        return None

    base_url = f"http://127.0.0.1:{port}"
    err_tail = deque(maxlen=80)
    err_thread = Thread(target=_drain_stream, args=(proc.stdout, err_tail), daemon=True)
    err_thread.start()

    try:
        healthy = wait_for_server(base_url, proc, timeout=startup_timeout)
    except BaseException:
        kill_server(proc)
        err_thread.join(timeout=2)
        raise

    if not healthy:
        if proc.poll() is not None:
            logger.error(
                "[%s] llama-server exited early (code=%s) before becoming healthy.",
                _LOG_TAG,
                proc.poll(),
            )
        else:
            logger.error(
                "[%s] llama-server did not become healthy within %ds.",
                _LOG_TAG,
                startup_timeout,
            )
        kill_server(proc)
        err_thread.join(timeout=2)
        _log_server_error(err_tail)
        return None

    model_name = discover_model_name(base_url)
    _print_safe(f"  [{_LOG_TAG}] Server ready. Model: {model_name}")
    return LLMServer(
        base_url=base_url,
        model_name=model_name,
        proc=proc,
        port=port,
        model_path=model_path,
        ctx_size=ctx_size,
        server_path=server_path,
        mmproj_path=mmproj_path,
        extra_flags=extra_flags,
    )


# --- Warm cache -----------------------------------------------------------
# The whole point of the loader/consumer split: a server started by the Load
# node is reused for the rest of the ComfyUI session (or until a Close node
# tears it down), so repeated queue items do not pay the cold-start cost.
_warm_servers: dict[tuple, LLMServer] = {}
_warm_lock = Lock()

# "Start generation": bumps every time a llama-server is cold-started. The Load
# node returns this as its cache fingerprint (``fingerprint_inputs``) so
# ComfyUI's output cache never hands out a stale (closed) LLMServer: the
# fingerprint changes exactly when a server is (re)started, forcing a cache
# miss and a fresh, live server. It is global (not per-config) on purpose: a
# bump may make a sibling Load node re-run, but re-running is a cheap warm-cache
# lookup, and the invariant that matters is that a Load node never returns a
# dead server.
_server_start_generation = 0


def server_generation() -> int:
    """Current start generation, used as the Load node's cache fingerprint."""
    return _server_start_generation


def get_or_start_server(
    server_path: str,
    model_path: str,
    ctx_size: int,
    mmproj_path: str = "",
    extra_flags: str = "",
    startup_timeout: int = _SERVER_STARTUP_TIMEOUT,
) -> LLMServer | None:
    """Return a live server for this config, reusing a warm one when possible.

    ``server_path`` and ``model_path`` must already be resolved and validated
    by the caller (the node layer does that). Returns a live :class:`LLMServer`
    or None when a cold start fails. Frees ComfyUI GPU memory only on a cold
    start, never when reusing a warm server.
    """
    global _server_start_generation
    check_interrupt()
    key = (server_path, model_path, ctx_size, mmproj_path, extra_flags)
    with _warm_lock:
        existing = _warm_servers.get(key)
        if existing is not None and existing.is_alive():
            _print_safe(
                f"  [{_LOG_TAG}] Reusing warm llama-server (port {existing.port}, model {existing.model_name})"
            )
            return existing

    free_gpu_memory()
    port = find_free_port()
    server = start_server(
        server_path,
        model_path,
        port,
        ctx_size,
        mmproj_path,
        extra_flags,
        startup_timeout,
    )
    if server is None:
        return None
    with _warm_lock:
        _warm_servers[key] = server
        _server_start_generation += 1
    return server


def _same_process(a: LLMServer, b: LLMServer) -> bool:
    """True when two connections wrap the very same operating-system process."""
    return a is b or (
        a.proc is not None and b.proc is not None and a.proc.pid == b.proc.pid
    )


def close_server(server: LLMServer | None) -> None:
    """Stop one server and drop it from the warm cache. Idempotent.

    The cache entry is dropped when the stored server is this object, or a
    different wrapper around the same process (duplicate objects can flow
    through the graph). A different PID - a newer server for the same config -
    is never torn down by mistake.
    """
    if server is None:
        return
    with _warm_lock:
        current = _warm_servers.get(server.config_key)
        if current is not None and _same_process(current, server):
            _warm_servers.pop(server.config_key, None)
    server.close()


def close_all_servers():
    """Stop every tracked server. Used at interpreter shutdown as a safety net."""
    with _warm_lock:
        servers = list(_warm_servers.values())
        _warm_servers.clear()
    for server in servers:
        server.close()


atexit.register(close_all_servers)
