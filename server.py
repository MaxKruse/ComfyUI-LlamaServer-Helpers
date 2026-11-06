"""llama-server lifecycle and the shared server handle.

The Load node emits an :class:`LLMServer` handle - a stable, cacheable object
carrying the full server configuration but no process. The actual
``llama-server`` (llama.cpp) process starts lazily on first real use, is kept
warm in a session-wide cache (keyed by configuration), and is torn down by a
Close node or ComfyUI shutdown.

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

    ``extra_flags`` is split with ``shlex`` (quoted arguments survive); user
    flags are appended last so they can override the defaults above.
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


def check_interrupt(server: LLMServer | None = None):
    """Raise if ComfyUI requested an interrupt. No-op outside ComfyUI (tests).

    When *server* is given, an interrupt kills that server (and drops it from
    the warm cache) before the exception propagates: the Close node downstream
    of the interrupted node never runs, so without this the llama-server would
    be left running and holding VRAM until the ComfyUI session ends.
    """
    try:
        import comfy.model_management
    except ImportError:
        return
    try:
        comfy.model_management.throw_exception_if_processing_interrupted()
    except BaseException:
        if server is not None:
            close_server(server)
        raise


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
class LLMServerProcess:
    """A running llama-server: the OS process plus how to talk to it.

    Internal object - nodes and tests should deal with the :class:`LLMServer`
    handle, which resolves (and caches) one of these per configuration.
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


class LLMServerStartError(RuntimeError):
    """A cold start of the llama-server failed (see the ComfyUI logs)."""


class LLMServer:
    """Stable, cacheable handle to a llama-server for one fixed configuration.

    The "meta-object" the Load node emits and every consumer node threads
    through: it carries the full server configuration (which is what makes the
    Load node cacheable by inputs) but no process. The llama-server starts
    lazily on first real use (:meth:`ensure_live`, or the first ``base_url`` /
    ``model_name`` read), is reused warm while alive, and is resolved again
    after a Close.
    """

    def __init__(
        self,
        server_path: str,
        model_path: str,
        ctx_size: int,
        mmproj_path: str = "",
        extra_flags: str = "",
        startup_timeout: int = _SERVER_STARTUP_TIMEOUT,
    ):
        self.server_path = server_path
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.mmproj_path = mmproj_path
        self.extra_flags = extra_flags
        self.startup_timeout = startup_timeout
        self._live: LLMServerProcess | None = None

    @property
    def config_key(self) -> tuple:
        """Warm-cache identity: the exact server configuration.

        ``startup_timeout`` is not part of the key: two Load nodes with the
        same model but different timeouts share one server.
        """
        return (
            self.server_path,
            self.model_path,
            self.ctx_size,
            self.mmproj_path,
            self.extra_flags,
        )

    def is_alive(self) -> bool:
        """True while the process behind this handle is running and not closed."""
        return self._live is not None and self._live.is_alive()

    def ensure_live(self) -> None:
        """Resolve a live process for this config, starting one if needed.

        Warm reuse never frees GPU memory; a cold start does. Raises the
        ComfyUI interrupt exception (killing the in-use server first) when an
        interrupt is requested, and :class:`LLMServerStartError` when a cold
        start fails.
        """
        check_interrupt(self)
        with _warm_lock:
            existing = _warm_servers.get(self.config_key)
        if existing is not None and existing.is_alive():
            if self._live is not existing:
                _print_safe(
                    f"  [{_LOG_TAG}] Reusing warm llama-server "
                    f"(port {existing.port}, model {existing.model_name})"
                )
            self._live = existing
            return

        free_gpu_memory()
        port = find_free_port()
        process = start_server(
            self.server_path,
            self.model_path,
            port,
            self.ctx_size,
            self.mmproj_path,
            self.extra_flags,
            self.startup_timeout,
        )
        if process is None:
            raise LLMServerStartError(
                "Failed to start llama-server. Check the model path, that the "
                "llama-server binary is correct, and the ComfyUI logs for the "
                "captured server output. Lower ctx_size if the model cannot fit."
            )
        with _warm_lock:
            _warm_servers[self.config_key] = process
        self._live = process

    def _resolve(self):
        """Return the live process, cold-starting it when needed."""
        if not self.is_alive():
            self.ensure_live()
        return self._live

    @property
    def base_url(self) -> str:
        return self._resolve().base_url

    @property
    def model_name(self) -> str:
        return self._resolve().model_name

    @property
    def port(self) -> int:
        return self._resolve().port

    def close(self):
        """Stop this config's server. Idempotent."""
        close_server(self)


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
) -> LLMServerProcess | None:
    """Start llama-server, wait until healthy, and return a live process object.

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
    return LLMServerProcess(
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
# A server started for one configuration is reused for the rest of the
# ComfyUI session (or until a Close node tears it down), so repeated queue
# items do not pay the cold-start cost.
_warm_servers: dict[tuple, LLMServerProcess] = {}
_warm_lock = Lock()


def close_server(server: "LLMServer | None") -> None:
    """Stop the server for this handle's configuration. Idempotent.

    Closing is config-scoped: every handle for the same configuration shares
    one process, so closing any one of them frees that config's server - the
    process currently owned by the config, not whatever a stale handle's own
    reference points at. ``None`` is a no-op.
    """
    if server is None:
        return
    with _warm_lock:
        process = _warm_servers.pop(server.config_key, None)
    server._live = None
    if process is not None and process.is_alive():
        process.close()


def close_all_servers():
    """Stop every tracked server. Used at interpreter shutdown as a safety net."""
    with _warm_lock:
        servers = list(_warm_servers.values())
        _warm_servers.clear()
    for server in servers:
        server.close()


atexit.register(close_all_servers)
