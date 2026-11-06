"""Tests for server.py: command building, port finding, lifecycle, warm cache."""

import json
import socket
import subprocess
import sys
import types
from urllib import error

import pytest

import server
from conftest import FakeProc


# --- build_command ---------------------------------------------------------


def test_build_command_base_flags():
    cmd = server.build_command("llama-server", "/m/model.gguf", 1234, ctx_size=4096)
    assert cmd[0] == "llama-server"
    assert "-m" in cmd and "/m/model.gguf" in cmd
    assert "--port" in cmd and "1234" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--no-ui" in cmd and "--no-warmup" in cmd
    assert "-c" in cmd and "4096" in cmd
    assert "--mmproj" not in cmd


def test_build_command_with_mmproj_and_extra_flags():
    cmd = server.build_command(
        "llama-server",
        "/m/model.gguf",
        1234,
        ctx_size=2048,
        mmproj_path="/m/mmproj.gguf",
        extra_flags="--temp 0.7 --flash-attn on",
    )
    assert "--mmproj" in cmd and "/m/mmproj.gguf" in cmd
    # extra flags are appended last and shlex-split
    assert cmd[-3:] == ["--temp", "0.7", "--flash-attn", "on"][-3:]
    assert "0.7" in cmd and "on" in cmd


def test_build_command_extra_flags_with_spaces():
    cmd = server.build_command(
        "s", "/m.gguf", 1, extra_flags='--path "C:/a b/model.gguf"'
    )
    # the quoted path stays a single token
    assert "C:/a b/model.gguf" in cmd


# --- find_free_port --------------------------------------------------------


def test_find_free_port_returns_bindable_port():
    port = server.find_free_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535
    # the returned port should still be bindable immediately after
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


# --- LLMServer lifecycle ---------------------------------------------------


def test_process_is_alive_and_close():
    # Process-level lifecycle (the object wrapping the real OS process).
    from conftest import fake_process

    proc = FakeProc()
    s = fake_process(proc=proc)
    assert s.is_alive()
    assert not s.closed
    s.close()
    assert s.closed
    assert not s.is_alive()
    # the child is reaped in both the tree-kill and graceful-kill paths
    assert proc.wait_called
    # idempotent
    s.close()
    assert not proc.killed


# --- kill_server -----------------------------------------------------------


def test_kill_server_tree_kill_path(fake_tree_kill):
    # Windows path: taskkill takes the whole tree, no graceful terminate.
    calls, _real = fake_tree_kill
    proc = FakeProc()
    server.kill_server(proc)
    assert calls == [proc.pid]
    assert not proc.terminated and not proc.killed
    assert proc.wait_called


def test_kill_server_fallback_terminate(fake_tree_kill, monkeypatch):
    # POSIX / taskkill unavailable: graceful terminate, no SIGKILL when the
    # process exits in time.
    calls, _real = fake_tree_kill
    monkeypatch.setattr(server, "_kill_process_tree", lambda pid: False)
    proc = FakeProc()
    server.kill_server(proc)
    assert calls == []
    assert proc.terminated and not proc.killed
    assert proc.wait_called


def test_kill_server_none_proc_is_noop(fake_tree_kill):
    server.kill_server(None)  # must not raise


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is Windows-only")
def test_kill_process_tree_kills_wrapper_children(fake_tree_kill, monkeypatch):
    # The real-world failure this guards: a wrapper/router (.bat/.cmd/shim,
    # llama-server router mode) spawns the actual model server as a child; a
    # plain proc.terminate() would kill only the wrapper and orphan the server
    # (and its VRAM). kill_server must take down the whole tree.
    _calls, real = fake_tree_kill
    monkeypatch.setattr(server, "_kill_process_tree", real)
    wrapper = subprocess.Popen(
        ["cmd", "/c", "ping", "-n", "300", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        server.kill_server(wrapper)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
    assert wrapper.poll() is not None
    children = subprocess.run(
        ["tasklist", "/FI", f"PARENTPID eq {wrapper.pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
    ).stdout
    assert children.strip() == "", f"orphaned children survived: {children}"


def test_handle_config_key():
    a = server.LLMServer("llama-server", "/a.gguf", 2048)
    b = server.LLMServer("llama-server", "/b.gguf", 2048)
    c = server.LLMServer("llama-server", "/a.gguf", 4096)
    d = server.LLMServer("llama-server", "/a.gguf", 2048, mmproj_path="/m/mm.gguf")
    assert a.config_key != b.config_key
    assert a.config_key != c.config_key
    assert a.config_key != d.config_key
    # same config -> same key
    assert a.config_key == server.LLMServer("llama-server", "/a.gguf", 2048).config_key


# --- wait_for_server -------------------------------------------------------


class _Resp:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_wait_for_server_healthy(monkeypatch):
    monkeypatch.setattr(server.request, "urlopen", lambda url, timeout=None: _Resp(200))
    assert (
        server.wait_for_server("http://127.0.0.1:9999", FakeProc(), timeout=5) is True
    )


def test_wait_for_server_dead_process_returns_false():
    # a process that has already exited returns False immediately (no polling)
    assert (
        server.wait_for_server("http://127.0.0.1:1", FakeProc(exit_code=1), timeout=5)
        is False
    )


# --- discover_model_name ---------------------------------------------------


def test_discover_model_name_prefers_loaded(monkeypatch):
    body = json.dumps(
        {
            "data": [
                {"id": "idle-model", "status": {"value": "unloaded"}},
                {"id": "active-model", "status": {"value": "loaded"}},
            ]
        }
    ).encode()

    def fake_urlopen(url, timeout=None):
        return _Resp(200, body)

    monkeypatch.setattr(server.request, "urlopen", fake_urlopen)
    assert server.discover_model_name("http://127.0.0.1:9999") == "active-model"


def test_discover_model_name_falls_back(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise error.URLError("down")

    monkeypatch.setattr(server.request, "urlopen", fake_urlopen)
    assert server.discover_model_name("http://127.0.0.1:9999") == "llama"


# --- start_server failure path --------------------------------------------


def test_start_server_returns_none_when_binary_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(server.subprocess, "Popen", boom)
    assert (
        server.start_server("/nonexistent/llama-server", "/m.gguf", 1234, 2048) is None
    )


# --- free_gpu_memory -------------------------------------------------------


def _fake_comfy(monkeypatch, calls):
    """Install a fake ``comfy`` package whose ``model_management`` records calls."""
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: calls.append("unload_all_models")
    mm.soft_empty_cache = lambda force=False: calls.append(("soft_empty_cache", force))
    mm.throw_exception_if_processing_interrupted = lambda: calls.append("interrupt")
    pkg = types.ModuleType("comfy")
    pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)


def _fake_torch(monkeypatch, cuda_available):
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return torch_mod


def test_free_gpu_memory_unloads_comfy_models(monkeypatch):
    # Regression: the local name bound for ``comfy`` must stay the module,
    # never a boolean flag - ``comfy.model_management`` crashed with
    # AttributeError: 'bool' object has no attribute 'model_management'.
    calls = []
    printed = []
    _fake_comfy(monkeypatch, calls)
    _fake_torch(monkeypatch, cuda_available=False)
    monkeypatch.setattr(server, "_print_safe", printed.append)
    server.free_gpu_memory()
    assert "unload_all_models" in calls
    assert ("soft_empty_cache", True) in calls
    assert any("Freeing GPU memory" in m for m in printed)
    assert any("GPU memory freed" in m for m in printed)


def test_free_gpu_memory_noop_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)  # force ImportError
    _fake_torch(monkeypatch, cuda_available=False)
    printed = []
    monkeypatch.setattr(server, "_print_safe", printed.append)
    server.free_gpu_memory()  # must not raise outside ComfyUI
    assert printed == []


def test_free_gpu_memory_still_clears_torch_cache_without_comfy(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "comfy", None)
    torch_mod = _fake_torch(monkeypatch, cuda_available=True)
    torch_mod.cuda.empty_cache = lambda: calls.append("empty_cache")
    torch_mod.cuda.ipc_collect = lambda: calls.append("ipc_collect")
    server.free_gpu_memory()
    assert calls == ["empty_cache", "ipc_collect"]


# --- server handle (lazy lifecycle) ----------------------------------------


def _patch_start(monkeypatch):
    """Patch the process spawn behind the handle. Returns (started, freed)."""
    from conftest import fake_process

    started = []
    freed = []

    def fake_start(
        server_path,
        model_path,
        port,
        ctx_size,
        mmproj_path="",
        extra_flags="",
        startup_timeout=120,
    ):
        p = fake_process(
            port=port,
            model_path=model_path,
            ctx_size=ctx_size,
            server_path=server_path,
            mmproj_path=mmproj_path,
            extra_flags=extra_flags,
            base_url=f"http://127.0.0.1:{port}",
        )
        started.append(p)
        return p

    monkeypatch.setattr(server, "start_server", fake_start)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: freed.append(1))
    return started, freed


def test_handle_starts_nothing_on_construction():
    # The whole point of the lazy handle: constructing one never touches the
    # model - the process starts only on first real use.
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    assert not h.is_alive()
    assert server._warm_servers == {}


def test_ensure_live_cold_starts(monkeypatch):
    started, freed = _patch_start(monkeypatch)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    h.ensure_live()
    assert len(started) == 1
    assert freed == [1]  # the cold path frees GPU memory
    assert h.is_alive()
    assert h.base_url == "http://127.0.0.1:12345"
    assert h.model_name == "fake-model"
    assert server._warm_servers[h.config_key] is h._live


def test_ensure_live_idempotent(monkeypatch):
    started, _ = _patch_start(monkeypatch)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    h.ensure_live()
    h.ensure_live()
    assert len(started) == 1


def test_base_url_property_lazy_starts(monkeypatch):
    # Consumers read base_url / model_name directly: the first read must
    # trigger the lazy start.
    started, _ = _patch_start(monkeypatch)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    assert h.base_url == "http://127.0.0.1:12345"
    assert len(started) == 1


def test_handle_warm_reuse_across_handles(monkeypatch):
    # Two handles with the same config (e.g. two Load nodes) share one
    # process: the second never pays the cold-start cost.
    started, freed = _patch_start(monkeypatch)
    a = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    a.ensure_live()
    b = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    b.ensure_live()
    assert len(started) == 1
    assert freed == [1]  # warm reuse never frees GPU memory
    assert b._live is a._live


def test_handle_distinct_configs(monkeypatch):
    started, _ = _patch_start(monkeypatch)
    a = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    b = server.LLMServer("llama-server", "/m/b.gguf", 2048)
    a.ensure_live()
    b.ensure_live()
    assert len(started) == 2
    assert a._live is not b._live


def test_close_server_then_ensure_live_restarts(monkeypatch):
    # After a Close, the handle cold-starts again on its next use.
    started, _ = _patch_start(monkeypatch)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    h.ensure_live()
    first = h._live
    server.close_server(h)
    assert not h.is_alive()
    assert server._warm_servers == {}
    h.ensure_live()
    assert len(started) == 2
    assert h.is_alive()
    assert h._live is not first


def test_ensure_live_restarts_dead_process(monkeypatch):
    # A process that dies mid-run is replaced on the next ensure_live: the
    # handle's contract is "give me a live server for this config".
    started, _ = _patch_start(monkeypatch)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    h.ensure_live()
    h._live.proc._exit = 1  # simulate a crash
    h.ensure_live()
    assert len(started) == 2
    assert h.is_alive()


def test_close_server_is_config_scoped(monkeypatch):
    # Same-config handles share one process: closing one frees the server
    # even if that handle's own reference is stale - and the other handle
    # cold-starts again on next use.
    started, _ = _patch_start(monkeypatch)
    a = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    a.ensure_live()
    b = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    b.ensure_live()  # shares a's process
    server.close_server(a)
    assert started[0].closed
    assert not b._live.is_alive()
    b.ensure_live()
    assert len(started) == 2
    assert b.is_alive()


def test_close_server_none_is_noop():
    server.close_server(None)  # must not raise


def test_close_all_servers_stops_everything(monkeypatch):
    started, _ = _patch_start(monkeypatch)
    a = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    b = server.LLMServer("llama-server", "/m/b.gguf", 2048)
    a.ensure_live()
    b.ensure_live()
    server.close_all_servers()
    assert all(p.closed for p in started)
    assert server._warm_servers == {}


def test_ensure_live_failed_start_raises(monkeypatch):
    # A failed cold start is a hard infrastructure error (never a silent
    # degrade): the handle stays dead and nothing is cached.
    monkeypatch.setattr(server, "start_server", lambda *a, **k: None)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    with pytest.raises(server.LLMServerStartError):
        h.ensure_live()
    assert not h.is_alive()
    assert server._warm_servers == {}


# --- interrupt handling -----------------------------------------------------


def test_check_interrupt_kills_server_before_raising(
    monkeypatch, make_fake_server, interrupting_comfy
):
    # An interrupt while a node is mid-run must kill the in-use server: the
    # Close node downstream of the interrupted node never runs, so without
    # this the llama-server would be left holding VRAM until session end.
    exc = interrupting_comfy
    srv = make_fake_server()
    with pytest.raises(exc):
        server.check_interrupt(srv)
    assert not srv.is_alive()


def test_check_interrupt_without_server_still_raises(interrupting_comfy):
    exc = interrupting_comfy
    with pytest.raises(exc):
        server.check_interrupt()


def test_check_interrupt_noop_without_comfy(monkeypatch, make_fake_server):
    monkeypatch.setitem(sys.modules, "comfy", None)  # force ImportError
    srv = make_fake_server()
    server.check_interrupt(srv)  # must not raise outside ComfyUI
    assert srv.is_alive()


def test_ensure_live_interrupt_kills_warm_server(monkeypatch, interrupting_comfy):
    # An interrupt during the warm-cache lookup must kill the warm server
    # before the exception propagates, and never reach the cold start.
    from conftest import fake_handle

    exc = interrupting_comfy
    h = fake_handle()  # warm process pre-seeded in the cache
    started = []
    monkeypatch.setattr(server, "start_server", lambda *a, **k: started.append(1))
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    freed = []
    monkeypatch.setattr(server, "free_gpu_memory", lambda: freed.append(1))
    with pytest.raises(exc):
        h.ensure_live()
    assert not h.is_alive()
    assert server._warm_servers == {}
    assert started == [] and freed == []  # interrupt before the cold start
