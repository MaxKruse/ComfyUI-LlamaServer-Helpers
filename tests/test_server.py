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


def test_llmserver_is_alive_and_close(make_fake_server):
    proc = FakeProc()
    s = make_fake_server(proc=proc)
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


def test_llmserver_config_key_distinct():
    a = server.LLMServer("u", "m", None, 1, "/a.gguf", 2048, server_path="s")
    b = server.LLMServer("u", "m", None, 1, "/b.gguf", 2048, server_path="s")
    c = server.LLMServer("u", "m", None, 1, "/a.gguf", 4096, server_path="s")
    assert a.config_key != b.config_key
    assert a.config_key != c.config_key
    # same config -> same key
    d = server.LLMServer("u", "m", None, 9, "/a.gguf", 2048, server_path="s")
    assert a.config_key == d.config_key


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


# --- warm cache ------------------------------------------------------------


def _patch_start(monkeypatch, make):
    started = []

    def fake_start(
        server_path,
        model_path,
        port,
        ctx_size,
        mmproj_path="",
        extra_flags="",
        startup_timeout=120,
    ):
        s = make(
            port=port,
            model_path=model_path,
            ctx_size=ctx_size,
            server_path=server_path,
            base_url=f"http://127.0.0.1:{port}",
        )
        started.append(s)
        return s

    monkeypatch.setattr(server, "start_server", fake_start)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    return started


def test_warm_cache_reuses_same_config(monkeypatch, make_fake_server):
    started = _patch_start(monkeypatch, make_fake_server)
    a = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    b = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    assert a is b
    assert len(started) == 1
    server.close_all_servers()


def test_warm_cache_starts_new_when_dead(monkeypatch, make_fake_server):
    started = _patch_start(monkeypatch, make_fake_server)
    a = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    # simulate the process dying
    a.proc._exit = 1
    a._closed = False
    b = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    assert b is not a
    assert len(started) == 2
    server.close_all_servers()


def test_warm_cache_distinct_configs(monkeypatch, make_fake_server):
    started = _patch_start(monkeypatch, make_fake_server)
    a = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    b = server.get_or_start_server("llama-server", "/m/b.gguf", 2048)
    assert a is not b
    assert len(started) == 2
    server.close_all_servers()


def test_close_server_removes_from_cache(monkeypatch, make_fake_server):
    _patch_start(monkeypatch, make_fake_server)
    a = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    assert a in server._warm_servers.values()
    server.close_server(a)
    assert a not in server._warm_servers.values()
    assert a.closed
    server.close_all_servers()


def test_close_server_pops_duplicate_wrapper(make_fake_server):
    # A second wrapper object around the same process must still drop the
    # cache entry when closed (duplicate objects can flow through the graph).
    proc = FakeProc()
    original = make_fake_server(proc=proc)
    server._warm_servers[original.config_key] = original
    duplicate = make_fake_server(proc=proc)
    assert duplicate is not original
    server.close_server(duplicate)
    assert original.config_key not in server._warm_servers
    assert duplicate.closed and not original.closed


def test_close_server_keeps_newer_server_for_same_config(make_fake_server):
    # A stale object must not evict a newer server (different PID) that is
    # currently cached for the same config.
    stale = make_fake_server(proc=FakeProc(pid=1111))
    fresh = make_fake_server(proc=FakeProc(pid=2222))
    server._warm_servers[fresh.config_key] = fresh
    server.close_server(stale)
    assert fresh.config_key in server._warm_servers
    assert not fresh.closed


def test_close_server_none_is_noop():
    server.close_server(None)  # must not raise


def test_get_or_start_returns_none_on_failed_cold_start(monkeypatch):
    monkeypatch.setattr(server, "start_server", lambda *a, **k: None)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    assert server.get_or_start_server("llama-server", "/m/a.gguf", 2048) is None
    server.close_all_servers()


# --- start generation (Load node cache fingerprint) ------------------------


def test_generation_increments_on_cold_start(monkeypatch, make_fake_server):
    started = _patch_start(monkeypatch, make_fake_server)
    before = server.server_generation()
    server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    assert len(started) == 1
    assert server.server_generation() == before + 1
    server.close_all_servers()


def test_generation_unchanged_on_warm_reuse(monkeypatch, make_fake_server):
    started = _patch_start(monkeypatch, make_fake_server)
    server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    after_start = server.server_generation()
    server.get_or_start_server("llama-server", "/m/a.gguf", 2048)  # warm reuse
    assert len(started) == 1
    assert server.server_generation() == after_start
    server.close_all_servers()


def test_generation_increments_on_restart_after_close(monkeypatch, make_fake_server):
    # The regression the fingerprint guards: after a Close tears the server
    # down, the next start must bump the generation so the Load node's cache
    # misses and a fresh (live) server is emitted, not the stale closed one.
    started = _patch_start(monkeypatch, make_fake_server)
    a = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    server.close_server(a)
    after_close = server.server_generation()
    b = server.get_or_start_server("llama-server", "/m/a.gguf", 2048)
    assert b is not a
    assert len(started) == 2
    assert server.server_generation() == after_close + 1
    server.close_all_servers()
