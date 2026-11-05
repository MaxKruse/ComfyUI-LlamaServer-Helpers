"""Tests for server.py: command building, port finding, lifecycle, warm cache."""

import json
import socket
from urllib import error

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
    assert proc.terminated
    # idempotent
    s.close()
    assert not proc.killed


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


def test_close_server_none_is_noop():
    server.close_server(None)  # must not raise


def test_get_or_start_returns_none_on_failed_cold_start(monkeypatch):
    monkeypatch.setattr(server, "start_server", lambda *a, **k: None)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    assert server.get_or_start_server("llama-server", "/m/a.gguf", 2048) is None
    server.close_all_servers()
