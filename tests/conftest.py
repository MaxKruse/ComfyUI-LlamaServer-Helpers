"""Shared test setup.

Puts the ComfyUI root (for ``comfy_api``) and this package root (for the
``server`` / ``client`` / ``enhance`` / ``judge`` / ``node_base`` / ``nodes``
modules) on sys.path so the modules import directly without the hyphenated
package name. Also provides a fake process and fake server handles for the
node and cache tests.
"""

import os
import sys
import types

NODE_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = os.path.dirname(os.path.dirname(NODE_PACKAGE_ROOT))

for _p in (COMFYUI_ROOT, NODE_PACKAGE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class FakeProc:
    """Just enough of subprocess.Popen to exercise the lifecycle code."""

    def __init__(self, exit_code=None, pid=4242):
        self._exit = exit_code
        self.terminated = False
        self.killed = False
        self.wait_called = False
        self.pid = pid

    def poll(self):
        return self._exit

    def terminate(self):
        self.terminated = True
        self._exit = 0

    def kill(self):
        self.killed = True
        self._exit = -9

    def wait(self, timeout=None):
        self.wait_called = True
        return self._exit


def fake_process(proc=None, **overrides):
    """A fake live server process (the object ``start_server`` returns)."""
    from server import LLMServerProcess

    defaults = dict(
        base_url="http://127.0.0.1:9999",
        model_name="fake-model",
        proc=proc if proc is not None else FakeProc(),
        port=9999,
        model_path="/tmp/fake-model.gguf",
        ctx_size=2048,
        server_path="llama-server",
        mmproj_path="",
        extra_flags="",
    )
    defaults.update(overrides)
    return LLMServerProcess(**defaults)


def fake_handle(proc=None, **overrides):
    """A server handle with a pre-seeded warm process (the common test shape).

    ``overrides`` apply to the process (``model_name``, ``proc``, ...); the
    config fields (``server_path`` / ``model_path`` / ``ctx_size`` / ...)
    propagate into the handle. Returns the handle.
    """
    import server

    process = fake_process(proc=proc, **overrides)
    handle = server.LLMServer(
        process.server_path,
        process.model_path,
        process.ctx_size,
        mmproj_path=process.mmproj_path,
        extra_flags=process.extra_flags,
    )
    server._warm_servers[handle.config_key] = process
    handle._live = process
    return handle


def _clear_warm_cache():
    import server

    server._warm_servers.clear()


import pytest


@pytest.fixture(autouse=True)
def clean_warm_cache():
    """Every test starts and ends with an empty warm cache."""
    _clear_warm_cache()
    yield
    _clear_warm_cache()


@pytest.fixture(autouse=True)
def fake_tree_kill(monkeypatch):
    """Keep kill_server's taskkill path hermetic.

    kill_server force-kills by PID and FakeProc PIDs are arbitrary - they may
    collide with live host processes, so the real taskkill must never run in
    tests. Yields (calls, real_fn); a test that wants the real process-tree
    kill restores real_fn via monkeypatch.
    """
    import server

    real = server._kill_process_tree
    calls = []

    def _fake(pid):
        calls.append(pid)
        return True

    monkeypatch.setattr(server, "_kill_process_tree", _fake)
    yield calls, real


class InterruptProcessingException(BaseException):
    """Stand-in for ComfyUI's interrupt exception (a BaseException, not Exception)."""


@pytest.fixture
def interrupting_comfy(monkeypatch):
    """Install a fake ``comfy`` package whose interrupt check always raises.

    Returns the exception class so tests can assert on it.
    """
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: (_ for _ in ()).throw(
        InterruptProcessingException()
    )
    pkg = types.ModuleType("comfy")
    pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return InterruptProcessingException


@pytest.fixture
def fake_server():
    """A live fake server handle (warm process pre-seeded)."""
    return fake_handle()


@pytest.fixture
def make_fake_server():
    """Factory for live fake server handles (kept for the long-standing name)."""
    return fake_handle
