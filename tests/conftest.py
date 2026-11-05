"""Shared test setup.

Puts the ComfyUI root (for ``comfy_api``) and this package root (for the
``server`` / ``client`` / ``enhance`` / ``judge`` / ``node_base`` / ``nodes``
modules) on sys.path so the modules import directly without the hyphenated
package name. Also provides a fake process and a fake LLMServer connection for
the node and cache tests.
"""

import os
import sys

NODE_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = os.path.dirname(os.path.dirname(NODE_PACKAGE_ROOT))

for _p in (COMFYUI_ROOT, NODE_PACKAGE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class FakeProc:
    """Just enough of subprocess.Popen to exercise the lifecycle code."""

    def __init__(self, exit_code=None):
        self._exit = exit_code
        self.terminated = False
        self.killed = False
        self.pid = 4242

    def poll(self):
        return self._exit

    def terminate(self):
        self.terminated = True
        self._exit = 0

    def kill(self):
        self.killed = True
        self._exit = -9

    def wait(self, timeout=None):
        return self._exit


def _make_fake_server(proc=None, **overrides):
    from server import LLMServer

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
    return LLMServer(**defaults)


def _clear_warm_cache():
    import server

    server._warm_servers.clear()


import pytest


@pytest.fixture
def fake_server():
    """A live fake LLMServer, and cleanup of the warm cache around the test."""
    _clear_warm_cache()
    yield _make_fake_server()
    _clear_warm_cache()


@pytest.fixture
def make_fake_server():
    _clear_warm_cache()
    yield _make_fake_server
    _clear_warm_cache()
