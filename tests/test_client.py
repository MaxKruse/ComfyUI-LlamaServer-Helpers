"""Tests for client.py: payload building and chat completion handling."""

import json
import time
from urllib import error

import pytest

import client


# --- _payload --------------------------------------------------------------


def _srv(**kw):
    # A handle with a pre-seeded warm process: chat's ensure_live hits the
    # warm cache and starts nothing.
    from conftest import fake_handle

    kw.setdefault("model_name", "m")
    return fake_handle(**kw)


def test_payload_no_sampling_params():
    p = client._payload(_srv(), "you are x", "hello", 42)
    assert p["model"] == "m"
    assert p["seed"] == 42
    assert p["messages"] == [
        {"role": "system", "content": "you are x"},
        {"role": "user", "content": "hello"},
    ]
    assert "temperature" not in p
    assert "max_tokens" not in p


def test_payload_omits_system_when_empty():
    p = client._payload(_srv(), "", "hello", 1)
    assert p["messages"] == [{"role": "user", "content": "hello"}]


def test_payload_includes_sampling_params_when_set():
    p = client._payload(_srv(), "s", "u", 1, temperature=0.7, max_tokens=128)
    assert p["temperature"] == 0.7
    assert p["max_tokens"] == 128


# --- build_user_content ----------------------------------------------------


def test_build_user_content_no_images():
    assert client.build_user_content("hello", None) == "hello"
    assert client.build_user_content("hello", []) == "hello"


def test_build_user_content_with_images(monkeypatch):
    monkeypatch.setattr(
        client, "_images_to_base64_jpegs", lambda imgs: ["b64one", "b64two"]
    )
    content = client.build_user_content("look at these", [object(), object()])
    assert isinstance(content, list)
    # first part is the text, then pairs of (label, image) per image
    assert content[0] == {"type": "text", "text": "look at these"}
    image_urls = [
        c["image_url"]["url"] for c in content if c.get("type") == "image_url"
    ]
    assert image_urls == [
        "data:image/jpeg;base64,b64one",
        "data:image/jpeg;base64,b64two",
    ]


# --- _is_context_error -----------------------------------------------------


def test_is_context_error():
    assert client._is_context_error("prompt out of context window")
    assert client._is_context_error("sequence length exceeds context limit")
    assert not client._is_context_error("server exploded")


# --- chat_completion -------------------------------------------------------


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, payload=None, exc=None):
    calls = {}

    def fake_urlopen(req, timeout=None):
        calls["req"] = req
        calls["timeout"] = timeout
        if exc is not None:
            raise exc
        return _Resp(payload)

    monkeypatch.setattr(client.request, "urlopen", fake_urlopen)
    return calls


def test_chat_completion_success(monkeypatch):
    _patch_urlopen(
        monkeypatch,
        payload={
            "choices": [{"message": {"content": "  the response  "}}],
            "timings": {
                "prompt_n": 10,
                "cache_n": 0,
                "predicted_n": 20,
                "prompt_per_second": 50.0,
                "predicted_per_second": 30.0,
            },
        },
    )
    out = client.chat_completion(
        _srv(), "sys", "user", 7, temperature=0.5, max_tokens=64
    )
    assert out == "the response"


def test_chat_completion_list_content_joined(monkeypatch):
    _patch_urlopen(
        monkeypatch,
        payload={
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "a "},
                            {"type": "text", "text": "b"},
                        ]
                    }
                }
            ],
        },
    )
    out = client.chat_completion(_srv(), "", "user", 1)
    assert out == "a b"


def test_chat_completion_http_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise error.HTTPError("url", 500, "server error", {}, None)

    monkeypatch.setattr(client.request, "urlopen", boom)
    assert client.chat_completion(_srv(), "s", "u", 1) is None


def test_chat_completion_url_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        client.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(error.URLError("down")),
    )
    assert client.chat_completion(_srv(), "s", "u", 1) is None


def test_chat_completion_sends_auth_and_model(monkeypatch):
    calls = _patch_urlopen(
        monkeypatch, payload={"choices": [{"message": {"content": "ok"}}]}
    )
    client.chat_completion(_srv(model_name="my-model"), "s", "u", 9)
    sent = json.loads(calls["req"].data.decode("utf-8"))
    assert sent["model"] == "my-model"
    assert calls["req"].get_header("Authorization") == "Bearer no-key"
    assert calls["timeout"] == client._REQUEST_TIMEOUT


# --- lazy start (the handle may not have a process yet) ---------------------


def _patch_lazy_start(monkeypatch):
    """Patch the process spawn behind ensure_live. Returns the started list."""
    import server
    from conftest import fake_process

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
        p = fake_process(
            port=port,
            model_path=model_path,
            ctx_size=ctx_size,
            server_path=server_path,
            base_url=f"http://127.0.0.1:{port}",
        )
        started.append(p)
        return p

    monkeypatch.setattr(server, "start_server", fake_start)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    return started


def test_chat_completion_lazy_starts_server(monkeypatch):
    # A handle straight from Load has no live process: the first chat must
    # cold-start the server and then talk to it.
    import server

    started = _patch_lazy_start(monkeypatch)
    _patch_urlopen(monkeypatch, payload={"choices": [{"message": {"content": "hi"}}]})
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    out = client.chat_completion(h, "s", "u", 1)
    assert out == "hi"
    assert len(started) == 1
    assert h.is_alive()


def test_chat_completion_start_failure_raises(monkeypatch):
    # A failed cold start (missing binary, model too big) is a hard
    # infrastructure error: it must surface, not return a silent None.
    import server

    monkeypatch.setattr(server, "start_server", lambda *a, **k: None)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    with pytest.raises(server.LLMServerStartError):
        client.chat_completion(h, "s", "u", 1)


# --- chat_completion_interruptible -----------------------------------------


def test_chat_completion_interruptible_kills_server_on_interrupt(
    monkeypatch, make_fake_server, interrupting_comfy
):
    # A long generation interrupted by the user must kill the server, not just
    # abandon it (killing it also unblocks the in-flight HTTP request).
    exc = interrupting_comfy
    srv = make_fake_server()

    def slow(*a, **k):
        time.sleep(5)
        return "done"

    monkeypatch.setattr(client, "chat_completion", slow)
    with pytest.raises(exc):
        client.chat_completion_interruptible(srv, "s", "u", 1)
    assert not srv.is_alive()


def test_chat_completion_interruptible_interrupt_during_start(
    monkeypatch, interrupting_comfy
):
    # An interrupt while the handle resolves a live process must propagate
    # as an interrupt (the run aborts), not become a silent None response.
    import server

    exc = interrupting_comfy
    monkeypatch.setattr(server, "start_server", lambda *a, **k: None)
    monkeypatch.setattr(server, "find_free_port", lambda: 12345)
    monkeypatch.setattr(server, "free_gpu_memory", lambda: None)
    h = server.LLMServer("llama-server", "/m/a.gguf", 2048)
    with pytest.raises(exc):
        client.chat_completion_interruptible(h, "s", "u", 1)


# --- live probe (the machine's llama-server on :8080, when up) --------------

_LIVE_URL = "http://127.0.0.1:8080"


class _LiveProc:
    """A stand-in subprocess for a server we do not own: always alive.

    The server on :8080 belongs to the machine (omp's own runtime): the
    lifecycle code must see it as running, and the test must never try to
    stop it.
    """

    pid = None

    def poll(self):
        return None


def test_live_server_chat_completion():
    # Probe the machine's real llama-server: when it is up, run one actual
    # completion through the full client path (payload, HTTP, parse). Skip
    # when it is down. The test never closes the server: the warm-cache
    # teardown only clears the registry.
    from urllib import request

    from conftest import fake_handle

    try:
        with request.urlopen(_LIVE_URL + "/v1/models", timeout=5) as r:
            model = json.loads(r.read())["data"][0]["id"]
    except Exception as e:
        pytest.skip(f"no llama-server at {_LIVE_URL}: {e}")

    srv = fake_handle(proc=_LiveProc(), base_url=_LIVE_URL, model_name=model, port=8080)
    # 256 tokens: the loaded model may be a reasoning model whose
    # reasoning_content eats a small budget before the visible answer.
    out = client.chat_completion(
        srv,
        "You are a helpful assistant.",
        "Reply with exactly one word: pong",
        42,
        max_tokens=256,
    )
    assert isinstance(out, str)
