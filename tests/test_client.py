"""Tests for client.py: payload building and chat completion handling."""

import json
import time
from urllib import error

import pytest

import client


# --- _payload --------------------------------------------------------------


def _srv(**kw):
    from server import LLMServer
    from conftest import FakeProc

    d = dict(
        base_url="http://127.0.0.1:9999",
        model_name="m",
        proc=FakeProc(),
        port=9999,
        model_path="/m.gguf",
        ctx_size=1024,
        server_path="s",
    )
    d.update(kw)
    return LLMServer(**d)


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
