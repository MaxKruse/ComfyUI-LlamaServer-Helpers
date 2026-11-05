"""Tests for the node layer: path validators and node execute() behavior.

The LLM/HTTP logic is mocked out (chat_completion_interruptible / enhance_prompt
/ check_rules / close_server / get_or_start_server) so these tests exercise only
the nodes' own logic: bypass handling, output shapes, judgement reporting, the
Close node's close-then-abort behaviour, and error paths.
"""

import math
import os
import tempfile

import pytest

import node_base
import nodes
import server
from enhance import EnhanceResult
from judge import CheckResult


def _missing_model_path() -> str:
    """An absolute path that does not exist, on any platform."""
    return os.path.join(
        tempfile.gettempdir(), "definitely_missing_dir_xyz", "model.gguf"
    )


# --- path validators -------------------------------------------------------


def test_resolve_server_path_bare():
    assert node_base.resolve_server_path("llama-server") == "llama-server"
    assert node_base.resolve_server_path("  ") == "llama-server"


def test_resolve_server_path_absolute():
    assert node_base.resolve_server_path("C:/bin/llama-server") == "C:/bin/llama-server"


def test_resolve_server_path_relative_raises():
    with pytest.raises(ValueError):
        node_base.resolve_server_path("some/relative/llama-server")


def test_require_model_path_empty_raises():
    with pytest.raises(ValueError, match="required"):
        node_base.require_model_path("")


def test_require_model_path_relative_raises():
    with pytest.raises(ValueError, match="absolute"):
        node_base.require_model_path("relative/model.gguf")


def test_require_model_path_missing_raises():
    with pytest.raises(ValueError, match="does not exist"):
        node_base.require_model_path(_missing_model_path())


def test_require_model_path_valid(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"x")
    assert node_base.require_model_path(str(f)) == str(f.resolve())


def test_optional_absolute_path():
    assert node_base.optional_absolute_path("", "mmproj") == ""
    assert node_base.optional_absolute_path("C:/x/mm.gguf", "mmproj") == "C:/x/mm.gguf"
    with pytest.raises(ValueError):
        node_base.optional_absolute_path("rel/mm.gguf", "mmproj")


# --- collect_images --------------------------------------------------------


def test_collect_images_empty():
    assert node_base.collect_images(None) == []
    assert node_base.collect_images({}) == []


def test_collect_images_sorted_and_skips_none():
    t1, t2 = object(), object()
    got = node_base.collect_images(
        {"ref_image_1": t2, "ref_image_0": t1, "ref_image_2": None}
    )
    assert got == [t1, t2]


# --- LLMPrompt -------------------------------------------------------------


def test_llm_prompt_bypass_empty(monkeypatch, make_fake_server):
    srv = make_fake_server()
    out = nodes.LLMPrompt.execute(llm_server=srv, user_prompt="   ", system_prompt="")
    assert out[0] == ""
    assert out[1] is srv


def test_llm_prompt_returns_response(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes, "chat_completion_interruptible", lambda *a, **k: "the answer"
    )
    out = nodes.LLMPrompt.execute(
        llm_server=srv, user_prompt="what?", system_prompt="be brief", seed=3
    )
    assert out[0] == "the answer"
    assert out[1] is srv


def test_llm_prompt_none_response_raises(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(nodes, "chat_completion_interruptible", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        nodes.LLMPrompt.execute(llm_server=srv, user_prompt="what?", system_prompt="")


# --- PromptEnhancer --------------------------------------------------------


def test_enhancer_bypass_empty(monkeypatch, make_fake_server):
    srv = make_fake_server()
    out = nodes.PromptEnhancer.execute(llm_server=srv, prompt="")
    assert tuple(out) == ("", "", srv)


def test_enhancer_returns_enhanced(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes,
        "enhance_prompt",
        lambda *a, **k: EnhanceResult("enhanced prompt", "blurry"),
    )
    out = nodes.PromptEnhancer.execute(llm_server=srv, prompt="a girl")
    assert tuple(out) == ("enhanced prompt", "blurry", srv)


def test_enhancer_degrades_on_failure(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes, "enhance_prompt", lambda *a, **k: EnhanceResult(None, None)
    )
    out = nodes.PromptEnhancer.execute(llm_server=srv, prompt="original prompt")
    # degrades to passing the original prompt through unchanged
    assert tuple(out) == ("original prompt", "", srv)


def test_enhancer_collects_ref_images(monkeypatch, make_fake_server):
    srv = make_fake_server()
    t1 = object()
    captured = {}

    def fake(server, prompt, system, *, images=None, **k):
        captured["images"] = images
        return EnhanceResult("ok", "")

    monkeypatch.setattr(nodes, "enhance_prompt", fake)
    nodes.PromptEnhancer.execute(
        llm_server=srv,
        prompt="a girl",
        ref_images={"ref_image_0": t1, "ref_image_1": None},
    )
    assert captured["images"] == [t1]


# --- RulesChecker ----------------------------------------------------------


def test_rules_bypass_empty_text(monkeypatch, make_fake_server):
    srv = make_fake_server()
    out = nodes.RulesChecker.execute(llm_server=srv, text="", rules="some rule")
    assert tuple(out) == ("", "", srv)


def test_rules_bypass_empty_rules(monkeypatch, make_fake_server):
    srv = make_fake_server()
    out = nodes.RulesChecker.execute(llm_server=srv, text="some text", rules="")
    assert tuple(out) == ("some text", "", srv)


def test_rules_pass(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes, "check_rules", lambda *a, **k: CheckResult(True, "", ok=True)
    )
    out = nodes.RulesChecker.execute(llm_server=srv, text="text", rules="rule")
    assert tuple(out) == ("text", "", srv)


def test_rules_fail_reports_judgement(monkeypatch, make_fake_server):
    # The checker never aborts: a failed check is reported on the judgement
    # output, and the text still passes through.
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes, "check_rules", lambda *a, **k: CheckResult(False, "breaks rule", ok=True)
    )
    out = nodes.RulesChecker.execute(llm_server=srv, text="text", rules="rule")
    assert tuple(out) == ("text", "Rules check failed: breaks rule", srv)


def test_rules_no_verdict_reports_error(monkeypatch, make_fake_server):
    # An infra failure (no parseable verdict) is reported as a non-empty
    # judgement (fail-closed, not a pass) - the Close node is what aborts.
    srv = make_fake_server()
    monkeypatch.setattr(
        nodes, "check_rules", lambda *a, **k: CheckResult(False, "no verdict", ok=False)
    )
    out = nodes.RulesChecker.execute(llm_server=srv, text="text", rules="rule")
    assert tuple(out) == (
        "text",
        "Rules check error: could not obtain a verdict from the LLM: no verdict",
        srv,
    )


# --- CloseLLMServer --------------------------------------------------------


def test_close_server_called(monkeypatch, make_fake_server):
    srv = make_fake_server()
    closed = {}

    def fake_close(server):
        closed["srv"] = server

    monkeypatch.setattr(nodes, "close_server", fake_close)
    out = nodes.CloseLLMServer.execute(llm_server=srv)
    assert closed["srv"] is srv
    assert tuple(out) == ()


def test_close_ignores_blank_judgement(monkeypatch, make_fake_server):
    # An empty / whitespace-only judgement is no abort: just close the server.
    srv = make_fake_server()
    closed = {}
    monkeypatch.setattr(nodes, "close_server", lambda s: closed.update(srv=s))
    out = nodes.CloseLLMServer.execute(llm_server=srv, judgement="   ")
    assert closed["srv"] is srv
    assert tuple(out) == ()


def test_close_aborts_on_judgement(monkeypatch, make_fake_server):
    # A non-empty judgement (a failed check) aborts the workflow.
    srv = make_fake_server()
    monkeypatch.setattr(nodes, "close_server", lambda s: None)
    with pytest.raises(RuntimeError, match="Rules check failed: breaks rule"):
        nodes.CloseLLMServer.execute(
            llm_server=srv, judgement="Rules check failed: breaks rule"
        )


def test_close_closes_before_aborting(monkeypatch, make_fake_server):
    # The server must be released even when the close then aborts - that is
    # the whole point of moving the abort out of the checker.
    srv = make_fake_server()
    order = []
    monkeypatch.setattr(nodes, "close_server", lambda s: order.append("closed"))
    with pytest.raises(RuntimeError):
        nodes.CloseLLMServer.execute(llm_server=srv, judgement="Rules check failed: x")
    assert order == ["closed"]


# --- LoadLLMServer ---------------------------------------------------------


def test_load_returns_server(monkeypatch, make_fake_server, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    srv = make_fake_server()
    captured = {}

    def fake_get(
        server_path,
        model_path,
        ctx_size,
        mmproj_path="",
        extra_flags="",
        startup_timeout=120,
    ):
        captured.update(
            server_path=server_path, model_path=model_path, ctx_size=ctx_size
        )
        return srv

    monkeypatch.setattr(nodes, "get_or_start_server", fake_get)
    out = nodes.LoadLLMServer.execute(
        model_path=str(model),
        llama_server_path="llama-server",
        ctx_size=4096,
        mmproj_path="",
        extra_server_args="",
        startup_timeout=60,
    )
    assert out[0] is srv
    assert captured["ctx_size"] == 4096
    assert captured["server_path"] == "llama-server"


def test_load_missing_model_raises(monkeypatch, make_fake_server):
    with pytest.raises(ValueError, match="does not exist"):
        nodes.LoadLLMServer.execute(model_path=_missing_model_path(), ctx_size=2048)


def test_load_failed_start_raises(monkeypatch, make_fake_server, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr(nodes, "get_or_start_server", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Failed to start"):
        nodes.LoadLLMServer.execute(model_path=str(model), ctx_size=2048)


# --- scheduling & caching (output-node flags + fingerprints) ---------------


def test_close_is_output_node():
    # Close must be scheduled: the executor only runs output nodes (or their
    # upstream dependencies), and Close has no outputs of its own.
    assert nodes.CloseLLMServer.OUTPUT_NODE is True


def test_load_is_not_output_node():
    # Sanity: Load is a plain producer, not an output node.
    assert nodes.LoadLLMServer.OUTPUT_NODE is False


def test_close_fingerprint_is_nan():
    # Closing is a side effect: it must re-run every time, never be served from
    # the persistent output cache (NaN != NaN, so the cache always misses).
    assert math.isnan(nodes.CloseLLMServer.fingerprint_inputs())


def test_load_fingerprint_tracks_generation(monkeypatch):
    # Load's cache key is the server start-generation, so it changes exactly
    # when a server (re)starts and the output cache never serves a stale
    # (closed) connection.
    monkeypatch.setattr(server, "_server_start_generation", 7)
    assert nodes.LoadLLMServer.fingerprint_inputs() == 7
