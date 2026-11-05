"""Tests for the model-specific enhancer nodes and the ImageRulesChecker node.

The LLM logic is mocked out (enhance_prompt / check_rules) so these tests
exercise only the node layer: the per-model defaults wired from the preset
specs, bypass handling, output shapes, judgement reporting, and the
ImageRulesChecker gate semantics.
"""

from pathlib import Path

import model_nodes
from enhance import EnhanceResult
from judge import CheckResult
from presets import ENHANCER_SPECS

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

# Model dialects mirrored verbatim from ../ComfyUI-PromptEnhancer: node_id ->
# preset filename. These six were missing from the first mirror pass.
MIRRORED_PRESETS = {
    "MossImagePromptEnhancer": "moss-image-sft-t2i.txt",
    "QwenImageEditPromptEnhancer": "qwenimage2.1-i2i.txt",
    "Ltx23PromptEnhancer": "ltx2.3-10eros-i2v.txt",
    "Ltx25PromptEnhancer": "ltx2.5-i2v.txt",
    "MiniMaxH3R2VPromptEnhancer": "minimax-h3-r2v.txt",
    "MiniMaxH3ErosMaxPromptEnhancer": "minimax-h3-eros-max-r2v.txt",
}


def _spec(node_id: str):
    return next(s for s in ENHANCER_SPECS if s.node_id == node_id)


def _boom(*args, **kwargs):
    raise AssertionError("LLM call should not happen on this path")


# --- preset specs -----------------------------------------------------------


def test_specs_unique_ids_and_complete_fields():
    ids = [s.node_id for s in ENHANCER_SPECS]
    assert len(ids) == len(set(ids))
    for s in ENHANCER_SPECS:
        assert s.node_id
        assert s.display_name
        assert s.description
        assert len(s.system_prompt) >= 100  # a real dialect prompt, not a stub
        assert 1 <= s.default_min_words <= 500


def test_no_spec_defaults_generate_negative():
    neg = {s.node_id for s in ENHANCER_SPECS if s.default_generate_negative}
    assert neg == set()


def test_video_specs_use_video_category():
    video = {s.node_id for s in ENHANCER_SPECS if s.category == "LlamaServer/Video"}
    assert video == {
        "MiniMaxH3PromptEnhancer",
        "Wan22PromptEnhancer",
        "Ltx23PromptEnhancer",
        "Ltx25PromptEnhancer",
        "MiniMaxH3R2VPromptEnhancer",
        "MiniMaxH3ErosMaxPromptEnhancer",
    }


def test_mirrored_preset_specs_present():
    ids = {s.node_id for s in ENHANCER_SPECS}
    for node_id in MIRRORED_PRESETS:
        assert node_id in ids, f"{node_id} missing from ENHANCER_SPECS"


def test_mirrored_preset_systems_match_preset_files():
    for node_id, filename in MIRRORED_PRESETS.items():
        spec = _spec(node_id)
        text = (PRESETS_DIR / filename).read_text(encoding="utf-8").strip()
        assert spec.system_prompt == text, (
            f"{node_id} system prompt differs from {filename}"
        )


def test_mirrored_preset_categories_and_defaults():
    expected = {
        "MossImagePromptEnhancer": ("LlamaServer/LLM", 40),
        "QwenImageEditPromptEnhancer": ("LlamaServer/LLM", 25),
        "Ltx23PromptEnhancer": ("LlamaServer/Video", 60),
        "Ltx25PromptEnhancer": ("LlamaServer/Video", 40),
        "MiniMaxH3R2VPromptEnhancer": ("LlamaServer/Video", 50),
        "MiniMaxH3ErosMaxPromptEnhancer": ("LlamaServer/Video", 50),
    }
    for node_id, (category, min_words) in expected.items():
        spec = _spec(node_id)
        assert spec.category == category
        assert spec.default_min_words == min_words
        assert spec.default_generate_negative is False


# --- registration ------------------------------------------------------------


def test_all_specs_registered_in_mappings():
    for s in ENHANCER_SPECS:
        assert s.node_id in model_nodes.MODEL_ENHANCER_NODES
        assert s.node_id in model_nodes.NODE_CLASS_MAPPINGS
        assert model_nodes.NODE_DISPLAY_NAME_MAPPINGS[s.node_id] == s.display_name
    assert "ImageRulesChecker" in model_nodes.NODE_CLASS_MAPPINGS


def test_generated_node_is_comfy_node_subclass():
    from comfy_api.latest import io

    for s in ENHANCER_SPECS:
        node_cls = model_nodes.MODEL_ENHANCER_NODES[s.node_id]
        assert issubclass(node_cls, io.ComfyNode)
        assert node_cls.__name__ == s.node_id


# --- per-model enhancer behaviour ---------------------------------------------


def _record_fake_enhance(captured):
    def fake(server, prompt, system, **kw):
        captured["system"] = system
        captured.update(kw)
        return EnhanceResult("enhanced", None)

    return fake


def test_model_node_bypass_empty_prompt(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(model_nodes, "enhance_prompt", _boom)
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    out = node.execute(llm_server=srv, prompt="   ")
    assert tuple(out) == ("", "", srv)


def test_model_node_uses_spec_system_by_default(monkeypatch, make_fake_server):
    srv = make_fake_server()
    captured = {}
    monkeypatch.setattr(model_nodes, "enhance_prompt", _record_fake_enhance(captured))
    spec = _spec("FluxPromptEnhancer")
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    # cleared system_prompt widget falls back to the model dialect
    out = node.execute(llm_server=srv, prompt="a cat", system_prompt="")
    assert captured["system"] == spec.system_prompt
    assert tuple(out) == ("enhanced", "", srv)


def test_model_node_custom_system_overrides_spec(monkeypatch, make_fake_server):
    srv = make_fake_server()
    captured = {}
    monkeypatch.setattr(model_nodes, "enhance_prompt", _record_fake_enhance(captured))
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    node.execute(llm_server=srv, prompt="a cat", system_prompt="my custom instructions")
    assert captured["system"] == "my custom instructions"


def test_model_node_min_words_and_negative_defaults(monkeypatch, make_fake_server):
    srv = make_fake_server()
    captured = {}
    monkeypatch.setattr(model_nodes, "enhance_prompt", _record_fake_enhance(captured))
    model_nodes.MODEL_ENHANCER_NODES["Ltx23PromptEnhancer"].execute(
        llm_server=srv, prompt="a cat"
    )
    assert captured["min_words"] == 60
    assert captured["generate_negative"] is False

    model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"].execute(
        llm_server=srv, prompt="a cat"
    )
    assert captured["min_words"] == 25
    assert captured["generate_negative"] is False


def test_model_node_returns_enhanced(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        model_nodes,
        "enhance_prompt",
        lambda *a, **k: EnhanceResult("enhanced", "blurry"),
    )
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    out = node.execute(llm_server=srv, prompt="a cat")
    assert tuple(out) == ("enhanced", "blurry", srv)


def test_model_node_degrades_on_failure(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        model_nodes, "enhance_prompt", lambda *a, **k: EnhanceResult(None, None)
    )
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    out = node.execute(llm_server=srv, prompt="original prompt")
    assert tuple(out) == ("original prompt", "", srv)


def test_model_node_forwards_ref_images(monkeypatch, make_fake_server):
    srv = make_fake_server()
    t1 = object()
    captured = {}
    monkeypatch.setattr(model_nodes, "enhance_prompt", _record_fake_enhance(captured))
    node = model_nodes.MODEL_ENHANCER_NODES["FluxPromptEnhancer"]
    out = node.execute(llm_server=srv, prompt="a cat", ref_images={"ref_image_0": t1})
    assert captured["images"] == [t1]
    assert tuple(out) == ("enhanced", "", srv)


# --- ImageRulesChecker --------------------------------------------------------


def test_image_qa_bypass_no_images(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(model_nodes, "check_rules", _boom)
    out = model_nodes.ImageRulesChecker.execute(llm_server=srv, rules="no watermark")
    assert tuple(out) == ("", srv)


def test_image_qa_bypass_empty_rules(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(model_nodes, "check_rules", _boom)
    out = model_nodes.ImageRulesChecker.execute(
        llm_server=srv, images={"image_0": object()}, rules="  "
    )
    assert tuple(out) == ("", srv)


def test_image_qa_pass(monkeypatch, make_fake_server):
    srv = make_fake_server()
    monkeypatch.setattr(
        model_nodes, "check_rules", lambda *a, **k: CheckResult(True, "")
    )
    out = model_nodes.ImageRulesChecker.execute(
        llm_server=srv, images={"image_0": object()}, rules="no watermark"
    )
    assert tuple(out) == ("", srv)


def test_image_qa_fail_reports_judgement(monkeypatch, make_fake_server):
    # The checker never aborts: a failed check is reported on the judgement
    # output; the Close node is what aborts.
    srv = make_fake_server()
    monkeypatch.setattr(
        model_nodes,
        "check_rules",
        lambda *a, **k: CheckResult(False, "watermark visible"),
    )
    out = model_nodes.ImageRulesChecker.execute(
        llm_server=srv, images={"image_0": object()}, rules="no watermark"
    )
    assert tuple(out) == ("Image rules check failed: watermark visible", srv)


def test_image_qa_no_verdict_reports_error(monkeypatch, make_fake_server):
    # An infra failure (no parseable verdict) is reported as a non-empty
    # judgement (fail-closed, not a pass) and still mentions the mmproj.
    srv = make_fake_server()
    monkeypatch.setattr(
        model_nodes,
        "check_rules",
        lambda *a, **k: CheckResult(False, "no valid verdict", ok=False),
    )
    out = model_nodes.ImageRulesChecker.execute(
        llm_server=srv, images={"image_0": object()}, rules="no watermark"
    )
    judgement, out_srv = tuple(out)
    assert judgement.startswith("Image rules check error: could not obtain a verdict")
    assert "mmproj" in judgement
    assert out_srv is srv


def test_image_qa_forwards_images(monkeypatch, make_fake_server):
    srv = make_fake_server()
    img = object()
    captured = {}

    def fake(server, text, rules, **kw):
        captured["images"] = kw.get("images")
        captured["text"] = text
        captured["rules"] = rules
        return CheckResult(True, "")

    monkeypatch.setattr(model_nodes, "check_rules", fake)
    model_nodes.ImageRulesChecker.execute(
        llm_server=srv, images={"image_1": img}, rules="no watermark"
    )
    assert captured["images"] == [img]
    assert captured["text"] == ""
    assert captured["rules"] == "no watermark"


def test_image_checker_is_output_node():
    # The gate must be scheduled even when its outputs are left unconnected:
    # the executor only runs output nodes (or their upstream dependencies).
    assert model_nodes.ImageRulesChecker.OUTPUT_NODE is True


def test_image_checker_fingerprint_is_nan():
    # Gating is a side effect: it must re-run every time, never be served from
    # the persistent output cache (a cached no-op gate would silently pass).
    fp = model_nodes.ImageRulesChecker.fingerprint_inputs()
    assert fp != fp  # NaN
