"""Tests for judge.py: verdict parsing and the rules-check retry loop."""

import pytest

import judge


# --- _coerce_bool ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("False", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
        (1.0, True),
        (None, None),
        ("maybe", None),
    ],
)
def test_coerce_bool(raw, expected):
    assert judge._coerce_bool(raw) is expected


# --- _parse_check_response -------------------------------------------------


def test_parse_pass():
    r = judge._parse_check_response('{"pass": true, "reason": ""}')
    assert r == judge.CheckResult(True, "", ok=True)


def test_parse_fail_with_reason():
    r = judge._parse_check_response('{"pass": false, "reason": "adds a second person"}')
    assert r.passed is False
    assert r.reason == "adds a second person"
    assert r.ok is True


def test_parse_reason_ignored_on_pass():
    r = judge._parse_check_response('{"pass": true, "reason": "should be cleared"}')
    assert r.passed is True
    assert r.reason == ""


def test_parse_string_pass_value():
    assert judge._parse_check_response('{"pass": "true"}').passed is True
    assert judge._parse_check_response('{"pass": "false"}').passed is False


def test_parse_invalid_returns_not_ok():
    assert judge._parse_check_response(None) == judge.CheckResult(False, "", ok=False)
    assert judge._parse_check_response("not json") == judge.CheckResult(
        False, "", ok=False
    )
    assert judge._parse_check_response('{"nope": 1}') == judge.CheckResult(
        False, "", ok=False
    )
    assert judge._parse_check_response('{"pass": "maybe"}') == judge.CheckResult(
        False, "", ok=False
    )


def test_parse_fenced_json():
    r = judge._parse_check_response(
        '```json\n{"pass": false, "reason": "too short"}\n```'
    )
    assert r.passed is False
    assert r.reason == "too short"
    assert r.ok is True


# --- check_rules (retry loop) ----------------------------------------------


def _srv(make_fake_server):
    return make_fake_server()


def _mock_chat(monkeypatch, side_effects):
    it = iter(side_effects)
    calls = []

    def fake(server, system, user, seed, images=None, temperature=None):
        calls.append(
            {
                "system": system,
                "user": user,
                "seed": seed,
                "images": images,
                "temperature": temperature,
            }
        )
        return next(it)

    monkeypatch.setattr(judge, "chat_completion_interruptible", fake)
    return calls


def test_check_rules_pass(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, ['{"pass": true, "reason": ""}'])
    r = judge.check_rules(_srv(make_fake_server), "some text", "RULE", seed=5)
    assert r.ok is True and r.passed is True and r.reason == ""
    assert len(calls) == 1
    # rules go in the system prompt, text in the user message
    assert "RULE" in calls[0]["system"]
    assert "some text" in calls[0]["user"]


def test_check_rules_fail(monkeypatch, make_fake_server):
    _mock_chat(monkeypatch, ['{"pass": false, "reason": "breaks rule 2"}'])
    r = judge.check_rules(_srv(make_fake_server), "text", "RULE", seed=5)
    assert r.passed is False
    assert r.reason == "breaks rule 2"
    assert r.ok is True


def test_check_rules_empty_text_vacuous_pass(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, [])
    r = judge.check_rules(_srv(make_fake_server), "   ", "RULE", seed=5)
    assert r == judge.CheckResult(True, "", ok=True)
    assert len(calls) == 0  # no LLM call for empty text


def test_check_rules_retries_then_valid(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, ["garbage", '{"pass": true, "reason": ""}'])
    r = judge.check_rules(_srv(make_fake_server), "text", "RULE", seed=1)
    assert r.ok is True and r.passed is True
    assert len(calls) == 2


def test_check_rules_no_verdict_returns_not_ok(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, [None, "garbage"])
    r = judge.check_rules(_srv(make_fake_server), "text", "RULE", seed=1, max_retries=2)
    assert r.ok is False
    assert r.passed is False
    assert len(calls) == 2


# --- image (vision) check path ---------------------------------------------


def test_judge_image_system_prompt_mentions_rules_and_contract():
    p = judge._judge_image_system_prompt("no text in the image")
    assert "no text in the image" in p
    assert '"pass"' in p
    assert "IMAGES" in p


def test_check_rules_with_images_uses_vision_prompt(monkeypatch, make_fake_server):
    img = object()
    calls = _mock_chat(monkeypatch, ['{"pass": true, "reason": ""}'])
    r = judge.check_rules(_srv(make_fake_server), "", "no watermark", images=[img])
    assert r.ok is True and r.passed is True
    assert len(calls) == 1
    # rules in the (vision) system prompt, image forwarded to the chat call
    assert "no watermark" in calls[0]["system"]
    assert "IMAGES" in calls[0]["system"]
    assert calls[0]["images"] == [img]


def test_check_rules_with_images_and_text_sends_both(monkeypatch, make_fake_server):
    img = object()
    calls = _mock_chat(
        monkeypatch, ['{"pass": false, "reason": "caption does not match the image"}']
    )
    r = judge.check_rules(
        _srv(make_fake_server),
        "a red car",
        "caption must match the image",
        images=[img],
    )
    assert r.ok is True and r.passed is False
    assert r.reason == "caption does not match the image"
    # vision prompt, related text in the user message, image forwarded
    assert "IMAGES" in calls[0]["system"]
    assert "a red car" in calls[0]["user"]
    assert calls[0]["images"] == [img]


def test_check_rules_text_only_sends_no_images(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, ['{"pass": true, "reason": ""}'])
    r = judge.check_rules(_srv(make_fake_server), "some text", "RULE", seed=1)
    assert r.ok is True
    assert "IMAGES" not in calls[0]["system"]
    assert calls[0]["images"] is None
