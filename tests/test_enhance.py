"""Tests for enhance.py: quality gate, parsing, and the retry loop."""

import pytest

import enhance

GOOD = (
    "a young woman with long blonde wavy hair standing in a rain-soaked cyberpunk city "
    "at night, neon signs reflecting in puddles, cinematic lighting, shallow depth of field, "
    "highly detailed, dramatic atmosphere"
)
SHORT = "a girl in a city"


# --- is_good_prompt --------------------------------------------------------


def test_is_good_prompt_accepts_expansion():
    assert (
        enhance.is_good_prompt(GOOD, "a girl in a cyberpunk city", min_words=10) is True
    )


def test_is_good_prompt_rejects_too_short():
    assert (
        enhance.is_good_prompt(SHORT, "something else entirely here", min_words=50)
        is False
    )


def test_is_good_prompt_rejects_refusal():
    assert (
        enhance.is_good_prompt(
            "I cannot help with that request, sorry.", "x" * 300, min_words=10
        )
        is False
    )


def test_is_good_prompt_rejects_exact_echo():
    assert enhance.is_good_prompt(GOOD, GOOD, min_words=5) is False


def test_is_good_prompt_rejects_near_echo_jaccard():
    # result keeps almost every original word -> high Jaccard overlap
    original = " ".join(f"w{i}" for i in range(40))
    result = original + " a bit"
    assert enhance.is_good_prompt(result, original, min_words=10) is False


def test_is_good_prompt_accepts_faithful_expansion():
    original = "a cat"
    result = (
        original
        + " sitting on a windowsill bathed in warm afternoon sunlight, detailed fur"
    )
    assert enhance.is_good_prompt(result, original, min_words=5) is True


# --- _json_loads_lenient ---------------------------------------------------


def test_json_loads_lenient_plain():
    assert enhance._json_loads_lenient('{"a": 1}') == {"a": 1}


def test_json_loads_lenient_fenced():
    assert enhance._json_loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_loads_lenient_surrounding_text():
    assert enhance._json_loads_lenient(
        'Sure! Here you go: {"a": 1} hope that helps'
    ) == {"a": 1}


def test_json_loads_lenient_invalid():
    assert enhance._json_loads_lenient("not json at all") is None


# --- _parse_enhancement_response -------------------------------------------


def test_parse_with_negative():
    pos, neg = enhance._parse_enhancement_response(
        '{"positive": "good prompt", "negative": "blurry, extra limbs"}', True
    )
    assert pos == "good prompt"
    assert neg == "blurry, extra limbs"


def test_parse_without_negative_ignores_field():
    pos, neg = enhance._parse_enhancement_response(
        '{"positive": "good prompt", "negative": "should be ignored"}', False
    )
    assert pos == "good prompt"
    assert neg is None


def test_parse_discards_refusal_negative():
    pos, neg = enhance._parse_enhancement_response(
        '{"positive": "ok prompt", "negative": "I cannot generate that"}', True
    )
    assert pos == "ok prompt"
    assert neg is None


def test_parse_missing_positive():
    # positive is missing but negative is present and clean -> (None, negative)
    assert enhance._parse_enhancement_response('{"negative": "x"}', True) == (None, "x")


def test_parse_invalid_or_empty():
    assert enhance._parse_enhancement_response(None, True) == (None, None)
    assert enhance._parse_enhancement_response("no json", True) == (None, None)


# --- _enhance_json_instruction ---------------------------------------------


def test_json_instruction_without_negative():
    s = enhance._enhance_json_instruction(False)
    assert '"positive"' in s
    assert '"negative"' not in s


def test_json_instruction_with_negative():
    s = enhance._enhance_json_instruction(True)
    assert '"positive"' in s and '"negative"' in s


# --- enhance_prompt (retry loop) -------------------------------------------


def _mock_chat(monkeypatch, side_effects):
    """Replace chat_completion_interruptible with a sequence of canned results."""
    it = iter(side_effects)
    calls = []

    def fake(server, system, user, seed, images=None, temperature=None):
        calls.append(
            {"system": system, "user": user, "seed": seed, "temperature": temperature}
        )
        return next(it)

    monkeypatch.setattr(enhance, "chat_completion_interruptible", fake)
    return calls


def test_enhance_passes_first_attempt(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, ['{"positive": "%s", "negative": "blurry"}' % GOOD])
    r = enhance.enhance_prompt(
        make_fake_server(),
        "a girl",
        "SYS",
        seed=10,
        generate_negative=True,
        min_words=10,
    )
    assert r.positive == GOOD
    assert r.negative == "blurry"
    assert len(calls) == 1
    # the JSON contract was appended to the user's system prompt
    assert calls[0]["system"].startswith("SYS")
    assert '"positive"' in calls[0]["system"]


def test_enhance_retries_then_passes(monkeypatch, make_fake_server):
    good = '{"positive": "%s"}' % GOOD
    calls = _mock_chat(monkeypatch, [SHORT, good])
    r = enhance.enhance_prompt(
        make_fake_server(), "a girl", "SYS", seed=10, min_words=10
    )
    assert r.positive == GOOD
    assert len(calls) == 2


def test_enhance_falls_back_to_longest_clean(monkeypatch, make_fake_server):
    # every attempt is a clean positive but below the word floor
    a = '{"positive": "%s"}' % ("word " * 10).strip()
    b = '{"positive": "%s"}' % ("word " * 30).strip()
    calls = _mock_chat(monkeypatch, [a, b])
    r = enhance.enhance_prompt(
        make_fake_server(), "zzz", "SYS", seed=1, max_retries=2, min_words=100
    )
    assert r.positive == ("word " * 30).strip()
    assert len(calls) == 2


def test_enhance_all_fail_returns_none(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, [None, None])
    r = enhance.enhance_prompt(
        make_fake_server(), "a girl", "SYS", seed=1, max_retries=2, min_words=10
    )
    assert r.positive is None
    assert r.negative is None
    assert len(calls) == 2


def test_enhance_refusals_are_not_returned(monkeypatch, make_fake_server):
    _mock_chat(
        monkeypatch,
        ['{"positive": "I cannot help with that"}', '{"positive": "I cannot assist"}'],
    )
    r = enhance.enhance_prompt(
        make_fake_server(), "a girl", "SYS", seed=1, max_retries=2, min_words=5
    )
    assert r.positive is None


def test_enhance_seed_advances_per_attempt(monkeypatch, make_fake_server):
    calls = _mock_chat(monkeypatch, [None, '{"positive": "%s"}' % GOOD])
    enhance.enhance_prompt(
        make_fake_server(), "a girl", "SYS", seed=100, max_retries=2, min_words=10
    )
    assert calls[0]["seed"] == 100
    assert calls[1]["seed"] == 101


# --- interrupt handling -----------------------------------------------------


def test_enhance_prompt_interrupt_kills_server(
    monkeypatch, make_fake_server, interrupting_comfy
):
    exc = interrupting_comfy
    srv = make_fake_server()
    with pytest.raises(exc):
        enhance.enhance_prompt(srv, "a girl", "SYS")
    assert not srv.is_alive()
