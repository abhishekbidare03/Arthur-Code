"""
Tests for the Ollama request shape and response handling.

These assert on settings that have no visible effect until they're wrong, and
then fail as "the model is stupid" rather than "the client is misconfigured".
Most of all num_ctx: leave it unset and llama.cpp truncates from the front,
silently deleting the system prompt mid-run.

No live model needed -- the HTTP layer is faked.
"""

import json

import pytest

from arthur import config
from arthur.llm_backend import OllamaBackend, strip_thinking

requests = pytest.importorskip("requests")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, lines=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


@pytest.fixture
def captured(monkeypatch):
    """Capture the JSON body of the next POST instead of sending it."""
    box = {}

    def fake_post(url, json=None, timeout=None, stream=False, **kw):
        box["url"] = url
        box["body"] = json
        box["timeout"] = timeout
        return FakeResponse({"message": {"content": "THOUGHT: ok\nFINAL: done"}})

    backend = OllamaBackend(model="test-model")
    monkeypatch.setattr(backend._requests, "post", fake_post)
    return backend, box


# --- request shape -----------------------------------------------------------

def test_num_ctx_is_always_sent(captured):
    """The one that silently eats the system prompt when missing."""
    backend, box = captured
    backend.chat([{"role": "user", "content": "hi"}])
    assert box["body"]["options"]["num_ctx"] == config.OLLAMA_NUM_CTX


def test_stop_sequences_prevent_self_dialogue(captured):
    """Small models otherwise write the tool's OBSERVATION: themselves."""
    backend, box = captured
    backend.chat([{"role": "user", "content": "hi"}])
    assert "OBSERVATION:" in box["body"]["options"]["stop"]


def test_temperature_is_low_for_protocol_compliance(captured):
    backend, box = captured
    backend.chat([{"role": "user", "content": "hi"}])
    assert box["body"]["options"]["temperature"] <= 0.2


def test_keep_alive_is_sent_so_the_model_is_not_evicted(captured):
    """Every loop step is a separate request; eviction costs a reload each time."""
    backend, box = captured
    backend.chat([{"role": "user", "content": "hi"}])
    assert box["body"]["keep_alive"] == config.OLLAMA_KEEP_ALIVE


def test_runtime_model_override_is_used(captured):
    backend, box = captured
    backend.chat([{"role": "user", "content": "hi"}])
    assert box["body"]["model"] == "test-model"


def test_options_follow_config_at_call_time(captured, monkeypatch):
    """Config is read per call, so /model and /ctx can change mid-session."""
    backend, box = captured
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 16384)
    backend.chat([{"role": "user", "content": "hi"}])
    assert box["body"]["options"]["num_ctx"] == 16384


# --- thinking blocks ---------------------------------------------------------

def test_think_block_is_stripped():
    text = "<think>Let me consider this carefully.</think>\nTHOUGHT: ok\nFINAL: done"
    assert strip_thinking(text).startswith("THOUGHT:")


def test_unclosed_think_block_is_stripped():
    """Generation cut short by num_predict or a stop token never closes the tag."""
    assert strip_thinking("THOUGHT: fine\n<think>wait, actually") == "THOUGHT: fine"


def test_text_without_thinking_is_untouched():
    text = 'THOUGHT: go\nACTION: read_file\nARGS: {"path": "a.py"}'
    assert strip_thinking(text) == text


def test_thinking_is_disabled_for_reasoning_models(monkeypatch):
    """
    Reasoning models are a bad fit for a rigid protocol: the format already IS
    the reasoning scaffold, so the scratchpad buys nothing and costs a large
    multiple in latency -- and can consume the whole num_predict budget before
    any protocol output appears.
    """
    box = {}
    backend = OllamaBackend(model="qwen3:4b")
    monkeypatch.setattr(backend._requests, "post",
                        lambda url, json=None, **kw: (box.update(body=json),
                                                      FakeResponse({"message": {"content": "FINAL: x"}}))[1])

    backend.chat([{"role": "system", "content": "protocol"},
                  {"role": "user", "content": "task"}])

    assert box["body"]["think"] is False
    assert box["body"]["messages"][0]["content"].endswith("/no_think")


def test_no_think_marker_goes_on_the_last_turn_without_a_system_message(monkeypatch):
    """The critic builds its own messages and may not include a system role."""
    box = {}
    backend = OllamaBackend(model="qwen3:4b")
    monkeypatch.setattr(backend._requests, "post",
                        lambda url, json=None, **kw: (box.update(body=json),
                                                      FakeResponse({"message": {"content": "x"}}))[1])

    backend.chat([{"role": "user", "content": "review this"}])
    assert box["body"]["messages"][-1]["content"].endswith("/no_think")


def test_non_thinking_models_are_left_alone(monkeypatch):
    box = {}
    backend = OllamaBackend(model="qwen2.5-coder:3b")
    monkeypatch.setattr(backend._requests, "post",
                        lambda url, json=None, **kw: (box.update(body=json),
                                                      FakeResponse({"message": {"content": "x"}}))[1])

    backend.chat([{"role": "system", "content": "protocol"}])

    assert "think" not in box["body"]
    assert box["body"]["messages"][0]["content"] == "protocol"


def test_caller_messages_are_not_mutated(monkeypatch):
    """The session reuses its history; appending markers to it would compound."""
    backend = OllamaBackend(model="qwen3:4b")
    monkeypatch.setattr(backend._requests, "post",
                        lambda *a, **k: FakeResponse({"message": {"content": "x"}}))

    original = [{"role": "system", "content": "protocol"}]
    backend.chat(original)
    assert original[0]["content"] == "protocol"


def test_chat_strips_thinking_before_returning(monkeypatch):
    backend = OllamaBackend(model="m")
    monkeypatch.setattr(
        backend._requests, "post",
        lambda *a, **k: FakeResponse({"message": {"content": "<think>hm</think>FINAL: x"}}),
    )
    assert backend.chat([{"role": "user", "content": "hi"}]).text == "FINAL: x"


# --- streaming ---------------------------------------------------------------

def test_streaming_forwards_chunks_and_returns_full_text(monkeypatch):
    chunks = [
        json.dumps({"message": {"content": "THOUGHT: "}}),
        json.dumps({"message": {"content": "looking"}}),
        json.dumps({"message": {"content": "\nFINAL: done"}, "done": True}),
    ]
    backend = OllamaBackend(model="m")
    monkeypatch.setattr(backend._requests, "post",
                        lambda *a, **k: FakeResponse(lines=chunks))

    seen = []
    result = backend.chat([{"role": "user", "content": "hi"}], on_token=seen.append)

    assert "".join(seen) == "THOUGHT: looking\nFINAL: done"
    assert result.text == "THOUGHT: looking\nFINAL: done"


def test_streaming_survives_a_partial_line(monkeypatch):
    lines = ["{ broken json", json.dumps({"message": {"content": "FINAL: ok"}, "done": True})]
    backend = OllamaBackend(model="m")
    monkeypatch.setattr(backend._requests, "post",
                        lambda *a, **k: FakeResponse(lines=lines))
    assert backend.chat([{"role": "user", "content": "x"}], on_token=lambda c: None).text == "FINAL: ok"


def test_streamed_error_is_raised(monkeypatch):
    lines = [json.dumps({"error": "model requires more system memory"})]
    backend = OllamaBackend(model="m")
    monkeypatch.setattr(backend._requests, "post",
                        lambda *a, **k: FakeResponse(lines=lines))
    with pytest.raises(RuntimeError, match="more system memory"):
        backend.chat([{"role": "user", "content": "x"}], on_token=lambda c: None)


# --- errors people actually hit ----------------------------------------------

def test_missing_model_says_how_to_pull_it(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:3b")
    monkeypatch.setattr(backend._requests, "post",
                        lambda *a, **k: FakeResponse(status_code=404, text="not found"))

    with pytest.raises(RuntimeError) as exc:
        backend.chat([{"role": "user", "content": "hi"}])
    assert "ollama pull qwen2.5-coder:3b" in str(exc.value)


def test_daemon_down_says_to_start_it(monkeypatch):
    backend = OllamaBackend(model="m")

    def refuse(*a, **k):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(backend._requests, "post", refuse)
    with pytest.raises(RuntimeError, match="ollama serve"):
        backend.chat([{"role": "user", "content": "hi"}])


def test_timeout_suggests_a_remedy(monkeypatch):
    backend = OllamaBackend(model="m")

    def slow(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(backend._requests, "post", slow)
    with pytest.raises(RuntimeError, match="OLLAMA_TIMEOUT|smaller model"):
        backend.chat([{"role": "user", "content": "hi"}])
