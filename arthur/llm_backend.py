"""
Pluggable LLM backend.

Why text-based ReAct instead of native function-calling?
----------------------------------------------------------
Small local models (the kind that actually fit your VRAM budget) are
unreliable at native structured tool-calling. A plain-text protocol that
the model just has to continue ("THOUGHT: / ACTION: / ARGS:") degrades
gracefully on weak models and is trivial to parse by hand -- which is also
the whole point of this project: understanding the loop, not hiding behind
a framework's tool-calling sugar.

The same protocol runs unchanged on all three backends below, which is what
makes the local-vs-cloud comparison in the writeup an apples-to-apples one.
"""

from dataclasses import dataclass
from typing import Callable, Iterator
import json
import re
import time

from . import config


@dataclass
class LLMResponse:
    text: str


class LLMBackend:
    def chat(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> LLMResponse:
        """
        Run one turn. If `on_token` is given and the backend supports
        streaming, it is called with each chunk as it arrives -- the caller
        uses this to render output live. The full text is always returned
        either way, so callers that don't care can ignore it.
        """
        raise NotImplementedError


# Reasoning-mode models (qwen3, deepseek-r1, ...) wrap their scratchpad in
# these. It is not protocol output and must not reach the parser.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> scratchpads.

    The unclosed case matters: if generation stops early (num_predict, a stop
    token) mid-thought, the closing tag never arrives and a naive regex leaves
    the entire scratchpad in place, where it looks exactly like a protocol
    violation to the parser.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()


class OllamaBackend(LLMBackend):
    """
    Local, offline. Requires `ollama serve` running and the model pulled.

    Everything interesting here is in `_options()`. A bare Ollama call with
    default settings will appear to work and then fail in ways that look like
    the model being stupid rather than the client being wrong -- see the
    comments on each option in config.py.
    """

    # Defaults are resolved in the body, not in the signature: a default
    # argument is evaluated once at import time, so `arthur --model X` (which
    # sets config.OLLAMA_MODEL at runtime) would be silently ignored.
    def __init__(self, model: str | None = None, host: str | None = None):
        import requests  # local import so the mock backend has zero deps
        self._requests = requests
        self.model = model or config.OLLAMA_MODEL
        self.host = host or config.OLLAMA_HOST

    @staticmethod
    def _options() -> dict:
        return {
            "num_ctx": config.OLLAMA_NUM_CTX,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "temperature": config.OLLAMA_TEMPERATURE,
            "top_p": config.OLLAMA_TOP_P,
            "stop": list(config.OLLAMA_STOP),
        }

    def _is_thinking_model(self) -> bool:
        name = self.model.lower()
        return any(name.startswith(p) for p in config.THINKING_MODEL_PREFIXES)

    def _prepare(self, messages: list[dict]) -> list[dict]:
        """
        Turn thinking off for models that do it.

        Two mechanisms because neither is universal: Ollama's `think` field is
        recent and not honoured by every build, while Qwen3 responds to a
        literal "/no_think" marker in the prompt. Sending both is harmless --
        the marker is ignored by models that don't know it.
        """
        if not (config.OLLAMA_DISABLE_THINKING and self._is_thinking_model()):
            return messages

        prepared = [dict(m) for m in messages]
        for m in prepared:
            if m.get("role") == "system":
                m["content"] = f"{m['content']}\n\n/no_think"
                return prepared
        # No system message (the critic builds its own): mark the last user turn.
        if prepared:
            prepared[-1]["content"] = f"{prepared[-1]['content']}\n\n/no_think"
        return prepared

    def _post(self, messages: list[dict], stream: bool):
        body = {
            "model": self.model,
            "messages": self._prepare(messages),
            "stream": stream,
            "options": self._options(),
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        if config.OLLAMA_DISABLE_THINKING and self._is_thinking_model():
            body["think"] = False

        return self._requests.post(
            f"{self.host}/api/chat",
            json=body,
            timeout=config.OLLAMA_TIMEOUT,
            stream=stream,
        )

    def _explain(self, exc: Exception) -> RuntimeError:
        """Turn transport-level failures into something actionable."""
        if isinstance(exc, self._requests.ConnectionError):
            return RuntimeError(
                f"cannot reach Ollama at {self.host}. Is the daemon running? "
                "Start it with `ollama serve`."
            )
        if isinstance(exc, self._requests.Timeout):
            return RuntimeError(
                f"Ollama timed out after {config.OLLAMA_TIMEOUT}s on model "
                f"'{self.model}'. A cold start on a large model can exceed "
                "this -- raise OLLAMA_TIMEOUT, or use a smaller model."
            )
        return RuntimeError(str(exc))

    def _check_status(self, resp) -> None:
        if resp.status_code == 200:
            return
        body = resp.text[:400]
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama does not have the model '{self.model}'.\n"
                f"  Pull it with:  ollama pull {self.model}\n"
                f"  Or list what you do have:  ollama list"
            )
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body}")

    def chat(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> LLMResponse:
        try:
            if on_token is None:
                resp = self._post(messages, stream=False)
                self._check_status(resp)
                text = resp.json()["message"]["content"]
            else:
                text = self._stream(messages, on_token)
        except self._requests.RequestException as e:
            raise self._explain(e) from e

        return LLMResponse(text=strip_thinking(text))

    def _stream(self, messages: list[dict], on_token: Callable[[str], None]) -> str:
        """
        Read the NDJSON stream, forwarding chunks to `on_token` as they land.

        Streaming is not cosmetic on this hardware: a 3B on a 4GB card takes
        tens of seconds for a full turn, and without it the terminal just sits
        there looking hung.
        """
        resp = self._post(messages, stream=True)
        self._check_status(resp)

        parts: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial line; the next iteration completes it
            if chunk.get("error"):
                raise RuntimeError(f"Ollama: {chunk['error']}")
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                parts.append(piece)
                on_token(piece)
            if chunk.get("done"):
                break
        return "".join(parts)

    # -- model management -----------------------------------------------------

    def installed_models(self) -> list[str]:
        resp = self._requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return sorted(m.get("name", "") for m in resp.json().get("models", []))

    def pull(self, model: str) -> Iterator[tuple[str, int, int]]:
        """
        Stream a model pull, yielding (status, completed_bytes, total_bytes)
        so a caller can draw a progress bar instead of blocking silently on a
        multi-gigabyte download.
        """
        resp = self._requests.post(
            f"{self.host}/api/pull",
            json={"model": model, "stream": True},
            timeout=None,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(f"pull failed: {chunk['error']}")
            yield (
                chunk.get("status", ""),
                int(chunk.get("completed", 0)),
                int(chunk.get("total", 0)),
            )


class GeminiBackend(LLMBackend):
    """
    Cloud, for when you want a reliable demo without fighting a 3B model.

    Talks to the Gemini REST API directly with `requests` rather than the
    google-genai SDK: this project already depends on requests for Ollama,
    and one visible HTTP call is easier to reason about (and to explain in
    an interview) than an SDK abstraction. The wire format differs from
    OpenAI/Anthropic in three ways worth knowing:

      1. the system prompt is its own top-level `system_instruction` field,
         not a message with role "system";
      2. the assistant role is called "model", not "assistant";
      3. 2.5-series models emit internal "thinking" parts in the response
         that must be filtered out before parsing.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import requests  # local import so the mock backend has zero deps
        self._requests = requests
        self.model = model or config.GEMINI_MODEL  # resolved late; see OllamaBackend
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY in your environment "
                "or put it in a .env file next to config.py:\n"
                "    GEMINI_API_KEY=your-key-here\n"
                "Get a key at https://aistudio.google.com/apikey"
            )

    # -- request building -----------------------------------------------------

    @staticmethod
    def _to_gemini_payload(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Map this project's OpenAI-shaped messages onto Gemini's format."""
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                # Multiple system messages get concatenated; Gemini takes one.
                system = m["content"] if system is None else f"{system}\n\n{m['content']}"
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def _body(self, messages: list[dict]) -> dict:
        system, contents = self._to_gemini_payload(messages)
        generation_config = {
            "temperature": config.GEMINI_TEMPERATURE,
            "maxOutputTokens": config.GEMINI_MAX_OUTPUT_TOKENS,
        }
        if config.GEMINI_THINKING_BUDGET is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": config.GEMINI_THINKING_BUDGET
            }
        body = {"contents": contents, "generationConfig": generation_config}
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        return body

    # -- response parsing -----------------------------------------------------

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            # Usually means the prompt itself was blocked by a safety filter.
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block!r})")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Skip thinking parts -- they are the model's scratchpad, not protocol output.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))

        if not text.strip():
            reason = candidate.get("finishReason")
            if reason == "MAX_TOKENS":
                raise RuntimeError(
                    "Gemini hit MAX_TOKENS before producing any output text. "
                    "Raise GEMINI_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_BUDGET."
                )
            raise RuntimeError(f"Gemini returned empty text (finishReason={reason!r})")
        return text

    # -- the call -------------------------------------------------------------

    def chat(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> LLMResponse:
        # Not streamed: the cloud path is fast enough that a spinner is fine,
        # and generateContent returns one blob. `on_token` is still honoured so
        # callers don't need to special-case the backend -- they just get the
        # whole turn in a single call.
        url = f"{config.GEMINI_API_BASE}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        body = self._body(messages)

        last_error = None
        for attempt in range(config.GEMINI_MAX_RETRIES):
            try:
                resp = self._requests.post(url, headers=headers, json=body,
                                           timeout=config.GEMINI_TIMEOUT)
            except self._requests.RequestException as e:
                last_error = e
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                text = self._extract_text(resp.json())
                if on_token:
                    on_token(text)
                return LLMResponse(text=text)

            # 429 = rate limited (very common on the free tier), 5xx = transient.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(2 ** attempt * 2)
                continue

            # 400/403/404 are our fault (bad key, bad model name) -- fail loudly.
            raise RuntimeError(
                f"Gemini API error {resp.status_code} for model '{self.model}':\n"
                f"{resp.text[:600]}"
            )

        raise RuntimeError(
            f"Gemini call failed after {config.GEMINI_MAX_RETRIES} attempts: {last_error}"
        )


class MockBackend(LLMBackend):
    """
    Deterministic, scripted backend so the whole pipeline (indexing,
    retrieval, the agent loop, the critic, the safety gate, patch application)
    can be exercised end-to-end with zero network access and zero local model.

    This is what lets you run `python main.py` right now and watch the full
    loop work, before you've plugged in Ollama or an API key.
    """

    def __init__(self, script: list[str]):
        self._script = list(script)
        self._i = 0

    def chat(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> LLMResponse:
        if self._i >= len(self._script):
            return LLMResponse(text="FINAL: (mock script exhausted) Stopping here.")
        text = self._script[self._i]
        self._i += 1
        if on_token:
            on_token(text)
        return LLMResponse(text=text)


def get_backend(name: str | None = None) -> LLMBackend:
    """
    Build a backend. Always returns a FRESH instance -- the critic in
    safety_gate.py needs its own, independent of the coder's conversation
    (and, for MockBackend, independent of the coder's script cursor).
    """
    name = (name or config.BACKEND).lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "gemini":
        return GeminiBackend()
    if name == "mock":
        return MockBackend(script=_demo_script())
    raise ValueError(f"Unknown backend {name!r}. Use one of: mock, ollama, gemini.")


def get_critic_backend(name: str | None = None) -> LLMBackend:
    """
    The critic's backend: always a fresh conversation, never the coder's.

    Locally it also runs a *smaller* model than the coder (see
    config.OLLAMA_CRITIC_MODEL). Judging a finished diff is an easier job than
    writing one, so the extra VRAM is better spent on the coder -- and a
    genuinely different set of weights makes the second opinion independent
    rather than the same model re-scoring its own work.
    """
    name = (name or config.BACKEND).lower()
    if name == "mock":
        return MockBackend(script=critic_script())
    if name == "ollama":
        return OllamaBackend(model=config.OLLAMA_CRITIC_MODEL)
    return get_backend(name)


def _demo_script() -> list[str]:
    """Scripted coder-agent turns for the bundled demo_repo scenario."""
    return [
        # Step 1: look around
        'THOUGHT: I need to see the file before changing it.\n'
        'ACTION: read_file\n'
        'ARGS: {"path": "calculator.py"}',

        # Step 2: change only the lines that change. The demo deliberately
        # models the behaviour we want from a real model -- naming the exact
        # lines rather than re-emitting the file -- since a whole-file rewrite
        # of an existing file is refused now.
        'THOUGHT: divide() does not guard against division by zero. '
        'I will add a check and raise a clear ValueError instead of letting '
        'ZeroDivisionError propagate.\n'
        'CONFIDENCE: 0.55\n'
        'ACTION: edit_file\n'
        'PATH: calculator.py\n'
        'FIND:\n```\n'
        'def divide(a, b):\n'
        '    return a / b\n'
        '```\n'
        'REPLACE:\n```\n'
        'def divide(a, b):\n'
        '    if b == 0:\n'
        '        raise ValueError("Cannot divide by zero")\n'
        '    return a / b\n'
        '```',

        # Step 3: after the gate/critic pass, report and stop.
        'THOUGHT: Patch applied and the critic approved it.\n'
        'FINAL: Added a zero-division guard in divide() in calculator.py: '
        'it now raises a ValueError("Cannot divide by zero") instead of '
        'crashing with ZeroDivisionError. 1 file changed.',
    ]


def critic_script() -> list[str]:
    """Scripted critic-agent verdicts, used by MockBackend-powered critic calls."""
    return [
        'VERDICT: APPROVE\n'
        'REASON: The guard correctly intercepts the zero-denominator case '
        'before the division executes and raises a clear, catchable error.'
    ]
