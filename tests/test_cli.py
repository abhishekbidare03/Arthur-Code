"""
Tests for the `arthur` entry point.

The thing worth pinning down here is the wiring, not the agent: that the repo
defaults to wherever you're standing, that `--model` actually reaches the
backend (it silently did not, when the backends read config in their default
arguments), and that a bad path fails with a message instead of a traceback.
"""

import os

import pytest

from arthur import cli, config


# --- repo resolution ---------------------------------------------------------

def test_repo_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli._resolve_repo(None) == os.path.abspath(str(tmp_path))


def test_repo_flag_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    assert cli._resolve_repo("sub") == os.path.join(str(tmp_path), "sub")


def test_missing_repo_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli._resolve_repo(str(tmp_path / "nope"))
    assert "not a directory" in str(exc.value)


# --- overrides reaching the backend ------------------------------------------

def test_model_override_reaches_ollama_backend(monkeypatch):
    """
    Regression: OllamaBackend used to read `config.OLLAMA_MODEL` as a default
    argument, which Python evaluates once at import. Overriding the config at
    runtime therefore did nothing and `--model` was a no-op.
    """
    monkeypatch.setattr(config, "OLLAMA_MODEL", "placeholder", raising=False)
    cli._apply_overrides("ollama", "qwen2.5-coder:7b")
    assert config.OLLAMA_MODEL == "qwen2.5-coder:7b"

    from arthur.llm_backend import OllamaBackend
    pytest.importorskip("requests")
    assert OllamaBackend().model == "qwen2.5-coder:7b"


def test_model_override_is_scoped_to_its_backend(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-untouched", raising=False)
    monkeypatch.setattr(config, "OLLAMA_MODEL", "ollama-untouched", raising=False)
    cli._apply_overrides("ollama", "some-coder-model")
    assert config.GEMINI_MODEL == "gemini-untouched"


def test_critic_uses_the_smaller_local_model(monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "coder-big", raising=False)
    monkeypatch.setattr(config, "OLLAMA_CRITIC_MODEL", "critic-small", raising=False)

    from arthur.llm_backend import get_critic_backend
    assert get_critic_backend("ollama").model == "critic-small"


# --- preflight ---------------------------------------------------------------

def test_gemini_without_key_exits_with_guidance(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None, raising=False)
    with pytest.raises(SystemExit) as exc:
        cli._preflight("gemini")
    assert "GEMINI_API_KEY" in str(exc.value)


def test_mock_backend_needs_no_preflight():
    cli._preflight("mock")  # must not raise


# --- end to end through main() -----------------------------------------------

def test_one_shot_run_on_a_temp_repo(tmp_path, monkeypatch, capsys):
    """The mock backend's script targets calculator.py, so give it one."""
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["-p", "Add error handling for division by zero in divide", "--backend", "mock"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "GATE" in out
    assert "DONE" in out
    assert "Cannot divide by zero" in (tmp_path / "calculator.py").read_text(encoding="utf-8")


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
