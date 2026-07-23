"""Wiring smoke test for the Layer-3 agent. No live API / subprocess.

Mocks the genai client + MCP stdio/session boundaries; asserts ask() passes
the prompt as contents, hands the session straight into tools=[session], and
returns the model's text — and that failures surface as SoccerhubError.
"""
import pytest

from soccerhub.errors import SoccerhubError


class _ACM:
    """Minimal async context manager yielding a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc):
        return False


class _AsyncCall:
    """Async callable that records kwargs and returns (or raises) a fixed result."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _wire(monkeypatch, gen_result):
    """Patch agent's genai/stdio/session boundaries. Returns (session, gen_call)."""
    from soccerhub import agent

    async def _init():
        return None

    session = type("Session", (), {"initialize": staticmethod(_init)})()
    monkeypatch.setattr(agent, "stdio_client", lambda server: _ACM((None, None)))
    monkeypatch.setattr(agent, "ClientSession", lambda read, write: _ACM(session))

    gen = _AsyncCall(gen_result)
    client = type("Client", (), {})()
    client.aio = type("Aio", (), {})()
    client.aio.models = type("Models", (), {"generate_content": gen})()
    fake_genai = type("genai", (), {"Client": staticmethod(lambda api_key: client)})
    monkeypatch.setattr(agent, "genai", fake_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return session, gen


def test_ask_returns_model_text(monkeypatch):
    resp = type("Resp", (), {"text": "Messi leads with 12 xG."})()
    session, gen = _wire(monkeypatch, resp)

    from soccerhub.agent import ask

    out = ask("who leads La Liga in xG?")

    assert out == "Messi leads with 12 xG."
    kwargs = gen.calls[0]
    assert kwargs["contents"] == "who leads La Liga in xG?"
    assert kwargs["config"].tools == [session]  # session handed straight to tools


def test_ask_wraps_errors(monkeypatch):
    _wire(monkeypatch, ValueError("boom"))

    from soccerhub.agent import ask

    with pytest.raises(SoccerhubError):
        ask("q")
