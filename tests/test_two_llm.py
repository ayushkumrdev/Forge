"""Two-model brain: the thinker model interprets the user's intent and briefs
the coder model; thinker failures degrade gracefully to single-model mode."""

from forge.chat.session import ChatSession
from forge.llm.base import ChatMessage, LLMError
from forge.llm.mock import MockLLMClient


def _user_content(request: list[ChatMessage]) -> str:
    return next(m.content for m in request if m.role == "user")


def test_thinker_brief_reaches_the_coder(workspace, monkeypatch):
    monkeypatch.setenv("FORGE_GATE_INTENT_BRIEF", "1")  # this test IS about briefing
    thinker = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                content="INTENT: create a greeting module. STEPS: write greet.py "
                "with greet(). WHERE: repo root. VERIFY: py_compile it.",
            )
        ]
    )
    coder = MockLLMClient([ChatMessage(role="assistant", content="done")])
    session = ChatSession(workspace, coder, session_id="brain", thinker_llm=thinker)
    session.send("make a greeting module")

    # thinker got the raw message
    assert _user_content(thinker.requests[0]) == "make a greeting module"
    # coder got the raw message PLUS the interpreted brief
    coder_prompt = _user_content(coder.requests[0])
    assert "make a greeting module" in coder_prompt
    assert "Intent brief" in coder_prompt
    assert "create a greeting module" in coder_prompt


class _BrokenLLM:
    def chat(self, *args, **kwargs):
        raise LLMError("thinker model is not pulled")


def test_thinker_failure_falls_back_to_single_model(workspace):
    coder = MockLLMClient([ChatMessage(role="assistant", content="handled")])
    session = ChatSession(workspace, coder, session_id="brain2", thinker_llm=_BrokenLLM())
    reply = session.send("explain this repo")
    assert reply == "handled"
    assert "Intent brief" not in _user_content(coder.requests[0])


def test_no_thinker_by_default(workspace):
    coder = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = ChatSession(workspace, coder, session_id="brain3")
    session.send("hello")
    assert session._thinker is None
    assert "Intent brief" not in _user_content(coder.requests[0])
