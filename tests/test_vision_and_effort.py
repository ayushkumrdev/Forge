"""Vision (read_image + @image mentions) and Fast/Smart/Genius effort levels."""

import base64
import json

import httpx
import pytest

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.safety.guard import SafetyGuard
from forge.tools.vision import ReadImageTool

# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _vision_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    assert payload["model"] == "llava:7b"
    assert payload["messages"][0]["images"], "image must be attached as base64"
    return httpx.Response(
        200, json={"message": {"content": "A login form with a red error banner."}}
    )


def test_read_image_describes_via_vision_model(workspace):
    (workspace / "shot.png").write_bytes(_PNG)
    tool = ReadImageTool(
        SafetyGuard(workspace), model="llava:7b",
        transport=httpx.MockTransport(_vision_handler),
    )
    result = tool.run(path="shot.png")
    assert result.ok
    assert "login form" in result.output
    assert "[image shot.png seen by llava:7b]" in result.output


def test_read_image_rejects_non_images_and_missing(workspace):
    (workspace / "notes.txt").write_text("hi")
    tool = ReadImageTool(SafetyGuard(workspace))
    assert not tool.run(path="notes.txt").ok
    assert not tool.run(path="ghost.png").ok


def test_read_image_missing_model_advises_pull(workspace):
    (workspace / "s.png").write_bytes(_PNG)
    tool = ReadImageTool(
        SafetyGuard(workspace), model="llava:7b",
        transport=httpx.MockTransport(lambda r: httpx.Response(404, json={})),
    )
    result = tool.run(path="s.png")
    assert not result.ok
    assert "ollama pull llava:7b" in result.error


def test_image_mention_is_described_for_the_coder(workspace):
    (workspace / "bug.png").write_bytes(_PNG)
    settings = ForgeSettings(vision_model="llava:7b")
    coder = MockLLMClient([ChatMessage(role="assistant", content="I see it.")])
    session = ChatSession(workspace, coder, settings, session_id="vis")
    # swap in a mocked vision tool for the session
    session._vision = ReadImageTool(
        SafetyGuard(workspace), model="llava:7b",
        transport=httpx.MockTransport(_vision_handler),
    )
    session.send("what's wrong in @bug.png ?")
    prompt = next(m.content for m in coder.requests[0] if m.role == "user")
    assert "red error banner" in prompt  # the description reached the coder


def test_image_mention_without_vision_model_notes_it(workspace):
    (workspace / "bug.png").write_bytes(_PNG)
    coder = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = ChatSession(workspace, coder, session_id="novis")
    session.send("look at @bug.png")
    prompt = next(m.content for m in coder.requests[0] if m.role == "user")
    assert "FORGE_VISION_MODEL" in prompt


# -- effort levels ----------------------------------------------------------------


def test_fast_skips_thinker_and_shrinks_budget(workspace):
    thinker = MockLLMClient([ChatMessage(role="assistant", content="brief")])
    coder = MockLLMClient([ChatMessage(role="assistant", content="quick answer")])
    settings = ForgeSettings(effort="fast")
    session = ChatSession(workspace, coder, settings, session_id="f", thinker_llm=thinker)
    session.send("add a helper")
    assert thinker.requests == []  # thinker never consulted
    assert session._step_budget() < settings.max_agent_steps


def test_genius_self_briefs_without_a_thinker(workspace):
    coder = MockLLMClient(
        [
            ChatMessage(role="assistant", content="INTENT: add helper. STEPS: ..."),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "h.py", "content": "def h():\n    pass\n"},
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Added h.py with helper."),
            ChatMessage(role="assistant", content="Everything done: h.py verified."),
        ]
    )
    settings = ForgeSettings(effort="genius")
    session = ChatSession(workspace, coder, settings, session_id="g")
    reply = session.send("add a helper module h.py")
    # self-brief happened (call 1) and the completeness check ran (call 4)
    assert reply == "Everything done: h.py verified."
    assert any("Final completeness check" in m.content for m in session.history)
    prompt = next(m.content for m in coder.requests[1] if m.role == "user")
    assert "Intent brief" in prompt
    assert session._step_budget() > settings.max_agent_steps


def test_set_effort_validates(workspace):
    session = ChatSession(workspace, MockLLMClient([]), session_id="v")
    session.set_effort("genius")
    assert session.effort == "genius"
    with pytest.raises(ValueError):
        session.set_effort("ludicrous")


def test_api_start_and_switch_effort(workspace):
    from tests.test_chat_api import _client, _llm_scripts

    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    state = client.post(
        "/api/chat/start", json={"workspace": str(workspace), "effort": "genius"}
    ).json()
    assert state["effort"] == "genius"
    state = client.post("/api/chat/effort", json={"effort": "fast"}).json()
    assert state["effort"] == "fast"
    assert client.post("/api/chat/effort", json={"effort": "warp"}).status_code == 422
