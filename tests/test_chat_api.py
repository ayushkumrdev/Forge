"""Milestone 7 tests: the desktop app's chat API — session start, background
turn execution, live updates, the blocking approval flow, model switching,
undo/diff, and the app page."""

import time

from fastapi.testclient import TestClient

from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.server.app import create_app


def _llm_scripts():
    """Each session start pops the next scripted client."""
    scripts = []

    def factory(model):
        client = scripts.pop(0) if scripts else MockLLMClient([])
        client.model = model or "qwen2.5-coder:7b"
        return client

    return scripts, factory


def _client(workspace, scripts_factory) -> TestClient:
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    return TestClient(create_app(workspace, chat_llm_factory=scripts_factory))


def _wait_idle(client: TestClient, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = client.get("/api/chat/updates?events_after=0&messages_after=0").json()
        if st["status"] == "idle":
            return st
        time.sleep(0.05)
    raise AssertionError("session did not become idle in time")


def _wait_pending(client: TestClient, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = client.get("/api/chat/updates").json()
        if st["pending_approval"]:
            return st
        time.sleep(0.05)
    raise AssertionError("no approval request appeared in time")


def _write_call(path="made.txt", content="hello") -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="write_file", arguments={"path": path, "content": content})],
    )


def test_start_requires_valid_folder(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    response = client.post("/api/chat/start", json={"workspace": "C:/definitely/not/here"})
    assert response.status_code == 400


def test_full_turn_auto_mode(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([_write_call(), ChatMessage(role="assistant", content="All done!")])
    )
    client = _client(workspace, factory)

    st = client.post("/api/chat/start", json={"mode": "auto"}).json()
    assert st["status"] == "idle"
    assert st["workspace"] == str(workspace.resolve())

    assert client.post("/api/chat/message", json={"text": "make made.txt"}).status_code == 202
    st = _wait_idle(client)

    texts = [m["text"] for m in st["messages"]]
    assert "make made.txt" in texts
    assert "All done!" in texts
    assert (workspace / "made.txt").exists()
    assert st["changed_files"] == ["made.txt"]
    # tool activity was streamed as events
    kinds = {e["kind"] for e in st["events"]}
    assert "tool_call" in kinds


def test_approval_flow_allow(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([_write_call(), ChatMessage(role="assistant", content="done")])
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "write the file"})

    pending = _wait_pending(client)["pending_approval"]
    assert pending["tool"] == "write_file"
    assert "made.txt" in pending["detail"]

    client.post("/api/chat/approval", json={"approved": True})
    _wait_idle(client)
    assert (workspace / "made.txt").exists()


def test_approval_flow_deny(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient(
            [_write_call(), ChatMessage(role="assistant", content="okay, I won't")]
        )
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "write the file"})

    _wait_pending(client)
    client.post("/api/chat/approval", json={"approved": False})
    st = _wait_idle(client)
    assert not (workspace / "made.txt").exists()
    assert "okay, I won't" in [m["text"] for m in st["messages"]]


def test_approval_allow_always_skips_next_prompt(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient(
            [
                _write_call("one.txt"),
                _write_call("two.txt"),
                ChatMessage(role="assistant", content="both written"),
            ]
        )
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "write both files"})

    _wait_pending(client)
    client.post("/api/chat/approval", json={"approved": True, "always": True})
    _wait_idle(client)  # second write must NOT block on approval
    assert (workspace / "one.txt").exists()
    assert (workspace / "two.txt").exists()


def test_busy_session_rejects_second_message(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([_write_call(), ChatMessage(role="assistant", content="ok")]))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "first"})
    _wait_pending(client)  # worker is now blocked on approval
    assert client.post("/api/chat/message", json={"text": "second"}).status_code == 409
    client.post("/api/chat/approval", json={"approved": True})
    _wait_idle(client)


def test_model_switch(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([]))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"model": "qwen2.5-coder:7b"})
    st = client.post("/api/chat/model", json={"model": "glm-5.2:latest"}).json()
    assert st["model"] == "glm-5.2:latest"


def test_slash_command_and_undo_endpoint(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([_write_call(), ChatMessage(role="assistant", content="written")])
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "auto"})
    client.post("/api/chat/message", json={"text": "write it"})
    _wait_idle(client)

    client.post("/api/chat/message", json={"text": "/diff"})
    st = _wait_idle(client)
    diff_message = [m for m in st["messages"] if m.get("kind") == "command"][-1]
    assert "+hello" in diff_message["text"]

    restored = client.post("/api/chat/undo").json()["restored"]
    assert restored == ["made.txt"]
    assert not (workspace / "made.txt").exists()


def test_no_session_yields_409(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    assert client.get("/api/chat/updates").status_code == 409
    assert client.post("/api/chat/message", json={"text": "hi"}).status_code == 409


def test_app_page_served(workspace):
    scripts, factory = _llm_scripts()
    response = _client(workspace, factory).get("/app")
    assert response.status_code == 200
    assert "FORGE" in response.text
    assert "pick_folder" in response.text  # native folder picker bridge


def test_models_endpoint_returns_list(workspace):
    scripts, factory = _llm_scripts()
    response = _client(workspace, factory).get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
