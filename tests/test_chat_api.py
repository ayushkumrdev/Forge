"""Desktop app chat API tests: session start, background turns, streaming
partials, stop, queueing, multiple conversations, the blocking approval flow,
model switching, undo/diff, recent folders, and the app page."""

import threading
import time

from fastapi.testclient import TestClient

from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.server.app import create_app


class BlockingLLM(MockLLMClient):
    """First chat call emits a token, then blocks until the gate opens."""

    def __init__(self, responses, gate: threading.Event, token: str = "Thinking hard"):
        super().__init__(responses)
        self._gate = gate
        self._token = token
        self._calls = 0

    def chat(self, messages, tools=None, temperature=None, on_token=None):
        call_index = self._calls
        self._calls += 1
        if call_index == 0:
            if on_token is not None:
                on_token(self._token)
            assert self._gate.wait(timeout=10), "test gate never opened"
        return super().chat(messages, tools, temperature, on_token=None)


def _llm_scripts():
    scripts = []

    def factory(model):
        client = scripts.pop(0) if scripts else MockLLMClient([])
        client.model = model or "qwen2.5-coder:7b"
        return client

    return scripts, factory


def _client(workspace, scripts_factory) -> TestClient:
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    return TestClient(
        create_app(
            workspace,
            chat_llm_factory=scripts_factory,
            app_state_path=workspace / "state" / "app_state.json",
        )
    )


def _wait(client: TestClient, predicate, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = client.get("/api/chat/updates?events_after=0&messages_after=0").json()
        if predicate(st):
            return st
        time.sleep(0.04)
    raise AssertionError("condition not reached in time")


def _wait_idle(client):
    return _wait(client, lambda st: st["status"] == "idle")


def _write_call(path="made.txt", content="hello") -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="write_file", arguments={"path": path, "content": content})],
    )


# -- basics ----------------------------------------------------------------------


def test_start_requires_valid_folder(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    assert client.post("/api/chat/start", json={"workspace": "C:/nope/never"}).status_code == 400


def test_full_turn_auto_mode(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([_write_call(), ChatMessage(role="assistant", content="All done!")])
    )
    client = _client(workspace, factory)
    st = client.post("/api/chat/start", json={"mode": "auto"}).json()
    assert st["status"] == "idle"

    assert client.post("/api/chat/message", json={"text": "make made.txt"}).status_code == 202
    st = _wait_idle(client)

    texts = [m["text"] for m in st["messages"]]
    assert "All done!" in texts
    assert (workspace / "made.txt").exists()
    assert st["changed_files"] == ["made.txt"]
    assert {e["kind"] for e in st["events"]} >= {"tool_call", "tool_result"}
    # tool_result events carry an output snippet for the UI cards
    results = [e for e in st["events"] if e["kind"] == "tool_result"]
    assert any("Wrote" in (e.get("output") or "") for e in results)


def test_no_session_yields_409(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    assert client.get("/api/chat/updates").status_code == 409
    assert client.post("/api/chat/message", json={"text": "hi"}).status_code == 409


# -- streaming, stop, queue --------------------------------------------------------


def test_streaming_partial_visible_while_working(workspace):
    scripts, factory = _llm_scripts()
    gate = threading.Event()
    scripts.append(BlockingLLM([ChatMessage(role="assistant", content="final")], gate))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "auto"})
    client.post("/api/chat/message", json={"text": "hi"})

    st = _wait(client, lambda s: s["partial"])
    assert st["partial"] == "Thinking hard"
    gate.set()
    st = _wait_idle(client)
    assert st["partial"] == ""
    assert "final" in [m["text"] for m in st["messages"]]


def test_stop_cancels_turn(workspace):
    scripts, factory = _llm_scripts()
    gate = threading.Event()
    scripts.append(
        BlockingLLM([_write_call(), ChatMessage(role="assistant", content="done")], gate)
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "auto"})
    client.post("/api/chat/message", json={"text": "write it"})

    _wait(client, lambda s: s["status"] == "working")
    client.post("/api/chat/stop")
    gate.set()
    st = _wait_idle(client)

    assert "Stopped by user." in [m["text"] for m in st["messages"]]
    assert not (workspace / "made.txt").exists()  # cancel landed before the tool ran


def test_second_message_queues_and_runs_after(workspace):
    scripts, factory = _llm_scripts()
    gate = threading.Event()
    scripts.append(
        BlockingLLM(
            [
                ChatMessage(role="assistant", content="first done"),
                ChatMessage(role="assistant", content="second done"),
            ],
            gate,
        )
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "auto"})
    client.post("/api/chat/message", json={"text": "one"})
    _wait(client, lambda s: s["status"] == "working")

    response = client.post("/api/chat/message", json={"text": "two"})
    assert response.status_code == 202
    assert response.json()["disposition"] == "queued"
    assert response.json()["queued"] == 1

    gate.set()
    st = _wait_idle(client)
    texts = [m["text"] for m in st["messages"]]
    assert texts.count("first done") == 1
    assert texts.count("second done") == 1


# -- approvals ---------------------------------------------------------------------


def test_approval_flow_allow(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([_write_call(), ChatMessage(role="assistant", content="done")]))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "write the file"})

    st = _wait(client, lambda s: s["pending_approval"])
    assert st["pending_approval"]["tool"] == "write_file"
    client.post("/api/chat/approval", json={"approved": True})
    _wait_idle(client)
    assert (workspace / "made.txt").exists()


def test_approval_flow_deny(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([_write_call(), ChatMessage(role="assistant", content="okay, I won't")])
    )
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "ask"})
    client.post("/api/chat/message", json={"text": "write the file"})
    _wait(client, lambda s: s["pending_approval"])
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
    # phrased as one outcome on purpose: this test is about the approval
    # flow, and "write both" now reads as multi-part, which engages
    # plan-first decomposition and changes the call sequence
    client.post("/api/chat/message", json={"text": "write the files"})
    _wait(client, lambda s: s["pending_approval"])
    client.post("/api/chat/approval", json={"approved": True, "always": True})
    _wait_idle(client)
    assert (workspace / "one.txt").exists()
    assert (workspace / "two.txt").exists()


# -- sessions / models / folders ----------------------------------------------------


def test_multiple_sessions_and_select(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([ChatMessage(role="assistant", content="hello from A")]))
    scripts.append(MockLLMClient([]))
    client = _client(workspace, factory)

    a = client.post("/api/chat/start", json={"mode": "auto"}).json()
    client.post("/api/chat/message", json={"text": "greetings forge"})
    _wait_idle(client)
    b = client.post("/api/chat/start", json={"mode": "auto"}).json()

    sessions = client.get("/api/chat/sessions").json()
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == b["session_id"]  # newest first
    assert sessions[0]["current"] is True
    titled = {s["session_id"]: s["title"] for s in sessions}
    assert titled[a["session_id"]] == "greetings forge"
    assert titled[b["session_id"]] == "New conversation"

    st = client.post("/api/chat/select", json={"session_id": a["session_id"]}).json()
    assert st["session_id"] == a["session_id"]
    assert client.post("/api/chat/select", json={"session_id": "nope"}).status_code == 404


def test_model_switch(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([]))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"model": "qwen2.5-coder:7b"})
    st = client.post("/api/chat/model", json={"model": "glm-5.2:latest"}).json()
    assert st["model"] == "glm-5.2:latest"


def test_recent_folders_remembered(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    assert client.get("/api/chat/recent").json() == []
    client.post("/api/chat/start", json={"mode": "auto"})
    recent = client.get("/api/chat/recent").json()
    assert str(workspace.resolve()) in recent


def test_undo_endpoint(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([_write_call(), ChatMessage(role="assistant", content="ok")]))
    client = _client(workspace, factory)
    client.post("/api/chat/start", json={"mode": "auto"})
    client.post("/api/chat/message", json={"text": "write it"})
    _wait_idle(client)
    assert client.post("/api/chat/undo").json()["restored"] == ["made.txt"]
    assert not (workspace / "made.txt").exists()


def test_app_page_served(workspace):
    scripts, factory = _llm_scripts()
    response = _client(workspace, factory).get("/app")
    assert response.status_code == 200
    assert "Forge" in response.text
    assert "pick_folder" in response.text
