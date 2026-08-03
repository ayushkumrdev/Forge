"""Conversation management: mode switching mid-chat, image attach uploads,
delete, and persistent conversations that survive an app restart."""

import base64
import time

from tests.test_chat_api import _client, _llm_scripts, _wait

from forge.llm.base import ChatMessage
from forge.llm.mock import MockLLMClient


def _start(client, workspace, **overrides):
    payload = {"workspace": str(workspace), "mode": "ask", **overrides}
    response = client.post("/api/chat/start", json=payload)
    assert response.status_code == 200
    return response.json()


def test_mode_switches_mid_chat(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    state = _start(client, workspace)
    assert state["mode"] == "ask"
    state = client.post("/api/chat/mode", json={"mode": "auto"}).json()
    assert state["mode"] == "auto"
    state = client.post("/api/chat/mode", json={"mode": "ask"}).json()
    assert state["mode"] == "ask"
    assert client.post("/api/chat/mode", json={"mode": "yolo"}).status_code == 422


def test_attach_saves_image_into_workspace(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    _start(client, workspace)
    data = base64.b64encode(b"\x89PNG fakebytes").decode()
    response = client.post(
        "/api/chat/attach", json={"filename": "my shot!.png", "data_b64": data}
    )
    assert response.status_code == 200
    rel = response.json()["path"]
    assert rel.startswith(".forge/uploads/")
    assert rel.endswith("_my_shot_.png")  # sanitized
    assert (workspace / rel).read_bytes() == b"\x89PNG fakebytes"
    assert client.post(
        "/api/chat/attach", json={"filename": "x.png", "data_b64": "@@not-base64@@"}
    ).status_code == 400


def test_delete_removes_session_and_transcript(workspace):
    scripts, factory = _llm_scripts()
    scripts.append(MockLLMClient([ChatMessage(role="assistant", content="hi there")]))
    client = _client(workspace, factory)
    first = _start(client, workspace)
    client.post("/api/chat/message", json={"text": "hello"})
    _wait(client, lambda st: st["status"] == "idle" and st["message_count"] >= 2)

    transcript = workspace / ".forge" / "chat" / f"app-{first['session_id']}.json"
    assert transcript.exists()

    second = _start(client, workspace)  # a second conversation becomes current
    r = client.post("/api/chat/delete", json={"session_id": first["session_id"]}).json()
    assert r["ok"] and r["current"] == second["session_id"]
    assert not transcript.exists()

    sessions = client.get("/api/chat/sessions").json()
    assert all(s["session_id"] != first["session_id"] for s in sessions)


def test_saved_conversations_survive_restart_and_resume(workspace):
    # launch 1: real conversation persisted to transcript
    scripts, factory = _llm_scripts()
    scripts.append(
        MockLLMClient([ChatMessage(role="assistant", content="Answer: 42.")])
    )
    client1 = _client(workspace, factory)
    first = _start(client1, workspace)
    client1.post("/api/chat/message", json={"text": "what is the answer?"})
    _wait(client1, lambda st: st["status"] == "idle" and st["message_count"] >= 2)

    # launch 2: fresh server process, same workspace
    scripts2, factory2 = _llm_scripts()
    client2 = _client(workspace, factory2)
    _start(client2, workspace)
    time.sleep(0)  # transcripts are read from disk on listing
    sessions = client2.get("/api/chat/sessions").json()
    saved = [s for s in sessions if s["status"] == "saved"]
    assert any(s["session_id"] == first["session_id"] for s in saved)
    assert any("what is the answer?" in s["title"] for s in saved)

    # resuming restores the visible message history
    state = client2.post(
        "/api/chat/select", json={"session_id": first["session_id"]}
    ).json()
    assert state["session_id"] == first["session_id"]
    updates = client2.get("/api/chat/updates?events_after=0&messages_after=0").json()
    texts = [m["text"] for m in updates["messages"]]
    assert "what is the answer?" in texts
    assert "Answer: 42." in texts


def test_empty_transcripts_are_not_listed(workspace):
    scripts, factory = _llm_scripts()
    client = _client(workspace, factory)
    _start(client, workspace)  # never sent a message -> empty transcript at most
    sessions = client.get("/api/chat/sessions").json()
    assert all(s["status"] != "saved" or s["message_count"] > 0 for s in sessions)
    # the only listed sessions are the live one and non-empty saved ones
    assert len([s for s in sessions if s["status"] == "saved"]) == 0
