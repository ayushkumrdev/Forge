"""Milestone 5 tests: the FastAPI service — health, repo, run submission with
background execution, event trace, memory, and the dashboard page."""

import json
import time

from fastapi.testclient import TestClient

from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.server.app import create_app


def _scripted_llm() -> MockLLMClient:
    plan = {
        "summary": "Add a greeting module.",
        "tasks": [
            {
                "id": 1,
                "title": "Create greeting module",
                "description": "Create greeting.py",
                "target_files": ["greeting.py"],
                "complexity": "low",
            }
        ],
    }
    return MockLLMClient(
        [
            ChatMessage(role="assistant", content=json.dumps(plan)),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "greeting.py", "content": "def greet():\n    pass\n"},
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Created greeting.py."),
            ChatMessage(
                role="assistant",
                content=json.dumps({"approved": True, "summary": "good", "issues": []}),
            ),
        ]
    )


def _client(workspace) -> TestClient:
    (workspace / "README.md").write_text("# sample\n", encoding="utf-8")
    app = create_app(workspace, llm_factory=_scripted_llm)
    return TestClient(app)


def _wait_for_completion(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] != "running":
            return detail
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_health_reports_fields(workspace):
    response = _client(workspace).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["workspace"] == str(workspace.resolve())
    assert "ollama_reachable" in body
    assert "model" in body


def test_repo_endpoint(workspace):
    response = _client(workspace).get("/api/repo")
    assert response.status_code == 200
    body = response.json()
    assert body["files"] == 1
    assert "README.md" in body["tree"]


def test_submit_run_executes_in_background(workspace):
    client = _client(workspace)
    response = client.post(
        "/api/runs", json={"request": "add a greeting module", "check_commands": []}
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    detail = _wait_for_completion(client, run_id)
    assert detail["status"] == "success"
    assert detail["report"]["changed_files"] == ["greeting.py"]
    assert (workspace / "greeting.py").exists()

    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == run_id

    events = client.get(f"/api/runs/{run_id}/events").json()
    kinds = {event["kind"] for event in events}
    assert "tool_call" in kinds
    assert "run_finished" in kinds

    lessons = client.get("/api/memory").json()
    assert lessons and lessons[0]["status"] == "approved"


def test_run_detail_404(workspace):
    assert _client(workspace).get("/api/runs/doesnotexist").status_code == 404


def test_run_request_validation(workspace):
    response = _client(workspace).post("/api/runs", json={"request": "x"})
    assert response.status_code == 422


def test_dashboard_served(workspace):
    response = _client(workspace).get("/")
    assert response.status_code == 200
    assert "FORGE" in response.text
