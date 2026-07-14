"""Milestone 4 tests: execution memory — lessons persisted per task and
recalled into the planner prompt for similar future requests."""

import json

from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.memory.service import ExecutionMemory
from forge.orchestrator.loop import ExecutionLoop


def test_lessons_round_trip(store):
    store.add_lesson("run1", "add auth endpoint", "Create login route", "rejected",
                     ["missing import of session module"])
    lessons = store.lessons()
    assert len(lessons) == 1
    assert lessons[0]["task_title"] == "Create login route"
    assert lessons[0]["issues"] == ["missing import of session module"]


def test_lessons_for_ranks_similar_request_first(store):
    memory = ExecutionMemory(store)
    memory.record_task("r1", "add login endpoint to the API", "Create login route",
                       "rejected", ["forgot to import session"])
    memory.record_task("r2", "optimize database queries", "Add query index",
                       "approved", [])
    hits = memory.lessons_for("create a login endpoint", k=1)
    assert hits and hits[0]["task_title"] == "Create login route"


def test_render_produces_prompt_block():
    text = ExecutionMemory.render(
        [
            {
                "request": "add login endpoint",
                "task_title": "Create login route",
                "status": "rejected",
                "issues": ["forgot to import session"],
            }
        ]
    )
    assert "Create login route" in text
    assert "forgot to import session" in text
    assert ExecutionMemory.render([]) == ""


def _plan(title: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=json.dumps(
            {
                "summary": "do the thing",
                "tasks": [
                    {
                        "id": 1,
                        "title": title,
                        "description": "Create greeting.py",
                        "target_files": ["greeting.py"],
                        "complexity": "low",
                    }
                ],
            }
        ),
    )


def _write() -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                name="write_file",
                arguments={"path": "greeting.py", "content": "def greet():\n    return 'hi'\n"},
            )
        ],
    )


def _review(approved: bool, issues=None) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=json.dumps({"approved": approved, "summary": "s", "issues": issues or []}),
    )


def test_orchestrator_records_lessons_and_recalls_them(workspace, store):
    # run 1: task gets rejected three times -> lesson with issues recorded
    llm1 = MockLLMClient(
        [
            _plan("Add greeting module"),
            ChatMessage(role="assistant", content="done (not really)"),
            _review(False, ["greeting.py never created"]),
            ChatMessage(role="assistant", content="done (not really)"),
            _review(False, ["greeting.py never created"]),
            ChatMessage(role="assistant", content="done (not really)"),
            _review(False, ["greeting.py never created"]),
        ]
    )
    loop1 = ExecutionLoop(workspace=workspace, llm=llm1, store=store, run_id="mem1")
    report1 = loop1.run("add a greeting module")
    assert report1.status == "failed"
    assert store.lessons()[0]["issues"] == ["greeting.py never created"]

    # run 2, similar request: the planner prompt must contain the past lesson
    llm2 = MockLLMClient([_plan("Add greeting module"), _write(),
                          ChatMessage(role="assistant", content="created"), _review(True)])
    loop2 = ExecutionLoop(workspace=workspace, llm=llm2, store=store, run_id="mem2")
    report2 = loop2.run("add a greeting module please")
    assert report2.status == "success"

    planner_prompt = llm2.requests[0][-1].content
    assert "Lessons from similar past runs" in planner_prompt
    assert "greeting.py never created" in planner_prompt
