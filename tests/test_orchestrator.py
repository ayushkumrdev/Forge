"""End-to-end tests of the full plan -> code -> review loop with a scripted
mock LLM: no network, fully deterministic."""

import json

from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.orchestrator.loop import ExecutionLoop


def _plan_message(description: str = "Create greeting.py with a greet() function.") -> ChatMessage:
    plan = {
        "summary": "Add a greeting module.",
        "tasks": [
            {
                "id": 1,
                "title": "Create greeting module",
                "description": description,
                "target_files": ["greeting.py"],
                "complexity": "low",
            }
        ],
    }
    return ChatMessage(role="assistant", content=json.dumps(plan))


def _write_call() -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                name="write_file",
                arguments={
                    "path": "greeting.py",
                    "content": "def greet(name):\n    return f'hello {name}'\n",
                },
            )
        ],
    )


def _coder_done() -> ChatMessage:
    return ChatMessage(role="assistant", content="Created greeting.py with greet().")


def _review(approved: bool, issues: list[str] | None = None) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=json.dumps(
            {"approved": approved, "summary": "checked", "issues": issues or []}
        ),
    )


def test_happy_path_run(workspace):
    llm = MockLLMClient([_plan_message(), _write_call(), _coder_done(), _review(True)])
    loop = ExecutionLoop(workspace=workspace, llm=llm, run_id="e2e1")
    report = loop.run("add a greeting module")

    assert report.status == "success"
    assert report.task_results[0].status == "approved"
    assert report.task_results[0].attempts == 1
    assert report.changed_files == ["greeting.py"]
    assert "+def greet(name):" in report.diff
    assert (workspace / "greeting.py").exists()
    assert (workspace / ".forge" / "runs" / "e2e1.json").exists()


def test_reviewer_rejection_triggers_revision_with_feedback(workspace):
    llm = MockLLMClient(
        [
            _plan_message(),
            # attempt 1: coder claims done without changing anything
            ChatMessage(role="assistant", content="Nothing to do."),
            _review(False, issues=["greeting.py was never created"]),
            # attempt 2: coder actually writes the file
            _write_call(),
            _coder_done(),
            _review(True),
        ]
    )
    loop = ExecutionLoop(workspace=workspace, llm=llm, run_id="e2e2")
    report = loop.run("add a greeting module")

    assert report.status == "success"
    assert report.task_results[0].attempts == 2
    assert (workspace / "greeting.py").exists()

    # the reviewer's issue must have been fed back to the coder on attempt 2
    second_attempt_prompt = llm.requests[3][-1].content
    assert "Reviewer feedback" in second_attempt_prompt
    assert "greeting.py was never created" in second_attempt_prompt


def test_check_output_is_fed_back_to_coder_on_rejection(workspace):
    llm = MockLLMClient(
        [
            _plan_message(),
            ChatMessage(role="assistant", content="attempt 1"),
            _review(False, issues=["tests fail"]),
            _write_call(),
            _coder_done(),
            _review(True),
        ]
    )
    loop = ExecutionLoop(
        workspace=workspace,
        llm=llm,
        run_id="e2e6",
        check_commands=['python -c "print(\'CHECK_MARKER_XYZ\')"'],
    )
    report = loop.run("add a greeting module")

    assert report.status == "success"
    second_attempt_prompt = llm.requests[3][-1].content
    assert "CHECK_MARKER_XYZ" in second_attempt_prompt


def test_run_fails_after_max_review_cycles(workspace):
    llm = MockLLMClient(
        [
            _plan_message(),
            ChatMessage(role="assistant", content="attempt 1"),
            _review(False, issues=["still wrong"]),
            ChatMessage(role="assistant", content="attempt 2"),
            _review(False, issues=["still wrong"]),
            ChatMessage(role="assistant", content="attempt 3"),
            _review(False, issues=["still wrong"]),
        ]
    )
    loop = ExecutionLoop(workspace=workspace, llm=llm, run_id="e2e3")
    report = loop.run("impossible task")

    assert report.status == "failed"
    assert report.task_results[0].status == "rejected"
    assert report.task_results[0].attempts == 3


def test_planner_recovers_from_invalid_json(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="I think we should refactor everything!"),
            _plan_message(),
            _write_call(),
            _coder_done(),
            _review(True),
        ]
    )
    loop = ExecutionLoop(workspace=workspace, llm=llm, run_id="e2e4")
    report = loop.run("add a greeting module")

    assert report.status == "success"
    # the retry prompt must have told the model its output was invalid
    retry_prompt = llm.requests[1][-1].content
    assert "not valid" in retry_prompt


def test_llm_failure_produces_failed_report_not_crash(workspace):
    llm = MockLLMClient([])  # no responses -> immediate LLMError
    loop = ExecutionLoop(workspace=workspace, llm=llm, run_id="e2e5")
    report = loop.run("anything")

    assert report.status == "failed"
    assert "LLMError" in report.error
    assert (workspace / ".forge" / "runs" / "e2e5.json").exists()
