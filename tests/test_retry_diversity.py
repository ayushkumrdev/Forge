"""Verifier-guided retry diversity and plan-path grounding."""

from forge.agents.planner import Plan, PlanTask
from forge.orchestrator.loop import ground_target_files, retry_temperature


def test_retry_temperature_escalates_and_caps():
    assert retry_temperature(0.2, 1) is None  # first attempt: configured default
    assert retry_temperature(0.2, 2) == 0.5
    assert retry_temperature(0.2, 3) == 0.8
    assert retry_temperature(0.2, 4) == 0.9  # capped
    assert retry_temperature(0.8, 2) == 0.9  # capped from a high base too


def test_ground_target_files_annotates_missing_paths():
    plan = Plan(
        summary="s",
        tasks=[
            PlanTask(
                id=1,
                title="t",
                description="d",
                target_files=["src/real.py", "src/imagined.py"],
            )
        ],
    )
    ground_target_files(plan, {"src/real.py"})
    assert plan.tasks[0].target_files[0] == "src/real.py"
    assert "new file" in plan.tasks[0].target_files[1]


def test_ground_target_files_normalizes_backslashes():
    plan = Plan(
        summary="s",
        tasks=[PlanTask(id=1, title="t", description="d", target_files=["src\\real.py"])],
    )
    ground_target_files(plan, {"src/real.py"})
    assert plan.tasks[0].target_files == ["src\\real.py"]  # recognized, unannotated


def test_coder_passes_temperature_through(workspace, recorder):
    from forge.agents.coder import Coder
    from forge.llm.base import ChatMessage
    from forge.llm.mock import MockLLMClient
    from forge.safety.guard import SafetyGuard
    from forge.tools.base import ToolRegistry
    from forge.tools.filesystem import ReadFileTool

    llm = MockLLMClient([ChatMessage(role="assistant", content="done")])
    registry = ToolRegistry([ReadFileTool(SafetyGuard(workspace))])
    coder = Coder(llm, registry, recorder)
    task = PlanTask(id=1, title="t", description="d")
    plan = Plan(summary="s", tasks=[task])

    coder.execute(task, plan, "repo", temperature=0.7)
    assert llm.temperatures == [0.7]
