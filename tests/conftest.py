from pathlib import Path

import pytest

from forge.memory.store import MemoryStore
from forge.safety.guard import SafetyGuard
from forge.telemetry import Recorder
from forge.tools.changes import ChangeLedger


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def guard(workspace: Path) -> SafetyGuard:
    return SafetyGuard(workspace)


@pytest.fixture()
def ledger(workspace: Path) -> ChangeLedger:
    return ChangeLedger(workspace, run_id="testrun")


@pytest.fixture()
def recorder(workspace: Path) -> Recorder:
    return Recorder("testrun", workspace, store=None, console=None)


@pytest.fixture()
def store(workspace: Path) -> MemoryStore:
    memory = MemoryStore(workspace)
    yield memory
    memory.close()


@pytest.fixture(autouse=True)
def _no_intent_brief(monkeypatch):
    """Self-briefing is off for the suite unless a test asks for it.

    Reasoning before acting costs a whole model call per action turn. These
    tests script exact call sequences against a mock, so leaving it on would
    mean threading one extra scripted reply through every action-turn test in
    the project — noise that would obscure what each test is actually about.

    The tests that care about briefing set FORGE_GATE_INTENT_BRIEF=1
    themselves, and the real default (on for smart and genius) lives in
    ForgeSettings where the product behaviour belongs.
    """
    monkeypatch.setenv("FORGE_GATE_INTENT_BRIEF", "0")
