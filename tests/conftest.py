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
