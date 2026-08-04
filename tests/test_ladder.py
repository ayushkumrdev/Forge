"""The verification ladder: ordered rungs, pre-existing-failure tolerance,
and the resolution rung that catches hallucinated imports."""

from pathlib import Path

from forge.verify.ladder import RESOLUTION, SYNTAX, Ladder
from forge.verify.resolution import public_names, resolution_errors

# -- L2 resolution ---------------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "utils.py").write_text(
        "import json\n\nCONSTANT = 3\n\n\ndef helper(x):\n    return x\n\n\n"
        "class Widget:\n    pass\n",
        encoding="utf-8",
    )
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "inner.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    return tmp_path


def test_stdlib_and_installed_imports_resolve(tmp_path):
    repo = _repo(tmp_path)
    source = "import json\nimport pathlib\nimport pytest\nfrom typing import Any\n"
    assert resolution_errors(repo / "m.py", source, repo) == []


def test_hallucinated_module_is_caught(tmp_path):
    repo = _repo(tmp_path)
    errors = resolution_errors(
        repo / "m.py", "import totally_made_up_lib\n", repo
    )
    assert len(errors) == 1
    assert "totally_made_up_lib" in errors[0]
    assert "not installed" in errors[0]


def test_hallucinated_name_in_real_repo_module_is_caught(tmp_path):
    """The characteristic failure: the module exists, the name does not."""
    repo = _repo(tmp_path)
    errors = resolution_errors(repo / "m.py", "from utils import nonexistent\n", repo)
    assert len(errors) == 1
    assert "'utils' does not define 'nonexistent'" in errors[0]


def test_near_miss_gets_a_did_you_mean(tmp_path):
    repo = _repo(tmp_path)
    errors = resolution_errors(repo / "m.py", "from utils import helpr\n", repo)
    assert "did you mean 'helper'" in errors[0]


def test_real_names_and_reexports_resolve(tmp_path):
    repo = _repo(tmp_path)
    source = "from utils import helper, Widget, CONSTANT, json\n"
    assert resolution_errors(repo / "m.py", source, repo) == []


def test_submodule_import_resolves(tmp_path):
    repo = _repo(tmp_path)
    assert resolution_errors(repo / "m.py", "from pkg import inner\n", repo) == []
    assert resolution_errors(repo / "m.py", "from pkg.inner import go\n", repo) == []


def test_relative_imports(tmp_path):
    repo = _repo(tmp_path)
    inner = repo / "pkg" / "inner.py"
    assert resolution_errors(inner, "from . import inner\n", repo) == []
    bad = resolution_errors(inner, "from .ghost import thing\n", repo)
    assert bad and "does not resolve" in bad[0]


def test_star_import_is_not_judged(tmp_path):
    repo = _repo(tmp_path)
    assert resolution_errors(repo / "m.py", "from utils import *\n", repo) == []


def test_syntax_errors_are_left_to_the_syntax_rung(tmp_path):
    repo = _repo(tmp_path)
    assert resolution_errors(repo / "m.py", "def broken(:\n", repo) == []


def test_public_names_collects_all_top_level_forms():
    names = public_names(
        "import os\nfrom sys import path\nX = 1\nY: int = 2\n"
        "def f(): pass\nasync def g(): pass\nclass C: pass\n"
    )
    assert names == {"os", "path", "X", "Y", "f", "g", "C"}


# -- the ladder ------------------------------------------------------------------


def test_ladder_passes_a_clean_change(tmp_path):
    repo = _repo(tmp_path)
    ladder = Ladder(repo)
    verdict = ladder.check(repo / "m.py", None, "from utils import helper\n")
    assert verdict.ok
    assert verdict.failed_rung is None
    assert verdict.highest_passed == RESOLUTION


def test_ladder_stops_at_syntax_before_paying_for_resolution(tmp_path):
    repo = _repo(tmp_path)
    verdict = Ladder(repo).check(repo / "m.py", None, "from utils import (\n")
    assert not verdict.ok
    assert verdict.failed_rung == SYNTAX
    # the resolution rung was never attempted
    assert [r.rung for r in verdict.results] == [SYNTAX]


def test_ladder_catches_hallucinated_import_that_parses_fine(tmp_path):
    """The whole point of L2: this file is perfectly valid Python."""
    repo = _repo(tmp_path)
    verdict = Ladder(repo).check(
        repo / "m.py", None, "from utils import make_everything_work\n"
    )
    assert not verdict.ok
    assert verdict.failed_rung == RESOLUTION
    assert "does not define" in verdict.diagnostic


def test_pre_existing_syntax_failure_never_blocks(tmp_path):
    repo = _repo(tmp_path)
    broken_before = "def f(:\n"
    still_broken = "def f(:\n# a comment\n"
    verdict = Ladder(repo).check(repo / "m.py", broken_before, still_broken)
    assert verdict.ok
    assert verdict.results[0].pre_existing


def test_pre_existing_bad_import_never_blocks(tmp_path):
    """Editing one line of a file that already had a bad import must not be
    blamed for it — otherwise the agent is trapped."""
    repo = _repo(tmp_path)
    before = "from utils import ghost\n\nx = 1\n"
    after = "from utils import ghost\n\nx = 2\n"
    verdict = Ladder(repo).check(repo / "m.py", before, after)
    assert verdict.ok
    assert any(r.rung == RESOLUTION and r.pre_existing for r in verdict.results)


def test_newly_introduced_bad_import_does_block(tmp_path):
    repo = _repo(tmp_path)
    before = "x = 1\n"
    after = "from utils import ghost\n\nx = 1\n"
    verdict = Ladder(repo).check(repo / "m.py", before, after)
    assert not verdict.ok
    assert verdict.failed_rung == RESOLUTION


def test_line_shift_does_not_resurrect_a_pre_existing_problem(tmp_path):
    """The bad import moves down a line; it is still pre-existing."""
    repo = _repo(tmp_path)
    before = "from utils import ghost\nx = 1\n"
    after = "import json\nfrom utils import ghost\nx = 1\n"
    verdict = Ladder(repo).check(repo / "m.py", before, after)
    assert verdict.ok


def test_disabled_resolution_still_detects_but_does_not_block(tmp_path):
    """Disabling a rung must stop it BLOCKING, not stop it MEASURING —
    otherwise an ablation cannot report the hallucinations it now allows."""
    repo = _repo(tmp_path)
    verdict = Ladder(repo, resolution=False).check(
        repo / "m.py", None, "from utils import ghost\n"
    )
    assert verdict.ok  # the write is allowed through
    unenforced = verdict.unenforced_failures
    assert [r.rung for r in unenforced] == [RESOLUTION]
    assert "does not define 'ghost'" in unenforced[0].diagnostic


def test_disabled_resolution_reports_nothing_when_code_is_clean(tmp_path):
    repo = _repo(tmp_path)
    verdict = Ladder(repo, resolution=False).check(
        repo / "m.py", None, "from utils import helper\n"
    )
    assert verdict.ok
    assert verdict.unenforced_failures == []


def test_types_rung_skipped_by_default(tmp_path):
    repo = _repo(tmp_path)
    verdict = Ladder(repo).check(repo / "m.py", None, "x = 1\n")
    assert verdict.summary().endswith("types:skip")


def test_non_python_files_pass_resolution(tmp_path):
    repo = _repo(tmp_path)
    verdict = Ladder(repo).check(repo / "data.json", None, '{"a": 1}')
    assert verdict.ok


# -- wiring: the tools actually climb the ladder ---------------------------------


def _session(workspace, **settings_kwargs):
    from forge.chat.session import ChatSession
    from forge.config import ForgeSettings
    from forge.llm.mock import MockLLMClient

    return ChatSession(
        workspace,
        MockLLMClient([]),
        ForgeSettings(**settings_kwargs),
        session_id="ladder-wiring",
    )


def test_write_file_refuses_a_hallucinated_import(tmp_path):
    repo = _repo(tmp_path)
    session = _session(repo)
    result = session.registry.execute(
        "write_file",
        {"path": "main.py", "content": "from utils import make_magic\n"},
    )
    assert not result.ok
    assert "does not define 'make_magic'" in result.error
    assert not (repo / "main.py").exists()  # nothing written


def test_write_file_allows_a_real_import(tmp_path):
    repo = _repo(tmp_path)
    session = _session(repo)
    result = session.registry.execute(
        "write_file", {"path": "ok.py", "content": "from utils import helper\n"}
    )
    assert result.ok
    assert (repo / "ok.py").exists()


def test_edit_file_refuses_an_edit_that_adds_a_bad_import(tmp_path):
    repo = _repo(tmp_path)
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    session = _session(repo)
    result = session.registry.execute(
        "edit_file",
        {"path": "mod.py", "old_string": "x = 1", "new_string": "from utils import ghost\nx = 1"},
    )
    assert not result.ok
    assert "resolution" in result.error
    assert (repo / "mod.py").read_text(encoding="utf-8") == "x = 1\n"  # untouched


def test_resolution_gate_can_be_ablated_but_still_measures(tmp_path):
    """Ablating the rung must not blind the metrics: the write lands, and the
    detection is reported so HIR stays comparable across configurations."""
    repo = _repo(tmp_path)
    session = _session(repo, gate_resolution=False)
    result = session.registry.execute(
        "write_file", {"path": "main.py", "content": "from utils import make_magic\n"}
    )
    assert result.ok  # syntax still checked, resolution not enforced
    assert (repo / "main.py").exists()
    assert "unenforced" in result.output
    assert "resolution check failed" in result.output


def test_hir_counts_hallucinations_in_both_configurations():
    from forge.evals.metrics import metrics_from_events

    blocked = metrics_from_events([
        {"kind": "tool_call", "tool": "write_file", "arguments": {"path": "a.py"}},
        {"kind": "tool_result", "tool": "write_file", "ok": False,
         "error": "Rejected — the resolution check failed: no such name."},
    ])
    allowed = metrics_from_events([
        {"kind": "tool_call", "tool": "write_file", "arguments": {"path": "a.py"}},
        {"kind": "tool_result", "tool": "write_file", "ok": True,
         "output": "Wrote 20 chars to a.py. [unenforced: resolution check failed: x]"},
    ])
    assert blocked.hallucinated_identifier == 1.0
    assert allowed.hallucinated_identifier == 1.0  # same rate, gate off


def test_clean_writes_have_zero_hir():
    from forge.evals.metrics import metrics_from_events

    m = metrics_from_events([
        {"kind": "tool_call", "tool": "write_file", "arguments": {"path": "a.py"}},
        {"kind": "tool_result", "tool": "write_file", "ok": True,
         "output": "Wrote 20 chars to a.py."},
    ])
    assert m.hallucinated_identifier == 0.0


# -- dangling references: the rename that missed a caller -------------------------
# Observed live on t2-rename-in-file: push -> enqueue was applied to the class
# and to one internal call, leaving queue.push(value) in a module-level helper.
# Valid Python, resolvable imports, broken at runtime.

BEFORE_RENAME = (
    "class Queue:\n"
    "    def push(self, item):\n"
    "        self._items.append(item)\n\n"
    "    def drain(self):\n"
    "        return self.push(1)\n\n\n"
    "def fill(queue, values):\n"
    "    queue.push(values)\n"
)


def test_missed_caller_is_caught(tmp_path):
    from forge.verify.resolution import dangling_reference_errors

    partial = BEFORE_RENAME.replace("def push", "def enqueue").replace(
        "self.push(1)", "self.enqueue(1)"
    )  # queue.push(values) left behind
    problems = dangling_reference_errors(BEFORE_RENAME, partial)
    assert len(problems) == 1
    assert "'push' is still called here" in problems[0]


def test_complete_rename_passes(tmp_path):
    from forge.verify.resolution import dangling_reference_errors

    complete = BEFORE_RENAME.replace("push", "enqueue")
    assert dangling_reference_errors(BEFORE_RENAME, complete) == []


def test_removing_a_function_and_its_callers_is_fine():
    from forge.verify.resolution import dangling_reference_errors

    after = "class Queue:\n    def drain(self):\n        return 1\n"
    assert dangling_reference_errors(BEFORE_RENAME, after) == []


def test_ladder_does_NOT_block_mid_rename(tmp_path):
    """A rename must be allowed to pass through an inconsistent state.
    Blocking each write trapped the agent mid-operation and measurably
    dropped tier 2 from 3/6 to 1/6 (83% wasted cycles). The check runs once
    at the end of a turn instead — see test_dangling_check_runs_at_turn_end."""
    repo = _repo(tmp_path)
    partial = BEFORE_RENAME.replace("def push", "def enqueue").replace(
        "self.push(1)", "self.enqueue(1)"
    )
    verdict = Ladder(repo).check(repo / "q.py", BEFORE_RENAME, partial)
    assert verdict.ok


def test_edit_tool_allows_the_first_step_of_a_rename(tmp_path):
    repo = _repo(tmp_path)
    (repo / "q.py").write_text(BEFORE_RENAME, encoding="utf-8")
    session = _session(repo)
    result = session.registry.execute(
        "edit_file",
        {"path": "q.py", "old_string": "    def push(self, item):",
         "new_string": "    def enqueue(self, item):"},
    )
    assert result.ok, result.error


def test_dangling_check_runs_at_turn_end(tmp_path):
    """The incomplete rename is caught once, when the turn tries to finish."""
    from forge.chat.session import ChatSession
    from forge.config import ForgeSettings
    from forge.llm.base import ChatMessage, ToolCall
    from forge.llm.mock import MockLLMClient

    repo = _repo(tmp_path)
    (repo / "q.py").write_text(BEFORE_RENAME, encoding="utf-8")
    finished = BEFORE_RENAME.replace("push", "enqueue")
    llm = MockLLMClient([
        # renames the definition but leaves fill() calling the old name
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "q.py",
                       "content": BEFORE_RENAME.replace("def push", "def enqueue")
                                               .replace("self.push(1)", "self.enqueue(1)")})]),
        ChatMessage(role="assistant", content="Renamed it."),
        # after being told, finishes the job
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "q.py", "content": finished})]),
        ChatMessage(role="assistant", content="Updated every caller."),
    ])
    session = ChatSession(repo, llm, ForgeSettings(), session_id="dang")
    reply = session.send("rename push to enqueue")
    assert reply == "Updated every caller."
    assert "queue.push(" not in (repo / "q.py").read_text(encoding="utf-8")
    assert any("no longer exists" in m.content for m in session.history)
