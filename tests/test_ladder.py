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


def test_resolution_can_be_disabled(tmp_path):
    repo = _repo(tmp_path)
    verdict = Ladder(repo, resolution=False).check(
        repo / "m.py", None, "from utils import ghost\n"
    )
    assert verdict.ok
    assert any(r.rung == RESOLUTION and r.skipped for r in verdict.results)


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


def test_resolution_gate_can_be_ablated(tmp_path):
    repo = _repo(tmp_path)
    session = _session(repo, gate_resolution=False)
    result = session.registry.execute(
        "write_file", {"path": "main.py", "content": "from utils import make_magic\n"}
    )
    assert result.ok  # syntax still checked, resolution skipped
    assert (repo / "main.py").exists()
