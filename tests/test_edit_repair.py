"""Tests for the self-repairing edit engine — the grounding layer that turns
near-miss edits (the #1 small-model failure) into successful edits or precise
corrections. Pure logic, so exhaustively unit-testable."""

from forge.tools.edit_repair import MatchOutcome, compute_edit
from forge.tools.filesystem import EditFileTool

# -- tier 1: exact ---------------------------------------------------------------


def test_exact_unique_applies():
    content = "a = 1\nb = 2\n"
    r = compute_edit(content, "b = 2", "b = 3")
    assert r.outcome == MatchOutcome.APPLIED
    assert r.tier == "exact"
    assert r.new_content == "a = 1\nb = 3\n"


def test_exact_multiple_is_ambiguous():
    content = "x = 1\nx = 1\n"
    r = compute_edit(content, "x = 1", "x = 2")
    assert r.outcome == MatchOutcome.AMBIGUOUS
    assert r.occurrences == 2


# -- tier 2: whitespace-tolerant auto-repair -------------------------------------


def test_trailing_whitespace_mismatch_auto_repairs():
    # model added a trailing space the file doesn't have -> not an exact
    # substring, so this must be repaired at the whitespace tier
    content = "def f():\n    return 1\n"
    r = compute_edit(content, "    return 1 ", "    return 2")
    assert r.outcome == MatchOutcome.APPLIED
    assert r.tier == "whitespace"
    assert "return 2" in r.new_content
    assert "return 1" not in r.new_content


def test_indentation_style_mismatch_auto_repairs():
    # file uses a tab; the model used four spaces
    content = "def f():\n\treturn total\n"
    r = compute_edit(content, "    return total", "    return total + 1")
    assert r.outcome == MatchOutcome.APPLIED
    assert r.tier == "whitespace"
    assert "total + 1" in r.new_content


def test_whitespace_match_edits_real_file_bytes():
    # the replacement must land on the file's real text, not the model's version
    content = "x   =   1\n"
    r = compute_edit(content, "x = 1", "x = 2")
    assert r.outcome == MatchOutcome.APPLIED
    assert r.new_content == "x = 2\n"


def test_whitespace_tolerant_ambiguity_is_reported():
    content = "y = 1 \ny = 1\n"  # both normalize to "y = 1"
    r = compute_edit(content, "y = 1", "y = 9")
    assert r.outcome == MatchOutcome.AMBIGUOUS


# -- tier 3: grounded correction -------------------------------------------------


def test_not_found_returns_closest_real_snippet():
    content = "def calculate_total(items):\n    return sum(items)\n"
    # model hallucinated a slightly different signature
    r = compute_edit(content, "def calculate_total(item):\n    return sum(item)", "x")
    assert r.outcome == MatchOutcome.NOT_FOUND
    assert r.suggestion is not None
    assert "def calculate_total(items):" in r.suggestion  # the REAL text


def test_totally_absent_has_no_suggestion():
    content = "print('hello world')\n"
    r = compute_edit(content, "class QuantumReactor:\n    def spin(self): ...", "x")
    assert r.outcome == MatchOutcome.NOT_FOUND
    assert r.suggestion is None  # nothing similar — don't mislead


# -- end-to-end through the tool -------------------------------------------------


def test_tool_auto_repairs_whitespace(guard, ledger, workspace):
    (workspace / "m.py").write_text("def f():\n\treturn x\n", encoding="utf-8")
    tool = EditFileTool(guard, ledger)
    # model uses spaces where the file uses a tab
    result = tool.run(path="m.py", old_string="    return x", new_string="    return y")
    assert result.ok
    assert "whitespace" in result.output
    assert "return y" in (workspace / "m.py").read_text(encoding="utf-8")


def test_tool_grounds_model_on_near_miss(guard, ledger, workspace):
    (workspace / "m.py").write_text(
        "def handler(request):\n    return process(request)\n", encoding="utf-8"
    )
    tool = EditFileTool(guard, ledger)
    result = tool.run(
        path="m.py",
        old_string="def handler(req):\n    return process(req)",
        new_string="x",
    )
    assert not result.ok
    assert "closest ACTUAL text" in result.error
    assert "def handler(request):" in result.error  # the real snippet to copy


def test_tool_still_reports_true_absence(guard, ledger, workspace):
    (workspace / "m.py").write_text("a = 1\n", encoding="utf-8")
    result = EditFileTool(guard, ledger).run(
        path="m.py", old_string="something entirely unrelated here", new_string="x"
    )
    assert not result.ok
    assert "read_file" in result.error
