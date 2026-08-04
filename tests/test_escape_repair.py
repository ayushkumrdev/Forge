"""Escape repair — the fix for a death-spiral observed live on the benchmark.

qwen2.5-coder:7b emitted edit_file arguments in which newlines arrived as the
two characters \\ and n and regex backslashes arrived doubled. Every match
tier failed, the fuzzy suggestion collapsed to a single line, and the model
re-sent the same broken needle 11 times in a row. These tests pin the repair.
"""

from forge.tools.edit_repair import (
    MatchOutcome,
    compute_edit,
    reindent_replacement,
    unescape_literals,
)

FILE = (
    "import re\n"
    "\n"
    "\n"
    "def titlecase(text):\n"
    '    """Capitalize the first letter of every word."""\n'
    "    return re.sub(r'\\w+', lambda m: m.group(0).capitalize(), text)\n"
)


def test_unescape_decodes_in_one_pass():
    assert unescape_literals("a\\nb") == "a\nb"
    assert unescape_literals("a\\tb") == "a\tb"
    assert unescape_literals("r'\\\\w+'") == "r'\\w+'"
    assert unescape_literals("no escapes here") == "no escapes here"


def test_literal_backslash_n_needle_now_applies():
    """The exact live failure: a 3-line needle arriving as one line."""
    needle = "def titlecase(text):\\n" '    """Capitalize the first letter of every word."""'
    result = compute_edit(FILE, needle, "def titlecase(text):\\n    # replaced")
    assert result.outcome == MatchOutcome.APPLIED
    assert result.tier == "escaped"
    assert "# replaced" in result.new_content
    # the replacement was decoded too, so no literal \n is written to disk
    assert "\\n" not in result.new_content.replace("\\w", "")


def test_doubled_regex_backslashes_are_repaired():
    needle = "    return re.sub(r'\\\\w+', lambda m: m.group(0).capitalize(), text)"
    result = compute_edit(FILE, needle, "    return text.title()")
    assert result.outcome == MatchOutcome.APPLIED
    assert result.tier == "escaped"
    assert "text.title()" in result.new_content


def test_repair_never_invents_a_location():
    """Decoding must not make a non-existent snippet 'match' something."""
    result = compute_edit(FILE, "def nonexistent():\\n    return 99", "x")
    assert result.outcome == MatchOutcome.NOT_FOUND
    assert result.new_content is None


def test_correctly_encoded_edits_still_take_the_exact_path():
    result = compute_edit(FILE, "import re", "import regex as re")
    assert result.outcome == MatchOutcome.APPLIED
    assert result.tier == "exact"


def test_ambiguity_is_still_reported_not_repaired():
    content = "a = 1\nb = 2\na = 1\n"
    result = compute_edit(content, "a = 1", "a = 3")
    assert result.outcome == MatchOutcome.AMBIGUOUS
    assert result.occurrences == 2


def test_escaped_needle_gets_a_multiline_suggestion_when_it_cannot_apply():
    """Regression for the loop itself: a near-miss escaped needle must yield a
    suggestion spanning the real lines, not a single collapsed line."""
    needle = "def titlecase(txt):\\n" '    """Capitalize the first letter of every word."""'
    result = compute_edit(FILE, needle, "y")
    assert result.outcome == MatchOutcome.NOT_FOUND
    assert result.suggestion is not None
    assert "\n" in result.suggestion  # spans multiple real lines
    assert "def titlecase(text):" in result.suggestion


# -- indentation repair ----------------------------------------------------------
# Observed live on t2-three-guards: the model replaced `return int(text)` with a
# multi-line guard block written as if it started at column zero. The result
# dedented `return` out of its function — and ast.parse ACCEPTED it, so the
# syntax gate let a file through that Python refuses to run.

FUNC = 'def parse_port(text):\n    """Parse."""\n    return int(text)\n'
BLOCK = "text = text.strip()\nif not text.isdigit():\n    return None\nreturn int(text)"


def test_reindent_lifts_a_flat_block_into_the_function():
    lifted = reindent_replacement(FUNC, "return int(text)", BLOCK)
    result = compute_edit(FUNC, "return int(text)", lifted)
    assert result.outcome == MatchOutcome.APPLIED
    compile(result.new_content, "<t>", "exec")  # raises if still broken


def test_reindent_preserves_relative_nesting():
    """A nested body must stay nested, not be flattened to the base depth."""
    fixed = reindent_replacement(FUNC, "return int(text)", BLOCK)
    lines = fixed.split("\n")
    assert lines[1] == "    if not text.isdigit():"
    assert lines[2] == "        return None"  # one level deeper, preserved


def test_reindent_leaves_single_line_replacements_alone():
    assert reindent_replacement(FUNC, "return int(text)", "return 0") == "return 0"


def test_reindent_leaves_blank_lines_blank():
    fixed = reindent_replacement(FUNC, "return int(text)", "a = 1\n\nb = 2")
    assert "\n\n" in fixed  # no whitespace-only padding introduced


def test_reindent_is_a_noop_at_column_zero():
    flat = "x = 1\ny = 2\n"
    assert reindent_replacement(flat, "x = 1", "a = 1\nb = 2") == "a = 1\nb = 2"


def test_reindent_ignores_a_mid_line_match():
    content = "value = compute(1)\n"
    assert reindent_replacement(content, "compute(1)", "a\nb") == "a\nb"


def test_compile_catches_what_ast_parse_missed():
    """The gate now uses compile(), which rejects return/break/yield outside
    their block — ast.parse accepts all three."""
    from forge.tools.syntax_check import syntax_error

    for source in ["return 1\n", "break\n", "yield 2\n"]:
        assert syntax_error("m.py", source) is not None, source
    assert syntax_error("m.py", "def f():\n    return 1\n") is None
