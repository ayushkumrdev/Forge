"""Escape repair — the fix for a death-spiral observed live on the benchmark.

qwen2.5-coder:7b emitted edit_file arguments in which newlines arrived as the
two characters \\ and n and regex backslashes arrived doubled. Every match
tier failed, the fuzzy suggestion collapsed to a single line, and the model
re-sent the same broken needle 11 times in a row. These tests pin the repair.
"""

from forge.tools.edit_repair import (
    MatchOutcome,
    compute_edit,
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
