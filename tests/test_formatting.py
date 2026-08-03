"""Structural repair of reply Markdown — the formatting failures a 7B
actually makes, and the prose that must survive untouched."""

from forge.chat.formatting import normalize_markdown


def test_unclosed_fence_is_closed():
    """An odd fence count hides everything after it in any renderer."""
    out = normalize_markdown("Here you go:\n\n```python\nx = 1\n")
    assert out.count("```") == 2
    assert out.endswith("```")


def test_balanced_fences_untouched():
    text = "Before\n\n```python\nx = 1\n```\n\nAfter"
    assert normalize_markdown(text).count("```") == 2


def test_run_on_numbered_list_is_split():
    out = normalize_markdown("1. Read the file 2. Edit it 3. Run the tests")
    lines = [line for line in out.split("\n") if line.strip()]
    assert lines == ["1. Read the file", "2. Edit it", "3. Run the tests"]


def test_prose_mentioning_numbers_is_not_split():
    """Conservative: only an ascending run starting at 1 counts as a list."""
    text = "See section 2. The parser handles 3. cases without issue."
    assert normalize_markdown(text) == text


def test_list_glued_to_paragraph_gets_separated():
    out = normalize_markdown("I changed three things:\n- one\n- two")
    assert out == "I changed three things:\n\n- one\n- two"


def test_consecutive_list_items_are_not_separated():
    out = normalize_markdown("- one\n- two\n- three")
    assert out == "- one\n- two\n- three"


def test_heading_glued_to_paragraph_gets_separated():
    out = normalize_markdown("Some intro text\n## Details\nmore")
    assert "\n\n## Details" in out


def test_fence_glued_to_paragraph_gets_separated():
    out = normalize_markdown("Run this:\n```bash\npytest\n```")
    assert out.startswith("Run this:\n\n```bash")


def test_content_inside_a_fence_is_never_reformatted():
    """Code that looks like a list must survive verbatim."""
    text = "```python\n# 1. first 2. second\n- not a list\nx = 1\n```"
    out = normalize_markdown(text)
    assert "# 1. first 2. second" in out
    assert "- not a list\nx = 1" in out


def test_trailing_whitespace_and_blank_runs_collapse():
    out = normalize_markdown("one   \n\n\n\n\ntwo")
    assert out == "one\n\ntwo"


def test_empty_and_whitespace_input_survives():
    assert normalize_markdown("") == ""
    assert normalize_markdown("   ") == "   "


def test_ordinary_prose_is_unchanged():
    text = "I read `app.py` and fixed the off-by-one in `chunk()`. Tests pass."
    assert normalize_markdown(text) == text


def test_session_normalizes_the_final_reply(workspace):
    from forge.chat.session import ChatSession
    from forge.llm.base import ChatMessage
    from forge.llm.mock import MockLLMClient

    llm = MockLLMClient(
        [ChatMessage(role="assistant", content="Steps:\n1. Read it 2. Fix it")]
    )
    reply = ChatSession(workspace, llm, session_id="fmt").send("what should I do?")
    assert reply == "Steps:\n\n1. Read it\n2. Fix it"
