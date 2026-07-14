import pytest

from forge.llm.json_utils import JSONExtractionError, extract_json, extract_tool_call


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = 'Here is the plan:\n```json\n{"tasks": [1, 2]}\n```\nDone.'
    assert extract_json(text) == {"tasks": [1, 2]}


def test_embedded_json_with_prose():
    text = 'Sure! The answer is {"approved": true, "issues": []} as requested.'
    assert extract_json(text) == {"approved": True, "issues": []}


def test_nested_braces_and_strings():
    text = 'result: {"msg": "use {braces} wisely", "n": {"x": 1}} trailing'
    assert extract_json(text) == {"msg": "use {braces} wisely", "n": {"x": 1}}


def test_no_json_raises():
    with pytest.raises(JSONExtractionError):
        extract_json("there is no json here")


def test_extract_tool_call_plain_json():
    call = extract_tool_call('{"name": "edit_file", "arguments": {"path": "a.py"}}')
    assert call is not None
    assert call.name == "edit_file"
    assert call.arguments == {"path": "a.py"}


def test_extract_tool_call_qwen_tags_and_fences():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "x"}}\n</tool_call>'
    call = extract_tool_call(text)
    assert call is not None and call.name == "read_file"

    fenced = '```json\n{"name": "grep", "arguments": {"pattern": "def"}}\n```'
    call = extract_tool_call(fenced)
    assert call is not None and call.name == "grep"


def test_extract_tool_call_rejects_non_tool_json():
    assert extract_tool_call('{"approved": true, "issues": []}') is None
    assert extract_tool_call("just a plain sentence") is None
    assert extract_tool_call('{"name": "x", "arguments": "not a dict"}') is None
