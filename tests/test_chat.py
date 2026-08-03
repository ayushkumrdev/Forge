"""Milestone 6 tests: the interactive chat session — multi-turn tool loop,
permissions, @mentions, compaction, transcript resume, undo, slash commands,
and project instructions."""

from forge.chat import commands
from forge.chat.instructions import load_project_instructions
from forge.chat.session import ChatSession
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.safety.permissions import PermissionPolicy


def _session(workspace, llm, policy=None, session_id="chat-test") -> ChatSession:
    return ChatSession(workspace, llm, policy=policy, session_id=session_id)


def _write_call(path="hello.py", content="print('hi')\n") -> ChatMessage:
    return ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="write_file", arguments={"path": path, "content": content})],
    )


# -- conversation ----------------------------------------------------------------


def test_multi_turn_history_persists(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="Hello! How can I help?"),
            _write_call(),
            ChatMessage(role="assistant", content="Created hello.py."),
        ]
    )
    session = _session(workspace, llm)

    reply1 = session.send("hi there")
    assert reply1 == "Hello! How can I help?"

    reply2 = session.send("create hello.py")
    assert "hello.py" in reply2
    assert (workspace / "hello.py").exists()

    # second turn's request must include the first turn's messages
    second_request = llm.requests[1]
    contents = [m.content for m in second_request]
    assert "hi there" in contents
    assert "Hello! How can I help?" in contents
    # and the tool result flowed back to the model
    final_request = llm.requests[2]
    assert any(m.role == "tool" for m in final_request)


def test_inline_json_tool_calls_recovered_in_chat(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                content='{"name": "write_file", "arguments": '
                '{"path": "x.txt", "content": "inline"}}',
            ),
            ChatMessage(role="assistant", content="done"),
        ]
    )
    _session(workspace, llm).send("write x.txt")
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "inline"


def test_should_stop_short_circuits_before_any_llm_call(workspace):
    llm = MockLLMClient([ChatMessage(role="assistant", content="never reached")])
    session = _session(workspace, llm)
    session.should_stop = lambda: True
    assert session.send("do something") == "Stopped by user."
    assert llm.requests == []


def test_on_stream_receives_final_text(workspace):
    llm = MockLLMClient([ChatMessage(role="assistant", content="streamed reply")])
    session = _session(workspace, llm)
    deltas: list[str] = []
    session.on_stream = deltas.append
    session.send("hi")
    assert "".join(deltas) == "streamed reply"


def test_template_token_leak_gets_one_nudge(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="<|im_start|>\n"),
            ChatMessage(role="assistant", content="Here is the real answer."),
        ]
    )
    session = _session(workspace, llm)
    assert session.send("do something") == "Here is the real answer."
    # the nudge went into history as a user message
    assert any("empty" in m.content for m in llm.requests[1] if m.role == "user")


# -- permissions -----------------------------------------------------------------


def test_ask_mode_denial_reaches_model_without_executing(workspace):
    llm = MockLLMClient(
        [_write_call(), ChatMessage(role="assistant", content="understood, stopping")]
    )
    policy = PermissionPolicy("ask", approver=lambda name, detail: False)
    session = _session(workspace, llm, policy=policy)
    session.send("write hello.py")

    assert not (workspace / "hello.py").exists()
    tool_messages = [m for m in llm.requests[-1] if m.role == "tool"]
    assert "Permission denied" in tool_messages[0].content


def test_ask_mode_approval_executes(workspace):
    llm = MockLLMClient([_write_call(), ChatMessage(role="assistant", content="done")])
    approvals = []

    def approver(name, detail):
        approvals.append(name)
        return True

    session = _session(workspace, llm, policy=PermissionPolicy("ask", approver))
    session.send("write hello.py")
    assert (workspace / "hello.py").exists()
    assert approvals == ["write_file"]


def test_read_only_tools_never_prompt(workspace):
    (workspace / "a.txt").write_text("data", encoding="utf-8")
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="read_file", arguments={"path": "a.txt"})],
            ),
            ChatMessage(role="assistant", content="read it"),
        ]
    )

    def approver(name, detail):
        raise AssertionError("read-only tool must not prompt")

    session = _session(workspace, llm, policy=PermissionPolicy("ask", approver))
    assert session.send("read a.txt") == "read it"


def test_allow_always_skips_future_prompts(workspace):
    policy = PermissionPolicy("ask", approver=lambda n, d: False)
    policy.allow_always("write_file")
    assert policy.check("write_file", True, {}) is None


# -- mentions -------------------------------------------------------------------


def test_file_mentions_inline_content(workspace):
    (workspace / "config.toml").write_text("key = 'value'\n", encoding="utf-8")
    llm = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = _session(workspace, llm)
    session.send("explain @config.toml please")

    user_message = llm.requests[0][-1].content
    assert "key = 'value'" in user_message
    assert "content of config.toml" in user_message


def test_unknown_mentions_are_ignored(workspace):
    llm = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    _session(workspace, llm).send("look at @nope/missing.py")
    # no crash, message goes through unchanged apart from no expansion


# -- compaction ------------------------------------------------------------------


def test_compaction_summarizes_old_history(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="summary of the early conversation"),
            ChatMessage(role="assistant", content="final answer"),
        ]
    )
    session = _session(workspace, llm)
    session.settings.chat_compact_threshold = 10
    # seed a long history
    for i in range(20):
        session.history.append(ChatMessage(role="user", content=f"message {i}"))
        session.history.append(ChatMessage(role="assistant", content=f"reply {i}"))

    session.send("next question")

    assert session.history[0].role == "system"
    assert "summary of the early conversation" in session.history[0].content
    assert len(session.history) < 15


# -- persistence and undo ---------------------------------------------------------


def test_transcript_save_and_resume(workspace):
    llm = MockLLMClient([ChatMessage(role="assistant", content="hello!")])
    session = _session(workspace, llm)
    session.send("hi")

    fresh = _session(workspace, MockLLMClient([]), session_id="chat-test")
    assert fresh.load_transcript()
    assert any(m.content == "hello!" for m in fresh.history)


def test_undo_restores_files(workspace):
    (workspace / "keep.py").write_text("original\n", encoding="utf-8")
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "keep.py", "content": "changed"},
                    ),
                    ToolCall(
                        name="write_file",
                        arguments={"path": "new.py", "content": "brand = 'new'"},
                    ),
                ],
            ),
            ChatMessage(role="assistant", content="done"),
        ]
    )
    session = _session(workspace, llm)
    session.send("change files")
    assert (workspace / "keep.py").read_text(encoding="utf-8") == "changed"

    restored = session.undo()
    assert set(restored) == {"keep.py", "new.py"}
    assert (workspace / "keep.py").read_text(encoding="utf-8") == "original\n"
    assert not (workspace / "new.py").exists()


def test_token_pressure_triggers_compaction(workspace):
    """Long tool outputs must trigger compaction well before the message-count
    threshold — the estimated token footprint is the real constraint."""
    from forge.config import ForgeSettings

    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="summary of the earlier work"),
            ChatMessage(role="assistant", content="continuing"),
        ]
    )
    settings = ForgeSettings(num_ctx=400, chat_compact_threshold=30)
    session = ChatSession(workspace, llm, settings=settings, session_id="tok-test")
    # 10 messages (< threshold) but each huge relative to num_ctx=400
    for i in range(10):
        session.history.append(ChatMessage(role="user", content=f"m{i} " + "x" * 500))

    session.send("continue")
    assert any(
        m.role == "system" and "compacted" in m.content for m in session.history
    )
    assert session.history[-1].content == "continuing"


# -- slash commands ---------------------------------------------------------------


def test_slash_command_dispatch(workspace):
    session = _session(workspace, MockLLMClient([]))
    assert commands.is_command("/help")
    assert not commands.is_command("hello /help")

    assert "/undo" in commands.execute(session, "/help").text
    assert commands.execute(session, "/exit").should_exit
    assert "Unknown command" in commands.execute(session, "/bogus").text
    assert "No file changes" in commands.execute(session, "/diff").text
    assert "Nothing to undo" in commands.execute(session, "/undo").text


def test_model_command_switches_model(workspace):
    llm = MockLLMClient([])
    llm.model = "qwen2.5-coder:7b"
    session = _session(workspace, llm)
    assert "qwen2.5-coder:7b" in commands.execute(session, "/model").text
    commands.execute(session, "/model qwen2.5-coder:14b")
    assert llm.model == "qwen2.5-coder:14b"


def test_diff_command_shows_changes(workspace):
    llm = MockLLMClient([_write_call(), ChatMessage(role="assistant", content="done")])
    session = _session(workspace, llm)
    session.send("write hello.py")
    assert "+print('hi')" in commands.execute(session, "/diff").text


# -- project instructions -----------------------------------------------------------


def test_project_instructions_loaded_into_system_prompt(workspace):
    (workspace / "FORGE.md").write_text("Always use tabs, never spaces.", encoding="utf-8")
    llm = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = _session(workspace, llm)
    session.send("hi")
    system = llm.requests[0][0]
    assert system.role == "system"
    assert "Always use tabs, never spaces." in system.content


def test_instructions_missing_is_empty(workspace):
    assert load_project_instructions(workspace) == ""
