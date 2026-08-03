"""Retrieval pre-flight: real code is injected before the model generates,
BM25-gated so unrelated messages inject nothing."""

from forge.llm.base import ChatMessage
from forge.llm.mock import MockLLMClient
from forge.repo.scanner import RepoScanner
from forge.retrieval.engine import RetrievalEngine


def _engine(workspace) -> RetrievalEngine:
    engine = RetrievalEngine(workspace)
    engine.build(RepoScanner(workspace).scan())
    return engine


def test_preflight_returns_real_code_for_matching_query(workspace):
    (workspace / "billing.py").write_text(
        "def calculate_invoice_total(items):\n    return sum(i.price for i in items)\n",
        encoding="utf-8",
    )
    block = _engine(workspace).preflight("fix the invoice total calculation")
    assert "billing.py" in block
    assert "calculate_invoice_total" in block
    assert block.startswith("## Relevant code")


def test_preflight_empty_for_unrelated_query(workspace):
    (workspace / "billing.py").write_text("def pay():\n    return 1\n", encoding="utf-8")
    assert _engine(workspace).preflight("zzz qqq xyzzy nothing") == ""


def test_preflight_respects_char_budget(workspace):
    big = "\n".join(f"def invoice_helper_{i}():\n    return {i}" for i in range(400))
    (workspace / "big.py").write_text(big, encoding="utf-8")
    block = _engine(workspace).preflight("invoice helper", max_chars=800)
    assert 0 < len(block) < 1_200  # bounded (budget + header/truncation slack)


def test_chat_turn_carries_preflight_context(workspace):
    from forge.chat.session import ChatSession

    (workspace / "orders.py").write_text(
        "def submit_order(cart):\n    return cart.total\n", encoding="utf-8"
    )
    llm = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = ChatSession(workspace, llm, session_id="pf-test")
    session.send("what does submit_order do?")
    user_message = next(m for m in llm.requests[0] if m.role == "user")
    assert "auto-retrieved" in user_message.content
    assert "submit_order" in user_message.content


def test_chat_mentions_suppress_preflight(workspace):
    from forge.chat.session import ChatSession

    (workspace / "orders.py").write_text(
        "def submit_order(cart):\n    return cart.total\n", encoding="utf-8"
    )
    llm = MockLLMClient([ChatMessage(role="assistant", content="ok")])
    session = ChatSession(workspace, llm, session_id="pf-test2")
    session.send("explain @orders.py submit_order")
    user_message = next(m for m in llm.requests[0] if m.role == "user")
    assert "content of orders.py" in user_message.content  # mention inlined
    assert "auto-retrieved" not in user_message.content  # no double context
