"""Milestone 3 tests: tokenizer, BM25 ranking, chunking, hybrid fusion, the
Ollama embedder wire format, and the search_code tool."""

import httpx

from forge.repo.scanner import RepoScanner
from forge.retrieval.bm25 import BM25Index, tokenize
from forge.retrieval.embeddings import Embedder, OllamaEmbedder, cosine
from forge.retrieval.engine import RetrievalEngine, _chunk_file
from forge.tools.retrieval_tool import SearchCodeTool

# -- tokenizer -------------------------------------------------------------------


def test_tokenizer_splits_code_identifiers():
    tokens = tokenize("class OrderService: def submit_order(self): pass")
    assert "orderservice" in tokens
    assert "order" in tokens
    assert "service" in tokens
    assert "submit" in tokens


# -- bm25 ------------------------------------------------------------------------


def test_bm25_ranks_relevant_doc_first():
    index = BM25Index()
    index.add_documents(
        [
            "def render_template(name): return html",
            "def validate_user_session(token): check expiry and signature",
            "SELECT * FROM orders WHERE id = ?",
        ]
    )
    top = index.top("where are user sessions validated", k=2)
    assert top and top[0][0] == 1


def test_bm25_empty_query_terms():
    index = BM25Index()
    index.add_documents(["some document"])
    assert index.top("zzz_unknown_term", k=3) == []


# -- chunking --------------------------------------------------------------------


def test_symbol_aware_chunking(workspace):
    source = (
        "import os\n"
        "\n"
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )
    (workspace / "mod.py").write_text(source, encoding="utf-8")
    snapshot = RepoScanner(workspace).scan()
    file = snapshot.file("mod.py")
    chunks = _chunk_file(file, source.splitlines())

    # preamble (imports), first(), second() — three chunks at symbol boundaries
    assert [c.start_line for c in chunks] == [1, 3, 6]
    assert "def second" in chunks[2].text


def test_window_chunking_for_symbolless_files(workspace):
    lines = [f"line {i}" for i in range(1, 151)]
    (workspace / "notes.md").write_text("\n".join(lines), encoding="utf-8")
    snapshot = RepoScanner(workspace).scan()
    chunks = _chunk_file(snapshot.file("notes.md"), lines)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    # windows overlap so nothing is lost at boundaries
    assert chunks[1].start_line <= chunks[0].end_line


# -- engine ----------------------------------------------------------------------


def _build_engine(workspace, embedder=None) -> RetrievalEngine:
    (workspace / "auth.py").write_text(
        "def validate_session(token):\n"
        '    """Check session token expiry and signature."""\n'
        "    return token.is_valid()\n",
        encoding="utf-8",
    )
    (workspace / "billing.py").write_text(
        "def charge_card(amount):\n    return gateway.charge(amount)\n",
        encoding="utf-8",
    )
    engine = RetrievalEngine(workspace, embedder=embedder)
    engine.build(RepoScanner(workspace).scan())
    return engine


def test_engine_bm25_only_search(workspace):
    engine = _build_engine(workspace)
    results = engine.search("session token validation")
    assert results
    assert results[0][0].path == "auth.py"


class _FakeEmbedder(Embedder):
    """Embeds 'billing-ish' text near the query vector to test fusion."""

    def embed(self, texts):
        return [[1.0, 0.0] if "charge" in t or "billing" in t else [0.0, 1.0] for t in texts]


def test_engine_hybrid_fusion_includes_embedding_signal(workspace):
    engine = _build_engine(workspace, embedder=_FakeEmbedder())
    results = engine.search("charge payment")
    assert results[0][0].path == "billing.py"


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# -- ollama embedder -------------------------------------------------------------


def test_ollama_embedder_wire_format():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}]})
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    embedder = OllamaEmbedder("nomic-embed-text", transport=httpx.MockTransport(handler))
    assert embedder.available()
    vectors = embedder.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_embedder_unavailable_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b"}]})

    embedder = OllamaEmbedder("nomic-embed-text", transport=httpx.MockTransport(handler))
    assert not embedder.available()


# -- tool ------------------------------------------------------------------------


def test_search_code_tool_formats_results(workspace):
    engine = _build_engine(workspace)
    result = SearchCodeTool(engine).run(query="session token validation")
    assert result.ok
    assert "### auth.py:" in result.output
    assert "validate_session" in result.output


def test_search_code_tool_no_results(workspace):
    engine = RetrievalEngine(workspace)
    engine.build(RepoScanner(workspace).scan())
    result = SearchCodeTool(engine).run(query="anything")
    assert result.ok
    assert "No relevant code" in result.output
