"""web_search tool: DuckDuckGo HTML parsing, redirect decoding, failure modes."""

import httpx

from forge.tools.web import WebSearchTool, _decode_ddg_href

_DDG_PAGE = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&amp;rut=abc">
     asyncio — Asynchronous I/O</a>
  <a class="result__snippet" href="#">High-level <b>async</b> framework.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://realpython.com/async-io-python/">
     Async IO in Python</a>
  <a class="result__snippet" href="#">A complete walkthrough.</a>
</div>
</body></html>
"""


def test_decode_ddg_redirect():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%20page&rut=x"
    assert _decode_ddg_href(href) == "https://example.com/a page"
    assert _decode_ddg_href("https://direct.example.com/") == "https://direct.example.com/"


def test_web_search_parses_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "python asyncio"
        return httpx.Response(200, text=_DDG_PAGE)

    tool = WebSearchTool(transport=httpx.MockTransport(handler))
    result = tool.run(query="python asyncio")
    assert result.ok
    assert "asyncio — Asynchronous I/O" in result.output
    assert "https://docs.python.org/3/library/asyncio.html" in result.output  # decoded
    assert "realpython.com" in result.output
    assert "High-level async framework." in result.output  # snippet, tags stripped


def test_web_search_respects_max_results():
    tool = WebSearchTool(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=_DDG_PAGE))
    )
    result = tool.run(query="q", max_results=1)
    assert result.ok
    assert "realpython.com" not in result.output


def test_web_search_no_results_is_ok_not_error():
    tool = WebSearchTool(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html></html>"))
    )
    result = tool.run(query="zxqv")
    assert result.ok
    assert "No results" in result.output


def test_web_search_http_error_is_friendly():
    tool = WebSearchTool(
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
    )
    result = tool.run(query="q")
    assert not result.ok
    assert "503" in result.error


def test_web_search_registered_read_only(workspace):
    from forge.chat.session import ChatSession
    from forge.llm.mock import MockLLMClient

    assert WebSearchTool.mutating is False
    session = ChatSession(workspace, MockLLMClient([]), session_id="ws-test")
    assert "web_search" in session.registry.names()
