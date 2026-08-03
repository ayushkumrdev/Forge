"""GitHub intelligence tools: repo analysis, file reads, spec parsing, and
friendly failure modes — all against a mocked GitHub API."""


import httpx

from forge.tools.github import GitHubFileTool, GitHubRepoTool, parse_repo


def test_parse_repo_accepts_common_forms():
    assert parse_repo("ayushkumrdev/Forge") == ("ayushkumrdev", "Forge")
    assert parse_repo("https://github.com/psf/requests") == ("psf", "requests")
    assert parse_repo("https://github.com/psf/requests.git") == ("psf", "requests")
    assert parse_repo("git@github.com:psf/requests.git") == ("psf", "requests")
    assert parse_repo("https://github.com/psf/requests/tree/main/src") == ("psf", "requests")
    assert parse_repo("not a repo at all !") is None


def _repo_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/repos/psf/requests":
        return httpx.Response(
            200,
            json={
                "full_name": "psf/requests",
                "description": "A simple HTTP library.",
                "language": "Python",
                "stargazers_count": 52000,
                "forks_count": 9000,
                "default_branch": "main",
                "license": {"spdx_id": "Apache-2.0"},
                "topics": ["http", "python"],
                "clone_url": "https://github.com/psf/requests.git",
            },
        )
    if path == "/repos/psf/requests/readme":
        return httpx.Response(200, text="# Requests\nHTTP for Humans.")
    if path == "/repos/psf/requests/git/trees/main":
        return httpx.Response(
            200,
            json={
                "tree": [
                    {"path": "src", "type": "tree"},
                    {"path": "src/requests/api.py", "type": "blob"},
                    {"path": "README.md", "type": "blob"},
                ]
            },
        )
    return httpx.Response(404, json={"message": "Not Found"})


def test_github_repo_builds_architecture_overview():
    tool = GitHubRepoTool(transport=httpx.MockTransport(_repo_handler))
    result = tool.run(repo="https://github.com/psf/requests")
    assert result.ok
    assert "psf/requests" in result.output
    assert "HTTP for Humans" in result.output          # README inlined
    assert "src/requests/api.py" in result.output      # file tree
    assert "git clone https://github.com/psf/requests.git" in result.output
    assert "Apache-2.0" in result.output


def test_github_repo_not_found_is_friendly():
    tool = GitHubRepoTool(transport=httpx.MockTransport(_repo_handler))
    result = tool.run(repo="nobody/nothing")
    assert not result.ok
    assert "not found" in result.error.lower()


def test_github_repo_rate_limit_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, headers={"X-RateLimit-Remaining": "0"}, json={"message": "rate limited"}
        )

    result = GitHubRepoTool(transport=httpx.MockTransport(handler)).run(repo="a/b")
    assert not result.ok
    assert "FORGE_GITHUB_TOKEN" in result.error


def test_github_file_reads_raw_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/psf/requests/contents/src/requests/api.py"
        assert request.url.params.get("ref") == "v2.31.0"
        return httpx.Response(200, text="def get(url, **kwargs):\n    ...\n")

    tool = GitHubFileTool(transport=httpx.MockTransport(handler))
    result = tool.run(repo="psf/requests", path="src/requests/api.py", ref="v2.31.0")
    assert result.ok
    assert result.output.startswith("[psf/requests:src/requests/api.py @ v2.31.0]")
    assert "def get(url" in result.output


def test_github_token_sent_as_bearer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, text="x = 1\n")

    GitHubFileTool(token="ghp_abc", transport=httpx.MockTransport(handler)).run(
        repo="a/b", path="x.py"
    )
    assert seen["auth"] == "Bearer ghp_abc"


def test_tools_are_read_only_and_registered(workspace):
    from forge.chat.session import ChatSession
    from forge.llm.mock import MockLLMClient

    assert GitHubRepoTool.mutating is False
    assert GitHubFileTool.mutating is False
    session = ChatSession(workspace, MockLLMClient([]), session_id="gh-test")
    names = session.registry.names()
    assert "github_repo" in names
    assert "github_file" in names
