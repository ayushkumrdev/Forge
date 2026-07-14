from forge.repo.scanner import RepoScanner
from forge.tools.search import GlobTool, GrepTool


def _make_sample_repo(workspace):
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text(
        "class Greeter:\n"
        "    def greet(self, name):\n"
        "        return f'hello {name}'\n"
        "\n"
        "def main():\n"
        "    print(Greeter().greet('world'))\n",
        encoding="utf-8",
    )
    (workspace / "README.md").write_text("# Sample\n", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("greet secret", encoding="utf-8")


def test_grep_finds_matches_and_skips_git(workspace):
    _make_sample_repo(workspace)
    result = GrepTool(workspace).run(pattern="greet")
    assert result.ok
    assert "src/app.py:2" in result.output
    assert ".git" not in result.output


def test_grep_invalid_regex(workspace):
    result = GrepTool(workspace).run(pattern="([unclosed")
    assert not result.ok
    assert "Invalid regex" in result.error


def test_glob_filters_by_extension(workspace):
    _make_sample_repo(workspace)
    result = GlobTool(workspace).run(glob="*.py")
    assert result.ok
    assert "src/app.py" in result.output
    assert "README.md" not in result.output


def test_scanner_extracts_python_symbols(workspace):
    _make_sample_repo(workspace)
    snapshot = RepoScanner(workspace).scan()

    app = next(f for f in snapshot.files if f.path == "src/app.py")
    names = {symbol.name for symbol in app.symbols}
    assert {"Greeter", "Greeter.greet", "main"} <= names
    assert snapshot.language_stats["Python"] >= 6

    summary = snapshot.summary()
    assert "src/" in summary
    assert "app.py" in summary
    assert "Greeter.greet" in summary


def test_scanner_ignores_noise_dirs(workspace):
    _make_sample_repo(workspace)
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "lib.js").write_text("x", encoding="utf-8")
    snapshot = RepoScanner(workspace).scan()
    assert all("node_modules" not in f.path for f in snapshot.files)
