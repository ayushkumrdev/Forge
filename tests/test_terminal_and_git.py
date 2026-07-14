import sys

from forge.tools.git_tool import GitTool
from forge.tools.terminal import RunCommandTool


def test_run_command_captures_output(guard, workspace):
    tool = RunCommandTool(guard, workspace)
    result = tool.run(command=f'"{sys.executable}" -c "print(40 + 2)"')
    assert result.ok
    assert "exit code: 0" in result.output
    assert "42" in result.output


def test_run_command_reports_nonzero_exit(guard, workspace):
    tool = RunCommandTool(guard, workspace)
    result = tool.run(command=f'"{sys.executable}" -c "import sys; sys.exit(3)"')
    assert result.ok  # non-zero exit is information, not a tool failure
    assert "exit code: 3" in result.output


def test_run_command_blocks_destructive(guard, workspace):
    from forge.tools.base import ToolRegistry

    registry = ToolRegistry([RunCommandTool(guard, workspace)])
    result = registry.execute("run_command", {"command": "rm -rf /"})
    assert not result.ok
    assert "blocked" in result.error.lower()


def test_run_command_timeout(guard, workspace):
    tool = RunCommandTool(guard, workspace)
    result = tool.run(
        command=f'"{sys.executable}" -c "import time; time.sleep(5)"', timeout_s=1
    )
    assert not result.ok
    assert "timed out" in result.error


def test_git_disallowed_subcommand(workspace):
    result = GitTool(workspace).run(args="push origin main")
    assert not result.ok
    assert "not allowed" in result.error


def test_git_status_in_fresh_repo(workspace):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    result = GitTool(workspace).run(args="status --short")
    assert result.ok
    assert "exit code: 0" in result.output
