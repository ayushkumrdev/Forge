"""The SWE-bench adapter: external calibration for the SWE-micro numbers.

A benchmark we wrote ourselves cannot be the only thing we report. These
tests cover the parts that do not need Docker or the network — the scoring
criterion above all, because getting that wrong would make every number
meaningless.
"""

import json
from pathlib import Path

from forge.evals.swebench import (
    Instance,
    InstanceResult,
    SWEBenchReport,
    _outcomes,
    _request,
    evaluate,
    write_report,
)


def _instance(**overrides) -> Instance:
    base = {
        "instance_id": "psf__requests-1234",
        "repo": "psf/requests",
        "base_commit": "abc123",
        "problem_statement": "Session.get drops the timeout argument.",
        "fail_to_pass": ["tests/test_x.py::test_a"],
        "pass_to_pass": ["tests/test_x.py::test_b"],
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
    }
    return Instance(**{**base, **overrides})


def test_the_image_name_matches_the_published_convention():
    """swebench mangles the id: django__django-11099 becomes
    django_1776_django-11099. Getting this wrong means every pull 404s."""
    assert _instance(instance_id="django__django-11099").image == (
        "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    )


def test_the_request_asks_for_a_fix_not_a_discussion():
    text = _request(_instance())
    assert "Session.get drops the timeout" in text
    assert "Do not write tests" in text  # they already exist and are held out


def test_a_huge_problem_statement_is_clipped():
    text = _request(_instance(problem_statement="x" * 50_000))
    assert len(text) < 8_000


def test_an_empty_patch_is_not_evaluated(tmp_path):
    """No patch is a different failure from a wrong patch, and it must never
    reach Docker."""
    result = evaluate(_instance(), "   \n", tmp_path)
    assert not result.resolved
    assert "no patch" in result.error


def test_outcomes_are_read_from_the_pytest_report():
    output = (
        "PASSED tests/test_x.py::test_a\n"
        "FAILED tests/test_x.py::test_b\n"
        "ERROR tests/test_x.py::test_c\n"
    )
    assert _outcomes(output) == {
        "tests/test_x.py::test_a": "PASSED",
        "tests/test_x.py::test_b": "FAILED",
        "tests/test_x.py::test_c": "ERROR",
    }


def test_resolved_requires_every_target_test_and_no_regression():
    """The official criterion. A patch that fixes the bug and breaks
    something else is not a resolved instance."""
    fixed_and_clean = InstanceResult(
        instance_id="x", fail_to_pass_passed=2, fail_to_pass_total=2, pass_to_pass_broken=0
    )
    fixed_and_clean.resolved = True
    partial = InstanceResult(
        instance_id="y", fail_to_pass_passed=1, fail_to_pass_total=2
    )
    regressed = InstanceResult(
        instance_id="z", fail_to_pass_passed=2, fail_to_pass_total=2, pass_to_pass_broken=1
    )
    report = SWEBenchReport(results=[fixed_and_clean, partial, regressed])
    assert report.summary()["resolved"] == 1
    assert report.resolved_rate() == 1 / 3


def test_the_report_separates_no_patch_from_a_wrong_patch(tmp_path: Path):
    report = SWEBenchReport(results=[
        InstanceResult(instance_id="a", patch="diff --git ...", error="0/1 target tests pass"),
        InstanceResult(instance_id="b", patch="", error="the agent produced no patch"),
    ])
    destination = tmp_path / "report.json"
    write_report(report, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["summary"]["produced_a_patch"] == 1
    assert payload["summary"]["instances"] == 2
    assert payload["results"][0]["patch_chars"] > 0


def test_an_empty_report_does_not_divide_by_zero():
    assert SWEBenchReport().resolved_rate() == 0.0
    assert SWEBenchReport().summary()["instances"] == 0
