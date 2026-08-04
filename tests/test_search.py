"""Execution-guided candidate search: isolation, scoring, and the fallbacks.

The benchmark showed mechanical failure was essentially gone while task
success stalled — the agent acts cleanly on the wrong thing. Gates cannot
supply a better idea; trying more than once can. These tests pin the part
that must be exactly right: a losing candidate leaves NOTHING behind."""

from forge.verify.search import (
    Candidate,
    _search_temperatures,
    capture,
    search,
)


def _seed(root):
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "keep.txt").write_text("untouched\n", encoding="utf-8")
    sub = root / "pkg"
    sub.mkdir(exist_ok=True)
    (sub / "b.py").write_text("y = 2\n", encoding="utf-8")


# -- snapshot / restore -----------------------------------------------------------


def test_restore_undoes_modifications(tmp_path):
    _seed(tmp_path)
    snap = capture(tmp_path)
    (tmp_path / "a.py").write_text("RUINED\n", encoding="utf-8")
    snap.restore()
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_restore_deletes_files_a_candidate_created(tmp_path):
    _seed(tmp_path)
    snap = capture(tmp_path)
    (tmp_path / "litter.py").write_text("junk\n", encoding="utf-8")
    snap.restore()
    assert not (tmp_path / "litter.py").exists()


def test_restore_recreates_a_deleted_file(tmp_path):
    _seed(tmp_path)
    snap = capture(tmp_path)
    (tmp_path / "a.py").unlink()
    snap.restore()
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_restore_is_byte_exact_including_line_endings(tmp_path):
    (tmp_path / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    snap = capture(tmp_path)
    (tmp_path / "crlf.py").write_bytes(b"changed\n")
    snap.restore()
    assert (tmp_path / "crlf.py").read_bytes() == b"x = 1\r\ny = 2\r\n"


def test_snapshot_skips_noise_directories(tmp_path):
    _seed(tmp_path)
    junk = tmp_path / "node_modules" / "dep"
    junk.mkdir(parents=True)
    (junk / "huge.js").write_text("x" * 1000, encoding="utf-8")
    snap = capture(tmp_path)
    assert not any("node_modules" in str(p) for p in snap.files)


def test_changed_since_reports_only_real_changes(tmp_path):
    _seed(tmp_path)
    snap = capture(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    changed = [p.name for p in snap.changed_since()]
    assert changed == ["a.py"]


# -- temperatures -----------------------------------------------------------------


def test_temperatures_start_at_default_then_diversify():
    temps = _search_temperatures(0.2, 3)
    assert temps[0] is None  # first attempt uses the configured setting
    assert temps[1:] == [0.5, 0.8]
    assert len(set(map(str, temps))) == 3  # genuinely different samples


def test_temperatures_are_capped():
    assert all(t is None or t <= 0.9 for t in _search_temperatures(0.8, 5))


def test_single_candidate_is_just_one_attempt():
    assert _search_temperatures(0.2, 1) == [None]


# -- scoring ----------------------------------------------------------------------


def test_satisfying_the_requirement_dominates():
    did_job = Candidate(index=0, temperature=None, satisfied=True, changed=["a", "b", "c"])
    tidy_but_wrong = Candidate(index=1, temperature=0.5, satisfied=False, changed=["a"])
    assert did_job.score > tidy_but_wrong.score


def test_smaller_change_wins_a_tie():
    big = Candidate(index=0, temperature=None, satisfied=True, verified=True,
                    changed=["a", "b", "c"])
    small = Candidate(index=1, temperature=0.5, satisfied=True, verified=True,
                      changed=["a"])
    assert small.score > big.score


# -- the search loop --------------------------------------------------------------


def test_only_the_winning_candidate_survives(tmp_path):
    """The heart of it: two attempts run, one wins, and the loser's edits are
    nowhere to be found."""
    _seed(tmp_path)

    def attempt(index, temperature):
        if index == 0:
            (tmp_path / "a.py").write_text("BAD ATTEMPT\n", encoding="utf-8")
            (tmp_path / "loser_litter.py").write_text("junk\n", encoding="utf-8")
            return True, False  # changed something, did not satisfy
        (tmp_path / "a.py").write_text("x = 99\n", encoding="utf-8")
        return True, True

    winner = search(tmp_path, attempt, [None, 0.5])
    assert winner is not None and winner.index == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 99\n"
    assert not (tmp_path / "loser_litter.py").exists()
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "untouched\n"


def test_search_stops_early_once_a_candidate_succeeds(tmp_path):
    _seed(tmp_path)
    tried = []

    def attempt(index, temperature):
        tried.append(index)
        (tmp_path / "a.py").write_text(f"x = {index}\n", encoding="utf-8")
        return True, True  # first one already satisfies

    search(tmp_path, attempt, [None, 0.5, 0.8])
    assert tried == [0]  # no compute wasted beating a good answer


def test_a_crashing_candidate_does_not_break_the_search(tmp_path):
    _seed(tmp_path)

    def attempt(index, temperature):
        if index == 0:
            (tmp_path / "a.py").write_text("half written", encoding="utf-8")
            raise RuntimeError("model died")
        (tmp_path / "a.py").write_text("x = 7\n", encoding="utf-8")
        return True, True

    winner = search(tmp_path, attempt, [None, 0.5])
    assert winner is not None and winner.index == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 7\n"


def test_no_candidate_changed_anything(tmp_path):
    _seed(tmp_path)
    winner = search(tmp_path, lambda i, t: (False, False), [None, 0.5])
    assert winner is None
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_best_of_failures_still_wins_if_it_changed_something(tmp_path):
    """When nothing satisfies, the attempt that at least did work is kept —
    that matches single-attempt behaviour rather than losing the work."""
    _seed(tmp_path)

    def attempt(index, temperature):
        (tmp_path / "a.py").write_text(f"attempt {index}\n", encoding="utf-8")
        return True, False

    winner = search(tmp_path, attempt, [None, 0.5])
    assert winner is not None
    assert (tmp_path / "a.py").read_text(encoding="utf-8").startswith("attempt")


def test_search_declines_on_a_workspace_too_large_to_snapshot(tmp_path, monkeypatch):
    """Returning None tells the caller to run once instead — never pretend a
    partial snapshot can be restored."""
    import forge.verify.search as mod

    monkeypatch.setattr(mod, "_MAX_SNAPSHOT_FILES", 2)
    for i in range(6):
        (tmp_path / f"f{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
    assert search(tmp_path, lambda i, t: (True, True), [None, 0.5]) is None
