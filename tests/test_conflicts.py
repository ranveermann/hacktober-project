from gittrace.conflicts import Operation, detect_conflicts, detect_operation, parse_conflict_markers
from gittrace.analysis import Analyzer

from conftest import BASE


def make_merge_conflict(r):
    r.commit("dev: tax 18%", **{"app.py": BASE.replace("x * 0.1", "x * 0.18")})
    dev = r.head()
    r.checkout("production")
    r.commit("prod: tax 12%", **{"app.py": BASE.replace("x * 0.1", "x * 0.12")})
    prod = r.head()
    r.git_cmd("merge", "development", check=False)
    return dev, prod


def test_parse_markers_merge_and_diff3_styles():
    text = "a\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> feature\nb\n<<<<<<< HEAD\nx\n||||||| base\no\n=======\ny\n>>>>>>> f\n"
    hunks = parse_conflict_markers(text)
    assert len(hunks) == 2
    assert hunks[0].current_lines == ["ours"] and hunks[0].incoming_lines == ["theirs"]
    assert hunks[0].base_lines is None and hunks[0].start_line == 2 and hunks[0].end_line == 6
    assert hunks[1].base_lines == ["o"] and hunks[1].incoming_marker == "f"


def test_parse_markers_ignores_truncated_block():
    assert parse_conflict_markers("<<<<<<< HEAD\nours\n=======\ntheirs\n") == []


def test_no_conflicts(base_repo):
    assert detect_operation(base_repo.git).operation is Operation.NONE
    assert detect_conflicts(base_repo.git) == []


def test_merge_conflict_detection(base_repo):
    dev, prod = make_merge_conflict(base_repo)
    ctx = detect_operation(base_repo.git)
    assert ctx.operation is Operation.MERGE
    assert ctx.current.rev == prod and ctx.incoming.rev == dev
    assert ctx.base.rev == base_repo.git.merge_base(prod, dev)
    files = detect_conflicts(base_repo.git)
    assert [f.path for f in files] == ["app.py"]
    f = files[0]
    assert set(f.stages) == {1, 2, 3} and f.kind == "both modified"
    assert f.hunks[0].current_lines == ["    return x * 0.12"]
    assert f.hunks[0].incoming_lines == ["    return x * 0.18"]


def test_analysis_does_not_modify_worktree(base_repo):
    make_merge_conflict(base_repo)
    before = base_repo.read("app.py")
    status_before = base_repo.git_cmd("status", "--porcelain").stdout
    Analyzer(base_repo.git).analyze()
    assert base_repo.read("app.py") == before
    assert base_repo.git_cmd("status", "--porcelain").stdout == status_before


def test_blame_attributes_each_side(base_repo):
    dev, prod = make_merge_conflict(base_repo)
    rep = Analyzer(base_repo.git).analyze()
    h = rep.files[0].hunks[0]
    assert [c.commit for c in h.current.commits] == [prod]
    assert [c.commit for c in h.incoming.commits] == [dev]
    assert h.current.attribution.method == "blame"


def test_cherry_pick_context_uses_parent_as_base(base_repo):
    r = base_repo
    r.commit("refactor", **{"app.py": BASE.replace("return x * 1.0", "return float(x)")})
    fix = r.commit("fix tax", **{"app.py": BASE.replace("return x * 1.0", "return float(x)").replace("x * 0.1", "x * 0.18")})
    fix_parent = r.git_cmd("rev-parse", f"{fix}^").stdout.strip()
    r.checkout("uat")
    r.commit("uat tax", **{"app.py": BASE.replace("x * 0.1", "x * 0.15")})
    r.git_cmd("cherry-pick", "-x", fix, check=False)
    ctx = detect_operation(r.git)
    assert ctx.operation is Operation.CHERRY_PICK
    assert ctx.incoming.rev == fix
    assert ctx.base.rev == fix_parent  # not the merge-base of the branches
    assert ctx.base.rev != r.git.merge_base("HEAD", fix)


def test_rebase_labels_are_swapped(base_repo):
    r = base_repo
    r.checkout("feature", create=True)
    mine = r.commit("my change", **{"app.py": BASE.replace("x * 0.1", "x * 0.2")})
    r.checkout("development")
    upstream = r.commit("upstream change", **{"app.py": BASE.replace("x * 0.1", "x * 0.3")})
    r.checkout("feature")
    r.git_cmd("rebase", "development", check=False)
    ctx = detect_operation(r.git)
    assert ctx.operation is Operation.REBASE
    assert ctx.current.rev == upstream  # stage 2 = upstream during a rebase
    assert ctx.incoming.rev == mine  # stage 3 = the commit being replayed
    assert "upstream" in ctx.current.label and "replayed" in ctx.incoming.label
    hunk = detect_conflicts(r.git)[0].hunks[0]
    assert hunk.current_lines == ["    return x * 0.3"]
    r.git_cmd("rebase", "--abort")


def test_revert_context(base_repo):
    r = base_repo
    a = r.commit("tax 18", **{"app.py": BASE.replace("x * 0.1", "x * 0.18")})
    r.commit("tax 20", **{"app.py": BASE.replace("x * 0.1", "x * 0.20")})
    r.git_cmd("revert", "--no-edit", a, check=False)
    ctx = detect_operation(r.git)
    assert ctx.operation is Operation.REVERT
    assert ctx.base.rev == a
    assert ctx.incoming.rev == r.git_cmd("rev-parse", f"{a}^").stdout.strip()


def test_modify_delete_conflict(base_repo):
    r = base_repo
    r.commit("modify", **{"app.py": BASE + "# more\n"})
    r.checkout("production")
    r.git_cmd("rm", "-q", "app.py")
    r.git_cmd("commit", "-q", "-m", "delete app")
    r.git_cmd("merge", "development", check=False)
    f = detect_conflicts(r.git)[0]
    assert f.kind == "deleted on current side, modified on incoming side"
    rep = Analyzer(r.git).analyze()
    assert any("Structural conflict" in a for a in rep.files[0].advisories)


def test_binary_conflict(base_repo):
    r = base_repo
    (r.path / "img.bin").write_bytes(b"\x00\x01base")
    r.commit("add bin")
    r.git_cmd("branch", "-f", "production")
    (r.path / "img.bin").write_bytes(b"\x00\x02dev")
    r.commit("dev bin")
    r.checkout("production")
    (r.path / "img.bin").write_bytes(b"\x00\x03prod")
    r.commit("prod bin")
    r.git_cmd("merge", "development", check=False)
    f = detect_conflicts(r.git)[0]
    assert f.binary and f.hunks == []
