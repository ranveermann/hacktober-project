import json

import pytest

from gittrace.provenance import ProvenanceIndex, Relation
from gittrace.status import deploy_status, test_status as compute_status
from gittrace.store import Store, ValidationError

from conftest import BASE

FIXED = BASE.replace("return x * 0.1", "return round(x * 0.18, 2)")


def test_record_stores_commit_and_tree(base_repo):
    r = base_repo
    c = r.commit("fix", **{"app.py": FIXED})
    rec = r.store.add_test(c[:8], "UAT", "pass", suite="regression", run_id="R1", metadata={"ci": "jenkins"})
    assert rec.commit == c and rec.environment == "uat" and rec.result == "passed"
    assert rec.tree == r.git_cmd("rev-parse", f"{c}^{{tree}}").stdout.strip()
    data = json.loads((r.git.common_dir / "gittrace" / "records.json").read_text())
    assert data["tests"][0]["run_id"] == "R1" and data["tests"][0]["metadata"] == {"ci": "jenkins"}


def test_validation(base_repo):
    with pytest.raises(ValidationError):
        base_repo.store.add_test("HEAD", "qa-lab", "passed")
    with pytest.raises(ValidationError):
        base_repo.store.add_test("HEAD", "uat", "green")
    with pytest.raises(ValidationError):
        base_repo.store.add_test("deadbeef", "uat", "passed")


def test_exact_status_and_latest_wins(base_repo):
    r = base_repo
    c = r.commit("fix", **{"app.py": FIXED})
    r.store.add_test(c, "uat", "failed", timestamp="2026-01-01T00:00:00+00:00")
    r.store.add_test(c, "uat", "passed", timestamp="2026-01-02T00:00:00+00:00")
    st = compute_status(r.git, r.store, None, c)
    assert st.by_environment == {"development": "untested", "uat": "passed", "production": "untested"}


def test_same_tree_counts_as_same_snapshot(base_repo):
    r = base_repo
    c = r.commit("fix", **{"app.py": FIXED})
    r.git_cmd("commit", "-q", "--amend", "-m", "fix (reworded)")
    amended = r.head()
    assert amended != c
    r.store.add_test(c, "uat", "passed")
    st = compute_status(r.git, r.store, None, amended)
    assert st.by_environment["uat"] == "passed" and st.same_snapshot and not st.exact


def test_backport_does_not_inherit_test_status(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.store.add_test(orig, "uat", "passed")
    r.checkout("uat")
    r.git_cmd("cherry-pick", "-x", orig)
    bp = r.head()
    st = compute_status(r.git, r.store, ProvenanceIndex(r.git), bp)
    assert st.by_environment["uat"] == "untested"  # different snapshot!
    assert st.related and st.related[0].relationship.relation is Relation.REFERENCED
    assert st.related_by_environment["uat"] == "passed"


def test_deployment_via_branch_ancestry_cherry_and_backport(base_repo):
    r = base_repo
    store = r.store
    store.set_env_branch("uat", "uat")
    store.set_env_branch("production", "production")
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", orig)  # equivalent patch, no reference
    r.checkout("production")
    r.commit(f"Fix GST prod\n\n(cherry picked from commit {orig})", **{"app.py": FIXED.replace("2)", "2)  # prod")})
    r.checkout("development")
    idx = ProvenanceIndex(r.git)
    ds = {f.environment: f for f in deploy_status(r.git, store, idx, orig).findings}
    assert ds["development"].state == "unconfigured"
    assert ds["uat"].state == "equivalent-present"
    assert ds["production"].state == "backport-present" and "MODIFIED" in ds["production"].detail
    store.set_env_branch("development", "development")
    ds = {f.environment: f for f in deploy_status(r.git, store, idx, orig).findings}
    assert ds["development"].state == "present"


def test_recorded_deployment_wins(base_repo):
    r = base_repo
    c = r.commit("fix", **{"app.py": FIXED})
    later = r.commit("more", **{"README.md": "x\n"})
    r.store.add_deploy(later, "production", reference="v1.2.0")
    f = {x.environment: x for x in deploy_status(r.git, r.store, None, c).findings}["production"]
    assert f.state == "deployed" and f.strength == "recorded" and later[:10] in f.detail


def test_export_import_roundtrip(base_repo, tmp_path):
    r = base_repo
    r.store.add_test("HEAD", "uat", "passed")
    dest = tmp_path / "records.json"
    r.store.export(dest)
    fresh = Store(r.git)
    fresh.data["tests"].clear()
    fresh.save()
    assert Store(r.git).import_(dest) == (1, 0)
    assert Store(r.git).import_(dest) == (0, 1)
