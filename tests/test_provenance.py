from gittrace.provenance import ProvenanceIndex, Relation, parse_references

from conftest import BASE

FIXED = BASE.replace("return x * 0.1", "return round(x * 0.18, 2)")


def test_parse_references_variants():
    msg = "Fix\n\n(cherry picked from commit abcdef1234567)\nBackport-of: 1234567abc\n(backported from commit fedcba9)"
    kinds = parse_references(msg)
    assert ("abcdef1234567", "cherry-pick -x line") in kinds
    assert ("1234567abc", "Backport-of trailer") in kinds
    assert ("fedcba9", "backport note") in kinds


def test_exact(base_repo):
    c = base_repo.commit("x", **{"app.py": FIXED})
    idx = ProvenanceIndex(base_repo.git)
    assert idx.classify(c, c).relation is Relation.EXACT


def test_unmodified_cherry_pick_x_is_referenced_and_unmodified(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", "-x", orig)
    bp = r.head()
    assert bp != orig
    rel = ProvenanceIndex(r.git).classify(bp, orig)
    assert rel.relation is Relation.REFERENCED
    assert rel.modified is False
    assert any("patch-id also identical" in e for e in rel.evidence)


def test_modified_backport_is_referenced_but_flagged_modified(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    bp = r.commit(f"Fix GST\n\n(cherry picked from commit {orig})", **{"app.py": FIXED.replace("2)", "2)  # uat")})
    rel = ProvenanceIndex(r.git).classify(bp, orig)
    assert rel.relation is Relation.REFERENCED and rel.modified is True
    assert any("MODIFIED" in e for e in rel.evidence)


def test_reference_chain_is_transitive(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", "-x", orig)
    uat = r.head()
    r.checkout("production")
    # names only the uat copy (git cherry-pick -x would also keep the original line)
    prod = r.commit(f"Fix GST\n\nBackport-of: {uat[:12]}", **{"app.py": FIXED})
    rel = ProvenanceIndex(r.git).classify(prod, orig)
    assert rel.relation is Relation.REFERENCED
    assert any("chain of 2 hops" in e for e in rel.evidence)


def test_equivalent_patch_without_reference(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", orig)  # no -x: no reference in message
    copy = r.head()
    rel = ProvenanceIndex(r.git).classify(copy, orig)
    assert rel.relation is Relation.EQUIVALENT and rel.confidence == "high"
    assert any("does not prove" in e for e in rel.evidence)


def test_whitespace_only_difference_is_still_equivalent(base_repo):
    r = base_repo
    orig = r.commit("Fix", **{"app.py": FIXED})
    r.checkout("uat")
    copy = r.commit("Fix (uat)", **{"app.py": FIXED.replace("round(x * 0.18, 2)", "round(x  *  0.18, 2)")})
    assert ProvenanceIndex(r.git).classify(copy, orig).relation is Relation.EQUIVALENT


def test_modified_copy_without_reference_is_only_probable(base_repo):
    r = base_repo
    orig = r.commit("Fix GST rate", **{"app.py": FIXED})
    r.checkout("uat")
    copy = r.commit("Fix GST rate for uat", **{"app.py": FIXED.replace("2)", "2)  # adjusted")})
    rel = ProvenanceIndex(r.git).classify(copy, orig)
    assert rel.relation is Relation.PROBABLE  # never claimed equivalent
    assert any("not proof" in e for e in rel.evidence)


def test_competing_edits_of_same_line_are_not_related(base_repo):
    r = base_repo
    a = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("production")
    b = r.commit("Hotfix tax", **{"app.py": BASE.replace("x * 0.1", "x * 0.12")})
    assert ProvenanceIndex(r.git).classify(a, b).relation is Relation.UNKNOWN


def test_unrelated_is_unknown(base_repo):
    r = base_repo
    a = r.commit("tax", **{"app.py": FIXED})
    b = r.commit("docs", **{"README.md": "hello world docs\n"})
    assert ProvenanceIndex(r.git).classify(a, b).relation is Relation.UNKNOWN


def test_lineage_lists_related_strongest_first(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", "-x", orig)
    bp = r.head()
    r.checkout("production")
    r.git_cmd("cherry-pick", orig)
    eq = r.head()
    lin = ProvenanceIndex(r.git).lineage(orig)
    rels = {x.target: x.relation for x in lin}
    assert rels[bp] is Relation.REFERENCED
    assert rels[eq] is Relation.EQUIVALENT
    assert lin[0].relation is Relation.REFERENCED


def test_git_cherry_wrapper(base_repo):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    other = r.commit("docs", **{"README.md": "x\n"})
    r.checkout("uat")
    r.git_cmd("cherry-pick", orig)
    rows = dict((c, s) for s, c in ProvenanceIndex(r.git).cherry("uat", "development"))
    assert rows[orig] == "-" and rows[other] == "+"
