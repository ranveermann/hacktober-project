"""Create a demo repository that is left in a merge-conflict state for GitTrace.

    python demo/make_demo_repo.py /tmp/gittrace-demo
    cd /tmp/gittrace-demo && streamlit run /path/to/gittrace/app.py

Scenario (development -> uat -> production), merging `uat` into `production`:
  Hunk 1 (tax): production has a hotfix that is LOGICALLY THE SAME fix as uat's
    backport, written differently and with no reference. Patch-ids differ and the
    heuristic can't link them; this is where Gemma helps.
  Hunk 2 (discount): uat and production changed the discount rule in genuinely
    DIFFERENT ways. Gemma should say so.
Test records: the development fix passed UAT; the production hotfix passed PROD.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gittrace.git import Git  # noqa: E402
from gittrace.store import Store  # noqa: E402

BASE = '''"""Billing helpers."""


def tax(amount):
    """GST on an amount."""
    return amount * 0.1


def shipping(weight):
    if weight < 1:
        return 40
    return 40 + (weight - 1) * 15


def describe(order):
    return f"{order['id']}: {order['total']}"


def discount(total):
    """Loyalty discount."""
    return total
'''


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_EDITOR": "true", "LC_ALL": "C"}
    return subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True, env=env).stdout.strip()


def commit(repo: Path, msg: str, content: str) -> str:
    (repo / "billing.py").write_text(content, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def main(path: str, merge: bool = True) -> None:
    repo = Path(path).resolve()
    if repo.exists() and any(repo.iterdir()):
        sys.exit(f"{repo} exists and is not empty")
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "development")
    for k, v in (("user.name", "Demo Dev"), ("user.email", "dev@example.com"), ("commit.gpgsign", "false"),
                 ("merge.conflictStyle", "merge")):
        git(repo, "config", k, v)
    commit(repo, "Initial billing module", BASE)
    git(repo, "branch", "uat")
    git(repo, "branch", "production")

    fixed_tax = BASE.replace("return amount * 0.1", "return round(amount * 0.18, 2)")
    dev_fix = commit(repo, "Fix GST: 18% rounded to 2 decimals", fixed_tax)

    git(repo, "checkout", "-q", "uat")
    uat_tax = BASE.replace("return amount * 0.1", "return round(amount * 0.18, 2)  # GST 18%")
    commit(repo, f"Fix GST: 18% rounded to 2 decimals\n\n(cherry picked from commit {dev_fix})", uat_tax)
    uat_disc = uat_tax.replace("    return total\n", "    if total > 1000:\n        return total * 0.9\n    return total\n")
    commit(repo, "Loyalty: 10% off orders above 1000", uat_disc)

    git(repo, "checkout", "-q", "production")
    prod_tax = BASE.replace("return amount * 0.1", 'gst = amount * 18 / 100\n    return float(f"{gst:.2f}")')
    prod_fix = commit(repo, "HOTFIX: customers were charged 10% GST instead of 18%", prod_tax)
    prod_disc = prod_tax.replace("    return total\n", "    if total >= 500:\n        return total * 0.95\n    return total\n")
    commit(repo, "Festive offer: 5% off from 500", prod_disc)

    store = Store(Git(repo))
    store.add_test(dev_fix, "uat", "passed", suite="billing-regression", run_id="UAT-2291", source="jenkins")
    store.add_test(prod_fix, "production", "passed", suite="smoke", run_id="PROD-77", source="jenkins")
    for env in ("development", "uat", "production"):
        store.set_env_branch(env, env)

    if merge:
        git(repo, "merge", "uat")
        print(f"Demo repo ready at {repo}, now MERGING 'uat' into 'production' with conflicts in billing.py.")
    else:
        print(f"Demo repo ready at {repo}. Run `git merge uat` to create the conflict.")
    print(f"Next: cd {repo} && streamlit run {Path(__file__).resolve().parents[1] / 'app.py'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python demo/make_demo_repo.py <empty-directory> [--no-merge]")
    main(sys.argv[1], merge="--no-merge" not in sys.argv)
