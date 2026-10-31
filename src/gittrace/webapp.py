"""GitTrace Streamlit UI.

Run from inside the repository you're merging:
    cd /path/to/your/repo
    streamlit run /path/to/gittrace/app.py
or:  gittrace ui

AI: only the local Ollama server (default http://localhost:11434, model gemma4:e4b).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import streamlit as st

from .ai import DEFAULT_HOST, DEFAULT_MODEL, AIAssessment, NonLocalEndpointError, OllamaClient
from .analysis import Analyzer, Badge, HunkReport, Report, SideReport, describe_choice, side_badge
from .conflicts import Operation
from .git import Git, GitError, NotARepositoryError
from .provenance import Relationship
from .resolve import ResolveError, finish, resolve_and_finish
from .store import ENVIRONMENTS, Store, ValidationError

LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".java": "java",
        ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp",
        ".kt": "kotlin", ".swift": "swift", ".sql": "sql", ".sh": "bash", ".yml": "yaml", ".yaml": "yaml",
        ".json": "json", ".html": "html", ".css": "css", ".md": "markdown", ".toml": "toml", ".xml": "xml"}
VERDICT_STYLE = {"SAME_FIX": ("success", "🟢 Logically the SAME fix"),
                 "PARTIALLY_SAME": ("warning", "🟡 PARTIALLY the same"),
                 "DIFFERENT_CHANGES": ("error", "🔴 DIFFERENT changes"),
                 "UNCLEAR": ("info", "⚪ Unclear"),
                 "ERROR": ("error", "⚠️ Local AI unavailable")}
ENV_GUESSES = {"development": ["development", "develop", "dev"], "uat": ["uat", "staging", "qa"],
               "production": ["production", "prod", "main", "master"]}


# --------------------------------------------------------------------------- setup
def _cli_repo() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.repo or os.environ.get("GITTRACE_REPO")


def _fingerprint(git: Git, store: Store) -> str:
    parts = [git.rev_parse("HEAD") or ""]
    for name in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD"):
        parts += git.pseudo_ref(name)
    for p in (git.git_dir / "index", store.path):
        parts.append(str(p.stat().st_mtime_ns) if p.exists() else "-")
    return "|".join(parts)


def _flash(level: str, text: str) -> None:
    st.session_state.setdefault("flash", []).append((level, text))


def _show_flash() -> None:
    for level, text in st.session_state.pop("flash", []):
        getattr(st, level)(text)


def _invalidate() -> None:
    for k in ("fp", "report", "analyzer"):
        st.session_state.pop(k, None)


def _get_report(git: Git, store: Store, history_limit: int) -> tuple[Report, Analyzer, str]:
    fp = _fingerprint(git, store)
    if st.session_state.get("fp") != fp or "report" not in st.session_state:
        analyzer = Analyzer(git, store, history_limit)
        with st.spinner("Analyzing conflicts, lineage, tests and deployments..."):
            st.session_state["report"] = analyzer.analyze()
        st.session_state["analyzer"] = analyzer
        st.session_state["fp"] = fp
    return st.session_state["report"], st.session_state["analyzer"], fp


def _guess_env_branches(git: Git) -> dict[str, str]:
    out = {}
    for env, names in ENV_GUESSES.items():
        for n in names:
            if git.branch_exists(n):
                out[env] = n
                break
    return out


# ------------------------------------------------------------------------- sidebar
def _sidebar(default_repo: str) -> dict:
    sb = st.sidebar
    sb.title("🧭 GitTrace")
    sb.caption("Conflict resolution with testing provenance. AI runs locally via Ollama.")
    repo = sb.text_input("Repository path", value=st.session_state.get("repo", default_repo), key="repo_path")
    st.session_state["repo"] = repo

    sb.subheader("Local AI (Ollama)")
    ai_on = sb.toggle("Use Gemma assessment", value=True)
    host = sb.text_input("Ollama URL (must be localhost)", value=os.environ.get("GITTRACE_OLLAMA_HOST", DEFAULT_HOST), key="ollama_host")
    model = sb.text_input("Model", value=os.environ.get("GITTRACE_OLLAMA_MODEL", DEFAULT_MODEL), key="ollama_model")
    auto = sb.toggle("Ask Gemma automatically for each hunk", value=True, disabled=not ai_on)
    client = None
    if ai_on:
        try:
            client = OllamaClient(host, model)
        except NonLocalEndpointError as exc:
            sb.error(str(exc))
        if client and sb.button("Check Ollama connection"):
            h = client.health()
            if not h["reachable"]:
                sb.error(f"Not reachable: {h['error']}\n\nStart it with `ollama serve`.")
            elif not h["model_available"]:
                sb.warning(f"Ollama is running, but '{model}' isn't installed. Run `ollama pull {model}`.\n\nInstalled: {', '.join(h['models']) or 'none'}")
            else:
                sb.success(f"Connected. '{model}' is available.")
    history = sb.number_input("History scanned for lineage", 100, 100000, 5000, step=500)
    if sb.button("🔄 Refresh analysis"):
        _invalidate()
    return {"repo": repo, "client": client, "auto": auto and client is not None, "history": int(history)}


def _sidebar_repo_tools(git: Git, store: Store) -> None:
    sb = st.sidebar
    with sb.expander("Environment → branch mapping", expanded=not store.env_branches):
        guesses = _guess_env_branches(git)
        values = {}
        for env in ENVIRONMENTS:
            values[env] = st.text_input(env, value=store.env_branches.get(env, guesses.get(env, "")), key=f"envb_{env}")
        if st.button("Save mapping"):
            for env, br in values.items():
                if br.strip():
                    store.set_env_branch(env, br.strip())
                else:
                    store.unset_env_branch(env)
            _invalidate()
            st.rerun()
    with sb.expander("Record a test result"):
        with st.form("record_test", clear_on_submit=True):
            commit = st.text_input("Commit (hash or ref)")
            env = st.selectbox("Environment", ENVIRONMENTS)
            result = st.selectbox("Result", ("passed", "failed", "unknown"))
            suite = st.text_input("Test suite (optional)")
            run_id = st.text_input("Run ID (optional)")
            if st.form_submit_button("Record"):
                try:
                    rec = store.add_test(commit, env, result, suite or None, run_id or None, source="ui")
                    _flash("success", f"Recorded {rec.environment} = {rec.result} for {rec.commit[:10]} (tree {rec.tree[:10]})")
                    _invalidate()
                    st.rerun()
                except ValidationError as exc:
                    st.error(str(exc))


# --------------------------------------------------------------------- rendering
def _badge(b: Badge) -> None:
    icon = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}[b.level]
    getattr(st, b.level)(f"**{b.text}**", icon=icon)


def _side_column(title: str, side: SideReport, badge: Badge, path: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(side.label)
    _badge(badge)
    with st.expander("Provenance details"):
        for d in badge.details:
            st.markdown(f"- {d}")
        for cc in side.commits:
            st.markdown(f"**{cc.commit[:10]}** — {cc.subject}  \n_{cc.author}, {cc.date[:10]}, {cc.lines or '-'} line(s)_")
            st.table({"environment": list(cc.tests.by_environment),
                      "test status (exact snapshot)": list(cc.tests.by_environment.values()),
                      "deployment": [next((f"{f.state}: {f.detail}" for f in cc.deploy.findings if f.environment == e), "-") for e in cc.tests.by_environment]})
            for r in cc.lineage[:4]:
                _relationship(r)
            for rt in cc.tests.related[:4]:
                st.caption(f"Related evidence (does NOT certify this commit): {rt.record.commit[:10]} "
                           f"{rt.relationship.relation.value} - {rt.record.environment} {rt.record.result}")
        if side.attribution.note:
            st.caption(side.attribution.note)
    code = "\n".join(side.lines) if side.lines else "(this side removes these lines)"
    st.code(code, language=LANG.get(Path(path).suffix.lower(), None), line_numbers=False)


def _relationship(r: Relationship) -> None:
    mod = " · MODIFIED" if r.modified else ""
    st.markdown(f"- `{r.source[:10]}` ~ `{r.target[:10]}` → **{r.relation.value}**{mod} (confidence {r.confidence})")
    for ev in r.evidence:
        st.caption(f"  • {ev}")


def _ai_panel(analyzer: Analyzer, client: OllamaClient | None, auto: bool, hr: HunkReport, path: str, key: str) -> AIAssessment | None:
    with st.container(border=True):
        st.markdown("##### 🤖 Gemma assessment: is this the same fix?")
        if client is None:
            st.caption("Local AI is turned off in the sidebar. Deterministic provenance is still shown.")
            _cross(hr.cross_side)
            return None
        results = st.session_state.setdefault("ai_results", {})
        ai_key = f"{key}:{client.base}:{client.model}"
        if ai_key not in results:
            if auto or st.button("Ask Gemma", key=f"ask_{key}"):
                with st.spinner(f"Asking {client.model} on {client.base} (runs locally)..."):
                    results[ai_key] = analyzer.compare_hunk(client, hr, path)
            else:
                st.caption("Click to send both sides' commit messages and diffs to your local model.")
                _cross(hr.cross_side)
                return None
        assessment, rel = results[ai_key]
        level, label = VERDICT_STYLE.get(assessment.verdict, ("info", assessment.verdict))
        meta = f"{assessment.model} · local" + (" · cached" if assessment.cached else f" · {assessment.duration_s:.1f}s")
        getattr(st, level)(f"**{label}**" + ("" if not assessment.ok else f" — confidence {assessment.confidence:.0%}"))
        if assessment.ok:
            st.progress(assessment.confidence)
        st.write(assessment.explanation)
        if assessment.key_differences:
            st.markdown("**Key behavioural differences:**\n" + "\n".join(f"- {d}" for d in assessment.key_differences))
        st.caption(meta + " · advisory only: patch-id/reference evidence takes precedence, and you decide.")
        if rel is not None:
            st.markdown("**Combined lineage verdict (deterministic + AI):**")
            _relationship(rel)
        elif hr.cross_side:
            _cross(hr.cross_side)
        if st.button("Re-ask", key=f"reask_{key}"):
            results.pop(ai_key, None)
            with st.spinner("Re-asking..."):
                results[ai_key] = analyzer.compare_hunk(client, hr, path, use_cache=False)
            st.rerun()
        return assessment


def _cross(rels: list[Relationship]) -> None:
    if not rels:
        st.caption("No deterministic relationship between the two sides' commits.")
    for r in rels:
        _relationship(r)


def _do_resolve(git: Git, store: Store, path: str, hunk_no: int, choice: str, side: SideReport, badge: Badge,
                ai: AIAssessment | None, whole_file: bool = False) -> None:
    name = "Side A" if choice == "current" else "Side B"
    prov = describe_choice(name, side, badge, ai)
    try:
        out = resolve_and_finish(git, store, path, hunk_no, choice, provenance=prov, whole_file=whole_file)
    except (ResolveError, GitError, OSError, UnicodeDecodeError) as exc:
        _flash("error", f"Could not resolve {path}: {exc}")
        return
    fin = out["finish"]
    if fin["action"] == "commit":
        kind = "merge commit" if len(fin.get("parents", [])) > 1 else "commit"
        _flash("success", f"✅ Created standard Git {kind} `{fin['commit'][:10]}` with parents "
                          f"{', '.join(p[:10] for p in fin.get('parents', []))}.")
        st.session_state["last_commit_message"] = fin["message"]
    elif fin["action"] in ("rebase-continue", "am-continue"):
        _flash("success", f"✅ Ran `git {fin['action'].replace('-', ' --')}`; HEAD is now `{(fin['commit'] or '')[:10]}`.")
    elif fin["action"] == "staged":
        _flash("info", f"{path} staged. {fin.get('note', '')}")
    else:
        left = out["resolution"].get("remaining_hunks", 0)
        _flash("info", f"Kept {name} in {path}. {left} hunk(s) left in this file; "
                       f"unmerged files: {', '.join(fin['remaining']) or 'none'}.")
    st.session_state.pop("ai_results", None)
    _invalidate()


def _render_hunk(git: Git, store: Store, analyzer: Analyzer, cfg: dict, path: str, hr: HunkReport, n_hunks: int, fp: str) -> None:
    h = hr.hunk
    key = hashlib.sha1(f"{fp}:{path}:{h.index}:{h.current_lines}:{h.incoming_lines}".encode()).hexdigest()[:16]
    st.markdown(f"**Hunk {h.index}/{n_hunks}** · lines {h.start_line}–{h.end_line}")
    badge_a, badge_b = side_badge(hr.current), side_badge(hr.incoming)

    col_a, col_b = st.columns(2)
    with col_a:
        _side_column("Side A", hr.current, badge_a, path)
    with col_b:
        _side_column("Side B", hr.incoming, badge_b, path)

    assessment = _ai_panel(analyzer, cfg["client"], cfg["auto"], hr, path, key)
    if hr.advisories:
        with st.expander("Context for your decision (GitTrace never picks a side)"):
            for a in hr.advisories:
                (st.error if a.startswith("WARNING") else st.markdown)(f"- {a}")

    btn_a, btn_b = st.columns(2)
    with btn_a:
        if st.button("✅ Keep Side A", key=f"a_{key}", type="primary", help="Write Side A's lines, stage, and commit once everything is resolved"):
            _do_resolve(git, store, path, h.index, "current", hr.current, badge_a, assessment)
            st.rerun()
    with btn_b:
        if st.button("✅ Keep Side B", key=f"b_{key}", type="primary", help="Write Side B's lines, stage, and commit once everything is resolved"):
            _do_resolve(git, store, path, h.index, "incoming", hr.incoming, badge_b, assessment)
            st.rerun()
    st.divider()


def main() -> None:
    st.set_page_config(page_title="GitTrace", page_icon="🧭", layout="wide")
    cfg = _sidebar(_cli_repo() or os.getcwd())
    try:
        git = Git(cfg["repo"])
        git.ensure_repo()
    except (NotARepositoryError, FileNotFoundError, OSError) as exc:
        st.error(f"Not a Git repository: {exc}\n\nStart the app from inside your cloned repo (`cd repo && streamlit run .../app.py`) or set the path in the sidebar.")
        st.stop()
    store = Store(git)
    if cfg["client"] is not None:
        cfg["client"].cache_path = store.dir / "ai-cache.json"
    _sidebar_repo_tools(git, store)

    st.title("GitTrace: intelligent conflict resolution")
    st.caption(f"Repository: `{git.toplevel}`")
    _show_flash()
    if st.session_state.get("last_commit_message"):
        with st.expander("Commit message of the last resolution", expanded=True):
            st.code(st.session_state["last_commit_message"], language=None)

    report, analyzer, fp = _get_report(git, store, cfg["history"])
    ctx = report.context

    if ctx.operation is Operation.NONE and not report.files:
        st.success("No merge conflict in progress in this repository.")
        st.markdown("Start a merge, cherry-pick, revert or rebase that conflicts (for example `git merge uat`), then click **Refresh analysis**.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Operation", ctx.operation.value)
    m2.metric("Conflicted files", len(report.files))
    m3.metric("Conflict hunks", sum(len(f.hunks) for f in report.files))
    st.markdown(f"**Side A** = stage 2 = {ctx.current.label}  \n**Side B** = stage 3 = {ctx.incoming.label}  \n**Base** = {ctx.base.label}")
    for n in ctx.notes:
        st.info(n)

    if not report.files:
        st.success("All conflicts are resolved and staged.")
        if st.button("Finish with a standard Git commit", type="primary"):
            try:
                fin = finish(git, store)
                if fin["committed"]:
                    _flash("success", f"✅ {fin['action']} → `{(fin.get('commit') or '')[:10]}`")
                    st.session_state["last_commit_message"] = fin.get("message")
                else:
                    _flash("info", fin.get("note", "nothing to do"))
            except (ResolveError, GitError) as exc:
                _flash("error", str(exc))
            _invalidate()
            st.rerun()
        return

    for fr in report.files:
        f = fr.file
        st.header(f"📄 {f.path}")
        st.caption(f"{f.kind} · current-side commits: {', '.join(c[:10] for c in fr.current_file_commits[:5]) or '-'} · "
                   f"incoming-side commits: {', '.join(c[:10] for c in fr.incoming_file_commits[:5]) or '-'}")
        if not f.hunks:
            for a in fr.advisories:
                st.warning(a)
            ca, cb = st.columns(2)
            key = hashlib.sha1(f"{fp}:{f.path}".encode()).hexdigest()[:12]
            empty = SideReport("current", ctx.current.label, ctx.current.rev, [], None, [])  # type: ignore[arg-type]
            with ca:
                st.markdown(f"#### Side A\n{ctx.current.label}\n\n{'present' if 2 in f.stages else '**deleted** on this side'}")
                if st.button("Keep Side A (whole file)", key=f"fa_{key}", type="primary"):
                    _do_resolve(git, store, f.path, 0, "current", empty, Badge("whole-file choice", "info"), None, whole_file=True)
                    st.rerun()
            with cb:
                st.markdown(f"#### Side B\n{ctx.incoming.label}\n\n{'present' if 3 in f.stages else '**deleted** on this side'}")
                if st.button("Keep Side B (whole file)", key=f"fb_{key}", type="primary"):
                    _do_resolve(git, store, f.path, 0, "incoming", empty, Badge("whole-file choice", "info"), None, whole_file=True)
                    st.rerun()
            st.divider()
            continue
        for hr in fr.hunks:
            _render_hunk(git, store, analyzer, cfg, f.path, hr, len(fr.hunks), fp)

    st.caption("Lineage uses explicit references, git patch-id and heuristics, plus your LOCAL Gemma model for the "
               "'same fix?' question. No code leaves this machine.")


if __name__ == "__main__":  # pragma: no cover
    main()
