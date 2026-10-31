"""GitTrace: AI-Powered Conflict Resolution (Streamlit app).

    cd /path/to/your/cloned/repo
    streamlit run /path/to/gittrace/app.py

Page 1 (landing): paste a GitHub repository link (or a local path) and click Analyze Repo.
Page 2 (dashboard): Gemma's local verdict, a Side A / Side B diff viewer, and Accept buttons.
Accepting plays a short resolve animation, then calls the backend resolve engine: write lines ->
`git add` -> a standard `git commit` (a normal two-parent merge commit during a merge).

AI runs only on your machine through Ollama (default http://localhost:11434, gemma4:e4b).
"""

from __future__ import annotations

import difflib
import hashlib
import html
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))  # works even without `pip install -e .`

import streamlit as st  # noqa: E402

from gittrace.ai import DEFAULT_HOST, DEFAULT_MODEL, NonLocalEndpointError, OllamaClient  # noqa: E402
from gittrace.analysis import Analyzer, Badge, HunkReport, SideReport, describe_choice, side_badge  # noqa: E402
from gittrace.git import Git, GitError  # noqa: E402
from gittrace.repolink import RepoLinkError, branches, is_clean, locate, start_merge  # noqa: E402
from gittrace.resolve import ResolveError, resolve_and_finish  # noqa: E402
from gittrace.store import Store  # noqa: E402

# =====================================================================================
# Copy: edit these to match your Beginner's Guide wording.
# =====================================================================================
HERO = {
    "eyebrow": "Hacktoberfest 2026 · Runs 100% on your machine",
    "title": "GitTrace",
    "subtitle": "AI-Powered Conflict Resolution",
    "lede": ("Merge conflicts between development, UAT and production, explained. "
             "See where each side came from, whether it was tested, and whether both sides "
             "are really the same fix, then resolve it in one click."),
}
HOW_IT_WORKS = [
    {
        "icon": "trace", "step": "01", "title": "Trace History",
        "body": ("GitTrace reads your repository's Git history to find where each conflicting line came from: "
                 "the original commit, cherry-picked backports, and identical patches. It also checks whether "
                 "that exact code passed testing in DEV, UAT or PROD."),
        "points": ["git blame & patch-id lineage", "Snapshot-exact test status", "Deployment awareness"],
    },
    {
        "icon": "ai", "step": "02", "title": "Local AI Analysis",
        "body": ("Gemma, running locally in Ollama, reads the commit messages and code of both sides and judges "
                 "whether they're logically the same fix, even when the code looks different. "
                 "Your code never leaves your laptop."),
        "points": ["Gemma via Ollama (localhost only)", "Same fix vs. different change", "Explains the key differences"],
    },
    {
        "icon": "resolve", "step": "03", "title": "One-Click Resolve",
        "body": ("Pick Side A or Side B. GitTrace writes the lines, stages the file and creates a standard Git "
                 "merge commit, and records why you chose that side in the commit message."),
        "points": ["Standard git add + commit", "Provenance in the commit message", "Merge, cherry-pick, revert, rebase"],
    },
]

ICONS = {
    "trace": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="5" r="2.2"/><circle cx="6" cy="19" r="2.2"/><circle cx="18" cy="9" r="2.2"/><path d="M6 7.2v9.6M18 11.2c0 3.8-4 4.6-9.8 6.4"/></svg>',
    "ai": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.6L18.5 9l-4.7 1.6L12 15l-1.8-4.4L5.5 9l4.7-1.4z"/><path d="M18.5 15.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg>',
    "resolve": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12.4l2.7 2.7L16.3 9.4"/></svg>',
    "lock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
    "commit": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="3.2"/><path d="M3 12h5.8M15.2 12H21"/></svg>',
}

WEB_SVG = """<svg class="gt-web" viewBox="0 0 200 200" fill="none" stroke="currentColor" stroke-width=".7">
<path d="M200 0L60 200M200 0L0 140M200 0L120 200M200 0L0 60"/>
<path d="M170 0q4 22 24 30M140 0q8 44 54 62M110 0q12 66 84 96M80 0q18 90 114 128"/></svg>"""

# =====================================================================================
# Styles
# =====================================================================================
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#09080C; --panel:#121017; --panel-2:#17141E; --line:rgba(255,255,255,.07); --line-2:rgba(255,255,255,.12);
  --text:#ECEAF3; --muted:#8C879E; --faint:#5E596F;
  --pumpkin:#FF7518; --pumpkin-2:#FF9A4D; --phantom:#8B5CF6; --phantom-2:#A78BFA;
  --ok:#4ADE80; --warn:#FBBF24; --bad:#F87171; --info:#A78BFA;
  --ease:cubic-bezier(.2,.7,.2,1);
}
html, body, .stApp{ background:var(--bg) !important; color:var(--text); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea{ font-family:'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important; }
.stApp::before{ content:""; position:fixed; inset:-20%; pointer-events:none; z-index:0;
  background:
    radial-gradient(40% 35% at 50% 0%, rgba(139,92,246,.16), transparent 70%),
    radial-gradient(30% 30% at 100% 100%, rgba(255,117,24,.08), transparent 70%),
    radial-gradient(25% 25% at 0% 60%, rgba(139,92,246,.06), transparent 70%);
  animation:gtDrift 22s ease-in-out infinite alternate; }
@keyframes gtDrift{ from{ transform:translate3d(0,0,0) } to{ transform:translate3d(-2%,1.5%,0) } }

/* chrome */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, [data-testid="stSidebarCollapsedControl"]{ display:none !important; }
.block-container, [data-testid="stMainBlockContainer"]{ max-width:1120px !important; padding:2.2rem 2rem 5rem !important; position:relative; z-index:1; }
[data-testid="stVerticalBlock"]{ gap:.9rem; }
[data-testid="stMarkdownContainer"] p{ margin:0; }

/* motion */
@keyframes gtFadeUp{ from{ opacity:0; transform:translateY(14px); filter:blur(4px) } to{ opacity:1; transform:none; filter:none } }
@keyframes gtFade{ from{ opacity:0 } to{ opacity:1 } }
.gt-fade{ animation:gtFadeUp .7s var(--ease) both; }
.gt-d1{ animation-delay:.06s } .gt-d2{ animation-delay:.14s } .gt-d3{ animation-delay:.22s } .gt-d4{ animation-delay:.3s }

/* nav */
.gt-nav{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.gt-brand{ display:flex; align-items:center; gap:10px; font-weight:700; letter-spacing:-.01em; font-size:15px; }
.gt-mark{ width:26px; height:26px; border-radius:8px; display:grid; place-items:center;
  background:linear-gradient(135deg, var(--pumpkin), #c2410c); box-shadow:0 0 18px rgba(255,117,24,.45), inset 0 1px 0 rgba(255,255,255,.25); }
.gt-mark i{ width:8px; height:8px; border-radius:50%; background:#1a0a00; box-shadow:0 0 0 2px rgba(26,10,0,.25); }
.gt-chip{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:500;
  color:var(--muted); background:rgba(255,255,255,.03); border:1px solid var(--line); }
.gt-dot{ width:7px; height:7px; border-radius:50%; background:var(--ok); box-shadow:0 0 10px var(--ok); }
.gt-dot.off{ background:var(--faint); box-shadow:none; }

/* hero */
.gt-hero{ text-align:center; padding:72px 0 26px; position:relative; }
.gt-eyebrow{ display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; font-size:12.5px; font-weight:500;
  color:#E9D5FF; background:rgba(139,92,246,.10); border:1px solid rgba(139,92,246,.28); }
.gt-eyebrow .p{ display:inline-block; animation:gtFlicker 3.2s ease-in-out infinite; }
@keyframes gtFlicker{ 0%,100%{ opacity:1; filter:drop-shadow(0 0 6px rgba(255,117,24,.8)) } 45%{ opacity:.85; filter:drop-shadow(0 0 2px rgba(255,117,24,.4)) } 50%{ opacity:1 } }
.gt-h1{ font-size:clamp(46px, 7vw, 76px); line-height:1.02; font-weight:800; letter-spacing:-.045em; margin:22px 0 6px; }
.gt-h2{ font-size:clamp(24px, 3.4vw, 36px); font-weight:700; letter-spacing:-.03em; margin:0;
  background:linear-gradient(92deg, var(--pumpkin-2) 0%, var(--pumpkin) 35%, var(--phantom-2) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.gt-lede{ max-width:640px; margin:18px auto 0; color:var(--muted); font-size:17px; line-height:1.6; }
.gt-web{ position:fixed; top:0; right:0; width:220px; height:220px; color:rgba(255,255,255,.05); pointer-events:none; z-index:0; }

/* repo input */
.st-key-repo_form [data-testid="stForm"]{ border:0; padding:0; background:transparent; }
.st-key-repo_url input{ height:58px; font-size:16px !important; padding:0 18px !important; border-radius:14px !important;
  background:var(--panel) !important; color:var(--text) !important; border:1px solid var(--line-2) !important;
  transition:border-color .25s var(--ease), box-shadow .25s var(--ease); }
.st-key-repo_url [data-baseweb="input"], .st-key-repo_url [data-baseweb="base-input"]{ background:transparent !important; border:0 !important; border-radius:14px !important; }
.st-key-repo_url input:focus{ border-color:rgba(255,117,24,.6) !important; box-shadow:0 0 0 4px rgba(255,117,24,.12), 0 0 40px -10px rgba(255,117,24,.5) !important; }
.st-key-repo_url input::placeholder{ color:var(--faint) !important; }
.st-key-repo_url label{ display:none !important; }
/* wrapper differs by Streamlit version: data-testid (new) or data-baseweb (older) */
.st-key-repo_url [data-testid="stTextInputRootElement"], .st-key-repo_url [data-baseweb="input"]{
  height:58px !important; border:0 !important; background:transparent !important; border-radius:14px !important; box-shadow:none !important; }
.st-key-repo_url [data-baseweb="base-input"]{ height:100% !important; }
.st-key-repo_url input{ height:58px !important; }

/* buttons */
.stButton button, [data-testid="stFormSubmitButton"] button{ border-radius:12px !important; font-weight:600 !important;
  transition:transform .2s var(--ease), box-shadow .25s var(--ease), background .25s var(--ease), border-color .25s var(--ease), opacity .2s; }
.stButton button[kind="secondary"]{ background:rgba(255,255,255,.03) !important; border:1px solid var(--line-2) !important; color:var(--text) !important; }
.stButton button[kind="secondary"]:hover{ border-color:rgba(255,255,255,.22) !important; transform:translateY(-1px); }
.stButton button[kind="tertiary"]{ color:var(--muted) !important; }
.stButton button[kind="tertiary"]:hover{ color:var(--text) !important; }
.st-key-analyze button[kind], .st-key-start_merge button[kind], .st-key-next_conflict button[kind]{
  height:58px; width:100%; color:#1b0b00 !important; border:0 !important; font-size:15.5px !important; letter-spacing:-.005em;
  background:linear-gradient(180deg, #FF8A3D 0%, var(--pumpkin) 55%, #E85F0C 100%) !important;
  box-shadow:0 10px 30px -10px rgba(255,117,24,.7), inset 0 1px 0 rgba(255,255,255,.35) !important;
  animation:gtPulse 2.8s ease-in-out infinite; }
.st-key-start_merge button[kind], .st-key-next_conflict button[kind]{ height:48px; }
.st-key-next_conflict{ margin-top:18px; }
.st-key-analyze button[kind]:hover, .st-key-start_merge button[kind]:hover, .st-key-next_conflict button[kind]:hover{ transform:translateY(-2px); filter:brightness(1.05); }
.st-key-analyze, .st-key-analyze .stButton, .st-key-analyze [data-testid="stFormSubmitButton"]{ width:100%; }
@keyframes gtPulse{
  0%,100%{ box-shadow:0 10px 30px -12px rgba(255,117,24,.65), 0 0 0 0 rgba(255,117,24,0), inset 0 1px 0 rgba(255,255,255,.35) }
  50%{ box-shadow:0 14px 38px -10px rgba(255,117,24,.85), 0 0 0 6px rgba(255,117,24,.07), inset 0 1px 0 rgba(255,255,255,.35) } }

.st-key-accept_a button[kind], .st-key-accept_b button[kind]{ width:100%; height:54px; font-size:15.5px !important; border:0 !important; letter-spacing:-.005em; }
.st-key-accept_a, .st-key-accept_b, .st-key-accept_a .stButton, .st-key-accept_b .stButton{ width:100%; }
.st-key-accept_a button[kind]{ color:#1b0b00 !important; background:linear-gradient(180deg,#FF8A3D,var(--pumpkin) 60%,#E85F0C) !important;
  box-shadow:0 12px 30px -14px rgba(255,117,24,.8), inset 0 1px 0 rgba(255,255,255,.35) !important; }
.st-key-accept_b button[kind]{ color:#fff !important; background:linear-gradient(180deg,#A07BFF,var(--phantom) 60%,#6D3FE0) !important;
  box-shadow:0 12px 30px -14px rgba(139,92,246,.85), inset 0 1px 0 rgba(255,255,255,.3) !important; }
.st-key-accept_a button[kind]:hover{ transform:translateY(-2px); box-shadow:0 18px 40px -12px rgba(255,117,24,.95), 0 0 0 5px rgba(255,117,24,.09), inset 0 1px 0 rgba(255,255,255,.35) !important; }
.st-key-accept_b button[kind]:hover{ transform:translateY(-2px); box-shadow:0 18px 40px -12px rgba(139,92,246,.95), 0 0 0 5px rgba(139,92,246,.10), inset 0 1px 0 rgba(255,255,255,.3) !important; }
.st-key-accept_a button[kind]:active, .st-key-accept_b button[kind]:active{ transform:translateY(0) scale(.99); }

/* cards */
.gt-card{ height:100%; min-height:372px; box-sizing:border-box; background:linear-gradient(180deg, var(--panel-2), var(--panel)); border:1px solid var(--line); border-radius:18px; padding:22px;
  transition:transform .35s var(--ease), border-color .35s var(--ease), box-shadow .35s var(--ease); }
.gt-card:hover{ transform:translateY(-4px); border-color:var(--line-2); box-shadow:0 24px 50px -28px rgba(0,0,0,.9), 0 0 0 1px rgba(255,255,255,.02); }
.gt-step-top{ display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
.gt-ico{ width:42px; height:42px; border-radius:12px; display:grid; place-items:center; color:var(--pumpkin-2);
  background:rgba(255,117,24,.09); border:1px solid rgba(255,117,24,.22); }
.gt-ico svg{ width:21px; height:21px; }
.gt-ico.ai{ color:var(--phantom-2); background:rgba(139,92,246,.1); border-color:rgba(139,92,246,.25); }
.gt-num{ font-family:'JetBrains Mono', ui-monospace, monospace; font-size:12px; color:var(--faint); }
.gt-card h3{ font-size:18px; font-weight:700; letter-spacing:-.02em; margin:0 0 8px; color:var(--text); padding:0; }
.gt-card p.b{ color:var(--muted); font-size:14.5px; line-height:1.62; margin:0 0 16px; }
.gt-points{ list-style:none; padding:0; margin:0; display:grid; gap:7px; }
.gt-points li{ font-size:13px; color:#C9C5D6; display:flex; gap:9px; align-items:center; margin:0; }
.gt-points li::before{ content:""; width:5px; height:5px; border-radius:50%; background:var(--pumpkin); box-shadow:0 0 8px var(--pumpkin); flex:none; }
.gt-section{ text-align:center; margin:84px 0 26px; }
.gt-kicker{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--pumpkin-2); font-weight:600; }
.gt-section h2{ font-size:32px; font-weight:750; letter-spacing:-.035em; margin:10px 0 0; padding:0; }
.gt-trust{ display:flex; gap:22px; justify-content:center; flex-wrap:wrap; margin-top:18px; color:var(--faint); font-size:13px; }
.gt-trust span{ display:inline-flex; gap:7px; align-items:center; }
.gt-trust svg{ width:15px; height:15px; color:var(--muted); }
.gt-hint{ text-align:center; color:var(--faint); font-size:13px; margin-top:4px; }

/* dashboard */
.gt-top{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
.gt-repo{ display:flex; align-items:center; gap:12px; }
.gt-repo b{ font-size:20px; letter-spacing:-.025em; }
.gt-sub{ color:var(--muted); font-size:13px; }
.gt-pills{ display:flex; gap:8px; flex-wrap:wrap; }
.gt-pill{ display:inline-flex; align-items:center; gap:7px; padding:5px 11px; border-radius:999px; font-size:12px; font-weight:600;
  border:1px solid var(--line-2); color:#D4D0E2; background:rgba(255,255,255,.03); }
.gt-pill.ok{ color:var(--ok); border-color:rgba(74,222,128,.32); background:rgba(74,222,128,.07); }
.gt-pill.warn{ color:#FDBA74; border-color:rgba(255,117,24,.35); background:rgba(255,117,24,.08); }
.gt-pill.bad{ color:var(--bad); border-color:rgba(248,113,113,.35); background:rgba(248,113,113,.08); }
.gt-pill.info{ color:var(--phantom-2); border-color:rgba(139,92,246,.35); background:rgba(139,92,246,.08); }
.gt-pill i{ width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 8px currentColor; }

/* AI verdict card: animated conic border */
@property --gt-a{ syntax:'<angle>'; inherits:false; initial-value:0deg; }
.gt-ai{ position:relative; border-radius:20px; padding:1.2px; isolation:isolate;
  background:conic-gradient(from var(--gt-a), rgba(255,117,24,.9), rgba(139,92,246,.9), rgba(56,189,248,.5), rgba(139,92,246,.9), rgba(255,117,24,.9));
  animation:gtSpin 7s linear infinite, gtFadeUp .7s var(--ease) both; }
.gt-ai::before{ content:""; position:absolute; inset:-1px; border-radius:22px; z-index:-1; filter:blur(22px); opacity:.33; background:inherit; }
@keyframes gtSpin{ to{ --gt-a:360deg; } }
.gt-ai-in{ background:linear-gradient(180deg,#15121D,#100E15); border-radius:19px; padding:24px 26px; }
.gt-ai-head{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.gt-ai-title{ display:flex; align-items:center; gap:10px; font-weight:650; font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:#DCD3FF; }
.gt-ai-title svg{ width:18px; height:18px; color:var(--phantom-2); filter:drop-shadow(0 0 6px rgba(167,139,250,.8)); }
.gt-verdict{ font-size:28px; font-weight:750; letter-spacing:-.03em; margin:2px 0 12px; }
.gt-verdict.same{ color:var(--ok); } .gt-verdict.diff{ color:var(--bad); } .gt-verdict.part{ color:var(--warn); } .gt-verdict.unk{ color:var(--muted); }
.gt-bar{ height:6px; border-radius:6px; background:rgba(255,255,255,.06); overflow:hidden; margin-bottom:16px; }
.gt-bar span{ display:block; height:100%; border-radius:6px; transform-origin:left; animation:gtGrow 1.1s var(--ease) .2s both; }
@keyframes gtGrow{ from{ transform:scaleX(0) } to{ transform:scaleX(1) } }
.gt-expl{ color:#D8D3E6; font-size:15px; line-height:1.65; }
.gt-diffs{ display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.gt-diffs span{ font-size:12.5px; padding:5px 10px; border-radius:8px; background:rgba(248,113,113,.08); color:#FCA5A5; border:1px solid rgba(248,113,113,.2); }
.gt-foot{ display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-top:18px; padding-top:14px; border-top:1px solid var(--line); color:var(--faint); font-size:12.5px; }
.gt-foot b{ color:#BFB9D3; font-weight:600; }
.gt-skel{ height:12px; border-radius:6px; margin:10px 0; background:linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.10), rgba(255,255,255,.04));
  background-size:200% 100%; animation:gtShimmer 1.4s linear infinite; }
@keyframes gtShimmer{ to{ background-position:-200% 0 } }

/* side cards + diff viewer */
.gt-side{ border-radius:18px; border:1px solid var(--line); background:linear-gradient(180deg,var(--panel-2),var(--panel)); overflow:hidden;
  transition:transform .35s var(--ease), border-color .35s var(--ease), box-shadow .35s var(--ease), opacity .6s var(--ease), filter .6s var(--ease); }
.gt-side:hover{ transform:translateY(-3px); border-color:var(--line-2); box-shadow:0 24px 50px -30px rgba(0,0,0,.95); }
.gt-side-h{ padding:18px 20px 14px; border-bottom:1px solid var(--line); }
.gt-side-t{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
.gt-side-t b{ font-size:15px; letter-spacing:-.01em; display:flex; align-items:center; gap:9px; }
.gt-tag{ width:22px; height:22px; border-radius:7px; display:grid; place-items:center; font-size:12px; font-weight:800; }
.gt-a .gt-tag{ background:rgba(255,117,24,.14); color:var(--pumpkin-2); border:1px solid rgba(255,117,24,.3); }
.gt-b .gt-tag{ background:rgba(139,92,246,.14); color:var(--phantom-2); border:1px solid rgba(139,92,246,.32); }
.gt-lbl{ color:var(--muted); font-size:12.5px; }
.gt-origin{ margin-top:10px; color:#BDB7CF; font-size:12.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.gt-origin code{ font-family:'JetBrains Mono', monospace !important; color:var(--faint); background:none; padding:0; font-size:12px; }
.gt-code{ font-family:'JetBrains Mono', ui-monospace, Menlo, monospace; font-size:13px; line-height:1.75; padding:12px 0; overflow-x:auto; background:rgba(0,0,0,.18); }
.gt-ln{ display:flex; padding:0 18px 0 0; white-space:pre; }
.gt-ln .n{ width:46px; flex:none; text-align:right; padding-right:14px; color:var(--faint); user-select:none; }
.gt-ln .c{ color:#DAD6E6; }
.gt-a .gt-ln.chg{ background:rgba(255,117,24,.07); box-shadow:inset 2px 0 0 var(--pumpkin); }
.gt-b .gt-ln.chg{ background:rgba(139,92,246,.08); box-shadow:inset 2px 0 0 var(--phantom); }
.gt-a mark{ background:rgba(255,117,24,.28); color:#FFE3CF; border-radius:3px; padding:0 1px; }
.gt-b mark{ background:rgba(139,92,246,.32); color:#EDE7FF; border-radius:3px; padding:0 1px; }
.gt-empty{ color:var(--faint); font-style:italic; padding:6px 18px 6px 60px; }
.gt-det{ padding:12px 20px 16px; display:grid; gap:5px; }
.gt-det span{ font-size:12px; color:var(--muted); }

/* resolution animation */
.gt-side.gt-rejected{ animation:gtReject 1.1s var(--ease) forwards; }
@keyframes gtReject{ 0%{ opacity:1; transform:none; filter:none } 100%{ opacity:.12; transform:translateY(10px) scale(.97); filter:blur(3px) grayscale(1) } }
.gt-side.gt-chosen{ animation:gtChosen 1.2s var(--ease) forwards; }
@keyframes gtChosen{ 0%{ box-shadow:0 0 0 0 rgba(74,222,128,0) } 40%{ box-shadow:0 0 0 2px rgba(74,222,128,.7), 0 0 60px -10px rgba(74,222,128,.55) }
  100%{ box-shadow:0 0 0 1px rgba(74,222,128,.55), 0 0 40px -16px rgba(74,222,128,.5); border-color:rgba(74,222,128,.4) } }
.gt-done{ text-align:center; padding:34px 20px 26px; border-radius:20px; border:1px solid rgba(74,222,128,.25);
  background:radial-gradient(60% 120% at 50% 0%, rgba(74,222,128,.10), transparent 70%), var(--panel); animation:gtFadeUp .6s var(--ease) both; }
.gt-check{ width:64px; height:64px; margin:0 auto 14px; }
.gt-check circle{ stroke:var(--ok); stroke-width:2; fill:none; stroke-dasharray:166; stroke-dashoffset:166; animation:gtDraw .7s var(--ease) .1s forwards; }
.gt-check path{ stroke:var(--ok); stroke-width:3; fill:none; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:48; stroke-dashoffset:48; animation:gtDraw .45s var(--ease) .65s forwards; }
.gt-check svg{ filter:drop-shadow(0 0 12px rgba(74,222,128,.6)); }
@keyframes gtDraw{ to{ stroke-dashoffset:0 } }
.gt-done h3{ font-size:24px; font-weight:750; letter-spacing:-.03em; margin:0 0 6px; padding:0; color:var(--text); }
.gt-done p{ color:var(--muted); font-size:14px; }
.gt-commit{ margin-top:16px; text-align:left; font-family:'JetBrains Mono', monospace; font-size:12.5px; line-height:1.7; color:#C8C2DA;
  background:rgba(0,0,0,.25); border:1px solid var(--line); border-radius:12px; padding:14px 16px; white-space:pre-wrap; max-height:280px; overflow:auto; }
.stApp code{ font-family:'JetBrains Mono', ui-monospace, monospace !important; background:rgba(255,255,255,.06) !important;
  color:#E9D5FF !important; padding:1px 6px; border-radius:6px; font-size:.86em; border:1px solid var(--line); }
.gt-commit{ white-space:normal; }
.gt-commit b{ color:var(--ok); font-weight:500; }
.gt-muted-card{ border:1px dashed var(--line-2); border-radius:18px; padding:26px; text-align:center; color:var(--muted); }
.gt-muted-card h3{ color:var(--text); font-size:20px; margin:0 0 6px; padding:0; letter-spacing:-.02em; }

/* streamlit widgets in dark */
[data-baseweb="select"] > div{ background:var(--panel) !important; border-color:var(--line-2) !important; border-radius:12px !important; }
[data-testid="stExpander"] details{ border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel) !important; }
[data-testid="stExpander"] summary{ color:var(--muted) !important; }
[data-testid="stPopover"] button, [data-testid="stPopoverButton"]{ background:rgba(255,255,255,.03) !important; color:var(--text) !important;
  border:1px solid var(--line-2) !important; border-radius:12px !important; }
[data-testid="stPopover"] button:hover{ border-color:rgba(255,255,255,.22) !important; }
[data-testid="stPopover"] button p, [data-testid="stPopover"] button span{ color:var(--text) !important; }
[data-testid="stPopoverBody"]{ background:var(--panel-2) !important; border:1px solid var(--line-2) !important; border-radius:14px !important; }
.stAlert{ border-radius:14px !important; }
</style>
"""


# =====================================================================================
# HTML helpers
# =====================================================================================
def esc(s: object) -> str:
    """HTML-escape, and flatten newlines: a blank line would end Streamlit's HTML block."""
    return html.escape(str(s), quote=True).replace("\r", "").replace("\n", " ")


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def nav(right: str = "") -> None:
    md(f'<div class="gt-nav gt-fade"><div class="gt-brand"><span class="gt-mark"><i></i></span>GitTrace</div>{right}</div>')


def ai_chip(client: OllamaClient | None) -> str:
    if client is None:
        return '<span class="gt-chip"><span class="gt-dot off"></span>Local AI off</span>'
    return f'<span class="gt-chip"><span class="gt-dot"></span>{esc(client.model)} · on-device</span>'


LEVEL_CLASS = {"success": "ok", "warning": "warn", "error": "bad", "info": "info"}


def _inline_marks(a: str, b: str) -> tuple[str, str]:
    """Highlight the characters that differ between two near-identical lines (else: plain)."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.ratio() < 0.6:  # mostly different: a line tint reads better than confetti highlights
        return esc(a), esc(b)
    oa, ob = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        sa, sb = esc(a[i1:i2]), esc(b[j1:j2])
        if op == "equal":
            oa.append(sa); ob.append(sb)
        else:
            if sa:
                oa.append(f"<mark>{sa}</mark>")
            if sb:
                ob.append(f"<mark>{sb}</mark>")
    return "".join(oa), "".join(ob)


def diff_pair(a_lines: list[str], b_lines: list[str]) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """Return per-line (html, changed) for both sides, with intra-line highlights."""
    out_a: list[tuple[str, bool]] = []
    out_b: list[tuple[str, bool]] = []
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out_a += [(esc(x), False) for x in a_lines[i1:i2]]
            out_b += [(esc(x), False) for x in b_lines[j1:j2]]
            continue
        pairs = min(i2 - i1, j2 - j1)
        for k in range(pairs):
            ha, hb = _inline_marks(a_lines[i1 + k], b_lines[j1 + k])
            out_a.append((ha, True)); out_b.append((hb, True))
        out_a += [(esc(x), True) for x in a_lines[i1 + pairs:i2]]
        out_b += [(esc(x), True) for x in b_lines[j1 + pairs:j2]]
    return out_a, out_b


def code_html(rows: list[tuple[str, bool]], start: int) -> str:
    if not rows:
        return '<div class="gt-code"><div class="gt-empty">This side removes these lines.</div></div>'
    body = "".join(f'<div class="gt-ln{" chg" if chg else ""}"><span class="n">{start + i}</span><span class="c">{h or " "}</span></div>'
                   for i, (h, chg) in enumerate(rows))
    return f'<div class="gt-code">{body}</div>'


def side_card(key: str, side: SideReport, badge: Badge, rows: list[tuple[str, bool]], start: int, state: str = "") -> str:
    origin = ""
    if side.commits:
        cc = side.commits[0]
        origin = f'<div class="gt-origin"><code>{esc(cc.commit[:7])}</code>&nbsp; {esc(cc.subject)} · {esc(cc.author)}</div>'
    details = "".join(f"<span>{esc(d)}</span>" for d in (badge.details[1:4] if len(badge.details) > 1 else badge.details))
    det_html = f'<div class="gt-det">{details}</div>' if details else ""
    return (f'<div class="gt-side gt-{key.lower()} gt-fade {state}">'
            f'<div class="gt-side-h"><div class="gt-side-t"><b><span class="gt-tag">{key}</span>Side {key}</b>'
            f'<span class="gt-lbl">{esc(side.label)}</span></div>'
            f'<span class="gt-pill {LEVEL_CLASS.get(badge.level, "info")}"><i></i>{esc(badge.text)}</span>{origin}</div>'
            f'{code_html(rows, start)}{det_html}</div>')


VERDICT_UI = {"SAME_FIX": ("Logically the same fix", "same", "var(--ok)"),
              "PARTIALLY_SAME": ("Partially the same", "part", "var(--warn)"),
              "DIFFERENT_CHANGES": ("Different changes", "diff", "var(--bad)"),
              "UNCLEAR": ("Not enough evidence", "unk", "var(--muted)")}


def ai_card(assessment, rel, client: OllamaClient | None) -> str:
    title = f'<div class="gt-ai-title">{ICONS["ai"]}Gemma analysis</div>'
    if client is None:
        body = ('<div class="gt-verdict unk">Local AI is off</div><div class="gt-expl">Turn on Gemma in '
                '<b>Local AI settings</b> to compare both sides on-device. Deterministic provenance is shown below.</div>')
        return f'<div class="gt-ai"><div class="gt-ai-in"><div class="gt-ai-head">{title}</div>{body}</div></div>'
    if assessment is None or not assessment.ok:
        err = esc(assessment.error if assessment is not None else "unavailable")
        body = (f'<div class="gt-verdict unk">Gemma is offline</div><div class="gt-expl">{err}</div>'
                '<div class="gt-foot"><span>Start it with <b>ollama serve</b> and <b>ollama pull ' + esc(client.model) +
                '</b>, then refresh.</span></div>')
        return f'<div class="gt-ai"><div class="gt-ai-in"><div class="gt-ai-head">{title}{ai_chip(client)}</div>{body}</div></div>'
    label, cls, color = VERDICT_UI.get(assessment.verdict, VERDICT_UI["UNCLEAR"])
    diffs = "".join(f"<span>{esc(d)}</span>" for d in assessment.key_differences[:6])
    lineage = ""
    if rel is not None:
        mod = " (modified)" if rel.modified else ""
        lineage = f"<span>Lineage: <b>{esc(rel.relation.value)}{mod}</b> · confidence {esc(rel.confidence)}</span>"
    meta = f"{esc(assessment.model)} · on-device" + (" · cached" if assessment.cached else f" · {assessment.duration_s:.1f}s")
    return (f'<div class="gt-ai"><div class="gt-ai-in">'
            f'<div class="gt-ai-head">{title}<span class="gt-chip"><span class="gt-dot"></span>{meta}</span></div>'
            f'<div class="gt-verdict {cls}">{label} <span style="font-size:15px;color:var(--muted);font-weight:500">· {assessment.confidence:.0%} confidence</span></div>'
            f'<div class="gt-bar"><span style="width:{assessment.confidence * 100:.0f}%;background:{color};box-shadow:0 0 14px {color}"></span></div>'
            f'<div class="gt-expl">{esc(assessment.explanation)}</div>'
            f'{f"<div class=gt-diffs>{diffs}</div>" if diffs else ""}'
            f'<div class="gt-foot">{lineage or "<span></span>"}<span>Advisory · patch-id & reference evidence take precedence · you decide</span></div>'
            f'</div></div>')


AI_SKELETON = ('<div class="gt-ai"><div class="gt-ai-in"><div class="gt-ai-head"><div class="gt-ai-title">' + ICONS["ai"] +
               'Gemma analysis</div><span class="gt-chip"><span class="gt-dot"></span>reading both sides…</span></div>'
               '<div class="gt-skel" style="width:46%;height:22px"></div><div class="gt-skel" style="width:100%"></div>'
               '<div class="gt-skel" style="width:92%"></div><div class="gt-skel" style="width:74%"></div></div></div>')


def done_card(title: str, sub: str, commit_msg: str | None = None, commit: str | None = None) -> str:
    check = ('<div class="gt-check"><svg viewBox="0 0 52 52"><circle cx="26" cy="26" r="25"/>'
             '<path d="M15 27l7 7 15-15"/></svg></div>')
    extra = ""
    if commit_msg:
        # no raw newlines inside the HTML block, or Streamlit's markdown parser takes over
        lines = "<br>".join(esc(ln).replace("  ", "&nbsp;&nbsp;") or "&nbsp;" for ln in commit_msg.splitlines())
        extra = f'<div class="gt-commit"><b>commit {esc(commit or "")}</b><br>{lines}</div>'
    return f'<div class="gt-done">{check}<h3>{esc(title)}</h3><p>{sub}</p>{extra}</div>'


# =====================================================================================
# State helpers
# =====================================================================================
def ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state[key]


def get_client() -> OllamaClient | None:
    s = st.session_state["settings"]
    if not s["ai"]:
        return None
    try:
        return OllamaClient(s["host"], s["model"])
    except NonLocalEndpointError:
        return None


def fingerprint(git: Git, store: Store) -> str:
    parts = [str(git.toplevel), git.rev_parse("HEAD") or ""]
    for name in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD"):
        parts += git.pseudo_ref(name)
    for p in (git.git_dir / "index", store.path):
        parts.append(str(p.stat().st_mtime_ns) if p.exists() else "-")
    return "|".join(parts)


def analysis(git: Git, store: Store, force: bool = False):
    fp = fingerprint(git, store)
    ss = st.session_state
    if force or ss.get("fp") != fp or "report" not in ss:
        ss["analyzer"] = Analyzer(git, store)
        ss["report"] = ss["analyzer"].analyze()
        ss["fp"] = fp
    return ss["report"], ss["analyzer"]


def hunk_key(path: str, hr: HunkReport, head: str | None) -> str:
    return hashlib.sha1(f"{path}|{hr.hunk.current_lines}|{hr.hunk.incoming_lines}|{head}".encode()).hexdigest()[:16]


# =====================================================================================
# Page 1: landing
# =====================================================================================
def page_landing() -> None:
    md(WEB_SVG)
    nav(ai_chip(get_client()))
    md(f'''<div class="gt-hero">
      <div class="gt-eyebrow gt-fade"><span class="p">🎃</span>{esc(HERO["eyebrow"])}</div>
      <div class="gt-h1 gt-fade gt-d1">{esc(HERO["title"])}</div>
      <div class="gt-h2 gt-fade gt-d2">{esc(HERO["subtitle"])}</div>
      <div class="gt-lede gt-fade gt-d3">{esc(HERO["lede"])}</div></div>''')

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        with st.container(key="repo_form"):
            with st.form("repo", border=False, clear_on_submit=False):
                c1, c2 = st.columns([4.2, 1.35], vertical_alignment="bottom", gap="small")
                url = c1.text_input("GitHub Repository Link", key="repo_url",
                                    placeholder="https://github.com/your-org/your-repo",
                                    value=st.session_state.get("last_url", ""))
                go = c2.form_submit_button("Analyze Repo  →", key="analyze", type="primary")
        cwd_repo = st.session_state.get("start_repo")
        if cwd_repo:
            md(f'<div class="gt-hint">Running inside <b style="color:#C9C5D6">{esc(Path(cwd_repo).name)}</b>. '
               f'Paste its GitHub link, or use it directly:</div>')
            _, b, _ = st.columns([2, 1.4, 2])
            if b.button("Use this folder", key="use_cwd", type="tertiary"):
                url, go = cwd_repo, True
        if go:
            submit_repo(url)
        md(f'''<div class="gt-trust gt-fade gt-d4">
          <span>{ICONS["lock"]}No cloud AI. Code stays on your machine</span>
          <span>{ICONS["commit"]}Produces standard Git commits</span>
          <span>{ICONS["trace"]}Works with any local clone</span></div>''')

    md('<div class="gt-section gt-fade"><div class="gt-kicker">How it works</div><h2>From conflict to commit in three steps</h2></div>')
    cols = st.columns(3, gap="medium")
    for col, step, delay in zip(cols, HOW_IT_WORKS, ("gt-d1", "gt-d2", "gt-d3")):
        points = "".join(f"<li>{esc(p)}</li>" for p in step["points"])
        col.markdown(f'''<div class="gt-card gt-fade {delay}">
            <div class="gt-step-top"><div class="gt-ico {step["icon"]}">{ICONS[step["icon"]]}</div><span class="gt-num">{step["step"]}</span></div>
            <h3>{esc(step["title"])}</h3><p class="b">{esc(step["body"])}</p><ul class="gt-points">{points}</ul></div>''',
                     unsafe_allow_html=True)
    md('<div class="gt-hint" style="margin-top:48px">Built for Hacktoberfest 2026 Hack Day · Problem 01</div>')


def submit_repo(url: str) -> None:
    starts = [Path(p) for p in (os.environ.get("GITTRACE_REPO"), os.getcwd()) if p]
    try:
        with st.spinner("Locating your local clone…"):
            loc = locate(url, starts)
    except RepoLinkError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # clone timeouts etc.
        st.error(f"Could not open the repository: {exc}")
        return
    ss = st.session_state
    ss.update(page="dashboard", repo_path=str(loc.path), repo_how=loc.how, last_url=url,
              repo_slug=loc.ref.slug if loc.ref else Path(loc.path).name)
    for k in ("fp", "report", "analyzer", "pending", "result"):
        ss.pop(k, None)
    st.rerun()


# =====================================================================================
# Page 2: dashboard
# =====================================================================================
def settings_popover(container) -> None:
    s = st.session_state["settings"]
    with container.popover("Local AI settings", icon=":material/tune:"):
        s["ai"] = st.toggle("Use Gemma (local Ollama)", value=s["ai"], key="set_ai")
        s["host"] = st.text_input("Ollama URL (localhost only)", value=s["host"], key="set_host")
        s["model"] = st.text_input("Model", value=s["model"], key="set_model")
        try:
            OllamaClient(s["host"], s["model"])
        except NonLocalEndpointError as exc:
            st.error(str(exc))
        if st.button("Check connection", key="check_ai"):
            h = OllamaClient(s["host"], s["model"]).health() if s["ai"] else {"reachable": False, "error": "AI is off"}
            if not h.get("reachable"):
                st.error(f"Not reachable: {h.get('error')}. Run `ollama serve`.")
            elif not h.get("model_available"):
                st.warning(f"Run `ollama pull {s['model']}`")
            else:
                st.success("Connected, model ready")


def page_dashboard() -> None:
    ss = st.session_state
    try:
        git = Git(ss["repo_path"])
        git.ensure_repo()
    except Exception as exc:
        st.error(f"Repository is no longer available: {exc}")
        if st.button("← Back"):
            ss["page"] = "landing"; st.rerun()
        return
    store = Store(git)
    client = get_client()
    if client is not None:
        client.cache_path = store.dir / "ai-cache.json"

    # top bar
    c1, c2, c3 = st.columns([1.1, 5, 2.2], vertical_alignment="center")
    if c1.button("← New repo", key="back", type="tertiary"):
        ss["page"] = "landing"
        for k in ("pending", "result"):
            ss.pop(k, None)
        st.rerun()
    how = {"current": "your current clone", "workspace": "GitTrace workspace clone", "cloned": "freshly cloned",
           "path": "local path"}.get(ss.get("repo_how", ""), "local")
    c2.markdown(f'<div class="gt-repo gt-fade"><span class="gt-mark"><i></i></span><div><b>{esc(ss.get("repo_slug", ""))}</b>'
                f'<div class="gt-sub">{esc(git.toplevel)} · {how}</div></div></div>', unsafe_allow_html=True)
    with c3:
        r1, r2 = st.columns([1, 1.4], vertical_alignment="center")
        if r1.button("Refresh", key="refresh", icon=":material/refresh:"):
            ss.pop("fp", None)
        settings_popover(r2)

    # a resolution that just happened
    if ss.get("result"):
        render_result(ss["result"])
        return

    report, analyzer = analysis(git, store)
    ctx = report.context
    files = [f for f in report.files if f.hunks]

    if not report.files:
        render_no_conflict(git, ctx)
        return

    pills = [f'<span class="gt-pill info"><i></i>{esc(ctx.operation.value)} in progress</span>']
    total = sum(len(f.hunks) for f in report.files)
    pills.append(f'<span class="gt-pill">{total} conflict{"s" if total != 1 else ""} remaining</span>')
    pills.append(f'<span class="gt-pill">{len(report.files)} file{"s" if len(report.files) != 1 else ""}</span>')
    md(f'<div class="gt-pills gt-fade" style="margin-top:6px">{"".join(pills)}</div>')

    if not files:  # structural conflicts only
        render_structural(git, store, report)
        return

    fr = files[0]
    hr = fr.hunks[0]
    key = hunk_key(fr.file.path, hr, ctx.head)
    badge_a, badge_b = side_badge(hr.current), side_badge(hr.incoming)
    pending = ss.get("pending")
    if pending and pending.get("key") != key:
        ss.pop("pending", None); pending = None

    # --- AI verdict card
    ai_slot = st.empty()
    cache = ss_default("ai_cache", {})
    cache_key = f"{key}:{client.base}:{client.model}" if client else None
    if client is not None and cache_key not in cache:
        ai_slot.markdown(AI_SKELETON, unsafe_allow_html=True)
        assessment, rel = analyzer.compare_hunk(client, hr, fr.file.path)
        if assessment.ok:
            cache[cache_key] = (assessment, rel)
    else:
        assessment, rel = cache.get(cache_key, (None, None)) if cache_key else (None, None)
    ai_slot.markdown(ai_card(assessment, rel, client), unsafe_allow_html=True)

    # --- file + diff viewer
    md(f'<div class="gt-top gt-fade" style="margin-top:10px"><div><b style="font-size:16px">{esc(fr.file.path)}</b>'
       f'<div class="gt-sub">Lines {hr.hunk.start_line}–{hr.hunk.end_line} · conflict {1} of {total}</div></div>'
       f'<div class="gt-sub">Side A = {esc(ctx.current.label)} &nbsp;·&nbsp; Side B = {esc(ctx.incoming.label)}</div></div>')
    rows_a, rows_b = diff_pair(hr.hunk.current_lines, hr.hunk.incoming_lines)
    start = hr.hunk.start_line + 1
    state_a = state_b = ""
    if pending:
        state_a, state_b = ("gt-chosen", "gt-rejected") if pending["choice"] == "current" else ("gt-rejected", "gt-chosen")
    ca, cb = st.columns(2, gap="medium")
    ca.markdown(side_card("A", hr.current, badge_a, rows_a, start, state_a), unsafe_allow_html=True)
    cb.markdown(side_card("B", hr.incoming, badge_b, rows_b, start, state_b), unsafe_allow_html=True)

    if pending:
        run_resolution(git, store, fr.file.path, hr, key, pending, {"current": badge_a, "incoming": badge_b}, assessment)
        return

    # --- actions
    ba, bb = st.columns(2, gap="medium")
    if ba.button("Accept Side A", key="accept_a", type="primary", icon=":material/check:"):
        ss["pending"] = {"key": key, "choice": "current"}; st.rerun()
    if bb.button("Accept Side B", key="accept_b", type="primary", icon=":material/check:"):
        ss["pending"] = {"key": key, "choice": "incoming"}; st.rerun()

    if hr.advisories:
        with st.expander("Why these badges? Full provenance context"):
            for a in hr.advisories:
                st.markdown(f"- {a}")


def run_resolution(git: Git, store: Store, path: str, hr: HunkReport, key: str, pending: dict,
                   badges: dict, assessment) -> None:
    """Play the success animation, THEN call the backend resolve engine."""
    ss = st.session_state
    name = "Side A" if pending["choice"] == "current" else "Side B"
    slot = st.empty()
    slot.markdown(done_card("Conflict resolved", f"Keeping <b>{name}</b>. Staging the file and writing a standard Git commit…"),
                  unsafe_allow_html=True)
    time.sleep(1.6)  # let the fade-out / check animation play before touching the repository
    side = hr.current if pending["choice"] == "current" else hr.incoming
    provenance = describe_choice(name, side, badges[pending["choice"]], assessment)
    try:
        out = resolve_and_finish(git, store, path, hr.hunk.index, pending["choice"], provenance=provenance)
    except (ResolveError, GitError, OSError, UnicodeDecodeError) as exc:
        ss.pop("pending", None)
        slot.error(f"Could not resolve: {exc}")
        st.button("Back to the conflict", key="err_back")
        return
    ss.pop("pending", None)
    ss.pop("fp", None)
    ss["result"] = {"name": name, "path": path, **out}
    st.rerun()


def render_result(res: dict) -> None:
    ss = st.session_state
    fin = res["finish"]
    if fin["action"] == "commit":
        parents = ", ".join(p[:7] for p in fin.get("parents", []))
        kind = "merge commit" if len(fin.get("parents", [])) > 1 else "commit"
        md(done_card("Merge complete",
                     f"Kept <b>{esc(res['name'])}</b>. All conflicts resolved. Created a standard Git {kind} "
                     f"<code>{esc(fin['commit'][:10])}</code> (parents {esc(parents)}).",
                     fin.get("message"), fin["commit"][:12]))
        _, b, _ = st.columns([2, 1.3, 2])
        if b.button("Analyze another repo", key="next_conflict", type="primary"):
            ss.pop("result", None); ss["page"] = "landing"; st.rerun()
    elif fin["action"] in ("rebase-continue", "am-continue"):
        md(done_card("Conflict resolved", f"Kept <b>{esc(res['name'])}</b> and ran <code>git {fin['action'].replace('-', ' --')}</code>."))
        _, b, _ = st.columns([2, 1.3, 2])
        if b.button("Continue", key="next_conflict", type="primary"):
            ss.pop("result", None); st.rerun()
    else:
        left = len(fin.get("remaining", []))
        md(done_card("Conflict resolved",
                     f"Kept <b>{esc(res['name'])}</b> in <code>{esc(res['path'])}</code>. "
                     f"{left} file{'s' if left != 1 else ''} still unmerged. The merge commit is written once everything is resolved."))
        _, b, _ = st.columns([2, 1.3, 2])
        if b.button("Next conflict  →", key="next_conflict", type="primary"):
            ss.pop("result", None); st.rerun()


def render_no_conflict(git: Git, ctx) -> None:
    md('<div class="gt-muted-card gt-fade"><h3>No conflict in progress</h3>'
       'This repository is clean. Start a standard <code>git merge</code> between two branches, and GitTrace will take you through any conflicts.</div>')
    local, remote = branches(git)
    targets = local + [r.split("/", 1)[1] for r in remote if r.split("/", 1)[1] not in local]
    incoming = local + remote
    if len(incoming) < 2:
        st.info("This repository has only one branch, so there's nothing to merge.")
        return
    cur = git.current_branch()
    c1, c2, c3 = st.columns([2, 2, 1.3], vertical_alignment="bottom")
    tgt = c1.selectbox("Merge into (target)", targets, index=targets.index(cur) if cur in targets else 0, key="merge_target")
    inc_opts = [b for b in incoming if b != tgt and b.split("/", 1)[-1] != tgt] or incoming
    inc = c2.selectbox("Bring in (incoming)", inc_opts, key="merge_incoming")
    if c3.button("Start merge", key="start_merge", type="primary"):
        try:
            res = start_merge(git, tgt, inc)
        except RepoLinkError as exc:
            st.error(str(exc)); return
        if not res["conflicts"]:
            st.success("Merged cleanly, no conflicts to resolve.")
        st.session_state.pop("fp", None); st.rerun()
    if not is_clean(git):
        st.caption("Your working tree has uncommitted changes; commit or stash them before starting a merge.")


def render_structural(git: Git, store: Store, report) -> None:
    for fr in report.files:
        f = fr.file
        md(f'<div class="gt-muted-card gt-fade"><h3>{esc(f.path)}</h3>{esc(f.kind)}. '
           f'This is a structural conflict, so choose which side\'s whole file to keep.</div>')
        ca, cb = st.columns(2, gap="medium")
        for col, choice, k in ((ca, "current", "a"), (cb, "incoming", "b")):
            if col.button(f"Accept Side {k.upper()} (whole file)", key=f"accept_{k}", type="primary"):
                name = f"Side {k.upper()}"
                try:
                    out = resolve_and_finish(git, store, f.path, 0, choice, provenance=f"kept {name} (whole file)", whole_file=True)
                except (ResolveError, GitError, OSError) as exc:
                    st.error(str(exc)); return
                st.session_state["result"] = {"name": name, "path": f.path, **out}
                st.session_state.pop("fp", None); st.rerun()
        break


# =====================================================================================
# Main
# =====================================================================================
def main() -> None:
    st.set_page_config(page_title="GitTrace: AI-Powered Conflict Resolution", page_icon="🎃", layout="wide",
                       initial_sidebar_state="collapsed")
    md(CSS)
    ss_default("page", "landing")
    ss_default("settings", {"ai": True, "host": os.environ.get("GITTRACE_OLLAMA_HOST", DEFAULT_HOST),
                            "model": os.environ.get("GITTRACE_OLLAMA_MODEL", DEFAULT_MODEL)})
    if "start_repo" not in st.session_state:
        start = os.environ.get("GITTRACE_REPO") or os.getcwd()
        g = Git(start)
        st.session_state["start_repo"] = str(g.toplevel) if g.ok("rev-parse", "--git-dir") else None
    if st.session_state["page"] == "dashboard" and st.session_state.get("repo_path"):
        page_dashboard()
    else:
        page_landing()


if __name__ == "__main__":
    main()
