# GitTrace: AI-powered Git conflict resolution in your terminal

**Hacktoberfest 2026 Hack Day · Problem 01**

GitTrace runs where developers already work: the terminal. When a merge (or cherry-pick, revert or rebase) conflicts, it shows Side A and Side B side by side, together with what Git alone can't tell you:

- **Where each side came from**: `git blame` on the conflicting lines, explicit backport references, `git patch-id` equivalence, ancestry.
- **Whether that exact code was tested** (*Tested in PROD*, *Unverified UAT change*, …). Results are tied to the exact snapshot (tree hash), and a backport never inherits its original's test status.
- **Whether it reached UAT or production**, from deployment records and environment branches.
- **Whether both sides are logically the same fix**, judged by **Gemma running locally in Ollama** from both sides' commit messages and diffs.

Press **A** or **B**. GitTrace writes those lines, stages the file, and creates a **standard Git merge commit** whose message explains the provenance of every choice.

![Conflict screen](docs/tui-conflict.png)

## Install

Requires Python 3.11+, Git, and (for the AI) [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt     # rich, requests  (the only dependencies)
ollama pull gemma4:e4b              # once; then make sure `ollama serve` is running
```

Optional: `pip install -e ".[dev]"` adds the `gittrace` command-suite and pytest.

## Run

```bash
python cli.py                                     # analyse the current directory
python cli.py ~/code/billing-service              # analyse a local clone
python cli.py https://github.com/user/repo.git    # clone to a temp dir, then analyse
python cli.py https://github.com/user/repo.git --merge production uat
                                                  # clone, then start `git merge uat` into production
```

| Key | Action |
|---|---|
| **A** | Keep Side A: write its lines, stage, commit once everything is resolved |
| **B** | Keep Side B |
| **S** | Skip this conflict for now |
| **D** | Details: originating commits, per-environment test status, deployment, lineage evidence |
| **Q** | Abort: nothing is written for the current conflict (`git merge --abort` undoes the merge) |

Options: `--no-ai`, `--model gemma4:e4b`, `--ollama-host http://localhost:11434` (must be local), `--merge TARGET INCOMING`, `--keep` (keep the temp clone), `--report` (show the first conflict and exit). Environment variables `GITTRACE_OLLAMA_HOST` and `GITTRACE_OLLAMA_MODEL` set the defaults.

### Remote URLs
A URL (`https://`, `git@`, `ssh://`, `file://`, or anything ending in `.git`) is cloned with `subprocess` + `git clone` into a `tempfile.mkdtemp()` directory, with `GIT_TERMINAL_PROMPT=0` so it never hangs on a password prompt.

A fresh clone has no merge in progress, so GitTrace offers to start one: pick a target and an incoming branch from a numbered list, or pass `--merge`. Branches that exist only on the remote are merged as `origin/<branch>`.

If you commit a resolution, the clone is **kept** and its path printed, so you can push it. Otherwise the temp directory is deleted. Private repositories: clone them yourself and run `python cli.py <path>`.

### Demo (30 seconds)

```bash
python demo/make_demo_repo.py /tmp/gittrace-demo   # builds a repo and leaves it mid-merge
cd /tmp/gittrace-demo && python /path/to/gittrace/cli.py
```

The demo merges `uat` into `production`, which conflicts in two places in `billing.py`:
1. **GST fix.** Production's hotfix (`amount * 18 / 100` formatted to 2 decimals, *Tested in PROD*) against UAT's modified backport (`round(amount * 0.18, 2)`). Patch IDs differ and the rule-based heuristics can't link them; **Gemma: logically the SAME fix.**
2. **Discount rule.** 5% off from 500 against 10% off above 1000. The heuristic wrongly calls these "probably related"; **Gemma: DIFFERENT changes**, and the false match is rejected.

Press **A**, then **B**:

![Merge complete](docs/tui-merged.png)

## Hackathon rules compliance

| Rule | How GitTrace meets it |
|---|---|
| No cloud AI APIs | The only AI call is `requests.post("http://localhost:11434/api/generate")` in `src/gittrace/ai.py`. No OpenAI, Anthropic or Gemini SDKs or URLs anywhere; a test (`test_no_cloud_ai_in_codebase`) checks this. |
| Local AI only | `ensure_local()` refuses any Ollama URL that isn't `localhost`, `127.0.0.1` or `::1`. Redirects are not followed and proxies are bypassed. |
| Standard Git workflow | Resolutions run real `git add` and `git commit` (a normal two-parent merge commit). Rebases finish with `git rebase --continue`. |
| Runs in the developer's repo | `python cli.py` with no arguments works in the current clone and reads its `.git` directly. |

## How it works

**Conflict detection** (`conflicts.py`) reads `git ls-files -u` stages and labels stage 2/3 by operation: merge, cherry-pick (base = picked commit's parent), revert, or rebase (where Git's "ours" is the *upstream*). It parses conflict markers (merge/diff3/zdiff3 styles) and blames each side's lines.

**Lineage** (`provenance.py`) classifies how two commits are related, with evidence:

| Class | Basis |
|---|---|
| exact original commit | same commit object |
| explicitly referenced backport | `(cherry picked from commit X)` or a `Backport-of:` trailer, followed through chains; flagged if the copy was **modified** |
| equivalent patch | identical `git patch-id --stable` |
| probably related | fuzzy overlap of changed lines, files and subjects (heuristic) |

**Local AI** (`ai.py`, `provenance.apply_ai`): Gemma judges whether the two sides are logically the same fix. Its verdict can confirm or reject a heuristic "probably related" match, but it never overrides the evidence-backed classes and can never produce "equivalent patch". Replies are parsed defensively and cached in `.git/gittrace/ai-cache.json`. If Ollama is down, the TUI shows a warning box and everything else keeps working.

**Testing and deployment provenance** (`store.py`, `status.py`): records live in `.git/gittrace/records.json` and are keyed by commit **and tree hash**. Deployment comes from recorded deploys, ancestry in environment branches, `git cherry` equivalence, or a referenced backport.

**Resolution** (`resolve.py`): write lines → `git add` → `git commit` (or `git rebase --continue`), with a provenance section and `GitTrace-Resolution:` trailers in the message.

## Project structure

```
gittrace/
├── cli.py                    # the terminal UI (Rich): python cli.py [path|url]
├── requirements.txt          # rich, requests
├── demo/make_demo_repo.py    # judge demo repository (left mid-merge)
├── src/gittrace/
│   ├── ai.py                 # local Ollama client (requests POST /api/generate, localhost guard, cache)
│   ├── analysis.py           # Analyzer, provenance badges, Gemma hunk comparison, commit text
│   ├── provenance.py         # references, patch-id, similarity, git cherry, apply_ai()
│   ├── conflicts.py          # operation detection, index stages, markers, blame
│   ├── resolve.py            # write -> git add -> git commit / rebase --continue
│   ├── repolink.py           # branch listing + starting a standard git merge
│   ├── status.py, store.py   # testing & deployment provenance
│   ├── git.py                # subprocess wrapper (no shell)
│   └── cli.py, render.py     # `gittrace` command-suite (record-test, trace, analyze --json, ...)
└── tests/                    # 92 tests
```

The `gittrace` command-suite (after `pip install -e .`) adds scripting commands: `gittrace record-test --commit H -e uat -r passed`, `gittrace record-deploy`, `gittrace trace <commit>`, `gittrace env-branch set production main`, `gittrace analyze --json`, `gittrace ai-check`.

## Tests

```bash
pytest -q
```

The 92 tests include:
- **The terminal UI**, driven by piped keystrokes:
  - the no-argument (current directory) run
  - the Gemma panel and side-by-side layout
  - A/B producing a two-parent merge commit
  - Q leaving the repo untouched, EOF, invalid keys, Details, Skip, report mode
  - remote-URL clone → `--merge` → resolve, with the clone kept and the cleanup path when nothing is committed
  - the interactive merge starter, and error handling
- **The Ollama client**, against a fake server on 127.0.0.1: request shape, cloud-endpoint refusal, the offline and missing-model messages, the cache, tolerant reply parsing, and every `apply_ai` rule.
- **The backend**: merge/cherry-pick/rebase/revert labelling, lineage classes, snapshot-exact testing, deployment evidence and the resolve engine.

## Limitations

- Gemma's verdict is advisory and can be wrong; deterministic evidence always takes precedence.
- Small local models can be slow on the first request while the model loads (240 s timeout); answers are cached after that.
- Test/deploy records live in `.git/gittrace/` and are not part of a clone. Share them with `gittrace export`/`import`.
- Line attribution needs the conflicting text verbatim at the side's commit; hand-edited hunks fall back to file-level commits.
