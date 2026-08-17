# Agent operating brief

This repository is designed to be operated by any capable coding agent. Read
`CLAUDE.md`, `notes/CONTRACT.md`, `notes/BUDGET.md`, `notes/METHOD.md`, and
`notes/EXPERIMENTS.md` before changing a generator. For first-time setup, see
`docs/SETUP.md`.

## Environment

The Cascade reference clone lives at `$CASCADE_DIR` (env var) or the path
passed to `--cascade-dir`. It is a read-only upstream reference — pull and
reinstall it, never edit it.

If `CASCADE_DIR` is not set, the controller defaults to `/root/cascade`. In
non-root environments, set it explicitly:

```bash
export CASCADE_DIR=/path/to/cascade
```

Before starting work, check what the host actually has — both commands are
read-only and exit 0 only when every stage is ready:

```bash
bash scripts/setup.sh --check                     # setup steps, verified in place
.venv/bin/python -m miner.status --doctor --json  # full checklist with a fix per stage
```

The competition itself moves. `notes/UPSTREAM.md` and
`notes/upstream-state.json` are **generated** from the clone by
`scripts/upstream_state.py` (refreshed by the `upstream-sync` workflow) — never
hand-edit them, and where they disagree with prose in `notes/`, they are right.
When the doctor's `upstream sync` check is red, or a sync PR reports failing
prose pins, folding those changes into `notes/CONTRACT.md` comes before any
generator work: designing against a stale contract wastes the round.

## For automated improvement events

- Work only in this repository and treat the Cascade reference clone as a
  read-only upstream that may be pulled and reinstalled, never edited.
- Create or improve one deployable candidate in `generators/candidate`.
- Keep each pass bounded to one stated hypothesis and record it in
  `notes/EXPERIMENTS.md`.
- Run static verification and inexpensive deterministic checks.
- Do not claim an improvement without comparable measured results.
- Request privileged work through `runs/agent-request.json`. Allowed actions
  are `gpu_evaluation`, `create_hotkey`, `register_hotkey`, and
  `submit_candidate`; include a clear reason and `estimated_hours` when useful.
  The controller will request human approval or invoke an explicitly
  allowlisted command in autonomous mode.
- Never read or print wallet secrets, rent pods, register, deploy, submit
  on-chain, commit, or push directly.
- Never place credentials, wallet files, mnemonics, private pool data, scores,
  or generated runtime state in Git.

## Agent-native interfaces

If your session supports MCP, prefer the typed tool surface over shelling into
`miner.*` modules:

```bash
claude mcp add cascade-miner -- .venv/bin/python -m miner.mcp_server
```

Key tools: `miner_status` and `get_brief` to orient, `run_quick_verify` for
free local checks, `list_heat_entrants` and `fetch_generator` to study public
generators from completed rounds (past kings and heat entrants — never submit
someone else's work), `log_experiment`/`list_experiments` for the structured
ledger (`runs/experiments.jsonl` — record the narrative in
`notes/EXPERIMENTS.md` as well), `report_issue`/`list_issues` to file bugs and
feature requests about the harness itself (`runs/issues.jsonl`, surfaced to
the operator via `miner_status` — check `list_issues` first to avoid
duplicates), and `request_action` to queue a privileged action for approval.
`request_action` and `runs/agent-request.json` are requests only — the
controller applies its mode/policy gate before anything runs. File contracts
are documented under `schemas/`.

Without MCP, the same issue tracker is
`python -m miner.issues report --kind bug|feature --title "..." --detail "..."`.
Use it for harness defects and missing capabilities, not experiment results —
those belong in the experiment ledger.

When a `policy.toml` exists, autonomous mode is bounded by per-action caps
(runs and estimated hours per trailing 24h). A declined action queues for
human approval; it is not an error, and not a loophole to work around.

## Agent backends

The improvement hook (`scripts/improve_candidate.py`) spawns a non-interactive
subagent for each pass. It supports `hermes`, `claude`, `codex`, `custom`, and
`auto` (first found). See `docs/SETUP.md` for details. The subagent has no
conversation context — everything it needs is in this file, `CLAUDE.md`, and
the `notes/` directory.

### Running inside an agent session

An agent that *is* the session has no CLI to invoke itself with. Set
`CASCADE_AGENT=hermes-native` and the hook publishes the prompt to
`runs/improve-request.json`, then blocks on `runs/improve-response.json`
(`CASCADE_IMPROVE_RESPONSE_TIMEOUT`, default 3600s). Read and answer it with:

```bash
python skills/cascade-miner/scripts/improve-request show
python skills/cascade-miner/scripts/improve-request respond --status completed \
    --detail "one line on what changed"
```

Statuses are `completed`, `failed`, `rejected`, and `skipped`; only `completed`
makes the pass succeed. Do the work before responding — the response ends the
pass. `hermes-native` is never selected by `auto`.

`skills/cascade-miner/SKILL.md` carries the same operating context for sessions
that load skills.
