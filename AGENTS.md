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

## Agent backends

The improvement hook (`scripts/improve_candidate.py`) spawns a non-interactive
subagent for each pass. It supports `hermes`, `claude`, `codex`, `custom`, and
`auto` (first found). See `docs/SETUP.md` for details. The subagent has no
conversation context — everything it needs is in this file, `CLAUDE.md`, and
the `notes/` directory.
