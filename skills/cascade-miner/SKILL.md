---
name: cascade-miner
description: Operate the cascade-miner harness for Bittensor subnet 91 (cascade) from inside an agent session — first-time setup, the improve → verify → screen → request-evaluation → analyse loop, and answering the controller's improvement requests in hermes-native mode. Use whenever the session mentions cascade, SN91, netuid 91, the eval pool, a king/challenger duel, or this repository's controller.
---

# cascade-miner

Operating a purely-algorithmic synthetic time-series generator against the
cascade subnet. The owner trains one fixed forecaster on our corpus and scores
it against the king's. Model, seeds and compute are identical on both sides —
**the generator is the only variable.**

Read `CLAUDE.md` and `AGENTS.md` in the repository root first; they are the
authority, and this file only routes you to the right command. `notes/METHOD.md`
is the canonical evaluation methodology.

## Hard limits

These are not style preferences. Breaking one burns a hotkey or spends money.

- **Never rent a pod, register, or submit on-chain yourself.** Request the
  action by writing `runs/agent-request.json`; the controller turns it into an
  approval. Allowed actions: `gpu_evaluation`, `create_hotkey`,
  `register_hotkey`, `submit_candidate`.
- **Never touch wallet secrets**, mnemonics, or `btcli`. Wallet work is
  delegated to operator-owned wrappers under `ops/` that refuse by default.
- **Never edit the Cascade reference clone** at `$CASCADE_DIR`. Pull and
  reinstall it; never modify it.
- **Never commit** credentials, pool data, scores, or runtime state.

## Setup

One command, idempotent, safe to re-run:

```bash
bash scripts/setup.sh --cascade-dir "$CASCADE_DIR"
```

It creates the venv, installs Cascade, installs the Lium CLI, generates an SSH
key, reports wallet status, mirrors the latest eval-pool snapshot, seeds
`runs/controller-state.json`, and verifies the starter candidate. `--dry-run`
prints the plan without changing anything. Add `--quick-verify` on a host with
under ~4GB RAM, where the full `cascade verify` corpus gets OOM-killed.

Set `CASCADE_DIR` and, in a non-root environment, `LIUM_SSH_KEY` if your key is
not at `~/.ssh/id_ed25519`.

## The loop

1. **Improve** one thing in `generators/candidate/` — a single stated
   hypothesis per pass. Prefer replacing a weak component over adding a new
   one; the model has fixed capacity, so added coverage displaces existing
   coverage rather than accumulating.
2. **Verify** it is deployable and deterministic:
   ```bash
   .venv/bin/cascade verify generators/candidate --chain-toml "$CASCADE_DIR/chain.toml"
   .venv/bin/python scripts/quick_verify.py generators/candidate   # low-memory hosts
   ```
3. **Screen** before asking anyone to spend money:
   ```bash
   .venv/bin/python -m miner.screen generators/candidate \
       --king generators/king-control --n-series 256 --claim <family>+
   ```
   Exit 1 `blocked` — a contract gate fails, fix it. Exit 3 `undosed` — the
   corpus is indistinguishable from the king's at this resolution, so a paired
   eval buys a null; raise the dose instead. Exit 4 — the corpus does not carry
   a claim you made, which means the code does not do what you think. Exit 0
   `measurable` licenses the paid step and nothing more: the screen ranks
   nothing and never predicts a duel. Read its `challenges` list before writing
   the request — it asks what you are trading away and what effect you expect.
4. **Request evaluation** — write `runs/agent-request.json`:
   ```json
   {"action": "gpu_evaluation", "reason": "...",
    "candidate_path": "generators/candidate", "estimated_hours": 1}
   ```
   The controller queues it for approval (human mode) or runs the allowlisted
   command (autonomous mode). Training is backgrounded on the pod and polled,
   so progress appears in the controller's event log as it happens.
5. **Analyse** the paired result — the point estimate is not the verdict:
   ```bash
   .venv/bin/python -m miner.analyze --scores-root scores --king king-control
   ```
   The throne is decided on the lower confidence bound of a paired cluster
   bootstrap clearing `[scoring] win_margin`, not on the geomean.
6. **Record** the hypothesis and outcome in `notes/EXPERIMENTS.md`. State the
   seed count and noise floor with every number. Say plainly when a result is
   null.

## Reading a result honestly

- A verdict needs paired window-level components from a **freshly trained**
  king and challenger in the same batch. An aggregate from an older run is not
  a control.
- Measure the noise floor before believing anything: run the same generator
  twice at the same seed and see how far apart the scores land. Anything inside
  that band is noise — report it as noise.
- Recompute from saved per-window components. The subnet's scoring code
  changes, and figures produced under different metric versions are not
  comparable.
- Diagnose per-source, not just in aggregate. An aggregate null can hide large
  gains on targeted sources offset by losses elsewhere.

## hermes-native mode

An agent that *is* the session cannot invoke itself as a CLI, so
`CASCADE_AGENT=hermes-native` replaces the subprocess call with a file
handshake. The improvement hook publishes the prompt and blocks:

```bash
python skills/cascade-miner/scripts/improve-request show     # pending request
python skills/cascade-miner/scripts/improve-request respond --status completed \
    --detail "replaced the seasonal family with a regime-switching one"
```

Statuses: `completed` (hook exits 0), `failed`, `rejected`, `skipped` (hook
exits 1). If nobody answers within `CASCADE_IMPROVE_RESPONSE_TIMEOUT`
(default 3600s) the hook exits 124 and leaves the request in place.

Do the work first, respond second — the response ends the pass.

## What this skill deliberately does not wrap

There is no `eval.sh`. Paid evaluation is an approval boundary, not a
convenience wrapper, and a script that rents a pod on an agent's behalf would
defeat it. Request it through `runs/agent-request.json` instead.
