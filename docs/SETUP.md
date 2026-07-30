# Setup Guide

This guide covers getting cascade-miner running on a fresh machine or container.
It works for both human operators and AI agents.

## The short version

`scripts/setup.sh` performs steps 2, 4, 5, 7, 8, and 9 below — everything that
neither spends money nor touches wallet secrets — and ends with a summary of
what is ready and what still needs you:

```bash
bash scripts/setup.sh --cascade-dir /path/to/cascade
bash scripts/setup.sh --cascade-dir /path/to/cascade --dry-run   # print the plan
```

Every step is idempotent, so re-running it is safe: an existing venv, SSH key,
or eval-pool snapshot is left alone. Use `--skip-venv`, `--skip-lium`,
`--skip-ssh-key`, `--skip-pool`, `--skip-seed`, or `--skip-verify` to drop
individual steps, and `--eval-pool-snapshot` to choose what to mirror.

It deliberately does **not** create wallets (step 6) or register (also step 6).
Read on for those, and for what each automated step is doing.

## Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv) (Python package manager)
- A GitHub PAT with access to `TensorLink-AI/cascade-miner` and `TensorLink-AI/cascade`
- A [Lium](https://lium.io) account for GPU pod rental
- A HuggingFace token (read access to `Tensor-Link/cascade-eval-pool`)

## Step 1: Clone the repos

```bash
git clone https://github.com/TensorLink-AI/cascade-miner.git
git clone https://github.com/TensorLink-AI/cascade.git   # read-only reference
```

The cascade reference clone is read-only. Never edit it. Refresh with `git pull`
and reinstall when the upstream changes.

## Step 2: Create the Python environment

```bash
cd cascade-miner
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python '/path/to/cascade[hippius,chain]'
```

The `train` extra (torch) is only needed on GPU pods — skip it locally if you
don't have a GPU or are disk-constrained. The `hippius` and `chain` extras are
needed for the controller, eval pool sync, and deployment.
`scripts/setup.sh` installs `train,hippius,chain`; pass
`--extras hippius,chain` to leave torch out.

Verify the install:

```bash
.venv/bin/python -c "import cascade; import bittensor; print('OK')"
```

## Step 3: Configure `.env`

```bash
cp example.env .env
# Edit .env and fill in your credentials
```

Required values:

| Key | What it's for | Where to get it |
|-----|--------------|----------------|
| `HF_TOKEN` | Download the gated eval pool dataset | https://huggingface.co/settings/tokens |
| `LIUM_API_KEY` | Rent GPU pods for evaluation | https://lium.io |
| `HIPPIUS_HUB_TOKEN` | Upload generators to Hippius registry | Hippius dashboard |
| `CASCADE_MINER_HOTKEY` | Your SS58 hotkey (for receipt lookups) | Created in step 6 |

Optional but recommended:

| Key | Default | What it does |
|-----|---------|-------------|
| `CASCADE_MODE` | `human` | `human` = approve GPU spend; `autonomous` = auto-allowlisted actions |
| `CASCADE_AGENT` | `auto` | Which agent CLI to use for improvement passes (`hermes`, `claude`, `codex`, `custom`) |
| `CASCADE_EVAL_POOL_SNAPSHOT` | `latest` | Which eval-pool snapshot to mirror (`latest`, `all`, or a dated name) |
| `CASCADE_EVAL_SEEDS` | `0,1,2` | Seeds for paired evaluation |
| `CASCADE_TRAIN_HOURS` | `1` | Training budget per run (heat budget) |
| `CASCADE_GPU` | `RTX4090` | GPU type to rent from Lium |

**Important:** Values containing spaces must be quoted in `.env`:
```bash
CASCADE_APPROVED_EVAL_COMMAND=".venv/bin/python scripts/run-gpu-evaluation"
```
An unquoted value with a space causes bash to execute the second word as a
command during `source .env`.

`.env` is gitignored. Never commit real credentials.

## Step 4: Install the Lium CLI

```bash
curl -fsSL https://lium.io/install.sh -o /tmp/lium_install.sh
bash /tmp/lium_install.sh
export PATH="$HOME/.lium/bin:$PATH"
lium balance   # verify it works
```

Lium is installed outside the Python venv deliberately. The CLI found as `lium`
on `PATH` is authoritative.

## Step 5: Generate an SSH key for pod access

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q
```

`pods.py` uses `~/.ssh/id_ed25519` to connect to rented GPU pods. The lium CLI
also reads this path from its config.

## Step 6: Create a Bittensor wallet

bittensor 10.5.0 ships the Wallet Python API but no `btcli` console script.
Create the wallet programmatically:

```bash
.venv/bin/python -c "
import bittensor
w = bittensor.Wallet(name='cascade-miner', path='wallets')
w.create_new_coldkey(n_words=12, use_password=False, overwrite=False, suppress=False)
w.create_new_hotkey(n_words=12, use_password=False, overwrite=False, suppress=False)
print(f'Coldkey: {w.coldkeypub.ss58_address}')
print(f'Hotkey:  {w.hotkeypub.ss58_address}')
"
```

**Save the mnemonics.** They are the only way to recover the keys. Keep
`overwrite=False`: with `overwrite=True` this silently replaces an existing
coldkey, and the funds under the old one are unrecoverable without its mnemonic.
Keep the wallet directory outside this repository.

This is the one step `scripts/setup.sh` will not do for you. It reports whether
the wallet named by `--wallet-name` exists, and `--with-wallet` delegates to
your own `CASCADE_CREATE_HOTKEY_COMMAND` (default `ops/create-next-hotkey`, a
refusing stub). Harness code never handles wallet secrets.

The hotkey must be registered on netuid 91 (mainnet/finney) before submitting.
Registration costs ~1126 τ and burns the registration fee. This step requires
TAO in the coldkey and is never automated — agents must request it through
`runs/agent-request.json` and a human must approve.

For testnet (netuid 259), registration is free.

## Step 7: Download the eval pool (latest snapshot only)

The eval pool is 10k+ files across 13 dated snapshots. The controller mirrors
only the newest one by default, which is a few minutes rather than 20+ and well
clear of the Hub's rate limits:

```bash
.venv/bin/python -m miner.controller --sync-pool
```

This checks the dataset revision first — one fast API call — prints the
snapshot, file count, and approximate size before downloading anything, records
the revision in `runs/controller-state.json`, and retries once if the Hub fails
mid-transfer. Re-running it downloads nothing when the revision is unchanged.

`--eval-pool-snapshot` chooses what to mirror:

| Selector | Effect |
|----------|--------|
| `latest` (default) | just the newest dated snapshot |
| `all` | the whole dataset — 10k+ files, 20+ min, 429-prone |
| `2026-07-16` | exactly that snapshot; an unknown name lists the real ones |

Snapshots already under `pools/snapshots/` are never deleted, so a narrower
selector shrinks what is *downloaded*, not what you can train against. Keep the
selector fixed across the arms of one experiment: a king and challenger scored
over different snapshot sets are not paired.

## Step 8: Seed the controller state

Run one poll. It records the Cascade head, `chain.toml`, the eval-pool
revision, and the latest round receipt — and deliberately invokes no
improvement hook on a first run:

```bash
.venv/bin/python -m miner.controller --once --cascade-dir /path/to/cascade
```

Do not hand-write `runs/controller-state.json`. Writing `initialized: true`
with an empty `last_receipt` makes the controller treat the next poll as a
resumption and start firing improvement hooks, while `scripts/run-gpu-evaluation`
has no `king_gen_ref` to fetch a control from. Each detection is persisted as
soon as it is made, so an interrupted poll never re-downloads the pool.

## Step 9: Verify the starter generator

```bash
.venv/bin/cascade verify ./generators/candidate --chain-toml /path/to/cascade/chain.toml
```

This runs every check the trainer runs: layout, static guard, hash-locked
deps, and the determinism check (generates the corpus twice and compares
digests). If this passes, your environment is ready.

**Note:** On machines with <4GB RAM, the determinism check may OOM on heavy
generators (20+ families, 4096 length, 16384 series). Reduce `batch_size` in
`config.json` or use a lighter generator for the check.

## Step 10: Start the loop

With state seeded in step 8, the second poll is the first one that can invoke
an improvement hook:

```bash
set -a; source .env; set +a
export PATH="$HOME/.lium/bin:$PATH"
.venv/bin/python -m miner.controller \
  --cascade-dir /path/to/cascade \
  --chain-toml /path/to/cascade/chain.toml \
  --interval 300 --hotkey "<your-ss58-hotkey>" \
  --improve-command ".venv/bin/python scripts/improve_candidate.py" \
  --approved-eval-command ".venv/bin/python scripts/run-gpu-evaluation"
```

Pass empty `--improve-command ""` and `--approved-eval-command ""` to watch
without acting. Nothing here spends money: in `human` mode (the default) GPU
evaluation waits for `--approve`, and submission is never automatic. See the
README's *Continuous controller* section for both modes.

## Non-root / container environments

If you're not running as root (containers, agent sessions, shared machines):

- `pods.py` uses `~/.ssh/id_ed25519` (not `/root/.ssh/`) — works for any user
- `pods.py` falls back to `tar | ssh` when `rsync` is not installed
- Set `CASCADE_DIR` env var to point at the cascade reference clone
- The controller accepts `--cascade-dir` and `--chain-toml` flags
- The GPU pods themselves run as root — the `/root/` paths in `pods.py`
  (like `REMOTE_ROOT = "/root/cascade-miner"`) are correct; they refer to the
  pod's filesystem, not yours

## Agent backends

The controller's improvement hook (`scripts/improve_candidate.py`) supports
multiple agent CLIs for the improve → verify → request-eval loop:

| `CASCADE_AGENT` | CLI command | Requirement |
|-----------------|-------------|-------------|
| `hermes` | `hermes chat --toolsets terminal -q "<prompt>"` | Hermes Agent installed |
| `claude` | `claude --print --permission-mode acceptEdits "<prompt>"` | Claude Code CLI installed |
| `codex` | `codex exec --ephemeral --sandbox workspace-write -` | Codex CLI installed |
| `custom` | `CASCADE_AGENT_COMMAND` template | Any non-interactive CLI |
| `auto` | tries claude → codex → hermes | first found |

Each backend spawns a fresh, non-interactive subagent that:
1. Reads `AGENTS.md`, `CLAUDE.md`, and `notes/` from the repo
2. Inspects the controller event (round result, cascade update, etc.)
3. Makes one bounded improvement to `generators/candidate/`
4. Runs `cascade verify` and deterministic checks
5. Writes `runs/agent-request.json` if it needs GPU eval or submission
6. Exits

The subagent has no conversation context — everything it needs is in the repo
files and the controller event.

## Quick reference: the eval loop

```bash
# 1. Fetch current king as control
KING_REF=$(.venv/bin/python -c "
import json; print(json.load(open('runs/controller-state.json'))['last_receipt']['summary']['king_gen_ref'])")
.venv/bin/cascade fetch "$KING_REF" --out generators/king-control \
    --chain-toml "$CASCADE_DIR/chain.toml" --network finney

# 2. Fork and modify the candidate (one change, per CLAUDE.md method)
cp -r generators/king-control generators/candidate
# edit generators/candidate/config.json or generator.py

# 3. Record the experiment in notes/EXPERIMENTS.md

# 4. Run paired eval (rents GPU, trains king + candidate, scores, pulls results)
bash scripts/run_paired_eval.sh

# 5. Analyze
.venv/bin/python -m miner.analyze --scores-root scores --king king-control
```
