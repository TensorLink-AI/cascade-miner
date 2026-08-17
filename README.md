# cascade-miner

A miner for the **cascade** Bittensor subnet (netuid 91, finney).

cascade holds the model byte-identical and scores the *data*: you submit a
purely-algorithmic time-series generator, the owner trains a fixed forecaster on
your corpus, and a private rotating held-out set decides whether your data
trains a better forecaster than the reigning king's.

Only the **single best** challenger from each heat reaches the duel, and it must
clear a confidence-bound margin to take the throne. Rounds are ~12h.

Since 2026-08-05 that training run is **warm-started**: after a ripe reign the
subnet promotes up to three reign checkpoints and rotates rounds across them,
so a round no longer necessarily starts from random init. King and challenger
still share the round's init — the generator remains the only variable — but
your corpus is increasingly judged on what it adds to a partially-trained
model. `notes/CONTRACT.md` has the mechanics; `notes/UPSTREAM.md` has the live
values.

## Ground rule

`/root/cascade` is a read-only reference clone, installed into this project's
venv as an ordinary library. **We never edit it.**

```bash
git -C /root/cascade pull
uv pip install --python .venv/bin/python --reinstall /root/cascade
```

## Staying in sync with the subnet

The competition changes as decisions land upstream, and stale notes cost real
rounds — warm-start went live on 2026-08-05 while these notes still described a
from-scratch trainer. Two halves keep that from repeating:

| | what | who runs it |
|---|---|---|
| **Facts** | `scripts/upstream_state.py` extracts every miner-facing `chain.toml` key and `DEC-CA-` node into `notes/upstream-state.json` + `notes/UPSTREAM.md` | the `upstream-sync` workflow, every 6h — it pushes the regenerated snapshot and opens/refreshes one PR |
| **Prose** | `notes/CONTRACT.md` interprets those facts | you (or your agent), prompted by that PR |

`tests/test_stale_references.py` pins the prose to the snapshot, so a fact that
contradicts a sentence fails the suite and the failure names the stale claim.
The sync PR runs the suite in-job and reports the result in its body: green
means facts-only, red means prose needs folding first. Keys we don't track yet
are inventoried by name, so a key *added* upstream also shows up in that diff,
and a broken sync run files an issue rather than going quiet.

Locally, `python -m miner.status --doctor` runs the same comparison against the
clone this host has, so "the clone was pulled but nothing was regenerated"
shows up in the readiness checklist.

Merging is **not** the end of it — the runner cannot touch the box where
scoring happens:

```bash
bash scripts/sync.sh          # pull + REINSTALL + regenerate + stamp + test
bash scripts/sync.sh --check  # cron-friendly: exit 1 when behind, change nothing
```

Scoring imports come from the *installed* package, so an un-reinstalled venv
keeps scoring with the old metric, and numbers from before the change are not
comparable to numbers after it (`CLAUDE.md` Rule 3).

## Layout

```
miner/
  pods.py        rent (-c 1), provision, assert deps, TTL guard, results puller
  evaluate.py    train N seeds x M snapshots; saves per-window score components
  analyze.py     live-metric geomean + paired cluster-bootstrap LCB vs the king
  submit.py      prepare -> verify -> upload -> commit --ref -> confirm reveal
  rounds.py      boundary / reveal-margin math; the real commit deadline
  controller.py  watch Cascade updates + round receipts; trigger improvement hooks
generators/      candidate generator repos; each is a deployable submission dir
pools/           eval-pool snapshots (gitignored — data, not code)
notes/           CONTRACT.md, BUDGET.md, METHOD.md, EXPERIMENTS.md
                 UPSTREAM.md + upstream-state.json (GENERATED — see below)
```

A submission is exactly three files — `generator.py`, `config.json`,
`requirements.txt`. Nothing else in this repo ships.

## Setup

One command does everything that does not spend money or touch wallet secrets —
venv, Cascade install, Lium CLI, SSH key, latest eval-pool snapshot, controller
state, `cascade verify` — and prints what is ready and what still needs you:

```bash
bash scripts/setup.sh --cascade-dir /path/to/cascade
bash scripts/setup.sh --cascade-dir /path/to/cascade --dry-run   # print the plan
bash scripts/setup.sh --check                                    # verify only, change nothing
```

The exit status is meaningful: 0 when nothing needs the operator, 1 while the
summary lists outstanding actions, so scripts and agents can gate on it.
`--check` re-runs the same readiness checks without installing or downloading
anything. Preflight also lints `.env` — an unquoted value containing a space
would make `source .env` execute the second word as a command, and leftover
`replace_me` placeholders are flagged before they fail a run later.

Every step is idempotent: an existing venv, SSH key, or eval-pool snapshot is
left alone. `--skip-venv`, `--skip-lium`, `--skip-ssh-key`, `--skip-pool`,
`--skip-seed`, and `--skip-verify` drop individual steps, and
`--eval-pool-snapshot` selects what to mirror (see *Continuous controller*).

Wallet creation is deliberately **not** done by the script. It reports whether
the wallet named by `--wallet-name` exists, and `--with-wallet` delegates to
your own `CASCADE_CREATE_HOTKEY_COMMAND` (default `ops/create-next-hotkey`,
which refuses until you implement it). Registration stays manual: it burns TAO.

The rest of this section is the same work done by hand.

You need a Bittensor coldkey and hotkey before submitting. Create them with
`btcli` if they do not already exist:

```bash
btcli wallet new-coldkey --wallet-name my-miner
btcli wallet new-hotkey --wallet-name my-miner --wallet-hotkey default

# register the hotkey on Cascade mainnet (burns the current registration cost)
btcli subnets register --netuid 91 --network finney \
  --wallet-name my-miner --wallet-hotkey default
```

Keep the wallet under Bittensor's wallet directory; never copy private wallet
files into this repository. Deploy commands require the corresponding
`--wallet-name my-miner --wallet-hotkey default` arguments.

Create the Python environment and install the local Cascade checkout:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python '/root/cascade[train,hippius,chain]'
```

Install the Lium CLI if this operator host will rent GPU pods:

```bash
curl -fsSL https://lium.io/install.sh | bash
lium --version
```

The executable found as `lium` on the operator's `PATH` is authoritative;
Lium is deliberately not installed into this project's Python venv. Restart
the shell or add the installer-reported binary directory to `PATH` if
`lium --version` is not found.

Copy the environment template and fill in the credentials you use:

```bash
cp example.env .env
# edit .env and replace hf_replace_me with a real read token, then export it
set -a
source .env
set +a
```

For Hippius, this harness only needs `HIPPIUS_HUB_TOKEN` to upload generator
artefacts for deployment; no Hippius S3 or pool-S3 credentials are required.
Set `HF_TOKEN` for the controller's evaluation-pool revision checks and downloads;
authenticated requests avoid the public Hub's low unauthenticated rate limits.
By default a pool sync fetches only the newest snapshot, and unchanged later
revisions download nothing at all. Set `LIUM_API_KEY` when renting GPU pods.
`.env` is gitignored; never commit real tokens, wallet secrets, or mnemonics.

Mirror the eval pool without starting the controller loop:

```bash
.venv/bin/python -m miner.controller --sync-pool                      # newest snapshot
.venv/bin/python -m miner.controller --sync-pool --eval-pool-snapshot all
```

## Operating with your agent

The harness is agent-neutral: Hermes, Claude Code, Codex, or any
non-interactive CLI or MCP client can run the whole mining loop. `AGENTS.md`
is the operating brief every backend reads; `skills/cascade-miner/SKILL.md`
packages the same rules for sessions that load skills. Three integration
paths, chosen by who is in charge:

| Who drives | Path | Start with |
|------------|------|------------|
| Your agent, interactively | MCP server | `.venv/bin/python -m miner.mcp_server` (stdio) |
| The controller, per round event | improvement hook | `CASCADE_AGENT=hermes` (or `claude`, `codex`, `custom`) |
| The controller, *inside* an agent session | file handshake | `CASCADE_AGENT=hermes-native` |

**Your agent drives (MCP).** The server is dependency-free stdio JSON-RPC —
point any MCP client at the command below. Tools cover status, receipts, heat
entrants, free verification, public-generator fetches, the experiment ledger,
the issue tracker (bug reports and feature requests about the harness), and
`request_action` (which queues an approval, never executes). See
*Agent-native interfaces* below for the full list.

```bash
claude mcp add cascade-miner -- .venv/bin/python -m miner.mcp_server
# any other MCP client: launch `.venv/bin/python -m miner.mcp_server` as a stdio server
```

**The controller drives (improvement hook).** The controller watches Cascade
updates, pool rotations, and round receipts, and invokes
`scripts/improve_candidate.py` once per event batch. That hook spawns your
agent non-interactively — `CASCADE_AGENT=hermes` runs
`hermes chat --toolsets terminal`, `claude`/`codex` use those CLIs, `custom`
takes any command template via `CASCADE_AGENT_COMMAND`, and `auto` picks the
first CLI found. The subagent gets no conversation context: everything it
needs is in `AGENTS.md`, `CLAUDE.md`, `notes/`, and the controller event.

```bash
set -a; source .env; set +a
CASCADE_AGENT=hermes .venv/bin/python -m miner.controller --mode human --interval 300 \
  --improve-command ".venv/bin/python scripts/improve_candidate.py"
```

**Inside an agent session (hermes-native).** An agent that *is* the session
has no CLI to invoke itself with, so the hook swaps the subprocess for a file
handshake: it publishes `runs/improve-request.json` and blocks until the
session answers. See *Continuous controller* below for the show/respond
commands.

Agents can self-serve setup and gate on readiness — both commands are
read-only and exit 0 only when the host is ready:

```bash
bash scripts/setup.sh --check
.venv/bin/python -m miner.status --doctor --json
```

Whichever path is used, the privilege boundary is identical: an agent may
edit `generators/candidate`, verify, analyze, and log experiments, and may
*request* `gpu_evaluation`, `create_hotkey`, `register_hotkey`, or
`submit_candidate` — but the controller's mode/policy gate decides whether
anything paid or on-chain actually runs. Agents never hold wallet secrets,
never rent pods directly, and never commit or push.

## The loop

The latest controller receipt identifies the reigning king by immutable
generator reference. Fetch that exact artefact and train it as a fresh control;
do not compare the challenger with a score from an earlier run.

```bash
# 1. fetch the reigning king named by the latest audited receipt
KING_REF=$(.venv/bin/python -c \
  'import json; print(json.load(open("runs/controller-state.json"))["last_receipt"]["summary"]["king_gen_ref"])')
.venv/bin/cascade fetch "$KING_REF" --out generators/king-control \
    --chain-toml /root/cascade/chain.toml --network finney

# 2. structural + determinism gate (every check the trainer runs)
.venv/bin/cascade verify ./generators/<name> --chain-toml /root/cascade/chain.toml

# 3. train both arms on identical seeds/windows, saving per-window components
for seed in 0 1 2; do
  .venv/bin/python -m miner.evaluate ./generators/king-control \
      --chain-toml /root/cascade/chain.toml --pools-root pools/snapshots --seed "$seed"
  .venv/bin/python -m miner.evaluate ./generators/<name> \
      --chain-toml /root/cascade/chain.toml --pools-root pools/snapshots --seed "$seed"
done

# 4. --king is the score-run name (generator directory basename), not a path
.venv/bin/python -m miner.analyze --scores-root scores --king king-control

# 5. submit — privileged; human-approved or explicitly allowlisted autonomous action
.venv/bin/python -c "from miner import submit; ..."   # commit(confirm=True)
```

The receipt also records the king UID and published verdict fields, but
`king_gen_ref` is the auditable control input. If no controller state exists
yet, run the controller once to seed it and again after a new receipt is
published. See `notes/METHOD.md` for the pairing and component-storage rules.

## Two things that will bite you

**Measurement noise.** Training is not bit-reproducible (nondeterministic CUDA
kernels compound over ~50k steps). Measure your noise floor by rerunning one
generator at one seed, and treat smaller differences as noise. Detecting a few
percent reliably takes closer to ten seeds than three.

**Metric drift.** The subnet's scoring code changes, and a change has already
reversed a candidate ranking outright. `evaluate.py` stores raw per-window
components precisely so results can be re-scored later; always recompute rather
than trusting a stored aggregate.

## Deploy notes

`cascade deploy` cannot upload — its uploader passes a token the registry
rejects for blob upload. Upload via `hippius_hub.upload_folder(token=None)` and
hand `cascade deploy` a `--ref`. `miner/submit.py` does this.

Your registry project must be **public** or the trainer's anonymous fetch fails
and the submission is rejected every round while the on-chain commit looks fine.
`submit.check_pullable()` verifies this with all credentials unset.

Use a fresh random repo name per deploy so content stays as hidden as the
timelocked pointer.

## Continuous controller

`miner/pods.py` is an explicit operator library; nothing imports or runs it
automatically. The controller deliberately never rents GPUs. An operator must
authorize `scripts/run-gpu-evaluation` as the improvement hook's
`gpu_evaluation` action. Human mode still requires approval, and autonomous
mode still requires the action to be allowlisted.

### Money-spending boundary

The commands configured for privileged actions are the deliberate boundary
between agent edits and real-world effects. `scripts/run-gpu-evaluation` is the
shipped, non-interactive GPU implementation: after approval it rents exactly
one Lium GPU, fetches the receipt-pinned king, trains king and candidate on the
same seeds, continuously pulls scores, and terminates the pod even on failure.
It reads `candidate_path` and `estimated_hours` from the approval context.

The three `ops/` commands ship only as refusing stubs. Operators must replace
them with reviewed wrappers appropriate to their protected wallet setup before
enabling `create_hotkey`, `register_hotkey`, or `submit_candidate`. This is
intentional: the repository does not guess how wallet secrets are unlocked.
The supplied `example.env` points `CASCADE_APPROVED_EVAL_COMMAND` at the GPU
script; copying it does not bypass either controller mode's gate.

`miner.controller` continuously fast-forwards the read-only Cascade reference
checkout, records changes to `chain.toml`, mirrors the
[`Tensor-Link/cascade-eval-pool`](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool)
dataset into `pools/` (its dated snapshots land in `pools/snapshots/`), audits
each newly published round receipt, and
invokes an improvement hook once for every new code, pool, or round event.

Multiple changes discovered in one poll are coalesced into one improvement
batch. By default there is at most one pass per round, with a two-hour agent
timeout.

Startup verifies that the selected venv interpreter can import both Cascade and
`huggingface_hub`. The first poll seeds controller state without invoking an
agent.

Each poll checks the dataset revision — one fast API call — before downloading
anything, and mirrors only the snapshot named by `--eval-pool-snapshot`
(`CASCADE_EVAL_POOL_SNAPSHOT`):

- `latest` (default) fetches just the newest dated snapshot. Mirroring all 13
  is 10k+ files, takes 20+ minutes, and invites Hub rate limiting (429).
- `all` mirrors the whole dataset.
- A dated name such as `2026-07-16` mirrors exactly that snapshot. An unknown
  name fails with the list of snapshots the dataset actually publishes.

Snapshots already under `pools/snapshots/` are never deleted, so a narrower
selector shrinks what is *downloaded*, not what you can train against. The
controller announces the snapshot, file count, and approximate size before a
download starts, and retries once if the Hub fails mid-transfer. An unchanged
revision downloads nothing; a first run against an already-populated pool
records the revision instead of re-fetching files that are on disk. Each
detection — Cascade head, pool revision, round receipt — is written to the state
file as soon as it is made, so a failure later in the poll cannot cost a second
20-minute download.

Because a narrower selector changes which snapshots `miner.evaluate` scores,
keep it fixed across the arms of one experiment: a king and challenger scored
over different snapshot sets are not paired. A dirty `/root/cascade`
checkout produces one actionable `cascade_dirty` event and exits nonzero rather
than repeating a merge failure. Revert local changes there; never commit them.

The controller has two explicit modes:

- `human` (default) stops when candidate changes need review, creates an
  approval request, and waits. It never starts paid evaluation without
  `--approve`.
- `autonomous` runs explicitly allowlisted action commands without asking and
  can perform multiple bounded improvement/evaluation passes. Set
  `--max-improvements-per-round` to the maximum passes per round: after each
  successful evaluation the controller immediately starts the next improvement
  pass in the same poll, until the cap (or a policy limit) stops it. Human
  mode instead pauses at every approval, so its cadence is one pass per
  approved request.

Agents never receive wallet secrets or execute privileged commands directly.
Human mode approves named actions individually. Autonomous mode can create and
register hotkeys or submit on-chain only when the operator both configures the
corresponding command and adds that action to `--autonomous-actions`.

Run one check:

```bash
set -a; source .env; set +a
.venv/bin/python -m miner.controller --once
```

Run continuously, checking every five minutes:

```bash
set -a; source .env; set +a
.venv/bin/python -m miner.controller \
  --mode human \
  --interval 300 \
  --hotkey "<miner-ss58-hotkey>" \
  --hook-timeout 7200 \
  --max-improvements-per-round 1 \
  --sync-command "uv pip install --python .venv/bin/python --reinstall /root/cascade" \
  --approved-eval-command ".venv/bin/python scripts/run-gpu-evaluation" \
  --improve-command ".venv/bin/python scripts/improve_candidate.py"
```

For bounded autonomous evaluation:

```bash
.venv/bin/python -m miner.controller \
  --mode autonomous \
  --interval 300 \
  --max-improvements-per-round 3 \
  --approved-eval-command "./scripts/run-gpu-evaluation" \
  --create-hotkey-command "./ops/create-next-hotkey" \
  --register-hotkey-command "./ops/register-next-hotkey" \
  --submit-command "./ops/submit-candidate" \
  --autonomous-actions "gpu_evaluation" \
  --improve-command ".venv/bin/python scripts/improve_candidate.py"
```

### Iterating several times in one round

To let the miner try, say, five experiments inside a single ~12h round, run
autonomous mode with the per-round cap raised:

```bash
.venv/bin/python -m miner.controller \
  --mode autonomous \
  --interval 300 \
  --max-improvements-per-round 5 \
  --approved-eval-command "./scripts/run-gpu-evaluation" \
  --autonomous-actions "gpu_evaluation" \
  --improve-command ".venv/bin/python scripts/improve_candidate.py"
```

Each pass is one bounded experiment: improve one thing, evaluate it paired
against the king, read the result, then the next pass starts from what was
learned (the agent sees the updated scores and experiment ledger). The chain
stops early when a pass fails, when the agent stops requesting evaluation,
or when a `policy.toml` cap declines the next evaluation — a declined action
queues for approval instead of erroring, so a budget cap pauses the chain
rather than killing the loop. Budget for it: five passes means up to five GPU
evaluations per round, so set the `policy.toml` per-24h run/hour caps to what
you are willing to spend. In human mode the same flag only sets an upper
bound — each evaluation still waits for `--approve`, one pass at a time.

The example paths above now all exist. The GPU command is operational and
spends money only after the configured approval/allowlist gate. The `ops/`
paths refuse with exit status 2 until an operator deliberately implements
them; do not allowlist those wallet actions while they remain stubs.

The hook receives `CASCADE_MINER_EVENT`, `CASCADE_MINER_EVENT_TYPE`,
`CASCADE_MINER_ROUND_ID`, `CASCADE_MINER_STATE`, and `CASCADE_CHAIN_TOML` in its
environment. State and append-only event history live under `runs/`.
When `--hotkey` (or `CASCADE_MINER_HOTKEY`) is set, round events include that
miner's participation and heat rank/status in addition to the signed verdict.

The improvement hook may request four privileged actions:
`gpu_evaluation`, `create_hotkey`, `register_hotkey`, and `submit_candidate`.
It cannot provide an arbitrary command. The controller maps each action to an
operator-owned command configured on startup.

The bundled `scripts/improve_candidate.py` hook is agent-neutral. Set
`CASCADE_AGENT` to `claude`, `codex`, `hermes`, or `auto` (the default) to use
the first installed supported CLI. Each backend receives the same controller
event and operating brief.

```bash
# Claude Code
CASCADE_AGENT=claude .venv/bin/python scripts/improve_candidate.py

# Codex
CASCADE_AGENT=codex .venv/bin/python scripts/improve_candidate.py

# Hermes Agent
CASCADE_AGENT=hermes .venv/bin/python scripts/improve_candidate.py
```

Other agents work through `CASCADE_AGENT=custom` and
`CASCADE_AGENT_COMMAND`. Include `{prompt}` in the command template or omit it
to receive the prompt on standard input. The command must be non-interactive
and return a nonzero exit status on failure.

When the controller runs *inside* an agent session there is no CLI for the
agent to invoke itself with. `CASCADE_AGENT=hermes-native` swaps the subprocess
for a file handshake: the hook publishes the prompt to
`runs/improve-request.json` and waits for `runs/improve-response.json`.

```bash
CASCADE_AGENT=hermes-native .venv/bin/python scripts/improve_candidate.py &
python skills/cascade-miner/scripts/improve-request show
python skills/cascade-miner/scripts/improve-request respond --status completed \
    --detail "one line on what changed"
```

`auto` never picks it — see `docs/SETUP.md` for the timeouts and statuses.

Every backend may update `generators/candidate` and the experiment log, and may
request a named privileged action. It cannot launch that action itself, read
wallet secrets, commit, or push.

### Privileged-action approval

In `human` mode, the controller creates a durable request in
`runs/approvals.json` for GPU evaluation, hotkey creation, subnet registration,
or submission. An agent explains the need through `runs/agent-request.json`,
but cannot approve or execute it.

```bash
# View pending requests
.venv/bin/python -m miner.controller --list-approvals

# Approve or reject one request
.venv/bin/python -m miner.controller --approve <request-id>
.venv/bin/python -m miner.controller --reject <request-id>
```

Configure `CASCADE_APPROVED_EVAL_COMMAND` with the GPU evaluation entry point.
Configure `CASCADE_CREATE_HOTKEY_COMMAND`, `CASCADE_REGISTER_HOTKEY_COMMAND`,
and `CASCADE_SUBMIT_COMMAND` with non-interactive, operator-owned wrappers for
the wallet/chain actions. Never put passwords or mnemonics in command strings;
wrappers should obtain secrets from protected local storage. The running
controller executes these commands only after approval. Set
`CASCADE_NOTIFY_COMMAND` to a Telegram, Slack, email, Hermes gateway, or other
notifier command to surface `approval_required` and `approval_result` events to
a person. Without a notifier, requests remain visible in controller output and
the approvals file.

In `autonomous` mode, only names in `CASCADE_AUTONOMOUS_ACTIONS` run without an
approval. The default is `gpu_evaluation` only. To permit the full lifecycle,
the operator must deliberately set:

```bash
CASCADE_AUTONOMOUS_ACTIONS=gpu_evaluation,create_hotkey,register_hotkey,submit_candidate
```

Action failures stop the loop, and the per-round iteration cap prevents an
unbounded cycle. Git commits and pushes are never autonomous.

## Agent-native interfaces

The improvement hook covers agents this repo spawns. For the other direction —
an agent the operator runs (Claude Code, Codex, anything speaking MCP) — the
harness exposes a typed tool surface:

```bash
claude mcp add cascade-miner -- .venv/bin/python -m miner.mcp_server
```

The server is stdio JSON-RPC with no extra dependencies. Its tools are reads
(`miner_status`, `get_state`, `get_latest_receipt`, `list_heat_entrants`,
`list_approvals`, `get_policy`, `get_brief`), cheap local checks
(`run_quick_verify`), free public-artefact fetches (`fetch_generator` — the
king, a past heat entrant, a dethroned king, into a guarded `generators/`
directory), the structured experiment ledger (`log_experiment`,
`list_experiments`), the issue tracker (`report_issue`, `list_issues` —
bugs and feature requests about the harness, appended to `runs/issues.jsonl`
and shown to the operator in `miner_status`), the hermes-native handshake
(`get_improve_request`, `respond_improve_request`), and `request_action` —
which queues a durable
approval request and never executes anything. The privilege boundary is
identical to every other agent path: no wallet secrets, no spending, no
chain writes.

Round receipts are stored with their full participant and heat-entrant
lists, not just this miner's entry — generators from completed (revealed)
rounds are public, so past entrants are study material: list them, fetch
them, diff them against your candidate. The current heat's live rivals stay
timelocked until reveal; that is the subnet's design.

`python -m miner.status [--json]` prints the same one-look summary for humans
and scripts: round, king ref, eval pool, candidate digest, pending approvals,
open issues, and the experiment ledger tail.

`python -m miner.status --doctor [--json]` runs the onboarding checklist
instead: every stage between a fresh clone and a submittable miner —
environment, credentials, cascade checkout, Lium, SSH key, eval pool,
controller state, candidate, wallet — each with the exact command that fixes
it. It is read-only and offline, exits 1 until the host is ready, and `next:`
always names the single command to run first.

`python -m miner.experiments log|list` maintains `runs/experiments.jsonl`, the
structured twin of `notes/EXPERIMENTS.md`: hypothesis, the one change per arm,
seeds, snapshots, status, and the noise floor each verdict was read against.

Machine-facing contracts live under `schemas/`: the agent-request file, the
approval-queue entry, and the experiment ledger entry.

### Autonomy policy

`policy.toml` (copy `policy.example.toml`) bounds what autonomous mode may do
per trailing 24-hour window — runs per day and estimated hours per run and per
day, per action. When present (or named via `--policy-file` /
`CASCADE_POLICY_FILE`) it replaces `--autonomous-actions` as the autonomy
authority. Anything the policy declines degrades to an ordinary pending
approval rather than an error, so a cap pauses spending instead of dropping
the request. Human mode is unaffected; wallet/chain actions should stay
non-autonomous until the `ops/` wrappers are deliberately implemented.

## Development

This miner harness was developed with Claude Code.
