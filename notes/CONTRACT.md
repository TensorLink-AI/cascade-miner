# The cascade contract — distilled

Everything here is read from `/root/cascade` (read-only reference clone,
`TensorLink-AI/cascade`). Source of truth is that repo; this file exists so we
don't re-read 1800 lines of docs every session. Re-check against
`chain.toml` / `docs/INTERFACE.md` when a round's behaviour surprises us.

**Numbers live next door.** `notes/UPSTREAM.md` (and its machine form,
`notes/upstream-state.json`) is generated straight from the reference clone
and refreshed automatically; this file is the *interpretation*, and
`tests/test_stale_references.py` fails when a sentence here contradicts a value
there. When the two disagree, the snapshot is right.

## What we are actually competing on

We submit a **data generator**: purely algorithmic code that emits synthetic
univariate time series. The subnet owner trains a **fixed Toto2-4M backbone**
on our corpus, then scores that forecaster on a **private, rotating held-out
set** (WQL + MASE; since DEC-CA-0009 the CRPS half is a **geomean of per-window
WQL**, not a pooled MWSQL, and MASE is a geometric mean — pooling weighted each
window by its magnitude and let three huge-scale windows dominate the round).

The model, optimiser, seeds, initialisation, and compute budget are
byte-identical between king and challenger. **The only variable is our
generator code.** So we are optimising one thing: *does data drawn from our
prior train a better general-purpose forecaster than the king's data?*

Consequences that should drive every design decision:

- **Since 2026-08-05 the run does NOT always start from noise.** Warm-start is
  armed (`cascade_enabled = true`): after `cascade_reign_rounds = 5`
  undethroned rounds the trainer promotes up to `cascade_top_k = 3` reign
  checkpoints into a member set, and subsequent rounds *rotate* across those
  members as the training init. Both duellists get the **same** init in a
  round, so the duel is as fair as ever — but our corpus is increasingly being
  asked to *improve a partially-trained model*, not to teach one from scratch.
  What a warm-started model still lacks is not what a random-init one lacks,
  so coverage that only re-teaches the basics is worth less than it used to be.
  See "Warm start" below.
- Regime diversity (trend, multi-seasonality, regime shifts, noise structure,
  realistic and varied scales) remains the core of the prior.
- The eval pool is **private and rotates every round** — distribution-matching
  a public benchmark cannot work. Robust general priors win.
- Corpus size can't be gamed: budget is `train_tokens`, not epochs.
- Generator **throughput is a compute multiplier** (DEC-CA-0001: "wall is the
  law"). A slow generator starves the trainer and eats its own token budget.
  `ref_throughput_tokens_per_s = 3_700_000`. Fast generate path is a feature,
  not an optimisation.

## Hard requirements (rejection = wasted round)

| rule | value |
|---|---|
| entrypoint | `class Generator(DataGenerator)` in `generator.py` |
| required files | `generator.py`, `config.json`, `requirements.txt` |
| determinism | same `seed` ⇒ **byte-identical** corpus, across *processes* |
| series length | `L ∈ [64, 4096]` |
| channels | `max_channels = 1` (1-D `(L,)`; `(C,L)` schema reserved) |
| count | `generate(n)` yields **exactly** `n` |
| finiteness | no NaN/inf; `max_abs_value = 0.0` ⇒ float32 ceiling |
| corpus cap | `max_total_points = 2e9`; `corpus_n_series = 16384` per run |
| generate budget | `max_generate_seconds = 1800`, `max_memory_mb = 4096` |
| repo size | `max_repo_mb = 128`, **code only** |
| deps | hash-locked `pkg==ver --hash=sha256:<64hex>`, ≤ 16 packages |

**No shipped weights of any kind** — not `*.pt/*.pkl` (arbitrary code) and not
`*.safetensors/*.npy/*.npz/*.onnx` (would let us distil a pretrained model).
`torch`/`gpytorch` are allowed *as compute libraries* only.

Allowed deps: `numpy, scipy, pandas, statsmodels, numba, torch,
scikit-learn, gpytorch, networkx`.

Blocked imports (AST-scanned): `socket, urllib*, requests, httpx, http.*,
subprocess, os.system, ctypes, cffi, pickle, marshal, shelve, multiprocessing,
asyncio.subprocess, cascade.trainer, cascade.validator, cascade.shared.chain`.

### Determinism gotchas that have actually bitten people

- `hash()` is salted per process by `PYTHONHASHSEED` — use `zlib.crc32`.
- Seed numpy **and** `random` **and** torch (`torch.manual_seed` +
  `torch.use_deterministic_algorithms(True)`, CPU) if used.
- No wall-clock, no `os.urandom`, no un-seeded global RNG.
- `cascade verify` runs the generator twice and compares digests.

## Round mechanics

- One round ≈ **12h** (`epoch_blocks = 3600`; was 7200 ≈ 24h). The switch is
  **block-gated**: `epoch_blocks_prev = 7200` applies strictly before
  `epoch_activation_block = 8726400`. Resolve the length through
  `cascade.shared.config.effective_epoch_blocks(round_cfg, block)` — never the
  raw key. Eligibility needs the **reveal** (not the commit) strictly before
  the epoch boundary.
- Two stages: **heat** (every challenger trained `heat_train_hours = 1.0` at
  `screen_size = toto2-4m`, top `finalists = 1` promoted) → **final** (king +
  finalist trained to `target_train_hours = 3.0` on `throne_sizes`).
- The heat now screens on **`heat_n_windows = 2000`** — the full `[eval]`
  window count, raised from 256 on 2026-08-07 because the short screen was
  noisy near the top. Our local screen should match it (`miner/evaluate.py`
  reads the key, so it follows a reinstall automatically), at ~8× the CPU
  cost per checkpoint: any heat noise floor we measured under 256 windows is
  no longer the live one.
- `finalists = 1` today, but DEC-CA-0012 has landed **inert** upstream: when
  the screen cannot separate its top entrants, the tied set advances and
  *duels as a cohort* under an `α/k` per-challenger correction. The code is in
  `cascade/eval/heat.py` (`tied_set`, `lcb_vs`); only the config still says 1.
  Treat "only the single best challenger reaches the duel" as a value to check,
  not a law.
- Verdict: paired-bootstrap LCB of geomean(WQL, MASE) **pooled across sizes**,
  finalist vs king; per DEC-CA-0009 the WQL half is a per-window geomean (zero
  `sum|y|` windows masked from that half, not floored). `dethrone_cp = 1` —
  one decisive round takes the throne.
- `one_submission_per_hotkey = true` — **a hotkey that enters a heat is burned
  and must re-register.** Submissions are expensive. Score locally first.
- Dedup is `enforce` on **exact identity only** (same tree / normalised tokens /
  tokens-modulo-renames, and identical probe output). Forking + genuinely
  modifying is the intended game; byte-identical re-uploads are dropped and
  still burn the submission.
- `commit_floor_block = 8622922` — reveals before this never compete.
- Mainnet netuid **91** (finney); testnet netuid **259** (`chain.testnet.toml`).
- **Heat standings publish when the heat settles** (DEC-CA-0011), hours before
  the validator receipt: `status/heat.json` (live pointer), `heats/round-<id>.json`,
  `heats/index.json` in the public logs store — including raw per-entrant
  CRPS/MASE, and a stated reason for rounds rejected at a gate. Unsigned and
  presentational; the signed manifest/receipt stays the record. Use these to
  see where an entry ranked without waiting a round.
- Every training run stamps a `host` record (lane geometry, capability, fixed
  calibration bench) into the public training log (DEC-CA-0010). Telemetry
  only — nothing consumes it and scores are NOT host-normalized, but it is the
  data for judging how much host variance sits in the noise floor.
- `cascade duel` prints the full settled-round verdict from the **signed**
  public receipts — king/challenger geomeans, the LCB against the required
  margin, window win-rate and per-domain breakdown, and whether the validators
  agreed. `--history` lists every settled round. Read-only, no wallet. This is
  the honest post-mortem for a lost round: the per-domain table says *where*
  the duel was lost, which is what `notes/METHOD.md` asks us to diagnose.

## Warm start (armed 2026-08-05 — read this before designing a corpus)

`cascade_enabled = true` on mainnet. The mechanics we compete under:

- **Promotion.** After `cascade_reign_rounds = 5` consecutive undethroned
  rounds the reign is "ripe". The **trainer** then selects up to
  `cascade_top_k = 3` reign checkpoints and publishes a signed
  `PromotionRecord` (`promotions/gen-<n>.json`); validators *verify* that
  declaration against an envelope (provenance, a quality floor of
  `cascade_quality_epsilon = 0.05`, ripeness, the k cap) rather than
  re-deriving it (DEC-CA-0013). Members are chosen by **measured error
  decorrelation** inside the quality frontier (DEC-CA-0015) — lineages that
  fail on *different* windows.
- **Rotation.** Later rounds rotate their warm-start init across the member
  set, so consecutive rounds can start from different checkpoints. The round's
  init is published in the heat standings, so we can see which one we drew.
- **The king persists** through promotion (DEC-CA-0004); the throne is never
  vacated, and promotion pays the checkpoint's owner nothing.
- **Fairness is unchanged**: king and challenger share the round's init, so
  there is still no incumbency advantage and the generator is still the only
  variable.
- **What it changes for us**: the marginal value of a corpus is now "what does
  this add to an already-trained forecaster", and it varies with which member
  a round drew. Two implications for `notes/METHOD.md` discipline — a control
  must be trained in the *same* round-init regime as its arm, and cross-round
  comparisons now carry an extra source of variance we did not have before.
- DEC-CA-0014 (staged, **not built yet**) keeps a from-scratch signal alive:
  a periodic shadow scratch control, then a reseed valve admitting scratch
  checkpoints through the same quality floor. Watch for stage 1 landing — it
  is the number that tells us whether the lineage is still compounding.

## Submission protection

`cascade deploy` defaults to a **timed reveal** (decrypts ~5 min before the
boundary) so a fresh submission can't be copied into its own round. Use
`--hub-namespace` (fresh non-guessable repo per deploy) rather than a fixed
`--hub-repo`. The Hippius project must be **public** or the trainer gets 401 and
we're skipped as `generator_artifact_unreachable`.

Commit **from an environment pinning `bittensor==10.5.0`** — other SDK lines
write reveals the subnet's decoder can't read, and you're silently skipped.

## The loop the docs recommend

```
cascade verify <gen>                       # every trainer-side check
cascade score  <gen> --pool-dir <held-out> # train at heat budget, score offline
cascade fetch  king --out ./king           # pull the current best to beat
cascade deploy <gen> --hub-namespace ...   # only once it beats the king locally
```

`score` is directional only — our pool is not the validator's private pool, and
overfitting a fixed local pool is exactly what the rotating eval punishes.

## Keeping this file in sync

The competition moves under us; this file goes stale, not the subnet. It went
stale once already — warm-start armed on 2026-08-05 and these notes still
said "trains from random init" ten days later — so the knowledge is now split
in two, and only one half needs a human.

**Facts sync themselves.** `scripts/upstream_state.py` reads every miner-facing
`chain.toml` key and every `DEC-CA-` node out of the reference clone into
`notes/upstream-state.json` + `notes/UPSTREAM.md`. The `upstream-sync`
workflow reruns it every 6h and, when a value moves, pushes the regenerated
snapshot to `automation/upstream-sync` and opens/refreshes one PR.

**Prose does not.** This file is pinned to the snapshot by
`tests/test_stale_references.py`: it asserts, on the values rather than the
wording, that what we claim about the init regime, the finalist rule, the heat
screen, the margin and the netuid matches upstream. So a contradicting fact
change turns the sync PR red and the failing assertion names the stale
sentence.

What still needs a human (or an agent) per sync:

1. Fold the flagged changes into this file, then restamp the line below — it
   means *prose reviewed at this revision*, which is why the bot never moves
   it.
2. Run `bash scripts/sync.sh` **on the operator host**: merging a PR does not
   reinstall the library, and scoring imports come from the installed package,
   so an un-reinstalled venv keeps scoring with the old metric.
3. Re-score saved per-window components with the now-live metric (Rule 3 in
   `CLAUDE.md`) before comparing any number to a pre-sync one.
4. Add a pin to `test_stale_references.py` whenever a sync catches something
   this file said wrongly — that is how the alarm gets sharper.

Last synced: 2026-08-16, cascade `90456ac`.
