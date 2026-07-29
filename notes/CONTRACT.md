# The cascade contract — distilled

Everything here is read from `/root/cascade` (read-only reference clone,
`TensorLink-AI/cascade`). Source of truth is that repo; this file exists so we
don't re-read 1800 lines of docs every session. Re-check against
`chain.toml` / `docs/INTERFACE.md` when a round's behaviour surprises us.

## What we are actually competing on

We submit a **data generator**: purely algorithmic code that emits synthetic
univariate time series. The subnet owner trains a **fixed Toto2-4M backbone
from random init** on our corpus, then scores that forecaster on a **private,
rotating held-out set** (CRPS/MWSQL + MASE, pooled geomean).

The model, optimiser, seeds, and compute budget are byte-identical between king
and challenger. **The only variable is our generator code.** So we are
optimising one thing: *does data drawn from our prior train a better
general-purpose forecaster than the king's data?*

Consequences that should drive every design decision:

- The model starts from **noise**, so it learns forecasting *only* from us.
  Regime diversity (trend, multi-seasonality, regime shifts, noise structure,
  realistic and varied scales) is the whole game.
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

- One round ≈ 24h (`epoch_blocks = 7200`). Eligibility needs the **reveal**
  (not the commit) strictly before the epoch boundary.
- Two stages: **heat** (every challenger trained ~1h at screen size, top
  `finalists = 1` promoted) → **final** (king + finalist trained to
  `target_train_hours = 3.0` at every size: 4M primary + 22M).
- Verdict: paired-bootstrap LCB of geomean(CRPS, MASE) **pooled across sizes**,
  finalist vs king. `dethrone_cp = 1` — one decisive round takes the throne.
- `one_submission_per_hotkey = true` — **a hotkey that enters a heat is burned
  and must re-register.** Submissions are expensive. Score locally first.
- Dedup is `enforce` on **exact identity only** (same tree / normalised tokens /
  tokens-modulo-renames, and identical probe output). Forking + genuinely
  modifying is the intended game; byte-identical re-uploads are dropped and
  still burn the submission.
- `commit_floor_block = 8622922` — reveals before this never compete.
- Mainnet netuid **91** (finney); testnet netuid **259** (`chain.testnet.toml`).

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
