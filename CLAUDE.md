# CLAUDE.md — cascade-miner

Operating notes for AI-assisted sessions.

## The game

We submit a purely-algorithmic synthetic time-series generator to the **cascade**
subnet (netuid 91, finney). The owner trains a fixed forecaster on our corpus and
scores it on a private, rotating held-out set. Model, seeds, init and compute are
identical between king and challenger — **the generator is the only variable**.
Only the single best challenger from each heat reaches the duel, and it must
clear a confidence-bound margin to take the throne.

Warm-start is armed (2026-08-05): the run starts from a promoted reign
checkpoint, rotating across a member set, not always from random init. The duel
stays fair — both sides share the round's init — but "what does this corpus add
to an already-trained model" is now the question. See `notes/CONTRACT.md`.

Read `notes/CONTRACT.md` (submission rules), `notes/BUDGET.md` (how the training
budget is derived), `notes/METHOD.md` (evaluation method), and
`notes/EXPERIMENTS.md` (what has been tried) before starting.

`notes/UPSTREAM.md` + `notes/upstream-state.json` are **generated** from the
reference clone by `scripts/upstream_state.py` and refreshed by the
`upstream-sync` workflow — never hand-edit them, and when they disagree with
prose, they are right.

**Never modify `/root/cascade`** — read-only reference clone, installed as a
library. Refresh with `git pull` + reinstall; never edit.

## Rules

1. **Submissions are one-shot per hotkey.** Entering a heat burns the hotkey and
   costs a re-registration. Never commit on-chain without explicit human
   approval — `submit.commit()` requires `confirm=True`.
2. **Know your noise floor before believing any result.** Measure it: run the
   same generator twice at the same seed and see how far apart the scores land.
   Treat anything inside that band as noise and say so, rather than reporting it
   as a finding.
3. **Score with the metric that is live now.** The subnet's scoring code
   changes. Recompute from saved per-window components rather than trusting
   numbers a remote job wrote with whatever version it had pinned. Never compare
   figures produced under different metric versions.
4. **Pull results continuously, starting when the run starts.** Rented pods
   expire; anything not already copied locally is gone.
5. **Use absolute paths in remote commands.** SSH sessions don't start in the
   project directory, and relative interpreter paths fail silently.
6. **Verify the environment before launching work.** Assert dependencies
   actually import on the remote host, and check the pod's remaining lifetime
   covers the run.
7. **Rent exactly the hardware you use.** Pin the GPU count.

Automated agents must request paid evaluation through
`runs/agent-request.json`; the controller turns that into a durable approval.
In human mode, only a human may approve it. In autonomous mode, only the
preconfigured evaluation command may act on it. Agents never rent pods directly.

Hotkey creation, subnet registration, and candidate submission use the same
named-action mechanism. Agents may request them but must never handle wallet
secrets or invoke wallet/chain commands directly.

## Method

`notes/METHOD.md` is the canonical, result-free evaluation methodology. In
particular, a verdict requires paired window-level components from a freshly
trained king and challenger; an aggregate from an older run is not a control.

- **Change one thing per arm**, with a freshly-trained control in the *same*
  batch — not a comparison against an older run.
- **Dose changes large enough to exceed the noise floor.** Small mixture tweaks
  are unmeasurable; screen at a size where a real effect would show.
  `python -m miner.screen <candidate> --king generators/king-control` does this
  for free: it expresses every corpus difference in units of the generator's own
  seed noise, names the king coverage a change trades away, and checks with
  `--claim` that the corpus carries the mechanism you believe you added. Its
  `measurable` verdict licenses a paid eval and is never evidence of a win.
- **Prefer replacing weak components over adding new ones.** The model has fixed
  capacity, so added coverage tends to displace existing coverage rather than
  accumulate. Test subtraction as seriously as addition.
- **Diagnose per-source, not just in aggregate.** An aggregate null can hide
  large gains on targeted sources offset by losses elsewhere — that distinction
  tells you whether a mechanism works.
- **Understand what moves the decision statistic.** A lower confidence bound on
  relative improvement is driven by the *spread* across resampled clusters, so
  reducing where you lose can matter more than increasing where you win.

## Deploy

Generators become public after submission, so do not include private reasoning
or sensitive information in their source files.

Upload to a fresh, non-guessable repo name each time so content stays as hidden
as the timelocked pointer. After committing, confirm the artefact is
**anonymously pullable** — the trainer fetches without credentials, and a
permissions failure looks identical to success on-chain.

Mind the real deadline: the reveal block minus a safety buffer, not the epoch
boundary.

## Reporting

State the seed count and noise floor with every number. Say plainly when a
result is null. If an experiment is lost, say so and what it cost. Distinguish
**measured** from **inferred**. Never present a ranking that can't survive the
noise floor.
