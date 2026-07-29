# Evaluation method

This note records reusable measurement method, not experiment results. Keep
candidate hypotheses, observations, and scores in the experiment log; do not
turn one run into a general rule.

## Pair the verdict by construction

The throne verdict is a paired statistic. Train and score the king and
challenger on exactly the same windows: the same seed must drive both
`RoundSeeds` and day-of-week/window selection, as it does in a real round. A
comparison with an older, independently sampled run is not a noisier paired
test; it is a different test.

## Preserve components, not only aggregates

Save every window's `qloss`, `abs_target`, `mase`, `source`, and `series_id`.
The decision is computed by a paired cluster bootstrap over window-level
components, clustered by source. A geomean cannot be decomposed back into
those inputs, so aggregate-only output cannot produce the real verdict or be
re-scored after a metric change.

## Optimise the lower bound, not just the mean

The lower confidence bound depends on the spread across resampled source
clusters. Reducing losses on weak sources will often move the lower bound more
than increasing gains on sources the candidate already wins. A higher geomean
with inconsistent source-level effects can therefore lose to a smaller but
more even improvement.

## Train a fresh control in the same batch

Always train the reigning king alongside the challenger with the same Cascade
revision, pool revision, environment, seeds, and window selection. Training
noise is heavy-tailed and a byte-identical rerun can move by several percent.
A delta smaller than that same-batch noise band is not a finding; an old score
is not an adequate control merely because the generator is unchanged.

## Keep metric versions comparable

Never compare figures computed by different metric versions. When scoring
changes, recompute both arms from their stored per-window components. Do not
trust an aggregate produced by a remote job whose pinned scoring revision is
unknown.

## Diagnose by source before concluding

Inspect paired effects per source before interpreting an aggregate. A null
aggregate can hide large gains on targeted sources offset by losses elsewhere.
That distinction separates a mechanism that works but competes for fixed model
capacity from one that has no measurable effect.
