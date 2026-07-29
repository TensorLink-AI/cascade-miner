# Training budget

Confirmed against `/root/cascade` source and against observed run lengths
(heat ≈ 50k steps, final ≈ 150k steps).

## The numbers

```
tokens/step = batch_size x context_length = 64 x 4096 = 262,144

HEAT   1.0h x 3,700,000 tok/s = 13.32e9 points ->  ~50,800 steps   (matches observed ~50k)
FINAL  3.0h x 3,700,000 tok/s = 39.96e9 points -> ~152,400 steps   (matches observed ~150k)

warmup_fraction = 0.05  ->  heat warmup  ~2,540 steps
                            final warmup ~7,620 steps
```

`train_tokens = target_train_hours x ref_throughput_tokens_per_s`
(`cascade/shared/config.py`), and the heat uses the same formula with
`[round] heat_train_hours`.

## `max_total_points` is not the training budget

`[generator] max_total_points = 2e9` is enforced **only** in
`cascade/trainer/corpus.py` (the `build_round_corpus` / `cache_reuse` path) and
as an rlimit fsize bound in `sandbox.py`. The live subnet runs
`corpus_mode = "stream_cpu"`, where `_FreshSeriesStream` streams until
`token_budget = contract.train_tokens` and never consults `max_total_points`.

So a generator that anchors a schedule to `2e9` points is using a horizon
**6.7x smaller than the heat budget and 20x smaller than the final budget.**

Derive schedules from the live contract's token budget. If a schedule must use
absolute point counts, verify where it completes in both the heat and final.
The heat should evaluate the same steady-state distribution intended for the
final rather than stopping partway through a ramp.

## Revisit when

`[round] heat_train_hours`, `[training] target_train_hours`,
`ref_throughput_tokens_per_s`, `batch_size`, or `context_length` change — every
number above is derived from those five.
