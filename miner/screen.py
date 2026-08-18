"""Pre-GPU screen: is this corpus different enough from the king's to be worth measuring?

The paid loop is train-both-arms-and-bootstrap. This is the free step in front of
it. It generates a small sample of the candidate corpus and the king's, reduces
each series to a fixed feature vector, and answers three questions that do not
need a GPU:

1. **Would the trainer reject or starve on this?** Length bounds, finiteness,
   duplicate fraction, and projected generate time, read from
   `notes/upstream-state.json` so the gates follow upstream rather than a
   hardcoded copy.
2. **Is the dose big enough to measure?** Every feature distance between
   candidate and king is expressed in units of the *same generator's* seed-to-seed
   distance. A difference smaller than a generator's own seed noise cannot
   survive a paired eval, and the eval costs money to learn that.
3. **What is the bet, stated in numbers?** Which feature families moved, how much
   king coverage was displaced (fixed model capacity means new coverage is
   usually traded, not added), and whether the corpus actually contains the
   property the author claims it does.

**What this cannot do.** It cannot rank two corpora, predict a duel, or replace
`miner.evaluate` + `miner.analyze`. Feature distance is not score improvement:
a corpus can differ wildly and train a worse forecaster. The screen is a
*falsifier* — it tells you when not to spend, and what you are actually
betting on when you do. Every verdict it can return is one of:

    blocked      a contract gate fails — fix before spending anything
    undosed      indistinguishable from the king at this sample size — a paired
                 eval would return a null, so raise the dose instead of paying
    measurable   the corpora differ beyond seed noise — the GPU eval can now
                 resolve *something*; it still says nothing about the sign

    python -m miner.screen generators/candidate --king generators/king-control \\
        --prior-king generators/prior-king --claim seasonality+ --json runs/screen.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "notes/upstream-state.json"

# Contract values used by the gates. The snapshot is authoritative when present;
# these are the fallback for a checkout without one, and are labelled as such in
# the report so no one mistakes a default for a synced fact.
GATE_DEFAULTS = {
    "generator.min_length": 64,
    "generator.max_length": 4096,
    "generator.max_abs_value": 0.0,
    "generator.max_dup_fraction": 0.05,
    "generator.reject_constant": False,
    "generator.corpus_n_series": 16384,
    "generator.max_generate_seconds": 1800,
    "training.ref_throughput_tokens_per_s": 3_700_000,
    "training.context_length": 4096,
}
FLOAT32_MAX = 3.4028234663852886e38

# Per-series features. Each is unitless or log-scaled so a single distance
# convention applies across all of them. `sign` is the direction that means
# "more of this family's property"; 0 means the feature has no natural direction
# and is used for distance only, never for a signed claim check.
FEATURES: tuple[tuple[str, str, int], ...] = (
    ("log_length", "length", 0),
    ("log_scale", "scale", 0),
    ("trend_strength", "trend", +1),
    ("diff_var_ratio", "trend", -1),
    ("seasonal_strength", "seasonality", +1),
    ("log_period", "seasonality", 0),
    ("season_skill", "seasonality", -1),
    ("spectral_entropy", "noise", +1),
    ("acf1", "noise", -1),
    ("ar_skill", "noise", +1),
    ("jump_rate", "regime", +1),
    ("level_shift", "regime", +1),
    ("skew", "tails", 0),
    ("tail_ratio", "tails", +1),
    ("hetero", "tails", +1),
    ("zero_frac", "intermittency", +1),
)
FEATURE_NAMES = tuple(name for name, _, _ in FEATURES)
FAMILIES = sorted({family for _, family, _ in FEATURES})

# A feature has "moved" when its candidate-vs-king distance is this many times
# the generators' own seed-to-seed distance. The band is a two-sample estimate,
# so a plain >1 crossing is within its own error; 2 is the working margin.
DOSE_THRESHOLD = 2.0
# Floor on the noise band, in the same standardised units. Without it, a feature
# that happens to be seed-invariant divides by ~0 and every trivial difference
# reads as an enormous dose.
BAND_FLOOR = 0.05
# Feature-scale floor, for features that are near-constant across both corpora.
SCALE_FLOOR = 1e-3
EPS = 1e-12


# -- contract gates ---------------------------------------------------------


def load_gates(snapshot: Path = SNAPSHOT) -> tuple[dict[str, Any], str]:
    """Gate values plus a one-line provenance string for the report."""
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        keys = payload.get("keys", {})
        source = payload.get("generated_from", {})
        gates = {name: keys.get(name, default) for name, default in GATE_DEFAULTS.items()}
        return gates, (f"notes/upstream-state.json @ cascade "
                       f"{source.get('revision', '?')} ({source.get('revision_date', '?')})")
    except (OSError, ValueError):
        return dict(GATE_DEFAULTS), "built-in defaults — no readable upstream snapshot"


# -- per-series features ----------------------------------------------------


def _robust_scale(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.349
    if scale <= 0:
        scale = float(np.std(x))
    return max(float(scale), EPS)


def _spectrum(resid: np.ndarray) -> tuple[float, float, float]:
    """(seasonal_strength, log2 dominant period, normalised spectral entropy)."""
    n = resid.size
    power = np.abs(np.fft.rfft(resid - resid.mean())) ** 2
    power = power[1:]                                    # drop DC
    total = float(power.sum())
    if n < 8 or power.size < 2 or total <= EPS:
        return 0.0, 0.0, 1.0
    density = power / total
    entropy = float(-(density * np.log(density + EPS)).sum() / np.log(density.size))
    peak = int(np.argmax(density))
    lo, hi = max(peak - 1, 0), min(peak + 2, density.size)
    strength = float(density[lo:hi].sum())
    period = n / (peak + 1)
    return strength, float(np.log2(min(max(period, 2.0), float(n)))), min(max(entropy, 0.0), 1.0)


def _ar_skill(y: np.ndarray, order: int = 4) -> float:
    """log10(AR(order) one-step MAE / last-value MAE) on a held-out tail.

    Negative means linear structure the model can exploit beyond a random walk;
    ~0 means the series is a random walk or noise. This is the closest cheap
    proxy for "is there anything here to learn".
    """
    n = y.size
    if n < 8 * order:
        return 0.0
    split = int(n * 0.8)
    lags = np.stack([y[order - k - 1: n - k - 1] for k in range(order)], axis=1)
    target = y[order:]
    cut = max(split - order, order)
    if cut >= target.size - 2:
        return 0.0
    design = np.hstack([lags, np.ones((lags.shape[0], 1))])
    try:
        coef, *_ = np.linalg.lstsq(design[:cut], target[:cut], rcond=None)
    except np.linalg.LinAlgError:                        # pragma: no cover - defensive
        return 0.0
    predicted = design[cut:] @ coef
    mae_ar = float(np.mean(np.abs(target[cut:] - predicted)))
    mae_naive = float(np.mean(np.abs(target[cut:] - lags[cut:, 0])))
    return float(np.log10((mae_ar + EPS) / (mae_naive + EPS)))


def series_features(y: np.ndarray) -> np.ndarray:
    """Reduce one series to the fixed feature vector, in `FEATURE_NAMES` order."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    out: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    out["log_length"] = float(np.log2(max(n, 1)))
    if n < 4:
        return np.array([out[name] for name in FEATURE_NAMES], dtype=np.float64)

    scale = _robust_scale(y)
    out["log_scale"] = float(np.log10(scale))
    out["zero_frac"] = float(np.mean(y == 0.0))

    time_index = np.arange(n, dtype=np.float64)
    centred = time_index - time_index.mean()
    slope = float(centred @ (y - y.mean()) / max(float(centred @ centred), EPS))
    resid = y - (y.mean() + slope * centred)
    var_y = float(np.var(y))
    out["trend_strength"] = 0.0 if var_y <= EPS else float(
        np.clip(1.0 - np.var(resid) / var_y, 0.0, 1.0))

    diff = np.diff(y)
    out["diff_var_ratio"] = float(np.clip(
        np.log10((np.var(diff) + EPS) / (var_y + EPS)), -6.0, 6.0))

    strength, log_period, entropy = _spectrum(resid)
    out["seasonal_strength"] = strength
    out["log_period"] = log_period
    out["spectral_entropy"] = entropy

    if float(np.var(resid)) > EPS and n > 2:
        centred_resid = resid - resid.mean()
        denom = float(centred_resid @ centred_resid)
        out["acf1"] = float(np.clip(
            (centred_resid[:-1] @ centred_resid[1:]) / max(denom, EPS), -1.0, 1.0))

    median_diff = float(np.median(diff))
    mad = float(np.median(np.abs(diff - median_diff)))
    if mad > EPS:
        out["jump_rate"] = float(np.mean(np.abs(diff - median_diff) > 6.0 * 1.4826 * mad))

    cuts = np.unique(np.clip((np.linspace(0.1, 0.9, 9) * n).astype(int), 1, n - 1))
    shifts = [abs(float(y[:c].mean() - y[c:].mean())) for c in cuts]
    out["level_shift"] = float(min(max(shifts) / scale, 20.0)) if shifts else 0.0

    q01, q25, q50, q75, q99 = np.percentile(y, [1, 25, 50, 75, 99])
    iqr = float(q75 - q25)
    if iqr > EPS:
        out["skew"] = float(np.clip((q75 + q25 - 2.0 * q50) / iqr, -1.0, 1.0))
        out["tail_ratio"] = float(np.clip(np.log10((q99 - q01 + EPS) / iqr), -2.0, 4.0))

    blocks = max(min(8, n // 16), 2)
    block_std = np.array([float(np.std(part)) for part in np.array_split(y, blocks)])
    mean_std = float(block_std.mean())
    if mean_std > EPS:
        out["hetero"] = float(np.clip(np.log10(block_std.std() / mean_std + 1e-3), -3.0, 3.0))

    season = int(round(2.0 ** log_period))
    if 2 <= season <= n // 3:
        mae_season = float(np.mean(np.abs(y[season:] - y[:-season])))
        mae_naive = float(np.mean(np.abs(diff)))
        out["season_skill"] = float(np.clip(
            np.log10((mae_season + EPS) / (mae_naive + EPS)), -4.0, 4.0))

    out["ar_skill"] = float(np.clip(_ar_skill(y), -4.0, 4.0))
    vector = np.array([out[name] for name in FEATURE_NAMES], dtype=np.float64)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# -- corpus profile ---------------------------------------------------------


@dataclass
class Profile:
    """A sampled corpus reduced to what the screen needs: features plus gate facts."""

    name: str
    seed: int
    matrix: np.ndarray                       # (n_series, len(FEATURES))
    n_series: int
    n_requested: int
    total_points: int
    generate_seconds: float
    lengths: np.ndarray
    max_abs: float
    dup_fraction: float
    constant_fraction: float
    nonfinite_series: int
    bad_dtype_series: int
    bad_shape_series: int

    def points_per_second(self) -> float:
        return self.total_points / max(self.generate_seconds, 1e-6)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name, "seed": self.seed, "n_series": self.n_series,
            "total_points": self.total_points,
            "median_length": int(np.median(self.lengths)) if self.lengths.size else 0,
            "generate_seconds": round(self.generate_seconds, 3),
            "points_per_second": round(self.points_per_second(), 1),
            "dup_fraction": round(self.dup_fraction, 4),
            "constant_fraction": round(self.constant_fraction, 4),
            "max_abs": self.max_abs,
        }


def profile_series(name: str, seed: int, series: Sequence[np.ndarray], *,
                   generate_seconds: float = 0.0, n_requested: int | None = None) -> Profile:
    """Build a Profile from series already in memory (the testable core)."""
    arrays = [np.asarray(s) for s in series]
    rows, lengths, digests = [], [], set()
    duplicates = constant = nonfinite = bad_dtype = bad_shape = 0
    max_abs = 0.0
    for item in arrays:
        if item.ndim != 1:
            bad_shape += 1
            item = item.ravel()
        if item.dtype != np.float64:
            bad_dtype += 1
        values = np.asarray(item, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.all():
            nonfinite += 1
            values = np.where(finite, values, 0.0)
        key = values.tobytes()
        if key in digests:
            duplicates += 1
        else:
            digests.add(key)
        if values.size and float(np.ptp(values)) == 0.0:
            constant += 1
        max_abs = max(max_abs, float(np.max(np.abs(values))) if values.size else 0.0)
        lengths.append(values.size)
        rows.append(series_features(values))
    count = len(arrays)
    matrix = np.vstack(rows) if rows else np.zeros((0, len(FEATURES)))
    return Profile(
        name=name, seed=seed, matrix=matrix, n_series=count,
        n_requested=count if n_requested is None else int(n_requested),
        total_points=int(sum(lengths)), generate_seconds=float(generate_seconds),
        lengths=np.array(lengths, dtype=np.int64),
        max_abs=max_abs,
        dup_fraction=duplicates / count if count else 0.0,
        constant_fraction=constant / count if count else 0.0,
        nonfinite_series=nonfinite, bad_dtype_series=bad_dtype, bad_shape_series=bad_shape,
    )


def profile_generator(repo: Path, seed: int, n_series: int, *, name: str | None = None) -> Profile:
    """Generate `n_series` series from a candidate directory and profile them.

    Generation is timed, so the same pass that measures distribution also
    measures throughput — the compute multiplier the contract cares about.
    """
    from scripts.quick_verify import load_generator_class

    try:
        generator_cls = load_generator_class(repo)
    except ModuleNotFoundError as error:
        # A generator subclasses cascade.interface.DataGenerator, so importing
        # one needs the installed library — the same requirement as the trainer.
        raise SystemExit(
            f"{repo}/generator.py needs {error.name!r}, which this interpreter "
            "cannot import; run the screen with .venv/bin/python (bash "
            "scripts/setup.sh installs the Cascade library into it)"
        ) from None
    generator = generator_cls(str(repo), seed=int(seed))
    start = time.perf_counter()
    produced: list[np.ndarray] = []
    for index, item in enumerate(generator.generate(n_series)):
        if index >= n_series:
            break
        produced.append(np.asarray(item))
    elapsed = time.perf_counter() - start
    return profile_series(name or repo.name, seed, produced,
                          generate_seconds=elapsed, n_requested=n_series)


# -- distances, bands, coverage ---------------------------------------------


_QUANTILES = np.linspace(0.005, 0.995, 199)


def _feature_scales(*matrices: np.ndarray) -> np.ndarray:
    """Per-feature scale from the pooled corpora, so distances are comparable."""
    usable = [m for m in matrices if m.size]
    if not usable:                    # a generator that yielded nothing; gates report it
        return np.full(len(FEATURES), SCALE_FLOOR)
    pooled = np.vstack(usable)
    q75, q25 = np.percentile(pooled, [75, 25], axis=0)
    return np.maximum((q75 - q25) / 1.349, SCALE_FLOOR)


def feature_distances(a: np.ndarray, b: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Per-feature 1-Wasserstein distance between two corpora, in scale units."""
    if not a.size or not b.size:
        return np.zeros(len(FEATURES))
    qa = np.quantile(a, _QUANTILES, axis=0)
    qb = np.quantile(b, _QUANTILES, axis=0)
    return np.mean(np.abs(qa - qb), axis=0) / scales


def _nn_distances(source: np.ndarray, target: np.ndarray, block: int = 128) -> np.ndarray:
    """For each row of `source`, the Euclidean distance to the nearest row of `target`."""
    if not source.size or not target.size:
        return np.zeros(source.shape[0])
    out = np.empty(source.shape[0])
    for start in range(0, source.shape[0], block):
        chunk = source[start:start + block]
        d2 = ((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk.shape[0]] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def coverage_report(candidate: np.ndarray, king: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    """How much of the king's corpus the candidate stops covering, and vice versa.

    Both directions are calibrated against the king's *own* internal spacing: the
    king is split in half and the nearest-neighbour distance between halves gives
    the radius at which "no neighbour" stops meaning "different sampling".
    """
    if candidate.shape[0] < 4 or king.shape[0] < 4:
        return {"note": "too few series to calibrate coverage"}
    cand = candidate / scales
    ref = king / scales
    half = ref.shape[0] // 2
    internal = _nn_distances(ref[:half], ref[half:])
    radius = float(np.percentile(internal, 95))
    cand_half = cand.shape[0] // 2
    cand_internal = _nn_distances(cand[:cand_half], cand[cand_half:])
    ref_to_cand = _nn_distances(ref, cand)
    cand_to_ref = _nn_distances(cand, ref)
    median_internal = float(np.median(internal))
    return {
        "radius": round(radius, 4),
        "coverage_loss": round(float(np.mean(ref_to_cand > radius)), 4),
        "novelty": round(float(np.mean(cand_to_ref > radius)), 4),
        "diversity_ratio": round(
            float(np.median(cand_internal) / max(median_internal, EPS)), 3),
        "median_nn_king": round(median_internal, 4),
        "median_nn_candidate": round(float(np.median(cand_internal)), 4),
    }


def history_axis(candidate: np.ndarray, king: np.ndarray, prior: np.ndarray,
                 scales: np.ndarray) -> dict[str, Any]:
    """Where the candidate sits relative to the prior-king → king transition.

    Two readings, both weak by construction. The *direction*: the only observed
    winning move is prior king → king, so a candidate moving the other way is
    betting against it. The *distance*: a candidate closer to the dethroned
    corpus than to the reigning one may be re-proposing a prior that already
    lost. One transition is one data point, and the round that produced it had
    its own noise — this is a prompt to justify a bet, never evidence for one.
    """
    to_king = float(np.mean(feature_distances(candidate, king, scales)))
    to_prior = float(np.mean(feature_distances(candidate, prior, scales)))
    king_to_prior = float(np.mean(feature_distances(king, prior, scales)))
    report = {
        "mean_distance_to_king": round(to_king, 3),
        "mean_distance_to_prior_king": round(to_prior, 3),
        "mean_king_to_prior_king": round(king_to_prior, 3),
        "closer_to_dethroned_king": bool(to_prior < to_king),
        "caveat": "n=1 transition, unmeasured noise — a prompt to justify, not evidence.",
    }
    axis = (np.median(king, axis=0) - np.median(prior, axis=0)) / scales
    move = (np.median(candidate, axis=0) - np.median(king, axis=0)) / scales
    axis_norm, move_norm = float(np.linalg.norm(axis)), float(np.linalg.norm(move))
    if axis_norm < EPS or move_norm < EPS:
        report["note"] = "prior king and king are indistinguishable on these features"
        return report
    cosine = float(axis @ move / (axis_norm * move_norm))
    report.update({
        "cosine_with_last_winning_move": round(cosine, 3),
        "axis_magnitude": round(axis_norm, 3),
        "move_magnitude": round(move_norm, 3),
        "reading": ("continues the last winning direction" if cosine > 0.3 else
                    "moves against the last winning direction" if cosine < -0.3 else
                    "orthogonal to the last winning direction"),
    })
    return report


# -- gates ------------------------------------------------------------------


@dataclass
class Flag:
    level: str            # "block" | "warn"
    name: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "name": self.name, "detail": self.detail}


def gate_flags(profile: Profile, gates: dict[str, Any]) -> list[Flag]:
    """Contract checks that decide whether spending anything is even sane."""
    flags: list[Flag] = []
    if profile.n_series != profile.n_requested:
        flags.append(Flag("block", "count", (
            f"asked for {profile.n_requested} series, generator yielded "
            f"{profile.n_series}; `generate(n)` must yield exactly n")))
    if profile.bad_shape_series:
        flags.append(Flag("block", "shape",
                          f"{profile.bad_shape_series} series are not 1-D"))
    if profile.bad_dtype_series:
        flags.append(Flag("block", "dtype",
                          f"{profile.bad_dtype_series} series are not float64"))
    if profile.nonfinite_series:
        flags.append(Flag("block", "finiteness",
                          f"{profile.nonfinite_series} series contain NaN or inf"))
    low, high = int(gates["generator.min_length"]), int(gates["generator.max_length"])
    if profile.lengths.size:
        out_of_range = int(np.sum((profile.lengths < low) | (profile.lengths > high)))
        if out_of_range:
            flags.append(Flag("block", "length", (
                f"{out_of_range}/{profile.n_series} series outside "
                f"[{low}, {high}] (min {int(profile.lengths.min())}, "
                f"max {int(profile.lengths.max())})")))
    ceiling = float(gates["generator.max_abs_value"]) or FLOAT32_MAX
    if profile.max_abs > ceiling:
        flags.append(Flag("block", "magnitude",
                          f"max |value| {profile.max_abs:.3e} exceeds the {ceiling:.3e} ceiling"))
    max_dup = float(gates["generator.max_dup_fraction"])
    if profile.dup_fraction > max_dup:
        flags.append(Flag("block", "duplicates", (
            f"{profile.dup_fraction:.1%} of the sample is byte-identical to another "
            f"series, over the {max_dup:.0%} cap (sample estimate)")))
    corpus_n = int(gates["generator.corpus_n_series"])
    budget = float(gates["generator.max_generate_seconds"])
    if profile.n_series and profile.generate_seconds > 0:
        projected = profile.generate_seconds * corpus_n / profile.n_series
        if projected > budget:
            flags.append(Flag("block", "generate_budget", (
                f"projected {projected:.0f}s to emit {corpus_n:,} series, over the "
                f"{budget:.0f}s budget (extrapolated from {profile.n_series})")))
        reference = float(gates["training.ref_throughput_tokens_per_s"])
        rate = profile.points_per_second()
        if rate < reference:
            flags.append(Flag("warn", "throughput", (
                f"{rate:,.0f} points/s is {rate / reference:.2f}× the "
                f"{reference:,.0f} reference; wall is the law, so the trainer sees "
                "fewer tokens than a faster king's run (single-process estimate)")))
    if profile.constant_fraction > 0:
        level = "block" if bool(gates["generator.reject_constant"]) else "warn"
        flags.append(Flag(level, "constant", (
            f"{profile.constant_fraction:.1%} of the sample is constant; "
            "a flat series carries no forecasting signal")))
    return flags


# -- claims -----------------------------------------------------------------


def parse_claim(claim: str) -> tuple[str, int]:
    text = claim.strip().lower()
    direction = 0
    if text.endswith("+"):
        direction, text = +1, text[:-1]
    elif text.endswith("-"):
        direction, text = -1, text[:-1]
    if text not in FAMILIES:
        raise ValueError(f"unknown claim family {text!r}; known: {', '.join(FAMILIES)}")
    return text, direction


def check_claims(claims: Sequence[str], doses: dict[str, float],
                 shifts: dict[str, float], threshold: float) -> list[dict[str, Any]]:
    """Does the corpus actually carry the property its author says it carries?

    A claim passes only when the family it names moved beyond seed noise *and*,
    when the claim is signed, moved in the direction claimed. A failing claim is
    the cheapest finding in the harness: the code does not do what you think.
    """
    results = []
    for raw in claims:
        family, direction = parse_claim(raw)
        members = [name for name, fam, _ in FEATURES if fam == family]
        dose = max(doses[name] for name in members)
        shift = shifts[family]
        moved = dose >= threshold
        directed = direction == 0 or (shift > 0 if direction > 0 else shift < 0)
        results.append({
            "claim": raw, "family": family, "max_dose": round(dose, 2),
            "signed_shift": round(shift, 2), "passed": bool(moved and directed),
            "why": ("family moved beyond seed noise in the claimed direction"
                    if moved and directed else
                    f"family moved (dose {dose:.1f}) but in the opposite direction"
                    if moved else
                    f"largest movement in {family} is {dose:.1f}× seed noise, "
                    f"under the {threshold:.1f}× threshold — the corpus does not "
                    "measurably carry this claim"),
        })
    return results


# -- the screen ------------------------------------------------------------


@dataclass
class ScreenInput:
    """Two seeds per arm: one supplies the sample, the pair supplies its noise band."""

    candidate: tuple[Profile, Profile]
    king: tuple[Profile, Profile]
    prior_king: Profile | None = None
    claims: Sequence[str] = field(default_factory=tuple)
    dose_threshold: float = DOSE_THRESHOLD


def screen(data: ScreenInput, gates: dict[str, Any], gates_source: str) -> dict[str, Any]:
    """Run every check and assemble the report. Pure — no I/O, no side effects."""
    cand_a, cand_b = data.candidate
    king_a, king_b = data.king
    scales = _feature_scales(cand_a.matrix, cand_b.matrix, king_a.matrix, king_b.matrix)

    across = feature_distances(cand_a.matrix, king_a.matrix, scales)
    band_cand = feature_distances(cand_a.matrix, cand_b.matrix, scales)
    band_king = feature_distances(king_a.matrix, king_b.matrix, scales)
    band = np.maximum(np.maximum(band_cand, band_king), BAND_FLOOR)
    dose = across / band

    doses = {name: float(dose[i]) for i, name in enumerate(FEATURE_NAMES)}
    per_feature = {
        name: {
            "distance": round(float(across[i]), 3),
            "seed_noise_band": round(float(band[i]), 3),
            "dose": round(float(dose[i]), 2),
            "candidate_median": round(float(np.median(cand_a.matrix[:, i])), 3)
            if cand_a.matrix.size else 0.0,
            "king_median": round(float(np.median(king_a.matrix[:, i])), 3)
            if king_a.matrix.size else 0.0,
        }
        for i, name in enumerate(FEATURE_NAMES)
    }

    # Signed family shift, in band units, positive = "more of the family property".
    shifts: dict[str, float] = {}
    for family in FAMILIES:
        parts = []
        for i, (name, fam, sign) in enumerate(FEATURES):
            if fam != family or sign == 0 or not cand_a.matrix.size or not king_a.matrix.size:
                continue
            delta = float(np.median(cand_a.matrix[:, i]) - np.median(king_a.matrix[:, i]))
            parts.append(sign * (delta / scales[i]) / band[i])
        shifts[family] = float(np.mean(parts)) if parts else 0.0

    coverage = coverage_report(cand_a.matrix, king_a.matrix, scales)
    flags = gate_flags(cand_a, gates)
    flags += _corpus_flags(cand_a, king_a, coverage, doses, data.dose_threshold)
    claims = check_claims(data.claims, doses, shifts, data.dose_threshold)

    moved = sorted((n for n, d in doses.items() if d >= data.dose_threshold),
                   key=lambda n: -doses[n])
    blocked = [f for f in flags if f.level == "block"]
    if blocked:
        verdict = "blocked"
    elif not moved and not _coverage_moved(coverage):
        verdict = "undosed"
    else:
        verdict = "measurable"

    report: dict[str, Any] = {
        "verdict": verdict,
        "verdict_meaning": VERDICT_MEANING[verdict],
        "dose_threshold": data.dose_threshold,
        "gates_source": gates_source,
        "candidate": cand_a.summary(),
        "king": king_a.summary(),
        "moved_features": moved,
        "per_feature": per_feature,
        "family_shift_in_band_units": {k: round(v, 2) for k, v in shifts.items()},
        "coverage": coverage,
        "flags": [f.as_dict() for f in flags],
        "claims": claims,
        "sample": {
            "n_series_per_profile": cand_a.n_series,
            "candidate_seeds": [cand_a.seed, cand_b.seed],
            "king_seeds": [king_a.seed, king_b.seed],
        },
    }
    if data.prior_king is not None:
        report["history"] = history_axis(
            cand_a.matrix, king_a.matrix, data.prior_king.matrix, scales)
        report["prior_king"] = data.prior_king.summary()
    report["challenges"] = challenges(report)
    report["limits"] = LIMITS
    return report


VERDICT_MEANING = {
    "blocked": "a contract gate fails — the round would be wasted; fix before spending",
    "undosed": ("indistinguishable from the king at this sample size — a paired eval "
                "would almost certainly return a null; raise the dose instead of paying"),
    "measurable": ("the corpora differ beyond seed noise, so a paired eval can resolve "
                   "something — this says nothing about which way"),
}

LIMITS = (
    "Feature distance is not score improvement: a corpus can differ on every "
    "feature and still train a worse forecaster. This screen can only rule "
    "spending out, never rule a win in. A verdict of `measurable` is a licence "
    "to run miner.evaluate on both arms in one batch, nothing more.",
    "Statistics come from a sample of a few hundred series, not the 16,384-series "
    "corpus; duplicate fraction and throughput are extrapolations.",
    "The noise band is a two-seed estimate of the generator's own variability. It "
    "is not the *training* noise floor, which is larger and can only be measured "
    "by training the same generator twice.",
)


def _coverage_moved(coverage: dict[str, Any]) -> bool:
    return (float(coverage.get("novelty", 0.0)) > 0.10
            or float(coverage.get("coverage_loss", 0.0)) > 0.10)


def _corpus_flags(candidate: Profile, king: Profile, coverage: dict[str, Any],
                  doses: dict[str, float], threshold: float) -> list[Flag]:
    """Warnings about the bet itself, once the contract gates are satisfied."""
    flags: list[Flag] = []
    loss = float(coverage.get("coverage_loss", 0.0))
    if loss > 0.10:
        flags.append(Flag("warn", "coverage_loss", (
            f"{loss:.0%} of king series have no candidate neighbour within the "
            "king's own spacing — model capacity is fixed, so this is a trade, "
            "not an addition; name what you expect to lose")))
    diversity = float(coverage.get("diversity_ratio", 1.0))
    if diversity < 0.5:
        flags.append(Flag("warn", "redundancy", (
            f"candidate series sit {1 / max(diversity, EPS):.1f}× closer together "
            "than the king's — the effective corpus is smaller than its series "
            "count suggests")))
    if candidate.matrix.size:
        entropy = candidate.matrix[:, FEATURE_NAMES.index("spectral_entropy")]
        ar_skill = candidate.matrix[:, FEATURE_NAMES.index("ar_skill")]
        unlearnable = float(np.mean((entropy > 0.95) & (ar_skill > -0.02)))
        if unlearnable > 0.20:
            flags.append(Flag("warn", "unlearnable", (
                f"{unlearnable:.0%} of the sample is spectrally flat with no linear "
                "structure beyond a random walk — those tokens spend budget "
                "teaching noise")))
        trivial = float(np.mean(ar_skill < -1.0))
        king_trivial = float(np.mean(
            king.matrix[:, FEATURE_NAMES.index("ar_skill")] < -1.0)) if king.matrix.size else 0.0
        if trivial > max(0.30, king_trivial * 1.5):
            flags.append(Flag("warn", "trivial", (
                f"{trivial:.0%} of the sample is near-deterministic (AR(4) beats a "
                f"random walk 10×) against the king's {king_trivial:.0%}; warm start "
                "means the model already has the easy structure")))
    if not any(d >= threshold for d in doses.values()):
        flags.append(Flag("warn", "undosed", (
            "no feature moved beyond seed noise; whatever the code intends, the "
            "corpus it emits is the king's corpus to this screen's resolution")))
    return flags


def challenges(report: dict[str, Any]) -> list[str]:
    """The questions this report forces on the author, each tied to a number."""
    out: list[str] = []
    coverage = report.get("coverage", {})
    moved = report.get("moved_features", [])
    per_feature = report.get("per_feature", {})
    if report["verdict"] == "undosed":
        out.append(
            "Nothing moved past this generator's own seed noise. Before paying for a "
            "paired eval, state what dose you expect to be visible and re-screen at "
            "that dose — a null result here costs nothing and a null on the GPU costs "
            "a rental.")
    if moved:
        top = moved[0]
        out.append(
            f"The largest movement is `{top}` at {per_feature[top]['dose']:.1f}× seed "
            f"noise ({per_feature[top]['king_median']} → "
            f"{per_feature[top]['candidate_median']}). Is that the change you meant to "
            "make, or a side effect of one you meant to make elsewhere?")
    loss = float(coverage.get("coverage_loss", 0.0))
    novelty = float(coverage.get("novelty", 0.0))
    if loss > 0.05 or novelty > 0.05:
        out.append(
            f"You add {novelty:.0%} new coverage and drop {loss:.0%} of the king's. "
            "Fixed capacity means the model trades one for the other — which of the "
            "king's regimes are you willing to forecast worse, and does the private "
            "pool contain them?")
    if novelty <= 0.05 and moved:
        out.append(
            "Movement without novelty: you re-weighted regimes the king already "
            "covers rather than adding coverage. That is a legitimate bet — the "
            "lower bound rewards evening out weak sources — but say so explicitly, "
            "because it predicts a small effect that needs more seeds to see.")
    history = report.get("history", {})
    if history.get("reading") == "moves against the last winning direction":
        out.append(
            f"This corpus moves against the only observed winning transition "
            f"(cosine {history['cosine_with_last_winning_move']}). That is one noisy "
            "data point, not a law — but if you are betting against it, say why.")
    if history.get("closer_to_dethroned_king"):
        out.append(
            f"The corpus sits closer to the dethroned king "
            f"({history['mean_distance_to_prior_king']}) than to the reigning one "
            f"({history['mean_distance_to_king']}). A prior that already lost a duel "
            "is a strange place to move towards; if the resemblance is incidental, "
            "name the feature that makes this different.")
    throughput = [f for f in report["flags"] if f["name"] == "throughput"]
    if throughput:
        out.append(
            "Throughput is below the reference: the trainer will see fewer tokens "
            "than a faster generator's run. Your prior has to beat the king's by "
            "more than that token deficit, not just beat it.")
    failed = [c["claim"] for c in report.get("claims", []) if not c["passed"]]
    if failed:
        out.append(
            f"Claims not carried by the corpus: {', '.join(failed)}. The code does "
            "not measurably do what its author says; fix the generator or the claim "
            "before spending a GPU hour on either.")
    out.append(
        "This screen ranks nothing. If you believe this corpus beats the king, name "
        "the mechanism, the source it should win on, and the size of the effect "
        "before the eval runs — then let miner.analyze contradict you.")
    return out


# -- rendering --------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    lines = [
        f"screen: {report['candidate']['name']} vs king {report['king']['name']}",
        f"verdict: {report['verdict'].upper()} — {report['verdict_meaning']}",
        f"gates from {report['gates_source']}",
        (f"sample: {report['sample']['n_series_per_profile']} series/profile, "
         f"candidate seeds {report['sample']['candidate_seeds']}, "
         f"king seeds {report['sample']['king_seeds']}"),
        "",
        ("dose = distance / seed-noise band: how reproducible a difference is, "
         "not how much it matters."),
        "read the medians beside it for the size of the change in the feature's own units.",
        "",
        f"{'feature':<20}{'king':>10}{'candidate':>12}{'distance':>10}{'band':>8}{'dose':>8}",
        "-" * 68,
    ]
    for name, row in sorted(report["per_feature"].items(), key=lambda kv: -kv[1]["dose"]):
        mark = " *" if row["dose"] >= report["dose_threshold"] else ""
        lines.append(f"{name:<20}{row['king_median']:>10.3f}{row['candidate_median']:>12.3f}"
                     f"{row['distance']:>10.3f}{row['seed_noise_band']:>8.3f}"
                     f"{row['dose']:>7.1f}x{mark}")
    lines.append("")
    coverage = report["coverage"]
    if "note" in coverage:
        lines.append(f"coverage: {coverage['note']}")
    else:
        lines.append(
            f"coverage: novelty {coverage['novelty']:.0%} of candidate series are outside "
            f"the king's spacing; {coverage['coverage_loss']:.0%} of king series lose "
            f"their neighbourhood; diversity {coverage['diversity_ratio']:.2f}× the king's")
    if "history" in report:
        history = report["history"]
        lines.append(
            f"history: distance to king {history['mean_distance_to_king']}, to the "
            f"dethroned king {history['mean_distance_to_prior_king']} "
            f"(king to dethroned king {history['mean_king_to_prior_king']})")
        lines.append("         " + (history.get("note") or
                     f"{history['reading']} (cosine "
                     f"{history['cosine_with_last_winning_move']}) — {history['caveat']}"))
    if report["claims"]:
        lines.append("")
        lines.append("claims:")
        for claim in report["claims"]:
            lines.append(f"  [{'PASS' if claim['passed'] else 'FAIL'}] {claim['claim']}: "
                         f"{claim['why']}")
    if report["flags"]:
        lines.append("")
        lines.append("flags:")
        for flag in report["flags"]:
            lines.append(f"  [{flag['level']}] {flag['name']}: {flag['detail']}")
    lines.append("")
    lines.append("challenge your own case:")
    for item in report["challenges"]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("limits:")
    for item in report["limits"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


# Exit codes, so a hook can branch without parsing prose: 0 go, 1 fix the
# generator, 3 raise the dose, 4 the corpus does not carry a stated claim.
# 2 is skipped deliberately — argparse exits 2 on a usage error, and a caller
# must not read "you typed the flag wrong" as "the corpus is undosed".
EXIT_CODES = {"measurable": 0, "blocked": 1, "undosed": 3}
CLAIM_FAILED = 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--king", type=Path, required=True,
                        help="the king's generator directory (cascade fetch <king_ref>)")
    parser.add_argument("--prior-king", type=Path, default=None,
                        help="optional: the king this king dethroned, for direction only")
    parser.add_argument("--n-series", type=int, default=256,
                        help="series per profile; four profiles are generated")
    parser.add_argument("--seed", type=int, default=0,
                        help="first seed; seed+1 supplies the noise band")
    parser.add_argument("--claim", action="append", default=[],
                        help=f"assert a family moved, e.g. seasonality+ ({', '.join(FAMILIES)})")
    parser.add_argument("--dose-threshold", type=float, default=DOSE_THRESHOLD)
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)
    if args.n_series < 8:
        parser.error("--n-series must be at least 8 for the distances to mean anything")
    for claim in args.claim:
        try:
            parse_claim(claim)
        except ValueError as error:
            parser.error(str(error))

    gates, gates_source = load_gates()
    data = ScreenInput(
        candidate=(profile_generator(args.candidate, args.seed, args.n_series),
                   profile_generator(args.candidate, args.seed + 1, args.n_series)),
        king=(profile_generator(args.king, args.seed, args.n_series),
              profile_generator(args.king, args.seed + 1, args.n_series)),
        prior_king=(profile_generator(args.prior_king, args.seed, args.n_series)
                    if args.prior_king else None),
        claims=tuple(args.claim),
        dose_threshold=args.dose_threshold,
    )
    report = screen(data, gates, gates_source)
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {args.json}")
    code = EXIT_CODES[report["verdict"]]
    if code == 0 and any(not claim["passed"] for claim in report["claims"]):
        return CLAIM_FAILED
    return code


if __name__ == "__main__":
    raise SystemExit(main())
