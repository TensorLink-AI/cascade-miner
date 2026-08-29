"""Aggregate the matrix: geomean per candidate + paired cluster-bootstrap LCB vs king.

The geomean is a point estimate; the KOTH throne is decided on the LCB of the
paired bootstrap clearing `[scoring] win_margin`. This computes the real thing:
resampling paired across king and challenger, clustered by `source` (windows from
one feed are correlated), aggregating MWSQL numerator/denominator before dividing.

Pairing requires identical windows, so king and challenger must come from the same
(seed, snapshot); series_id order is asserted rather than assumed.

    python -m miner.analyze --scores-root scores --king king-control
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def effective_alpha(alpha: float, cohort_k: int) -> float:
    """Per-challenger alpha in a k-challenger cohort duel (DEC-CA-0012).

    Since max_finalists armed (2026-08-20) a statistically tied heat top duels
    as a cohort, and the validator judges each challenger at ``alpha / k`` —
    the quantile tightens, never the margin. An LCB computed at the solo alpha
    can clear the margin and still lose the same duel judged in a cohort, so
    size the target against the worst case you might land in.
    """
    return alpha / max(1, cohort_k)


def load(scores_root: Path):
    """-> {candidate: {seed: {snapshot: npz_dict}}} plus geomean summaries."""
    runs: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    geo: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for d in sorted(scores_root.iterdir()):
        if not d.is_dir() or "__seed" not in d.name:
            continue
        cand, seed = d.name.rsplit("__seed", 1)
        seed = int(seed)
        for npz in sorted(d.glob("*.npz")):
            with np.load(npz, allow_pickle=True) as z:
                runs[cand][seed][npz.stem] = {k: z[k] for k in z.files}
        s = d / "summary.json"
        if s.exists():
            for r in json.loads(s.read_text()).get("per_snapshot", []):
                if "geomean" in r:
                    geo[cand][seed][r["snapshot"]] = r["geomean"]
    return runs, geo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-root", type=Path, default=Path("scores"))
    ap.add_argument("--king", default="king_live")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cohort-k", type=int, default=1,
                    help="challengers in the duel cohort; the LCB is computed "
                         "at alpha/k as the validator judges a tied cohort "
                         "(DEC-CA-0012). 1 = solo duel, today's default; the "
                         "worst case is round.max_finalists.")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="the bar the LCB must clear; decays with king tenure "
                         "(DEC-CA-0016) from win_margin_start to win_margin_end")
    ap.add_argument("--out", type=Path, default=Path("results/matrix.json"))
    args = ap.parse_args()

    from cascade.eval.bootstrap import paired_bootstrap_lcb_aggregated

    alpha = effective_alpha(args.alpha, args.cohort_k)
    if args.cohort_k > 1:
        print(f"cohort_k = {args.cohort_k}: LCB quantile at alpha "
              f"{args.alpha:g}/{args.cohort_k} = {alpha:g}")

    runs, geo = load(args.scores_root)
    if args.king not in runs:
        raise SystemExit(f"no king runs under {args.scores_root} (looked for '{args.king}')")

    report = {}
    for cand in sorted(runs):
        if cand == args.king:
            continue
        lcbs, pairs = [], 0
        for seed in sorted(runs[cand]):
            if seed not in runs[args.king]:
                continue
            for snap in sorted(runs[cand][seed]):
                k = runs[args.king][seed].get(snap)
                c = runs[cand][seed][snap]
                if k is None:
                    continue
                if not np.array_equal(k["series_id"], c["series_id"]):
                    print(f"  !! {cand} seed{seed} {snap}: window order differs — skipping")
                    continue
                lcb = paired_bootstrap_lcb_aggregated(
                    k["qloss"], k["abs_target"], k["mase"],
                    c["qloss"], c["abs_target"], c["mase"],
                    alpha=alpha, B=args.B, seed=seed,
                    clusters=list(k["source"]),
                )
                lcbs.append(lcb)
                pairs += 1
        g = [v for s in geo[cand].values() for v in s.values()]
        gk = [v for s in geo[args.king].values() for v in s.values()]
        report[cand] = {
            "mean_geomean": float(np.mean(g)) if g else None,
            "king_mean_geomean": float(np.mean(gk)) if gk else None,
            "mean_lcb": float(np.mean(lcbs)) if lcbs else None,
            "min_lcb": float(np.min(lcbs)) if lcbs else None,
            "frac_lcb_over_margin": float(np.mean([x > args.margin for x in lcbs])) if lcbs else None,
            "n_pairs": pairs,
            "n_seeds": len(runs[cand]),
            "alpha": alpha,
            "cohort_k": args.cohort_k,
        }

    hdr = f"{'candidate':<18}{'geomean':>10}{'vs king':>9}{'mean LCB':>10}{'min LCB':>9}{'>margin':>9}{'n':>5}"
    print(hdr); print("-" * len(hdr))
    kg = next((v["king_mean_geomean"] for v in report.values() if v["king_mean_geomean"]), None)
    for cand, r in sorted(report.items(), key=lambda x: x[1]["mean_geomean"] or 9e9):
        edge = 100 * (kg - r["mean_geomean"]) / kg if kg and r["mean_geomean"] else float("nan")
        print(f"{cand:<18}{r['mean_geomean']:>10.5f}{edge:>8.1f}%{r['mean_lcb']:>10.4f}"
              f"{r['min_lcb']:>9.4f}{r['frac_lcb_over_margin']:>8.0%}{r['n_pairs']:>5}")
    print(f"\nking ({args.king}) mean geomean: {kg:.5f}" if kg else "")
    print(f"win_margin = {args.margin}; LCB must exceed it to dethrone.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
