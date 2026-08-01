"""Structured experiment ledger: what was tried, under what pairing, and how it ended.

`notes/EXPERIMENTS.md` is prose for humans; agents need the same record as
data they can query without parsing markdown. Each entry is one line of JSON
in append-only ``runs/experiments.jsonl``. The methodology fields exist so a
later reader can tell whether two numbers are comparable at all: which seeds,
which eval-pool snapshots, and the noise floor the verdict was read against.

    python -m miner.experiments log --hypothesis "..." --change "..." \
        --seeds 0,1,2 --status planned
    python -m miner.experiments list --status measured --json

The ledger never replaces the markdown log — record the narrative there too.
Statuses: ``planned`` (not yet run), ``running``, ``measured`` (components
saved, verdict pending), ``null`` (inside the noise floor), ``win``, ``loss``,
``abandoned`` (say what it cost).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

LEDGER = Path("runs/experiments.jsonl")
STATUSES = ("planned", "running", "measured", "null", "win", "loss", "abandoned")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def log_entry(root: Path, *, hypothesis: str, change: str = "",
              arm: str = "generators/candidate", seeds: list[Any] | None = None,
              snapshots: list[str] | None = None, status: str = "planned",
              noise_floor: float | None = None, result: str = "",
              supersedes: str = "", notes: str = "") -> dict[str, Any]:
    """Append one experiment entry and return it (with its assigned id)."""
    hypothesis = hypothesis.strip()
    if not hypothesis:
        raise ValueError("an experiment entry needs a hypothesis")
    if status not in STATUSES:
        raise ValueError(
            f"unknown status {status!r}; expected one of {', '.join(STATUSES)}"
        )
    entry = {
        "at": utc_now(),
        "hypothesis": hypothesis,
        "change": change.strip(),
        "arm": arm,
        "seeds": seeds or [],
        "snapshots": snapshots or [],
        "status": status,
        "noise_floor": noise_floor,
        "result": result.strip(),
        "supersedes": supersedes.strip(),
        "notes": notes.strip(),
    }
    entry["id"] = hashlib.sha256(
        json.dumps(entry, sort_keys=True).encode()
    ).hexdigest()[:12]
    path = root / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def list_entries(root: Path, *, status: str = "",
                 limit: int = 0) -> list[dict[str, Any]]:
    """Ledger entries oldest-first, optionally filtered; ``limit`` keeps the newest."""
    path = root / LEDGER
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if status and entry.get("status") != status:
            continue
        entries.append(entry)
    return entries[-limit:] if limit > 0 else entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)

    log = commands.add_parser("log", help="append one experiment entry")
    log.add_argument("--hypothesis", required=True)
    log.add_argument("--change", default="",
                     help="the one thing this arm changes vs its control")
    log.add_argument("--arm", default="generators/candidate")
    log.add_argument("--seeds", default="", help="comma-separated seed list")
    log.add_argument("--snapshots", default="",
                     help="comma-separated eval-pool snapshots the arms were scored on")
    log.add_argument("--status", default="planned", choices=STATUSES)
    log.add_argument("--noise-floor", type=float, default=None,
                     help="the measured noise floor the verdict was read against")
    log.add_argument("--result", default="")
    log.add_argument("--supersedes", default="", help="id of the entry this updates")
    log.add_argument("--notes", default="")

    show = commands.add_parser("list", help="print ledger entries")
    show.add_argument("--status", default="", choices=("",) + STATUSES)
    show.add_argument("--limit", type=int, default=0)
    show.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "log":
        entry = log_entry(
            root, hypothesis=args.hypothesis, change=args.change, arm=args.arm,
            seeds=_split_csv(args.seeds), snapshots=_split_csv(args.snapshots),
            status=args.status, noise_floor=args.noise_floor, result=args.result,
            supersedes=args.supersedes, notes=args.notes,
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 0
    entries = list_entries(root, status=args.status, limit=args.limit)
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return 0
    for entry in entries:
        floor = entry.get("noise_floor")
        print(f"{entry.get('id', '?'):<14}{entry.get('status', ''):<10}"
              f"{entry.get('at', ''):<22}"
              f"floor={floor if floor is not None else '—'}  "
              f"{entry.get('hypothesis', '')}")
    if not entries:
        print("no matching experiment entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
