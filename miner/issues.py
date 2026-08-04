"""Structured issue tracker: bugs and feature requests agents can file in-repo.

Agents in this harness have no GitHub access and no channel for "the harness
itself is broken" or "the harness should do X" — the experiment ledger records
what was *tried*, not what is *wrong*. Each report is one line of JSON in
append-only ``runs/issues.jsonl``, write-only for agents and surfaced to the
operator through ``miner.status``. Filing an issue never executes anything.

    python -m miner.issues report --kind bug --title "..." --detail "..."
    python -m miner.issues list --status open --json

Kinds: ``bug`` (something is broken) and ``feature`` (something is missing).
Statuses: ``open`` (the only status an agent should file), ``resolved``, and
``wontfix`` — a triaged entry supersedes the original by id, mirroring the
experiment ledger, so history is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

LEDGER = Path("runs/issues.jsonl")
KINDS = ("bug", "feature")
STATUSES = ("open", "resolved", "wontfix")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def report_entry(root: Path, *, kind: str, title: str, detail: str = "",
                 area: str = "", status: str = "open",
                 supersedes: str = "") -> dict[str, Any]:
    """Append one issue entry and return it (with its assigned id)."""
    title = title.strip()
    if not title:
        raise ValueError("an issue entry needs a title")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    if status not in STATUSES:
        raise ValueError(
            f"unknown status {status!r}; expected one of {', '.join(STATUSES)}"
        )
    entry = {
        "at": utc_now(),
        "kind": kind,
        "title": title,
        "detail": detail.strip(),
        "area": area.strip(),
        "status": status,
        "supersedes": supersedes.strip(),
    }
    entry["id"] = hashlib.sha256(
        json.dumps(entry, sort_keys=True).encode()
    ).hexdigest()[:12]
    path = root / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def list_entries(root: Path, *, kind: str = "", status: str = "",
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
        if kind and entry.get("kind") != kind:
            continue
        if status and entry.get("status") != status:
            continue
        entries.append(entry)
    return entries[-limit:] if limit > 0 else entries


def open_entries(root: Path) -> list[dict[str, Any]]:
    """Open issues after applying supersedes: the operator's live worklist."""
    superseded = set()
    entries = list_entries(root)
    for entry in entries:
        if entry.get("supersedes"):
            superseded.add(entry["supersedes"])
    return [entry for entry in entries
            if entry.get("status") == "open" and entry.get("id") not in superseded]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="file one bug or feature request")
    report.add_argument("--kind", required=True, choices=KINDS)
    report.add_argument("--title", required=True)
    report.add_argument("--detail", default="",
                        help="reproduction steps for a bug; motivation for a feature")
    report.add_argument("--area", default="",
                        help="what it concerns, e.g. miner/controller.py or docs")
    report.add_argument("--status", default="open", choices=STATUSES)
    report.add_argument("--supersedes", default="",
                        help="id of the entry this triages (e.g. open -> resolved)")

    show = commands.add_parser("list", help="print issue entries")
    show.add_argument("--kind", default="", choices=("",) + KINDS)
    show.add_argument("--status", default="", choices=("",) + STATUSES)
    show.add_argument("--open", action="store_true", dest="open_only",
                      help="only issues still open after supersedes are applied")
    show.add_argument("--limit", type=int, default=0)
    show.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "report":
        entry = report_entry(
            root, kind=args.kind, title=args.title, detail=args.detail,
            area=args.area, status=args.status, supersedes=args.supersedes,
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 0
    if args.open_only:
        entries = open_entries(root)
        if args.kind:
            entries = [e for e in entries if e.get("kind") == args.kind]
        entries = entries[-args.limit:] if args.limit > 0 else entries
    else:
        entries = list_entries(root, kind=args.kind, status=args.status,
                               limit=args.limit)
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return 0
    for entry in entries:
        print(f"{entry.get('id', '?'):<14}{entry.get('kind', ''):<9}"
              f"{entry.get('status', ''):<9}{entry.get('at', ''):<22}"
              f"{entry.get('title', '')}")
    if not entries:
        print("no matching issue entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
