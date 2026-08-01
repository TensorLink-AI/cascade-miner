"""One-look miner status assembled from the controller's ``runs/`` files.

Everything here is a read: the controller state, the approval queue, the event
history tail, and the candidate working-tree digest. ``--json`` emits the same
summary as data, which is also what the MCP ``miner_status`` tool returns.

    python -m miner.status
    python -m miner.status --json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

from miner import experiments
from miner.controller import candidate_digest, candidate_dirty, read_json

STATE_FILE = Path("runs/controller-state.json")
APPROVALS_FILE = Path("runs/approvals.json")
EVENTS_FILE = Path("runs/controller-events.jsonl")
EVENT_TAIL = 10


def tail_events(path: Path, limit: int = EVENT_TAIL) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                lines.append(line)
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize(root: Path) -> dict[str, Any]:
    state = read_json(root / STATE_FILE)
    receipt = state.get("last_receipt", {}) if isinstance(state, dict) else {}
    summary = receipt.get("summary", {}) if isinstance(receipt, dict) else {}
    approvals = read_json(root / APPROVALS_FILE).get("requests", [])
    if not isinstance(approvals, list):
        approvals = []
    try:
        candidate = {
            "digest": candidate_digest(root),
            "dirty": candidate_dirty(root),
        }
    except Exception:  # noqa: BLE001 - status must render without git
        candidate = {"digest": None, "dirty": None}
    ledger = experiments.list_entries(root, limit=5)
    return {
        "initialized": bool(state.get("initialized")),
        "updated_at": state.get("updated_at"),
        "round": {
            "id": state.get("last_round_id"),
            "status": receipt.get("status"),
            "king_gen_ref": summary.get("king_gen_ref"),
            "miner_participant": receipt.get("miner_participant"),
            "miner_heat": receipt.get("miner_heat"),
        },
        "eval_pool": {
            "repo": state.get("eval_pool_repo"),
            "revision": state.get("eval_pool_revision"),
            "snapshot": state.get("eval_pool_snapshot"),
            "local_snapshots": state.get("eval_pool_local_snapshots", []),
        },
        "cascade_head": state.get("cascade_head"),
        "candidate": candidate,
        "approvals": {
            "pending": [r for r in approvals if r.get("status") == "pending"],
            "counts": {
                status: sum(1 for r in approvals if r.get("status") == status)
                for status in sorted({str(r.get("status")) for r in approvals})
            },
        },
        "experiments": ledger,
        "recent_events": tail_events(root / EVENTS_FILE),
    }


def render(summary: dict[str, Any]) -> str:
    lines = []
    round_info = summary["round"]
    lines.append(f"round:       {round_info['id'] or '(none)'}"
                 f"  status={round_info['status'] or '?'}")
    lines.append(f"king ref:    {round_info['king_gen_ref'] or '(unknown)'}")
    heat = round_info.get("miner_heat")
    if heat:
        lines.append(f"our heat:    {json.dumps(heat, sort_keys=True)}")
    pool = summary["eval_pool"]
    revision = str(pool["revision"] or "")
    lines.append(f"eval pool:   {pool['repo'] or '(unsynced)'}"
                 f"@{revision[:12]}  local={','.join(pool['local_snapshots']) or 'none'}")
    candidate = summary["candidate"]
    digest = str(candidate["digest"] or "")
    dirty = candidate["dirty"]
    lines.append(f"candidate:   {digest[:12] or '(unknown)'}"
                 f"  {'dirty (changes awaiting review)' if dirty else 'clean' if dirty is not None else ''}")
    pending = summary["approvals"]["pending"]
    if pending:
        lines.append(f"approvals:   {len(pending)} pending")
        for request in pending:
            lines.append(f"  [{request.get('id')}] {request.get('action')}: "
                         f"{request.get('reason', '')}")
        lines.append("  approve with: .venv/bin/python -m miner.controller --approve <id>")
    else:
        lines.append("approvals:   none pending")
    entries = summary["experiments"]
    if entries:
        lines.append(f"experiments: last {len(entries)} ledger entries")
        for entry in entries:
            lines.append(f"  {entry.get('id')} {entry.get('status'):<9} "
                         f"{entry.get('hypothesis', '')}")
    lines.append(f"updated at:  {summary['updated_at'] or '(controller has not run)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(args.root.resolve())
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
