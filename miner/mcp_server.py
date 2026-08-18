"""MCP stdio server: the harness as tools for any MCP-capable agent.

The improvement backends in ``scripts/improve_candidate.py`` cover agents this
repo knows how to spawn. This is the other direction — an agent the *operator*
runs (Claude Code, Codex, anything speaking MCP) connects to the harness and
mines through a typed tool surface instead of shelling into ``miner.*``
modules and parsing prose.

The privilege boundary is identical to every other agent path: reads, cheap
local verification, the experiment ledger, and *requests*. ``request_action``
writes a durable entry into the controller's approval queue; it never executes
anything, and the controller still applies its mode/policy gates before any
approved command runs. Wallet secrets are never readable here.

Dependency-free by design: MCP's stdio transport is newline-delimited
JSON-RPC 2.0, which the standard library covers. Register with e.g.:

    claude mcp add cascade-miner -- .venv/bin/python -m miner.mcp_server

The server is stateless between calls; all state lives in ``runs/`` where the
controller and humans already look.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from miner import experiments
from miner import issues
from miner import policy as policy_module
from miner import status as status_module
from miner.controller import queue_approval, read_json

SERVER_NAME = "cascade-miner"
SERVER_VERSION = "0.1.0"
KNOWN_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
IMPROVE_REQUEST = Path("runs/improve-request.json")
IMPROVE_RESPONSE = Path("runs/improve-response.json")
IMPROVE_STATUSES = ("completed", "failed", "rejected", "skipped")
QUICK_VERIFY_TIMEOUT = 900
FETCH_TIMEOUT = 600
SCREEN_TIMEOUT = 1800
SCREEN_DIR = Path("runs/screen")
GENERATOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

HARD_LIMITS = (
    "Never rent a pod, register a hotkey, or submit on-chain yourself; use "
    "request_action and let the controller gate it.",
    "Never touch wallet secrets, mnemonics, or wallet/chain commands.",
    "Never edit the read-only Cascade reference checkout.",
    "Change one thing per experiment arm, against a freshly trained control "
    "in the same batch.",
    "State the seed count and noise floor with every number; a result inside "
    "the noise floor is null and must be reported as null.",
    "Record every experiment in the ledger (log_experiment) and in "
    "notes/EXPERIMENTS.md.",
)


class ToolError(RuntimeError):
    """A tool-level failure reported to the caller, not a protocol error."""


def _interpreter(root: Path) -> str:
    venv = root / ".venv/bin/python"
    return str(venv) if venv.exists() else sys.executable


def _candidate_path(root: Path, value: Any) -> Path:
    """Resolve a candidate path argument, refusing anything outside generators/."""
    raw = str(value or "generators/candidate")
    path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    generators = (root / "generators").resolve()
    if generators != path and generators not in path.parents:
        raise ToolError(f"candidate_path must live under generators/, got {raw!r}")
    return path


def default_chain_toml(root: Path) -> Path:
    """Resolve the chain config the same way the rest of the harness does."""
    named = os.environ.get("CASCADE_CHAIN_TOML", "").strip()
    if named:
        return Path(named)
    cascade_dir = os.environ.get("CASCADE_DIR", "").strip() or "/root/cascade"
    return Path(cascade_dir) / "chain.toml"


class Server:
    def __init__(self, root: Path, chain_toml: Path | None = None,
                 network: str = "finney"):
        self.root = root
        self.chain_toml = chain_toml or default_chain_toml(root)
        self.network = network
        self.tools: list[dict[str, Any]] = []
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._register_tools()

    # -- tool registry -----------------------------------------------------

    def _tool(self, name: str, description: str, schema: dict[str, Any],
              handler: Callable[[dict[str, Any]], Any]) -> None:
        self.tools.append({
            "name": name,
            "description": description,
            "inputSchema": {"type": "object", "additionalProperties": False,
                            **schema},
        })
        self.handlers[name] = handler

    def _register_tools(self) -> None:
        no_args: dict[str, Any] = {"properties": {}}
        self._tool(
            "miner_status",
            "One-look harness status: round, king ref, eval pool, candidate "
            "digest, pending approvals, open issues, recent events, "
            "experiment ledger tail.",
            no_args, lambda args: status_module.summarize(self.root),
        )
        self._tool(
            "get_state",
            "Raw controller state (runs/controller-state.json).",
            no_args,
            lambda args: read_json(self.root / status_module.STATE_FILE),
        )
        self._tool(
            "get_latest_receipt",
            "The latest audited round receipt recorded by the controller, "
            "including summary.king_gen_ref — the auditable control input.",
            no_args,
            lambda args: read_json(
                self.root / status_module.STATE_FILE).get("last_receipt", {}),
        )
        self._tool(
            "list_heat_entrants",
            "Every participant and heat entrant from the latest stored round "
            "receipt — not just this miner's entry. Past rounds' generators "
            "are public after reveal, so entrants from completed rounds are "
            "study material for fetch_generator.",
            no_args, self._list_heat_entrants,
        )
        self._tool(
            "fetch_generator",
            "Fetch any public generator artefact by reference into "
            "generators/<out_name> — the king from the latest receipt, a "
            "past heat entrant, a dethroned king. Free and read-only; only "
            "artefacts from completed (revealed) rounds are fetchable.",
            {
                "properties": {
                    "ref": {"type": "string",
                            "description": "Immutable generator reference, "
                                           "e.g. summary.king_gen_ref."},
                    "out_name": {"type": "string",
                                 "description": "Directory name under "
                                                "generators/. Derived from "
                                                "the ref when omitted."},
                    "overwrite": {"type": "boolean",
                                  "description": "Replace an existing "
                                                 "non-empty destination."},
                },
                "required": ["ref"],
            },
            self._fetch_generator,
        )
        self._tool(
            "list_approvals",
            "Privileged-action approval queue (runs/approvals.json), "
            "optionally filtered by status.",
            {"properties": {"status": {
                "type": "string",
                "description": "pending, approved, rejected, completed, "
                               "failed, or blocked",
            }}},
            self._list_approvals,
        )
        self._tool(
            "request_action",
            "Request a privileged action (gpu_evaluation, create_hotkey, "
            "register_hotkey, submit_candidate). Writes a durable approval "
            "request the controller executes only after its mode/policy gate. "
            "This never runs anything itself.",
            {
                "properties": {
                    "action": {"type": "string",
                               "enum": list(policy_module.SUPPORTED_ACTIONS)},
                    "reason": {"type": "string",
                               "description": "Why this spend/action is needed "
                                              "now; shown to the approver."},
                    "candidate_path": {"type": "string",
                                       "description": "Defaults to "
                                                      "generators/candidate."},
                    "estimated_hours": {"type": "number",
                                        "exclusiveMinimum": 0},
                },
                "required": ["action", "reason"],
            },
            self._request_action,
        )
        self._tool(
            "get_policy",
            "The operator's autonomy policy (per-action caps) plus current "
            "trailing-window usage per action.",
            no_args, self._get_policy,
        )
        self._tool(
            "log_experiment",
            "Append one entry to the structured experiment ledger "
            "(runs/experiments.jsonl). Also record the narrative in "
            "notes/EXPERIMENTS.md.",
            {
                "properties": {
                    "hypothesis": {"type": "string"},
                    "change": {"type": "string"},
                    "arm": {"type": "string"},
                    "seeds": {"type": "array", "items": {"type": "integer"}},
                    "snapshots": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string",
                               "enum": list(experiments.STATUSES)},
                    "noise_floor": {"type": "number"},
                    "result": {"type": "string"},
                    "supersedes": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["hypothesis"],
            },
            self._log_experiment,
        )
        self._tool(
            "list_experiments",
            "Query the experiment ledger: what has been tried, its pairing "
            "(seeds/snapshots), and how it ended.",
            {"properties": {
                "status": {"type": "string",
                           "enum": list(experiments.STATUSES)},
                "limit": {"type": "integer", "minimum": 1},
            }},
            lambda args: experiments.list_entries(
                self.root, status=str(args.get("status", "")),
                limit=int(args.get("limit", 0) or 0)),
        )
        self._tool(
            "report_issue",
            "File a bug report or feature request about the harness itself "
            "into the append-only issue ledger (runs/issues.jsonl). "
            "Surfaced to the operator via miner_status; filing one never "
            "executes anything. Not for experiment results — use "
            "log_experiment for those.",
            {
                "properties": {
                    "kind": {"type": "string", "enum": list(issues.KINDS)},
                    "title": {"type": "string",
                              "description": "One line naming the problem or "
                                             "the missing capability."},
                    "detail": {"type": "string",
                               "description": "Reproduction steps for a bug; "
                                              "motivation for a feature."},
                    "area": {"type": "string",
                             "description": "What it concerns, e.g. "
                                            "miner/controller.py or docs."},
                },
                "required": ["kind", "title"],
            },
            self._report_issue,
        )
        self._tool(
            "list_issues",
            "Query the issue ledger: bugs and feature requests already filed, "
            "so duplicates can be avoided. open_only applies triage "
            "(supersedes) and keeps the live worklist.",
            {"properties": {
                "kind": {"type": "string", "enum": list(issues.KINDS)},
                "status": {"type": "string", "enum": list(issues.STATUSES)},
                "open_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1},
            }},
            self._list_issues,
        )
        self._tool(
            "run_quick_verify",
            "Run the memory-bounded determinism and contract check on a "
            "candidate. Free and local. NOT a substitute for full `cascade "
            "verify` before submission.",
            {"properties": {
                "candidate_path": {"type": "string"},
                "n_series": {"type": "integer", "minimum": 1},
            }},
            self._run_quick_verify,
        )
        self._tool(
            "screen_candidate",
            "Free pre-GPU screen of a candidate corpus against the king's "
            "(fetch it first): contract gates, per-feature distance expressed "
            "in units of the generator's own seed noise, coverage traded away, "
            "and a check that the corpus carries the properties you claim for "
            "it. Verdicts are `blocked` (fix it), `undosed` (a paired eval "
            "would return a null — raise the dose) and `measurable` (worth "
            "paying for). It never predicts a winner.",
            {"properties": {
                "candidate_path": {"type": "string",
                                   "description": "Defaults to generators/candidate."},
                "king_path": {"type": "string",
                              "description": "The fetched king's directory, "
                                             "default generators/king-control."},
                "prior_king_path": {"type": "string",
                                    "description": "Optional dethroned king, for "
                                                   "direction only (n=1, weak)."},
                "n_series": {"type": "integer", "minimum": 8},
                "seed": {"type": "integer"},
                "claims": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Properties you believe the corpus adds, e.g. "
                                   "['regime+', 'seasonality-']. A failing claim "
                                   "means the code does not do what you think.",
                },
            }},
            self._screen_candidate,
        )
        self._tool(
            "get_improve_request",
            "The pending improvement request published by the controller's "
            "hermes-native hook (runs/improve-request.json), if any.",
            no_args, self._get_improve_request,
        )
        self._tool(
            "respond_improve_request",
            "Answer the pending improvement request so the blocked hook can "
            "finish. Status `completed` makes the pass succeed.",
            {
                "properties": {
                    "status": {"type": "string",
                               "enum": list(IMPROVE_STATUSES)},
                    "detail": {"type": "string",
                               "description": "One line on what changed."},
                },
                "required": ["status"],
            },
            self._respond_improve_request,
        )
        self._tool(
            "get_brief",
            "Operating brief for an agent joining cold: the hard limits, "
            "current status, autonomy policy, and where the canonical docs "
            "live.",
            no_args, self._get_brief,
        )

    # -- tool handlers -----------------------------------------------------

    def _list_heat_entrants(self, args: dict[str, Any]) -> Any:
        receipt = read_json(self.root / status_module.STATE_FILE).get(
            "last_receipt", {})
        if not isinstance(receipt, dict) or not receipt:
            return {"round_id": None, "entrants": [], "participants": [],
                    "note": "no receipt stored yet; run the controller once"}
        heat = receipt.get("heat")
        entrants = (heat or {}).get("entrants", [])
        participants = receipt.get("participants", [])
        payload: dict[str, Any] = {
            "round_id": receipt.get("round_id"),
            "king_gen_ref": (receipt.get("summary") or {}).get("king_gen_ref"),
            "heat": heat,
            "entrants": entrants,
            "participants": participants,
        }
        if heat is None and not participants:
            payload["note"] = (
                "this receipt was stored before entrant lists were kept; "
                "the next controller poll of a new round records them"
            )
        return payload

    def _fetch_generator(self, args: dict[str, Any]) -> Any:
        ref = str(args.get("ref", "")).strip()
        if not ref:
            raise ToolError("a generator reference is required")
        out_name = str(args.get("out_name", "")).strip()
        if not out_name:
            out_name = re.sub(r"[^A-Za-z0-9._-]+", "-",
                              ref.rstrip("/").rsplit("/", 1)[-1]).strip("-.")
        if not GENERATOR_NAME.match(out_name):
            raise ToolError(
                f"cannot derive a safe directory name from {ref!r}; "
                "pass out_name matching [A-Za-z0-9][A-Za-z0-9._-]*"
            )
        destination = self.root / "generators" / out_name
        if destination.exists() and any(destination.iterdir()) and not bool(
                args.get("overwrite")):
            raise ToolError(
                f"{destination} already exists and is not empty; "
                "pass overwrite=true to replace it"
            )
        cli = self.root / ".venv/bin/cascade"
        argv = [str(cli) if cli.exists() else "cascade", "fetch", ref,
                "--out", str(destination),
                "--chain-toml", str(self.chain_toml),
                "--network", self.network]
        try:
            result = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True,
                timeout=FETCH_TIMEOUT,
            )
        except FileNotFoundError:
            raise ToolError(
                "cascade CLI not found; run setup to build the venv"
            ) from None
        except subprocess.TimeoutExpired:
            raise ToolError(f"fetch timed out after {FETCH_TIMEOUT}s") from None
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "path": str(destination),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "note": "Fetched artefacts are read-only study material and "
                    "controls; never submit someone else's generator.",
        }

    def _list_approvals(self, args: dict[str, Any]) -> Any:
        requests = read_json(self.root / status_module.APPROVALS_FILE).get(
            "requests", [])
        if not isinstance(requests, list):
            return []
        wanted = str(args.get("status", "")).strip()
        if wanted:
            requests = [r for r in requests if r.get("status") == wanted]
        return requests

    def _request_action(self, args: dict[str, Any]) -> Any:
        action = str(args.get("action", ""))
        if action not in policy_module.SUPPORTED_ACTIONS:
            raise ToolError(
                f"unknown action {action!r}; supported: "
                + ", ".join(policy_module.SUPPORTED_ACTIONS)
            )
        reason = str(args.get("reason", "")).strip()
        if not reason:
            raise ToolError("a request needs a reason the approver can act on")
        candidate = _candidate_path(self.root, args.get("candidate_path"))
        estimated = args.get("estimated_hours")
        if estimated is not None:
            estimated = float(estimated)
            if estimated <= 0:
                raise ToolError("estimated_hours must be positive")
        try:
            candidate_value = str(candidate.relative_to(self.root.resolve()))
        except ValueError:
            candidate_value = str(candidate)
        context = {
            "candidate_path": candidate_value,
            "estimated_hours": estimated,
            "requested_via": "mcp",
        }
        request = queue_approval(
            self.root / status_module.APPROVALS_FILE,
            action=action, reason=reason, context=context,
        )
        return {
            "request": request,
            "note": "Queued for approval. The controller executes it only "
                    "after a human approves (or an autonomous-mode policy "
                    "allows it). Approve with: .venv/bin/python -m "
                    f"miner.controller --approve {request['id']}",
        }

    def _get_policy(self, args: dict[str, Any]) -> Any:
        path = self.root / "policy.toml"
        if not path.is_file():
            return {
                "policy": None,
                "note": "no policy.toml; autonomous mode uses the flat "
                        "--autonomous-actions allowlist",
            }
        policy = policy_module.load_policy(path)
        events = self.root / status_module.EVENTS_FILE
        return {
            "policy": policy.describe(),
            "usage_last_24h": {
                action: policy_module.action_usage(events, action)
                for action in policy_module.SUPPORTED_ACTIONS
            },
        }

    def _log_experiment(self, args: dict[str, Any]) -> Any:
        try:
            return experiments.log_entry(
                self.root,
                hypothesis=str(args.get("hypothesis", "")),
                change=str(args.get("change", "")),
                arm=str(args.get("arm", "generators/candidate")),
                seeds=list(args.get("seeds", []) or []),
                snapshots=[str(s) for s in args.get("snapshots", []) or []],
                status=str(args.get("status", "planned")),
                noise_floor=args.get("noise_floor"),
                result=str(args.get("result", "")),
                supersedes=str(args.get("supersedes", "")),
                notes=str(args.get("notes", "")),
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def _report_issue(self, args: dict[str, Any]) -> Any:
        try:
            entry = issues.report_entry(
                self.root,
                kind=str(args.get("kind", "")),
                title=str(args.get("title", "")),
                detail=str(args.get("detail", "")),
                area=str(args.get("area", "")),
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "entry": entry,
            "note": f"filed as [{entry['id']}]; the operator sees open issues "
                    "in miner_status. Nothing executes from this.",
        }

    def _list_issues(self, args: dict[str, Any]) -> Any:
        kind = str(args.get("kind", ""))
        limit = int(args.get("limit", 0) or 0)
        if args.get("open_only"):
            entries = issues.open_entries(self.root)
            if kind:
                entries = [e for e in entries if e.get("kind") == kind]
            return entries[-limit:] if limit > 0 else entries
        return issues.list_entries(
            self.root, kind=kind, status=str(args.get("status", "")),
            limit=limit)

    def _run_quick_verify(self, args: dict[str, Any]) -> Any:
        candidate = _candidate_path(self.root, args.get("candidate_path"))
        n_series = int(args.get("n_series", 32) or 32)
        argv = [_interpreter(self.root), "scripts/quick_verify.py",
                str(candidate), "--n-series", str(n_series)]
        try:
            result = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True,
                timeout=QUICK_VERIFY_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"quick_verify timed out after {QUICK_VERIFY_TIMEOUT}s"
            ) from None
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "note": "quick_verify skips the static guard, sandbox, and "
                    "packaging checks; run full `cascade verify` before any "
                    "submission request.",
        }

    def _screen_candidate(self, args: dict[str, Any]) -> Any:
        candidate = _candidate_path(self.root, args.get("candidate_path"))
        king = _candidate_path(self.root, args.get("king_path")
                               or "generators/king-control")
        if not king.is_dir():
            raise ToolError(
                f"{king} does not exist; fetch the king first with "
                "fetch_generator(ref=<summary.king_gen_ref>, out_name='king-control')"
            )
        report_path = (self.root / SCREEN_DIR /
                       f"{candidate.name}--vs--{king.name}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [_interpreter(self.root), "-m", "miner.screen", str(candidate),
                "--king", str(king),
                "--n-series", str(int(args.get("n_series", 256) or 256)),
                "--seed", str(int(args.get("seed", 0) or 0)),
                "--json", str(report_path)]
        prior = args.get("prior_king_path")
        if prior:
            argv += ["--prior-king", str(_candidate_path(self.root, prior))]
        for claim in args.get("claims") or []:
            argv += ["--claim", str(claim)]
        try:
            result = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True,
                timeout=SCREEN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"screen timed out after {SCREEN_TIMEOUT}s") from None
        report = read_json(report_path) if report_path.is_file() else {}
        if not report:
            raise ToolError(
                f"screen exited {result.returncode} without a report: "
                f"{(result.stderr or result.stdout)[-1000:]}"
            )
        return {
            "verdict": report.get("verdict"),
            "exit_code": result.returncode,
            "report_path": str(report_path.relative_to(self.root)),
            "report": report,
            "note": "A `measurable` verdict licenses a paired GPU eval; it does "
                    "not predict the duel. Only miner.evaluate + miner.analyze "
                    "on a same-batch control can.",
        }

    def _get_improve_request(self, args: dict[str, Any]) -> Any:
        request = read_json(self.root / IMPROVE_REQUEST)
        if not request.get("id"):
            return {"pending": False}
        return {"pending": True, "request": request}

    def _respond_improve_request(self, args: dict[str, Any]) -> Any:
        request = read_json(self.root / IMPROVE_REQUEST)
        if not request.get("id"):
            raise ToolError(f"no pending improvement request at {IMPROVE_REQUEST}")
        status = str(args.get("status", ""))
        if status not in IMPROVE_STATUSES:
            raise ToolError(
                f"status must be one of {', '.join(IMPROVE_STATUSES)}"
            )
        payload = {
            "id": request["id"],
            "status": status,
            "detail": str(args.get("detail", "")).strip(),
            "responded_at": experiments.utc_now(),
        }
        path = self.root / IMPROVE_RESPONSE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True),
                             encoding="utf-8")
        temporary.replace(path)
        return {"response": payload}

    def _get_brief(self, args: dict[str, Any]) -> Any:
        policy = self._get_policy({})
        return {
            "subnet": "cascade (netuid 91, finney): submit a purely-"
                      "algorithmic time-series generator; a fixed forecaster "
                      "is trained on your corpus and scored on a private "
                      "rotating held-out set against the reigning king.",
            "hard_limits": list(HARD_LIMITS),
            "canonical_docs": [
                "CLAUDE.md", "AGENTS.md", "notes/CONTRACT.md",
                "notes/BUDGET.md", "notes/METHOD.md", "notes/EXPERIMENTS.md",
            ],
            "status": status_module.summarize(self.root),
            "policy": policy,
        }

    # -- JSON-RPC ----------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; None means no response (notification)."""
        method = message.get("method")
        message_id = message.get("id")
        if not isinstance(method, str):
            if message_id is None:
                return None
            return {"jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32600, "message": "invalid request"}}
        if method.startswith("notifications/"):
            return None
        params = message.get("params") or {}
        try:
            if method == "initialize":
                requested = str(params.get("protocolVersion", ""))
                version = (requested if requested in KNOWN_PROTOCOL_VERSIONS
                           else KNOWN_PROTOCOL_VERSIONS[0])
                result: Any = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME,
                                   "version": SERVER_VERSION},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tools}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                return {"jsonrpc": "2.0", "id": message_id,
                        "error": {"code": -32601,
                                  "message": f"method not found: {method}"}}
        except Exception as exc:  # noqa: BLE001 - protocol errors must not kill the server
            return {"jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32603, "message": repr(exc)}}
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name", ""))
        handler = self.handlers.get(name)
        if handler is None:
            raise RuntimeError(f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        try:
            value = handler(arguments)
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}],
                    "isError": True}
        return {
            "content": [{"type": "text",
                         "text": json.dumps(value, indent=2, sort_keys=True)}],
            "isError": False,
        }


def serve(root: Path, stdin=None, stdout=None, *,
          chain_toml: Path | None = None, network: str = "finney") -> int:
    server = Server(root, chain_toml=chain_toml, network=network)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            if not isinstance(message, dict):
                continue
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--chain-toml", default="",
                        help="Chain config for fetch_generator; defaults to "
                             "CASCADE_CHAIN_TOML, then $CASCADE_DIR/chain.toml.")
    parser.add_argument("--network", default="finney")
    args = parser.parse_args(argv)
    return serve(
        args.root.resolve(),
        chain_toml=Path(args.chain_toml) if args.chain_toml else None,
        network=args.network,
    )


if __name__ == "__main__":
    raise SystemExit(main())
