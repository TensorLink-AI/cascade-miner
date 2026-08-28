"""Goal-driven mining campaign: iterate improve -> screen -> eval -> verdict until
the candidate is competitive, the budget runs out, or the loop stalls.

The controller (``miner.controller``) is event-driven: it wakes on upstream
changes and runs one improvement pass per event. That never closes the loop —
after a paid evaluation nothing computes the verdict, and the next pass's
prompt carries no measured feedback. This module is the other shape: a
campaign has a *goal* (a paired-bootstrap LCB target against a fresh king
control), a *budget* (GPU-hours, passes, an optional deadline), and a loop
that hands every improvement pass the numbers the previous pass earned.

One pass is:

1. **Improve** — run the configured agent hook with a feedback event carrying
   the goal, the remaining budget, and the previous pass's screen report or
   eval verdict. Same backends as the controller (hermes/claude/codex/native).
2. **Gate for free** — the candidate must have actually changed, and
   ``miner.screen`` must return ``measurable``. ``blocked``/``undosed``/failed
   claims cost nothing and become the next pass's feedback.
3. **Pay through the same boundary as everything else** — a `gpu_evaluation`
   approval is queued. Human mode waits for a human decision; autonomous mode
   asks the policy (a ``policy.toml`` is *required* — an uncapped autonomous
   campaign is a runaway bill) and waits out exhausted caps rather than
   exceeding them. The campaign never rents anything itself; it runs the same
   approved eval command the controller would.
4. **Verdict** — re-run ``miner.analyze`` over the saved per-window components
   (live metric, real paired cluster bootstrap) and compare the candidate's
   mean LCB against the target. Meeting it queues a ``submit_candidate``
   approval — submission itself stays human-gated — and ends the campaign.

The campaign stops on: goal met, GPU-hour budget spent, pass limit, deadline,
too many consecutive free failures (the agent is stalled), or too many failed
hooks. Every pass is durable in ``runs/campaign.json`` and the shared events
file, so a killed campaign resumes where it stopped.

Do not run this at the same time as a controller that executes approvals: both
would race to execute the same approved request. The campaign marks requests
it executes as ``running`` first, but the safe configuration is one executor.

    .venv/bin/python -m miner.campaign --mode human --max-gpu-hours 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miner import policy as policy_module
from miner.controller import (
    append_event,
    queue_approval,
    read_json,
    run_hook,
    utc_now,
    write_json,
)


def tree_digest(path: Path) -> str:
    """Digest of the candidate directory itself, whatever it is named.

    ``controller.candidate_digest`` hardcodes ``generators/candidate``; the
    campaign's no-change check must follow ``--candidate`` instead.
    """
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("notes/upstream-state.json")
STATE_NAME = Path("runs/campaign.json")
SCREEN_DIR = Path("runs/screen")
MATRIX_PATH = Path("results/matrix.json")

# How many recent pass summaries ride along in each improvement event. Enough
# to show a trend; small enough not to drown the prompt.
HISTORY_WINDOW = 6
SUBPROCESS_TAIL = 4000


def _interpreter(root: Path) -> str:
    venv = root / ".venv/bin/python"
    return str(venv) if venv.exists() else sys.executable


def default_target_lcb(root: Path) -> float:
    """The margin a duel against a FRESH king requires (``win_margin_start``).

    Since DEC-CA-0016 the dethrone margin decays with king tenure, from
    ``win_margin_start`` down to the ``win_margin_end`` floor over
    ``margin_warmup_rounds``. The campaign has no chain access to read the
    live king's tenure, so it targets the conservative fresh-king bar —
    reading ``win_margin_end`` here (as this once did, when start == end)
    would declare "goal met" at a quarter of what a fresh king demands.
    Attacking a long-tenured king, pass ``--target-lcb`` with the decayed
    value instead.
    """
    try:
        keys = json.loads((root / SNAPSHOT).read_text(encoding="utf-8"))["keys"]
        return float(keys["scoring.win_margin_start"])
    except (OSError, ValueError, KeyError):
        return 0.02


def eval_seed_count() -> int:
    raw = os.environ.get("CASCADE_EVAL_SEEDS", "0,1,2")
    return max(1, len([part for part in raw.split(",") if part.strip()]))


def default_eval_hours() -> float:
    """Estimated GPU-hours for one paired eval: seeds x two arms x train hours."""
    train_hours = float(os.environ.get("CASCADE_TRAIN_HOURS", "1") or 1)
    return round(eval_seed_count() * 2 * train_hours, 2)


@dataclass
class Config:
    root: Path
    candidate: Path
    king_name: str
    mode: str                              # "human" | "autonomous"
    improve_command: str
    eval_command: str
    target_lcb: float
    min_pairs: int
    max_gpu_hours: float
    max_passes: int
    max_free_fails: int
    max_hook_fails: int
    deadline_at: float | None              # time.time() epoch, or None
    eval_hours: float
    poll_seconds: float
    hook_timeout: int
    state_file: Path
    events_file: Path
    approvals_file: Path
    controller_state: Path
    chain_toml: Path
    policy: policy_module.Policy | None
    screen_args: tuple[str, ...] = ()

    # Time seams; tests override per instance (instance attributes are not
    # subject to the bound-method transformation class attributes get).
    sleep = staticmethod(time.sleep)
    now = staticmethod(time.time)


# -- subprocess seams (patched in tests) -------------------------------------


def run_improve(cfg: Config, event: dict[str, Any]) -> dict[str, Any]:
    return run_hook(cfg.improve_command, root=cfg.root, event=event,
                    state_file=cfg.controller_state, chain_toml=cfg.chain_toml,
                    timeout=cfg.hook_timeout)


def run_screen(cfg: Config, pass_index: int) -> dict[str, Any]:
    """Run miner.screen; the JSON report is the result, exit codes are advisory."""
    report_path = cfg.root / SCREEN_DIR / f"campaign-pass-{pass_index}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [_interpreter(cfg.root), "-m", "miner.screen", str(cfg.candidate),
            "--king", str(cfg.root / "generators" / cfg.king_name),
            "--json", str(report_path), *cfg.screen_args]
    result = subprocess.run(argv, cwd=cfg.root, capture_output=True, text=True,
                            timeout=1800)
    report = read_json(report_path)
    if not report:
        return {"verdict": "error",
                "error": (result.stderr or result.stdout)[-SUBPROCESS_TAIL:]}
    report["report_path"] = str(report_path.relative_to(cfg.root))
    return report


def run_eval(cfg: Config, approval: dict[str, Any]) -> dict[str, Any]:
    return run_hook(cfg.eval_command, root=cfg.root,
                    event={"type": "approved_action", "approval": approval},
                    state_file=cfg.controller_state, chain_toml=cfg.chain_toml,
                    timeout=24 * 3600)


def run_analyze(cfg: Config) -> dict[str, Any]:
    """Recompute the verdict from saved components with the live metric."""
    out = cfg.root / MATRIX_PATH
    argv = [_interpreter(cfg.root), "-m", "miner.analyze",
            "--scores-root", str(cfg.root / "scores"),
            "--king", cfg.king_name, "--out", str(out)]
    result = subprocess.run(argv, cwd=cfg.root, capture_output=True, text=True,
                            timeout=3600)
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout)[-SUBPROCESS_TAIL:]}
    return read_json(out)


# -- pure pieces --------------------------------------------------------------


def verdict_from_matrix(matrix: dict[str, Any], candidate_name: str,
                        target_lcb: float, min_pairs: int) -> dict[str, Any]:
    """Compare one candidate's analyze row against the campaign goal."""
    row = matrix.get(candidate_name)
    if not isinstance(row, dict):
        return {"competitive": False, "reason": (
            f"no analyze row for {candidate_name!r}; candidates present: "
            f"{sorted(k for k in matrix if isinstance(matrix[k], dict))}")}
    mean_lcb = row.get("mean_lcb")
    n_pairs = int(row.get("n_pairs") or 0)
    if mean_lcb is None or n_pairs == 0:
        return {"competitive": False, "row": row,
                "reason": "no paired seed/snapshot runs — arms were not paired"}
    verdict = {
        "competitive": bool(mean_lcb >= target_lcb and n_pairs >= min_pairs),
        "mean_lcb": mean_lcb,
        "min_lcb": row.get("min_lcb"),
        "frac_lcb_over_margin": row.get("frac_lcb_over_margin"),
        "n_pairs": n_pairs,
        "target_lcb": target_lcb,
        "min_pairs": min_pairs,
        "row": row,
    }
    if not verdict["competitive"]:
        short = []
        if mean_lcb < target_lcb:
            short.append(f"mean LCB {mean_lcb:.4f} < target {target_lcb:g}")
        if n_pairs < min_pairs:
            short.append(f"{n_pairs} pairs < required {min_pairs}")
        verdict["reason"] = "; ".join(short)
    return verdict


def improvement_event(cfg: Config, state: dict[str, Any]) -> dict[str, Any]:
    """The feedback the agent gets: goal, budget, and what the last pass earned."""
    passes = state.get("passes", [])
    return {
        "type": "campaign_pass",
        "mode": cfg.mode,
        "at": utc_now(),
        "campaign_id": state["campaign_id"],
        "pass": len(passes) + 1,
        "goal": {
            "metric": ("mean paired cluster-bootstrap LCB vs a fresh "
                       f"{cfg.king_name} control, live scoring"),
            "target_lcb": cfg.target_lcb,
            "min_pairs": cfg.min_pairs,
        },
        "budget": {
            "gpu_hours_spent": state.get("gpu_hours_spent", 0.0),
            "gpu_hours_max": cfg.max_gpu_hours,
            "passes_used": len(passes),
            "passes_max": cfg.max_passes,
        },
        "last_pass": passes[-1] if passes else None,
        "history": [
            {k: p.get(k) for k in ("pass", "outcome", "detail")}
            for p in passes[-HISTORY_WINDOW:]
        ],
        "instructions": (
            "This is a goal-driven campaign pass. Study last_pass: if the "
            "screen said blocked, fix the named gate; if undosed, raise the "
            "dose; if a claim failed, the generator does not do what you "
            "think; if an eval verdict is present, read its numbers and "
            "per-source components under scores/ before choosing the next "
            "single change. One hypothesis per pass. Do not write "
            "runs/agent-request.json — the campaign gates evaluation itself."
        ),
    }


# -- gated evaluation ---------------------------------------------------------


class CampaignStop(Exception):
    """Raised to end the campaign with a status and a reason."""

    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


def _deadline_exceeded(cfg: Config) -> bool:
    return cfg.deadline_at is not None and cfg.now() >= cfg.deadline_at


def gated_evaluation(cfg: Config, state: dict[str, Any],
                     screen_report: dict[str, Any]) -> dict[str, Any]:
    """Queue the approval, wait for the gate to open, execute, and account for it.

    Returns the hook result. Raises CampaignStop when the campaign cannot
    continue (deadline while waiting). A human rejection is returned as an
    outcome, not an exception — the loop feeds it back and keeps iterating.
    """
    context = {
        "campaign_id": state["campaign_id"],
        "pass": len(state.get("passes", [])) + 1,
        "candidate_path": str(cfg.candidate.relative_to(cfg.root)),
        "candidate_digest": tree_digest(cfg.candidate),
        "estimated_hours": cfg.eval_hours,
        "screen_verdict": screen_report.get("verdict"),
        "screen_report": screen_report.get("report_path"),
    }
    reason = (f"campaign {state['campaign_id']} pass {context['pass']}: screen "
              f"verdict {context['screen_verdict']}, moved "
              f"{screen_report.get('moved_features', [])[:4]}")
    request = queue_approval(cfg.approvals_file, action="gpu_evaluation",
                             reason=reason, context=context)
    append_event(cfg.events_file, {"type": "campaign_eval_requested",
                                   "at": utc_now(), "approval": request})

    if cfg.mode == "autonomous":
        while True:
            usage = policy_module.action_usage(cfg.events_file, "gpu_evaluation")
            decision = cfg.policy.decide("gpu_evaluation",
                                         estimated_hours=cfg.eval_hours,
                                         usage=usage)
            if decision.allowed:
                break
            if _deadline_exceeded(cfg):
                raise CampaignStop("deadline", (
                    f"deadline passed while policy capped: {decision.reason}"))
            append_event(cfg.events_file, {
                "type": "campaign_waiting", "at": utc_now(),
                "reason": decision.reason})
            cfg.sleep(max(cfg.poll_seconds, 60.0))
        # This event shape is what policy_module.action_usage counts, so the
        # campaign's spends draw down the same 24h allowance as the controller's.
        _mark_approval(cfg, request["id"], "running")
        autonomous_event = {
            "type": "autonomous_action", "mode": "autonomous", "at": utc_now(),
            "action": "gpu_evaluation", "reason": reason, "context": context,
        }
        result = run_eval(cfg, request)
        autonomous_event["result"] = {k: result[k] for k in ("exit_code",)}
        append_event(cfg.events_file, autonomous_event)
    else:
        request = _wait_for_human(cfg, request["id"])
        if request.get("status") == "rejected":
            return {"exit_code": -1, "rejected": True,
                    "reason": "operator rejected the evaluation request"}
        _mark_approval(cfg, request["id"], "running")
        result = run_eval(cfg, request)

    final = "completed" if result["exit_code"] == 0 else "failed"
    _mark_approval(cfg, request["id"], final)
    if result["exit_code"] == 0:
        state["gpu_hours_spent"] = round(
            float(state.get("gpu_hours_spent", 0.0)) + cfg.eval_hours, 3)
    return result


def _mark_approval(cfg: Config, request_id: str, status: str) -> None:
    doc = read_json(cfg.approvals_file)
    requests = doc.get("requests", [])
    for request in requests:
        if request.get("id") == request_id:
            request["status"] = status
            request["updated_at"] = utc_now()
    write_json(cfg.approvals_file, {"requests": requests})


def _wait_for_human(cfg: Config, request_id: str) -> dict[str, Any]:
    """Block until the operator approves or rejects the queued request."""
    announced = False
    while True:
        doc = read_json(cfg.approvals_file)
        for request in doc.get("requests", []):
            if request.get("id") != request_id:
                continue
            status = request.get("status")
            if status in ("approved", "rejected"):
                return request
        if not announced:
            print(f"waiting for approval of {request_id} — decide with: "
                  f"python -m miner.controller --approve {request_id} "
                  f"(or --reject)", flush=True)
            announced = True
        if _deadline_exceeded(cfg):
            raise CampaignStop(
                "deadline", f"deadline passed awaiting approval {request_id}")
        cfg.sleep(max(cfg.poll_seconds, 1.0))


# -- the loop -----------------------------------------------------------------


def current_king_ref(cfg: Config) -> str:
    receipt = read_json(cfg.controller_state).get("last_receipt", {})
    return str((receipt.get("summary") or {}).get("king_gen_ref") or "")


def load_state(cfg: Config) -> dict[str, Any]:
    state = read_json(cfg.state_file)
    if state.get("campaign_id") and state.get("status") == "running":
        return state                        # resume a killed campaign
    state = {
        "campaign_id": f"campaign-{int(cfg.now())}",
        "status": "running",
        "started_at": utc_now(),
        "goal": {"target_lcb": cfg.target_lcb, "min_pairs": cfg.min_pairs,
                 "king": cfg.king_name},
        "king_ref": current_king_ref(cfg),
        "gpu_hours_spent": 0.0,
        "passes": [],
    }
    write_json(cfg.state_file, state)
    return state


def record_pass(cfg: Config, state: dict[str, Any], entry: dict[str, Any]) -> None:
    entry.setdefault("at", utc_now())
    entry["pass"] = len(state["passes"]) + 1
    state["passes"].append(entry)
    write_json(cfg.state_file, state)
    append_event(cfg.events_file, {"type": "campaign_pass_result", **entry,
                                   "campaign_id": state["campaign_id"]})
    print(f"[pass {entry['pass']}] {entry['outcome']}: {entry['detail']}",
          flush=True)


def check_budget(cfg: Config, state: dict[str, Any]) -> None:
    started_against = str(state.get("king_ref") or "")
    reigning = current_king_ref(cfg)
    if started_against and reigning and reigning != started_against:
        # The throne changed under us: every verdict so far compared against a
        # dethroned king, and the local king-control no longer matches reality.
        raise CampaignStop("king_changed", (
            f"the reigning king moved from {started_against} to {reigning}; "
            f"re-fetch generators/{cfg.king_name} and start a new campaign — "
            "prior verdicts are against a dethroned control"))
    if len(state["passes"]) >= cfg.max_passes:
        raise CampaignStop("budget", f"pass limit {cfg.max_passes} reached")
    if state.get("gpu_hours_spent", 0.0) + cfg.eval_hours > cfg.max_gpu_hours:
        raise CampaignStop("budget", (
            f"{state['gpu_hours_spent']:g}h spent of {cfg.max_gpu_hours:g}h; "
            f"another eval (~{cfg.eval_hours:g}h) would exceed the budget"))
    if _deadline_exceeded(cfg):
        raise CampaignStop("deadline", "campaign deadline passed")
    recent = [p["outcome"] for p in state["passes"]]
    tail_free = 0
    for outcome in reversed(recent):
        if outcome in ("screen_blocked", "screen_undosed", "screen_error",
                       "no_change"):
            tail_free += 1
        else:
            break
    if tail_free >= cfg.max_free_fails:
        raise CampaignStop("stalled", (
            f"{tail_free} consecutive passes produced nothing measurable; "
            "the improvement loop is not converging — a human should look"))
    tail_hooks = 0
    for outcome in reversed(recent):
        if outcome in ("improve_failed", "eval_failed"):
            tail_hooks += 1
        else:
            break
    if tail_hooks >= cfg.max_hook_fails:
        raise CampaignStop("hook_failure", (
            f"{tail_hooks} consecutive hook failures; fix the environment "
            "before burning more passes"))


def run_pass(cfg: Config, state: dict[str, Any]) -> None:
    before = tree_digest(cfg.candidate)
    improve = run_improve(cfg, improvement_event(cfg, state))
    if improve["exit_code"] != 0:
        record_pass(cfg, state, {
            "outcome": "improve_failed",
            "detail": f"improve hook exit {improve['exit_code']}",
            "stderr": improve["stderr"][-1000:]})
        return
    if tree_digest(cfg.candidate) == before:
        record_pass(cfg, state, {
            "outcome": "no_change",
            "detail": "improve pass left the candidate byte-identical; "
                      "there is nothing new to measure"})
        return

    report = run_screen(cfg, len(state["passes"]) + 1)
    verdict = report.get("verdict")
    if verdict != "measurable":
        outcome = {"blocked": "screen_blocked", "undosed": "screen_undosed"}.get(
            verdict, "screen_error")
        record_pass(cfg, state, {
            "outcome": outcome,
            "detail": report.get("verdict_meaning") or report.get("error", ""),
            "screen": _screen_digest(report)})
        return
    failed_claims = [c["claim"] for c in report.get("claims", [])
                     if not c.get("passed")]
    if failed_claims:
        record_pass(cfg, state, {
            "outcome": "screen_blocked",
            "detail": f"corpus does not carry claimed properties: {failed_claims}",
            "screen": _screen_digest(report)})
        return

    result = gated_evaluation(cfg, state, report)
    if result.get("rejected"):
        record_pass(cfg, state, {
            "outcome": "eval_rejected",
            "detail": result["reason"], "screen": _screen_digest(report)})
        return
    if result["exit_code"] != 0:
        record_pass(cfg, state, {
            "outcome": "eval_failed",
            "detail": f"evaluation exit {result['exit_code']}",
            "stderr": result["stderr"][-1000:]})
        return

    matrix = run_analyze(cfg)
    if "error" in matrix:
        record_pass(cfg, state, {
            "outcome": "eval_failed",
            "detail": f"analyze failed: {matrix['error'][:500]}"})
        return
    verdict = verdict_from_matrix(matrix, cfg.candidate.name,
                                  cfg.target_lcb, cfg.min_pairs)
    if verdict["competitive"]:
        request = queue_approval(
            cfg.approvals_file, action="submit_candidate",
            reason=(f"campaign {state['campaign_id']}: mean LCB "
                    f"{verdict['mean_lcb']:.4f} >= target {cfg.target_lcb:g} "
                    f"over {verdict['n_pairs']} pairs"),
            context={"campaign_id": state["campaign_id"],
                     "candidate_path": str(cfg.candidate.relative_to(cfg.root)),
                     "candidate_digest": tree_digest(cfg.candidate),
                     "verdict": {k: verdict[k] for k in
                                 ("mean_lcb", "min_lcb", "n_pairs",
                                  "frac_lcb_over_margin", "target_lcb")}})
        record_pass(cfg, state, {
            "outcome": "goal_met",
            "detail": (f"mean LCB {verdict['mean_lcb']:.4f} over "
                       f"{verdict['n_pairs']} pairs beats target "
                       f"{cfg.target_lcb:g}; submission queued as "
                       f"{request['id']} (human-gated)"),
            "verdict": verdict})
        raise CampaignStop("goal_met", "candidate is competitive locally; "
                           "submission awaits human approval")
    record_pass(cfg, state, {
        "outcome": "eval_short",
        "detail": verdict.get("reason", "below target"),
        "verdict": verdict, "screen": _screen_digest(report)})


def _screen_digest(report: dict[str, Any]) -> dict[str, Any]:
    """The screen fields worth carrying into feedback, without the full table."""
    return {
        "verdict": report.get("verdict"),
        "moved_features": report.get("moved_features", [])[:8],
        "coverage": report.get("coverage", {}),
        "flags": report.get("flags", []),
        "challenges": report.get("challenges", []),
        "claims": report.get("claims", []),
        "report_path": report.get("report_path"),
    }


def run_campaign(cfg: Config) -> dict[str, Any]:
    state = load_state(cfg)
    try:
        while True:
            check_budget(cfg, state)
            run_pass(cfg, state)
    except CampaignStop as stop:
        state["status"] = stop.status
        state["stop_reason"] = stop.reason
        state["finished_at"] = utc_now()
        write_json(cfg.state_file, state)
        append_event(cfg.events_file, {
            "type": "campaign_finished", "at": utc_now(),
            "campaign_id": state["campaign_id"], "status": stop.status,
            "reason": stop.reason,
            "passes": len(state["passes"]),
            "gpu_hours_spent": state.get("gpu_hours_spent", 0.0)})
        print(f"campaign {state['campaign_id']} finished: {stop.status} — "
              f"{stop.reason}", flush=True)
    return state


# -- CLI ----------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> Config:
    root = args.root.resolve()
    candidate = (root / args.candidate).resolve()
    if not (candidate / "generator.py").is_file():
        raise SystemExit(f"{candidate} has no generator.py")
    king_dir = root / "generators" / args.king
    if not king_dir.is_dir():
        raise SystemExit(
            f"{king_dir} does not exist. Fetch the reigning king first:\n"
            "  KING_REF=$(python -c 'import json; print(json.load(open("
            '"runs/controller-state.json"))["last_receipt"]["summary"]'
            '["king_gen_ref"])\')\n'
            f"  cascade fetch \"$KING_REF\" --out generators/{args.king} ...")

    policy = None
    if args.mode == "autonomous":
        policy_file = args.policy_file or (root / "policy.toml")
        if not Path(policy_file).is_file():
            raise SystemExit(
                "autonomous mode requires a policy file with per-day caps "
                f"(looked for {policy_file}); an uncapped autonomous campaign "
                "is a runaway bill. Copy policy.example.toml and review it.")
        policy = policy_module.load_policy(Path(policy_file))
        decision = policy.decide("gpu_evaluation", estimated_hours=args.eval_hours)
        if not decision.allowed and "not autonomous" in decision.reason:
            raise SystemExit(f"policy refuses autonomy: {decision.reason}")

    improve_command = args.improve_command or os.environ.get(
        "CASCADE_IMPROVE_COMMAND",
        f"{_interpreter(root)} scripts/improve_candidate.py")
    eval_command = args.eval_command or os.environ.get(
        "CASCADE_APPROVED_EVAL_COMMAND", "")
    if not eval_command:
        raise SystemExit(
            "no evaluation command: pass --eval-command or set "
            "CASCADE_APPROVED_EVAL_COMMAND (scripts/run-gpu-evaluation)")

    chain_toml = args.chain_toml or Path(
        os.environ.get("CASCADE_DIR", "/root/cascade")) / "chain.toml"
    deadline_at = (time.time() + args.deadline_hours * 3600
                   if args.deadline_hours else None)
    screen_args = tuple(shlex.split(args.screen_args)) if args.screen_args else ()
    return Config(
        root=root, candidate=candidate, king_name=args.king, mode=args.mode,
        improve_command=improve_command, eval_command=eval_command,
        target_lcb=(args.target_lcb if args.target_lcb is not None
                    else default_target_lcb(root)),
        min_pairs=args.min_pairs, max_gpu_hours=args.max_gpu_hours,
        max_passes=args.max_passes, max_free_fails=args.max_free_fails,
        max_hook_fails=args.max_hook_fails, deadline_at=deadline_at,
        eval_hours=args.eval_hours, poll_seconds=args.poll_seconds,
        hook_timeout=args.hook_timeout,
        state_file=root / STATE_NAME,
        events_file=root / "runs/controller-events.jsonl",
        approvals_file=root / "runs/approvals.json",
        controller_state=root / "runs/controller-state.json",
        chain_toml=chain_toml, policy=policy, screen_args=screen_args,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidate", default="generators/candidate")
    parser.add_argument("--king", default="king-control",
                        help="directory name under generators/ of the fetched king")
    parser.add_argument("--mode", choices=("human", "autonomous"), default="human")
    parser.add_argument("--target-lcb", type=float, default=None,
                        help="goal: mean paired-bootstrap LCB; default is the "
                             "fresh-king win_margin_start from "
                             "notes/upstream-state.json (the margin decays to "
                             "win_margin_end over margin_warmup_rounds of king "
                             "tenure — pass the decayed value to target a "
                             "long-tenured king)")
    parser.add_argument("--min-pairs", type=int, default=3,
                        help="paired (seed, snapshot) runs required for a verdict")
    parser.add_argument("--max-gpu-hours", type=float, default=12.0)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--max-free-fails", type=int, default=3,
                        help="consecutive unmeasurable passes before stopping")
    parser.add_argument("--max-hook-fails", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=default_eval_hours(),
                        help="estimated GPU-hours per paired eval (budget unit)")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--hook-timeout", type=int, default=7200)
    parser.add_argument("--improve-command", default="")
    parser.add_argument("--eval-command", default="")
    parser.add_argument("--policy-file", type=Path, default=None)
    parser.add_argument("--chain-toml", type=Path, default=None)
    parser.add_argument("--screen-args", default="",
                        help="extra arguments for miner.screen, e.g. "
                             "'--n-series 512 --claim regime+'")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    state = run_campaign(cfg)
    return 0 if state.get("status") == "goal_met" else 1


if __name__ == "__main__":
    raise SystemExit(main())
