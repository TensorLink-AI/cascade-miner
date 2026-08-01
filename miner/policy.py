"""Operator-owned autonomy policy for privileged controller actions.

``--autonomous-actions`` is a coarse on/off switch. A policy file bounds *how
much* an allowlisted action may do without a human: runs per day, hours per
run, hours per day. The controller consults the policy at the moment it would
execute an autonomous action; anything the policy declines degrades to an
ordinary approval request rather than an error, so raising autonomy never
removes the human fallback.

The file is TOML, reviewed and owned by the operator like the ``ops/``
wrappers — agents may read it (``get_policy`` over MCP) but have no reason to
write it, and the controller only ever reads it at startup.

    [actions.gpu_evaluation]
    autonomous = true
    max_runs_per_day = 4
    max_hours_per_run = 2
    max_hours_per_day = 6

    [actions.submit_candidate]
    autonomous = false

An action with no section is never autonomous. Unknown action names or keys
fail loading loudly — a typo must not silently widen or narrow the policy.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SUPPORTED_ACTIONS = (
    "gpu_evaluation", "create_hotkey", "register_hotkey", "submit_candidate",
)
ACTION_KEYS = frozenset(
    {"autonomous", "max_runs_per_day", "max_hours_per_run", "max_hours_per_day"}
)
# A request without estimated_hours still counts against the hour caps.
DEFAULT_HOURS_PER_RUN = 1.0
USAGE_WINDOW_HOURS = 24


@dataclass(frozen=True)
class ActionPolicy:
    autonomous: bool = False
    max_runs_per_day: int | None = None
    max_hours_per_run: float | None = None
    max_hours_per_day: float | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class Policy:
    source: str = ""
    actions: dict[str, ActionPolicy] = field(default_factory=dict)

    def action(self, name: str) -> ActionPolicy:
        return self.actions.get(name, ActionPolicy())

    def autonomous_actions(self) -> tuple[str, ...]:
        return tuple(
            name for name in SUPPORTED_ACTIONS if self.action(name).autonomous
        )

    def decide(self, action: str, *, estimated_hours: float | None = None,
               usage: dict[str, float] | None = None) -> Decision:
        """Would the policy run this action autonomously right now?

        ``usage`` is the trailing-window consumption from :func:`action_usage`.
        Declining is never an error: the controller turns a declined decision
        into a pending approval, so a cap pauses spending instead of dropping
        the request.
        """
        rule = self.action(action)
        if not rule.autonomous:
            return Decision(False, f"{action} is not autonomous in {self.source}")
        hours = DEFAULT_HOURS_PER_RUN if estimated_hours is None else float(estimated_hours)
        if rule.max_hours_per_run is not None and hours > rule.max_hours_per_run:
            return Decision(False, (
                f"{action} estimated {hours:g}h exceeds "
                f"max_hours_per_run={rule.max_hours_per_run:g}"
            ))
        spent = usage or {"runs": 0, "hours": 0.0}
        if rule.max_runs_per_day is not None and spent["runs"] + 1 > rule.max_runs_per_day:
            return Decision(False, (
                f"{action} already ran {spent['runs']:g} times in the last "
                f"{USAGE_WINDOW_HOURS}h (max_runs_per_day={rule.max_runs_per_day})"
            ))
        if rule.max_hours_per_day is not None and spent["hours"] + hours > rule.max_hours_per_day:
            return Decision(False, (
                f"{action} would reach {spent['hours'] + hours:g}h in the last "
                f"{USAGE_WINDOW_HOURS}h (max_hours_per_day={rule.max_hours_per_day:g})"
            ))
        return Decision(True, f"within policy {self.source}")

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "actions": {
                name: {
                    "autonomous": rule.autonomous,
                    "max_runs_per_day": rule.max_runs_per_day,
                    "max_hours_per_run": rule.max_hours_per_run,
                    "max_hours_per_day": rule.max_hours_per_day,
                }
                for name, rule in sorted(self.actions.items())
            },
        }


def _positive(name: str, value: Any, kind: type) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"policy key {name} must be a positive number")
    if value <= 0:
        raise RuntimeError(f"policy key {name} must be a positive number")
    return kind(value)


def load_policy(path: Path) -> Policy:
    """Load and validate a policy file, failing loudly on anything unknown."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read policy file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"policy file {path} is not valid TOML: {exc}") from exc
    known_top = {"version", "actions"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise RuntimeError(
            f"policy file {path} has unknown keys: {', '.join(sorted(unknown_top))}"
        )
    sections = raw.get("actions", {})
    if not isinstance(sections, dict):
        raise RuntimeError(f"policy file {path}: [actions] must be a table")
    actions: dict[str, ActionPolicy] = {}
    for name, section in sections.items():
        if name not in SUPPORTED_ACTIONS:
            raise RuntimeError(
                f"policy file {path} names unknown action {name!r}; "
                f"supported: {', '.join(SUPPORTED_ACTIONS)}"
            )
        if not isinstance(section, dict):
            raise RuntimeError(f"policy file {path}: [actions.{name}] must be a table")
        unknown = set(section) - ACTION_KEYS
        if unknown:
            raise RuntimeError(
                f"policy file {path}: [actions.{name}] has unknown keys: "
                + ", ".join(sorted(unknown))
            )
        autonomous = section.get("autonomous", False)
        if not isinstance(autonomous, bool):
            raise RuntimeError(
                f"policy file {path}: actions.{name}.autonomous must be a boolean"
            )
        actions[name] = ActionPolicy(
            autonomous=autonomous,
            max_runs_per_day=(
                _positive(f"actions.{name}.max_runs_per_day",
                          section["max_runs_per_day"], int)
                if "max_runs_per_day" in section else None
            ),
            max_hours_per_run=(
                _positive(f"actions.{name}.max_hours_per_run",
                          section["max_hours_per_run"], float)
                if "max_hours_per_run" in section else None
            ),
            max_hours_per_day=(
                _positive(f"actions.{name}.max_hours_per_day",
                          section["max_hours_per_day"], float)
                if "max_hours_per_day" in section else None
            ),
        )
    return Policy(source=str(path), actions=actions)


def action_usage(events_file: Path, action: str, *,
                 now: datetime | None = None,
                 window_hours: int = USAGE_WINDOW_HOURS) -> dict[str, float]:
    """Autonomous runs and estimated hours spent on ``action`` in the window.

    Only ``autonomous_action`` events count: a human-approved run was a human
    decision and must not consume the autonomous allowance.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(hours=window_hours)
    runs, hours = 0, 0.0
    if not events_file.exists():
        return {"runs": runs, "hours": hours}
    for line in events_file.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "autonomous_action":
            continue
        if str(event.get("action", "")) != action:
            continue
        try:
            at = datetime.fromisoformat(str(event.get("at", "")))
        except ValueError:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if at < cutoff:
            continue
        runs += 1
        estimate = (event.get("context") or {}).get("estimated_hours")
        try:
            hours += float(estimate) if estimate is not None else DEFAULT_HOURS_PER_RUN
        except (TypeError, ValueError):
            hours += DEFAULT_HOURS_PER_RUN
    return {"runs": runs, "hours": hours}
