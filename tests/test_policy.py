import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner import controller as controller_module
from miner import policy
from miner.controller import Controller


def make_controller(root: Path, **overrides) -> Controller:
    values = {
        "root": root,
        "cascade_dir": root / "cascade",
        "chain_toml": root / "cascade/chain.toml",
        "cascade_python": Path(sys.executable),
        "state_file": root / "runs/state.json",
        "events_file": root / "runs/events.jsonl",
        "approvals_file": root / "runs/approvals.json",
        "eval_pool_dir": root / "pools",
    }
    values.update(overrides)
    return Controller(**values)


def write_policy(root: Path, text: str) -> Path:
    path = root / "policy.toml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = """
version = 1

[actions.gpu_evaluation]
autonomous = true
max_runs_per_day = 2
max_hours_per_run = 2
max_hours_per_day = 3

[actions.submit_candidate]
autonomous = false
"""


class PolicyLoadTests(TestCase):
    def test_valid_policy_loads_with_defaults_for_missing_actions(self):
        with TemporaryDirectory() as directory:
            loaded = policy.load_policy(write_policy(Path(directory), VALID))
        self.assertEqual(loaded.autonomous_actions(), ("gpu_evaluation",))
        self.assertTrue(loaded.action("gpu_evaluation").autonomous)
        self.assertFalse(loaded.action("create_hotkey").autonomous)
        self.assertIsNone(loaded.action("create_hotkey").max_runs_per_day)

    def test_unknown_action_name_fails_loudly(self):
        with TemporaryDirectory() as directory:
            path = write_policy(Path(directory),
                                "[actions.gpu_evaluatoin]\nautonomous = true\n")
            with self.assertRaisesRegex(RuntimeError, "gpu_evaluatoin"):
                policy.load_policy(path)

    def test_unknown_key_fails_loudly(self):
        with TemporaryDirectory() as directory:
            path = write_policy(
                Path(directory),
                "[actions.gpu_evaluation]\nautonomous = true\nmax_usd = 5\n",
            )
            with self.assertRaisesRegex(RuntimeError, "max_usd"):
                policy.load_policy(path)

    def test_caps_must_be_positive_numbers(self):
        with TemporaryDirectory() as directory:
            path = write_policy(
                Path(directory),
                "[actions.gpu_evaluation]\nautonomous = true\n"
                "max_runs_per_day = 0\n",
            )
            with self.assertRaisesRegex(RuntimeError, "max_runs_per_day"):
                policy.load_policy(path)

    def test_autonomous_must_be_boolean(self):
        with TemporaryDirectory() as directory:
            path = write_policy(Path(directory),
                                "[actions.gpu_evaluation]\nautonomous = 'yes'\n")
            with self.assertRaisesRegex(RuntimeError, "boolean"):
                policy.load_policy(path)

    def test_invalid_toml_names_the_file(self):
        with TemporaryDirectory() as directory:
            path = write_policy(Path(directory), "[actions\n")
            with self.assertRaisesRegex(RuntimeError, "not valid TOML"):
                policy.load_policy(path)


class PolicyDecisionTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.policy = policy.load_policy(
            write_policy(Path(self.directory.name), VALID))

    def test_non_autonomous_action_is_declined(self):
        decision = self.policy.decide("submit_candidate")
        self.assertFalse(decision.allowed)
        self.assertIn("not autonomous", decision.reason)

    def test_unlisted_action_is_declined(self):
        self.assertFalse(self.policy.decide("create_hotkey").allowed)

    def test_within_all_caps_is_allowed(self):
        decision = self.policy.decide(
            "gpu_evaluation", estimated_hours=1,
            usage={"runs": 1, "hours": 1.0},
        )
        self.assertTrue(decision.allowed)

    def test_single_run_over_hours_cap_is_declined(self):
        decision = self.policy.decide("gpu_evaluation", estimated_hours=3)
        self.assertFalse(decision.allowed)
        self.assertIn("max_hours_per_run", decision.reason)

    def test_runs_per_day_cap_is_declined(self):
        decision = self.policy.decide(
            "gpu_evaluation", estimated_hours=1,
            usage={"runs": 2, "hours": 2.0},
        )
        self.assertFalse(decision.allowed)
        self.assertIn("max_runs_per_day", decision.reason)

    def test_hours_per_day_cap_is_declined(self):
        decision = self.policy.decide(
            "gpu_evaluation", estimated_hours=2,
            usage={"runs": 1, "hours": 1.5},
        )
        self.assertFalse(decision.allowed)
        self.assertIn("max_hours_per_day", decision.reason)

    def test_missing_estimate_counts_as_default_hours(self):
        decision = self.policy.decide(
            "gpu_evaluation", usage={"runs": 0, "hours": 2.5},
        )
        self.assertFalse(decision.allowed)
        self.assertIn("max_hours_per_day", decision.reason)


class UsageAccountingTests(TestCase):
    def _events(self, root: Path, events: list[dict]) -> Path:
        path = root / "runs/events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )
        return path

    def test_only_recent_autonomous_actions_count(self):
        now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as directory:
            events = self._events(Path(directory), [
                {"type": "autonomous_action", "action": "gpu_evaluation",
                 "at": (now - timedelta(hours=2)).isoformat(),
                 "context": {"estimated_hours": 2}},
                # Outside the 24h window: must not count.
                {"type": "autonomous_action", "action": "gpu_evaluation",
                 "at": (now - timedelta(hours=30)).isoformat(),
                 "context": {"estimated_hours": 4}},
                # Human-approved run: a human decision, not autonomous spend.
                {"type": "approval_result", "action": "gpu_evaluation",
                 "at": (now - timedelta(hours=1)).isoformat()},
                # Different action: separate allowance.
                {"type": "autonomous_action", "action": "submit_candidate",
                 "at": (now - timedelta(hours=1)).isoformat()},
                # No estimate: counted at the default hours.
                {"type": "autonomous_action", "action": "gpu_evaluation",
                 "at": (now - timedelta(hours=3)).isoformat(), "context": {}},
            ])
            with events.open("a", encoding="utf-8") as fh:
                fh.write("not json at all\n")
            usage = policy.action_usage(events, "gpu_evaluation", now=now)
        self.assertEqual(usage["runs"], 2)
        self.assertEqual(usage["hours"], 2 + policy.DEFAULT_HOURS_PER_RUN)

    def test_missing_events_file_is_zero_usage(self):
        with TemporaryDirectory() as directory:
            usage = policy.action_usage(
                Path(directory) / "runs/events.jsonl", "gpu_evaluation")
        self.assertEqual(usage, {"runs": 0, "hours": 0.0})


class ControllerGateTests(TestCase):
    def test_without_policy_the_flat_allowlist_decides(self):
        with TemporaryDirectory() as directory:
            controller = make_controller(
                Path(directory), autonomous_actions=("gpu_evaluation",))
            self.assertTrue(
                controller.autonomy_gate("gpu_evaluation", {}).allowed)
            declined = controller.autonomy_gate("submit_candidate", {})
        self.assertFalse(declined.allowed)
        self.assertIn("--autonomous-actions", declined.reason)

    def test_policy_overrides_the_flat_allowlist(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = policy.load_policy(write_policy(root, VALID))
            # The allowlist claims submit_candidate; the policy must win.
            controller = make_controller(
                root, policy=loaded,
                autonomous_actions=("gpu_evaluation", "submit_candidate"),
            )
            self.assertFalse(
                controller.autonomy_gate("submit_candidate", {}).allowed)
            self.assertTrue(
                controller.autonomy_gate(
                    "gpu_evaluation", {"estimated_hours": 1}).allowed)

    def test_policy_gate_reads_usage_from_the_events_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = policy.load_policy(write_policy(root, VALID))
            controller = make_controller(root, policy=loaded)
            events = root / "runs/events.jsonl"
            events.parent.mkdir(parents=True)
            now = datetime.now(timezone.utc)
            events.write_text("\n".join(
                json.dumps({
                    "type": "autonomous_action", "action": "gpu_evaluation",
                    "at": (now - timedelta(hours=1)).isoformat(),
                    "context": {"estimated_hours": 1},
                }) for _ in range(2)) + "\n", encoding="utf-8")
            declined = controller.autonomy_gate(
                "gpu_evaluation", {"estimated_hours": 1})
        self.assertFalse(declined.allowed)
        self.assertIn("max_runs_per_day", declined.reason)

    def test_parser_accepts_policy_file(self):
        args = controller_module.build_parser().parse_args(
            ["--policy-file", "policy.toml"])
        self.assertEqual(str(args.policy_file), "policy.toml")
