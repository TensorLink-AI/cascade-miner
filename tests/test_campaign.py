"""Offline tests for the goal-driven campaign loop.

The improve, screen, eval, and analyze steps are module seams patched with
fakes, so the suite never spawns an agent, imports cascade, or rents anything.
What is under test is the loop itself: gating, budgets, feedback, approval
handling, stop conditions, and durability.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import campaign, policy as policy_module
from miner.controller import read_json


POLICY = """
version = 1
[actions.gpu_evaluation]
autonomous = true
max_runs_per_day = 10
max_hours_per_day = 100
"""


class FakeSteps:
    """Scriptable stand-ins for the four subprocess seams."""

    def __init__(self, root: Path):
        self.root = root
        self.improve_outcomes: list[dict] = []
        self.screen_outcomes: list[dict] = []
        self.eval_outcomes: list[dict] = []
        self.analyze_outcomes: list[dict] = []
        self.improve_events: list[dict] = []
        self.evals_run = 0

    def improve(self, cfg, event):
        self.improve_events.append(event)
        outcome = self.improve_outcomes.pop(0) if self.improve_outcomes else {}
        if outcome.get("touch", True):
            marker = cfg.candidate / "generator.py"
            marker.write_text(marker.read_text() + f"\n# pass {len(self.improve_events)}\n")
        return {"command": "fake-improve", "exit_code": outcome.get("exit_code", 0),
                "stdout": "", "stderr": outcome.get("stderr", "")}

    def screen(self, cfg, pass_index):
        if self.screen_outcomes:
            return self.screen_outcomes.pop(0)
        return {"verdict": "measurable", "moved_features": ["level_shift"],
                "claims": [], "coverage": {}, "flags": [], "challenges": [],
                "report_path": f"runs/screen/campaign-pass-{pass_index}.json"}

    def evaluate(self, cfg, approval):
        self.evals_run += 1
        outcome = self.eval_outcomes.pop(0) if self.eval_outcomes else {}
        return {"command": "fake-eval", "exit_code": outcome.get("exit_code", 0),
                "stdout": "", "stderr": outcome.get("stderr", "")}

    def analyze(self, cfg):
        if self.analyze_outcomes:
            return self.analyze_outcomes.pop(0)
        return {"candidate": {"mean_lcb": -0.01, "min_lcb": -0.02,
                              "frac_lcb_over_margin": 0.0, "n_pairs": 6,
                              "mean_geomean": 1.0}}


class CampaignHarness(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        candidate = self.root / "generators/candidate"
        candidate.mkdir(parents=True)
        (candidate / "generator.py").write_text("# starter\n")
        (self.root / "generators/king-control").mkdir()
        (self.root / "runs").mkdir()
        self.steps = FakeSteps(self.root)
        for name, attr in (("run_improve", "improve"), ("run_screen", "screen"),
                           ("run_eval", "evaluate"), ("run_analyze", "analyze")):
            patcher = patch.object(campaign, name, getattr(self.steps, attr))
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, **overrides) -> campaign.Config:
        policy_path = self.root / "policy.toml"
        policy_path.write_text(POLICY)
        values = {
            "root": self.root,
            "candidate": self.root / "generators/candidate",
            "king_name": "king-control",
            "mode": "autonomous",
            "improve_command": "fake-improve",
            "eval_command": "fake-eval",
            "target_lcb": 0.02,
            "min_pairs": 3,
            "max_gpu_hours": 30.0,
            "max_passes": 10,
            "max_free_fails": 3,
            "max_hook_fails": 2,
            "deadline_at": None,
            "eval_hours": 6.0,
            "poll_seconds": 0.0,
            "hook_timeout": 60,
            "state_file": self.root / "runs/campaign.json",
            "events_file": self.root / "runs/controller-events.jsonl",
            "approvals_file": self.root / "runs/approvals.json",
            "controller_state": self.root / "runs/controller-state.json",
            "chain_toml": self.root / "chain.toml",
            "policy": policy_module.load_policy(policy_path),
        }
        values.update(overrides)
        cfg = campaign.Config(**values)
        cfg.sleep = lambda seconds: None
        return cfg

    def events(self) -> list[dict]:
        path = self.root / "runs/controller-events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]


class GoalTests(CampaignHarness):
    def test_campaign_stops_when_the_candidate_beats_the_target(self):
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": -0.01, "n_pairs": 6, "min_lcb": -0.02,
                           "frac_lcb_over_margin": 0.0}},
            {"candidate": {"mean_lcb": 0.031, "n_pairs": 6, "min_lcb": 0.021,
                           "frac_lcb_over_margin": 1.0}},
        ]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["status"], "goal_met")
        self.assertEqual(len(state["passes"]), 2)
        self.assertEqual(state["passes"][0]["outcome"], "eval_short")
        self.assertEqual(state["passes"][1]["outcome"], "goal_met")
        self.assertEqual(state["gpu_hours_spent"], 12.0)
        submissions = [r for r in read_json(self.root / "runs/approvals.json")
                       .get("requests", []) if r["action"] == "submit_candidate"]
        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0]["status"], "pending",
                         "submission must stay human-gated")

    def test_a_verdict_needs_enough_pairs_not_just_a_high_mean(self):
        verdict = campaign.verdict_from_matrix(
            {"candidate": {"mean_lcb": 0.5, "n_pairs": 1}}, "candidate",
            target_lcb=0.02, min_pairs=3)
        self.assertFalse(verdict["competitive"])
        self.assertIn("1 pairs < required 3", verdict["reason"])

    def test_missing_analyze_row_is_not_a_win(self):
        verdict = campaign.verdict_from_matrix({}, "candidate", 0.02, 3)
        self.assertFalse(verdict["competitive"])


class FeedbackTests(CampaignHarness):
    def test_each_pass_receives_the_previous_verdict_and_budget(self):
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": -0.04, "n_pairs": 6}},
            {"candidate": {"mean_lcb": 0.05, "n_pairs": 6}},
        ]
        campaign.run_campaign(self.config())
        second = self.steps.improve_events[1]
        self.assertEqual(second["type"], "campaign_pass")
        self.assertEqual(second["pass"], 2)
        self.assertEqual(second["last_pass"]["outcome"], "eval_short")
        self.assertEqual(second["last_pass"]["verdict"]["mean_lcb"], -0.04)
        self.assertEqual(second["budget"]["gpu_hours_spent"], 6.0)
        self.assertEqual(second["goal"]["target_lcb"], 0.02)
        self.assertIn("campaign gates evaluation itself", second["instructions"])

    def test_screen_failures_feed_back_without_spending(self):
        self.steps.screen_outcomes = [
            {"verdict": "undosed", "verdict_meaning": "raise the dose",
             "moved_features": [], "claims": [], "coverage": {}, "flags": [],
             "challenges": ["raise it"]},
        ]
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": 0.05, "n_pairs": 6}},
        ]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["passes"][0]["outcome"], "screen_undosed")
        self.assertEqual(self.steps.evals_run, 1, "undosed pass must not pay")
        second = self.steps.improve_events[1]
        self.assertEqual(second["last_pass"]["outcome"], "screen_undosed")

    def test_failed_claims_block_payment(self):
        self.steps.screen_outcomes = [
            {"verdict": "measurable", "moved_features": ["hetero"],
             "claims": [{"claim": "regime+", "passed": False}],
             "coverage": {}, "flags": [], "challenges": []},
        ]
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": 0.05, "n_pairs": 6}},
        ]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["passes"][0]["outcome"], "screen_blocked")
        self.assertIn("regime+", state["passes"][0]["detail"])
        self.assertEqual(self.steps.evals_run, 1)


class BudgetTests(CampaignHarness):
    def test_gpu_hour_budget_stops_the_campaign_before_overspending(self):
        state = campaign.run_campaign(self.config(max_gpu_hours=13.0))
        self.assertEqual(state["status"], "budget")
        self.assertEqual(state["gpu_hours_spent"], 12.0)
        self.assertEqual(self.steps.evals_run, 2)
        self.assertIn("would exceed the budget", state["stop_reason"])

    def test_pass_limit_stops_the_campaign(self):
        state = campaign.run_campaign(self.config(max_passes=3))
        self.assertEqual(state["status"], "budget")
        self.assertEqual(len(state["passes"]), 3)

    def test_consecutive_unmeasurable_passes_stall_out(self):
        undosed = {"verdict": "undosed", "verdict_meaning": "x",
                   "moved_features": [], "claims": [], "coverage": {},
                   "flags": [], "challenges": []}
        self.steps.screen_outcomes = [dict(undosed) for _ in range(5)]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["status"], "stalled")
        self.assertEqual(len(state["passes"]), 3)
        self.assertEqual(self.steps.evals_run, 0)

    def test_a_no_change_pass_counts_as_free_failure(self):
        self.steps.improve_outcomes = [{"touch": False}, {"touch": False},
                                       {"touch": False}]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["status"], "stalled")
        self.assertTrue(all(p["outcome"] == "no_change" for p in state["passes"]))

    def test_repeated_hook_failures_stop_the_campaign(self):
        self.steps.improve_outcomes = [{"exit_code": 1, "touch": False},
                                       {"exit_code": 1, "touch": False}]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["status"], "hook_failure")

    def test_deadline_stops_the_loop(self):
        cfg = self.config(deadline_at=0.0)   # already in the past
        cfg.now = lambda: 1.0
        state = campaign.run_campaign(cfg)
        self.assertEqual(state["status"], "deadline")
        self.assertEqual(state["passes"], [])


class ApprovalTests(CampaignHarness):
    def test_autonomous_spend_is_logged_in_the_policy_countable_shape(self):
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": 0.05, "n_pairs": 6}},
        ]
        campaign.run_campaign(self.config())
        autonomous = [e for e in self.events()
                      if e.get("type") == "autonomous_action"]
        self.assertEqual(len(autonomous), 1)
        usage = policy_module.action_usage(
            self.root / "runs/controller-events.jsonl", "gpu_evaluation")
        self.assertEqual(usage["runs"], 1)
        self.assertEqual(usage["hours"], 6.0)

    def test_policy_caps_pause_the_campaign_instead_of_overspending(self):
        cfg = self.config(deadline_at=100.0)
        capped = policy_module.load_policy(self._write_policy(
            "version = 1\n[actions.gpu_evaluation]\nautonomous = true\n"
            "max_runs_per_day = 1\n"))
        cfg.policy = capped
        # One run is already in the trailing window, so the cap is exhausted.
        campaign.append_event(cfg.events_file, {
            "type": "autonomous_action", "action": "gpu_evaluation",
            "at": campaign.utc_now(), "context": {"estimated_hours": 6}})
        cfg.now = lambda: 200.0            # deadline already passed while capped
        state = campaign.run_campaign(cfg)
        self.assertEqual(state["status"], "deadline")
        self.assertEqual(self.steps.evals_run, 0)
        waits = [e for e in self.events() if e.get("type") == "campaign_waiting"]
        self.assertEqual(waits, [], "deadline check precedes any waiting sleep")

    def _write_policy(self, text: str) -> Path:
        path = self.root / "policy-capped.toml"
        path.write_text(text)
        return path

    def test_human_mode_waits_for_approval_then_executes(self):
        cfg = self.config(mode="human", policy=None)
        approvals = self.root / "runs/approvals.json"

        def approve_when_asked(seconds):
            doc = read_json(approvals)
            for request in doc.get("requests", []):
                if request["status"] == "pending":
                    request["status"] = "approved"
            approvals.write_text(json.dumps(doc))

        cfg.sleep = approve_when_asked
        self.steps.analyze_outcomes = [
            {"candidate": {"mean_lcb": 0.05, "n_pairs": 6}},
        ]
        state = campaign.run_campaign(cfg)
        self.assertEqual(state["status"], "goal_met")
        self.assertEqual(self.steps.evals_run, 1)
        statuses = {r["action"]: r["status"] for r in
                    read_json(approvals).get("requests", [])}
        self.assertEqual(statuses["gpu_evaluation"], "completed")
        self.assertEqual(statuses["submit_candidate"], "pending")

    def test_human_rejection_feeds_back_and_continues(self):
        cfg = self.config(mode="human", policy=None, max_passes=2)
        approvals = self.root / "runs/approvals.json"

        def reject_when_asked(seconds):
            doc = read_json(approvals)
            for request in doc.get("requests", []):
                if request["status"] == "pending":
                    request["status"] = "rejected"
            approvals.write_text(json.dumps(doc))

        cfg.sleep = reject_when_asked
        state = campaign.run_campaign(cfg)
        self.assertEqual(state["status"], "budget")
        self.assertEqual(self.steps.evals_run, 0)
        self.assertTrue(all(p["outcome"] == "eval_rejected"
                            for p in state["passes"]))


class KingChangeTests(CampaignHarness):
    def write_receipt(self, king_ref: str) -> None:
        (self.root / "runs/controller-state.json").write_text(json.dumps(
            {"last_receipt": {"summary": {"king_gen_ref": king_ref}}}))

    def test_a_dethroned_king_stops_the_campaign(self):
        self.write_receipt("hippius://kings/old")
        cfg = self.config(max_passes=4)
        original_improve = self.steps.improve

        def improve_then_dethrone(inner_cfg, event):
            result = original_improve(inner_cfg, event)
            if event["pass"] == 2:
                self.write_receipt("hippius://kings/new")
            return result

        with patch.object(campaign, "run_improve", improve_then_dethrone):
            state = campaign.run_campaign(cfg)
        self.assertEqual(state["status"], "king_changed")
        self.assertIn("dethroned control", state["stop_reason"])
        self.assertEqual(len(state["passes"]), 2)

    def test_no_receipt_means_no_king_guard(self):
        state = campaign.run_campaign(self.config(max_passes=1))
        self.assertEqual(state["status"], "budget")


class DurabilityTests(CampaignHarness):
    def test_a_killed_campaign_resumes_with_its_history_and_spend(self):
        state = campaign.run_campaign(self.config(max_passes=2))
        self.assertEqual(state["status"], "budget")
        # Force it back to running, as a kill mid-pass would leave it.
        state["status"] = "running"
        (self.root / "runs/campaign.json").write_text(json.dumps(state))
        resumed = campaign.run_campaign(self.config(max_passes=3))
        self.assertEqual(resumed["campaign_id"], state["campaign_id"])
        self.assertEqual(len(resumed["passes"]), 3)
        self.assertEqual(resumed["gpu_hours_spent"], 18.0)

    def test_eval_failure_is_recorded_not_swallowed(self):
        self.steps.eval_outcomes = [{"exit_code": 3, "stderr": "pod died"},
                                    {"exit_code": 3, "stderr": "pod died"}]
        state = campaign.run_campaign(self.config())
        self.assertEqual(state["status"], "hook_failure")
        self.assertEqual(state["passes"][0]["outcome"], "eval_failed")
        self.assertIn("pod died", state["passes"][0]["stderr"])
        self.assertEqual(state["gpu_hours_spent"], 0.0,
                         "a failed eval must not count as budget spent"
                         " (the pod is torn down by the eval command)")


class ConfigTests(CampaignHarness):
    def test_autonomous_mode_without_a_policy_file_refuses_to_start(self):
        args = campaign.build_parser().parse_args([
            "--root", str(self.root), "--mode", "autonomous",
            "--eval-command", "x"])
        with self.assertRaises(SystemExit) as caught:
            campaign.build_config(args)
        self.assertIn("policy", str(caught.exception))

    def test_missing_king_names_the_fetch_fix(self):
        args = campaign.build_parser().parse_args([
            "--root", str(self.root), "--king", "absent",
            "--eval-command", "x"])
        with self.assertRaises(SystemExit) as caught:
            campaign.build_config(args)
        self.assertIn("cascade fetch", str(caught.exception))

    def test_default_target_comes_from_the_upstream_snapshot(self):
        notes = self.root / "notes"
        notes.mkdir()
        (notes / "upstream-state.json").write_text(json.dumps(
            {"keys": {"scoring.win_margin_end": 0.045}}))
        self.assertEqual(campaign.default_target_lcb(self.root), 0.045)
        self.assertEqual(campaign.default_target_lcb(Path("/nonexistent")), 0.02)


if __name__ == "__main__":
    import unittest
    unittest.main()
