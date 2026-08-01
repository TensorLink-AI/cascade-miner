import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner import experiments, status


def seed_runs(root: Path) -> None:
    runs = root / "runs"
    runs.mkdir(parents=True)
    (runs / "controller-state.json").write_text(json.dumps({
        "initialized": True,
        "updated_at": "2026-08-01T10:00:00+00:00",
        "last_round_id": "round-42",
        "cascade_head": "abc123",
        "eval_pool_repo": "Tensor-Link/cascade-eval-pool",
        "eval_pool_revision": "rev-1234567890ab",
        "eval_pool_snapshot": "2026-07-16",
        "eval_pool_local_snapshots": ["2026-07-16"],
        "last_receipt": {
            "status": "final",
            "summary": {"king_gen_ref": "hippius://king/ref"},
            "miner_heat": {"rank": 3},
        },
    }), encoding="utf-8")
    (runs / "approvals.json").write_text(json.dumps({"requests": [
        {"id": "aaa111bbb222", "action": "gpu_evaluation",
         "reason": "screen arm A", "status": "pending"},
        {"id": "ccc333ddd444", "action": "gpu_evaluation",
         "reason": "done earlier", "status": "completed"},
    ]}), encoding="utf-8")
    (runs / "controller-events.jsonl").write_text(
        "\n".join(json.dumps({"type": "round_result", "round_id": f"r{i}"})
                  for i in range(15)) + "\nnot json\n",
        encoding="utf-8",
    )


class StatusTests(TestCase):
    def test_summary_reads_all_runs_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed_runs(root)
            experiments.log_entry(root, hypothesis="ledger shows up")
            summary = status.summarize(root)
        self.assertEqual(summary["round"]["id"], "round-42")
        self.assertEqual(summary["round"]["king_gen_ref"], "hippius://king/ref")
        self.assertEqual(summary["round"]["miner_heat"], {"rank": 3})
        self.assertEqual(summary["eval_pool"]["snapshot"], "2026-07-16")
        self.assertEqual(
            [r["id"] for r in summary["approvals"]["pending"]],
            ["aaa111bbb222"],
        )
        self.assertEqual(summary["approvals"]["counts"]["completed"], 1)
        # The corrupt trailing line occupies one tail slot and is skipped.
        self.assertEqual(len(summary["recent_events"]), status.EVENT_TAIL - 1)
        self.assertEqual(summary["recent_events"][-1]["round_id"], "r14")
        self.assertEqual(summary["experiments"][0]["hypothesis"],
                         "ledger shows up")

    def test_summary_renders_before_the_controller_ever_ran(self):
        with TemporaryDirectory() as directory:
            summary = status.summarize(Path(directory))
            text = status.render(summary)
        self.assertFalse(summary["initialized"])
        self.assertIn("(none)", text)
        self.assertIn("none pending", text)

    def test_render_shows_pending_approvals_with_the_approve_command(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed_runs(root)
            text = status.render(status.summarize(root))
        self.assertIn("aaa111bbb222", text)
        self.assertIn("--approve <id>", text)
        self.assertIn("hippius://king/ref", text)
