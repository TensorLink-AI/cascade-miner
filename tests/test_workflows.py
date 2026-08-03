"""Offline checks for .github/workflows — string-level, no YAML dependency."""

from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
TESTS_WF = ROOT / ".github/workflows/tests.yml"
WATCH_WF = ROOT / ".github/workflows/upstream-watch.yml"


class WorkflowTests(TestCase):
    def test_ci_runs_the_offline_suite(self):
        source = TESTS_WF.read_text()
        self.assertIn("unittest discover -s tests", source)
        self.assertIn("numpy", source)

    def test_watch_compares_against_the_contract_stamp(self):
        # The watcher and scripts/sync.sh must agree on where the sync state
        # lives (the CONTRACT.md stamp) and what counts as miner-facing.
        source = WATCH_WF.read_text()
        self.assertIn("Last synced:", source)
        self.assertIn("notes/CONTRACT.md", source)
        for path in ("chain.toml", "decisions", "cascade/eval", "cascade/shared/config.py"):
            self.assertIn(path, source)

    def test_watch_cannot_write_code_only_issues(self):
        source = WATCH_WF.read_text()
        self.assertIn("contents: read", source)
        self.assertIn("issues: write", source)

    def test_watch_points_at_the_operator_side_sync(self):
        self.assertIn("scripts/sync.sh", WATCH_WF.read_text())
