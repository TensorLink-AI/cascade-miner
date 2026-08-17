"""Offline checks for .github/workflows — string-level, no YAML dependency."""

from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
TESTS_WF = ROOT / ".github/workflows/tests.yml"
SYNC_WF = ROOT / ".github/workflows/upstream-sync.yml"


class WorkflowTests(TestCase):
    def test_ci_runs_the_offline_suite(self):
        source = TESTS_WF.read_text()
        self.assertIn("unittest discover -s tests", source)
        self.assertIn("numpy", source)

    def test_sync_regenerates_the_snapshot_from_upstream(self):
        # The workflow and scripts/sync.sh must run the SAME generator, or CI
        # and the operator host end up with different pictures of upstream.
        source = SYNC_WF.read_text()
        self.assertIn("TensorLink-AI/cascade", source)
        self.assertIn("scripts/upstream_state.py", source)
        self.assertIn("notes/upstream-state.json", source)
        self.assertIn("notes/UPSTREAM.md", source)

    def test_sync_opens_a_pull_request_rather_than_only_reporting(self):
        # The previous watcher only filed an issue; nothing landed unless a
        # human acted, and the notes went ten days stale through an armed
        # warm-start. The sync now proposes the change itself.
        source = SYNC_WF.read_text()
        self.assertIn("gh pr create", source)
        self.assertIn("gh pr edit", source)
        self.assertIn("contents: write", source)
        self.assertIn("pull-requests: write", source)

    def test_sync_never_force_pushes_over_hand_folded_prose(self):
        source = SYNC_WF.read_text()
        self.assertNotIn("push --force", source)
        self.assertNotIn("force-with-lease", source)

    def test_sync_stages_only_the_generated_files(self):
        # The cascade clone lives inside the tree during the run; a blanket
        # `git add` would commit upstream's source into this repo.
        source = SYNC_WF.read_text()
        self.assertIn("git add notes/upstream-state.json notes/UPSTREAM.md", source)
        self.assertNotIn("git add -A", source)
        self.assertNotIn("git add .", source)
        self.assertIn("/cascade/", (ROOT / ".gitignore").read_text())

    def test_sync_runs_the_suite_itself(self):
        # A PR opened with GITHUB_TOKEN does not trigger workflow runs, so the
        # prose-vs-snapshot assertions have to run inside the sync job.
        source = SYNC_WF.read_text()
        self.assertIn("unittest discover -s tests", source)
        self.assertIn("GITHUB_TOKEN", source)

    def test_sync_points_at_the_operator_side_reinstall(self):
        # A runner cannot reinstall the library where scoring happens; merging
        # the PR alone still leaves the venv scoring with the old metric.
        source = SYNC_WF.read_text()
        self.assertIn("scripts/sync.sh", source)
        self.assertIn("reinstall", source)

    def test_sync_reports_its_own_breakage(self):
        # "No sync PR" must not be ambiguous between "upstream is quiet" and
        # "the robot has been erroring for a fortnight" — which is the same
        # silent-staleness failure the sync exists to prevent.
        source = SYNC_WF.read_text()
        self.assertIn("if: failure()", source)
        self.assertIn("issues: write", source)
        self.assertIn("gh issue create", source)
        self.assertIn("gh issue close", source)

    def test_the_superseded_watcher_is_gone(self):
        self.assertFalse((ROOT / ".github/workflows/upstream-watch.yml").exists())
