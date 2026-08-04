import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner import issues


class LedgerTests(TestCase):
    def test_report_and_list_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entry = issues.report_entry(
                root, kind="bug", title="quick_verify crashes on empty config",
                detail="run quick_verify against a candidate with config.json {}",
                area="scripts/quick_verify.py",
            )
            listed = issues.list_entries(root)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0], entry)
        self.assertRegex(entry["id"], r"^[0-9a-f]{12}$")
        self.assertEqual(entry["status"], "open")

    def test_kind_and_status_filters_and_limit_keep_the_newest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            issues.report_entry(root, kind="bug", title="bug 0")
            issues.report_entry(root, kind="feature", title="feature 1")
            issues.report_entry(root, kind="bug", title="bug 2")
            bugs = issues.list_entries(root, kind="bug")
            newest = issues.list_entries(root, limit=1)
        self.assertEqual([e["title"] for e in bugs], ["bug 0", "bug 2"])
        self.assertEqual([e["title"] for e in newest], ["bug 2"])

    def test_entry_requires_title_and_known_kind_and_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "title"):
                issues.report_entry(root, kind="bug", title="   ")
            with self.assertRaisesRegex(ValueError, "unknown kind"):
                issues.report_entry(root, kind="complaint", title="x")
            with self.assertRaisesRegex(ValueError, "unknown status"):
                issues.report_entry(root, kind="bug", title="x", status="fixed")

    def test_open_entries_apply_supersedes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = issues.report_entry(root, kind="bug", title="gets fixed")
            still = issues.report_entry(root, kind="feature", title="still open")
            issues.report_entry(root, kind="bug", title="gets fixed",
                                status="resolved", supersedes=fixed["id"])
            open_now = issues.open_entries(root)
        self.assertEqual([e["id"] for e in open_now], [still["id"]])

    def test_ledger_tolerates_a_corrupt_line(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            issues.report_entry(root, kind="bug", title="survives corruption")
            with (root / issues.LEDGER).open("a", encoding="utf-8") as fh:
                fh.write("{corrupt\n")
            issues.report_entry(root, kind="feature", title="second entry")
            listed = issues.list_entries(root)
        self.assertEqual(
            [entry["title"] for entry in listed],
            ["survives corruption", "second entry"],
        )

    def test_missing_ledger_lists_nothing(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(issues.list_entries(Path(directory)), [])
            self.assertEqual(issues.open_entries(Path(directory)), [])


class CliTests(TestCase):
    def test_report_then_list_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = io.StringIO()
            with redirect_stdout(out):
                code = issues.main([
                    "--root", str(root), "report",
                    "--kind", "feature", "--title", "cli roundtrip",
                    "--detail", "agents need a bug channel",
                ])
            self.assertEqual(code, 0)
            reported = json.loads(out.getvalue())
            out = io.StringIO()
            with redirect_stdout(out):
                code = issues.main(["--root", str(root), "list", "--json"])
            self.assertEqual(code, 0)
            listed = json.loads(out.getvalue())
        self.assertEqual(listed, [reported])
        self.assertEqual(reported["kind"], "feature")

    def test_list_open_hides_triaged_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            triaged = issues.report_entry(root, kind="bug", title="triaged away")
            issues.report_entry(root, kind="bug", title="triaged away",
                                status="wontfix", supersedes=triaged["id"])
            issues.report_entry(root, kind="bug", title="live bug")
            out = io.StringIO()
            with redirect_stdout(out):
                code = issues.main(
                    ["--root", str(root), "list", "--open", "--json"])
            self.assertEqual(code, 0)
            listed = json.loads(out.getvalue())
        self.assertEqual([e["title"] for e in listed], ["live bug"])
