import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner import experiments


class LedgerTests(TestCase):
    def test_log_and_list_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entry = experiments.log_entry(
                root, hypothesis="replace weakest family with GARCH bursts",
                change="config.json families[7]", seeds=[0, 1, 2],
                snapshots=["2026-07-16"], status="planned",
            )
            listed = experiments.list_entries(root)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0], entry)
        self.assertRegex(entry["id"], r"^[0-9a-f]{12}$")
        self.assertEqual(listed[0]["seeds"], [0, 1, 2])

    def test_status_filter_and_limit_keep_the_newest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                experiments.log_entry(
                    root, hypothesis=f"arm {index}",
                    status="null" if index == 1 else "planned",
                )
            nulls = experiments.list_entries(root, status="null")
            newest = experiments.list_entries(root, limit=1)
        self.assertEqual([entry["hypothesis"] for entry in nulls], ["arm 1"])
        self.assertEqual([entry["hypothesis"] for entry in newest], ["arm 2"])

    def test_entry_requires_hypothesis_and_known_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "hypothesis"):
                experiments.log_entry(root, hypothesis="   ")
            with self.assertRaisesRegex(ValueError, "unknown status"):
                experiments.log_entry(root, hypothesis="x", status="great")

    def test_ledger_tolerates_a_corrupt_line(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiments.log_entry(root, hypothesis="survives corruption")
            with (root / experiments.LEDGER).open("a", encoding="utf-8") as fh:
                fh.write("{corrupt\n")
            experiments.log_entry(root, hypothesis="second entry")
            listed = experiments.list_entries(root)
        self.assertEqual(
            [entry["hypothesis"] for entry in listed],
            ["survives corruption", "second entry"],
        )

    def test_missing_ledger_lists_nothing(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(experiments.list_entries(Path(directory)), [])


class CliTests(TestCase):
    def test_log_then_list_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = io.StringIO()
            with redirect_stdout(out):
                code = experiments.main([
                    "--root", str(root), "log",
                    "--hypothesis", "cli roundtrip",
                    "--seeds", "0,1", "--status", "measured",
                    "--noise-floor", "0.004",
                ])
            self.assertEqual(code, 0)
            logged = json.loads(out.getvalue())
            out = io.StringIO()
            with redirect_stdout(out):
                code = experiments.main(
                    ["--root", str(root), "list", "--json"])
            self.assertEqual(code, 0)
            listed = json.loads(out.getvalue())
        self.assertEqual(listed, [logged])
        self.assertEqual(logged["seeds"], ["0", "1"])
        self.assertEqual(logged["noise_floor"], 0.004)
