from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner.evaluate import snapshot_dirs


class SnapshotLayoutTests(TestCase):
    def test_accepts_dated_snapshot_directories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pools/snapshots"
            snapshot = root / "2026-07-16"
            snapshot.mkdir(parents=True)
            snapshot.joinpath("series.npy").write_bytes(b"offline fixture")
            self.assertEqual(snapshot_dirs(root), [snapshot])

    def test_rejects_root_one_level_too_high_with_hint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pools"
            snapshot = root / "snapshots/2026-07-16"
            snapshot.mkdir(parents=True)
            snapshot.joinpath("series.npy").write_bytes(b"offline fixture")
            with self.assertRaisesRegex(ValueError, r"use --pools-root .*snapshots"):
                snapshot_dirs(root)
