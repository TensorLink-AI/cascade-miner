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


class LiveRuleBlockTests(TestCase):
    """The local draw must emulate the block-gated rules a round scored today
    runs under — passing no block silently replays the retired uniform draw."""

    @staticmethod
    def _cfg(mix_from_block=0, mix_tier_from_block=0):
        from types import SimpleNamespace
        return SimpleNamespace(eval=SimpleNamespace(
            mix_from_block=mix_from_block, mix_tier_from_block=mix_tier_from_block))

    def test_nothing_armed_keeps_the_legacy_draw(self):
        from miner.evaluate import live_rule_block
        self.assertIsNone(live_rule_block(self._cfg()))

    def test_armed_mix_activates_at_its_block(self):
        from miner.evaluate import live_rule_block
        self.assertEqual(live_rule_block(self._cfg(mix_from_block=8895600)), 8895600)

    def test_latest_armed_rule_wins(self):
        from miner.evaluate import live_rule_block
        cfg = self._cfg(mix_from_block=8895600, mix_tier_from_block=8942400)
        self.assertEqual(live_rule_block(cfg), 8942400)
