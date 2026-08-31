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

    def test_rejects_root_one_level_too_deep_with_hint(self):
        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "pools/snapshots/2026-07-16"
            snapshot.mkdir(parents=True)
            snapshot.joinpath("series.npy").write_bytes(b"offline fixture")
            with self.assertRaisesRegex(
                    ValueError, r"is itself a dated snapshot .* use --pools-root .*snapshots"):
                snapshot_dirs(snapshot)

    def test_never_hints_at_a_directory_without_snapshots(self):
        # The old hint suggested a literal `snapshots/` child whether or not it
        # held dated snapshots, sending people to a path just as wrong.
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pools"
            (root / "snapshots").mkdir(parents=True)      # exists, but empty
            try:
                snapshot_dirs(root)
            except ValueError as error:
                self.assertNotIn("use --pools-root", str(error))
                self.assertIn("it contains: snapshots", str(error))
            else:
                self.fail("expected ValueError")


class WarmKwargTests(TestCase):
    """A requested warm start must reach the trainer or fail loudly — an arm
    silently trained from scratch pairs against a warm king as a fake result."""

    def test_auto_detects_a_known_parameter_name(self):
        from miner.evaluate import warm_train_kwarg

        class Trainer:
            def train(self, series, contract, *, training_seed, token_budget,
                      out_dir, init_dir=None):
                pass

        self.assertEqual(warm_train_kwarg(Trainer()), "init_dir")

    def test_refuses_a_trainer_with_no_warm_parameter(self):
        from miner.evaluate import warm_train_kwarg

        class Trainer:
            def train(self, series, contract, *, training_seed, token_budget,
                      out_dir):
                pass

        with self.assertRaisesRegex(ValueError, "--warm-init-kwarg"):
            warm_train_kwarg(Trainer())

    def test_explicit_kwarg_is_validated_against_the_signature(self):
        from miner.evaluate import warm_train_kwarg

        class Trainer:
            def train(self, series, contract, *, training_seed, token_budget,
                      out_dir, start_checkpoint=None):
                pass

        self.assertEqual(warm_train_kwarg(Trainer(), "start_checkpoint"),
                         "start_checkpoint")
        with self.assertRaisesRegex(ValueError, "no parameter"):
            warm_train_kwarg(Trainer(), "warm_dir")

    def test_explicit_kwarg_passes_through_var_keywords(self):
        from miner.evaluate import warm_train_kwarg

        class Trainer:
            def train(self, series, contract, **kwargs):
                pass

        self.assertEqual(warm_train_kwarg(Trainer(), "init_dir"), "init_dir")


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
