"""Offline tests for the pre-GPU corpus screen.

Every case builds its corpora in memory or from a throwaway generator file, so
the suite stays offline: no cascade library, no pool data, no GPU.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from miner import screen

ROOT = Path(__file__).resolve().parents[1]

KING_SOURCE = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)

    def generate(self, n_series):
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            length = int(rng.integers(128, 513))
            t = np.arange(length, dtype=np.float64)
            period = float(rng.choice([7.0, 12.0, 24.0]))
            y = (rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * t / period)
                 + rng.normal(0.0, 0.2, length))
            yield y.astype(np.float64)
"""

# Same skeleton, plus large random level shifts: a change big enough to see.
REGIME_SOURCE = KING_SOURCE.replace(
    "            yield y.astype(np.float64)",
    """            for cut in np.sort(rng.integers(10, length, size=3)):
                y[cut:] += rng.normal(0.0, 5.0)
            yield y.astype(np.float64)""",
)

BROKEN_SOURCE = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)

    def generate(self, n_series):
        for _ in range(n_series):
            yield np.zeros(32, dtype=np.float64)
"""


def write_generator(root: Path, name: str, source: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "generator.py").write_text(source, encoding="utf-8")
    return repo


def gates() -> tuple[dict, str]:
    return screen.load_gates(ROOT / "notes/upstream-state.json")


def sine_corpus(rng: np.random.Generator, n: int = 64, *, shift: float = 0.0,
                noise: float = 0.2) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        length = int(rng.integers(128, 513))
        t = np.arange(length, dtype=np.float64)
        period = float(rng.choice([7.0, 12.0, 24.0]))
        y = np.sin(2 * np.pi * t / period) + rng.normal(0.0, noise, length)
        if shift:
            for cut in np.sort(rng.integers(10, length, size=3)):
                y[cut:] += rng.normal(0.0, shift)
        out.append(y)
    return out


def profiles(shift: float = 0.0, noise: float = 0.2, name: str = "arm",
             seeds: tuple[int, int] = (0, 1)) -> tuple[screen.Profile, screen.Profile]:
    return tuple(
        screen.profile_series(name, seed, sine_corpus(np.random.default_rng(seed),
                                                      shift=shift, noise=noise),
                              generate_seconds=0.01)
        for seed in seeds
    )


class FeatureTests(TestCase):
    def test_features_are_finite_for_degenerate_series(self):
        for series in (np.zeros(64), np.ones(64), np.arange(64, dtype=np.float64),
                       np.array([1.0, 2.0]), np.zeros(0)):
            vector = screen.series_features(series)
            self.assertEqual(vector.shape, (len(screen.FEATURES),))
            self.assertTrue(np.isfinite(vector).all(), f"non-finite features for {series[:4]}")

    def test_features_separate_trend_seasonality_and_noise(self):
        t = np.arange(512, dtype=np.float64)
        index = {name: i for i, name in enumerate(screen.FEATURE_NAMES)}
        trend = screen.series_features(t)
        seasonal = screen.series_features(np.sin(2 * np.pi * t / 24.0))
        noise = screen.series_features(np.random.default_rng(0).normal(0, 1, 512))
        self.assertGreater(trend[index["trend_strength"]], 0.9)
        self.assertLess(seasonal[index["trend_strength"]], 0.1)
        self.assertGreater(seasonal[index["seasonal_strength"]],
                           noise[index["seasonal_strength"]])
        self.assertGreater(noise[index["spectral_entropy"]],
                           seasonal[index["spectral_entropy"]])
        # log2(24) ≈ 4.58: the dominant period is recovered, not merely nonzero.
        self.assertAlmostEqual(seasonal[index["log_period"]], np.log2(24.0), places=1)

    def test_features_are_deterministic(self):
        series = np.random.default_rng(7).normal(0, 1, 300)
        np.testing.assert_array_equal(screen.series_features(series),
                                      screen.series_features(series.copy()))


class ProfileTests(TestCase):
    def test_profile_counts_duplicates_constants_and_points(self):
        flat = np.zeros(128)
        profile = screen.profile_series("x", 0, [flat, flat.copy(), np.arange(128.0)],
                                        generate_seconds=0.5)
        self.assertEqual(profile.n_series, 3)
        self.assertEqual(profile.total_points, 384)
        self.assertAlmostEqual(profile.dup_fraction, 1 / 3)
        self.assertAlmostEqual(profile.constant_fraction, 2 / 3)
        self.assertAlmostEqual(profile.points_per_second(), 768.0)

    def test_profile_records_contract_violations_without_crashing(self):
        profile = screen.profile_series(
            "x", 0, [np.array([np.nan, 1.0, 2.0, 3.0]), np.zeros(8, dtype=np.float32)])
        self.assertEqual(profile.nonfinite_series, 1)
        self.assertEqual(profile.bad_dtype_series, 1)
        self.assertTrue(np.isfinite(profile.matrix).all())

    def test_profile_generator_reads_a_directory_and_times_it(self):
        with TemporaryDirectory() as tmp:
            repo = write_generator(Path(tmp), "king", KING_SOURCE)
            profile = screen.profile_generator(repo, 0, 16)
        self.assertEqual(profile.n_series, 16)
        self.assertEqual(profile.n_requested, 16)
        self.assertGreater(profile.generate_seconds, 0.0)


class VerdictTests(TestCase):
    def screen_arms(self, candidate, king, **kwargs):
        data = screen.ScreenInput(candidate=candidate, king=king, **kwargs)
        gate_values, source = gates()
        return screen.screen(data, gate_values, source)

    def test_identical_generators_are_undosed(self):
        report = self.screen_arms(profiles(name="candidate"), profiles(name="king"))
        self.assertEqual(report["verdict"], "undosed")
        self.assertEqual(report["moved_features"], [])
        self.assertIn("undosed", [flag["name"] for flag in report["flags"]])

    def test_a_change_inside_seed_noise_is_undosed(self):
        report = self.screen_arms(profiles(noise=0.205, name="candidate"),
                                  profiles(name="king"))
        self.assertEqual(report["verdict"], "undosed")

    def test_a_large_change_is_measurable_and_names_what_moved(self):
        report = self.screen_arms(profiles(shift=5.0, name="candidate"),
                                  profiles(name="king"))
        self.assertEqual(report["verdict"], "measurable")
        self.assertIn("level_shift", report["moved_features"])
        self.assertGreater(report["coverage"]["novelty"], 0.1)

    def test_contract_violations_block_before_anything_else(self):
        broken = screen.profile_series("candidate", 0, [np.zeros(32)] * 32,
                                       generate_seconds=0.01)
        report = self.screen_arms((broken, broken), profiles(name="king"))
        self.assertEqual(report["verdict"], "blocked")
        blocking = {flag["name"] for flag in report["flags"] if flag["level"] == "block"}
        self.assertIn("length", blocking)
        self.assertIn("duplicates", blocking)

    def test_a_generator_that_yields_nothing_is_blocked_not_a_crash(self):
        empty = screen.profile_series("candidate", 0, [], n_requested=64)
        report = self.screen_arms((empty, empty), profiles(name="king"))
        self.assertEqual(report["verdict"], "blocked")
        self.assertIn("count", {flag["name"] for flag in report["flags"]})

    def test_generate_budget_is_projected_to_the_full_corpus(self):
        slow = screen.profile_series("candidate", 0, sine_corpus(np.random.default_rng(0), 64),
                                     generate_seconds=60.0)
        report = self.screen_arms((slow, slow), profiles(name="king"))
        blocking = {flag["name"] for flag in report["flags"] if flag["level"] == "block"}
        self.assertIn("generate_budget", blocking)

    def test_report_always_states_its_limits_and_challenges(self):
        report = self.screen_arms(profiles(shift=5.0, name="candidate"), profiles(name="king"))
        self.assertTrue(report["limits"])
        self.assertTrue(report["challenges"])
        self.assertIn("ranks nothing", report["challenges"][-1])
        json.dumps(report)          # the report must survive --json


class ClaimTests(TestCase):
    def test_parse_claim_rejects_unknown_families(self):
        self.assertEqual(screen.parse_claim("regime+"), ("regime", 1))
        self.assertEqual(screen.parse_claim("noise-"), ("noise", -1))
        self.assertEqual(screen.parse_claim("trend"), ("trend", 0))
        with self.assertRaises(ValueError):
            screen.parse_claim("vibes+")

    def test_a_claim_the_corpus_does_not_carry_fails(self):
        data = screen.ScreenInput(candidate=profiles(shift=5.0, name="candidate"),
                                  king=profiles(name="king"),
                                  claims=("regime+", "intermittency+"))
        gate_values, source = gates()
        report = screen.screen(data, gate_values, source)
        results = {claim["claim"]: claim["passed"] for claim in report["claims"]}
        self.assertTrue(results["regime+"])
        self.assertFalse(results["intermittency+"])
        self.assertIn("intermittency+", report["challenges"][-2])

    def test_a_claim_in_the_wrong_direction_fails(self):
        data = screen.ScreenInput(candidate=profiles(shift=5.0, name="candidate"),
                                  king=profiles(name="king"), claims=("regime-",))
        gate_values, source = gates()
        report = screen.screen(data, gate_values, source)
        self.assertFalse(report["claims"][0]["passed"])
        self.assertIn("opposite direction", report["claims"][0]["why"])


class HistoryTests(TestCase):
    def test_history_axis_reads_direction_against_the_last_transition(self):
        prior, _ = profiles(name="prior")
        king, king_b = profiles(shift=2.0, name="king")
        candidate = profiles(shift=6.0, name="candidate")
        data = screen.ScreenInput(candidate=candidate, king=(king, king_b), prior_king=prior)
        gate_values, source = gates()
        report = screen.screen(data, gate_values, source)
        self.assertIn("history", report)
        self.assertGreater(report["history"]["cosine_with_last_winning_move"], 0.0)
        self.assertIn("n=1", report["history"]["caveat"])
        self.assertFalse(report["history"]["closer_to_dethroned_king"])

    def test_a_candidate_resembling_the_dethroned_king_is_called_out(self):
        prior, _ = profiles(name="prior")
        king = profiles(shift=6.0, name="king")
        candidate = profiles(shift=0.2, name="candidate")   # nearly the old corpus
        data = screen.ScreenInput(candidate=candidate, king=king, prior_king=prior)
        gate_values, source = gates()
        report = screen.screen(data, gate_values, source)
        self.assertTrue(report["history"]["closer_to_dethroned_king"])
        self.assertTrue(any("dethroned king" in line for line in report["challenges"]))


class CommandLineTests(TestCase):
    def test_cli_exit_codes_distinguish_blocked_from_undosed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            king = write_generator(root, "king", KING_SOURCE)
            twin = write_generator(root, "twin", KING_SOURCE)
            regime = write_generator(root, "regime", REGIME_SOURCE)
            broken = write_generator(root, "broken", BROKEN_SOURCE)
            out = root / "report.json"
            common = ["--king", str(king), "--n-series", "48"]
            with redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(screen.main([str(twin), *common]), 3)
                self.assertEqual(screen.main([str(regime), *common, "--json", str(out)]), 0)
                self.assertEqual(screen.main([str(broken), *common]), 1)
                self.assertEqual(
                    screen.main([str(regime), *common, "--claim", "intermittency+"]), 4)
            self.assertIn("UNDOSED", captured.getvalue())
            # 2 is argparse's usage-error code and must never be a verdict.
            self.assertNotIn(2, screen.EXIT_CODES.values())
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["verdict"], "measurable")
            self.assertIn("per_feature", report)


if __name__ == "__main__":
    import unittest
    unittest.main()
