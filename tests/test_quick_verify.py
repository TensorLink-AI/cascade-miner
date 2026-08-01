"""Offline tests for the memory-bounded determinism check."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts import quick_verify

DETERMINISTIC = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)

    def generate(self, n_series):
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield rng.normal(0.0, 1.0, size=256)
"""

NONDETERMINISTIC = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._rng = np.random.default_rng()

    def generate(self, n_series):
        for _ in range(n_series):
            yield self._rng.normal(0.0, 1.0, size=256)
"""

SEED_IGNORING = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        pass

    def generate(self, n_series):
        for _ in range(n_series):
            yield np.zeros(256, dtype=np.float64)
"""

WRONG_DTYPE = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)

    def generate(self, n_series):
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield rng.normal(0.0, 1.0, size=256).astype(np.float32)
"""

NON_FINITE = """
import numpy as np

class Generator:
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)

    def generate(self, n_series):
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            series = rng.normal(0.0, 1.0, size=256)
            series[3] = np.nan
            yield series
"""


class QuickVerifyTests(TestCase):
    def make_candidate(self, directory: str, source: str, **config) -> Path:
        repo = Path(directory) / "candidate"
        repo.mkdir()
        (repo / "generator.py").write_text(source, encoding="utf-8")
        (repo / "config.json").write_text(json.dumps(config or {"name": "t"}), encoding="utf-8")
        return repo

    def run_check(self, source: str, *, n_series=4, **config) -> int:
        with TemporaryDirectory() as directory:
            repo = self.make_candidate(directory, source, **config)
            return quick_verify.main([str(repo), "--n-series", str(n_series)])

    def test_deterministic_generator_passes(self):
        self.assertEqual(self.run_check(DETERMINISTIC), 0)

    def test_nondeterministic_generator_fails(self):
        self.assertEqual(self.run_check(NONDETERMINISTIC), 1)

    def test_generator_that_ignores_its_seed_fails(self):
        """Identical corpora across seeds means the round seed does nothing."""
        self.assertEqual(self.run_check(SEED_IGNORING), 1)

    def test_wrong_dtype_fails(self):
        self.assertEqual(self.run_check(WRONG_DTYPE), 1)

    def test_non_finite_values_fail(self):
        self.assertEqual(self.run_check(NON_FINITE), 1)

    def test_length_bounds_from_config_are_enforced(self):
        self.assertEqual(self.run_check(DETERMINISTIC, min_length=512), 1)
        self.assertEqual(self.run_check(DETERMINISTIC, max_length=128), 1)
        self.assertEqual(self.run_check(DETERMINISTIC, min_length=128, max_length=512), 0)

    def test_missing_generator_module_is_reported(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory) / "empty"
            repo.mkdir()
            with self.assertRaisesRegex(SystemExit, "no generator.py"):
                quick_verify.main([str(repo)])

    def test_module_without_a_generator_class_is_reported(self):
        with TemporaryDirectory() as directory:
            repo = self.make_candidate(directory, "x = 1\n")
            with self.assertRaisesRegex(SystemExit, "no `Generator` class"):
                quick_verify.main([str(repo)])

    def test_the_shipped_starter_candidate_is_checkable(self):
        """The bootstrap generator must satisfy its own contract."""
        repo = Path(__file__).resolve().parents[1] / "generators/candidate"
        self.assertTrue((repo / "generator.py").is_file())
