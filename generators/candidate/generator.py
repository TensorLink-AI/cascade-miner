"""Minimal deterministic Cascade generator intended as an editable starter."""

from collections.abc import Iterator

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "bootstrap-seasonal"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            length = int(rng.integers(128, 1025))
            time = np.arange(length, dtype=np.float64)
            period = float(rng.choice(np.array([7, 12, 24, 52])))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            trend = rng.normal(0.0, 0.005) * time
            seasonal = rng.uniform(0.5, 2.0) * np.sin(2.0 * np.pi * time / period + phase)
            noise = rng.normal(0.0, 0.2, size=length)
            yield (trend + seasonal + noise).astype(np.float64)
