"""Round timing: boundaries, the reveal margin, and the real commit deadline.

The deadline is NOT the epoch boundary. `cascade deploy` defaults to a timed
reveal that decrypts `reveal_margin_blocks` BEFORE the boundary, and eligibility
depends on the reveal landing strictly before it. Reveal timing jitters by a few
blocks, so committing inside the margin risks missing the round entirely.

Round cadence is contract-controlled and may change. Never hardcode it; read
``epoch_blocks`` from the current ``chain.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_BLOCK = 12.0


@dataclass(frozen=True)
class RoundClock:
    current_block: int
    epoch_blocks: int
    reveal_margin_blocks: int

    @property
    def boundary(self) -> int:
        """Next epoch boundary (boundaries sit on the epoch grid)."""
        return ((self.current_block // self.epoch_blocks) + 1) * self.epoch_blocks

    @property
    def reveal_block(self) -> int:
        return self.boundary - self.reveal_margin_blocks

    @property
    def commit_deadline(self) -> int:
        """Latest block we're willing to commit at.

        A safety buffer below the reveal block: reveal timing jitters, and a
        reveal landing at or after the boundary misses the round. Prefer
        `--next-epoch` to squeezing into the margin.
        """
        return self.reveal_block - 25

    def seconds_until(self, block: int) -> float:
        return max(0, block - self.current_block) * SECONDS_PER_BLOCK

    def can_finish(self, work_hours: float) -> bool:
        """Is there time to run `work_hours` of work AND still commit?"""
        return self.seconds_until(self.commit_deadline) > work_hours * 3600

    def summary(self) -> str:
        h = self.seconds_until(self.commit_deadline) / 3600
        return (
            f"block {self.current_block:,} | boundary {self.boundary:,} | "
            f"reveal ~{self.reveal_block:,} | commit before {self.commit_deadline:,} "
            f"({h:.1f}h)"
        )


def from_chain(subtensor, chain_cfg) -> RoundClock:
    return RoundClock(
        current_block=subtensor.get_current_block(),
        epoch_blocks=int(chain_cfg.round.epoch_blocks),
        reveal_margin_blocks=int(getattr(chain_cfg.round, "reveal_margin_blocks", 25)),
    )
