"""Port of RewardBooster.sol — see reference-sim/ProverRewardBoostProposal.sol.

Line-for-line mirror of the Solidity. Integer arithmetic throughout to match
EVM semantics (Python `//` matches Solidity unsigned division).
"""

from __future__ import annotations

from dataclasses import dataclass

DECAY_PER_EPOCH = 100_000


@dataclass(frozen=True)
class BoostConfig:
    increment: int
    max_score: int
    a: int
    minimum: int
    k: int

    @classmethod
    def current(cls) -> "BoostConfig":
        return cls(increment=125_000, max_score=15_000_000, a=1_000, minimum=100_000, k=1_000_000)

    @classmethod
    def proposed(cls) -> "BoostConfig":
        return cls(increment=101_400, max_score=367_500, a=250_000, minimum=10_000, k=1_000_000)


@dataclass
class ActivityScore:
    # time=-1 sentinel means "never submitted" — decay from -1 still gives 0 value.
    time: int = -1
    value: int = 0


def decayed_value(score: ActivityScore, epoch: int) -> int:
    decrease = (epoch - score.time) * DECAY_PER_EPOCH
    return 0 if decrease > score.value else score.value - decrease


def record_submission(score: ActivityScore, epoch: int, cfg: BoostConfig) -> ActivityScore:
    """Mirror of RewardBooster.updateAndGetShares state mutation.

    The Solidity only increments value when `store.time != currentEpoch`, so
    repeat submissions within the same epoch are no-ops.
    """
    decayed = decayed_value(score, epoch)
    if score.time != epoch:
        new_value = min(decayed + cfg.increment, cfg.max_score)
    else:
        new_value = decayed
    return ActivityScore(time=epoch, value=new_value)


def to_shares(value: int, cfg: BoostConfig) -> int:
    if value >= cfg.max_score:
        return cfg.k
    t = cfg.max_score - value
    rhs = cfg.a * t * t // 10**10
    if cfg.k < rhs:
        return cfg.minimum
    return max(cfg.k - rhs, cfg.minimum)


def shares_at(score: ActivityScore, epoch: int, cfg: BoostConfig) -> int:
    """View-only shares: decay to `epoch` without recording a submission."""
    return to_shares(decayed_value(score, epoch), cfg)
