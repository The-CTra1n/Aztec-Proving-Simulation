"""Epoch-loop simulator over a per-prover submission matrix.

Produces per-epoch rewards (AZTEC + USD), per-epoch shares, per-prover USD profit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from proving_sim.booster import ActivityScore, BoostConfig, record_submission, to_shares


@dataclass(frozen=True)
class EconomicConfig:
    epoch_reward: float                  # AZTEC pool minted per epoch (pre sequencer split).
                                         # On-chain: getCheckpointReward() × 32 (32 checkpoints/epoch).
    sequencer_bps: int                   # sequencer take in basis points
    aztec_usd: float                     # AZTEC → USD conversion
    hardware_cost_per_epoch_usd: float   # per-prover infra cost each epoch (sunk, whether submitting or not)
    gas_cost_per_submission_usd: float = 0.0  # L1 gas paid only on epochs the prover submits

    @property
    def prover_pool(self) -> float:
        return self.epoch_reward * (1.0 - self.sequencer_bps / 10_000.0)


def simulate(
    submissions: np.ndarray,
    booster: BoostConfig,
    econ: EconomicConfig,
    distribution: str = "per_submission",
    prover_costs_usd_per_epoch: np.ndarray | None = None,
    online_mask: np.ndarray | None = None,
) -> dict:
    """Run the booster + reward distribution over the given submission matrix.

    Args:
        submissions: (n_provers, n_epochs) bool. True = prover submitted ≥1 proof that epoch.
        booster: RewardBooster config.
        econ: Economic config (pool size, token price, default hardware cost).
        distribution:
          per_submission — each submitter independently receives pool * shares/K.
          pro_rata       — submitters split the pool weighted by shares.
        prover_costs_usd_per_epoch: optional (n_provers,) override of econ.hardware_cost_per_epoch_usd.
        online_mask: optional (n_provers, n_epochs) bool — True = prover is online and incurring
                     hardware cost. Default: always online. Provers that don't submit still burn cost
                     while online, modeling infra-running-but-losing-to-competition.

    Returns dict with per-epoch and aggregate metrics.
    """
    n_provers, n_epochs = submissions.shape
    scores = [ActivityScore() for _ in range(n_provers)]
    rewards = np.zeros((n_provers, n_epochs), dtype=float)
    shares_history = np.zeros((n_provers, n_epochs), dtype=float)

    for e in range(n_epochs):
        epoch = e + 1
        epoch_shares = np.zeros(n_provers, dtype=float)
        for i in range(n_provers):
            if submissions[i, e]:
                scores[i] = record_submission(scores[i], epoch, booster)
                epoch_shares[i] = to_shares(scores[i].value, booster)

        shares_history[:, e] = epoch_shares

        if distribution == "per_submission":
            rewards[:, e] = econ.prover_pool * epoch_shares / booster.k
        elif distribution == "pro_rata":
            total = epoch_shares.sum()
            if total > 0:
                rewards[:, e] = econ.prover_pool * epoch_shares / total
        else:
            raise ValueError(f"unknown distribution: {distribution}")

    if prover_costs_usd_per_epoch is None:
        prover_costs_usd_per_epoch = np.full(n_provers, econ.hardware_cost_per_epoch_usd, dtype=float)
    if online_mask is None:
        online_mask = np.ones((n_provers, n_epochs), dtype=bool)

    hw_costs = prover_costs_usd_per_epoch[:, None] * online_mask.astype(float)
    gas_costs = submissions.astype(float) * econ.gas_cost_per_submission_usd
    costs_per_epoch_usd = hw_costs + gas_costs
    rewards_usd = rewards * econ.aztec_usd
    pnl_per_epoch_usd = rewards_usd - costs_per_epoch_usd

    return {
        "rewards_per_epoch_aztec": rewards,
        "rewards_per_epoch_usd": rewards_usd,
        "costs_per_epoch_usd": costs_per_epoch_usd,
        "pnl_per_epoch_usd": pnl_per_epoch_usd,
        "shares_per_epoch": shares_history,
        "total_rewards_aztec": rewards.sum(axis=1),
        "total_rewards_usd": rewards_usd.sum(axis=1),
        "total_costs_usd": costs_per_epoch_usd.sum(axis=1),
        "profit_usd": pnl_per_epoch_usd.sum(axis=1),
    }
