"""Mechanism C — supply-target dynamic reward.

Each epoch the pool walks multiplicatively from the previous epoch's pool:
    pool[e] = pool[e-1] × (1 + delta(n_submitters[e]))
    delta   = max_move_pct/100 × clip((target − n_submitters)/target, -1, 1)

The update rule is memoryless (delta depends only on the current submitter
count) but the pool accumulates: a sustained undersupply ratchets the pool
up indefinitely; oversupply ratchets it down. Initial pool = base_reward.
Synthetic scenario for testing undersupply behaviour (real data has too many
provers to ever fall below realistic targets).

Provers drop out after `dropout_window` consecutive active epochs in which
cost ≥ dropout_ratio × reward. They re-enter when their hypothetical reward
at max booster shares (k) would beat their expected per-epoch cost.

Booster math is plugged in unchanged — Mechanism C is run twice (once with
booster A's params, once with booster B's) over the same prover realisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from proving_sim.booster import ActivityScore, BoostConfig, record_submission, to_shares


# Lognormal fitted to observed L1 gas-cost distribution at ETH=$3500
# (median 1.97, mean 3.44, p90 6.42 → σ ≈ 1.055).
GAS_LOGNORMAL_SIGMA = 1.05


@dataclass(frozen=True)
class MechanismCConfig:
    n_provers: int = 10
    n_epochs: int = 987
    target_provers: int = 5
    max_move_pct: float = 12.5
    base_reward: float = 500.0
    aztec_usd: float = 0.02
    hw_cost_mean: float = 5.0
    hw_cost_stddev_pct: float = 30.0
    gas_cost_median_usd: float = 2.0
    gas_lognormal_sigma: float = GAS_LOGNORMAL_SIGMA
    dropout_ratio: float = 2.0
    dropout_window: int = 5
    seed: int = 42
    # Per-prover hardware cost override (USD/epoch). When set, takes precedence
    # over hw_cost_mean / hw_cost_stddev_pct. Length must equal n_provers.
    hw_costs_override: tuple[float, ...] | None = None


def adjusted_pool(prev_pool: float, n_submitters: int, target: int, max_move_pct: float) -> float:
    """One memoryless step of the pool walk.

    Returns prev_pool × (1 + delta), where delta scales linearly with the
    distance from target and is clamped to ±max_move_pct%.
    """
    if target <= 0:
        return prev_pool
    raw = (target - n_submitters) / target
    delta = max_move_pct / 100.0 * max(-1.0, min(1.0, raw))
    return prev_pool * (1.0 + delta)


def _hardware_costs(cfg: MechanismCConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.hw_costs_override is not None:
        if len(cfg.hw_costs_override) != cfg.n_provers:
            raise ValueError(
                f"hw_costs_override length {len(cfg.hw_costs_override)} != n_provers {cfg.n_provers}"
            )
        return np.array(cfg.hw_costs_override, dtype=float)
    if cfg.hw_cost_stddev_pct <= 0:
        return np.full(cfg.n_provers, cfg.hw_cost_mean, dtype=float)
    factors = rng.normal(loc=1.0, scale=cfg.hw_cost_stddev_pct / 100.0, size=cfg.n_provers)
    return np.clip(factors, 0.1, None) * cfg.hw_cost_mean


def _gas_costs(cfg: MechanismCConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-epoch L1 gas cost. One draw per epoch, shared by all provers in that epoch
    (gas is a property of the L1 block, not the prover)."""
    if cfg.gas_cost_median_usd <= 0:
        return np.zeros(cfg.n_epochs, dtype=float)
    mu = math.log(cfg.gas_cost_median_usd)
    return rng.lognormal(mean=mu, sigma=cfg.gas_lognormal_sigma, size=cfg.n_epochs)


def simulate_mechanism_c(cfg: MechanismCConfig, booster: BoostConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n, T = cfg.n_provers, cfg.n_epochs

    hw = _hardware_costs(cfg, rng)
    gas = _gas_costs(cfg, rng)
    expected_gas = (
        math.exp(math.log(cfg.gas_cost_median_usd) + cfg.gas_lognormal_sigma**2 / 2.0)
        if cfg.gas_cost_median_usd > 0 else 0.0
    )

    scores = [ActivityScore() for _ in range(n)]
    active = np.ones(n, dtype=bool)
    consecutive_active = np.zeros(n, dtype=int)

    submissions = np.zeros((n, T), dtype=bool)
    rewards_aztec = np.zeros((n, T), dtype=float)
    shares_history = np.zeros((n, T), dtype=float)
    costs_usd = np.zeros((n, T), dtype=float)
    pool_history = np.zeros(T, dtype=float)
    submitters_history = np.zeros(T, dtype=int)
    active_history = np.zeros((n, T), dtype=bool)

    prev_pool = cfg.base_reward

    for e in range(T):
        epoch = e + 1

        # ── Re-entry check (uses prior-epoch state as snapshot) ──
        if e > 0:
            prev_shares = shares_history[:, e - 1]
            for i in range(n):
                if active[i]:
                    continue
                other_active_mask = active.copy()
                other_active_mask[i] = False
                sum_other_shares = float(prev_shares[other_active_mask].sum())
                hypo_n = int(other_active_mask.sum()) + 1
                pool_hypo = adjusted_pool(
                    prev_pool, hypo_n, cfg.target_provers, cfg.max_move_pct
                )
                hypo_share = booster.k
                denom = sum_other_shares + hypo_share
                hypo_reward_aztec = pool_hypo * hypo_share / denom if denom > 0 else 0.0
                hypo_reward_usd = hypo_reward_aztec * cfg.aztec_usd
                if hypo_reward_usd >= hw[i] + expected_gas:
                    active[i] = True
                    consecutive_active[i] = 0  # grace window restarts

        # ── Submissions + booster shares ──
        epoch_shares = np.zeros(n, dtype=float)
        for i in range(n):
            if active[i]:
                scores[i] = record_submission(scores[i], epoch, booster)
                epoch_shares[i] = to_shares(scores[i].value, booster)
                submissions[i, e] = True
                consecutive_active[i] += 1
            else:
                consecutive_active[i] = 0

        n_submitters = int(submissions[:, e].sum())
        submitters_history[e] = n_submitters
        pool = adjusted_pool(
            prev_pool, n_submitters, cfg.target_provers, cfg.max_move_pct
        )
        pool_history[e] = pool
        prev_pool = pool

        # Pro-rata over active provers' shares
        total_shares = epoch_shares.sum()
        if total_shares > 0:
            rewards_aztec[:, e] = pool * epoch_shares / total_shares

        shares_history[:, e] = epoch_shares
        active_history[:, e] = active

        # ── Costs ──
        costs_usd[:, e] = active.astype(float) * (hw + gas[e])

        # ── Dropout check: last `window` epochs all satisfy cost ≥ ratio × reward ──
        for i in range(n):
            if not active[i] or consecutive_active[i] < cfg.dropout_window:
                continue
            lo = e - cfg.dropout_window + 1
            window_cost = costs_usd[i, lo : e + 1]
            window_reward_usd = rewards_aztec[i, lo : e + 1] * cfg.aztec_usd
            if np.all(window_cost >= cfg.dropout_ratio * window_reward_usd):
                active[i] = False
                consecutive_active[i] = 0

    rewards_usd = rewards_aztec * cfg.aztec_usd
    pnl_per_epoch_usd = rewards_usd - costs_usd

    return {
        "rewards_per_epoch_aztec": rewards_aztec,
        "rewards_per_epoch_usd": rewards_usd,
        "costs_per_epoch_usd": costs_usd,
        "pnl_per_epoch_usd": pnl_per_epoch_usd,
        "shares_per_epoch": shares_history,
        "submissions": submissions,
        "active_per_epoch": active_history,
        "submitters_per_epoch": submitters_history,
        "pool_per_epoch_aztec": pool_history,
        "hw_costs": hw,
        "gas_costs": gas,
        "total_rewards_aztec": rewards_aztec.sum(axis=1),
        "total_rewards_usd": rewards_usd.sum(axis=1),
        "total_costs_usd": costs_usd.sum(axis=1),
        "profit_usd": pnl_per_epoch_usd.sum(axis=1),
    }
