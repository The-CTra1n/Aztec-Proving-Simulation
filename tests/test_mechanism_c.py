"""Tests for Mechanism C (supply-target dynamic reward)."""

import math

import numpy as np
import pytest

from proving_sim.booster import BoostConfig
from proving_sim.mechanism_c import MechanismCConfig, adjusted_pool, simulate_mechanism_c


def test_adjusted_pool_endpoints():
    # One step: prev_pool × (1 + delta). At n=target, delta=0; at n=0, +max%;
    # at n=2*target, −max%; clamped beyond.
    assert adjusted_pool(500.0, n_submitters=5, target=5, max_move_pct=12.5) == pytest.approx(500.0)
    assert adjusted_pool(500.0, n_submitters=0, target=5, max_move_pct=12.5) == pytest.approx(562.5)
    assert adjusted_pool(500.0, n_submitters=10, target=5, max_move_pct=12.5) == pytest.approx(437.5)
    assert adjusted_pool(500.0, n_submitters=20, target=5, max_move_pct=12.5) == pytest.approx(437.5)
    assert adjusted_pool(500.0, n_submitters=2, target=4, max_move_pct=10.0) == pytest.approx(525.0)
    # Walks compound: applying twice from a fresh pool is multiplicative.
    once = adjusted_pool(500.0, n_submitters=0, target=5, max_move_pct=12.5)
    twice = adjusted_pool(once, n_submitters=0, target=5, max_move_pct=12.5)
    assert twice == pytest.approx(500.0 * 1.125 * 1.125)


def test_simulation_runs_and_shapes_match():
    cfg = MechanismCConfig(n_provers=10, n_epochs=100, target_provers=5, base_reward=500.0,
                           aztec_usd=0.02, hw_cost_mean=5.0, gas_cost_median_usd=2.0,
                           dropout_ratio=2.0, dropout_window=5, seed=7)
    res = simulate_mechanism_c(cfg, BoostConfig.current())
    assert res["rewards_per_epoch_aztec"].shape == (10, 100)
    assert res["pool_per_epoch_aztec"].shape == (100,)
    # Per-step move bounded by ±max_move_pct (the rule itself, not absolute level).
    pool = res["pool_per_epoch_aztec"]
    prev = np.concatenate(([500.0], pool[:-1]))
    step_ratio = pool / prev
    assert step_ratio.min() >= 1 - 0.125 - 1e-9
    assert step_ratio.max() <= 1 + 0.125 + 1e-9
    # Submitter count never exceeds n_provers
    assert res["submitters_per_epoch"].max() <= 10


def test_dropout_after_window_of_loss():
    # Force everyone unprofitable: hw=$100, base reward=1 AZTEC at $0.02
    cfg = MechanismCConfig(n_provers=4, n_epochs=20, target_provers=2, max_move_pct=12.5,
                           base_reward=1.0, aztec_usd=0.02, hw_cost_mean=100.0,
                           hw_cost_stddev_pct=0.0, gas_cost_median_usd=0.0,
                           dropout_ratio=2.0, dropout_window=5, seed=1)
    res = simulate_mechanism_c(cfg, BoostConfig.current())
    # By the end everyone should have dropped at least once
    final_active = res["active_per_epoch"][:, -1]
    assert final_active.sum() == 0


def test_no_dropout_when_profitable():
    # Generous base, near-zero costs → no one drops
    cfg = MechanismCConfig(n_provers=4, n_epochs=50, target_provers=4, max_move_pct=12.5,
                           base_reward=100_000.0, aztec_usd=0.02, hw_cost_mean=0.01,
                           hw_cost_stddev_pct=0.0, gas_cost_median_usd=0.0,
                           dropout_ratio=2.0, dropout_window=5, seed=2)
    res = simulate_mechanism_c(cfg, BoostConfig.current())
    assert res["active_per_epoch"][:, -1].all()
    # Submitter count == n_provers throughout
    assert (res["submitters_per_epoch"] == 4).all()


def test_pool_above_base_when_undersupplied():
    # Submit only via 2 of target=10 provers' worth of activity — by forcing many drops
    cfg = MechanismCConfig(n_provers=10, n_epochs=200, target_provers=20, max_move_pct=12.5,
                           base_reward=500.0, aztec_usd=0.02, hw_cost_mean=5.0,
                           gas_cost_median_usd=2.0, dropout_ratio=2.0, dropout_window=5, seed=3)
    res = simulate_mechanism_c(cfg, BoostConfig.current())
    # 10 provers vs target=20 → always below target → pool above base
    assert res["pool_per_epoch_aztec"].mean() > 500.0
