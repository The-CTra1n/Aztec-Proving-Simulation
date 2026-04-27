"""Smoke-test the new PnL-aware simulation output with per-prover costs."""
import numpy as np
from proving_sim.real_data import load_submissions, submissions_matrix
from proving_sim.booster import BoostConfig
from proving_sim.simulation import EconomicConfig, simulate

df = load_submissions("data/submissions.parquet")
matrix, provers, _ = submissions_matrix(df, mode="active_only")
econ = EconomicConfig(200.0, 0, 0.02, 5.0)

rng = np.random.default_rng(42)
costs = np.clip(rng.normal(1.0, 0.2, size=matrix.shape[0]), 0.1, None) * 5.0

res = simulate(
    matrix, BoostConfig.proposed(), econ,
    distribution="per_submission",
    prover_costs_usd_per_epoch=costs,
)

print("shape checks:")
for k in ("rewards_per_epoch_usd", "costs_per_epoch_usd", "pnl_per_epoch_usd", "shares_per_epoch"):
    print(f"  {k}: {res[k].shape}")

print(f"\ntotals:")
print(f"  rewards: ${res['total_rewards_usd'].sum():,.0f}")
print(f"  costs:   ${res['total_costs_usd'].sum():,.0f}")
print(f"  pnl:     ${res['profit_usd'].sum():,.0f}")
print(f"  profitable provers: {(res['profit_usd'] > 0).sum()} / {len(res['profit_usd'])}")

top = res["profit_usd"].argsort()[::-1][:5]
print(f"\ntop 5 by PnL:")
for i in top:
    cov = matrix[i].mean() * 100
    print(f"  {provers[i][:12]}  cov={cov:>5.1f}%  cost=${costs[i]:.2f}  rewards=${res['total_rewards_usd'][i]:,.0f}  pnl=${res['profit_usd'][i]:,.0f}")
