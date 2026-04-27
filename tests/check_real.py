"""Quick sanity check on fetched real data."""
from proving_sim.real_data import load_submissions, submissions_matrix, coverage_summary
from proving_sim.booster import BoostConfig
from proving_sim.simulation import EconomicConfig, simulate

df = load_submissions("data/submissions.parquet")
print(f"logs: {len(df):,}  checkpoints: {df.checkpoint.min()}..{df.checkpoint.max()}  provers: {df.prover.nunique()}")

matrix, provers, checkpoints = submissions_matrix(df)
print(f"matrix: {matrix.shape}  density: {matrix.mean():.3f}  active checkpoints: {len(checkpoints)}")

summary = coverage_summary(matrix, provers)
print("\ntop 10 provers by submission count:")
print(summary.head(10).to_string(index=False))
print("\ncoverage distribution:")
print(summary.coverage_pct.describe().to_string())

prover_to_cov = summary.set_index("prover")["coverage_pct"].to_dict()

econ = EconomicConfig(200.0, 0, 0.02, 1.0)
for label, cfg in [("current", BoostConfig.current()), ("proposed", BoostConfig.proposed())]:
    res = simulate(matrix, cfg, econ, distribution="per_submission")
    top = res["total_rewards_aztec"].argsort()[::-1][:5]
    total = res["total_rewards_aztec"].sum()
    print(f"\n{label}: total = {total:,.0f} AZTEC")
    for i in top:
        addr = provers[i]
        cov = prover_to_cov[addr]
        rew = res["total_rewards_aztec"][i]
        print(f"  {addr[:10]}  cov={cov:>5.1f}%  rewards={rew:>10,.0f} AZTEC")
