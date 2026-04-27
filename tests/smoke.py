"""End-to-end smoke test — run from repo root with `python tests/smoke.py`."""
from proving_sim.synth import load_reference_csv, synthesize_submissions, infer_n_epochs
from proving_sim.booster import BoostConfig
from proving_sim.simulation import EconomicConfig, simulate

ref = load_reference_csv("reference-sim/reward_comparison.csv")
n = infer_n_epochs(ref)
print(f"provers={len(ref)} n_epochs={n}")

import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "bernoulli"
sub = synthesize_submissions(ref, n, mode=mode, seed=42)
print(f"mode={mode} submissions shape={sub.shape} density={sub.mean():.3f}")

econ = EconomicConfig(checkpoint_reward=200.0, sequencer_bps=0, aztec_usd=0.02, hardware_cost_per_epoch_usd=1.0)

for label, cfg, csv_col in [("current", BoostConfig.current(), "Current_Earned"),
                            ("proposed", BoostConfig.proposed(), "Proposed_Earned")]:
    res = simulate(sub, cfg, econ, distribution="per_submission")
    top = res["total_rewards_aztec"].argsort()[::-1][:3]
    total = res["total_rewards_aztec"].sum()
    print(f"\n{label}: total paid = {total:,.0f} AZTEC")
    for i in top:
        addr = ref["Address"].iloc[i][:10]
        cov = ref["Coverage%"].iloc[i]
        sim = res["total_rewards_aztec"][i]
        csv = ref[csv_col].iloc[i]
        print(f"  {addr}  cov={cov:.1f}%  sim={sim:>10,.0f}  csv={csv:>10,.0f}  ratio={sim/csv:.2f}")
