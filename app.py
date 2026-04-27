"""Streamlit interactive page for the Aztec prover reward simulation.

Launch: `streamlit run app.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from proving_sim.booster import ActivityScore, BoostConfig, record_submission, to_shares
from proving_sim.mechanism_c import MechanismCConfig, simulate_mechanism_c
from proving_sim.real_data import coverage_summary, gas_series, load_submissions, submissions_matrix
from proving_sim.simulation import EconomicConfig, simulate

REAL_SUBMISSIONS = Path(__file__).parent / "data" / "submissions.parquet"

st.set_page_config(page_title="Aztec prover reward simulation", layout="wide")
st.title("Aztec prover reward simulation")
st.caption(
    "Real per-checkpoint submissions fetched from L1 (Rollup 0xae2001…) · "
    "booster math ports RewardBooster.sol line-for-line."
)

with st.expander("About the mechanisms", expanded=False):
    st.markdown(
        """
**Mechanism A — current `RewardBooster`** (deployed at `0xae2001…`).
Each prover has an integer activity score. A submission adds `increment` (default
**125,000**), capped at `maxScore` (**15,000,000**). Inactivity decays the score
by 100,000/epoch (hard-coded in the contract). Shares paid per checkpoint are a
quadratic function: `k − a·(maxScore − score)² / 1e10`, floored at `minimum`
(**100,000**) and ceilinged at `k` (**1,000,000**). With these params it takes
~120 consecutive submissions to reach the cap, and most provers sit near the
floor most of the time.

**Mechanism B — proposed `RewardBooster`** (governance proposal). Same math,
much sharper. `maxScore=367,500` (40× lower) and `increment=101,400` → reach
the cap in ~4 submissions. `a=250,000` (250× higher) makes the quadratic bite
hard for any score below max. `minimum=10,000` (10× lower) so missing the cap
genuinely hurts. Net effect: rewards consistency aggressively, punishes
inconsistency aggressively.

**Mechanism C — supply-target dynamic pool**. Wraps either A or B. The
per-checkpoint reward pool is no longer fixed: it scales linearly with the
gap between actual submitters and a target —
`pool = base × (1 + max_move × clamp((target − n)/target, ±1))`. Below target
the pool grows (max +max_move%); above 2×target it's capped at −max_move%.
Uses 10 synthetic provers because real data has 76 active and never falls
below any plausible target. Provers drop out after `window` consecutive
epochs with `cost ≥ ratio × reward`, and re-enter when their hypothetical
reward at maximum booster shares would beat their expected per-epoch cost.
        """
    )

with st.expander("Parameter guide", expanded=False):
    st.markdown(
        """
**Booster (A and B)** — both scenarios share the same math, only the params differ.
- `increment` — score gained per submission. ↑ ramps to max faster · ↓ slower ramp, bigger relative cost of a miss.
- `maxScore` — score ceiling. ↑ harder to reach max (most provers stuck near floor) · ↓ easier; tighter response window.
- `a` — quadratic coefficient. ↑ shares fall off the cap faster (steeper penalty below max) · ↓ flatter curve.
- `minimum` — share floor when far from max. ↑ bigger consolation prize · ↓ harsher penalty for low activity.
- `k` — share cap, paid when at maxScore. Sets the unit; only meaningful relative to `a` and `minimum`.

**Economic** (apply to both A and B; Mechanism C reuses these too):
- `checkpoint reward` — AZTEC pool minted per checkpoint. On-chain default is **500** (read from `Rollup.getCheckpointReward()`). ↑ proportionally scales all rewards.
- `sequencer split (bps)` — fraction taken by the sequencer before the prover pool. ↑ less for provers.
- `AZTEC → USD` — token price for USD profitability conversion. ↑ rewards worth more in USD; provers more profitable.
- `Hardware cost per epoch` — baseline infra cost paid every epoch a prover is online (sunk in mechanisms A/B; gated on `active` in C). Default **$1.14/epoch** is derived from the conservative 1-yr reserved AWS `c6i.16xlarge` price (~$1,300/mo, 64 vCPU / 128 GiB, matching the 32-core / 64 vCPU / 128 GB / 10 GB SSD reference spec). Epoch = 32 checkpoints × 72 s = 2304 s, giving ~1,141 epochs/month. Pricing reference: [instances.vantage.sh/aws/ec2/c6i.16xlarge](https://instances.vantage.sh/aws/ec2/c6i.16xlarge). Other points on the curve: on-demand ≈$1.71–$1.84/epoch, 3-yr reserved ≈$0.66–$0.74/epoch, spot ≈$0.44–$0.79/epoch.
- `Cost heterogeneity %` — stddev of per-prover hw cost around the baseline. ↑ wider spread → some provers very profitable while others bleed.
- `Proof submission gas cost` — L1 gas paid only on epochs the prover submits a proof. The sidebar value is the **median** of a lognormal whose σ is fitted to observed on-chain data (held constant). Observed defaults: median ≈ $2 at ETH=$3500.
- `reward distribution`:
  - `per_submission` — each submitter independently receives `pool × shares/k`. Total emission can exceed the pool when many submit.
  - `pro_rata` — `pool` is split among submitters weighted by shares. Total emission = pool exactly.

**Mechanism C** (only used in the Mechanism C tab):
- `provers` — size of the synthetic prover set. Default 10. ↑ approaches real-data behaviour; ↓ stresses undersupply dynamics.
- `epochs` — simulation length. Default 987 matches the real-data window so booster decay/build behaviour is comparable.
- `target submitters` — ideal count per epoch. At n=target the pool is unchanged; below target it ratchets up, above target it ratchets down.
- `max move %` — per-epoch step cap (±%). The pool walks multiplicatively: `pool[e] = pool[e-1] × (1 + delta)`, where `delta` is bounded by ±max_move_pct and scaled by distance from target. Sustained undersupply ramps the pool indefinitely; oversupply decays it. Default 12.5%.
- `dropout cost:reward ratio` — drop trigger threshold. ↑ provers tolerate more loss before quitting · ↓ fast attrition.
- `dropout window` — consecutive epochs of loss required to drop. ↑ smoother, ignores noisy single-epoch spikes · ↓ trigger-happy.

**Seed** — controls hardware-cost sampling and Mechanism C's gas/hw realisations. Same seed → reproducible run.
        """
    )


# ──────────────── Cached data ────────────────
@st.cache_data
def load_real() -> tuple[np.ndarray, list[str], pd.DataFrame, list[int], pd.DataFrame]:
    df = load_submissions(REAL_SUBMISSIONS)
    matrix, provers, checkpoints = submissions_matrix(df, mode="active_only")
    summary = coverage_summary(matrix, provers)
    return matrix, provers, summary, checkpoints, df


def get_submissions() -> tuple[np.ndarray, pd.DataFrame]:
    matrix, provers, summary, _checkpoints, _raw = load_real()
    labels = pd.DataFrame({
        "Address": provers,
        "Coverage%": summary.set_index("prover").loc[provers, "coverage_pct"].to_numpy(),
    })
    return matrix, labels


@st.cache_data
def get_gas_series(eth_usd: float) -> pd.DataFrame:
    _matrix, _provers, _summary, checkpoints, raw_df = load_real()
    return gas_series(raw_df, checkpoints, eth_usd)


def _prover_costs(n_provers: int, baseline: float, stddev_pct: float, seed: int) -> np.ndarray:
    if stddev_pct <= 0:
        return np.full(n_provers, baseline, dtype=float)
    rng = np.random.default_rng(seed)
    factors = rng.normal(loc=1.0, scale=stddev_pct / 100.0, size=n_provers)
    return np.clip(factors, 0.1, None) * baseline


@st.cache_data
def run_sim_real(
    cfg_tuple: tuple,
    econ_tuple: tuple,
    distribution: str,
    cost_stddev_pct: float,
    cost_seed: int,
) -> dict:
    matrix, _provers, _summary, _ckpts, _raw = load_real()
    cfg = BoostConfig(*cfg_tuple)
    econ = EconomicConfig(*econ_tuple)
    costs = _prover_costs(matrix.shape[0], econ.hardware_cost_per_epoch_usd, cost_stddev_pct, cost_seed)
    return simulate(matrix, cfg, econ, distribution=distribution, prover_costs_usd_per_epoch=costs)


@st.cache_data
def run_mechanism_c(cfg_tuple: tuple, mc_tuple: tuple) -> dict:
    return simulate_mechanism_c(MechanismCConfig(*mc_tuple), BoostConfig(*cfg_tuple))


# ──────────────── Sidebar ────────────────
st.sidebar.header("🌐 Global parameters")
st.sidebar.caption("Applied to every mechanism and every tab.")
aztec_usd = st.sidebar.number_input(
    "AZTEC → USD", 0.0, 10.0, 0.02, 0.001, format="%.4f", key="global_aztec_usd",
    help="Token price used for all USD profitability conversions (Mechanisms A, B, and C).",
)
eth_usd = st.sidebar.number_input(
    "ETH → USD", 0.0, 100_000.0, 3500.0, 50.0, format="%.2f", key="global_eth_usd",
    help=(
        "Used to price the historical gas series (gas_used × gas_price → USD) for the "
        "real-data gas overlay. Does not affect Mechanism C's synthetic gas draw."
    ),
)
st.sidebar.divider()

st.sidebar.header("Scenario A — booster params")
aA = st.sidebar.columns(2)
sA_increment = aA[0].number_input("increment", 1, 10_000_000, 125_000, 1_000, key="a_inc")
sA_max = aA[1].number_input("maxScore", 1, 100_000_000, 15_000_000, 10_000, key="a_max")
sA_a = aA[0].number_input("a", 1, 10_000_000, 1_000, 100, key="a_a")
sA_min = aA[1].number_input("minimum", 0, 10_000_000, 100_000, 1_000, key="a_min")
sA_k = st.sidebar.number_input("k (max shares)", 1, 10_000_000, 1_000_000, 10_000, key="a_k")

st.sidebar.header("Scenario B — booster params")
aB = st.sidebar.columns(2)
sB_increment = aB[0].number_input("increment", 1, 10_000_000, 101_400, 1_000, key="b_inc")
sB_max = aB[1].number_input("maxScore", 1, 100_000_000, 367_500, 1_000, key="b_max")
sB_a = aB[0].number_input("a", 1, 10_000_000, 250_000, 1_000, key="b_a")
sB_min = aB[1].number_input("minimum", 0, 10_000_000, 10_000, 1_000, key="b_min")
sB_k = st.sidebar.number_input("k (max shares)", 1, 10_000_000, 1_000_000, 10_000, key="b_k")

st.sidebar.header("Economic")
checkpoint_reward = st.sidebar.number_input(
    "checkpoint reward (AZTEC / checkpoint)", 0.0, 100_000.0, 500.0, 10.0, format="%.2f",
    help="On-chain default from Rollup.getCheckpointReward() = 500 AZTEC. Tunable.",
)
sequencer_bps = st.sidebar.slider("sequencer split (bps)", 0, 10_000, 0, 100)
hw_cost = st.sidebar.number_input(
    "Hardware cost per epoch (USD/prover)", 0.0, 10_000.0, 1.14, 0.01, format="%.2f",
    help=(
        "Default: **$1.14/epoch** ≈ $1,300/month. Epoch = 32 checkpoints × 72 s = 2304 s "
        "(~1,141 epochs/month).\n\n"
        "Spec basis: 32 core / 64 vCPU, 128 GB RAM, 10 GB SSD — closest match is "
        "AWS `c6i.16xlarge` / `c7i.16xlarge` (64 vCPU, 128 GiB).\n\n"
        "Cost options for that spec (Apr 2026):\n"
        "- On-demand: ~$1,950–2,100/mo (~$2.72/hr) → ~$1.71–$1.84/epoch\n"
        "- **1-yr reserved (no upfront): ~$1,200–1,300/mo → ~$1.05–$1.14/epoch ← default uses high end**\n"
        "- 3-yr reserved (all upfront): ~$750–850/mo → ~$0.66–$0.74/epoch\n"
        "- Spot: ~$500–900/mo (volatile) → ~$0.44–$0.79/epoch\n\n"
        "Reference: https://instances.vantage.sh/aws/ec2/c6i.16xlarge"
    ),
)
hw_cost_stddev_pct = st.sidebar.slider(
    "Cost heterogeneity (% stddev)", 0, 100, 0, 5,
    help="Randomize each prover's cost by N% around the baseline to simulate different infra choices.",
)
gas_cost = st.sidebar.number_input(
    "Proof submission gas cost (USD)", 0.0, 1_000.0, 2.0, 0.5, format="%.2f",
    help="L1 gas charged only on epochs where a prover submits a proof. Observed median from on-chain data ≈ $2 (mean $3.4, p90 $6.4) at ETH=$3500.",
)
show_gas_overlay = st.sidebar.checkbox(
    "Overlay historical gas price", value=False,
    help="Shows median gas price (gwei) per checkpoint on a secondary axis in the real-data time-series tabs, to help diagnose spikes. Uses the global ETH → USD value above.",
)
distribution = st.sidebar.selectbox(
    "reward distribution",
    ["per_submission", "pro_rata"],
    index=0,
    help="per_submission: each submitter independently receives pool × shares/K. "
    "pro_rata: submitters split the pool weighted by shares.",
)

st.sidebar.header("Data")
if not REAL_SUBMISSIONS.exists():
    st.sidebar.error(f"Missing {REAL_SUBMISSIONS.name}. Run `python -m proving_sim.fetch`.")
    st.stop()
data_source = "real (fetched L1 logs)"
seed = st.sidebar.number_input("seed", 0, 100_000, 42, 1, help="Used for hardware-cost heterogeneity.")

st.sidebar.header("Mechanism C — supply-target dynamic")
mc_n_provers = st.sidebar.number_input("provers", 2, 100, 10, 1, key="mc_n",
    help="Synthetic prover set, separate from real-data tabs.")
mc_n_epochs = st.sidebar.number_input("epochs", 10, 50_000, 987, 10, key="mc_T",
    help="Default 987 matches the real-data length.")
mc_target = st.sidebar.number_input("target submitters", 1, 100, 5, 1, key="mc_target",
    help="Pool flat at n=target, +max% at n=0, −max% at n≥2·target.")
mc_base_reward = st.sidebar.number_input(
    "starting per-epoch reward (AZTEC)", 0.0, 1_000_000.0, float(checkpoint_reward), 10.0,
    key="mc_base_reward", format="%.2f",
    help="Pool minted per epoch at n_submitters = target. Defaults to the global checkpoint reward "
    "but can be tuned independently for Mechanism C.",
)
mc_max_move = st.sidebar.slider("max move %", 0.0, 50.0, 12.5, 0.5, key="mc_move", format="%.1f")
mc_dropout_ratio = st.sidebar.number_input("dropout cost:reward ratio", 1.0, 10.0, 2.0, 0.1, key="mc_drop_r",
    format="%.2f",
    help="Drop after `window` consecutive epochs with cost ≥ ratio × reward.")
mc_dropout_window = st.sidebar.number_input("dropout window (epochs)", 1, 100, 5, 1, key="mc_drop_w")

st.sidebar.markdown("**Per-prover hardware cost (USD/epoch)**")
st.sidebar.caption(
    "Edit individual provers' costs. Overrides the global `Hardware cost per epoch` and "
    "`Cost heterogeneity %` for the Mechanism C tab."
)
_mc_costs_default = pd.DataFrame({
    "Prover": [f"P{i}" for i in range(int(mc_n_provers))],
    "HW $/epoch": [float(hw_cost)] * int(mc_n_provers),
})
mc_costs_df = st.sidebar.data_editor(
    _mc_costs_default,
    key=f"mc_costs_editor_{int(mc_n_provers)}_{float(hw_cost):.4f}",
    num_rows="fixed",
    hide_index=True,
    disabled=["Prover"],
    column_config={
        "HW $/epoch": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.4f"),
    },
)
mc_hw_costs = tuple(float(v) for v in mc_costs_df["HW $/epoch"].tolist())


# ──────────────── Run ────────────────
cfg_A = (sA_increment, sA_max, sA_a, sA_min, sA_k)
cfg_B = (sB_increment, sB_max, sB_a, sB_min, sB_k)
econ = (float(checkpoint_reward), int(sequencer_bps), float(aztec_usd), float(hw_cost), float(gas_cost))

submissions, labels = get_submissions()
res_A = run_sim_real(cfg_A, econ, distribution, float(hw_cost_stddev_pct), int(seed))
res_B = run_sim_real(cfg_B, econ, distribution, float(hw_cost_stddev_pct), int(seed))
active_epochs = submissions.shape[1]


# ──────────────── Headline metrics ────────────────
cols = st.columns(4)
cols[0].metric("Provers", len(labels))
cols[1].metric("Epochs", active_epochs)
cols[2].metric("A · total USD paid", f"${res_A['total_rewards_usd'].sum():,.0f}")
cols[3].metric("B · total USD paid", f"${res_B['total_rewards_usd'].sum():,.0f}")


# ──────────────── Tabs ────────────────
tab7, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Mechanism C", "Per-prover rewards", "Cumulative over time", "Coverage → share curve",
     "Profitability", "PnL over time", "Prover drill-down"]
)

with tab1:
    df = pd.DataFrame(
        {
            "Address": labels["Address"].astype(str).str[:10] + "…",
            "Coverage%": labels["Coverage%"],
            "A (AZTEC)": res_A["total_rewards_aztec"],
            "B (AZTEC)": res_B["total_rewards_aztec"],
        }
    ).sort_values("Coverage%", ascending=False)

    fig = go.Figure()
    fig.add_bar(x=df["Address"], y=df["A (AZTEC)"], name="Scenario A (sim)")
    fig.add_bar(x=df["Address"], y=df["B (AZTEC)"], name="Scenario B (sim)")
    fig.update_layout(barmode="group", height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

with tab2:
    cum_A = np.cumsum(res_A["rewards_per_epoch_aztec"], axis=1)
    cum_B = np.cumsum(res_B["rewards_per_epoch_aztec"], axis=1)
    top = labels.nlargest(10, "Coverage%").index.to_list()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i in top:
        addr = str(labels["Address"].iloc[i])
        label_txt = f"{addr[:8]}… ({labels['Coverage%'].iloc[i]:.1f}%)"
        fig.add_scatter(y=cum_A[i], mode="lines", name=f"A · {label_txt}",
                        legendgroup="A", secondary_y=False)
        fig.add_scatter(y=cum_B[i], mode="lines", line=dict(dash="dash"),
                        name=f"B · {label_txt}", legendgroup="B", secondary_y=False)
    if show_gas_overlay:
        gas_df = get_gas_series(float(eth_usd))
        fig.add_scatter(y=gas_df["gas_price_gwei"], mode="lines",
                        name="gas (gwei, median)", legendgroup="gas",
                        line=dict(color="rgba(140,140,140,0.55)", width=1),
                        secondary_y=True)
    fig.update_layout(height=500, xaxis_title="Epoch")
    fig.update_yaxes(title_text="Cumulative reward (AZTEC)", secondary_y=False)
    fig.update_yaxes(title_text="Gas price (gwei)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top 10 provers by coverage — solid lines = A, dashed = B."
               + (" Grey line = median gas price (gwei) at each active checkpoint." if show_gas_overlay else ""))

with tab3:
    # Long-run mean share when submitting at steady-state coverage p.
    coverages = np.linspace(0.0, 1.0, 101)

    def steady_share(p: float, cfg: BoostConfig) -> float:
        rng = np.random.default_rng(0)
        score = ActivityScore()
        total_shares, submits = 0.0, 0
        for e in range(1, 2001):
            if rng.random() < p:
                score = record_submission(score, e, cfg)
                total_shares += to_shares(score.value, cfg)
                submits += 1
        return total_shares / submits / cfg.k if submits else 0.0

    cfg_A_obj = BoostConfig(*cfg_A)
    cfg_B_obj = BoostConfig(*cfg_B)
    shares_A = [steady_share(float(p), cfg_A_obj) for p in coverages]
    shares_B = [steady_share(float(p), cfg_B_obj) for p in coverages]

    fig = go.Figure()
    fig.add_scatter(x=coverages * 100, y=shares_A, name="Scenario A", mode="lines")
    fig.add_scatter(x=coverages * 100, y=shares_B, name="Scenario B", mode="lines")
    fig.update_layout(
        height=500, xaxis_title="Steady-state coverage %", yaxis_title="Mean share / K (fraction of max)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Simulated steady state at Bernoulli(p) submission for 2000 epochs.")

with tab4:
    df = pd.DataFrame(
        {
            "Address": labels["Address"].astype(str).str[:10] + "…",
            "Coverage%": labels["Coverage%"],
            "A rewards USD": res_A["total_rewards_usd"],
            "A costs USD": res_A["total_costs_usd"],
            "A profit USD": res_A["profit_usd"],
            "B rewards USD": res_B["total_rewards_usd"],
            "B costs USD": res_B["total_costs_usd"],
            "B profit USD": res_B["profit_usd"],
            "Δ profit (B − A)": res_B["profit_usd"] - res_A["profit_usd"],
        }
    ).sort_values("Coverage%", ascending=False)
    st.dataframe(df, use_container_width=True)

    agg = pd.DataFrame(
        {
            "Scenario": ["A", "B"],
            "Total rewards USD": [df["A rewards USD"].sum(), df["B rewards USD"].sum()],
            "Total costs USD": [df["A costs USD"].sum(), df["B costs USD"].sum()],
            "Total profit USD": [df["A profit USD"].sum(), df["B profit USD"].sum()],
            "Profitable provers": [(df["A profit USD"] > 0).sum(), (df["B profit USD"] > 0).sum()],
        }
    )
    st.dataframe(agg, use_container_width=True, hide_index=True)


with tab5:
    st.caption("Cumulative PnL (rewards − hardware costs) in USD across selected provers.")
    scenario_for_pnl = st.radio("scenario", ["A", "B", "both"], index=2, horizontal=True, key="pnl_scenario")
    default_top = labels.nlargest(10, "Coverage%").index.to_list()
    all_idx = list(range(len(labels)))
    sel_labels = [f"{str(labels['Address'].iloc[i])[:10]}… ({labels['Coverage%'].iloc[i]:.1f}%)" for i in all_idx]
    selected = st.multiselect(
        "provers", options=all_idx, default=default_top,
        format_func=lambda i: sel_labels[i], key="pnl_select",
    )
    if selected:
        cum_A = np.cumsum(res_A["pnl_per_epoch_usd"], axis=1)
        cum_B = np.cumsum(res_B["pnl_per_epoch_usd"], axis=1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for i in selected:
            label_txt = sel_labels[i]
            if scenario_for_pnl in ("A", "both"):
                fig.add_scatter(y=cum_A[i], mode="lines", name=f"A · {label_txt}",
                                legendgroup="A", secondary_y=False)
            if scenario_for_pnl in ("B", "both"):
                fig.add_scatter(y=cum_B[i], mode="lines", line=dict(dash="dash"),
                                name=f"B · {label_txt}", legendgroup="B", secondary_y=False)
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        if show_gas_overlay:
            gas_df = get_gas_series(float(eth_usd))
            fig.add_scatter(y=gas_df["gas_price_gwei"], mode="lines",
                            name="gas (gwei, median)", legendgroup="gas",
                            line=dict(color="rgba(140,140,140,0.55)", width=1),
                            secondary_y=True)
        fig.update_layout(height=600, xaxis_title="Epoch")
        fig.update_yaxes(title_text="Cumulative PnL (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Gas price (gwei)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


with tab6:
    st.caption("Deep-dive on one prover: score trajectory, per-epoch rewards, PnL under both scenarios.")
    options = list(range(len(labels)))
    selected_one = st.selectbox(
        "prover", options=options,
        format_func=lambda i: f"{str(labels['Address'].iloc[i])[:12]}… ({labels['Coverage%'].iloc[i]:.1f}%)",
        key="drill_select",
    )

    epochs_axis = np.arange(active_epochs)
    fig_shares = go.Figure()
    fig_shares.add_scatter(x=epochs_axis, y=res_A["shares_per_epoch"][selected_one],
                           mode="lines", name="A shares", legendgroup="A")
    fig_shares.add_scatter(x=epochs_axis, y=res_B["shares_per_epoch"][selected_one],
                           mode="lines", line=dict(dash="dash"), name="B shares", legendgroup="B")
    fig_shares.update_layout(height=300, xaxis_title="Epoch", yaxis_title="Shares (0 on missed epochs)",
                             title="Shares per epoch")
    st.plotly_chart(fig_shares, use_container_width=True)

    fig_rewards = go.Figure()
    fig_rewards.add_scatter(x=epochs_axis, y=np.cumsum(res_A["rewards_per_epoch_usd"][selected_one]),
                            mode="lines", name="A cum. rewards")
    fig_rewards.add_scatter(x=epochs_axis, y=np.cumsum(res_B["rewards_per_epoch_usd"][selected_one]),
                            mode="lines", line=dict(dash="dash"), name="B cum. rewards")
    fig_rewards.add_scatter(x=epochs_axis, y=np.cumsum(res_A["costs_per_epoch_usd"][selected_one]),
                            mode="lines", name="costs (same for A/B)", line=dict(color="gray"))
    fig_rewards.update_layout(height=300, xaxis_title="Epoch", yaxis_title="USD",
                              title="Cumulative rewards vs. costs")
    st.plotly_chart(fig_rewards, use_container_width=True)

    pnl_A = np.cumsum(res_A["pnl_per_epoch_usd"][selected_one])
    pnl_B = np.cumsum(res_B["pnl_per_epoch_usd"][selected_one])
    fig_pnl = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pnl.add_scatter(x=epochs_axis, y=pnl_A, mode="lines", name="A PnL", secondary_y=False)
    fig_pnl.add_scatter(x=epochs_axis, y=pnl_B, mode="lines", line=dict(dash="dash"),
                        name="B PnL", secondary_y=False)
    fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
    if show_gas_overlay:
        gas_df = get_gas_series(float(eth_usd))
        fig_pnl.add_scatter(x=epochs_axis, y=gas_df["gas_price_gwei"], mode="lines",
                            name="gas (gwei, median)",
                            line=dict(color="rgba(140,140,140,0.55)", width=1),
                            secondary_y=True)
    fig_pnl.update_layout(height=300, xaxis_title="Epoch",
                          title="Cumulative PnL (rewards − costs)")
    fig_pnl.update_yaxes(title_text="Cumulative PnL (USD)", secondary_y=False)
    fig_pnl.update_yaxes(title_text="Gas price (gwei)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_pnl, use_container_width=True)

    stats = pd.DataFrame({
        "Scenario": ["A", "B"],
        "Coverage %": [labels["Coverage%"].iloc[selected_one]] * 2,
        "Final rewards USD": [res_A["total_rewards_usd"][selected_one], res_B["total_rewards_usd"][selected_one]],
        "Total costs USD": [res_A["total_costs_usd"][selected_one], res_B["total_costs_usd"][selected_one]],
        "Final PnL USD": [res_A["profit_usd"][selected_one], res_B["profit_usd"][selected_one]],
        "Avg shares/K": [res_A["shares_per_epoch"][selected_one].mean() / sA_k,
                         res_B["shares_per_epoch"][selected_one].mean() / sB_k],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)


with tab7:
    st.caption(
        f"Synthetic {int(mc_n_provers)}-prover scenario · pool walks each epoch by up to ±{mc_max_move:.1f}% "
        f"(scaled by distance from target={int(mc_target)}) · dropout after {int(mc_dropout_window)} "
        f"consecutive epochs with cost ≥ {mc_dropout_ratio:g}× reward · re-entry when hypothetical "
        "max-shares reward beats expected cost."
    )

    mc_tuple = (
        int(mc_n_provers), int(mc_n_epochs), int(mc_target), float(mc_max_move),
        float(mc_base_reward), float(aztec_usd), float(hw_cost), float(hw_cost_stddev_pct),
        float(gas_cost), 1.05, float(mc_dropout_ratio), int(mc_dropout_window), int(seed),
        mc_hw_costs,
    )
    mc_A = run_mechanism_c(cfg_A, mc_tuple)
    mc_B = run_mechanism_c(cfg_B, mc_tuple)

    cols = st.columns(4)
    cols[0].metric("Avg active (A)", f"{mc_A['active_per_epoch'].sum(axis=0).mean():.2f}")
    cols[1].metric("Avg active (B)", f"{mc_B['active_per_epoch'].sum(axis=0).mean():.2f}")
    cols[2].metric("Avg pool A (AZTEC)", f"{mc_A['pool_per_epoch_aztec'].mean():.1f}")
    cols[3].metric("Avg pool B (AZTEC)", f"{mc_B['pool_per_epoch_aztec'].mean():.1f}")

    epochs_axis = np.arange(int(mc_n_epochs))
    mc_gas = mc_A["gas_costs"]  # identical to mc_B (seed-shared, booster-independent)

    def _add_mc_gas(fig: go.Figure) -> None:
        fig.add_scatter(x=epochs_axis, y=mc_gas, mode="lines",
                        name="gas cost ($/epoch)", legendgroup="gas",
                        line=dict(color="rgba(140,140,140,0.55)", width=1),
                        secondary_y=True)

    st.markdown(f"**Active provers per epoch** &nbsp;·&nbsp; "
                f"gas cost: median ${np.median(mc_gas):.2f}, mean ${mc_gas.mean():.2f}, "
                f"p90 ${np.quantile(mc_gas, 0.9):.2f}")
    show_gas_active = st.checkbox("Overlay gas cost (USD/epoch)", value=False, key="mc_gas_active")
    fig_active = make_subplots(specs=[[{"secondary_y": True}]])
    fig_active.add_scatter(x=epochs_axis, y=mc_A["active_per_epoch"].sum(axis=0),
                           mode="lines", name="A active", legendgroup="A", secondary_y=False)
    fig_active.add_scatter(x=epochs_axis, y=mc_B["active_per_epoch"].sum(axis=0),
                           mode="lines", line=dict(dash="dash"), name="B active",
                           legendgroup="B", secondary_y=False)
    fig_active.add_hline(y=int(mc_target), line_dash="dot", line_color="green", annotation_text="target")
    if show_gas_active:
        _add_mc_gas(fig_active)
    fig_active.update_layout(height=300, xaxis_title="Epoch", title="Active provers per epoch")
    fig_active.update_yaxes(title_text="Active provers", secondary_y=False)
    fig_active.update_yaxes(title_text="Gas cost (USD/epoch)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_active, use_container_width=True)

    st.markdown("**Adjusted pool per epoch**")
    show_gas_pool = st.checkbox("Overlay gas cost (USD/epoch)", value=False, key="mc_gas_pool")
    fig_pool = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pool.add_scatter(x=epochs_axis, y=mc_A["pool_per_epoch_aztec"], mode="lines",
                         name="A pool", secondary_y=False)
    fig_pool.add_scatter(x=epochs_axis, y=mc_B["pool_per_epoch_aztec"], mode="lines",
                         line=dict(dash="dash"), name="B pool", secondary_y=False)
    fig_pool.add_hline(y=float(checkpoint_reward), line_dash="dot", line_color="gray",
                       annotation_text="base")
    if show_gas_pool:
        _add_mc_gas(fig_pool)
    fig_pool.update_layout(height=300, xaxis_title="Epoch", title="Adjusted pool per epoch")
    fig_pool.update_yaxes(title_text="Pool (AZTEC)", secondary_y=False)
    fig_pool.update_yaxes(title_text="Gas cost (USD/epoch)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_pool, use_container_width=True)

    st.markdown("**Per-prover cumulative PnL**")
    show_gas_pnl = st.checkbox("Overlay gas cost (USD/epoch)", value=False, key="mc_gas_pnl")
    fig_pnl = make_subplots(specs=[[{"secondary_y": True}]])
    cum_pnl_A = np.cumsum(mc_A["pnl_per_epoch_usd"], axis=1)
    cum_pnl_B = np.cumsum(mc_B["pnl_per_epoch_usd"], axis=1)
    for i in range(int(mc_n_provers)):
        fig_pnl.add_scatter(x=epochs_axis, y=cum_pnl_A[i], mode="lines",
                            name=f"A · P{i} (hw=${mc_A['hw_costs'][i]:.2f})",
                            legendgroup="A", secondary_y=False)
        fig_pnl.add_scatter(x=epochs_axis, y=cum_pnl_B[i], mode="lines", line=dict(dash="dash"),
                            name=f"B · P{i}", legendgroup="B", secondary_y=False)
    fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
    if show_gas_pnl:
        _add_mc_gas(fig_pnl)
    fig_pnl.update_layout(height=500, xaxis_title="Epoch",
                          title="Per-prover cumulative PnL — solid=A, dashed=B")
    fig_pnl.update_yaxes(title_text="Cumulative PnL (USD)", secondary_y=False)
    fig_pnl.update_yaxes(title_text="Gas cost (USD/epoch)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_pnl, use_container_width=True)

    summary = pd.DataFrame({
        "Prover": [f"P{i}" for i in range(int(mc_n_provers))],
        "HW $/epoch": mc_A["hw_costs"],
        "A · profit USD": mc_A["profit_usd"],
        "A · submissions": mc_A["submissions"].sum(axis=1),
        "A · final active": mc_A["active_per_epoch"][:, -1],
        "B · profit USD": mc_B["profit_usd"],
        "B · submissions": mc_B["submissions"].sum(axis=1),
        "B · final active": mc_B["active_per_epoch"][:, -1],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
