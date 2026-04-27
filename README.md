# Aztec prover reward simulation

Interactive simulator for Aztec L2 prover-reward mechanisms, driven by real
on-chain submission data fetched from the Rollup contract
(`0xAe2001f7e21d5EcABf6234E9FDd1E76F50F74962`). Compares the currently deployed
booster ("Mechanism A"), a proposed governance update ("Mechanism B"), and a
supply-target dynamic-pool variant ("Mechanism C") under identical economic
assumptions.

```
streamlit run app.py
```

## Quick start

```bash
# 1. Create a venv and install
python -m venv .venv
. .venv/Scripts/activate     # Windows; use bin/activate on Unix
pip install -e ".[dev]"

# 2. Run the app (uses bundled data/submissions.parquet)
streamlit run app.py

# 3. Optional — refresh on-chain data
export ETHERSCAN_API_KEY=...
python -m proving_sim.fetch        # rewrites data/submissions.parquet

# 4. Tests
pytest
```

`ETHERSCAN_API_KEY` is read from the environment by `proving_sim.fetch`. No key
is committed to the repo; the bundled `data/submissions.parquet` and
`data/submissions.raw.json` were produced from a previous fetch and contain only
public on-chain event logs.

## Layout

```
app.py                       Streamlit UI — sidebar controls + 7 tabs
proving_sim/
  booster.py                 RewardBooster.sol port (line-for-line)
  simulation.py              Epoch-loop simulator over real submissions
  mechanism_c.py             Supply-target dynamic-pool variant
  real_data.py               Parquet → (n_provers × n_epochs) bool matrix + gas series
  fetch.py                   Etherscan → submissions.parquet (event logs + gas)
  synth.py                   Synthetic prover generators (used by Mech C)
data/
  submissions.parquet        Per-submission rows (block, tx, checkpoint, prover, gas)
  submissions.raw.json       Raw Etherscan response (cache; safe to delete)
reference-sim/               Solidity reference + previously published comparisons
tests/                       pytest suite (booster math, Mech C invariants)
```

## Time, space, and units

- **Checkpoint** — on-chain unit: `Rollup.L2ProofVerified(checkpointNumber, prover)`.
  72 seconds.
- **Epoch** — what the simulation iterates over: 32 checkpoints × 72 s = **2304 s
  (≈38.4 min)**, ~1,141 epochs/month. The `active_only` matrix used by the app
  drops empty checkpoint runs (testnet downtime) and treats columns as discrete
  prover-decision points.
- **Active provers (real-data tabs)** — currently 76 unique addresses across
  987 active checkpoints in the bundled dataset.
- **Currency** — pools and shares are denominated in **AZTEC**; profit/PnL is
  shown in **USD** via the sidebar `AZTEC → USD` rate.

## Mechanisms

Each mechanism reuses the same booster math; what differs is the parameter set
and (for C) how the per-epoch reward pool is sized.

### Booster math (shared)

Each prover holds an integer `score` (capped at `maxScore`). On submission,
`score += increment` (clamped). Inactivity decays the score by 100,000 per
epoch (this constant is hard-coded in the deployed contract). The per-prover
share paid out at a checkpoint is

```
shares = clamp( k − a · (maxScore − score)² / 1e10, minimum, k )
```

— a quadratic that floors at `minimum` and ceilings at `k` when at maxScore.

### Mechanism A — current `RewardBooster` (deployed)

Default params: `increment=125,000`, `maxScore=15,000,000`, `a=1,000`,
`minimum=100,000`, `k=1,000,000`. With these, ~120 consecutive submissions are
needed to reach the cap. **Expected behaviour:** most provers sit near the
floor; rewards scale roughly linearly with coverage. Top-coverage provers pull
slowly ahead.

### Mechanism B — proposed `RewardBooster` (governance proposal)

Same math, sharper params: `increment=101,400`, `maxScore=367,500` (40× lower
than A → cap reached in ~4 submissions), `a=250,000` (250× higher), `minimum=10,000`
(10× lower). **Expected behaviour:** rewards consistency aggressively; missing
even a few epochs collapses share toward `minimum`. High-coverage provers
dominate; low-coverage provers earn ~order of magnitude less than under A.

### Mechanism C — supply-target dynamic pool

Wraps Mechanism A (or B) with a memoryless multiplicative pool walk:

```
pool[e] = pool[e-1] × (1 + δ)
δ       = max_move_pct/100 × clip((target − n_submitters) / target, ±1)
```

When submitters fall below `target`, the pool ratchets up (capped at +max_move%
per step). Above `2×target` it ratchets down (−max_move%). Sustained undersupply
grows the pool indefinitely — that's the design intent.

Because the real dataset has 76 active provers (always far above any plausible
target), Mech C uses a **synthetic** 10-prover scenario. Provers drop out after
`dropout_window` consecutive epochs where `cost ≥ ratio × reward`, and re-enter
when their hypothetical reward at maximum booster shares would beat their
expected per-epoch cost.

**Expected behaviour:**
- If `target` is set above the synthetic prover count, the pool grows
  unboundedly — rewards rise until enough provers find re-entry profitable.
- Under typical params (target=5, n=10) the system equilibrates near `target`,
  with periodic dropouts/re-entries depending on dropout sensitivity.

## Economic inputs

The sidebar opens with a **🌐 Global parameters** section (separated from the
per-mechanism params by a divider) — these apply to every mechanism and every
tab:

- **`AZTEC → USD`** — token price used for all USD profitability conversions.
- **`ETH → USD`** — used to price the historical gas series for the real-data
  gas overlay (does not affect Mechanism C's synthetic gas draw).

The remaining economic inputs are in the sidebar under "Economic":

- **`prover reward pool`** — AZTEC paid to **provers** per epoch (one proof
  submission per epoch). Default **4,800 AZTEC** = 150 AZTEC/checkpoint × 32
  checkpoints/epoch. This is the prover share only; the sequencer's 70 % cut is
  out of scope. On-chain values verified from the live Rollup contract — see
  [`docs/mainnet-rewards.md`](docs/mainnet-rewards.md).
- **`sequencer split (bps)`** — fraction skimmed before the prover pool.
- **`Hardware cost per epoch`** — baseline per-prover infra cost. Default
  **$1.14/epoch**, derived from the conservative 1-yr-reserved AWS
  `c6i.16xlarge` price ($1,300/mo, 64 vCPU / 128 GiB ≈ the 32-core / 64 vCPU /
  128 GB / 10 GB SSD reference spec) at 2304 s/epoch (~1,141 epochs/mo).
  Pricing curve for that spec (Apr 2026, [Vantage](https://instances.vantage.sh/aws/ec2/c6i.16xlarge)):
  - On-demand: ~$1,950–2,100/mo → ~$1.71–$1.84/epoch
  - 1-yr reserved (no upfront): ~$1,200–1,300/mo → ~$1.05–$1.14/epoch ← default
  - 3-yr reserved (all upfront): ~$750–850/mo → ~$0.66–$0.74/epoch
  - Spot: ~$500–900/mo → ~$0.44–$0.79/epoch
- **`Cost heterogeneity (%)`** — stddev around the baseline so different provers
  represent different infra choices.
- **`Proof submission gas cost`** — L1 gas charged only on submission epochs.
  Median of a lognormal whose σ is fitted to observed on-chain data; default
  $2 (mean $3.4, p90 $6.4) at ETH=$3,500.

## Gas overlays

Two independent overlays are available:

- **Real-data tabs** (Cumulative-over-time, PnL, Drill-down): a "Overlay
  historical gas price" sidebar checkbox draws the **median gas price (gwei)**
  per active checkpoint from the bundled parquet on a secondary y-axis.
- **Mechanism C tab**: each plot has its own "Overlay gas cost (USD/epoch)"
  checkbox, drawing the **per-epoch realised gas cost** (in USD) that the Mech C
  simulation actually applied that epoch (lognormal draw, seed-deterministic).

The two are different signals — historical gwei vs. synthetic per-epoch USD
cost — and they live on different axes.

## Tabs

The app opens on **Mechanism C** by default.

1. **Mechanism C** *(default)* — synthetic supply-target run: active provers per
   epoch, pool trajectory, per-prover cumulative PnL. Each plot has an
   independent gas-cost overlay toggle.
2. **Per-prover rewards** — bar chart of total AZTEC paid under A vs. B per
   address.
3. **Cumulative over time** — top-10 covered provers' running rewards. Optional
   gas-price overlay.
4. **Coverage → share curve** — simulated steady-state mean share at Bernoulli(p)
   submission for 2,000 epochs. Shows the curvature difference between A and B.
5. **Profitability** — per-prover rewards/costs/profit table; aggregate summary.
6. **PnL over time** — cumulative USD PnL per selected prover. Optional gas-price
   overlay.
7. **Prover drill-down** — single-prover deep-dive: shares, cumulative rewards
   vs. costs, cumulative PnL. Optional gas-price overlay on the PnL chart.

## Refreshing the dataset

`python -m proving_sim.fetch` paginates `eth_getLogs` over the rollup contract,
caches the raw response to `data/submissions.raw.json`, then parses gas data and
writes `data/submissions.parquet`. If the raw cache exists the refetch is
skipped — delete the file to force a re-pull.

## Tests

```bash
pytest -q
```

Covers booster decay/score math (`tests/test_booster.py`) and Mechanism C
invariants like pool walk bounds, dropout/re-entry, and gas-cost application
(`tests/test_mechanism_c.py`). Real-data smoke checks live in
`tests/check_real.py`, `tests/check_pnl.py`, `tests/smoke.py`.
