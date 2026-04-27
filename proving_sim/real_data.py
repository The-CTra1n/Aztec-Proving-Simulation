"""Load historical submissions from fetched Parquet → (n_provers, n_epochs) bool matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_submissions(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def submissions_matrix(
    df: pd.DataFrame, mode: str = "active_only"
) -> tuple[np.ndarray, list[str], list[int]]:
    """Convert raw event rows into an (n_provers, n_epochs) bool matrix.

    mode:
      active_only — columns are only checkpoints with ≥1 submission (compacted).
                    Matches the forum's coverage definition. Loses gap-decay info
                    across inactive checkpoint runs but that's appropriate when
                    gaps represent testnet downtime, not prover misses.
      dense       — columns span [min_checkpoint, max_checkpoint]; empty
                    checkpoints decay prover scores on real-calendar time.

    Returns:
      matrix — (n_provers, n_epochs) bool, True = prover submitted in that column.
      provers — row index → prover address (lowercase).
      checkpoints — column index → real checkpoint number.
    """
    provers = sorted(df["prover"].unique())
    prover_idx = {p: i for i, p in enumerate(provers)}

    if mode == "active_only":
        active = sorted(df["checkpoint"].unique().tolist())
        col_idx = {c: i for i, c in enumerate(active)}
        out = np.zeros((len(provers), len(active)), dtype=bool)
        for _, row in df.iterrows():
            out[prover_idx[row["prover"]], col_idx[int(row["checkpoint"])]] = True
        return out, provers, active

    if mode == "dense":
        ep_min = int(df["checkpoint"].min())
        ep_max = int(df["checkpoint"].max())
        out = np.zeros((len(provers), ep_max - ep_min + 1), dtype=bool)
        for _, row in df.iterrows():
            out[prover_idx[row["prover"]], int(row["checkpoint"]) - ep_min] = True
        return out, provers, list(range(ep_min, ep_max + 1))

    raise ValueError(f"unknown mode: {mode}")


def gas_series(
    df: pd.DataFrame,
    checkpoints: list[int],
    eth_usd: float,
) -> pd.DataFrame:
    """Per-checkpoint gas summary aligned to the active_only matrix columns.

    Returns a DataFrame indexed 0..len(checkpoints)-1 with columns:
      checkpoint     — the on-chain checkpoint number
      timestamp      — median submission timestamp at that checkpoint (unix seconds)
      gas_price_gwei — median gas price across submissions at that checkpoint
      gas_cost_usd   — median (gas_used × gas_price_wei × eth_usd / 1e18)
      n_submissions  — number of provers that submitted at that checkpoint

    Checkpoints with no submissions get NaN for gas fields.
    """
    if not {"gas_price_wei", "gas_used", "timestamp"}.issubset(df.columns):
        raise ValueError(
            "submissions DataFrame missing gas columns — re-run `python -m proving_sim.fetch` "
            "after pulling the updated parser."
        )
    df = df.copy()
    df["gas_price_gwei"] = df["gas_price_wei"] / 1e9
    df["cost_usd"] = df["gas_used"] * df["gas_price_wei"] * eth_usd / 1e18
    grouped = df.groupby("checkpoint").agg(
        timestamp=("timestamp", "median"),
        gas_price_gwei=("gas_price_gwei", "median"),
        gas_cost_usd=("cost_usd", "median"),
        n_submissions=("prover", "count"),
    )
    out = grouped.reindex(checkpoints).reset_index()
    return out


def coverage_summary(matrix: np.ndarray, provers: list[str]) -> pd.DataFrame:
    n_epochs = matrix.shape[1]
    counts = matrix.sum(axis=1)
    return pd.DataFrame(
        {
            "prover": provers,
            "submissions": counts,
            "coverage_pct": counts / n_epochs * 100,
        }
    )
