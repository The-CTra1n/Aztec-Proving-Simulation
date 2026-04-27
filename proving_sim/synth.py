"""Synthesize per-epoch submission matrices from the reference CSV.

Each prover's aggregate coverage% is preserved; the per-epoch pattern is fabricated.
This means cascade statistics (consecutive-miss distributions) are model-dependent,
not historical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_reference_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_n_epochs(reference: pd.DataFrame) -> int:
    """Recover total epoch count from Proofs / Coverage%.

    Coverage% is rounded to one decimal in the CSV, so very low-coverage rows
    (e.g. 0.0 from a single proof) produce huge quantization error. Use only
    rows with ≥50% coverage for the estimate.
    """
    high = reference[reference["Coverage%"] >= 50.0]
    return int(round((high["Proofs"] / (high["Coverage%"] / 100.0)).mean()))


def synthesize_submissions(
    reference: pd.DataFrame,
    n_epochs: int,
    mode: str = "bernoulli",
    seed: int = 42,
) -> np.ndarray:
    """Return (n_provers, n_epochs) bool matrix — True = prover submitted in that epoch.

    Modes:
      bernoulli — Bernoulli(coverage%) per epoch, independent across provers.
                  Miss clustering emerges from random runs; realistic shape, stochastic.
      even      — Evenly space each prover's target submission count.
                  Deterministic, no miss clusters. Underestimates cascade penalties.
      clustered — Blocky on/off runs with mean-coverage calibrated run lengths.
                  Produces worst-case cascades for stress-testing the booster.
    """
    rng = np.random.default_rng(seed)
    n = len(reference)
    out = np.zeros((n, n_epochs), dtype=bool)
    coverage = reference["Coverage%"].to_numpy() / 100.0
    target_counts = reference["Proofs"].to_numpy().astype(int)

    if mode == "bernoulli":
        for i in range(n):
            out[i] = rng.random(n_epochs) < coverage[i]
    elif mode == "even":
        for i in range(n):
            k = min(target_counts[i], n_epochs)
            if k > 0:
                idx = np.linspace(0, n_epochs - 1, k, dtype=int)
                out[i, idx] = True
    elif mode == "clustered":
        # Alternating on/off runs; average run length 8, coverage preserved via on/off ratio.
        avg_run = 8
        for i in range(n):
            p = coverage[i]
            pos = 0
            on = rng.random() < p
            while pos < n_epochs:
                run = max(1, int(rng.exponential(avg_run)))
                end = min(pos + run, n_epochs)
                if on:
                    out[i, pos:end] = True
                pos = end
                # Flip with probability that keeps long-run coverage = p.
                on = rng.random() < (p if not on else 1 - p)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return out
