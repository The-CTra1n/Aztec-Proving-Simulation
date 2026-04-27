"""Investigate real-data shape — checkpoint vs block, proofs per checkpoint, active ranges."""
import pandas as pd
from proving_sim.real_data import load_submissions

df = load_submissions("data/submissions.parquet")
print(f"rows: {len(df):,}")
print(f"blocks: {df.block.min()}..{df.block.max()}  ({df.block.max() - df.block.min():,} span)")
print(f"checkpoints: {df.checkpoint.min()}..{df.checkpoint.max()}  ({df.checkpoint.max() - df.checkpoint.min():,} span)")

# Proofs per checkpoint
by_cp = df.groupby("checkpoint").size()
print(f"\nproofs per checkpoint: min={by_cp.min()} max={by_cp.max()} mean={by_cp.mean():.1f} median={by_cp.median()}")
print(f"  distinct checkpoints with ≥1 proof: {by_cp.size:,}")
print(f"  checkpoints in range with zero proofs: {(df.checkpoint.max() - df.checkpoint.min() + 1) - by_cp.size:,}")

print(f"\nproofs per checkpoint histogram:")
print(by_cp.value_counts().sort_index().head(10).to_string())
print(f"  (showing top 10 bucket sizes)")

# Block vs checkpoint relationship — monotonic?
df_sorted = df.sort_values("block")
is_monotonic = (df_sorted["checkpoint"].diff().fillna(0) >= -100).all()
print(f"\nblock→checkpoint monotonic-ish (jitter ≤100): {is_monotonic}")
print(f"max backward jump (checkpoint decrease between consecutive blocks): "
      f"{-df_sorted.groupby('block').checkpoint.max().diff().min():.0f}")

# Is activity concentrated in a subrange of checkpoints?
print(f"\ncheckpoints with zero proofs by range (bucket size 1000):")
cp_range = range(df.checkpoint.min(), df.checkpoint.max() + 1, 1000)
for start in cp_range:
    end = start + 999
    n_active = by_cp.loc[start:end].size if (by_cp.index >= start).any() else 0
    total = min(end, df.checkpoint.max()) - start + 1
    if n_active > 0:
        print(f"  cp {start:>6}..{end:>6}  active={n_active:>4}/{total:>4}  proofs={df[(df.checkpoint >= start) & (df.checkpoint <= end)].shape[0]:>5}")

# Check block ranges vs checkpoint ranges
print(f"\nblocks → checkpoints mapping (sample):")
sample = df.groupby(df.block // 10000 * 10000).agg(
    cp_min=("checkpoint", "min"), cp_max=("checkpoint", "max"), count=("checkpoint", "size")
).head(10)
print(sample.to_string())
