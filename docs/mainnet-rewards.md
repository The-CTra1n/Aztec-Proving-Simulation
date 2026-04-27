# Aztec mainnet — prover reward issuance per checkpoint

**Source:** read directly from the live Rollup contract on Ethereum mainnet
`0xae2001f7e21d5ecabf6234e9fdd1e76f50f74962` via `getRewardConfig()` /
`getCheckpointReward()` (verified against `aztec-packages` `next` branch).

## Per-checkpoint issuance

| Field | On-chain value | Deploy-script default |
|---|---|---|
| `checkpointReward` | **500 × 10¹⁸** (500 AZTEC) | 400 × 10¹⁸ |
| `sequencerBps` | **7000** (70%) | 7000 |
| `rewardDistributor` | `0x3d6a1b00c830c5f278fc5dfb3f6ff0b74db6dfe0` | — |
| `boosterCore` | `0x1cbb707bd7b4fd2bced6d96d84372fb428e93d80` | — |
| `feeAsset` | `0xa27ec0006e59f245217ff08cd52a7e8b169e62d2` (AZTEC) | — |

Per checkpoint:

- **350 AZTEC → sequencer** (70 % of 500)
- **150 AZTEC → prover pool** (30 % of 500), split among provers by
  `RewardBooster` shares for that epoch

The live `checkpointReward` (500) is higher than the 400 default baked into
`l1-contracts/script/deploy/RollupConfiguration.sol:74`, confirming
governance/deployment override.

## What this implies for the simulation

This repo only models **prover** payouts, so the relevant figure is the prover
pool — not the gross checkpoint issuance:

- **Per checkpoint:** 150 AZTEC to provers
- **Per epoch (32 checkpoints):** **4,800 AZTEC** to provers

The sidebar `epoch reward` default is set to 4,800 AZTEC. The sequencer split
is out of scope and is not modeled here.

## How prover rewards work (recap)

- Provers earn from two sources accrued at epoch-proof submission time:
  1. **Proving fee** per checkpoint:
     `min(manaUsed * provingCostPerMana, totalFee - congestionBurn)`
     — entirely to provers
     (`l1-contracts/src/core/libraries/rollup/RewardLib.sol:235`).
  2. **Checkpoint reward** from `RewardDistributor` (only if the rollup is
     canonical), split with sequencer via `sequencerBps`
     (`RewardLib.sol:187-214`).
- Distribution among provers for the same epoch is share-weighted by
  `RewardBooster` (quadratic activity score, decays over time —
  `RewardBooster.sol:117-130`).
- Permissionless — no stake required to prove. Multiple provers can submit
  proofs for the same epoch; only those proving the same longest prefix earn.
- Rewards accrue on proof verification (`EpochProofLib.sol:135` →
  `handleRewardsAndFees`) and are claimed via
  `claimProverRewards(address, Epoch[])` (`RollupCore.sol:425`) after the
  unlock timestamp. Paid in the fee asset (AZTEC ERC-20).

## Funding model

`RewardDistributor` is a **pre-funded bucket**, not a per-block mint. The
rollup `claim()`s from it on each epoch proof. Replenishment is governed
externally; there is no `BLOCK_REWARD` constant.

## Reproducing the read

```bash
# checkpointReward
cast call 0xae2001f7e21d5ecabf6234e9fdd1e76f50f74962 \
  "getCheckpointReward()(uint256)" --rpc-url <eth-rpc>

# Full RewardConfig: (rewardDistributor, sequencerBps, booster, checkpointReward)
cast call 0xae2001f7e21d5ecabf6234e9fdd1e76f50f74962 \
  "getRewardConfig()((address,uint32,address,uint96))" --rpc-url <eth-rpc>
```

Selectors used: `getCheckpointReward()` = `0x86a0d763`,
`getRewardConfig()` = `0xec147806`.
