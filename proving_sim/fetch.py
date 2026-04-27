"""Fetch Aztec L2ProofVerified events from Ethereum mainnet via Etherscan.

Event: L2ProofVerified(uint256 indexed checkpointNumber, address indexed proverId)
  topic0 = 0x034dd13d657aeb14f8dec7291c4a8ddb3b20d40cf2412714e72f97f19c735609
  topic1 = checkpointNumber (big-endian uint256)
  topic2 = proverId (left-padded to 32 bytes)

Usage:
    export ETHERSCAN_API_KEY=...
    python -m proving_sim.fetch  # writes data/submissions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROLLUP_ADDRESS = "0xAe2001f7e21d5EcABf6234E9FDd1E76F50F74962"
TOPIC_L2_PROOF_VERIFIED = "0x034dd13d657aeb14f8dec7291c4a8ddb3b20d40cf2412714e72f97f19c735609"
ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
ETHERSCAN_CHAIN_ID = 1  # Ethereum mainnet
PAGE_SIZE = 1000
RATE_LIMIT_SLEEP = 0.25  # 4 req/s — stays under free-tier 5 req/s limit


def _get(params: dict, api_key: str) -> dict:
    params = {**params, "chainid": ETHERSCAN_CHAIN_ID, "apikey": api_key}
    r = requests.get(ETHERSCAN_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") == "0" and j.get("message") != "No records found":
        raise RuntimeError(f"Etherscan error: {j.get('message')} — {j.get('result')}")
    return j


def get_contract_creation_block(address: str, api_key: str) -> int:
    """Return the block number where the contract was created."""
    j = _get(
        {"module": "contract", "action": "getcontractcreation", "contractaddresses": address},
        api_key,
    )
    creation_tx = j["result"][0]["txHash"]
    time.sleep(RATE_LIMIT_SLEEP)
    tx = _get({"module": "proxy", "action": "eth_getTransactionByHash", "txhash": creation_tx}, api_key)
    return int(tx["result"]["blockNumber"], 16)


def get_latest_block(api_key: str) -> int:
    j = _get({"module": "proxy", "action": "eth_blockNumber"}, api_key)
    return int(j["result"], 16)


def _fetch_page(from_block: int, to_block: int, page: int, api_key: str) -> list[dict]:
    j = _get(
        {
            "module": "logs",
            "action": "getLogs",
            "address": ROLLUP_ADDRESS,
            "topic0": TOPIC_L2_PROOF_VERIFIED,
            "fromBlock": from_block,
            "toBlock": to_block,
            "page": page,
            "offset": PAGE_SIZE,
        },
        api_key,
    )
    result = j.get("result") or []
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected result: {result}")
    return result


MAX_PAGES = 10_000 // PAGE_SIZE  # Etherscan hard cap: page × offset ≤ 10 000


def fetch_logs_range(from_block: int, to_block: int, api_key: str) -> list[dict]:
    """Fetch all L2ProofVerified logs in [from_block, to_block].

    Paginates with offset=PAGE_SIZE until a short page ends the range. If all MAX_PAGES
    come back full, the range exceeds Etherscan's 10k-log cap — discard the partial
    fetch and bisect into two sub-ranges. Wasted calls on a bisection: up to MAX_PAGES.
    """
    logs: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        result = _fetch_page(from_block, to_block, page, api_key)
        time.sleep(RATE_LIMIT_SLEEP)
        logs.extend(result)
        if len(result) < PAGE_SIZE:
            return logs
    # All MAX_PAGES pages came back full → range has >10k logs. Discard and bisect.
    if from_block >= to_block:
        return logs  # single block with >10k logs — shouldn't happen in practice
    mid = (from_block + to_block) // 2
    return fetch_logs_range(from_block, mid, api_key) + fetch_logs_range(mid + 1, to_block, api_key)


def _hex_int(s: str) -> int:
    """Etherscan sometimes returns '0x' for zero-valued hex fields."""
    return int(s, 16) if len(s) > 2 else 0


def parse_logs(raw_logs: list[dict]) -> pd.DataFrame:
    rows = []
    for log in raw_logs:
        topics = log["topics"]
        checkpoint = _hex_int(topics[1])
        prover = "0x" + topics[2][-40:]
        rows.append(
            {
                "block": _hex_int(log["blockNumber"]),
                "tx_hash": log["transactionHash"],
                "log_index": _hex_int(log["logIndex"]),
                "checkpoint": checkpoint,
                "prover": prover.lower(),
                "timestamp": _hex_int(log.get("timeStamp", "0x0")),
                "gas_price_wei": _hex_int(log.get("gasPrice", "0x0")),
                "gas_used": _hex_int(log.get("gasUsed", "0x0")),
            }
        )
    return pd.DataFrame(rows)




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/submissions.parquet"))
    parser.add_argument("--from-block", type=int, default=None)
    parser.add_argument("--to-block", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=100_000, help="blocks per Etherscan query")
    args = parser.parse_args()

    api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        raise SystemExit("Set ETHERSCAN_API_KEY in the environment.")

    # Fetch raw first, cache to disk, then parse — so a parse bug doesn't force a re-fetch.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_cache = args.out.with_suffix(".raw.json")
    if raw_cache.exists() and not args.from_block and not args.to_block:
        print(f"Using cached raw logs at {raw_cache}")
        all_logs = json.loads(raw_cache.read_text())
    else:
        from_block = args.from_block or get_contract_creation_block(ROLLUP_ADDRESS, api_key)
        to_block = args.to_block or get_latest_block(api_key)
        print(f"Fetch range: {from_block}..{to_block}")
        all_logs = []
        cur = from_block
        while cur <= to_block:
            end = min(cur + args.chunk - 1, to_block)
            raw = fetch_logs_range(cur, end, api_key)
            all_logs.extend(raw)
            print(f"  blocks {cur:>10}..{end:>10}  +{len(raw):>5} logs (total {len(all_logs)})")
            cur = end + 1
        raw_cache.write_text(json.dumps(all_logs))
        print(f"Cached raw response to {raw_cache}")
    df = parse_logs(all_logs)
    print(f"\nWrote {len(df)} rows to {args.out}")
    if len(df):
        print(f"  checkpoints: {df['checkpoint'].min()}..{df['checkpoint'].max()}")
        print(f"  unique provers: {df['prover'].nunique()}")


if __name__ == "__main__":
    main()
