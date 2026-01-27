#!/usr/bin/env python3
"""Re-extract funding sources for accounts that failed during the initial run.

This script:
1. Loads the current extraction results
2. Identifies accounts with no transfers found
3. Re-runs extraction for just those accounts with more conservative rate limiting
4. Merges the results back into the main data files

Usage:
    export ETHERSCAN_API_KEY=your_key_here
    python scripts/reextract_failed.py
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Same constants as in extraction script
MIN_TRANSFER_MATIC = 0.1
MIN_TRANSFER_USDC = 1.0
USDC_CONTRACTS = {
    "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359": "USDC",
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "USDC.e",
}


@dataclass
class TransferRecord:
    """A single transfer record."""
    tx_hash: str
    from_addr: str
    to_addr: str
    value_matic: float
    timestamp: str
    block_number: int
    transfer_type: str


@dataclass
class WalletFundFlow:
    """Complete fund flow data for a wallet."""
    wallet: str
    profit_usd: float
    funding_source: Optional[str] = None
    funding_tx_hash: Optional[str] = None
    funding_timestamp: Optional[str] = None
    funding_amount_matic: Optional[float] = None
    funding_source_type: Optional[str] = None
    all_funding_sources: list[TransferRecord] = field(default_factory=list)
    withdrawal_destinations: list[TransferRecord] = field(default_factory=list)
    primary_withdrawal_dest: Optional[str] = None
    total_withdrawn_matic: float = 0.0
    error: Optional[str] = None


class PolygonscanClient:
    """Client for Polygon blockchain via Etherscan V2 API."""

    BASE_URL = "https://api.etherscan.io/v2/api"
    CHAIN_ID = "137"

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.5):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY", "")
        self.rate_limit_delay = rate_limit_delay  # More conservative for re-extraction
        self._last_request_time = 0.0
        self._consecutive_errors = 0

        if not self.api_key:
            print("\n⚠️  WARNING: No API key found!")
            print("   Set ETHERSCAN_API_KEY environment variable\n")

    async def _rate_limit(self):
        """Enforce rate limiting with backoff on errors."""
        now = asyncio.get_event_loop().time()
        delay = self.rate_limit_delay * (1 + self._consecutive_errors * 0.5)
        elapsed = now - self._last_request_time
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def get_transactions(self, address: str, client: httpx.AsyncClient,
                               sort: str = "asc", limit: int = 100) -> list[dict]:
        """Get transactions for an address with better error handling."""
        await self._rate_limit()

        params = {
            "chainid": self.CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(limit),
            "sort": sort,
        }
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            response = await client.get(self.BASE_URL, params=params, timeout=30.0)
            data = response.json()

            if data.get("status") == "1" and isinstance(data.get("result"), list):
                self._consecutive_errors = 0
                return data["result"]
            elif data.get("message") == "No transactions found":
                self._consecutive_errors = 0
                return []
            else:
                self._consecutive_errors += 1
                print(f"    API error for {address[:10]}: {data.get('message')}")
                return []
        except Exception as e:
            self._consecutive_errors += 1
            print(f"    Exception for {address[:10]}: {e}")
            return []

    async def get_internal_transactions(self, address: str, client: httpx.AsyncClient,
                                        sort: str = "asc", limit: int = 100) -> list[dict]:
        """Get internal transactions."""
        await self._rate_limit()

        params = {
            "chainid": self.CHAIN_ID,
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(limit),
            "sort": sort,
        }
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            response = await client.get(self.BASE_URL, params=params, timeout=30.0)
            data = response.json()

            if data.get("status") == "1" and isinstance(data.get("result"), list):
                self._consecutive_errors = 0
                return data["result"]
            return []
        except Exception as e:
            self._consecutive_errors += 1
            return []

    async def get_token_transfers(self, address: str, client: httpx.AsyncClient,
                                  sort: str = "asc", limit: int = 100) -> list[dict]:
        """Get ERC-20 token transfers."""
        await self._rate_limit()

        params = {
            "chainid": self.CHAIN_ID,
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": str(limit),
            "sort": sort,
        }
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            response = await client.get(self.BASE_URL, params=params, timeout=30.0)
            data = response.json()

            if data.get("status") == "1" and isinstance(data.get("result"), list):
                self._consecutive_errors = 0
                return data["result"]
            return []
        except Exception as e:
            self._consecutive_errors += 1
            print(f"    Token TX error for {address[:10]}: {e}")
            return []


async def extract_fund_flow(
    wallet: str,
    profit: float,
    polygonscan: PolygonscanClient,
    client: httpx.AsyncClient,
) -> WalletFundFlow:
    """Extract complete fund flow for a single wallet."""

    flow = WalletFundFlow(wallet=wallet, profit_usd=profit)
    wallet_lower = wallet.lower()

    # Get all transaction types
    txs_asc = await polygonscan.get_transactions(wallet, client, sort="asc", limit=100)
    internal_txs = await polygonscan.get_internal_transactions(wallet, client, sort="asc", limit=100)
    token_txs = await polygonscan.get_token_transfers(wallet, client, sort="asc", limit=100)

    # Combine all transfers
    all_txs = []

    # Process native MATIC transfers
    for tx in txs_asc:
        value_wei = int(tx.get("value", "0"))
        if value_wei < MIN_TRANSFER_MATIC * 1e18:
            continue

        all_txs.append({
            "from": tx.get("from", "").lower(),
            "to": tx.get("to", "").lower(),
            "value": value_wei / 1e18,
            "asset": "MATIC",
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", "0")),
            "block": int(tx.get("blockNumber", "0")),
        })

    # Process internal transactions
    for tx in internal_txs:
        value_wei = int(tx.get("value", "0"))
        if value_wei < MIN_TRANSFER_MATIC * 1e18:
            continue

        all_txs.append({
            "from": tx.get("from", "").lower(),
            "to": tx.get("to", "").lower(),
            "value": value_wei / 1e18,
            "asset": "MATIC",
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", "0")),
            "block": int(tx.get("blockNumber", "0")),
        })

    # Process USDC token transfers
    for tx in token_txs:
        contract = tx.get("contractAddress", "").lower()

        if contract not in USDC_CONTRACTS:
            continue

        decimals = int(tx.get("tokenDecimal", "6"))
        value_raw = int(tx.get("value", "0"))
        value_usdc = value_raw / (10 ** decimals)

        if value_usdc < MIN_TRANSFER_USDC:
            continue

        all_txs.append({
            "from": tx.get("from", "").lower(),
            "to": tx.get("to", "").lower(),
            "value": value_usdc,
            "asset": USDC_CONTRACTS[contract],
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", "0")),
            "block": int(tx.get("blockNumber", "0")),
        })

    # Sort by timestamp
    all_txs.sort(key=lambda x: x["timestamp"])

    # Extract funding sources (inbound)
    for tx in all_txs:
        if tx["to"] == wallet_lower:
            record = TransferRecord(
                tx_hash=tx["hash"],
                from_addr=tx["from"],
                to_addr=tx["to"],
                value_matic=tx["value"],
                timestamp=datetime.fromtimestamp(tx["timestamp"]).isoformat(),
                block_number=tx["block"],
                transfer_type=f"funding:{tx['asset']}",
            )
            flow.all_funding_sources.append(record)

            # Set primary funding source (prefer USDC)
            if not flow.funding_source or (tx["asset"] in ("USDC", "USDC.e") and "USDC" not in (flow.funding_source_type or "")):
                flow.funding_source = tx["from"]
                flow.funding_tx_hash = tx["hash"]
                flow.funding_timestamp = record.timestamp
                flow.funding_amount_matic = tx["value"]
                flow.funding_source_type = f"eoa:{tx['asset']}"

    # Extract withdrawals (outbound)
    for tx in all_txs:
        if tx["from"] == wallet_lower:
            record = TransferRecord(
                tx_hash=tx["hash"],
                from_addr=tx["from"],
                to_addr=tx["to"],
                value_matic=tx["value"],
                timestamp=datetime.fromtimestamp(tx["timestamp"]).isoformat(),
                block_number=tx["block"],
                transfer_type=f"withdrawal:{tx['asset']}",
            )
            flow.withdrawal_destinations.append(record)
            flow.total_withdrawn_matic += tx["value"]

    # Determine primary withdrawal destination
    if flow.withdrawal_destinations:
        dest_totals = defaultdict(float)
        for wd in flow.withdrawal_destinations:
            dest_totals[wd.to_addr] += wd.value_matic
        flow.primary_withdrawal_dest = max(dest_totals, key=dest_totals.get)

    if not flow.funding_source and not flow.withdrawal_destinations:
        flow.error = "no_transfers_found"

    return flow


def load_failed_accounts(data_dir: Path) -> dict[str, float]:
    """Load accounts that had no transfers found."""

    wallet_file = data_dir / "profitable_wallets_full.json"
    with open(wallet_file) as f:
        data = json.load(f)

    failed = {}
    for w in data["wallets"]:
        if not w.get("funding_source") and not w.get("primary_withdrawal_dest"):
            failed[w["wallet"]] = w["profit_usd"]

    return failed


def merge_results(data_dir: Path, new_results: list[WalletFundFlow]):
    """Merge new extraction results into existing files."""

    # 1. Load existing wallet data
    wallet_file = data_dir / "profitable_wallets_full.json"
    with open(wallet_file) as f:
        wallet_data = json.load(f)

    # Create lookup
    wallet_map = {w["wallet"]: i for i, w in enumerate(wallet_data["wallets"])}

    # 2. Update wallets with new data
    updated_count = 0
    recovered_count = 0

    for flow in new_results:
        if flow.wallet in wallet_map:
            idx = wallet_map[flow.wallet]
            old = wallet_data["wallets"][idx]

            # Only update if we found new data
            if flow.funding_source or flow.primary_withdrawal_dest:
                wallet_data["wallets"][idx] = {
                    "wallet": flow.wallet,
                    "profit_usd": flow.profit_usd,
                    "funding_source": flow.funding_source,
                    "funding_source_type": flow.funding_source_type,
                    "funding_timestamp": flow.funding_timestamp,
                    "funding_amount_matic": flow.funding_amount_matic,
                    "primary_withdrawal_dest": flow.primary_withdrawal_dest,
                    "total_withdrawn_matic": flow.total_withdrawn_matic,
                    "funding_count": len(flow.all_funding_sources),
                    "withdrawal_count": len(flow.withdrawal_destinations),
                }
                updated_count += 1
                if not old.get("funding_source") and not old.get("primary_withdrawal_dest"):
                    recovered_count += 1

    # Save updated wallet file
    wallet_data["updated_at"] = datetime.now().isoformat()
    wallet_data["recovery_stats"] = {
        "attempted": len(new_results),
        "recovered": recovered_count,
        "updated": updated_count,
    }

    with open(wallet_file, "w") as f:
        json.dump(wallet_data, f, indent=2)

    print(f"Updated {wallet_file}")
    print(f"  Attempted: {len(new_results)}")
    print(f"  Recovered: {recovered_count}")
    print(f"  Updated: {updated_count}")

    # 3. Update funding sources
    funding_file = data_dir / "funding_sources.json"
    with open(funding_file) as f:
        funding_data = json.load(f)

    # Add new funding sources
    existing_sources = {s["address"]: s for s in funding_data["sources"]}

    for flow in new_results:
        if flow.funding_source:
            addr = flow.funding_source.lower()
            if addr not in existing_sources:
                existing_sources[addr] = {
                    "address": addr,
                    "source_type": "eoa",
                    "labels": [],
                    "funded_wallets": [],
                    "total_profit_funded": 0,
                }

            if flow.wallet not in existing_sources[addr]["funded_wallets"]:
                existing_sources[addr]["funded_wallets"].append(flow.wallet)
                existing_sources[addr]["total_profit_funded"] += flow.profit_usd

    # Sort and save
    funding_data["sources"] = sorted(
        existing_sources.values(),
        key=lambda x: len(x["funded_wallets"]),
        reverse=True,
    )
    funding_data["total_unique_sources"] = len(funding_data["sources"])
    funding_data["updated_at"] = datetime.now().isoformat()

    with open(funding_file, "w") as f:
        json.dump(funding_data, f, indent=2)

    print(f"Updated {funding_file}")

    # 4. Update withdrawal destinations
    withdrawal_file = data_dir / "withdrawal_destinations.json"
    with open(withdrawal_file) as f:
        withdrawal_data = json.load(f)

    existing_dests = {d["address"]: d for d in withdrawal_data["destinations"]}

    for flow in new_results:
        for wd in flow.withdrawal_destinations:
            addr = wd.to_addr.lower()
            if addr not in existing_dests:
                existing_dests[addr] = {
                    "address": addr,
                    "dest_type": "eoa",
                    "labels": [],
                    "received_from_traders": [],
                    "total_profit_source": 0,
                    "also_funded_traders": [],
                }

            if flow.wallet not in existing_dests[addr]["received_from_traders"]:
                existing_dests[addr]["received_from_traders"].append(flow.wallet)
                existing_dests[addr]["total_profit_source"] += flow.profit_usd

    # Sort and save
    withdrawal_data["destinations"] = sorted(
        existing_dests.values(),
        key=lambda x: len(x["received_from_traders"]),
        reverse=True,
    )
    withdrawal_data["total_unique_destinations"] = len(withdrawal_data["destinations"])
    withdrawal_data["updated_at"] = datetime.now().isoformat()

    with open(withdrawal_file, "w") as f:
        json.dump(withdrawal_data, f, indent=2)

    print(f"Updated {withdrawal_file}")


async def main():
    data_dir = PROJECT_ROOT / "data"

    print("=" * 60)
    print("RE-EXTRACTION OF FAILED ACCOUNTS")
    print("=" * 60)

    # Load failed accounts
    failed = load_failed_accounts(data_dir)
    print(f"\nFound {len(failed)} accounts with no transfers")

    if not failed:
        print("No failed accounts to re-extract!")
        return

    # Sort by profit (highest first)
    sorted_failed = sorted(failed.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 failed accounts by profit:")
    for wallet, profit in sorted_failed[:10]:
        print(f"  {wallet[:20]}... ${profit:,.0f}")

    print(f"\nRe-extracting with conservative rate limiting...")
    print("-" * 60)

    polygonscan = PolygonscanClient(rate_limit_delay=0.5)  # More conservative
    results = []

    async with httpx.AsyncClient() as client:
        for i, (wallet, profit) in enumerate(sorted_failed, 1):
            if i % 10 == 0 or i == 1:
                pct = i / len(sorted_failed) * 100
                print(f"  [{i:5}/{len(sorted_failed)}] {pct:5.1f}% - {wallet[:16]}...", end="", flush=True)

            flow = await extract_fund_flow(wallet, profit, polygonscan, client)
            results.append(flow)

            if i % 10 == 0 or i == 1:
                if flow.funding_source:
                    print(f" ✓ Found: {flow.funding_source[:12]}...")
                elif flow.primary_withdrawal_dest:
                    print(f" ✓ Found WD: {flow.primary_withdrawal_dest[:12]}...")
                else:
                    print(" ✗ Still no data")

    # Stats
    recovered = sum(1 for r in results if r.funding_source or r.withdrawal_destinations)
    still_failed = len(results) - recovered

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total attempted: {len(results)}")
    print(f"Recovered: {recovered} ({100*recovered/len(results):.1f}%)")
    print(f"Still failed: {still_failed}")

    if recovered > 0:
        print(f"\nMerging results into data files...")
        merge_results(data_dir, results)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
