#!/usr/bin/env python3
"""Extract funding sources AND withdrawal destinations for profitable Polymarket traders.

This script builds a complete fund flow graph:
1. Collects all traders with profit >= threshold from Polymarket leaderboard
2. For each wallet, queries Polygonscan API for:
   - First inbound MATIC transfer (funding source)
   - All significant outbound MATIC transfers (withdrawal destinations)
3. Builds a graph connecting: funding_source -> trader -> withdrawal_dest
4. Detects chains: X funds A, A withdraws to Y, Y funds B = X->A->Y->B cluster

This enables footprinting:
- When a new account is funded, check if funder is in the graph
- If funder was a withdrawal destination of a known profitable trader, flag it

Output:
- fund_flow_graph.json: Complete graph of funding relationships
- funding_sources.json: Master list of funding sources
- withdrawal_destinations.json: Master list of withdrawal destinations
- profitable_wallets_full.json: Each wallet with funding + withdrawal data
- cluster_report.txt: Human-readable cluster analysis

Rate Limits:
- Polygonscan free tier: 5 calls/sec, 100k calls/day
- This script uses 0.25 sec delay between calls (4 calls/sec max)
- For 500 wallets: ~6 minutes (3 API calls per wallet: txlist, internal, tokentx)
- For 2000 wallets: ~25 minutes

Prerequisites:
    - ETHERSCAN_API_KEY environment variable (get free at https://etherscan.io/myapikey)
    - Uses Etherscan V2 API with chainid=137 for Polygon (Polygonscan V1 is deprecated)

Usage:
    # Set API key first
    export ETHERSCAN_API_KEY=your_key_here

    python extract_funding_sources.py --min-profit 20000
    python extract_funding_sources.py --min-profit 20000 --resume
    python extract_funding_sources.py --input-file data/profitable_accounts.txt

    # Check if a new wallet is connected to known profitable traders
    python extract_funding_sources.py --lookup 0x1234...
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import httpx

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.discovery.service import LeaderboardClient, LEADERBOARD_CATEGORIES
from src.discovery.analyzer import LeaderboardEntry


# Minimum values to consider as significant transfer (filters dust)
MIN_TRANSFER_MATIC = 0.1  # 0.1 MATIC minimum
MIN_TRANSFER_USDC = 1.0   # $1 USDC minimum

# USDC contract addresses on Polygon
USDC_CONTRACTS = {
    "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359": "USDC",      # Native USDC
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "USDC.e",    # Bridged USDC.e
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
    transfer_type: str  # "funding" or "withdrawal"


@dataclass
class WalletFundFlow:
    """Complete fund flow data for a wallet."""
    wallet: str
    profit_usd: float

    # Funding sources (inbound transfers)
    funding_source: Optional[str] = None  # Primary (first) funding source
    funding_tx_hash: Optional[str] = None
    funding_timestamp: Optional[str] = None
    funding_amount_matic: Optional[float] = None
    funding_source_type: Optional[str] = None
    all_funding_sources: list[TransferRecord] = field(default_factory=list)

    # Withdrawal destinations (outbound transfers)
    withdrawal_destinations: list[TransferRecord] = field(default_factory=list)
    primary_withdrawal_dest: Optional[str] = None  # Most common or largest
    total_withdrawn_matic: float = 0.0

    error: Optional[str] = None


@dataclass
class FundFlowNode:
    """A node in the fund flow graph."""
    address: str
    node_type: str  # "trader", "funder", "withdrawal_dest", "bridge"

    # Connections
    funded_wallets: list[str] = field(default_factory=list)  # Wallets this address funded
    received_from: list[str] = field(default_factory=list)   # Wallets that sent to this address

    # If this is a trader
    profit_usd: Optional[float] = None

    # Metadata
    labels: list[str] = field(default_factory=list)
    first_seen: Optional[str] = None
    total_volume_matic: float = 0.0


@dataclass
class FundFlowGraph:
    """Complete fund flow graph for cluster detection."""
    nodes: dict[str, FundFlowNode] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)  # [{from, to, value, tx_hash, timestamp}]

    # Quick lookups
    trader_wallets: set[str] = field(default_factory=set)
    funding_sources: set[str] = field(default_factory=set)
    withdrawal_dests: set[str] = field(default_factory=set)

    # Cluster data
    clusters: list[dict] = field(default_factory=list)


# Known exchange hot wallets on Polygon
KNOWN_EXCHANGES = {
    # Binance
    "0xf977814e90da44bfa03b6295a0616a897441acec": "Binance",
    "0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245": "Binance Hot 2",
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance Hot 3",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "Coinbase Hot",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase Commerce",
    # Kraken
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Kraken",
    "0xda9dfa130df4de4673b89022ee50ff26f6ea73cf": "Kraken Hot",
    # OKX
    "0x5041ed759dd4afc3a72b8192c143f72f4724081a": "OKX",
    "0x98ec059dc3adfbdd63429454aeb0c990fba4a128": "OKX Hot",
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX Hot 2",
    # Crypto.com
    "0xcffad3200574698b78f32232aa9d63eabd290703": "Crypto.com",
    "0x6262998ced04146fa42253a5c0af90ca02dfd2a3": "Crypto.com Hot",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit",
    "0x1db92e2eebc8e0c075a02bea49a2935bcd2dfcf4": "Bybit Hot",
    # KuCoin
    "0xd6216fc19db775df9774a6e33526131da7d19a2c": "KuCoin",
    "0xeb2629a2734e272bcc07bda959863f316f4bd4cf": "KuCoin Hot",
    # Gate.io
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Gate.io",
    # HTX (Huobi)
    "0x18709e89bd403f470088abdacebe86cc60dda12e": "HTX",
    # MEXC
    "0x75e89d5979e4f6fba9f97c104c2f0afb3f1dcb88": "MEXC",
    # Polygon Bridge (from Ethereum)
    "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": "Polygon Bridge",
    "0x5773ff0a3b8b5e92f9e4f0e73f72a96f2ce8b4a0": "Polygon POS Bridge",
    "0x401f6c983ea34274ec46f84d70b31c151321188b": "Polygon Plasma Bridge",
    # LayerSwap
    "0x8898504c4a3ce3fef3e0000008c8e91ef63b5b97": "LayerSwap",
    # Multichain Bridge
    "0x2ef4a574b72e1f555185afa8a09c6d1a8ac4025c": "Multichain",
    # Hop Protocol
    "0x86ca30bef97fb651b8d866d45503684b90cb3312": "Hop Protocol",
    "0x553bc791d746767166fa3888432038193ceed5e2": "Hop MATIC",
    # Across Protocol
    "0x9295ee1d8c5b022be115a2ad3c30c72e34e7f096": "Across",
    # Stargate
    "0x45a01e4e04f14f7a4a6702c74187c5f6222033cd": "Stargate",
    # Synapse
    "0x1c6ae197ff4bf7ba96c66c5fd64cb22450af9cc8": "Synapse",
    # Celer cBridge
    "0x5427fefa711eff984124bfbb1ab6fbf5e3da1820": "Celer cBridge",
    # Wormhole
    "0x5a58505a96d1dbf8df91cb21b54419fc36e93fde": "Wormhole",
}


class PolygonscanClient:
    """Client for Polygon blockchain via Etherscan V2 API.

    NOTE: Polygonscan V1 API is deprecated. This uses Etherscan's unified V2 API
    with chainid=137 for Polygon network.

    Requires API key - get one free at https://etherscan.io/myapikey

    Rate limiting: Uses adaptive rate limiting with backoff on errors.
    The free tier allows 5 calls/sec, but we use 0.4s delay (2.5 calls/sec)
    to leave headroom and reduce rate limit errors.
    """

    # Etherscan V2 API (unified across all chains)
    BASE_URL = "https://api.etherscan.io/v2/api"
    CHAIN_ID = "137"  # Polygon mainnet

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.4):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY") or os.getenv("POLYGONSCAN_API_KEY", "")
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        self._consecutive_errors = 0
        self._max_retries = 3

        if not self.api_key:
            print("\n⚠️  WARNING: No API key found!")
            print("   Set ETHERSCAN_API_KEY environment variable")
            print("   Get a free key at: https://etherscan.io/myapikey\n")

    async def _rate_limit(self):
        """Enforce rate limiting with adaptive backoff on errors."""
        now = asyncio.get_event_loop().time()
        # Increase delay based on consecutive errors (exponential backoff)
        delay = self.rate_limit_delay * (1 + self._consecutive_errors * 0.5)
        elapsed = now - self._last_request_time
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        params: dict,
        action_name: str,
        address: str,
    ) -> list[dict]:
        """Make an API request with retry logic."""
        for attempt in range(self._max_retries):
            await self._rate_limit()

            try:
                response = await client.get(self.BASE_URL, params=params, timeout=30.0)
                data = response.json()

                # Success case
                if data.get("status") == "1" and isinstance(data.get("result"), list):
                    self._consecutive_errors = 0
                    return data["result"]

                # "No transactions found" is a valid response, not an error
                if data.get("message") == "No transactions found":
                    self._consecutive_errors = 0
                    return []

                # Rate limit or other API error - retry with backoff
                if "rate limit" in str(data.get("message", "")).lower():
                    self._consecutive_errors += 1
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue

                # Other API errors
                self._consecutive_errors += 1
                return []

            except Exception as e:
                self._consecutive_errors += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry
                    continue
                print(f"  Error in {action_name} for {address[:10]}: {e}")
                return []

        return []

    async def get_transactions(
        self,
        address: str,
        client: httpx.AsyncClient,
        sort: str = "asc",
        limit: int = 100,
    ) -> list[dict]:
        """Get transactions for an address."""
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

        return await self._make_request(client, params, "txlist", address)

    async def get_internal_transactions(
        self,
        address: str,
        client: httpx.AsyncClient,
        sort: str = "asc",
        limit: int = 100,
    ) -> list[dict]:
        """Get internal transactions (for contract interactions)."""
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

        return await self._make_request(client, params, "txlistinternal", address)

    async def get_token_transfers(
        self,
        address: str,
        client: httpx.AsyncClient,
        sort: str = "asc",
        limit: int = 100,
    ) -> list[dict]:
        """Get ERC-20 token transfers for an address (includes USDC)."""
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

        return await self._make_request(client, params, "tokentx", address)

    def identify_address_type(self, address: str) -> tuple[str, list[str]]:
        """Identify if an address is a known exchange, bridge, or EOA.

        Returns: (type, labels)
        """
        addr_lower = address.lower()

        # Check known exchanges/bridges
        if addr_lower in KNOWN_EXCHANGES:
            name = KNOWN_EXCHANGES[addr_lower]
            if "Bridge" in name or "Hop" in name or "Stargate" in name or "Synapse" in name:
                return "bridge", [name]
            return "exchange", [name]

        return "eoa", []


async def collect_profitable_wallets(
    min_profit: float,
    max_wallets: int = 10000,
) -> dict[str, float]:
    """Collect wallets with profit >= min_profit."""

    wallets: dict[str, float] = {}

    print(f"\nCollecting wallets with >= ${min_profit:,.0f} profit...")

    async with LeaderboardClient() as client:
        for category in LEADERBOARD_CATEGORIES:
            print(f"  [{category:12}] ", end="", flush=True)

            offset = 0
            category_count = 0

            while len(wallets) < max_wallets:
                try:
                    raw_entries = await client.get_leaderboard(
                        category=category,
                        time_period="ALL",
                        order_by="PNL",
                        limit=50,
                        offset=offset,
                    )

                    if not raw_entries:
                        break

                    should_stop = False
                    for raw in raw_entries:
                        pnl = float(raw.get("pnl", 0))

                        if pnl < min_profit:
                            should_stop = True
                            break

                        wallet = raw.get("proxyWallet", "").lower()
                        if wallet and wallet not in wallets:
                            wallets[wallet] = pnl
                            category_count += 1

                    if should_stop:
                        break

                    offset += 50
                    await asyncio.sleep(0.03)

                except Exception as e:
                    print(f"ERROR: {e}")
                    break

            print(f"{category_count:4} new wallets")

    print(f"\nTotal unique wallets: {len(wallets):,}")
    return wallets


async def extract_fund_flow(
    wallet: str,
    profit: float,
    polygonscan: PolygonscanClient,
    client: httpx.AsyncClient,
) -> WalletFundFlow:
    """Extract complete fund flow for a single wallet (funding + withdrawals).

    Tracks both:
    - Native MATIC transfers
    - USDC token transfers (primary currency on Polymarket)
    """

    flow = WalletFundFlow(wallet=wallet, profit_usd=profit)
    wallet_lower = wallet.lower()

    # Get normal transactions (MATIC transfers)
    txs_asc = await polygonscan.get_transactions(wallet, client, sort="asc", limit=50)

    # Get internal transactions (contract-mediated MATIC)
    internal_txs = await polygonscan.get_internal_transactions(wallet, client, sort="asc", limit=50)

    # Get ERC-20 token transfers (USDC - primary Polymarket currency)
    token_txs = await polygonscan.get_token_transfers(wallet, client, sort="asc", limit=100)

    # Combine all transfers into unified format
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
            "value_usd": None,  # MATIC value varies
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
            "value_usd": None,
            "asset": "MATIC",
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", "0")),
            "block": int(tx.get("blockNumber", "0")),
        })

    # Process USDC token transfers (most important for Polymarket)
    for tx in token_txs:
        contract = tx.get("contractAddress", "").lower()

        # Only process USDC transfers
        if contract not in USDC_CONTRACTS:
            continue

        # USDC has 6 decimals
        decimals = int(tx.get("tokenDecimal", "6"))
        value_raw = int(tx.get("value", "0"))
        value_usdc = value_raw / (10 ** decimals)

        if value_usdc < MIN_TRANSFER_USDC:
            continue

        all_txs.append({
            "from": tx.get("from", "").lower(),
            "to": tx.get("to", "").lower(),
            "value": value_usdc,
            "value_usd": value_usdc,  # USDC = USD
            "asset": USDC_CONTRACTS[contract],
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", "0")),
            "block": int(tx.get("blockNumber", "0")),
        })

    # Sort by timestamp (oldest first)
    all_txs.sort(key=lambda x: x["timestamp"])

    # Extract funding sources (inbound transfers)
    for tx in all_txs:
        if tx["to"] == wallet_lower:
            record = TransferRecord(
                tx_hash=tx["hash"],
                from_addr=tx["from"],
                to_addr=tx["to"],
                value_matic=tx["value"],  # Actually value in native unit (MATIC or USDC)
                timestamp=datetime.fromtimestamp(tx["timestamp"]).isoformat(),
                block_number=tx["block"],
                transfer_type=f"funding:{tx['asset']}",
            )
            flow.all_funding_sources.append(record)

            # Set primary funding source (first significant one)
            # Prefer USDC over MATIC for primary since that's what Polymarket uses
            if not flow.funding_source or (tx["asset"] in ("USDC", "USDC.e") and "USDC" not in (flow.funding_source_type or "")):
                flow.funding_source = tx["from"]
                flow.funding_tx_hash = tx["hash"]
                flow.funding_timestamp = record.timestamp
                flow.funding_amount_matic = tx["value"]
                addr_type, labels = polygonscan.identify_address_type(tx["from"])
                flow.funding_source_type = f"{addr_type}:{tx['asset']}"
                if labels:
                    flow.funding_source_type = f"{addr_type}:{labels[0]}:{tx['asset']}"

    # Extract withdrawal destinations (outbound transfers)
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

    # Determine primary withdrawal destination (most volume)
    if flow.withdrawal_destinations:
        dest_totals = defaultdict(float)
        for wd in flow.withdrawal_destinations:
            dest_totals[wd.to_addr] += wd.value_matic
        flow.primary_withdrawal_dest = max(dest_totals, key=dest_totals.get)

    if not flow.funding_source and not flow.withdrawal_destinations:
        flow.error = "no_transfers_found"

    return flow


def build_fund_flow_graph(flows: list[WalletFundFlow]) -> FundFlowGraph:
    """Build a complete fund flow graph from wallet flows."""

    graph = FundFlowGraph()

    for flow in flows:
        wallet = flow.wallet.lower()

        # Add trader node
        if wallet not in graph.nodes:
            graph.nodes[wallet] = FundFlowNode(
                address=wallet,
                node_type="trader",
                profit_usd=flow.profit_usd,
            )
        graph.trader_wallets.add(wallet)

        # Add funding source node and edge
        if flow.funding_source:
            funder = flow.funding_source.lower()

            if funder not in graph.nodes:
                addr_type, labels = PolygonscanClient().identify_address_type(funder)
                graph.nodes[funder] = FundFlowNode(
                    address=funder,
                    node_type=addr_type,
                    labels=labels,
                )

            graph.nodes[funder].funded_wallets.append(wallet)
            graph.nodes[wallet].received_from.append(funder)
            graph.funding_sources.add(funder)

            # Add edge
            graph.edges.append({
                "from": funder,
                "to": wallet,
                "value_matic": flow.funding_amount_matic,
                "tx_hash": flow.funding_tx_hash,
                "timestamp": flow.funding_timestamp,
                "type": "funding",
            })

        # Add withdrawal destination nodes and edges
        for wd in flow.withdrawal_destinations:
            dest = wd.to_addr.lower()

            if dest not in graph.nodes:
                addr_type, labels = PolygonscanClient().identify_address_type(dest)
                graph.nodes[dest] = FundFlowNode(
                    address=dest,
                    node_type=addr_type,
                    labels=labels,
                )

            graph.nodes[dest].received_from.append(wallet)
            graph.nodes[wallet].funded_wallets.append(dest)  # trader funded this dest
            graph.withdrawal_dests.add(dest)

            # Add edge
            graph.edges.append({
                "from": wallet,
                "to": dest,
                "value_matic": wd.value_matic,
                "tx_hash": wd.tx_hash,
                "timestamp": wd.timestamp,
                "type": "withdrawal",
            })

    # Detect clusters: withdrawal destinations that are also funding sources
    graph.clusters = detect_clusters(graph)

    return graph


def detect_clusters(graph: FundFlowGraph) -> list[dict]:
    """Detect clusters where withdrawal destinations fund new wallets.

    Pattern: Trader A withdraws to X, X funds Trader B
    This suggests A and B are same operator.
    """
    clusters = []

    # Find addresses that are both withdrawal destinations AND funding sources
    bridge_addresses = graph.withdrawal_dests & graph.funding_sources

    for bridge in bridge_addresses:
        node = graph.nodes.get(bridge)
        if not node:
            continue

        # Skip known exchanges/bridges (not interesting for clustering)
        if node.node_type in ("exchange", "bridge"):
            continue

        # Get traders that withdrew to this address
        withdrew_to = [addr for addr in node.received_from if addr in graph.trader_wallets]

        # Get traders funded by this address
        funded_by = [addr for addr in node.funded_wallets if addr in graph.trader_wallets]

        if withdrew_to and funded_by:
            total_profit = sum(
                graph.nodes[w].profit_usd or 0
                for w in set(withdrew_to + funded_by)
            )

            clusters.append({
                "bridge_address": bridge,
                "withdrew_to_bridge": withdrew_to,
                "funded_by_bridge": funded_by,
                "total_connected_wallets": len(set(withdrew_to + funded_by)),
                "total_profit_usd": total_profit,
                "cluster_type": "withdrawal_refund",
            })

    # Also find wallets with same funding source (simpler clustering)
    funder_groups = defaultdict(list)
    for wallet in graph.trader_wallets:
        node = graph.nodes[wallet]
        for funder in node.received_from:
            if funder not in graph.trader_wallets:  # Don't count trader-to-trader
                funder_groups[funder].append(wallet)

    for funder, wallets in funder_groups.items():
        if len(wallets) >= 2:
            node = graph.nodes.get(funder)
            if node and node.node_type not in ("exchange", "bridge"):
                total_profit = sum(
                    graph.nodes[w].profit_usd or 0 for w in wallets
                )
                clusters.append({
                    "bridge_address": funder,
                    "funded_by_bridge": wallets,
                    "withdrew_to_bridge": [],
                    "total_connected_wallets": len(wallets),
                    "total_profit_usd": total_profit,
                    "cluster_type": "shared_funder",
                })

    # Sort by profit
    clusters.sort(key=lambda x: x["total_profit_usd"], reverse=True)
    return clusters


def lookup_wallet_connections(
    new_wallet_funder: str,
    graph_file: Path,
) -> dict:
    """Check if a wallet's funder is connected to known profitable traders.

    Use this when a new wallet appears - check if its funding source
    is connected to any known profitable traders.
    """

    with open(graph_file) as f:
        graph_data = json.load(f)

    funder = new_wallet_funder.lower()

    result = {
        "funder": funder,
        "is_known": False,
        "connection_type": None,
        "connected_traders": [],
        "total_connected_profit": 0,
        "chain_depth": 0,
    }

    nodes = {n["address"]: n for n in graph_data.get("nodes", [])}

    if funder in nodes:
        node = nodes[funder]
        result["is_known"] = True

        # Direct connections
        if node.get("node_type") == "trader":
            result["connection_type"] = "is_profitable_trader"
            result["connected_traders"] = [funder]
            result["total_connected_profit"] = node.get("profit_usd", 0)
            result["chain_depth"] = 0
        else:
            # This funder previously funded traders or received from traders
            funded = node.get("funded_wallets", [])
            received = node.get("received_from", [])

            traders = []
            for addr in funded + received:
                if addr in nodes and nodes[addr].get("node_type") == "trader":
                    traders.append(addr)

            if traders:
                result["connection_type"] = "funded_by_or_withdrew_to"
                result["connected_traders"] = traders
                result["total_connected_profit"] = sum(
                    nodes[t].get("profit_usd", 0) for t in traders
                )
                result["chain_depth"] = 1

    # Check one more level deep (funder's funders)
    if not result["is_known"]:
        # Would need to do additional API lookup here
        result["connection_type"] = "unknown_not_in_graph"

    return result


async def process_wallets(
    wallets: dict[str, float],
    output_dir: Path,
    resume: bool = False,
) -> tuple[list[WalletFundFlow], FundFlowGraph]:
    """Process all wallets and extract fund flows."""

    results: list[WalletFundFlow] = []

    # Load checkpoint if resuming
    checkpoint_file = output_dir / "extraction_checkpoint.json"
    processed_wallets: set[str] = set()

    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            processed_wallets = set(checkpoint.get("processed", []))
            # Load previous results
            for r in checkpoint.get("results", []):
                flow = WalletFundFlow(
                    wallet=r["wallet"],
                    profit_usd=r["profit_usd"],
                    funding_source=r.get("funding_source"),
                    funding_tx_hash=r.get("funding_tx_hash"),
                    funding_timestamp=r.get("funding_timestamp"),
                    funding_amount_matic=r.get("funding_amount_matic"),
                    funding_source_type=r.get("funding_source_type"),
                    primary_withdrawal_dest=r.get("primary_withdrawal_dest"),
                    total_withdrawn_matic=r.get("total_withdrawn_matic", 0),
                    error=r.get("error"),
                )
                # Reconstruct transfer records
                for fs in r.get("all_funding_sources", []):
                    flow.all_funding_sources.append(TransferRecord(**fs))
                for wd in r.get("withdrawal_destinations", []):
                    flow.withdrawal_destinations.append(TransferRecord(**wd))
                results.append(flow)
            print(f"Resuming from checkpoint: {len(processed_wallets)} already processed")

    # Filter out already processed
    remaining = {w: p for w, p in wallets.items() if w not in processed_wallets}
    total = len(remaining)

    if total == 0:
        print("All wallets already processed!")
        graph = build_fund_flow_graph(results)
        return results, graph

    print(f"\nExtracting fund flows for {total} wallets...")
    print(f"Estimated time: {total * 0.75 / 60:.1f} minutes (3 API calls per wallet: txlist, internal, tokentx)")
    print("-" * 60)

    polygonscan = PolygonscanClient()

    async with httpx.AsyncClient() as client:
        for i, (wallet, profit) in enumerate(remaining.items(), 1):
            # Progress indicator
            if i % 10 == 0 or i == 1:
                pct = i / total * 100
                print(f"  [{i:5}/{total}] {pct:5.1f}% - {wallet[:12]}...", end="", flush=True)

            flow = await extract_fund_flow(wallet, profit, polygonscan, client)
            results.append(flow)
            processed_wallets.add(wallet)

            if i % 10 == 0 or i == 1:
                parts = []
                if flow.funding_source:
                    parts.append(f"fund:{flow.funding_source[:8]}")
                if flow.primary_withdrawal_dest:
                    parts.append(f"wd:{flow.primary_withdrawal_dest[:8]}")
                if parts:
                    print(f" -> {', '.join(parts)}")
                else:
                    print(f" -> {flow.error or 'no data'}")

            # Checkpoint every 50 wallets
            if i % 50 == 0:
                _save_checkpoint(checkpoint_file, processed_wallets, results)

    # Final checkpoint
    _save_checkpoint(checkpoint_file, processed_wallets, results)

    # Build graph
    print("\nBuilding fund flow graph...")
    graph = build_fund_flow_graph(results)
    print(f"  Nodes: {len(graph.nodes):,}")
    print(f"  Edges: {len(graph.edges):,}")
    print(f"  Clusters detected: {len(graph.clusters)}")

    return results, graph


def _save_checkpoint(path: Path, processed: set[str], results: list[WalletFundFlow]):
    """Save checkpoint for resume capability."""

    def transfer_to_dict(t: TransferRecord) -> dict:
        return asdict(t)

    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "processed": list(processed),
        "results": [],
    }

    for r in results:
        rd = {
            "wallet": r.wallet,
            "profit_usd": r.profit_usd,
            "funding_source": r.funding_source,
            "funding_tx_hash": r.funding_tx_hash,
            "funding_timestamp": r.funding_timestamp,
            "funding_amount_matic": r.funding_amount_matic,
            "funding_source_type": r.funding_source_type,
            "all_funding_sources": [transfer_to_dict(t) for t in r.all_funding_sources],
            "withdrawal_destinations": [transfer_to_dict(t) for t in r.withdrawal_destinations],
            "primary_withdrawal_dest": r.primary_withdrawal_dest,
            "total_withdrawn_matic": r.total_withdrawn_matic,
            "error": r.error,
        }
        checkpoint["results"].append(rd)

    with open(path, "w") as f:
        json.dump(checkpoint, f)


def save_results(
    output_dir: Path,
    results: list[WalletFundFlow],
    graph: FundFlowGraph,
):
    """Save all results to JSON files."""

    # 1. Save complete wallet fund flow data
    wallet_file = output_dir / "profitable_wallets_full.json"
    wallet_data = {
        "generated_at": datetime.now().isoformat(),
        "total_wallets": len(results),
        "wallets": [],
    }
    for r in results:
        wallet_data["wallets"].append({
            "wallet": r.wallet,
            "profit_usd": r.profit_usd,
            "funding_source": r.funding_source,
            "funding_source_type": r.funding_source_type,
            "funding_timestamp": r.funding_timestamp,
            "funding_amount_matic": r.funding_amount_matic,
            "primary_withdrawal_dest": r.primary_withdrawal_dest,
            "total_withdrawn_matic": r.total_withdrawn_matic,
            "funding_count": len(r.all_funding_sources),
            "withdrawal_count": len(r.withdrawal_destinations),
        })
    with open(wallet_file, "w") as f:
        json.dump(wallet_data, f, indent=2)
    print(f"Saved wallet data to: {wallet_file}")

    # 2. Save fund flow graph
    graph_file = output_dir / "fund_flow_graph.json"
    graph_data = {
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "trader_wallets": len(graph.trader_wallets),
            "funding_sources": len(graph.funding_sources),
            "withdrawal_destinations": len(graph.withdrawal_dests),
            "clusters_detected": len(graph.clusters),
        },
        "nodes": [
            {
                "address": n.address,
                "node_type": n.node_type,
                "labels": n.labels,
                "profit_usd": n.profit_usd,
                "funded_wallets": n.funded_wallets,
                "received_from": n.received_from,
            }
            for n in graph.nodes.values()
        ],
        "edges": graph.edges,
        "clusters": graph.clusters,
    }
    with open(graph_file, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"Saved fund flow graph to: {graph_file}")

    # 3. Save funding sources summary
    funding_file = output_dir / "funding_sources.json"
    funding_sources = {}
    for wallet in graph.trader_wallets:
        node = graph.nodes[wallet]
        for funder in node.received_from:
            if funder not in funding_sources:
                fnode = graph.nodes[funder]
                funding_sources[funder] = {
                    "address": funder,
                    "source_type": fnode.node_type,
                    "labels": fnode.labels,
                    "funded_wallets": [],
                    "total_profit_funded": 0,
                }
            funding_sources[funder]["funded_wallets"].append(wallet)
            funding_sources[funder]["total_profit_funded"] += node.profit_usd or 0

    sorted_funders = sorted(
        funding_sources.values(),
        key=lambda x: len(x["funded_wallets"]),
        reverse=True,
    )
    with open(funding_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_unique_sources": len(sorted_funders),
            "sources": sorted_funders,
        }, f, indent=2)
    print(f"Saved funding sources to: {funding_file}")

    # 4. Save withdrawal destinations summary
    withdrawal_file = output_dir / "withdrawal_destinations.json"
    withdrawal_dests = {}
    for wallet in graph.trader_wallets:
        node = graph.nodes[wallet]
        for dest in node.funded_wallets:
            if dest in graph.trader_wallets:
                continue  # Skip trader-to-trader (already captured)
            if dest not in withdrawal_dests:
                dnode = graph.nodes[dest]
                withdrawal_dests[dest] = {
                    "address": dest,
                    "dest_type": dnode.node_type,
                    "labels": dnode.labels,
                    "received_from_traders": [],
                    "total_profit_source": 0,
                    "also_funded_traders": [],
                }
            withdrawal_dests[dest]["received_from_traders"].append(wallet)
            withdrawal_dests[dest]["total_profit_source"] += node.profit_usd or 0
            # Check if this dest also funded traders
            dnode = graph.nodes[dest]
            for funded in dnode.funded_wallets:
                if funded in graph.trader_wallets and funded not in withdrawal_dests[dest]["also_funded_traders"]:
                    withdrawal_dests[dest]["also_funded_traders"].append(funded)

    sorted_dests = sorted(
        withdrawal_dests.values(),
        key=lambda x: len(x["received_from_traders"]),
        reverse=True,
    )
    with open(withdrawal_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_unique_destinations": len(sorted_dests),
            "destinations": sorted_dests,
        }, f, indent=2)
    print(f"Saved withdrawal destinations to: {withdrawal_file}")

    # 5. Save cluster report (human-readable)
    report_file = output_dir / "cluster_report.txt"
    with open(report_file, "w") as f:
        f.write("FUND FLOW CLUSTER ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total wallets analyzed: {len(results):,}\n")
        f.write(f"Unique funding sources: {len(graph.funding_sources):,}\n")
        f.write(f"Unique withdrawal destinations: {len(graph.withdrawal_dests):,}\n")
        f.write(f"Clusters detected: {len(graph.clusters)}\n\n")

        # Funding success rate
        funded = sum(1 for r in results if r.funding_source)
        withdrew = sum(1 for r in results if r.withdrawal_destinations)
        f.write(f"Wallets with funding source found: {funded:,} ({100*funded/len(results):.1f}%)\n")
        f.write(f"Wallets with withdrawals found: {withdrew:,} ({100*withdrew/len(results):.1f}%)\n\n")

        f.write("=" * 70 + "\n")
        f.write("DETECTED CLUSTERS\n")
        f.write("=" * 70 + "\n\n")
        f.write("These are addresses that connect multiple profitable traders,\n")
        f.write("suggesting they may be operated by the same entity.\n\n")

        for i, cluster in enumerate(graph.clusters[:30], 1):
            f.write(f"CLUSTER {i}: {cluster['cluster_type'].upper()}\n")
            f.write(f"  Bridge Address: {cluster['bridge_address']}\n")
            f.write(f"  Total Connected Profit: ${cluster['total_profit_usd']:,.0f}\n")

            if cluster.get('withdrew_to_bridge'):
                f.write(f"  Traders that withdrew to this address:\n")
                for w in cluster['withdrew_to_bridge'][:5]:
                    profit = graph.nodes[w].profit_usd or 0
                    f.write(f"    - {w} (${profit:,.0f})\n")
                if len(cluster['withdrew_to_bridge']) > 5:
                    f.write(f"    ... and {len(cluster['withdrew_to_bridge'])-5} more\n")

            if cluster.get('funded_by_bridge'):
                f.write(f"  Traders funded by this address:\n")
                for w in cluster['funded_by_bridge'][:5]:
                    profit = graph.nodes[w].profit_usd or 0
                    f.write(f"    - {w} (${profit:,.0f})\n")
                if len(cluster['funded_by_bridge']) > 5:
                    f.write(f"    ... and {len(cluster['funded_by_bridge'])-5} more\n")

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("TOP FUNDING SOURCES (by wallet count, excluding exchanges/bridges)\n")
        f.write("=" * 70 + "\n\n")

        eoa_funders = [s for s in sorted_funders if s["source_type"] == "eoa"]
        for i, funder in enumerate(eoa_funders[:20], 1):
            f.write(f"{i:2}. {funder['address']}\n")
            f.write(f"    Funded {len(funder['funded_wallets'])} traders | Total profit: ${funder['total_profit_funded']:,.0f}\n")
            f.write(f"    Wallets: {', '.join(funder['funded_wallets'][:3])}")
            if len(funder['funded_wallets']) > 3:
                f.write(f" +{len(funder['funded_wallets'])-3} more")
            f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("WITHDRAWAL DESTINATIONS THAT ALSO FUNDED TRADERS\n")
        f.write("=" * 70 + "\n\n")
        f.write("These are HIGH INTEREST addresses - traders withdrew here,\n")
        f.write("and this address then funded NEW traders (possible recycling).\n\n")

        recyclers = [d for d in sorted_dests if d.get("also_funded_traders")]
        for i, dest in enumerate(recyclers[:20], 1):
            f.write(f"{i:2}. {dest['address']}\n")
            f.write(f"    Received from: {', '.join(dest['received_from_traders'][:3])}\n")
            f.write(f"    Then funded: {', '.join(dest['also_funded_traders'][:3])}\n")
            f.write(f"    Total profit connected: ${dest['total_profit_source']:,.0f}\n\n")

    print(f"Saved cluster report to: {report_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Extract funding sources and withdrawal destinations for profitable traders"
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=20000,
        help="Minimum profit threshold in USD (default: 20000)",
    )
    parser.add_argument(
        "--max-wallets",
        type=int,
        default=10000,
        help="Maximum wallets to process (default: 10000)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Use pre-collected wallet list from file instead of fetching",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--lookup",
        type=str,
        help="Check if a wallet's funder is connected to known profitable traders",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lookup mode
    if args.lookup:
        graph_file = output_dir / "fund_flow_graph.json"
        if not graph_file.exists():
            print(f"ERROR: Graph file not found: {graph_file}")
            print("Run extraction first to build the graph.")
            return

        result = lookup_wallet_connections(args.lookup, graph_file)
        print("\n" + "=" * 60)
        print("WALLET LOOKUP RESULT")
        print("=" * 60)
        print(f"Funder address: {result['funder']}")
        print(f"Is in graph: {result['is_known']}")
        print(f"Connection type: {result['connection_type']}")
        if result['connected_traders']:
            print(f"Connected traders: {len(result['connected_traders'])}")
            for t in result['connected_traders'][:5]:
                print(f"  - {t}")
            print(f"Total connected profit: ${result['total_connected_profit']:,.0f}")
        return

    print("=" * 60)
    print("FUND FLOW EXTRACTION")
    print("=" * 60)
    print(f"Min profit threshold: ${args.min_profit:,.0f}")
    print(f"Max wallets: {args.max_wallets:,}")
    print(f"Output directory: {output_dir}")
    print(f"Resume mode: {args.resume}")

    start_time = datetime.now()

    # Step 1: Collect or load wallets
    if args.input_file:
        print(f"\nLoading wallets from: {args.input_file}")
        wallets = {}
        with open(args.input_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        wallet = parts[0].lower()
                        profit = float(parts[1])
                        if profit >= args.min_profit:
                            wallets[wallet] = profit
        print(f"Loaded {len(wallets)} wallets with >= ${args.min_profit:,.0f} profit")
    else:
        wallets = await collect_profitable_wallets(
            args.min_profit,
            args.max_wallets,
        )

    if not wallets:
        print("No wallets to process!")
        return

    # Step 2: Extract fund flows
    results, graph = await process_wallets(
        wallets,
        output_dir,
        resume=args.resume,
    )

    # Step 3: Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_results(output_dir, results, graph)

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total wallets processed: {len(results):,}")
    print(f"Unique funding sources: {len(graph.funding_sources):,}")
    print(f"Unique withdrawal destinations: {len(graph.withdrawal_dests):,}")
    print(f"Clusters detected: {len(graph.clusters)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())
