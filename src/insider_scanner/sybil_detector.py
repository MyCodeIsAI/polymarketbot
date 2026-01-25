"""Sybil detection service for Insider Scanner.

Detects when known profitable traders create new accounts by matching:
1. Funding sources - new wallet funded from same source as profitable trader
2. Withdrawal destinations - new wallet funded from address that received
   withdrawals from profitable traders (chain link detection)

Also tracks suspected insider traders separately when pattern analysis
triggers but no funding match is found.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session

from .models import (
    FlagType,
    FlaggedWallet,
    InsiderPriority,
    WalletStatus,
    KnownProfitableTrader,
    KnownFundingSource,
    KnownWithdrawalDest,
    FundFlowEdge,
    SybilDetection,
)


@dataclass
class SybilMatch:
    """Result of a sybil detection match."""
    matched: bool
    match_type: Optional[str] = None  # funding_source, withdrawal_dest, chain_link
    match_address: Optional[str] = None
    linked_trader_wallet: Optional[str] = None
    linked_trader_id: Optional[int] = None
    linked_trader_profit: Optional[Decimal] = None
    chain_path: Optional[List[str]] = None
    confidence: float = 0.0


class SybilDetector:
    """Service for detecting sybil accounts (profitable trader new accounts).

    Maintains an in-memory index for fast lookups, backed by database
    for persistence. Index is rebuilt on startup.
    """

    def __init__(self, session_factory):
        """Initialize the sybil detector.

        Args:
            session_factory: Callable that returns a database session
        """
        self.session_factory = session_factory

        # In-memory indices for fast lookup
        self._funding_source_index: Dict[str, List[Dict]] = {}  # address -> [trader info]
        self._withdrawal_dest_index: Dict[str, List[Dict]] = {}  # address -> [trader info]
        self._bridge_wallet_index: Dict[str, List[str]] = {}  # address -> [traders it funded]

        # Configuration
        self.sybil_funding_threshold_usd = 20000  # Min funding for sybil detection
        self.insider_funding_threshold_usd = 1000  # Min funding for insider detection

        # High-volume source threshold - sources funding more than this many traders
        # are likely exchanges, OTC desks, or market makers, not personal wallets
        self.max_funded_traders_threshold = 10

        # Exchange addresses (fund many people, less suspicious)
        self._known_exchanges: set = set()

        # High-volume sources (auto-detected during index rebuild)
        self._high_volume_sources: set = set()

        # =====================================================================
        # LOW-HANGING FRUIT EXCLUSION LISTS
        # These are ALWAYS excluded regardless of connection count.
        # Original data is kept in the database - filtering is at index time.
        # =====================================================================

        # Known exchange hot wallet addresses (verified)
        self.KNOWN_EXCHANGE_ADDRESSES: set = {
            # Binance Hot Wallet 2 - funds 232 traders
            "0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245",
            # Bybit - funds 37 traders
            "0xf89d7b9c864f589bbf53a82105107622b35eaa40",
            # Gate.io - funds 4 traders
            "0x0d0707963952f2fba59dd06f2b425ace40b492fe",
            # OKX
            "0x98ec059dc3adfbdd63429454aeb0c990fba4a128",
            # Crypto.com
            "0x6262998ced04146fa42253a5c0af90ca02dfd2a3",
            # KuCoin
            "0xf16e9b0d03470827a95cdfd0cb8a8a3b46969b91",
        }

        # Null/zero/burn addresses - no value for sybil detection
        self.NULL_ADDRESSES: set = {
            "0x0000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000001",  # Near-null
            "0x0000000000000000000000000000000000000002",  # Near-null
            "0x000000000000000000000000000000000000dead",
        }

        # Test/placeholder address patterns (obvious test addresses)
        self.TEST_ADDRESS_PATTERNS: set = {
            "0x1234567890123456789012345678901234567890",  # Sequential hex test
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",  # Repeating pattern
            "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",  # Classic test
        }

        # Known bridge/aggregator contracts - funds many unrelated users
        self.BRIDGE_ADDRESSES: set = {
            # Polygon Bridge
            "0xa0c68c638235ee32657e8f720a23cec1bfc77c77",
            # LayerSwap
            "0x45a318273749d6eb00f5f6ca3bc7cd3de26d642a",
            # Hop Protocol
            "0x25d8039bb044dc227f741a9e381ca4ceae2e6ae8",
            # Multichain (Anyswap)
            "0x4f3aff3a747fcade12598081e80c6605a8be192f",
        }

        # All addresses to exclude (combined)
        self._always_exclude: set = set()

    def rebuild_index(self) -> Dict[str, int]:
        """Rebuild in-memory indices from database.

        Uses a multi-pass approach:
        1. First: Build the always-exclude set (exchanges, nulls, bridges)
        2. Second: Compute TOTAL connections for each address
        3. Third: Apply high-volume threshold filtering
        4. Fourth: Index only non-excluded sources

        Original data in database is NEVER modified - all filtering happens
        at index time. This means if extraction scripts run again, the new
        data will auto-populate and filtering will apply automatically.

        Returns:
            Dict with counts of indexed items and exclusions
        """
        self._funding_source_index.clear()
        self._withdrawal_dest_index.clear()
        self._bridge_wallet_index.clear()
        self._known_exchanges.clear()
        self._high_volume_sources.clear()

        # ====================================================================
        # STEP 1: Build the always-exclude set from known addresses
        # These are ALWAYS excluded regardless of connection count
        # ====================================================================
        self._always_exclude = set()

        # Add known exchange addresses
        for addr in self.KNOWN_EXCHANGE_ADDRESSES:
            self._always_exclude.add(addr.lower())
            self._known_exchanges.add(addr.lower())

        # Add null/burn addresses
        for addr in self.NULL_ADDRESSES:
            self._always_exclude.add(addr.lower())

        # Add known test address patterns
        for addr in self.TEST_ADDRESS_PATTERNS:
            self._always_exclude.add(addr.lower())

        # Add known bridge addresses
        for addr in self.BRIDGE_ADDRESSES:
            self._always_exclude.add(addr.lower())

        always_excluded_count = len(self._always_exclude)

        with self.session_factory() as session:
            # ================================================================
            # STEP 2: Build total connection counts for all addresses
            # ================================================================
            address_connections: Dict[str, Dict[str, int]] = {}

            # Count funding connections
            funding_sources = session.query(KnownFundingSource).all()
            for fs in funding_sources:
                addr_lower = fs.address.lower()
                funded_count = len(fs.funded_trader_wallets or [])

                if addr_lower not in address_connections:
                    address_connections[addr_lower] = {"funding": 0, "withdrawal": 0, "is_exchange": False}

                address_connections[addr_lower]["funding"] = funded_count
                if fs.is_exchange:
                    address_connections[addr_lower]["is_exchange"] = True

            # Count withdrawal connections
            withdrawal_dests = session.query(KnownWithdrawalDest).all()
            for wd in withdrawal_dests:
                addr_lower = wd.address.lower()
                received_count = len(wd.received_from_traders or [])

                if addr_lower not in address_connections:
                    address_connections[addr_lower] = {"funding": 0, "withdrawal": 0, "is_exchange": False}

                address_connections[addr_lower]["withdrawal"] = received_count

            # ================================================================
            # STEP 3: Apply high-volume threshold filtering
            # ================================================================
            for addr, counts in address_connections.items():
                # Skip if already in always-exclude
                if addr in self._always_exclude:
                    self._high_volume_sources.add(addr)
                    continue

                total_connections = counts["funding"] + counts["withdrawal"]

                # Mark as high-volume if EITHER:
                # 1. Total connections > threshold, OR
                # 2. It's marked as an exchange in the database
                if total_connections > self.max_funded_traders_threshold or counts["is_exchange"]:
                    self._high_volume_sources.add(addr)
                    if counts["is_exchange"]:
                        self._known_exchanges.add(addr)

            # ================================================================
            # STEP 4: Index only non-excluded sources
            # ================================================================

            # Index funding sources (skip excluded)
            for fs in funding_sources:
                addr_lower = fs.address.lower()

                # Skip always-excluded and high-volume
                if addr_lower in self._always_exclude or addr_lower in self._high_volume_sources:
                    continue

                if addr_lower not in self._funding_source_index:
                    self._funding_source_index[addr_lower] = []

                for trader_wallet in (fs.funded_trader_wallets or []):
                    self._funding_source_index[addr_lower].append({
                        "trader_wallet": trader_wallet,
                        "source_type": fs.source_type,
                        "is_exchange": fs.is_exchange,
                    })

            # Index withdrawal destinations (skip excluded)
            for wd in withdrawal_dests:
                addr_lower = wd.address.lower()

                # Skip always-excluded and high-volume
                if addr_lower in self._always_exclude or addr_lower in self._high_volume_sources:
                    continue

                if addr_lower not in self._withdrawal_dest_index:
                    self._withdrawal_dest_index[addr_lower] = []

                for trader_wallet in (wd.received_from_traders or []):
                    self._withdrawal_dest_index[addr_lower].append({
                        "trader_wallet": trader_wallet,
                        "is_bridge": wd.is_bridge_wallet,
                    })

                # Bridge wallets that also funded traders
                if wd.is_bridge_wallet and wd.also_funded_traders:
                    self._bridge_wallet_index[addr_lower] = wd.also_funded_traders

        return {
            "funding_sources": len(self._funding_source_index),
            "withdrawal_dests": len(self._withdrawal_dest_index),
            "bridge_wallets": len(self._bridge_wallet_index),
            "exchanges": len(self._known_exchanges),
            "high_volume_excluded": len(self._high_volume_sources),
            "always_excluded": always_excluded_count,
        }

    def check_funding_source(self, funding_address: str) -> SybilMatch:
        """Check if a funding address matches a known profitable trader's source.

        Args:
            funding_address: The address that funded a new wallet

        Returns:
            SybilMatch with match details if found
        """
        addr_lower = funding_address.lower()

        # Skip high-volume sources (exchanges, OTC desks, market makers)
        if addr_lower in self._high_volume_sources:
            return SybilMatch(matched=False)

        # Skip known exchanges
        if addr_lower in self._known_exchanges:
            return SybilMatch(matched=False)

        # Check direct funding source match
        if addr_lower in self._funding_source_index:
            matches = self._funding_source_index[addr_lower]

            # Skip if it's an exchange (too many false positives)
            if addr_lower in self._known_exchanges:
                return SybilMatch(matched=False)

            if matches:
                # Get the most profitable trader linked to this source
                best_match = matches[0]
                trader_info = self._get_trader_info(best_match["trader_wallet"])

                return SybilMatch(
                    matched=True,
                    match_type="funding_source",
                    match_address=funding_address,
                    linked_trader_wallet=best_match["trader_wallet"],
                    linked_trader_id=trader_info.get("id") if trader_info else None,
                    linked_trader_profit=trader_info.get("profit") if trader_info else None,
                    confidence=0.95 if not best_match.get("is_exchange") else 0.3,
                )

        # Check withdrawal destination match (chain link)
        if addr_lower in self._withdrawal_dest_index:
            matches = self._withdrawal_dest_index[addr_lower]

            if matches:
                best_match = matches[0]
                trader_info = self._get_trader_info(best_match["trader_wallet"])

                # Chain: Trader withdrew to this address, which now funds a new wallet
                chain_path = [best_match["trader_wallet"], funding_address]

                return SybilMatch(
                    matched=True,
                    match_type="withdrawal_dest",
                    match_address=funding_address,
                    linked_trader_wallet=best_match["trader_wallet"],
                    linked_trader_id=trader_info.get("id") if trader_info else None,
                    linked_trader_profit=trader_info.get("profit") if trader_info else None,
                    chain_path=chain_path,
                    confidence=0.85,  # Slightly lower confidence for chain link
                )

        return SybilMatch(matched=False)

    def _get_trader_info(self, wallet_address: str) -> Optional[Dict]:
        """Get trader info from database."""
        with self.session_factory() as session:
            trader = session.query(KnownProfitableTrader).filter(
                KnownProfitableTrader.wallet_address.ilike(wallet_address)
            ).first()

            if trader:
                return {
                    "id": trader.id,
                    "wallet": trader.wallet_address,
                    "profit": trader.profit_usd,
                }
        return None

    def flag_sybil_wallet(
        self,
        new_wallet: str,
        funding_address: str,
        funding_amount_matic: Optional[Decimal] = None,
        funding_tx_hash: Optional[str] = None,
    ) -> Optional[FlaggedWallet]:
        """Check and flag a new wallet as a sybil if it matches.

        Args:
            new_wallet: The new wallet address being funded
            funding_address: The address that funded it
            funding_amount_matic: Amount of MATIC transferred
            funding_tx_hash: Transaction hash

        Returns:
            FlaggedWallet if sybil detected, None otherwise
        """
        match = self.check_funding_source(funding_address)

        if not match.matched:
            return None

        with self.session_factory() as session:
            # Check if wallet already flagged
            existing = session.query(FlaggedWallet).filter(
                FlaggedWallet.wallet_address.ilike(new_wallet)
            ).first()

            if existing:
                # Update existing flag if not already sybil
                if existing.flag_type != FlagType.SYBIL:
                    existing.flag_type = FlagType.SYBIL
                    existing.linked_trader_id = match.linked_trader_id
                    existing.funding_match_address = match.match_address
                    session.commit()
                return existing

            # Create new flagged wallet
            flagged = FlaggedWallet(
                wallet_address=new_wallet.lower(),
                flag_type=FlagType.SYBIL,
                linked_trader_id=match.linked_trader_id,
                funding_match_address=match.match_address,
                insider_score=Decimal("75"),  # High score for sybil
                priority=InsiderPriority.HIGH,
                status=WalletStatus.NEW,
                notes=f"Sybil detected: {match.match_type} match via {match.match_address[:10]}...",
            )
            session.add(flagged)

            # Create sybil detection record
            detection = SybilDetection(
                new_wallet_address=new_wallet.lower(),
                linked_trader_wallet=match.linked_trader_wallet,
                linked_trader_id=match.linked_trader_id,
                linked_trader_profit=match.linked_trader_profit,
                match_type=match.match_type,
                match_address=match.match_address,
                chain_path=match.chain_path,
                funding_amount_matic=funding_amount_matic,
                funding_tx_hash=funding_tx_hash,
            )
            session.add(detection)

            session.commit()

            # Update flagged wallet ID in detection
            detection.flagged_wallet_id = flagged.id
            session.commit()

            return flagged

    # =========================================================================
    # Data Import Methods
    # =========================================================================

    def import_profitable_wallets(self, json_path: str) -> Dict[str, int]:
        """Import profitable_wallets_full.json data.

        Args:
            json_path: Path to the JSON file

        Returns:
            Dict with import statistics
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path, "r") as f:
            data = json.load(f)

        stats = {"imported": 0, "updated": 0, "skipped": 0}

        with self.session_factory() as session:
            for wallet_data in data.get("wallets", []):
                wallet_addr = wallet_data.get("wallet", "").lower()
                if not wallet_addr:
                    stats["skipped"] += 1
                    continue

                existing = session.query(KnownProfitableTrader).filter(
                    KnownProfitableTrader.wallet_address == wallet_addr
                ).first()

                if existing:
                    # Update existing
                    existing.profit_usd = Decimal(str(wallet_data.get("profit_usd", 0)))
                    existing.funding_source = wallet_data.get("funding_source", "").lower() if wallet_data.get("funding_source") else None
                    existing.funding_source_type = wallet_data.get("funding_source_type")
                    existing.primary_withdrawal_dest = wallet_data.get("primary_withdrawal_dest", "").lower() if wallet_data.get("primary_withdrawal_dest") else None
                    existing.funding_count = wallet_data.get("funding_count", 1)
                    existing.withdrawal_count = wallet_data.get("withdrawal_count", 0)
                    stats["updated"] += 1
                else:
                    # Create new
                    trader = KnownProfitableTrader(
                        wallet_address=wallet_addr,
                        profit_usd=Decimal(str(wallet_data.get("profit_usd", 0))),
                        funding_source=wallet_data.get("funding_source", "").lower() if wallet_data.get("funding_source") else None,
                        funding_source_type=wallet_data.get("funding_source_type"),
                        funding_amount_matic=Decimal(str(wallet_data.get("funding_amount_matic", 0))) if wallet_data.get("funding_amount_matic") else None,
                        funding_count=wallet_data.get("funding_count", 1),
                        primary_withdrawal_dest=wallet_data.get("primary_withdrawal_dest", "").lower() if wallet_data.get("primary_withdrawal_dest") else None,
                        total_withdrawn_matic=Decimal(str(wallet_data.get("total_withdrawn_matic", 0))) if wallet_data.get("total_withdrawn_matic") else None,
                        withdrawal_count=wallet_data.get("withdrawal_count", 0),
                        data_source="profitable_wallets_full.json",
                    )
                    session.add(trader)
                    stats["imported"] += 1

            session.commit()

        return stats

    def import_funding_sources(self, json_path: str) -> Dict[str, int]:
        """Import funding_sources.json data.

        Args:
            json_path: Path to the JSON file

        Returns:
            Dict with import statistics
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path, "r") as f:
            data = json.load(f)

        stats = {"imported": 0, "updated": 0, "skipped": 0}

        # Known exchange addresses (can expand this list)
        exchange_labels = {"binance", "coinbase", "kraken", "ftx", "okx", "huobi", "kucoin", "bybit"}

        with self.session_factory() as session:
            for source_data in data.get("sources", []):
                addr = source_data.get("address", "").lower()
                if not addr:
                    stats["skipped"] += 1
                    continue

                labels = source_data.get("labels", [])
                is_exchange = any(
                    label.lower() in exchange_labels
                    for label in labels
                ) or source_data.get("source_type") == "exchange"

                existing = session.query(KnownFundingSource).filter(
                    KnownFundingSource.address == addr
                ).first()

                if existing:
                    existing.funded_trader_wallets = [w.lower() for w in source_data.get("funded_wallets", [])]
                    existing.funded_trader_count = len(existing.funded_trader_wallets)
                    existing.total_profit_funded = Decimal(str(source_data.get("total_profit_funded", 0)))
                    existing.is_exchange = is_exchange
                    existing.labels = labels
                    stats["updated"] += 1
                else:
                    funded_wallets = [w.lower() for w in source_data.get("funded_wallets", [])]
                    fs = KnownFundingSource(
                        address=addr,
                        source_type=source_data.get("source_type"),
                        labels=labels,
                        funded_trader_wallets=funded_wallets,
                        funded_trader_count=len(funded_wallets),
                        total_profit_funded=Decimal(str(source_data.get("total_profit_funded", 0))),
                        is_exchange=is_exchange,
                        risk_score=20 if is_exchange else 70,  # Exchanges are less suspicious
                    )
                    session.add(fs)
                    stats["imported"] += 1

            session.commit()

        return stats

    def import_withdrawal_destinations(self, json_path: str) -> Dict[str, int]:
        """Import withdrawal_destinations.json data.

        Args:
            json_path: Path to the JSON file

        Returns:
            Dict with import statistics
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path, "r") as f:
            data = json.load(f)

        stats = {"imported": 0, "updated": 0, "skipped": 0}

        with self.session_factory() as session:
            for dest_data in data.get("destinations", []):
                addr = dest_data.get("address", "").lower()
                if not addr:
                    stats["skipped"] += 1
                    continue

                also_funded = dest_data.get("also_funded_traders", [])
                is_bridge = bool(also_funded)  # Bridge wallet if it also funded traders

                existing = session.query(KnownWithdrawalDest).filter(
                    KnownWithdrawalDest.address == addr
                ).first()

                if existing:
                    existing.received_from_traders = [w.lower() for w in dest_data.get("received_from_traders", [])]
                    existing.received_from_count = len(existing.received_from_traders)
                    existing.total_profit_source = Decimal(str(dest_data.get("total_profit_source", 0)))
                    existing.also_funded_traders = [w.lower() for w in also_funded] if also_funded else None
                    existing.is_bridge_wallet = is_bridge
                    stats["updated"] += 1
                else:
                    received_from = [w.lower() for w in dest_data.get("received_from_traders", [])]
                    wd = KnownWithdrawalDest(
                        address=addr,
                        dest_type=dest_data.get("dest_type"),
                        labels=dest_data.get("labels"),
                        received_from_traders=received_from,
                        received_from_count=len(received_from),
                        total_profit_source=Decimal(str(dest_data.get("total_profit_source", 0))),
                        also_funded_traders=[w.lower() for w in also_funded] if also_funded else None,
                        is_bridge_wallet=is_bridge,
                    )
                    session.add(wd)
                    stats["imported"] += 1

            session.commit()

        return stats

    def import_fund_flow_edges(self, json_path: str) -> Dict[str, int]:
        """Import fund_flow_graph.json edges.

        Args:
            json_path: Path to the JSON file

        Returns:
            Dict with import statistics
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path, "r") as f:
            data = json.load(f)

        stats = {"imported": 0, "skipped": 0}

        with self.session_factory() as session:
            for edge_data in data.get("edges", []):
                from_addr = edge_data.get("from", "").lower()
                to_addr = edge_data.get("to", "").lower()

                if not from_addr or not to_addr:
                    stats["skipped"] += 1
                    continue

                edge = FundFlowEdge(
                    from_address=from_addr,
                    to_address=to_addr,
                    value_matic=Decimal(str(edge_data.get("value_matic", 0))) if edge_data.get("value_matic") else None,
                    tx_hash=edge_data.get("tx_hash"),
                    timestamp=datetime.fromisoformat(edge_data["timestamp"]) if edge_data.get("timestamp") else None,
                    edge_type=edge_data.get("type", "unknown"),
                )
                session.add(edge)
                stats["imported"] += 1

            session.commit()

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the sybil detection database.

        Returns:
            Dict with counts and statistics
        """
        with self.session_factory() as session:
            return {
                "known_profitable_traders": session.query(KnownProfitableTrader).count(),
                "known_funding_sources": session.query(KnownFundingSource).count(),
                "known_withdrawal_dests": session.query(KnownWithdrawalDest).count(),
                "fund_flow_edges": session.query(FundFlowEdge).count(),
                "sybil_detections": session.query(SybilDetection).count(),
                "indexed_funding_sources": len(self._funding_source_index),
                "indexed_withdrawal_dests": len(self._withdrawal_dest_index),
                "indexed_bridge_wallets": len(self._bridge_wallet_index),
                "known_exchanges": len(self._known_exchanges),
                "high_volume_sources_excluded": len(self._high_volume_sources),
                "max_funded_traders_threshold": self.max_funded_traders_threshold,
            }

    def get_recent_sybil_detections(self, limit: int = 50) -> List[Dict]:
        """Get recent sybil detections.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of sybil detection dicts
        """
        with self.session_factory() as session:
            detections = session.query(SybilDetection).order_by(
                SybilDetection.detected_at.desc()
            ).limit(limit).all()

            return [d.to_dict() for d in detections]

    def get_high_volume_sources(self) -> List[Dict]:
        """Get list of high-volume sources that are excluded from sybil detection.

        These are likely exchanges, OTC desks, or market makers.

        Returns:
            List of high-volume source info dicts with total_connections
        """
        results = []
        with self.session_factory() as session:
            for addr in self._high_volume_sources:
                funding_count = 0
                withdrawal_count = 0
                labels = None
                source_type = None

                # Check funding sources
                fs = session.query(KnownFundingSource).filter(
                    KnownFundingSource.address == addr
                ).first()
                if fs:
                    funding_count = fs.funded_trader_count or 0
                    labels = fs.labels
                    source_type = fs.source_type

                # Check withdrawal destinations
                wd = session.query(KnownWithdrawalDest).filter(
                    KnownWithdrawalDest.address == addr
                ).first()
                if wd:
                    withdrawal_count = wd.received_from_count or 0
                    if not labels:
                        labels = wd.labels

                total_connections = funding_count + withdrawal_count

                # Determine type based on which table has more connections
                if funding_count >= withdrawal_count and funding_count > 0:
                    primary_type = "funding_source"
                elif withdrawal_count > 0:
                    primary_type = "withdrawal_dest"
                else:
                    primary_type = "unknown"

                results.append({
                    "address": addr,
                    "type": primary_type,
                    "funded_trader_count": funding_count,
                    "withdrawal_received_count": withdrawal_count,
                    "total_connections": total_connections,
                    "labels": labels,
                    "source_type": source_type,
                    "is_exchange": addr in self._known_exchanges,
                })

        # Sort by total connections (most prolific first)
        results.sort(key=lambda x: x["total_connections"], reverse=True)
        return results

    def get_flagged_funding_sources(self) -> set:
        """Get set of all indexed funding source addresses.

        These are known addresses that have funded profitable traders,
        useful for checking if a new wallet's funder is suspicious.

        Returns:
            Set of lowercase funding source addresses
        """
        return set(self._funding_source_index.keys())

    def get_wallet_cluster(self, wallet_address: str) -> Optional[Dict]:
        """Check if a wallet belongs to a known cluster.

        Clusters are groups of wallets sharing funding sources or
        withdrawal destinations.

        Args:
            wallet_address: Wallet to check

        Returns:
            Cluster info dict if found, None otherwise
        """
        wallet_lower = wallet_address.lower()

        with self.session_factory() as session:
            # Check if wallet is a known profitable trader
            trader = session.query(KnownProfitableTrader).filter(
                KnownProfitableTrader.wallet_address == wallet_lower
            ).first()

            if not trader or not trader.funding_source:
                return None

            funding_source = trader.funding_source.lower()

            # Find other wallets funded by the same source
            cluster_wallets = []
            if funding_source in self._funding_source_index:
                for info in self._funding_source_index[funding_source]:
                    if info.get("wallet") != wallet_lower:
                        cluster_wallets.append(info.get("wallet"))

            if cluster_wallets:
                return {
                    "funding_source": funding_source,
                    "wallets": cluster_wallets,
                    "cluster_size": len(cluster_wallets) + 1,
                }

        return None

    def set_max_funded_traders_threshold(self, threshold: int) -> Dict[str, int]:
        """Update the max funded traders threshold and rebuild the index.

        Args:
            threshold: New threshold value (sources funding more traders are excluded)

        Returns:
            Updated index stats
        """
        self.max_funded_traders_threshold = threshold
        return self.rebuild_index()

    def is_high_volume_source(self, address: str) -> bool:
        """Check if an address is a high-volume source.

        Args:
            address: The address to check

        Returns:
            True if address is a high-volume source (excluded from sybil detection)
        """
        return address.lower() in self._high_volume_sources

    def is_excluded(self, address: str) -> Tuple[bool, Optional[str]]:
        """Check if an address is excluded from sybil detection and why.

        Args:
            address: The address to check

        Returns:
            Tuple of (is_excluded, reason)
            Reasons: 'known_exchange', 'null_address', 'bridge', 'high_volume', 'test_address', None
        """
        addr_lower = address.lower()

        if addr_lower in self.KNOWN_EXCHANGE_ADDRESSES:
            return True, "known_exchange"
        if addr_lower in self.NULL_ADDRESSES:
            return True, "null_address"
        if addr_lower in self.BRIDGE_ADDRESSES:
            return True, "bridge"
        if addr_lower in self.TEST_ADDRESS_PATTERNS:
            return True, "test_address"
        if addr_lower in self._high_volume_sources:
            return True, "high_volume"

        # Check for test address patterns dynamically
        is_test, reason = self.is_test_address(addr_lower)
        if is_test:
            return True, reason

        return False, None

    def is_test_address(self, address: str) -> Tuple[bool, Optional[str]]:
        """Check if an address appears to be a test/placeholder address.

        Detects patterns like:
        - Sequential hex (0x123456...)
        - Near-null addresses (0x00000...001, 0x00000...002)
        - Repeating patterns (0xabcdabcd...)
        - All same character (0xaaaa...)

        Args:
            address: The address to check (should be lowercase)

        Returns:
            Tuple of (is_test, reason)
        """
        addr_lower = address.lower()

        # Remove 0x prefix for pattern checking
        hex_part = addr_lower[2:] if addr_lower.startswith("0x") else addr_lower

        # Check for near-null (mostly zeros with small suffix)
        if hex_part.startswith("0" * 30):  # 30+ leading zeros
            return True, "near_null_address"

        # Check for sequential hex pattern (1234567890...)
        sequential = "1234567890abcdef"
        if any(sequential[i:i+10] in hex_part for i in range(len(sequential) - 9)):
            return True, "sequential_test_pattern"

        # Check for repeating 4-char patterns (abcdabcd...)
        if len(hex_part) == 40:
            chunk = hex_part[:4]
            if hex_part == chunk * 10:
                return True, "repeating_pattern"

            # Check for 8-char repeating patterns
            chunk8 = hex_part[:8]
            if hex_part == chunk8 * 5:
                return True, "repeating_pattern"

        # Check for all same character (0xaaaaaa...)
        if len(set(hex_part)) == 1:
            return True, "single_char_pattern"

        return False, None

    def get_exclusion_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about what's being excluded.

        Returns:
            Dict with breakdown of exclusion reasons
        """
        # Count sources that are dynamically excluded (high-volume but not hardcoded)
        # Use set difference to avoid negative numbers
        dynamic_high_volume = len(self._high_volume_sources - self._always_exclude)

        return {
            "always_excluded": {
                "known_exchanges": len(self.KNOWN_EXCHANGE_ADDRESSES),
                "null_addresses": len(self.NULL_ADDRESSES),
                "bridge_addresses": len(self.BRIDGE_ADDRESSES),
                "total": len(self._always_exclude),
            },
            "dynamic_excluded": {
                "high_volume_threshold": self.max_funded_traders_threshold,
                "high_volume_sources": dynamic_high_volume,
            },
            "total_excluded": len(self._high_volume_sources),
            "indexed": {
                "funding_sources": len(self._funding_source_index),
                "withdrawal_dests": len(self._withdrawal_dest_index),
            },
        }

    def add_exchange_address(self, address: str) -> None:
        """Add an address to the known exchange list.

        This persists in memory until the next rebuild.
        For permanent addition, add to KNOWN_EXCHANGE_ADDRESSES.

        Args:
            address: The exchange address to add
        """
        addr_lower = address.lower()
        self.KNOWN_EXCHANGE_ADDRESSES.add(addr_lower)
        self._always_exclude.add(addr_lower)
        self._known_exchanges.add(addr_lower)
        self._high_volume_sources.add(addr_lower)

        # Remove from indexes if present
        if addr_lower in self._funding_source_index:
            del self._funding_source_index[addr_lower]
        if addr_lower in self._withdrawal_dest_index:
            del self._withdrawal_dest_index[addr_lower]
