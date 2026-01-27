"""
Per-Account P/L Tracking for Copy Trading

This module tracks profitability per copied account by:
1. Recording executed trades with source account attribution
2. Syncing with Polymarket API to validate actual positions
3. Calculating realized and unrealized P/L per account

The P/L data is validated against actual Polymarket positions to ensure accuracy.
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set
import aiohttp

# Project root for state file
PROJECT_ROOT = Path(__file__).parent
PNL_STATE_FILE = PROJECT_ROOT / "account_pnl_state.json"


@dataclass
class ExecutedTrade:
    """A trade that was actually executed (confirmed on Polymarket)."""

    id: str  # Unique trade ID
    timestamp: datetime
    source_account: str  # The account we copied from
    market_id: str  # Condition ID
    token_id: str  # Asset ID (crucial for matching positions)
    market_name: str
    outcome: str  # "Yes" or "No"
    side: str  # "BUY" or "SELL"
    size: Decimal  # Number of shares
    entry_price: Decimal  # Price we entered at
    usd_amount: Decimal  # Total USD spent/received

    # Validation
    validated: bool = False  # Confirmed via Polymarket API
    order_id: Optional[str] = None  # Polymarket order ID if available
    tx_hash: Optional[str] = None  # Blockchain transaction hash

    # Exit tracking
    is_closed: bool = False
    exit_price: Optional[Decimal] = None
    exit_timestamp: Optional[datetime] = None
    realized_pnl: Optional[Decimal] = None

    # Market resolution
    market_resolved: bool = False
    resolution_outcome: Optional[str] = None  # "Yes" or "No" winner
    redeemed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_account": self.source_account,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "market_name": self.market_name,
            "outcome": self.outcome,
            "side": self.side,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "usd_amount": str(self.usd_amount),
            "validated": self.validated,
            "order_id": self.order_id,
            "tx_hash": self.tx_hash,
            "is_closed": self.is_closed,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
            "market_resolved": self.market_resolved,
            "resolution_outcome": self.resolution_outcome,
            "redeemed": self.redeemed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutedTrade":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_account=data["source_account"],
            market_id=data["market_id"],
            token_id=data["token_id"],
            market_name=data["market_name"],
            outcome=data["outcome"],
            side=data["side"],
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            usd_amount=Decimal(data["usd_amount"]),
            validated=data.get("validated", False),
            order_id=data.get("order_id"),
            tx_hash=data.get("tx_hash"),
            is_closed=data.get("is_closed", False),
            exit_price=Decimal(data["exit_price"]) if data.get("exit_price") else None,
            exit_timestamp=datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None,
            realized_pnl=Decimal(data["realized_pnl"]) if data.get("realized_pnl") else None,
            market_resolved=data.get("market_resolved", False),
            resolution_outcome=data.get("resolution_outcome"),
            redeemed=data.get("redeemed", False),
        )


@dataclass
class AccountPnL:
    """P/L summary for a single copied account."""

    account_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_trades: int = 0

    # P/L in USD
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    # Volume
    total_volume: Decimal = Decimal("0")

    # Validation stats
    validated_trades: int = 0
    unvalidated_trades: int = 0

    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "account_name": self.account_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "open_trades": self.open_trades,
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.total_pnl),
            "total_volume": float(self.total_volume),
            "validated_trades": self.validated_trades,
            "unvalidated_trades": self.unvalidated_trades,
            "win_rate": round(self.winning_trades / max(self.winning_trades + self.losing_trades, 1) * 100, 1),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class AccountPnLTracker:
    """
    Tracks P/L per copied account with Polymarket API validation.

    Key features:
    - Records executed trades with source account attribution
    - Syncs with Polymarket API to validate positions exist
    - Calculates realized P/L from closed positions and market resolutions
    - Calculates unrealized P/L from current market prices
    - Handles unredeemed resolved positions
    """

    POLYMARKET_DATA_API = "https://data-api.polymarket.com"
    POLYMARKET_CLOB_API = "https://clob.polymarket.com"

    def __init__(self, our_wallet: Optional[str] = None):
        self.our_wallet = our_wallet.lower() if our_wallet else None

        # All executed trades, keyed by trade ID
        self.executed_trades: Dict[str, ExecutedTrade] = {}

        # Token ID to trade ID mapping for quick lookups
        self._token_to_trades: Dict[str, Set[str]] = {}

        # Per-account P/L summaries
        self.account_pnl: Dict[str, AccountPnL] = {}

        # Last sync timestamp
        self.last_sync: Optional[datetime] = None

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load state from file."""
        if PNL_STATE_FILE.exists():
            try:
                with open(PNL_STATE_FILE, "r") as f:
                    data = json.load(f)

                # Load executed trades
                for trade_data in data.get("executed_trades", []):
                    trade = ExecutedTrade.from_dict(trade_data)
                    self.executed_trades[trade.id] = trade

                    # Update token mapping
                    if trade.token_id not in self._token_to_trades:
                        self._token_to_trades[trade.token_id] = set()
                    self._token_to_trades[trade.token_id].add(trade.id)

                # Load wallet
                self.our_wallet = data.get("our_wallet")

                print(f"  [PnL Tracker] Loaded {len(self.executed_trades)} executed trades")
            except Exception as e:
                print(f"  [PnL Tracker] Error loading state: {e}")

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "our_wallet": self.our_wallet,
                "executed_trades": [t.to_dict() for t in self.executed_trades.values()],
                "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            }
            with open(PNL_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [PnL Tracker] Error saving state: {e}")

    def set_wallet(self, wallet: str):
        """Set our wallet address for position validation."""
        self.our_wallet = wallet.lower()
        self._save_state()

    def record_trade(
        self,
        trade_id: str,
        source_account: str,
        market_id: str,
        token_id: str,
        market_name: str,
        outcome: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        order_id: Optional[str] = None,
        tx_hash: Optional[str] = None,
    ) -> ExecutedTrade:
        """
        Record a trade that we executed (or attempted to execute).

        Call this when a copy trade order is submitted successfully.
        The trade will be validated against Polymarket API on next sync.
        """
        usd_amount = size * entry_price

        trade = ExecutedTrade(
            id=trade_id,
            timestamp=datetime.utcnow(),
            source_account=source_account,
            market_id=market_id,
            token_id=token_id,
            market_name=market_name,
            outcome=outcome,
            side=side,
            size=size,
            entry_price=entry_price,
            usd_amount=usd_amount,
            order_id=order_id,
            tx_hash=tx_hash,
        )

        self.executed_trades[trade_id] = trade

        # Update token mapping
        if token_id not in self._token_to_trades:
            self._token_to_trades[token_id] = set()
        self._token_to_trades[token_id].add(trade_id)

        self._save_state()
        print(f"  [PnL Tracker] Recorded trade {trade_id[:8]}... from {source_account}")

        return trade

    def record_exit(
        self,
        trade_id: str,
        exit_price: Decimal,
        partial_size: Optional[Decimal] = None,
    ):
        """Record when we exit a position (sell or market resolution)."""
        if trade_id not in self.executed_trades:
            return

        trade = self.executed_trades[trade_id]
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.utcnow()
        trade.is_closed = True

        # Calculate realized P/L
        if trade.side == "BUY":
            # Bought low, sold high = profit
            trade.realized_pnl = (exit_price - trade.entry_price) * trade.size
        else:
            # Sold high, bought back low = profit (short)
            trade.realized_pnl = (trade.entry_price - exit_price) * trade.size

        self._save_state()

    async def sync_with_polymarket(self) -> Dict[str, AccountPnL]:
        """
        Sync with Polymarket API to validate positions and calculate P/L.

        This is the key validation step that ensures P/L is accurate:
        1. Fetch our actual positions from Polymarket
        2. Match positions to our recorded trades
        3. Update validation status
        4. Calculate realized and unrealized P/L
        5. Check for resolved markets and redemption status

        Returns per-account P/L summaries.
        """
        if not self.our_wallet:
            print("  [PnL Tracker] No wallet set, skipping sync")
            return {}

        print(f"  [PnL Tracker] Syncing with Polymarket for {self.our_wallet[:10]}...")

        async with aiohttp.ClientSession() as session:
            # Step 1: Fetch our positions
            positions = await self._fetch_positions(session)

            # Step 2: Fetch our recent activity (trade history)
            activity = await self._fetch_activity(session)

            # Step 3: Match positions to our trades and validate
            await self._validate_trades(positions, activity)

            # Step 4: Check for resolved markets
            await self._check_resolutions(session)

            # Step 5: Calculate per-account P/L
            self._calculate_pnl(positions)

        self.last_sync = datetime.utcnow()
        self._save_state()

        return self.account_pnl

    async def _fetch_positions(self, session: aiohttp.ClientSession) -> List[dict]:
        """Fetch our actual positions from Polymarket."""
        try:
            url = f"{self.POLYMARKET_DATA_API}/positions"
            params = {
                "user": self.our_wallet,
                "limit": 500,
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    positions = data if isinstance(data, list) else data.get("positions", [])
                    print(f"  [PnL Tracker] Fetched {len(positions)} positions")
                    return positions
                else:
                    print(f"  [PnL Tracker] Position fetch failed: {resp.status}")
                    return []
        except Exception as e:
            print(f"  [PnL Tracker] Position fetch error: {e}")
            return []

    async def _fetch_activity(self, session: aiohttp.ClientSession) -> List[dict]:
        """Fetch our trade activity from Polymarket."""
        try:
            url = f"{self.POLYMARKET_DATA_API}/activity"
            params = {
                "user": self.our_wallet,
                "type": "TRADE",
                "limit": 500,
                "sortDirection": "DESC",
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    activity = data if isinstance(data, list) else data.get("activity", [])
                    print(f"  [PnL Tracker] Fetched {len(activity)} activity records")
                    return activity
                else:
                    print(f"  [PnL Tracker] Activity fetch failed: {resp.status}")
                    return []
        except Exception as e:
            print(f"  [PnL Tracker] Activity fetch error: {e}")
            return []

    async def _validate_trades(self, positions: List[dict], activity: List[dict]):
        """Match our recorded trades against actual Polymarket data."""

        # Build lookup sets from Polymarket data
        position_tokens = {p.get("assetId", p.get("asset_id", "")).lower() for p in positions}
        activity_tokens = {a.get("asset", a.get("assetId", "")).lower() for a in activity}

        # Also track by transaction hash
        activity_by_tx = {a.get("transactionHash", ""): a for a in activity if a.get("transactionHash")}

        for trade in self.executed_trades.values():
            if trade.validated:
                continue

            # Method 1: Match by transaction hash (most reliable)
            if trade.tx_hash and trade.tx_hash in activity_by_tx:
                trade.validated = True
                continue

            # Method 2: Check if we have a position for this token
            if trade.token_id.lower() in position_tokens:
                trade.validated = True
                continue

            # Method 3: Check if we have activity for this token
            if trade.token_id.lower() in activity_tokens:
                trade.validated = True
                continue

        validated_count = sum(1 for t in self.executed_trades.values() if t.validated)
        print(f"  [PnL Tracker] Validated {validated_count}/{len(self.executed_trades)} trades")

    async def _check_resolutions(self, session: aiohttp.ClientSession):
        """Check if any markets have resolved."""
        # Get unique market IDs from open trades
        open_market_ids = {
            t.market_id for t in self.executed_trades.values()
            if not t.is_closed and not t.market_resolved
        }

        for market_id in open_market_ids:
            try:
                # Check market status via CLOB API
                url = f"{self.POLYMARKET_CLOB_API}/markets/{market_id}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Check if resolved
                        if data.get("closed") or data.get("resolved"):
                            resolution = data.get("resolution", data.get("winning_outcome"))

                            # Update all trades for this market
                            for trade in self.executed_trades.values():
                                if trade.market_id == market_id and not trade.is_closed:
                                    trade.market_resolved = True
                                    trade.resolution_outcome = resolution

                                    # Calculate P/L from resolution
                                    if resolution:
                                        won = (trade.outcome.lower() == resolution.lower())
                                        exit_price = Decimal("1.0") if won else Decimal("0.0")

                                        if trade.side == "BUY":
                                            trade.realized_pnl = (exit_price - trade.entry_price) * trade.size
                                        else:
                                            trade.realized_pnl = (trade.entry_price - exit_price) * trade.size

                                        trade.exit_price = exit_price
                                        trade.is_closed = True
            except Exception as e:
                # Silently continue on errors
                pass

    def _calculate_pnl(self, positions: List[dict]):
        """Calculate P/L per account from validated trades."""

        # Build position lookup by token ID
        position_by_token = {}
        for pos in positions:
            token_id = pos.get("assetId", pos.get("asset_id", "")).lower()
            position_by_token[token_id] = pos

        # Reset summaries
        self.account_pnl = {}

        for trade in self.executed_trades.values():
            if not trade.validated:
                continue

            account = trade.source_account

            # Initialize account if needed
            if account not in self.account_pnl:
                self.account_pnl[account] = AccountPnL(account_name=account)

            pnl = self.account_pnl[account]
            pnl.total_trades += 1
            pnl.total_volume += trade.usd_amount
            pnl.validated_trades += 1

            if trade.is_closed:
                # Realized P/L
                if trade.realized_pnl is not None:
                    pnl.realized_pnl += trade.realized_pnl
                    if trade.realized_pnl > 0:
                        pnl.winning_trades += 1
                    elif trade.realized_pnl < 0:
                        pnl.losing_trades += 1
            else:
                # Unrealized P/L from current position
                pnl.open_trades += 1

                pos_data = position_by_token.get(trade.token_id.lower())
                if pos_data:
                    # Use Polymarket's calculated unrealized P/L if available
                    unrealized = Decimal(str(pos_data.get("unrealizedPnl", 0)))

                    # Scale by our trade's proportion of the position
                    pos_size = Decimal(str(pos_data.get("size", 1)))
                    if pos_size > 0:
                        proportion = trade.size / pos_size
                        pnl.unrealized_pnl += unrealized * proportion

            pnl.total_pnl = pnl.realized_pnl + pnl.unrealized_pnl
            pnl.last_updated = datetime.utcnow()

        # Count unvalidated trades
        for trade in self.executed_trades.values():
            if not trade.validated:
                account = trade.source_account
                if account not in self.account_pnl:
                    self.account_pnl[account] = AccountPnL(account_name=account)
                self.account_pnl[account].unvalidated_trades += 1

    def _calculate_local_pnl(self) -> Dict[str, AccountPnL]:
        """
        Calculate P/L from local trade records without API validation.

        This provides estimates based on recorded trades. Use this for:
        - Local testing without API access
        - Real-time estimates before next sync
        - Offline mode

        For validated/accurate P/L, use sync_with_polymarket() first.
        """
        local_pnl: Dict[str, AccountPnL] = {}

        for trade in self.executed_trades.values():
            account = trade.source_account
            if account not in local_pnl:
                local_pnl[account] = AccountPnL(account_name=account)

            pnl = local_pnl[account]
            pnl.total_trades += 1
            pnl.total_volume += Decimal(str(trade.usd_amount))

            if trade.validated:
                pnl.validated_trades += 1
            else:
                pnl.unvalidated_trades += 1

            if trade.is_closed:
                # Closed trade - use recorded P/L
                realized = Decimal(str(trade.realized_pnl)) if trade.realized_pnl else Decimal("0")
                pnl.realized_pnl += realized

                if realized > 0:
                    pnl.winning_trades += 1
                elif realized < 0:
                    pnl.losing_trades += 1
            else:
                # Open trade
                pnl.open_trades += 1

            pnl.total_pnl = pnl.realized_pnl + pnl.unrealized_pnl
            pnl.win_rate = pnl.winning_trades / max(1, pnl.winning_trades + pnl.losing_trades)
            pnl.last_updated = datetime.utcnow()

        return local_pnl

    def get_account_pnl(self, account_name: str) -> Optional[AccountPnL]:
        """Get P/L summary for a specific account."""
        # Try synced data first
        if account_name in self.account_pnl:
            return self.account_pnl.get(account_name)

        # Fall back to local calculation
        local = self._calculate_local_pnl()
        return local.get(account_name)

    def get_all_account_pnl(self) -> Dict[str, dict]:
        """Get P/L summaries for all accounts."""
        # If we have synced data, use it
        if self.account_pnl:
            return {name: pnl.to_dict() for name, pnl in self.account_pnl.items()}

        # Fall back to local calculation
        local = self._calculate_local_pnl()
        return {name: pnl.to_dict() for name, pnl in local.items()}

    def get_trades_for_account(self, account_name: str) -> List[dict]:
        """Get all trades copied from a specific account."""
        return [
            t.to_dict() for t in self.executed_trades.values()
            if t.source_account == account_name
        ]


# Singleton instance for global access
_tracker: Optional[AccountPnLTracker] = None


def get_pnl_tracker() -> AccountPnLTracker:
    """Get the global P/L tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = AccountPnLTracker()
    return _tracker


def init_pnl_tracker(wallet: str) -> AccountPnLTracker:
    """Initialize the P/L tracker with our wallet address."""
    global _tracker
    _tracker = AccountPnLTracker(our_wallet=wallet)
    return _tracker
