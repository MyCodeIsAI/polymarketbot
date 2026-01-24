"""Profile Fetcher for Insider Scanner.

Fetches comprehensive wallet profile data from Polymarket APIs:
- Full trade history
- All positions (current + closed)
- On-chain activity (splits, merges, redemptions)
- Derived metrics (win rate, PnL, market diversity)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from collections import defaultdict

from ..api.data import DataAPIClient, Position, Activity, ActivityType, TradeSide
from ..api.gamma import GammaAPIClient, Market
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradeEntry:
    """A single trade entry."""

    timestamp: datetime
    market_id: str
    market_title: str
    token_id: str
    side: TradeSide
    size: Decimal
    price: Decimal
    usd_value: Decimal
    tx_hash: Optional[str] = None


@dataclass
class PositionSummary:
    """Summary of a position in a market."""

    market_id: str
    market_title: str
    side: str  # YES or NO
    total_size: Decimal
    total_usd: Decimal
    entry_count: int
    avg_price: Decimal
    first_entry: datetime
    last_entry: datetime
    is_resolved: bool = False
    won: Optional[bool] = None
    pnl: Decimal = Decimal("0")


@dataclass
class WalletProfile:
    """Complete profile for a wallet address."""

    wallet_address: str
    fetched_at: datetime

    # Account metrics
    first_seen: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    account_age_days: int = 0
    transaction_count: int = 0

    # Position metrics
    total_positions: int = 0
    active_positions: int = 0
    resolved_positions: int = 0
    total_position_usd: Decimal = Decimal("0")
    largest_position_usd: Decimal = Decimal("0")

    # Performance metrics
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0

    # Market diversity
    unique_markets: int = 0
    market_categories: dict = field(default_factory=dict)

    # Activity breakdown
    trade_count: int = 0
    split_count: int = 0
    merge_count: int = 0
    redeem_count: int = 0

    # Raw data
    trades: list[TradeEntry] = field(default_factory=list)
    positions: list[PositionSummary] = field(default_factory=list)
    activities: list[Activity] = field(default_factory=list)


class ProfileFetcher:
    """Fetches comprehensive wallet profiles from Polymarket APIs.

    Uses both the Data API (for positions, activity) and Gamma API
    (for market metadata and outcome resolution).

    Example:
        async with ProfileFetcher() as fetcher:
            profile = await fetcher.fetch_profile("0x...")
            print(f"Win rate: {profile.win_rate:.1%}")
    """

    def __init__(
        self,
        data_client: Optional[DataAPIClient] = None,
        gamma_client: Optional[GammaAPIClient] = None,
    ):
        """Initialize the profile fetcher.

        Args:
            data_client: Optional Data API client (created if not provided)
            gamma_client: Optional Gamma API client (created if not provided)
        """
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._owns_clients = data_client is None or gamma_client is None
        self._market_cache: dict[str, Market] = {}

    async def __aenter__(self) -> "ProfileFetcher":
        """Async context manager entry."""
        if self._data_client is None:
            self._data_client = DataAPIClient()
            await self._data_client.__aenter__()
        if self._gamma_client is None:
            self._gamma_client = GammaAPIClient()
            await self._gamma_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._owns_clients:
            if self._data_client:
                await self._data_client.__aexit__(exc_type, exc_val, exc_tb)
            if self._gamma_client:
                await self._gamma_client.__aexit__(exc_type, exc_val, exc_tb)

    async def fetch_profile(
        self,
        wallet_address: str,
        include_trades: bool = True,
        include_activity: bool = True,
        trade_limit: int = 500,
        activity_limit: int = 500,
    ) -> WalletProfile:
        """Fetch comprehensive profile for a wallet.

        Args:
            wallet_address: The wallet address to profile
            include_trades: Whether to fetch full trade history
            include_activity: Whether to fetch on-chain activity
            trade_limit: Maximum trades to fetch
            activity_limit: Maximum activity entries to fetch

        Returns:
            WalletProfile with all available data
        """
        logger.info("fetching_profile", wallet=wallet_address[:10] + "...")

        profile = WalletProfile(
            wallet_address=wallet_address.lower(),
            fetched_at=datetime.utcnow(),
        )

        # Fetch positions
        positions = await self._fetch_positions(wallet_address)
        await self._process_positions(profile, positions)

        # Fetch activity
        if include_activity:
            activities = await self._fetch_activity(wallet_address, limit=activity_limit)
            self._process_activity(profile, activities)

        # Calculate derived metrics
        self._calculate_metrics(profile)

        logger.info(
            "profile_fetched",
            wallet=wallet_address[:10] + "...",
            positions=profile.total_positions,
            win_rate=f"{profile.win_rate:.1%}",
            total_usd=float(profile.total_position_usd),
        )

        return profile

    async def _fetch_positions(self, wallet_address: str) -> list[Position]:
        """Fetch all positions for a wallet."""
        all_positions = []
        offset = 0
        limit = 100

        while True:
            positions = await self._data_client.get_positions(
                user=wallet_address,
                limit=limit,
                offset=offset,
            )

            if not positions:
                break

            all_positions.extend(positions)
            offset += limit

            if len(positions) < limit:
                break

        logger.debug("positions_fetched", count=len(all_positions))
        return all_positions

    async def _fetch_activity(
        self,
        wallet_address: str,
        activity_type: Optional[ActivityType] = None,
        limit: int = 500,
    ) -> list[Activity]:
        """Fetch on-chain activity for a wallet."""
        all_activity = []
        offset = 0
        batch_size = 100

        while len(all_activity) < limit:
            batch = await self._data_client.get_activity(
                user=wallet_address,
                activity_type=activity_type,
                limit=min(batch_size, limit - len(all_activity)),
                offset=offset,
            )

            if not batch:
                break

            all_activity.extend(batch)
            offset += batch_size

            if len(batch) < batch_size:
                break

        logger.debug("activity_fetched", count=len(all_activity))
        return all_activity

    async def _get_market(self, condition_id: str) -> Optional[Market]:
        """Get market info with caching."""
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]

        market = await self._gamma_client.get_market(condition_id)
        if market:
            self._market_cache[condition_id] = market
        return market

    async def _process_positions(
        self,
        profile: WalletProfile,
        positions: list[Position],
    ) -> None:
        """Process positions into profile summaries."""
        # Group by market + outcome
        market_positions: dict[str, dict] = defaultdict(lambda: {
            "market_id": "",
            "market_title": "",
            "side": "",
            "total_size": Decimal("0"),
            "total_usd": Decimal("0"),
            "entries": [],
            "avg_price": Decimal("0"),
            "realized_pnl": Decimal("0"),
            "unrealized_pnl": Decimal("0"),
        })

        for pos in positions:
            key = f"{pos.condition_id}:{pos.outcome}"
            mp = market_positions[key]

            mp["market_id"] = pos.condition_id
            mp["market_title"] = pos.market_title or ""
            mp["side"] = pos.outcome
            mp["total_size"] += pos.size
            mp["total_usd"] += pos.current_value
            mp["entries"].append(pos)
            mp["realized_pnl"] += pos.realized_pnl
            mp["unrealized_pnl"] += pos.unrealized_pnl

        # Convert to summaries
        unique_markets = set()

        for key, mp in market_positions.items():
            if mp["total_size"] == 0:
                continue

            unique_markets.add(mp["market_id"])

            # Get market info for resolution status
            market = await self._get_market(mp["market_id"])
            is_resolved = market.closed if market else False

            # Determine if position won (if resolved)
            won = None
            if is_resolved and market:
                for token in market.tokens:
                    if token.outcome == mp["side"] and token.winner is not None:
                        won = token.winner
                        break

            # Calculate average entry price
            total_cost = sum(e.average_price * e.size for e in mp["entries"])
            avg_price = total_cost / mp["total_size"] if mp["total_size"] > 0 else Decimal("0")

            summary = PositionSummary(
                market_id=mp["market_id"],
                market_title=mp["market_title"],
                side=mp["side"],
                total_size=mp["total_size"],
                total_usd=mp["total_usd"],
                entry_count=len(mp["entries"]),
                avg_price=avg_price,
                first_entry=datetime.utcnow(),  # Will be updated from activity
                last_entry=datetime.utcnow(),
                is_resolved=is_resolved,
                won=won,
                pnl=mp["realized_pnl"] + mp["unrealized_pnl"],
            )

            profile.positions.append(summary)

            # Update profile metrics
            profile.total_position_usd += mp["total_usd"]
            profile.realized_pnl += mp["realized_pnl"]
            profile.unrealized_pnl += mp["unrealized_pnl"]

            if mp["total_usd"] > profile.largest_position_usd:
                profile.largest_position_usd = mp["total_usd"]

            if is_resolved:
                profile.resolved_positions += 1
                if won is True:
                    profile.win_count += 1
                elif won is False:
                    profile.loss_count += 1
            else:
                profile.active_positions += 1

        profile.total_positions = len(profile.positions)
        profile.unique_markets = len(unique_markets)
        profile.total_pnl = profile.realized_pnl + profile.unrealized_pnl

    def _process_activity(
        self,
        profile: WalletProfile,
        activities: list[Activity],
    ) -> None:
        """Process activity data into profile."""
        profile.activities = activities

        if not activities:
            return

        # Track first and last activity
        timestamps = [a.timestamp for a in activities]
        profile.first_seen = min(timestamps)
        profile.last_activity = max(timestamps)
        profile.account_age_days = (datetime.utcnow() - profile.first_seen).days

        # Count by type
        for activity in activities:
            if activity.type == ActivityType.TRADE:
                profile.trade_count += 1

                # Create trade entry
                trade = TradeEntry(
                    timestamp=activity.timestamp,
                    market_id=activity.condition_id,
                    market_title=activity.market_title or "",
                    token_id=activity.token_id,
                    side=activity.side or TradeSide.BUY,
                    size=activity.size,
                    price=activity.price,
                    usd_value=activity.usd_value,
                    tx_hash=activity.tx_hash,
                )
                profile.trades.append(trade)

            elif activity.type == ActivityType.SPLIT:
                profile.split_count += 1
            elif activity.type == ActivityType.MERGE:
                profile.merge_count += 1
            elif activity.type == ActivityType.REDEEM:
                profile.redeem_count += 1

        profile.transaction_count = len(activities)

    def _calculate_metrics(self, profile: WalletProfile) -> None:
        """Calculate derived metrics."""
        # Win rate
        total_resolved = profile.win_count + profile.loss_count
        if total_resolved > 0:
            profile.win_rate = profile.win_count / total_resolved

        # Sort trades by timestamp
        profile.trades.sort(key=lambda t: t.timestamp)

        # Update position entry times from trades
        trade_times: dict[str, list[datetime]] = defaultdict(list)
        for trade in profile.trades:
            if trade.side == TradeSide.BUY:
                trade_times[f"{trade.market_id}:{trade.token_id}"].append(trade.timestamp)

        for pos in profile.positions:
            key_yes = f"{pos.market_id}:YES"
            key_no = f"{pos.market_id}:NO"

            times = trade_times.get(key_yes, []) + trade_times.get(key_no, [])
            if times:
                pos.first_entry = min(times)
                pos.last_entry = max(times)

    async def fetch_cumulative_positions(
        self,
        wallet_address: str,
    ) -> dict[str, PositionSummary]:
        """Fetch cumulative position data grouped by market+side.

        This is specifically for insider detection - tracking total position
        across multiple entry points to detect split entry patterns.

        Args:
            wallet_address: The wallet address to analyze

        Returns:
            Dict mapping "market_id:side" to PositionSummary
        """
        profile = await self.fetch_profile(
            wallet_address,
            include_trades=True,
            include_activity=True,
        )

        # Group trades by market+side to detect cumulative positions
        cumulative: dict[str, PositionSummary] = {}

        for trade in profile.trades:
            if trade.side != TradeSide.BUY:
                continue

            # Determine side from token (YES or NO based on market structure)
            side = "YES"  # Default, would need market lookup for accuracy

            key = f"{trade.market_id}:{side}"

            if key not in cumulative:
                cumulative[key] = PositionSummary(
                    market_id=trade.market_id,
                    market_title=trade.market_title,
                    side=side,
                    total_size=Decimal("0"),
                    total_usd=Decimal("0"),
                    entry_count=0,
                    avg_price=Decimal("0"),
                    first_entry=trade.timestamp,
                    last_entry=trade.timestamp,
                )

            pos = cumulative[key]
            pos.total_size += trade.size
            pos.total_usd += trade.usd_value
            pos.entry_count += 1
            pos.last_entry = max(pos.last_entry, trade.timestamp)

        # Calculate average entry price
        for pos in cumulative.values():
            if pos.entry_count > 0:
                pos.avg_price = pos.total_usd / pos.total_size if pos.total_size > 0 else Decimal("0")

        return cumulative
