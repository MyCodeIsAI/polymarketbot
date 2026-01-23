"""Polymarket Data API client.

This module provides access to the Data API for:
- User positions
- Trade activity
- Trade history
- Market holders

The Data API is used for monitoring target wallets and tracking positions.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from .base import BaseAPIClient, APIResponse
from .rate_limiter import RateLimiter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ActivityType(str, Enum):
    """Types of on-chain activity."""

    TRADE = "TRADE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    REDEEM = "REDEEM"
    REWARD = "REWARD"
    CONVERSION = "CONVERSION"


class TradeSide(str, Enum):
    """Trade side."""

    BUY = "BUY"
    SELL = "SELL"


class PositionSortBy(str, Enum):
    """Sort options for positions."""

    TOKENS = "TOKENS"
    CURRENT = "CURRENT"
    INITIAL = "INITIAL"
    CASHPNL = "CASHPNL"
    PERCENTPNL = "PERCENTPNL"


class SortDirection(str, Enum):
    """Sort direction."""

    ASC = "ASC"
    DESC = "DESC"


@dataclass
class Position:
    """A user's position in a market."""

    condition_id: str
    token_id: str
    outcome: str
    size: Decimal
    average_price: Decimal
    current_value: Decimal
    initial_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    market_slug: Optional[str] = None
    market_title: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "Position":
        """Create Position from API response."""
        return cls(
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            token_id=data.get("assetId", data.get("asset_id", data.get("tokenId", ""))),
            outcome=data.get("outcome", ""),
            size=Decimal(str(data.get("size", 0))),
            average_price=Decimal(str(data.get("avgPrice", data.get("average_price", 0)))),
            current_value=Decimal(str(data.get("currentValue", data.get("current_value", 0)))),
            initial_value=Decimal(str(data.get("initialValue", data.get("initial_value", 0)))),
            realized_pnl=Decimal(str(data.get("realizedPnl", data.get("realized_pnl", 0)))),
            unrealized_pnl=Decimal(str(data.get("unrealizedPnl", data.get("unrealized_pnl", 0)))),
            market_slug=data.get("marketSlug", data.get("market_slug")),
            market_title=data.get("title", data.get("market_title")),
        )


@dataclass
class Activity:
    """An on-chain activity event (trade, split, merge, etc.)."""

    id: str
    type: ActivityType
    timestamp: datetime
    condition_id: str
    token_id: str
    outcome: str
    side: Optional[TradeSide]
    size: Decimal
    price: Decimal
    usd_value: Decimal
    tx_hash: Optional[str] = None
    user: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "Activity":
        """Create Activity from API response."""
        # Parse timestamp
        ts = data.get("timestamp", data.get("createdAt", 0))
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            timestamp = datetime.fromtimestamp(ts)

        # Parse side
        side_str = data.get("side")
        side = TradeSide(side_str) if side_str else None

        # Parse type
        type_str = data.get("type", "TRADE")
        activity_type = ActivityType(type_str) if type_str in ActivityType.__members__ else ActivityType.TRADE

        return cls(
            id=data.get("id", data.get("_id", "")),
            type=activity_type,
            timestamp=timestamp,
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            token_id=data.get("assetId", data.get("asset_id", data.get("tokenId", ""))),
            outcome=data.get("outcome", ""),
            side=side,
            size=Decimal(str(data.get("size", data.get("amount", 0)))),
            price=Decimal(str(data.get("price", 0))),
            usd_value=Decimal(str(data.get("usdcSize", data.get("value", data.get("usdValue", 0))))),
            tx_hash=data.get("transactionHash", data.get("tx_hash")),
            user=data.get("proxyWallet", data.get("user")),
        )


@dataclass
class Trade:
    """A completed trade."""

    id: str
    timestamp: datetime
    market_id: str
    token_id: str
    side: TradeSide
    size: Decimal
    price: Decimal
    maker_address: Optional[str] = None
    taker_address: Optional[str] = None
    tx_hash: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "Trade":
        """Create Trade from API response."""
        ts = data.get("timestamp", data.get("matchTime", 0))
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)

        return cls(
            id=data.get("id", ""),
            timestamp=timestamp,
            market_id=data.get("market", data.get("conditionId", "")),
            token_id=data.get("asset_id", data.get("assetId", "")),
            side=TradeSide(data.get("side", "BUY")),
            size=Decimal(str(data.get("size", 0))),
            price=Decimal(str(data.get("price", 0))),
            maker_address=data.get("maker_address"),
            taker_address=data.get("taker_address"),
            tx_hash=data.get("transactionHash"),
        )


class DataAPIClient(BaseAPIClient):
    """Client for Polymarket Data API.

    Used for reading positions, activity, and trade history.
    No authentication required for public endpoints.

    Example:
        async with DataAPIClient() as client:
            positions = await client.get_positions("0x...")
            activity = await client.get_activity("0x...", type=ActivityType.TRADE)
    """

    DEFAULT_BASE_URL = "https://data-api.polymarket.com"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 10.0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize Data API client.

        Args:
            base_url: Data API base URL
            timeout_s: Request timeout in seconds
            rate_limiter: Optional rate limiter
        """
        if rate_limiter is None:
            rate_limiter = RateLimiter()

        super().__init__(
            base_url=base_url,
            timeout_s=timeout_s,
            rate_limiter=rate_limiter,
        )

    async def get_positions(
        self,
        user: str,
        size_threshold: Optional[float] = None,
        sort_by: PositionSortBy = PositionSortBy.CURRENT,
        sort_direction: SortDirection = SortDirection.DESC,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Position]:
        """Get positions for a user.

        Args:
            user: Wallet address
            size_threshold: Minimum position size to return
            sort_by: Sort field
            sort_direction: Sort direction
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of Position objects
        """
        params = {
            "user": user.lower(),
            "sortBy": sort_by.value,
            "sortDirection": sort_direction.value,
            "limit": limit,
            "offset": offset,
        }
        if size_threshold is not None:
            params["sizeThreshold"] = size_threshold

        response = await self.get("/positions", params=params)

        positions = []
        data = response.data if isinstance(response.data, list) else response.data.get("positions", [])
        for item in data:
            try:
                positions.append(Position.from_api(item))
            except Exception as e:
                logger.warning("position_parse_error", error=str(e), data=item)

        return positions

    async def get_activity(
        self,
        user: str,
        activity_type: Optional[ActivityType] = None,
        condition_id: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        side: Optional[TradeSide] = None,
        sort_by: str = "TIMESTAMP",
        sort_direction: SortDirection = SortDirection.DESC,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Activity]:
        """Get on-chain activity for a user.

        Args:
            user: Wallet address
            activity_type: Filter by activity type (TRADE, SPLIT, etc.)
            condition_id: Filter by market condition ID
            start_timestamp: Filter activities after this timestamp (seconds)
            end_timestamp: Filter activities before this timestamp (seconds)
            side: Filter by trade side (BUY/SELL)
            sort_by: Sort field (TIMESTAMP, TOKENS, CASH)
            sort_direction: Sort direction
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of Activity objects
        """
        params = {
            "user": user.lower(),
            "sortBy": sort_by,
            "sortDirection": sort_direction.value,
            "limit": limit,
            "offset": offset,
        }

        if activity_type:
            params["type"] = activity_type.value
        if condition_id:
            params["market"] = condition_id
        if start_timestamp:
            params["start"] = start_timestamp
        if end_timestamp:
            params["end"] = end_timestamp
        if side:
            params["side"] = side.value

        response = await self.get("/activity", params=params)

        activities = []
        data = response.data if isinstance(response.data, list) else response.data.get("activity", response.data.get("data", []))
        for item in data:
            try:
                activities.append(Activity.from_api(item))
            except Exception as e:
                logger.warning("activity_parse_error", error=str(e), data=item)

        return activities

    async def get_trades(
        self,
        user: Optional[str] = None,
        market: Optional[str] = None,
        maker: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Trade]:
        """Get trade history.

        Args:
            user: Filter by user wallet
            market: Filter by market condition ID
            maker: Filter by maker address
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of Trade objects
        """
        params = {
            "limit": limit,
            "offset": offset,
        }
        if user:
            params["user"] = user.lower()
        if market:
            params["market"] = market
        if maker:
            params["maker"] = maker.lower()

        response = await self.get("/trades", params=params)

        trades = []
        data = response.data if isinstance(response.data, list) else response.data.get("trades", response.data.get("data", []))
        for item in data:
            try:
                trades.append(Trade.from_api(item))
            except Exception as e:
                logger.warning("trade_parse_error", error=str(e), data=item)

        return trades

    async def get_trades_for_user(
        self,
        user: str,
        since_timestamp: Optional[int] = None,
        limit: int = 100,
    ) -> list[Activity]:
        """Get recent trades for a user (convenience method).

        This is optimized for copy-trading by filtering to TRADE type only
        and sorting by timestamp descending (most recent first).

        Args:
            user: Wallet address
            since_timestamp: Only get trades after this timestamp (seconds)
            limit: Maximum results

        Returns:
            List of Activity objects (trades only)
        """
        return await self.get_activity(
            user=user,
            activity_type=ActivityType.TRADE,
            start_timestamp=since_timestamp,
            sort_by="TIMESTAMP",
            sort_direction=SortDirection.DESC,
            limit=limit,
        )

    async def get_portfolio_value(self, user: str) -> dict:
        """Get portfolio value summary for a user.

        Args:
            user: Wallet address

        Returns:
            Dictionary with portfolio value data
        """
        response = await self.get("/value", params={"user": user.lower()})
        return response.data

    async def get_market_holders(
        self,
        condition_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get top holders for a market.

        Args:
            condition_id: Market condition ID
            limit: Maximum results

        Returns:
            List of holder data dictionaries
        """
        response = await self.get("/holders", params={
            "market": condition_id,
            "limit": limit,
        })

        return response.data if isinstance(response.data, list) else response.data.get("holders", [])
