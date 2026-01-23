"""Market feed for real-time order book updates.

This module manages subscriptions to market data feeds and
maintains local order book state for quick price lookups.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from ..utils.logging import get_logger
from .client import WebSocketClient, ConnectionState, WS_ENDPOINT

logger = get_logger(__name__)


@dataclass
class BookLevel:
    """A single price level in the order book."""

    price: Decimal
    size: Decimal

    def __lt__(self, other: "BookLevel") -> bool:
        return self.price < other.price


@dataclass
class BookUpdate:
    """An order book update from WebSocket."""

    token_id: str
    timestamp: datetime
    sequence: int
    bids: list[BookLevel]
    asks: list[BookLevel]
    is_snapshot: bool = False


@dataclass
class LocalOrderBook:
    """Local order book maintained from WebSocket updates.

    Uses sorted lists for efficient best bid/ask lookups.
    """

    token_id: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)  # price -> size
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    sequence: int = 0
    last_update: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best (highest) bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best (lowest) ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())

    @property
    def best_bid_size(self) -> Optional[Decimal]:
        """Get size at best bid."""
        price = self.best_bid
        return self.bids.get(price) if price else None

    @property
    def best_ask_size(self) -> Optional[Decimal]:
        """Get size at best ask."""
        price = self.best_ask
        return self.asks.get(price) if price else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return ask - bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get mid-market price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return (bid + ask) / 2
        return None

    def apply_update(self, update: BookUpdate) -> bool:
        """Apply an order book update.

        Args:
            update: The update to apply

        Returns:
            True if update was applied, False if stale
        """
        # Check sequence
        if not update.is_snapshot and update.sequence <= self.sequence:
            return False

        # Snapshot replaces entire book
        if update.is_snapshot:
            self.bids.clear()
            self.asks.clear()

        # Apply bid updates
        for level in update.bids:
            if level.size == Decimal("0"):
                self.bids.pop(level.price, None)
            else:
                self.bids[level.price] = level.size

        # Apply ask updates
        for level in update.asks:
            if level.size == Decimal("0"):
                self.asks.pop(level.price, None)
            else:
                self.asks[level.price] = level.size

        self.sequence = update.sequence
        self.last_update = update.timestamp

        return True

    def get_depth(self, side: str, levels: int = 5) -> list[BookLevel]:
        """Get order book depth.

        Args:
            side: "BUY" for bids, "SELL" for asks
            levels: Number of levels to return

        Returns:
            List of BookLevel sorted by price
        """
        if side == "BUY":
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return [BookLevel(p, self.bids[p]) for p in prices]
        else:
            prices = sorted(self.asks.keys())[:levels]
            return [BookLevel(p, self.asks[p]) for p in prices]

    def get_liquidity(self, side: str, depth_usd: Decimal = Decimal("100")) -> Decimal:
        """Calculate available liquidity up to a USD amount.

        Args:
            side: "BUY" for asks (liquidity to buy), "SELL" for bids
            depth_usd: Maximum USD to sweep

        Returns:
            Total shares available within depth
        """
        book = self.asks if side == "BUY" else self.bids
        prices = sorted(book.keys()) if side == "BUY" else sorted(book.keys(), reverse=True)

        total_shares = Decimal("0")
        remaining_usd = depth_usd

        for price in prices:
            size = book[price]
            level_usd = size * price

            if level_usd <= remaining_usd:
                total_shares += size
                remaining_usd -= level_usd
            else:
                # Partial fill
                shares_at_level = remaining_usd / price
                total_shares += shares_at_level
                break

        return total_shares

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "best_bid": str(self.best_bid) if self.best_bid else None,
            "best_ask": str(self.best_ask) if self.best_ask else None,
            "spread": str(self.spread) if self.spread else None,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "sequence": self.sequence,
        }


# Callback types
BookUpdateHandler = Callable[[str, LocalOrderBook], Awaitable[None]]


@dataclass
class SubscriptionStats:
    """Statistics for market subscriptions."""

    updates_received: int = 0
    snapshots_received: int = 0
    stale_updates: int = 0
    errors: int = 0


class MarketFeed:
    """Manages market data subscriptions and local order books.

    Subscribes to order book updates for markets of interest
    and maintains local order book state.

    Example:
        feed = MarketFeed()
        await feed.start()

        # Subscribe to markets
        await feed.subscribe_market("0xtoken123")

        # Get current order book
        book = feed.get_order_book("0xtoken123")
        print(f"Best bid: {book.best_bid}")

        # Stop
        await feed.stop()
    """

    def __init__(
        self,
        on_update: Optional[BookUpdateHandler] = None,
        auto_reconnect: bool = True,
    ):
        """Initialize market feed.

        Args:
            on_update: Callback when order book updates
            auto_reconnect: Whether to auto-reconnect
        """
        self._on_update = on_update
        self._auto_reconnect = auto_reconnect

        # WebSocket client
        self._client: Optional[WebSocketClient] = None

        # Order books keyed by token_id
        self._books: dict[str, LocalOrderBook] = {}

        # Subscribed markets
        self._subscribed: set[str] = set()
        self._pending_subs: set[str] = set()

        # Statistics per market
        self._stats: dict[str, SubscriptionStats] = defaultdict(SubscriptionStats)

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    async def start(self) -> None:
        """Start the market feed."""
        if self._running:
            return

        self._running = True

        # Create WebSocket client
        self._client = WebSocketClient(
            url=WS_ENDPOINT,
            on_message=self._handle_message,
            on_state_change=self._handle_state_change,
        )

        # Connect
        await self._client.connect()

        logger.info("market_feed_started")

    async def stop(self) -> None:
        """Stop the market feed."""
        if not self._running:
            return

        self._running = False

        if self._client:
            await self._client.close()
            self._client = None

        logger.info(
            "market_feed_stopped",
            markets=len(self._subscribed),
            total_updates=sum(s.updates_received for s in self._stats.values()),
        )

    async def subscribe_market(self, token_id: str) -> bool:
        """Subscribe to order book updates for a market.

        Args:
            token_id: Token ID to subscribe to

        Returns:
            True if subscription sent
        """
        if token_id in self._subscribed:
            return True

        # Initialize order book
        if token_id not in self._books:
            self._books[token_id] = LocalOrderBook(token_id=token_id)

        # Build subscription message
        subscription = {
            "type": "subscribe",
            "channel": "book",
            "market": token_id,
        }

        if self.is_connected:
            success = await self._client.subscribe(subscription)
            if success:
                self._subscribed.add(token_id)
                logger.debug("market_subscribed", token_id=token_id[:16] + "...")
                return True
        else:
            # Queue for when connected
            self._pending_subs.add(token_id)
            return True

        return False

    async def unsubscribe_market(self, token_id: str) -> bool:
        """Unsubscribe from a market.

        Args:
            token_id: Token ID to unsubscribe from

        Returns:
            True if unsubscription sent
        """
        if token_id not in self._subscribed:
            return True

        unsubscription = {
            "type": "unsubscribe",
            "channel": "book",
            "market": token_id,
        }

        if self.is_connected:
            success = await self._client.send(unsubscription)
            if success:
                self._subscribed.discard(token_id)
                logger.debug("market_unsubscribed", token_id=token_id[:16] + "...")
                return True

        return False

    def get_order_book(self, token_id: str) -> Optional[LocalOrderBook]:
        """Get local order book for a market.

        Args:
            token_id: Token ID

        Returns:
            LocalOrderBook or None if not subscribed
        """
        return self._books.get(token_id)

    def get_best_bid(self, token_id: str) -> Optional[Decimal]:
        """Get best bid price for a market."""
        book = self._books.get(token_id)
        return book.best_bid if book else None

    def get_best_ask(self, token_id: str) -> Optional[Decimal]:
        """Get best ask price for a market."""
        book = self._books.get(token_id)
        return book.best_ask if book else None

    def get_stats(self, token_id: str) -> Optional[SubscriptionStats]:
        """Get subscription statistics for a market."""
        return self._stats.get(token_id)

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all markets."""
        return {
            tid: {
                "updates": s.updates_received,
                "snapshots": s.snapshots_received,
                "stale": s.stale_updates,
                "errors": s.errors,
            }
            for tid, s in self._stats.items()
        }

    async def _handle_message(self, message: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            message: Parsed JSON message
        """
        msg_type = message.get("type", "")

        if msg_type == "book":
            await self._handle_book_update(message)
        elif msg_type == "snapshot":
            await self._handle_snapshot(message)
        elif msg_type == "subscribed":
            self._handle_subscribed(message)
        elif msg_type == "error":
            self._handle_error(message)

    async def _handle_book_update(self, message: dict) -> None:
        """Handle incremental book update.

        Args:
            message: Book update message
        """
        try:
            token_id = message.get("market", "")
            if not token_id or token_id not in self._books:
                return

            # Parse update
            update = self._parse_book_message(message, is_snapshot=False)

            # Apply to local book
            book = self._books[token_id]
            applied = book.apply_update(update)

            stats = self._stats[token_id]
            if applied:
                stats.updates_received += 1

                # Notify handler
                if self._on_update:
                    await self._on_update(token_id, book)
            else:
                stats.stale_updates += 1

        except Exception as e:
            logger.warning("book_update_error", error=str(e))
            self._stats[token_id].errors += 1

    async def _handle_snapshot(self, message: dict) -> None:
        """Handle full book snapshot.

        Args:
            message: Snapshot message
        """
        try:
            token_id = message.get("market", "")
            if not token_id:
                return

            # Ensure book exists
            if token_id not in self._books:
                self._books[token_id] = LocalOrderBook(token_id=token_id)

            # Parse snapshot
            update = self._parse_book_message(message, is_snapshot=True)

            # Apply to local book
            book = self._books[token_id]
            book.apply_update(update)

            self._stats[token_id].snapshots_received += 1

            logger.debug(
                "book_snapshot_received",
                token_id=token_id[:16] + "...",
                bids=len(book.bids),
                asks=len(book.asks),
            )

            # Notify handler
            if self._on_update:
                await self._on_update(token_id, book)

        except Exception as e:
            logger.warning("snapshot_error", error=str(e))

    def _parse_book_message(self, message: dict, is_snapshot: bool) -> BookUpdate:
        """Parse a book message into BookUpdate.

        Args:
            message: Raw message
            is_snapshot: Whether this is a snapshot

        Returns:
            Parsed BookUpdate
        """
        token_id = message.get("market", "")
        sequence = message.get("sequence", 0)

        # Parse bids
        bids = []
        for bid in message.get("bids", []):
            if isinstance(bid, dict):
                price = Decimal(str(bid.get("price", "0")))
                size = Decimal(str(bid.get("size", "0")))
            else:
                # Array format [price, size]
                price = Decimal(str(bid[0]))
                size = Decimal(str(bid[1]))
            bids.append(BookLevel(price=price, size=size))

        # Parse asks
        asks = []
        for ask in message.get("asks", []):
            if isinstance(ask, dict):
                price = Decimal(str(ask.get("price", "0")))
                size = Decimal(str(ask.get("size", "0")))
            else:
                price = Decimal(str(ask[0]))
                size = Decimal(str(ask[1]))
            asks.append(BookLevel(price=price, size=size))

        return BookUpdate(
            token_id=token_id,
            timestamp=datetime.utcnow(),
            sequence=sequence,
            bids=bids,
            asks=asks,
            is_snapshot=is_snapshot,
        )

    def _handle_subscribed(self, message: dict) -> None:
        """Handle subscription confirmation.

        Args:
            message: Subscribed message
        """
        market = message.get("market", "")
        if market:
            self._subscribed.add(market)
            logger.debug("subscription_confirmed", market=market[:16] + "...")

    def _handle_error(self, message: dict) -> None:
        """Handle error message.

        Args:
            message: Error message
        """
        error = message.get("error", "Unknown error")
        market = message.get("market", "")

        logger.warning(
            "market_feed_error",
            error=error,
            market=market[:16] + "..." if market else None,
        )

        if market:
            self._stats[market].errors += 1

    async def _handle_state_change(
        self,
        old_state: ConnectionState,
        new_state: ConnectionState,
    ) -> None:
        """Handle WebSocket state changes.

        Args:
            old_state: Previous state
            new_state: New state
        """
        logger.info(
            "market_feed_state_change",
            old_state=old_state.value,
            new_state=new_state.value,
        )

        if new_state == ConnectionState.CONNECTED:
            # Resubscribe to all markets
            await self._resubscribe_all()

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all markets after reconnect."""
        markets = list(self._subscribed) + list(self._pending_subs)
        self._subscribed.clear()
        self._pending_subs.clear()

        for token_id in markets:
            await self.subscribe_market(token_id)

        logger.info("markets_resubscribed", count=len(markets))


class MultiMarketFeed:
    """Convenience wrapper for subscribing to multiple markets.

    Manages subscriptions based on active targets and their positions.
    """

    def __init__(self, feed: MarketFeed):
        """Initialize multi-market feed.

        Args:
            feed: Underlying market feed
        """
        self.feed = feed
        self._active_markets: set[str] = set()

    async def track_markets(self, token_ids: list[str]) -> None:
        """Start tracking a list of markets.

        Args:
            token_ids: Markets to track
        """
        for token_id in token_ids:
            if token_id not in self._active_markets:
                await self.feed.subscribe_market(token_id)
                self._active_markets.add(token_id)

    async def untrack_markets(self, token_ids: list[str]) -> None:
        """Stop tracking markets.

        Args:
            token_ids: Markets to stop tracking
        """
        for token_id in token_ids:
            if token_id in self._active_markets:
                await self.feed.unsubscribe_market(token_id)
                self._active_markets.discard(token_id)

    async def untrack_all(self) -> None:
        """Stop tracking all markets."""
        await self.untrack_markets(list(self._active_markets))

    def get_best_prices(self) -> dict[str, tuple[Optional[Decimal], Optional[Decimal]]]:
        """Get best bid/ask for all tracked markets.

        Returns:
            Dict mapping token_id to (best_bid, best_ask)
        """
        prices = {}
        for token_id in self._active_markets:
            book = self.feed.get_order_book(token_id)
            if book:
                prices[token_id] = (book.best_bid, book.best_ask)
        return prices
