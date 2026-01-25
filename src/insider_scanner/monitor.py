"""Real-Time Monitoring for Insider Scanner.

Provides live trade monitoring with:
- WebSocket stream for real-time trades
- Polling fallback when WebSocket unavailable
- Event processing pipeline (Parse → Enrich → Evaluate → Store/Alert)
- New wallet detection
- Automatic scoring of suspicious activity
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Awaitable, List, Dict, Any
from enum import Enum
from collections import defaultdict

from ..websocket import (
    WebSocketClient,
    ConnectionState,
    ReconnectConfig,
)
from ..api.data import DataAPIClient, Activity, ActivityType
from ..utils.logging import get_logger
from .scoring import InsiderScorer, ScoringResult, MarketCategory
from .models import InsiderPriority

logger = get_logger(__name__)


class MonitorState(str, Enum):
    """Monitor operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"  # Fallback to polling
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TradeEvent:
    """A single trade event from the stream."""
    trade_id: str
    timestamp: datetime
    wallet_address: str
    market_id: str
    market_title: str
    token_id: str
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal
    usd_value: Decimal
    tx_hash: Optional[str] = None

    @classmethod
    def from_websocket(cls, data: dict) -> "TradeEvent":
        """Parse from WebSocket message."""
        return cls(
            trade_id=data.get("id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            wallet_address=data.get("maker", data.get("taker", "")),
            market_id=data.get("market", data.get("condition_id", "")),
            market_title=data.get("market_title", ""),
            token_id=data.get("asset_id", data.get("token_id", "")),
            side=data.get("side", "BUY"),
            size=Decimal(str(data.get("size", "0"))),
            price=Decimal(str(data.get("price", "0"))),
            usd_value=Decimal(str(data.get("size", "0"))) * Decimal(str(data.get("price", "0"))),
            tx_hash=data.get("transaction_hash"),
        )

    @classmethod
    def from_activity(cls, activity: Activity) -> "TradeEvent":
        """Parse from Activity dataclass."""
        return cls(
            trade_id=activity.id,
            timestamp=activity.timestamp,
            wallet_address="",  # Not in activity
            market_id=activity.condition_id,
            market_title=activity.market_title or "",
            token_id=activity.token_id,
            side=activity.side.value if activity.side else "BUY",
            size=activity.size,
            price=activity.price,
            usd_value=activity.usd_value,
            tx_hash=activity.tx_hash,
        )


@dataclass
class WalletAccumulation:
    """Tracks a wallet's accumulation in a market."""
    wallet_address: str
    market_id: str
    market_title: str
    total_size: Decimal = Decimal("0")
    total_usd: Decimal = Decimal("0")
    entry_count: int = 0
    first_entry: Optional[datetime] = None
    last_entry: Optional[datetime] = None
    avg_price: Decimal = Decimal("0")
    entry_prices: List[Decimal] = field(default_factory=list)

    def add_trade(self, trade: TradeEvent) -> None:
        """Add a trade to accumulation tracking."""
        if trade.side == "BUY":
            self.total_size += trade.size
            self.total_usd += trade.usd_value
            self.entry_count += 1
            self.entry_prices.append(trade.price)

            if self.first_entry is None:
                self.first_entry = trade.timestamp
            self.last_entry = trade.timestamp

            # Update average price
            if self.entry_count > 0:
                self.avg_price = sum(self.entry_prices) / len(self.entry_prices)


@dataclass
class MonitorStats:
    """Statistics for monitor operation."""
    started_at: Optional[datetime] = None
    trades_processed: int = 0
    wallets_evaluated: int = 0
    alerts_generated: int = 0
    websocket_uptime_s: float = 0
    polling_cycles: int = 0
    last_trade_at: Optional[datetime] = None
    last_evaluation_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "trades_processed": self.trades_processed,
            "wallets_evaluated": self.wallets_evaluated,
            "alerts_generated": self.alerts_generated,
            "websocket_uptime_s": round(self.websocket_uptime_s, 2),
            "polling_cycles": self.polling_cycles,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
        }


# Callback type for alerts
AlertCallback = Callable[[str, ScoringResult, WalletAccumulation], Awaitable[None]]


class RealTimeMonitor:
    """Real-time trade monitoring with WebSocket and polling fallback.

    Features:
    - WebSocket stream for live trades
    - Automatic fallback to polling when WebSocket unavailable
    - Wallet accumulation tracking
    - Automatic scoring against insider patterns
    - Alert callbacks for suspicious activity

    Example:
        async def on_alert(wallet, score, accumulation):
            print(f"Alert: {wallet} scored {score.score}")

        monitor = RealTimeMonitor(on_alert=on_alert)
        await monitor.start()
    """

    # Trade stream WebSocket endpoint (market channel for trade data)
    WS_TRADES_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Gamma API for polling fallback (public, no auth required)
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        scorer: Optional[InsiderScorer] = None,
        data_client: Optional[DataAPIClient] = None,
        on_alert: Optional[AlertCallback] = None,
        alert_threshold: float = 55.0,
        polling_interval_s: float = 30.0,
        accumulation_window_hours: float = 24.0,
        min_position_usd: float = 5000.0,
    ):
        """Initialize real-time monitor.

        Args:
            scorer: InsiderScorer instance (created if not provided)
            data_client: DataAPIClient for polling fallback
            on_alert: Callback for when suspicious wallet detected
            alert_threshold: Minimum score to trigger alert
            polling_interval_s: Seconds between polling cycles
            accumulation_window_hours: Hours to track accumulation
            min_position_usd: Minimum USD position to evaluate
        """
        self.scorer = scorer or InsiderScorer()
        self._data_client = data_client
        self._owns_client = data_client is None
        self.on_alert = on_alert
        self.alert_threshold = alert_threshold
        self.polling_interval_s = polling_interval_s
        self.accumulation_window_hours = accumulation_window_hours
        self.min_position_usd = min_position_usd

        # State
        self._state = MonitorState.STOPPED
        self._ws_client: Optional[WebSocketClient] = None
        self._should_run = False

        # Tasks
        self._ws_task: Optional[asyncio.Task] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Tracking
        self._accumulations: Dict[str, WalletAccumulation] = {}  # wallet:market -> accumulation
        self._alerted_wallets: set = set()  # Don't re-alert same wallet
        self._processed_trade_ids: set = set()  # Dedup across polling cycles
        self._max_processed_ids = 5000  # Limit memory usage
        self._last_poll_timestamp: Optional[datetime] = None

        # Statistics
        self.stats = MonitorStats()

    @property
    def state(self) -> MonitorState:
        """Get current monitor state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._state in (MonitorState.RUNNING, MonitorState.DEGRADED)

    async def start(self) -> bool:
        """Start the real-time monitor.

        Returns:
            True if started successfully
        """
        if self.is_running:
            return True

        self._state = MonitorState.STARTING
        self._should_run = True
        self.stats.started_at = datetime.utcnow()

        # Initialize data client if needed
        if self._data_client is None:
            self._data_client = DataAPIClient()
            await self._data_client.__aenter__()

        # Try WebSocket first
        ws_connected = await self._start_websocket()

        if ws_connected:
            self._state = MonitorState.RUNNING
            logger.info("monitor_started", mode="websocket")
        else:
            # Fallback to polling
            self._state = MonitorState.DEGRADED
            self._start_polling()
            logger.info("monitor_started", mode="polling")

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        return True

    async def stop(self) -> None:
        """Stop the real-time monitor."""
        self._state = MonitorState.STOPPING
        self._should_run = False

        # Cancel tasks
        for task in [self._ws_task, self._polling_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if self._ws_client:
            await self._ws_client.close()
            self._ws_client = None

        # Close data client if we own it
        if self._owns_client and self._data_client:
            await self._data_client.__aexit__(None, None, None)
            self._data_client = None

        self._state = MonitorState.STOPPED
        logger.info("monitor_stopped", stats=self.stats.to_dict())

    async def _start_websocket(self) -> bool:
        """Initialize and connect WebSocket client.

        Returns:
            True if connected successfully
        """
        self._ws_client = WebSocketClient(
            url=self.WS_TRADES_ENDPOINT,
            on_message=self._handle_ws_message,
            on_state_change=self._handle_ws_state_change,
            reconnect_config=ReconnectConfig(
                initial_delay_ms=1000,
                max_delay_ms=60000,
                max_attempts=0,  # Unlimited
            ),
        )

        connected = await self._ws_client.connect()

        if connected:
            # Get high-risk market token IDs to subscribe to
            token_ids = await self._get_high_risk_token_ids()

            if token_ids:
                # Subscribe to market channel with specific token IDs
                # Polymarket CLOB WS expects: {"type": "subscribe", "channel": "market", "assets_ids": [...]}
                # Note: "markets" key causes immediate disconnect, must use "assets_ids"
                await self._ws_client.subscribe({
                    "type": "subscribe",
                    "channel": "market",
                    "assets_ids": token_ids,
                })
                logger.info("ws_subscribed", token_count=len(token_ids))
            else:
                logger.warning("ws_no_tokens_to_subscribe")
                # Fall back to polling if no tokens found
                return False

        return connected

    async def _get_high_risk_token_ids(self) -> list[str]:
        """Get token IDs for high-risk markets (ending within 48 hours).

        Returns:
            List of token IDs to monitor
        """
        import httpx

        token_ids = []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Get active events
                response = await client.get(
                    f"{self.GAMMA_API_BASE}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 50,
                    }
                )

                if response.status_code != 200:
                    return token_ids

                events = response.json()
                now = datetime.utcnow()

                for event in events:
                    end_date_str = event.get("endDate")
                    if not end_date_str:
                        continue

                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", ""))
                        hours_until_end = (end_date - now).total_seconds() / 3600

                        # Focus on events ending in next 48 hours
                        if 0 < hours_until_end < 48:
                            # Get token IDs from markets
                            markets = event.get("markets", [])
                            for market in markets:
                                clob_tokens = market.get("clobTokenIds", [])
                                token_ids.extend(clob_tokens)

                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            logger.debug("get_token_ids_error", error=str(e))

        # Limit to reasonable number
        return token_ids[:100]

    def _start_polling(self) -> None:
        """Start polling fallback."""
        self._polling_task = asyncio.create_task(self._polling_loop())

    async def _handle_ws_message(self, message: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            message: Parsed WebSocket message
        """
        msg_type = message.get("type", "")

        if msg_type == "trade":
            try:
                trade = TradeEvent.from_websocket(message)
                await self._process_trade(trade)
            except Exception as e:
                logger.error("trade_parse_error", error=str(e), message=message)

        elif msg_type == "trades":
            # Batch of trades
            for trade_data in message.get("trades", []):
                try:
                    trade = TradeEvent.from_websocket(trade_data)
                    await self._process_trade(trade)
                except Exception as e:
                    logger.error("trade_parse_error", error=str(e))

    async def _handle_ws_state_change(
        self,
        old_state: ConnectionState,
        new_state: ConnectionState,
    ) -> None:
        """Handle WebSocket state changes.

        Args:
            old_state: Previous connection state
            new_state: New connection state
        """
        logger.info(
            "ws_state_change",
            old_state=old_state.value,
            new_state=new_state.value,
        )

        if new_state == ConnectionState.CONNECTED:
            self._state = MonitorState.RUNNING
            # Stop polling if it was running
            if self._polling_task:
                self._polling_task.cancel()
                self._polling_task = None

        elif new_state in (ConnectionState.RECONNECTING, ConnectionState.DISCONNECTED):
            self._state = MonitorState.DEGRADED
            # Start polling as fallback
            if not self._polling_task or self._polling_task.done():
                self._start_polling()

    async def _polling_loop(self) -> None:
        """Polling fallback loop."""
        logger.info("polling_started", interval_s=self.polling_interval_s)

        while self._should_run:
            try:
                await self._poll_recent_trades()
                self.stats.polling_cycles += 1
                await asyncio.sleep(self.polling_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("polling_error", error=str(e))
                await asyncio.sleep(self.polling_interval_s)

    async def _poll_recent_trades(self) -> None:
        """Poll for recent trades via Data API.

        Fetches actual trades from the public Data API and processes them
        through the insider detection pipeline.
        """
        import httpx

        DATA_API_BASE = "https://data-api.polymarket.com"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Get recent trades from Data API
                # At ~20 trades/sec, we need 1000 to cover 50+ seconds of activity
                response = await client.get(
                    f"{DATA_API_BASE}/trades",
                    params={"limit": 1000}  # API max - covers ~50 sec of trades
                )

                if response.status_code != 200:
                    logger.debug("data_api_error", status=response.status_code)
                    return

                trades_data = response.json()

                if not isinstance(trades_data, list):
                    trades_data = trades_data.get("data", trades_data.get("trades", []))

                trades_processed = 0
                trades_skipped_dup = 0

                for trade_data in trades_data:
                    try:
                        # Get trade ID for deduplication
                        trade_id = str(trade_data.get("id", ""))
                        if not trade_id:
                            trade_id = f"{trade_data.get('proxyWallet', '')}_{trade_data.get('timestamp', '')}"

                        # Skip if already processed (dedup across polling cycles)
                        if trade_id in self._processed_trade_ids:
                            trades_skipped_dup += 1
                            continue

                        # Parse timestamp (API returns Unix epoch as int)
                        ts_raw = trade_data.get("timestamp")
                        if isinstance(ts_raw, int):
                            timestamp = datetime.utcfromtimestamp(ts_raw)
                        elif isinstance(ts_raw, str):
                            timestamp = datetime.fromisoformat(ts_raw.replace("Z", ""))
                        else:
                            timestamp = datetime.utcnow()

                        # Parse trade from Data API format
                        trade = TradeEvent(
                            trade_id=trade_id,
                            timestamp=timestamp,
                            wallet_address=trade_data.get("proxyWallet", ""),
                            market_id=trade_data.get("conditionId", ""),
                            market_title=trade_data.get("title", ""),
                            token_id=trade_data.get("asset", ""),
                            side=trade_data.get("side", "BUY").upper(),
                            size=Decimal(str(trade_data.get("size", "0"))),
                            price=Decimal(str(trade_data.get("price", "0"))),
                            usd_value=Decimal(str(trade_data.get("size", "0"))) * Decimal(str(trade_data.get("price", "0"))),
                            tx_hash=trade_data.get("txHash"),
                        )

                        # Skip if no wallet address
                        if not trade.wallet_address:
                            continue

                        # Mark as processed
                        self._processed_trade_ids.add(trade_id)

                        # Trim dedup set if too large
                        if len(self._processed_trade_ids) > self._max_processed_ids:
                            # Remove oldest half
                            to_remove = list(self._processed_trade_ids)[:self._max_processed_ids // 2]
                            for tid in to_remove:
                                self._processed_trade_ids.discard(tid)

                        await self._process_trade(trade)
                        trades_processed += 1

                    except Exception as e:
                        logger.debug("trade_parse_error", error=str(e))
                        continue

                if trades_processed > 0 or trades_skipped_dup > 0:
                    logger.debug(
                        "trades_polled",
                        new=trades_processed,
                        duplicates=trades_skipped_dup,
                        total_fetched=len(trades_data),
                    )

            self._last_poll_timestamp = datetime.utcnow()

        except Exception as e:
            logger.debug("poll_trades_error", error=str(e))

    async def _process_trade(self, trade: TradeEvent) -> None:
        """Process a single trade event.

        Pipeline: Parse → Enrich → Evaluate → Store/Alert
        """
        self.stats.trades_processed += 1
        self.stats.last_trade_at = trade.timestamp

        # Skip small trades
        if trade.usd_value < self.min_position_usd:
            return

        # Skip sells for accumulation tracking
        if trade.side != "BUY":
            return

        # Track accumulation
        key = f"{trade.wallet_address}:{trade.market_id}"
        if key not in self._accumulations:
            self._accumulations[key] = WalletAccumulation(
                wallet_address=trade.wallet_address,
                market_id=trade.market_id,
                market_title=trade.market_title,
            )

        accumulation = self._accumulations[key]
        accumulation.add_trade(trade)

        # Evaluate if accumulated position meets threshold
        if accumulation.total_usd >= self.min_position_usd:
            await self._evaluate_wallet(trade.wallet_address, accumulation)

    async def _evaluate_wallet(
        self,
        wallet_address: str,
        accumulation: WalletAccumulation,
    ) -> None:
        """Evaluate a wallet for insider patterns.

        Args:
            wallet_address: Wallet to evaluate
            accumulation: Current accumulation data
        """
        # Skip if already alerted for this wallet+market
        alert_key = f"{wallet_address}:{accumulation.market_id}"
        if alert_key in self._alerted_wallets:
            return

        self.stats.wallets_evaluated += 1
        self.stats.last_evaluation_at = datetime.utcnow()

        # Build position data for scorer
        positions = [{
            "market_id": accumulation.market_id,
            "side": "YES",
            "size_usd": float(accumulation.total_usd),
            "size": float(accumulation.total_size),
            "entry_odds": float(accumulation.avg_price),
            "resolved": False,
            "won": None,
        }]

        # Calculate timing (hours since first entry)
        hours_accumulating = 0
        if accumulation.first_entry:
            hours_accumulating = (datetime.utcnow() - accumulation.first_entry).total_seconds() / 3600

        # Score the wallet
        result = self.scorer.score_wallet(
            wallet_address=wallet_address,
            positions=positions,
            event_hours_away=hours_accumulating if hours_accumulating < 72 else None,
        )

        logger.debug(
            "wallet_evaluated",
            wallet=wallet_address[:10] + "...",
            score=result.score,
            priority=result.priority,
            signals=result.signal_count,
        )

        # Check if meets alert threshold
        if result.score >= self.alert_threshold:
            await self._generate_alert(wallet_address, result, accumulation)

    async def _generate_alert(
        self,
        wallet_address: str,
        result: ScoringResult,
        accumulation: WalletAccumulation,
    ) -> None:
        """Generate an alert for suspicious wallet.

        Args:
            wallet_address: Suspicious wallet
            result: Scoring result
            accumulation: Position accumulation data
        """
        alert_key = f"{wallet_address}:{accumulation.market_id}"
        self._alerted_wallets.add(alert_key)
        self.stats.alerts_generated += 1

        logger.warning(
            "insider_alert",
            wallet=wallet_address,
            score=result.score,
            priority=result.priority,
            market=accumulation.market_id,
            position_usd=float(accumulation.total_usd),
            entries=accumulation.entry_count,
        )

        # Call alert callback
        if self.on_alert:
            try:
                await self.on_alert(wallet_address, result, accumulation)
            except Exception as e:
                logger.error("alert_callback_error", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old accumulation data."""
        cleanup_interval = 3600  # 1 hour

        while self._should_run:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    async def _cleanup_old_data(self) -> None:
        """Remove old accumulation entries."""
        cutoff = datetime.utcnow() - timedelta(hours=self.accumulation_window_hours)

        keys_to_remove = []
        for key, acc in self._accumulations.items():
            if acc.last_entry and acc.last_entry < cutoff:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._accumulations[key]

        if keys_to_remove:
            logger.debug("accumulations_cleaned", count=len(keys_to_remove))

    def get_accumulations(self, wallet_address: Optional[str] = None) -> List[WalletAccumulation]:
        """Get current accumulation data.

        Args:
            wallet_address: Filter by wallet (optional)

        Returns:
            List of accumulation records
        """
        accumulations = list(self._accumulations.values())

        if wallet_address:
            accumulations = [a for a in accumulations if a.wallet_address == wallet_address]

        return sorted(accumulations, key=lambda a: a.total_usd, reverse=True)

    def clear_alerts(self) -> None:
        """Clear alerted wallets to allow re-alerting."""
        self._alerted_wallets.clear()
        logger.info("alerts_cleared")


class NewWalletMonitor:
    """Monitor for newly created wallets.

    Detects new Polymarket proxy wallet creations and tracks
    initial deposits that may indicate insider activity.
    """

    def __init__(
        self,
        on_new_wallet: Optional[Callable[[str, Decimal], Awaitable[None]]] = None,
        min_deposit_usd: float = 1000.0,
    ):
        """Initialize new wallet monitor.

        Args:
            on_new_wallet: Callback when new wallet with deposit detected
            min_deposit_usd: Minimum deposit to trigger callback
        """
        self.on_new_wallet = on_new_wallet
        self.min_deposit_usd = min_deposit_usd
        self._known_wallets: set = set()
        self._is_running = False

    async def check_wallet(self, wallet_address: str, deposit_amount: Decimal) -> bool:
        """Check if wallet is new and notify if significant deposit.

        Args:
            wallet_address: Wallet to check
            deposit_amount: Deposit amount in USD

        Returns:
            True if new wallet with significant deposit
        """
        wallet_lower = wallet_address.lower()

        if wallet_lower in self._known_wallets:
            return False

        self._known_wallets.add(wallet_lower)

        if deposit_amount >= self.min_deposit_usd:
            logger.info(
                "new_wallet_detected",
                wallet=wallet_address[:10] + "...",
                deposit_usd=float(deposit_amount),
            )

            if self.on_new_wallet:
                try:
                    await self.on_new_wallet(wallet_address, deposit_amount)
                except Exception as e:
                    logger.error("new_wallet_callback_error", error=str(e))

            return True

        return False
