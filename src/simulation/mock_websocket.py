"""Mock WebSocket feed for simulation testing.

Provides a mock implementation of the Polymarket WebSocket
that can either connect to the real service or generate
synthetic events for testing.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, AsyncIterator, Optional, Set

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Real Polymarket WebSocket endpoints
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_USER_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


class MockWebSocketFeed:
    """Mock WebSocket feed for simulation.

    Can operate in two modes:
    1. real_connection=True: Connects to actual Polymarket WebSocket
    2. real_connection=False: Generates mock events locally
    """

    def __init__(
        self,
        wallets: List[str],
        real_connection: bool = False,
        markets: Optional[List[str]] = None,
    ):
        self.wallets = set(w.lower() for w in wallets)
        self.real_connection = real_connection
        self.markets = markets or []

        self._ws = None
        self._running = False
        self._trade_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> bool:
        """Connect to WebSocket (real or mock).

        Returns:
            True if connected successfully
        """
        if self.real_connection:
            return await self._connect_real()
        else:
            self._running = True
            return True

    async def _connect_real(self) -> bool:
        """Connect to real Polymarket WebSocket."""
        try:
            import websockets

            self._ws = await websockets.connect(
                POLYMARKET_USER_WS,
                ping_interval=30,
                ping_timeout=10,
            )

            # Subscribe to user activity for each wallet
            for wallet in self.wallets:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "user",
                    "user": wallet,
                }
                await self._ws.send(json.dumps(subscribe_msg))

            self._running = True
            logger.info("websocket_connected", wallets=len(self.wallets))
            return True

        except ImportError:
            logger.error("websockets_not_installed")
            return False
        except Exception as e:
            logger.error("websocket_connect_failed", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

    async def trades(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream trades as async iterator.

        Yields:
            Trade dictionaries from watched wallets
        """
        if not self._running:
            await self.connect()

        if self.real_connection and self._ws:
            async for trade in self._stream_real():
                yield trade
        else:
            async for trade in self._stream_mock():
                yield trade

    async def _stream_real(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream from real WebSocket."""
        try:
            async for message in self._ws:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    trade = self._parse_ws_message(data)

                    if trade:
                        yield trade

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error("parse_message_error", error=str(e))

        except Exception as e:
            logger.error("websocket_stream_error", error=str(e))

    def _parse_ws_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse WebSocket message into trade format.

        Args:
            data: Raw WebSocket message

        Returns:
            Parsed trade dict or None
        """
        msg_type = data.get("type", "")

        # Handle different message types from Polymarket WS
        if msg_type in ("trade", "order_filled", "user_trade"):
            user = data.get("user", "").lower()

            # Check if this is from a watched wallet
            if user not in self.wallets:
                return None

            return {
                "trade_id": data.get("id") or data.get("trade_id"),
                "wallet": user,
                "market_id": data.get("market") or data.get("condition_id"),
                "market_name": data.get("market_title", "Unknown"),
                "outcome": data.get("outcome", "Yes"),
                "side": data.get("side", "BUY").upper(),
                "size": float(data.get("size", 0)),
                "price": float(data.get("price", 0.5)),
                "timestamp": datetime.utcnow(),
            }

        return None

    async def _stream_mock(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream mock trades for testing."""
        from .trade_generator import TradeGenerator

        generator = TradeGenerator(
            wallets=list(self.wallets),
            trades_per_hour=10,
        )

        async for trade in generator.trades():
            if not self._running:
                break
            yield trade


class LiveTradeMonitor:
    """Monitors live trades from target wallets.

    Connects to Polymarket WebSocket and filters for
    trades from specified wallet addresses.
    """

    def __init__(
        self,
        target_wallets: List[str],
        on_trade_callback: Optional[callable] = None,
    ):
        self.target_wallets = set(w.lower() for w in target_wallets)
        self.on_trade_callback = on_trade_callback
        self._ws = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    async def start(self) -> None:
        """Start monitoring for trades."""
        self._running = True

        while self._running:
            try:
                await self._connect_and_monitor()
            except Exception as e:
                logger.error("monitor_error", error=str(e))

                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay,
                    )

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_monitor(self) -> None:
        """Connect to WebSocket and monitor trades."""
        try:
            import websockets

            async with websockets.connect(
                POLYMARKET_USER_WS,
                ping_interval=30,
            ) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0  # Reset on successful connect

                # Subscribe to all target wallets
                for wallet in self.target_wallets:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channel": "user",
                        "user": wallet,
                    }))

                logger.info(
                    "trade_monitor_connected",
                    wallets=len(self.target_wallets),
                )

                # Process messages
                async for message in ws:
                    if not self._running:
                        break

                    await self._process_message(message)

        except ImportError:
            logger.error("websockets_library_required")
            raise

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type in ("trade", "order_filled"):
                user = data.get("user", "").lower()

                if user in self.target_wallets:
                    trade = {
                        "wallet": user,
                        "market_id": data.get("market"),
                        "side": data.get("side", "BUY").upper(),
                        "size": float(data.get("size", 0)),
                        "price": float(data.get("price", 0)),
                        "timestamp": datetime.utcnow(),
                    }

                    logger.info(
                        "target_trade_detected",
                        wallet=user[:10] + "...",
                        side=trade["side"],
                        size=trade["size"],
                    )

                    if self.on_trade_callback:
                        await self.on_trade_callback(trade)

        except Exception as e:
            logger.debug("message_parse_error", error=str(e))


class TradeDetectionBenchmark:
    """Benchmark trade detection latency."""

    def __init__(self, target_wallet: str):
        self.target_wallet = target_wallet.lower()
        self.detection_times: List[float] = []

    async def run_benchmark(self, duration_s: float = 60.0) -> Dict[str, Any]:
        """Run detection latency benchmark.

        Args:
            duration_s: Benchmark duration in seconds

        Returns:
            Benchmark results
        """
        import time

        start_time = time.time()
        trade_count = 0

        async def on_trade(trade: Dict):
            nonlocal trade_count
            detection_time = time.time() - trade.get("block_timestamp", time.time())
            self.detection_times.append(detection_time * 1000)  # Convert to ms
            trade_count += 1

        monitor = LiveTradeMonitor(
            target_wallets=[self.target_wallet],
            on_trade_callback=on_trade,
        )

        # Run for specified duration
        monitor_task = asyncio.create_task(monitor.start())

        await asyncio.sleep(duration_s)
        await monitor.stop()
        monitor_task.cancel()

        # Calculate statistics
        if not self.detection_times:
            return {
                "trades_detected": 0,
                "duration_s": duration_s,
            }

        return {
            "trades_detected": trade_count,
            "duration_s": duration_s,
            "avg_latency_ms": sum(self.detection_times) / len(self.detection_times),
            "min_latency_ms": min(self.detection_times),
            "max_latency_ms": max(self.detection_times),
            "p50_latency_ms": sorted(self.detection_times)[len(self.detection_times) // 2],
            "p95_latency_ms": sorted(self.detection_times)[int(len(self.detection_times) * 0.95)],
        }
