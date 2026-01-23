"""Connection health monitoring for WebSocket feeds.

This module provides:
- Health checks for WebSocket connections
- Latency monitoring
- Stale data detection
- Connection metrics aggregation
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Awaitable

from ..utils.logging import get_logger
from .client import WebSocketClient, ConnectionState
from .market_feed import MarketFeed, LocalOrderBook
from .user_feed import UserFeed

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class LatencyMetrics:
    """Latency metrics for a connection."""

    samples: list[float] = field(default_factory=list)
    max_samples: int = 100

    def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        if not self.samples:
            return 0
        return sum(self.samples) / len(self.samples)

    @property
    def min_latency_ms(self) -> float:
        """Get minimum latency."""
        return min(self.samples) if self.samples else 0

    @property
    def max_latency_ms(self) -> float:
        """Get maximum latency."""
        return max(self.samples) if self.samples else 0

    @property
    def p95_latency_ms(self) -> float:
        """Get 95th percentile latency."""
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "avg_ms": round(self.avg_latency_ms, 2),
            "min_ms": round(self.min_latency_ms, 2),
            "max_ms": round(self.max_latency_ms, 2),
            "p95_ms": round(self.p95_latency_ms, 2),
            "samples": len(self.samples),
        }


@dataclass
class ConnectionHealth:
    """Health status for a single connection."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    connected: bool = False
    last_message_at: Optional[datetime] = None
    last_check_at: Optional[datetime] = None
    uptime_s: float = 0
    reconnects: int = 0
    errors: int = 0
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    # Thresholds
    stale_threshold_s: float = 30.0

    @property
    def is_stale(self) -> bool:
        """Check if data is stale."""
        if not self.last_message_at:
            return True
        age = (datetime.utcnow() - self.last_message_at).total_seconds()
        return age > self.stale_threshold_s

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "connected": self.connected,
            "stale": self.is_stale,
            "uptime_s": round(self.uptime_s, 2),
            "reconnects": self.reconnects,
            "errors": self.errors,
            "latency": self.latency.to_dict(),
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus = HealthStatus.UNKNOWN
    market_feed: Optional[ConnectionHealth] = None
    user_feed: Optional[ConnectionHealth] = None
    order_books_healthy: int = 0
    order_books_stale: int = 0
    last_check: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "market_feed": self.market_feed.to_dict() if self.market_feed else None,
            "user_feed": self.user_feed.to_dict() if self.user_feed else None,
            "order_books": {
                "healthy": self.order_books_healthy,
                "stale": self.order_books_stale,
            },
            "last_check": self.last_check.isoformat() if self.last_check else None,
        }


# Alert callback type
AlertHandler = Callable[[str, HealthStatus, str], Awaitable[None]]


class HealthMonitor:
    """Monitors health of WebSocket connections.

    Tracks:
    - Connection status
    - Message freshness (stale data detection)
    - Latency metrics
    - Reconnection history

    Example:
        monitor = HealthMonitor(
            market_feed=market_feed,
            user_feed=user_feed,
            on_alert=handle_alert,
        )
        await monitor.start()

        # Check health
        health = monitor.get_health()
        print(health.to_dict())
    """

    def __init__(
        self,
        market_feed: Optional[MarketFeed] = None,
        user_feed: Optional[UserFeed] = None,
        check_interval_s: float = 5.0,
        stale_threshold_s: float = 30.0,
        on_alert: Optional[AlertHandler] = None,
    ):
        """Initialize health monitor.

        Args:
            market_feed: Market feed to monitor
            user_feed: User feed to monitor
            check_interval_s: Health check interval
            stale_threshold_s: Seconds before data is considered stale
            on_alert: Callback for health alerts
        """
        self.market_feed = market_feed
        self.user_feed = user_feed
        self.check_interval_s = check_interval_s
        self.stale_threshold_s = stale_threshold_s
        self._on_alert = on_alert

        # Health state
        self._market_health = ConnectionHealth(
            name="market_feed",
            stale_threshold_s=stale_threshold_s,
        )
        self._user_health = ConnectionHealth(
            name="user_feed",
            stale_threshold_s=stale_threshold_s,
        )

        # Previous states for change detection
        self._prev_market_status: Optional[HealthStatus] = None
        self._prev_user_status: Optional[HealthStatus] = None

        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

        logger.info(
            "health_monitor_started",
            check_interval_s=self.check_interval_s,
        )

    async def stop(self) -> None:
        """Stop health monitoring."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("health_monitor_stopped")

    def get_health(self) -> SystemHealth:
        """Get current system health.

        Returns:
            SystemHealth with all connection states
        """
        # Count order book states
        healthy_books = 0
        stale_books = 0

        if self.market_feed:
            for token_id in self.market_feed._subscribed:
                book = self.market_feed.get_order_book(token_id)
                if book:
                    if book.last_update:
                        age = (datetime.utcnow() - book.last_update).total_seconds()
                        if age < self.stale_threshold_s:
                            healthy_books += 1
                        else:
                            stale_books += 1
                    else:
                        stale_books += 1

        # Determine overall status
        if self._market_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif self._user_health.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif (
            self._market_health.status == HealthStatus.DEGRADED
            or self._user_health.status == HealthStatus.DEGRADED
        ):
            overall_status = HealthStatus.DEGRADED
        elif (
            self._market_health.status == HealthStatus.HEALTHY
            and self._user_health.status == HealthStatus.HEALTHY
        ):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status,
            market_feed=self._market_health,
            user_feed=self._user_health,
            order_books_healthy=healthy_books,
            order_books_stale=stale_books,
            last_check=datetime.utcnow(),
        )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_health()
                await asyncio.sleep(self.check_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))
                await asyncio.sleep(self.check_interval_s)

    async def _check_health(self) -> None:
        """Perform health check."""
        # Check market feed
        if self.market_feed:
            await self._check_market_feed_health()

        # Check user feed
        if self.user_feed:
            await self._check_user_feed_health()

    async def _check_market_feed_health(self) -> None:
        """Check market feed health."""
        health = self._market_health
        feed = self.market_feed

        health.connected = feed.is_connected
        health.last_check_at = datetime.utcnow()

        if feed._client:
            stats = feed._client.stats
            health.reconnects = stats.reconnects
            health.last_message_at = stats.last_message_at

            if stats.connected_at:
                health.uptime_s = (datetime.utcnow() - stats.connected_at).total_seconds()

        # Determine status
        old_status = health.status

        if not health.connected:
            health.status = HealthStatus.UNHEALTHY
        elif health.is_stale:
            health.status = HealthStatus.DEGRADED
        else:
            health.status = HealthStatus.HEALTHY

        # Send alert on status change
        if old_status != health.status and self._on_alert:
            await self._on_alert(
                "market_feed",
                health.status,
                f"Market feed status changed from {old_status.value} to {health.status.value}",
            )

    async def _check_user_feed_health(self) -> None:
        """Check user feed health."""
        health = self._user_health
        feed = self.user_feed

        health.connected = feed.is_connected
        health.last_check_at = datetime.utcnow()

        if feed._client:
            stats = feed._client.stats
            health.reconnects = stats.reconnects
            health.last_message_at = stats.last_message_at

            if stats.connected_at:
                health.uptime_s = (datetime.utcnow() - stats.connected_at).total_seconds()

        # Determine status
        old_status = health.status

        if not health.connected:
            health.status = HealthStatus.UNHEALTHY
        elif health.is_stale:
            health.status = HealthStatus.DEGRADED
        else:
            health.status = HealthStatus.HEALTHY

        # Send alert on status change
        if old_status != health.status and self._on_alert:
            await self._on_alert(
                "user_feed",
                health.status,
                f"User feed status changed from {old_status.value} to {health.status.value}",
            )


class LatencyTracker:
    """Tracks WebSocket message latency.

    Measures round-trip time by sending pings and measuring
    time to receive pongs.
    """

    def __init__(
        self,
        client: WebSocketClient,
        sample_interval_s: float = 10.0,
    ):
        """Initialize latency tracker.

        Args:
            client: WebSocket client to track
            sample_interval_s: Interval between samples
        """
        self.client = client
        self.sample_interval_s = sample_interval_s
        self.metrics = LatencyMetrics()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ping_times: dict[str, datetime] = {}

    async def start(self) -> None:
        """Start latency tracking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        """Stop latency tracking."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def record_pong(self, ping_id: str) -> None:
        """Record pong response.

        Args:
            ping_id: ID of the ping that was responded to
        """
        if ping_id in self._ping_times:
            latency_ms = (datetime.utcnow() - self._ping_times[ping_id]).total_seconds() * 1000
            self.metrics.add_sample(latency_ms)
            del self._ping_times[ping_id]

    async def _sample_loop(self) -> None:
        """Sampling loop."""
        while self._running:
            try:
                await asyncio.sleep(self.sample_interval_s)

                if not self.client.is_connected:
                    continue

                # Send ping with ID
                import uuid
                ping_id = str(uuid.uuid4())
                self._ping_times[ping_id] = datetime.utcnow()

                await self.client.send({
                    "type": "ping",
                    "id": ping_id,
                })

                # Clean up old pings (no response after 30s)
                cutoff = datetime.utcnow() - timedelta(seconds=30)
                self._ping_times = {
                    k: v for k, v in self._ping_times.items()
                    if v > cutoff
                }

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("latency_sample_error", error=str(e))


class ConnectionWatchdog:
    """Watchdog that ensures connections stay alive.

    Automatically restarts connections if they become unhealthy.
    """

    def __init__(
        self,
        market_feed: Optional[MarketFeed] = None,
        user_feed: Optional[UserFeed] = None,
        check_interval_s: float = 10.0,
        max_unhealthy_duration_s: float = 60.0,
    ):
        """Initialize watchdog.

        Args:
            market_feed: Market feed to watch
            user_feed: User feed to watch
            check_interval_s: Check interval
            max_unhealthy_duration_s: Max time unhealthy before restart
        """
        self.market_feed = market_feed
        self.user_feed = user_feed
        self.check_interval_s = check_interval_s
        self.max_unhealthy_duration_s = max_unhealthy_duration_s

        self._market_unhealthy_since: Optional[datetime] = None
        self._user_unhealthy_since: Optional[datetime] = None

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start watchdog."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._watchdog_loop())

        logger.info("connection_watchdog_started")

    async def stop(self) -> None:
        """Stop watchdog."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("connection_watchdog_stopped")

    async def _watchdog_loop(self) -> None:
        """Main watchdog loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_s)

                if self.market_feed:
                    await self._check_feed(
                        "market_feed",
                        self.market_feed,
                        self._market_unhealthy_since,
                        lambda since: setattr(self, "_market_unhealthy_since", since),
                    )

                if self.user_feed:
                    await self._check_feed(
                        "user_feed",
                        self.user_feed,
                        self._user_unhealthy_since,
                        lambda since: setattr(self, "_user_unhealthy_since", since),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("watchdog_error", error=str(e))

    async def _check_feed(
        self,
        name: str,
        feed,
        unhealthy_since: Optional[datetime],
        set_unhealthy: Callable,
    ) -> None:
        """Check a feed and restart if needed.

        Args:
            name: Feed name for logging
            feed: Feed to check
            unhealthy_since: When feed became unhealthy
            set_unhealthy: Setter for unhealthy_since
        """
        is_healthy = feed.is_connected

        if is_healthy:
            set_unhealthy(None)
            return

        # Feed is unhealthy
        now = datetime.utcnow()

        if unhealthy_since is None:
            set_unhealthy(now)
            return

        # Check if unhealthy for too long
        unhealthy_duration = (now - unhealthy_since).total_seconds()

        if unhealthy_duration > self.max_unhealthy_duration_s:
            logger.warning(
                "watchdog_restarting_feed",
                feed=name,
                unhealthy_duration_s=unhealthy_duration,
            )

            # Restart feed
            await feed.stop()
            await asyncio.sleep(1)
            await feed.start()

            set_unhealthy(None)
