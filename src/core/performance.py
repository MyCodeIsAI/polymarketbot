"""High-performance configuration and optimization for copy trading.

This module provides:
- Aggressive performance tuning for low-latency trading
- Connection warm-up and keep-alive
- Real-time latency monitoring with alerts
- Optimized execution paths

CRITICAL: This is real money trading. Speed is measured in milliseconds.
Every optimization matters for execution quality and slippage.
"""

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable, List
from collections import deque
from statistics import mean, median, stdev

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMode(str, Enum):
    """Trading performance modes."""

    # Maximum speed, minimal checks - for copy trading
    AGGRESSIVE = "aggressive"

    # Balanced speed and safety - default
    BALANCED = "balanced"

    # Maximum safety, slower - for testing
    CONSERVATIVE = "conservative"


@dataclass
class PerformanceConfig:
    """Performance tuning configuration.

    Optimized defaults for low-latency copy trading.
    """

    mode: PerformanceMode = PerformanceMode.AGGRESSIVE

    # HTTP Client Settings
    http_timeout_ms: int = 2000  # 2s max for any HTTP call
    http_connect_timeout_ms: int = 1000  # 1s to establish connection
    http_pool_size: int = 30  # Connection pool
    http_keepalive_s: int = 60  # Keep connections alive longer
    http2_enabled: bool = True  # Use HTTP/2 multiplexing

    # Order Submission (CRITICAL PATH)
    order_submit_timeout_ms: int = 1500  # 1.5s max for order submission
    order_status_poll_ms: int = 100  # Poll every 100ms (was 500ms)
    order_max_polls: int = 20  # Max 2s of polling

    # Pre-flight Checks
    skip_preflight_for_copy: bool = True  # Skip order book fetch for copy trades
    max_price_drift_pct: Decimal = Decimal("0.03")  # 3% drift tolerance

    # Retry Settings (FAST)
    retry_initial_ms: int = 50  # Start at 50ms (was 100ms)
    retry_max_ms: int = 500  # Cap at 500ms (was 2000ms)
    retry_multiplier: float = 1.5  # Slower backoff
    max_retries: int = 2  # Fewer retries, fail fast

    # WebSocket Settings
    ws_ping_interval_ms: int = 5000  # Ping every 5s (per Polymarket docs)
    ws_reconnect_initial_ms: int = 50  # Fast reconnect start
    ws_reconnect_max_ms: int = 5000  # Cap reconnect at 5s
    ws_message_queue_size: int = 1000  # Buffer more messages

    # Latency Thresholds (ms)
    latency_target_detection_ms: int = 50
    latency_target_execution_ms: int = 100
    latency_target_e2e_ms: int = 200
    latency_alert_threshold_ms: int = 500
    latency_critical_threshold_ms: int = 1000

    @classmethod
    def aggressive(cls) -> "PerformanceConfig":
        """Maximum speed configuration."""
        return cls(mode=PerformanceMode.AGGRESSIVE)

    @classmethod
    def balanced(cls) -> "PerformanceConfig":
        """Balanced configuration."""
        return cls(
            mode=PerformanceMode.BALANCED,
            skip_preflight_for_copy=False,
            order_submit_timeout_ms=3000,
            max_retries=3,
        )

    @classmethod
    def conservative(cls) -> "PerformanceConfig":
        """Safe configuration for testing."""
        return cls(
            mode=PerformanceMode.CONSERVATIVE,
            skip_preflight_for_copy=False,
            order_submit_timeout_ms=5000,
            order_status_poll_ms=500,
            max_retries=5,
            retry_initial_ms=200,
        )


class LatencyTracker:
    """High-precision latency tracking with percentile calculations.

    Uses time.perf_counter() for microsecond precision.
    Maintains rolling window for accurate percentiles.
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_callback: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ):
        """Initialize latency tracker.

        Args:
            window_size: Number of samples to keep for percentiles
            alert_callback: Async callback for latency alerts
        """
        self._measurements: dict[str, deque] = {}
        self._window_size = window_size
        self._alert_callback = alert_callback
        self._config = PerformanceConfig()

    def start(self) -> float:
        """Start timing. Returns start timestamp."""
        return time.perf_counter()

    def stop(self, start: float) -> float:
        """Stop timing. Returns elapsed milliseconds."""
        return (time.perf_counter() - start) * 1000

    async def record(self, stage: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            stage: Pipeline stage name
            latency_ms: Latency in milliseconds
        """
        if stage not in self._measurements:
            self._measurements[stage] = deque(maxlen=self._window_size)

        self._measurements[stage].append(latency_ms)

        # Check thresholds and alert
        if latency_ms >= self._config.latency_critical_threshold_ms:
            logger.error(
                "latency_critical",
                stage=stage,
                latency_ms=round(latency_ms, 2),
            )
            if self._alert_callback:
                await self._alert_callback(stage, latency_ms)
        elif latency_ms >= self._config.latency_alert_threshold_ms:
            logger.warning(
                "latency_high",
                stage=stage,
                latency_ms=round(latency_ms, 2),
            )

    def get_stats(self, stage: str) -> dict:
        """Get statistics for a stage.

        Args:
            stage: Pipeline stage name

        Returns:
            Dict with min, max, avg, p50, p95, p99
        """
        if stage not in self._measurements or len(self._measurements[stage]) == 0:
            return {}

        data = list(self._measurements[stage])
        sorted_data = sorted(data)
        n = len(sorted_data)

        return {
            "count": n,
            "min_ms": round(min(data), 3),
            "max_ms": round(max(data), 3),
            "avg_ms": round(mean(data), 3),
            "p50_ms": round(sorted_data[n // 2], 3),
            "p95_ms": round(sorted_data[int(n * 0.95)], 3) if n >= 20 else round(sorted_data[-1], 3),
            "p99_ms": round(sorted_data[int(n * 0.99)], 3) if n >= 100 else round(sorted_data[-1], 3),
            "stddev_ms": round(stdev(data), 3) if n >= 2 else 0,
        }

    def get_all_stats(self) -> dict:
        """Get statistics for all stages."""
        return {stage: self.get_stats(stage) for stage in self._measurements}

    def is_healthy(self) -> tuple[bool, dict]:
        """Check if latencies are within targets.

        Returns:
            Tuple of (is_healthy, details)
        """
        issues = {}

        for stage, data in self._measurements.items():
            if len(data) < 10:
                continue

            avg = mean(data)

            # Check against targets
            if stage == "detection" and avg > self._config.latency_target_detection_ms:
                issues[stage] = f"avg {avg:.1f}ms > target {self._config.latency_target_detection_ms}ms"
            elif stage == "execution" and avg > self._config.latency_target_execution_ms:
                issues[stage] = f"avg {avg:.1f}ms > target {self._config.latency_target_execution_ms}ms"
            elif stage == "e2e" and avg > self._config.latency_target_e2e_ms:
                issues[stage] = f"avg {avg:.1f}ms > target {self._config.latency_target_e2e_ms}ms"

        return len(issues) == 0, issues


class ConnectionWarmer:
    """Pre-warms HTTP and WebSocket connections for minimal first-request latency.

    Cold connections add 100-500ms latency. Warming connections at startup
    ensures the critical copy-trade path has hot connections ready.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._http_warmed = False
        self._ws_warmed = False

    async def warm_http_connections(
        self,
        client,  # httpx.AsyncClient
        endpoints: List[str],
    ) -> dict:
        """Warm HTTP connection pool by making test requests.

        Args:
            client: httpx AsyncClient instance
            endpoints: List of endpoints to warm

        Returns:
            Dict with warm-up results per endpoint
        """
        results = {}

        for endpoint in endpoints:
            start = time.perf_counter()
            try:
                # Make a lightweight request to establish connection
                response = await asyncio.wait_for(
                    client.head(endpoint),
                    timeout=self.config.http_connect_timeout_ms / 1000,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                results[endpoint] = {
                    "status": "warmed",
                    "latency_ms": round(latency_ms, 2),
                    "http_version": str(response.http_version),
                }
                logger.debug(
                    "http_connection_warmed",
                    endpoint=endpoint,
                    latency_ms=round(latency_ms, 2),
                )
            except Exception as e:
                results[endpoint] = {
                    "status": "failed",
                    "error": str(e),
                }
                logger.warning(
                    "http_warmup_failed",
                    endpoint=endpoint,
                    error=str(e),
                )

        self._http_warmed = True
        return results

    async def warm_websocket(
        self,
        ws_client,  # WebSocketClient instance
        test_subscription: Optional[dict] = None,
    ) -> dict:
        """Ensure WebSocket connection is established and subscriptions active.

        Args:
            ws_client: WebSocket client instance
            test_subscription: Optional test subscription to verify

        Returns:
            Dict with connection status
        """
        start = time.perf_counter()

        try:
            # Connect if not already connected
            if not ws_client.is_connected:
                await ws_client.connect()

            latency_ms = (time.perf_counter() - start) * 1000

            # Send test subscription if provided
            if test_subscription:
                await ws_client.send(test_subscription)

            self._ws_warmed = True

            result = {
                "status": "connected",
                "connect_latency_ms": round(latency_ms, 2),
            }

            logger.info(
                "websocket_warmed",
                latency_ms=round(latency_ms, 2),
            )

            return result

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }

    @property
    def is_warmed(self) -> bool:
        """Check if all connections are warmed."""
        return self._http_warmed and self._ws_warmed


@dataclass
class LatencyBudget:
    """Tracks remaining latency budget for a trade execution.

    Helps enforce end-to-end latency targets by tracking time
    spent at each stage and calculating remaining budget.
    """

    total_budget_ms: float
    start_time: float = field(default_factory=time.perf_counter)
    stages: dict = field(default_factory=dict)

    def stage_start(self, stage: str) -> float:
        """Mark stage start. Returns start timestamp."""
        return time.perf_counter()

    def stage_end(self, stage: str, start: float) -> float:
        """Mark stage end. Returns stage duration in ms."""
        duration = (time.perf_counter() - start) * 1000
        self.stages[stage] = duration
        return duration

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time in ms."""
        return (time.perf_counter() - self.start_time) * 1000

    @property
    def remaining_ms(self) -> float:
        """Remaining budget in ms."""
        return max(0, self.total_budget_ms - self.elapsed_ms)

    @property
    def is_over_budget(self) -> bool:
        """Check if budget is exceeded."""
        return self.elapsed_ms > self.total_budget_ms

    def get_timeout_for_stage(self, stage: str, default_ms: float) -> float:
        """Calculate timeout for a stage based on remaining budget.

        Args:
            stage: Stage name
            default_ms: Default timeout if budget allows

        Returns:
            Timeout in milliseconds
        """
        remaining = self.remaining_ms

        # If over budget, use minimum viable timeout
        if remaining <= 0:
            return min(default_ms, 500)  # At least 500ms

        # Use remaining budget or default, whichever is smaller
        return min(remaining, default_ms)

    def summary(self) -> dict:
        """Get summary of latency budget usage."""
        return {
            "total_budget_ms": self.total_budget_ms,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "remaining_ms": round(self.remaining_ms, 2),
            "is_over_budget": self.is_over_budget,
            "stages": {k: round(v, 2) for k, v in self.stages.items()},
        }


# Global latency tracker instance
_global_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    """Get or create global latency tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker


def create_latency_budget(target_ms: Optional[float] = None) -> LatencyBudget:
    """Create a latency budget for a trade execution.

    Args:
        target_ms: Target total latency in ms (default: from config)

    Returns:
        LatencyBudget instance
    """
    config = PerformanceConfig()
    budget_ms = target_ms or config.latency_target_e2e_ms
    return LatencyBudget(total_budget_ms=budget_ms)
