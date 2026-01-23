"""Real-time latency monitoring for copy trading pipeline.

Provides:
- Live latency tracking at each pipeline stage
- Percentile calculations (p50, p95, p99)
- Threshold alerts
- Performance degradation detection
- Export to dashboard/API

CRITICAL: Latency monitoring is essential for live trading.
This module should have minimal overhead itself.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from statistics import mean, median, stdev
from typing import Optional, Callable, Awaitable, Dict, List

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LatencyStage(str, Enum):
    """Pipeline stages for latency tracking."""

    # WebSocket stages
    WS_RECEIVE = "ws_receive"  # Message received from WebSocket
    WS_PARSE = "ws_parse"  # Message parsed

    # Detection stages
    DETECTION = "detection"  # Trade detected
    VALIDATION = "validation"  # Trade validated

    # Execution stages
    SIZING = "sizing"  # Position size calculated
    ORDER_BUILD = "order_build"  # Order constructed
    SIGNING = "signing"  # Order signed
    SUBMISSION = "submission"  # Order submitted to CLOB
    CONFIRMATION = "confirmation"  # Fill confirmation received

    # Aggregate
    E2E = "e2e"  # End-to-end total


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LatencyAlert:
    """Latency threshold breach alert."""

    stage: LatencyStage
    level: AlertLevel
    latency_ms: float
    threshold_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "level": self.level.value,
            "latency_ms": round(self.latency_ms, 2),
            "threshold_ms": self.threshold_ms,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage: LatencyStage
    window_size: int = 1000

    # Rolling window of measurements
    _measurements: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Counters
    count: int = 0
    total_ms: float = 0

    # Extremes (all-time)
    min_ms: float = float("inf")
    max_ms: float = 0

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._measurements.append(latency_ms)
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)

    @property
    def avg_ms(self) -> float:
        """Average latency."""
        return self.total_ms / self.count if self.count > 0 else 0

    @property
    def recent_avg_ms(self) -> float:
        """Average of recent measurements."""
        if not self._measurements:
            return 0
        return mean(self._measurements)

    @property
    def p50_ms(self) -> float:
        """50th percentile (median)."""
        if not self._measurements:
            return 0
        return median(self._measurements)

    @property
    def p95_ms(self) -> float:
        """95th percentile."""
        if len(self._measurements) < 20:
            return self.max_ms
        sorted_m = sorted(self._measurements)
        idx = int(len(sorted_m) * 0.95)
        return sorted_m[idx]

    @property
    def p99_ms(self) -> float:
        """99th percentile."""
        if len(self._measurements) < 100:
            return self.max_ms
        sorted_m = sorted(self._measurements)
        idx = int(len(sorted_m) * 0.99)
        return sorted_m[idx]

    @property
    def stddev_ms(self) -> float:
        """Standard deviation."""
        if len(self._measurements) < 2:
            return 0
        return stdev(self._measurements)

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "stage": self.stage.value,
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "recent_avg_ms": round(self.recent_avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
        }


@dataclass
class LatencyThresholds:
    """Configurable thresholds for latency alerts."""

    # Per-stage warning thresholds (ms)
    warning: Dict[LatencyStage, float] = field(default_factory=lambda: {
        LatencyStage.WS_RECEIVE: 50,
        LatencyStage.WS_PARSE: 5,
        LatencyStage.DETECTION: 100,
        LatencyStage.VALIDATION: 10,
        LatencyStage.SIZING: 5,
        LatencyStage.ORDER_BUILD: 20,
        LatencyStage.SIGNING: 30,
        LatencyStage.SUBMISSION: 150,
        LatencyStage.CONFIRMATION: 200,
        LatencyStage.E2E: 300,
    })

    # Per-stage critical thresholds (ms)
    critical: Dict[LatencyStage, float] = field(default_factory=lambda: {
        LatencyStage.WS_RECEIVE: 200,
        LatencyStage.WS_PARSE: 20,
        LatencyStage.DETECTION: 300,
        LatencyStage.VALIDATION: 50,
        LatencyStage.SIZING: 20,
        LatencyStage.ORDER_BUILD: 50,
        LatencyStage.SIGNING: 100,
        LatencyStage.SUBMISSION: 500,
        LatencyStage.CONFIRMATION: 1000,
        LatencyStage.E2E: 1000,
    })


# Type for alert callbacks
AlertCallback = Callable[[LatencyAlert], Awaitable[None]]


class LatencyMonitor:
    """Real-time latency monitoring system.

    Features:
    - Track latency at each pipeline stage
    - Calculate rolling percentiles
    - Fire alerts on threshold breaches
    - Detect performance degradation
    - Export metrics for dashboard

    Example:
        monitor = LatencyMonitor()

        # Record latencies
        t_start = monitor.start()
        # ... do work ...
        await monitor.record(LatencyStage.DETECTION, t_start)

        # Get metrics
        metrics = monitor.get_metrics()
    """

    def __init__(
        self,
        thresholds: Optional[LatencyThresholds] = None,
        window_size: int = 1000,
        alert_callback: Optional[AlertCallback] = None,
    ):
        """Initialize latency monitor.

        Args:
            thresholds: Alert thresholds
            window_size: Rolling window size for percentiles
            alert_callback: Async callback for alerts
        """
        self.thresholds = thresholds or LatencyThresholds()
        self.window_size = window_size
        self._alert_callback = alert_callback

        # Stage metrics
        self._stages: Dict[LatencyStage, StageMetrics] = {
            stage: StageMetrics(stage=stage, window_size=window_size)
            for stage in LatencyStage
        }

        # Recent alerts (rolling window)
        self._alerts: deque[LatencyAlert] = deque(maxlen=100)

        # Trade-level tracking
        self._active_trades: Dict[str, float] = {}  # trade_id -> start_time

        # Health status
        self._degraded_stages: set = set()

    def start(self) -> float:
        """Start timing. Returns high-precision timestamp."""
        return time.perf_counter()

    def elapsed_ms(self, start: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.perf_counter() - start) * 1000

    async def record(
        self,
        stage: LatencyStage,
        start: float,
        trade_id: Optional[str] = None,
    ) -> float:
        """Record latency for a stage.

        Args:
            stage: Pipeline stage
            start: Start timestamp from start()
            trade_id: Optional trade ID for correlation

        Returns:
            Latency in milliseconds
        """
        latency_ms = self.elapsed_ms(start)

        # Record to stage metrics
        self._stages[stage].record(latency_ms)

        # Check thresholds and alert
        await self._check_thresholds(stage, latency_ms)

        return latency_ms

    async def record_value(
        self,
        stage: LatencyStage,
        latency_ms: float,
    ) -> None:
        """Record a pre-calculated latency value.

        Args:
            stage: Pipeline stage
            latency_ms: Latency in milliseconds
        """
        self._stages[stage].record(latency_ms)
        await self._check_thresholds(stage, latency_ms)

    def start_trade(self, trade_id: str) -> float:
        """Start tracking a trade end-to-end.

        Args:
            trade_id: Unique trade identifier

        Returns:
            Start timestamp
        """
        start = time.perf_counter()
        self._active_trades[trade_id] = start
        return start

    async def end_trade(self, trade_id: str) -> Optional[float]:
        """End trade tracking and record E2E latency.

        Args:
            trade_id: Trade identifier

        Returns:
            E2E latency in ms, or None if trade not found
        """
        start = self._active_trades.pop(trade_id, None)
        if start is None:
            return None

        latency_ms = self.elapsed_ms(start)
        self._stages[LatencyStage.E2E].record(latency_ms)
        await self._check_thresholds(LatencyStage.E2E, latency_ms)
        return latency_ms

    async def _check_thresholds(
        self,
        stage: LatencyStage,
        latency_ms: float,
    ) -> None:
        """Check latency against thresholds and fire alerts."""
        critical_threshold = self.thresholds.critical.get(stage, float("inf"))
        warning_threshold = self.thresholds.warning.get(stage, float("inf"))

        alert = None

        if latency_ms >= critical_threshold:
            alert = LatencyAlert(
                stage=stage,
                level=AlertLevel.CRITICAL,
                latency_ms=latency_ms,
                threshold_ms=critical_threshold,
                message=f"CRITICAL: {stage.value} latency {latency_ms:.1f}ms >= {critical_threshold}ms",
            )
            self._degraded_stages.add(stage)
            logger.error(
                "latency_critical",
                stage=stage.value,
                latency_ms=round(latency_ms, 2),
                threshold_ms=critical_threshold,
            )

        elif latency_ms >= warning_threshold:
            alert = LatencyAlert(
                stage=stage,
                level=AlertLevel.WARNING,
                latency_ms=latency_ms,
                threshold_ms=warning_threshold,
                message=f"WARNING: {stage.value} latency {latency_ms:.1f}ms >= {warning_threshold}ms",
            )
            logger.warning(
                "latency_warning",
                stage=stage.value,
                latency_ms=round(latency_ms, 2),
                threshold_ms=warning_threshold,
            )

        else:
            # Performance recovered
            self._degraded_stages.discard(stage)

        if alert:
            self._alerts.append(alert)
            if self._alert_callback:
                try:
                    await self._alert_callback(alert)
                except Exception as e:
                    logger.error("alert_callback_error", error=str(e))

    def get_metrics(self) -> Dict[str, dict]:
        """Get all stage metrics."""
        return {stage.value: self._stages[stage].to_dict() for stage in LatencyStage}

    def get_stage_metrics(self, stage: LatencyStage) -> dict:
        """Get metrics for a specific stage."""
        return self._stages[stage].to_dict()

    def get_recent_alerts(self, limit: int = 20) -> List[dict]:
        """Get recent alerts."""
        return [alert.to_dict() for alert in list(self._alerts)[-limit:]]

    def get_health_status(self) -> dict:
        """Get overall health status."""
        # Calculate health score (0-100)
        e2e_metrics = self._stages[LatencyStage.E2E]

        if e2e_metrics.count < 10:
            health_score = 100  # Not enough data
        else:
            target = self.thresholds.warning[LatencyStage.E2E]
            avg = e2e_metrics.recent_avg_ms

            if avg <= target * 0.5:
                health_score = 100
            elif avg <= target:
                health_score = int(100 - (avg / target) * 30)
            elif avg <= target * 2:
                health_score = int(70 - ((avg - target) / target) * 40)
            else:
                health_score = max(0, int(30 - ((avg - target * 2) / target) * 30))

        return {
            "status": "degraded" if self._degraded_stages else "healthy",
            "health_score": health_score,
            "degraded_stages": [s.value for s in self._degraded_stages],
            "e2e_avg_ms": round(e2e_metrics.recent_avg_ms, 2),
            "e2e_p95_ms": round(e2e_metrics.p95_ms, 2),
            "total_trades": e2e_metrics.count,
            "recent_alerts": len([a for a in self._alerts if a.level == AlertLevel.CRITICAL]),
        }

    def get_dashboard_data(self) -> dict:
        """Get data formatted for dashboard display."""
        return {
            "health": self.get_health_status(),
            "stages": self.get_metrics(),
            "alerts": self.get_recent_alerts(10),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        for stage in self._stages.values():
            stage._measurements.clear()
            stage.count = 0
            stage.total_ms = 0
            stage.min_ms = float("inf")
            stage.max_ms = 0

        self._alerts.clear()
        self._active_trades.clear()
        self._degraded_stages.clear()


# Global monitor instance
_global_monitor: Optional[LatencyMonitor] = None


def get_latency_monitor() -> LatencyMonitor:
    """Get or create global latency monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = LatencyMonitor()
    return _global_monitor
