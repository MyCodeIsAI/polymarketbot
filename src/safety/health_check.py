"""Health check system for trading bot monitoring.

This module provides:
- API connectivity checks
- WebSocket connection status
- Rate limit headroom monitoring
- System resource monitoring (memory/CPU)
- Order queue depth tracking
- Aggregated health status
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Awaitable, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    last_check: datetime = field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Aggregated system health status."""

    overall_status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    components: list[ComponentHealth] = field(default_factory=list)
    uptime_seconds: float = 0
    checks_performed: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == HealthStatus.HEALTHY

    @property
    def unhealthy_components(self) -> list[ComponentHealth]:
        """Get list of unhealthy components."""
        return [c for c in self.components if c.status == HealthStatus.UNHEALTHY]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "checks_performed": self.checks_performed,
            "components": [c.to_dict() for c in self.components],
            "unhealthy_count": len(self.unhealthy_components),
        }


# Type for health check functions
HealthCheckFunc = Callable[[], Awaitable[ComponentHealth]]


class APIHealthChecker:
    """Checks API connectivity and response times.

    Pings the API endpoint periodically to verify connectivity
    and measure latency.
    """

    def __init__(
        self,
        ping_func: Callable[[], Awaitable[bool]],
        name: str = "api",
        timeout_ms: float = 5000,
        degraded_threshold_ms: float = 1000,
    ):
        """Initialize API health checker.

        Args:
            ping_func: Async function that returns True if API is reachable
            name: Component name
            timeout_ms: Timeout for ping in milliseconds
            degraded_threshold_ms: Latency threshold for degraded status
        """
        self.ping_func = ping_func
        self.name = name
        self.timeout_ms = timeout_ms
        self.degraded_threshold_ms = degraded_threshold_ms

        self._last_success: Optional[datetime] = None
        self._consecutive_failures = 0

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth with current status
        """
        start_time = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self.ping_func(),
                timeout=self.timeout_ms / 1000,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if result:
                self._last_success = datetime.utcnow()
                self._consecutive_failures = 0

                if latency_ms > self.degraded_threshold_ms:
                    status = HealthStatus.DEGRADED
                    message = f"High latency: {latency_ms:.0f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = "OK"
            else:
                self._consecutive_failures += 1
                status = HealthStatus.UNHEALTHY
                message = "Ping returned false"

        except asyncio.TimeoutError:
            latency_ms = self.timeout_ms
            self._consecutive_failures += 1
            status = HealthStatus.UNHEALTHY
            message = f"Timeout after {self.timeout_ms}ms"

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._consecutive_failures += 1
            status = HealthStatus.UNHEALTHY
            message = f"Error: {str(e)}"

        return ComponentHealth(
            name=self.name,
            status=status,
            latency_ms=latency_ms,
            message=message,
            details={
                "consecutive_failures": self._consecutive_failures,
                "last_success": self._last_success.isoformat() if self._last_success else None,
            },
        )


class WebSocketHealthChecker:
    """Checks WebSocket connection health.

    Monitors connection state, message flow, and latency.
    """

    def __init__(
        self,
        is_connected_func: Callable[[], bool],
        get_latency_func: Optional[Callable[[], float]] = None,
        name: str = "websocket",
        max_silence_s: float = 30,
    ):
        """Initialize WebSocket health checker.

        Args:
            is_connected_func: Function returning True if connected
            get_latency_func: Function returning current latency in ms
            name: Component name
            max_silence_s: Max seconds without messages before unhealthy
        """
        self.is_connected_func = is_connected_func
        self.get_latency_func = get_latency_func
        self.name = name
        self.max_silence_s = max_silence_s

        self._last_message_time: Optional[datetime] = None

    def record_message(self) -> None:
        """Record that a message was received."""
        self._last_message_time = datetime.utcnow()

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth with current status
        """
        is_connected = self.is_connected_func()
        latency_ms = self.get_latency_func() if self.get_latency_func else None

        # Check silence duration
        silence_s = None
        if self._last_message_time:
            silence_s = (datetime.utcnow() - self._last_message_time).total_seconds()

        # Determine status
        if not is_connected:
            status = HealthStatus.UNHEALTHY
            message = "WebSocket disconnected"
        elif silence_s and silence_s > self.max_silence_s:
            status = HealthStatus.DEGRADED
            message = f"No messages for {silence_s:.0f}s"
        elif latency_ms and latency_ms > 500:
            status = HealthStatus.DEGRADED
            message = f"High latency: {latency_ms:.0f}ms"
        else:
            status = HealthStatus.HEALTHY
            message = "Connected"

        return ComponentHealth(
            name=self.name,
            status=status,
            latency_ms=latency_ms,
            message=message,
            details={
                "connected": is_connected,
                "silence_seconds": silence_s,
                "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
            },
        )


class RateLimitHealthChecker:
    """Monitors rate limit headroom.

    Tracks API rate limit usage and warns when approaching limits.
    """

    def __init__(
        self,
        get_usage_func: Callable[[], tuple[int, int]],
        name: str = "rate_limit",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        """Initialize rate limit health checker.

        Args:
            get_usage_func: Function returning (used, limit) tuple
            name: Component name
            warning_threshold: Usage ratio for degraded status
            critical_threshold: Usage ratio for unhealthy status
        """
        self.get_usage_func = get_usage_func
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth with current status
        """
        try:
            used, limit = self.get_usage_func()

            if limit == 0:
                return ComponentHealth(
                    name=self.name,
                    status=HealthStatus.UNKNOWN,
                    message="Rate limit not configured",
                )

            usage_ratio = used / limit
            headroom = limit - used

            if usage_ratio >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: {used}/{limit} ({usage_ratio*100:.0f}%)"
            elif usage_ratio >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Warning: {used}/{limit} ({usage_ratio*100:.0f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"OK: {headroom} remaining"

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                details={
                    "used": used,
                    "limit": limit,
                    "usage_percent": round(usage_ratio * 100, 1),
                    "headroom": headroom,
                },
            )

        except Exception as e:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Error: {str(e)}",
            )


class SystemResourceChecker:
    """Monitors system resources (memory, CPU).

    Uses psutil if available, falls back to /proc on Linux.
    """

    def __init__(
        self,
        name: str = "system",
        memory_warning_percent: float = 80,
        memory_critical_percent: float = 95,
        cpu_warning_percent: float = 80,
    ):
        """Initialize system resource checker.

        Args:
            name: Component name
            memory_warning_percent: Memory usage for degraded status
            memory_critical_percent: Memory usage for unhealthy status
            cpu_warning_percent: CPU usage for degraded status
        """
        self.name = name
        self.memory_warning_percent = memory_warning_percent
        self.memory_critical_percent = memory_critical_percent
        self.cpu_warning_percent = cpu_warning_percent

        self._has_psutil = False
        try:
            import psutil
            self._has_psutil = True
        except ImportError:
            pass

    def _get_memory_usage(self) -> Optional[float]:
        """Get memory usage percentage."""
        if self._has_psutil:
            import psutil
            return psutil.virtual_memory().percent

        # Fallback to /proc/meminfo on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                mem_total = None
                mem_available = None
                for line in lines:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])
                if mem_total and mem_available:
                    return ((mem_total - mem_available) / mem_total) * 100
        except Exception:
            pass

        return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Get CPU usage percentage."""
        if self._has_psutil:
            import psutil
            return psutil.cpu_percent(interval=0.1)

        # Fallback to /proc/loadavg on Linux
        try:
            with open("/proc/loadavg", "r") as f:
                load_1min = float(f.read().split()[0])
                cpu_count = os.cpu_count() or 1
                return min(100, (load_1min / cpu_count) * 100)
        except Exception:
            pass

        return None

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth with current status
        """
        memory_percent = self._get_memory_usage()
        cpu_percent = self._get_cpu_usage()

        status = HealthStatus.HEALTHY
        issues = []

        if memory_percent is not None:
            if memory_percent >= self.memory_critical_percent:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Memory critical: {memory_percent:.0f}%")
            elif memory_percent >= self.memory_warning_percent:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Memory high: {memory_percent:.0f}%")

        if cpu_percent is not None:
            if cpu_percent >= self.cpu_warning_percent:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"CPU high: {cpu_percent:.0f}%")

        message = "; ".join(issues) if issues else "OK"

        return ComponentHealth(
            name=self.name,
            status=status,
            message=message,
            details={
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
            },
        )


class QueueDepthChecker:
    """Monitors order queue depth.

    Warns when queue is backing up, indicating processing issues.
    """

    def __init__(
        self,
        get_queue_size_func: Callable[[], int],
        name: str = "order_queue",
        warning_threshold: int = 10,
        critical_threshold: int = 50,
    ):
        """Initialize queue depth checker.

        Args:
            get_queue_size_func: Function returning current queue size
            name: Component name
            warning_threshold: Queue size for degraded status
            critical_threshold: Queue size for unhealthy status
        """
        self.get_queue_size_func = get_queue_size_func
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self._max_observed = 0

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth with current status
        """
        try:
            queue_size = self.get_queue_size_func()
            self._max_observed = max(self._max_observed, queue_size)

            if queue_size >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Queue backing up: {queue_size} orders"
            elif queue_size >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Queue elevated: {queue_size} orders"
            else:
                status = HealthStatus.HEALTHY
                message = f"OK: {queue_size} orders"

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                details={
                    "current_size": queue_size,
                    "max_observed": self._max_observed,
                },
            )

        except Exception as e:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Error: {str(e)}",
            )


# Callback for health status changes
HealthChangeCallback = Callable[[SystemHealth], Awaitable[None]]


class HealthCheckManager:
    """Coordinates all health checks.

    Runs periodic health checks and aggregates results.
    Notifies on status changes.

    Example:
        manager = HealthCheckManager(check_interval_s=10)

        manager.add_checker(APIHealthChecker(ping_func))
        manager.add_checker(WebSocketHealthChecker(is_connected_func))

        await manager.start()

        # Get current health
        health = await manager.get_health()
        if not health.is_healthy:
            # Take action
            pass
    """

    def __init__(
        self,
        check_interval_s: float = 10,
        on_status_change: Optional[HealthChangeCallback] = None,
    ):
        """Initialize health check manager.

        Args:
            check_interval_s: Interval between health checks
            on_status_change: Callback when overall status changes
        """
        self.check_interval_s = check_interval_s
        self._on_status_change = on_status_change

        self._checkers: list[Any] = []  # Objects with check() method
        self._custom_checks: list[HealthCheckFunc] = []

        self._start_time: Optional[datetime] = None
        self._last_health: Optional[SystemHealth] = None
        self._checks_performed = 0

        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_checker(self, checker: Any) -> None:
        """Add a health checker.

        Args:
            checker: Object with async check() method returning ComponentHealth
        """
        self._checkers.append(checker)

    def add_custom_check(self, check_func: HealthCheckFunc) -> None:
        """Add a custom health check function.

        Args:
            check_func: Async function returning ComponentHealth
        """
        self._custom_checks.append(check_func)

    async def start(self) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._task = asyncio.create_task(self._check_loop())

        logger.info(
            "health_check_manager_started",
            check_interval_s=self.check_interval_s,
            checkers=len(self._checkers),
        )

    async def stop(self) -> None:
        """Stop periodic health checks."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            "health_check_manager_stopped",
            checks_performed=self._checks_performed,
        )

    async def get_health(self) -> SystemHealth:
        """Get current system health.

        Performs a health check and returns aggregated results.

        Returns:
            SystemHealth with all component statuses
        """
        components = []

        # Run all checker checks
        for checker in self._checkers:
            try:
                health = await checker.check()
                components.append(health)
            except Exception as e:
                components.append(ComponentHealth(
                    name=getattr(checker, "name", "unknown"),
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                ))

        # Run custom checks
        for check_func in self._custom_checks:
            try:
                health = await check_func()
                components.append(health)
            except Exception as e:
                components.append(ComponentHealth(
                    name="custom",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                ))

        # Determine overall status
        overall_status = self._aggregate_status(components)

        # Calculate uptime
        uptime_s = 0.0
        if self._start_time:
            uptime_s = (datetime.utcnow() - self._start_time).total_seconds()

        return SystemHealth(
            overall_status=overall_status,
            components=components,
            uptime_seconds=uptime_s,
            checks_performed=self._checks_performed,
        )

    def _aggregate_status(self, components: list[ComponentHealth]) -> HealthStatus:
        """Aggregate component statuses into overall status.

        Args:
            components: List of component health statuses

        Returns:
            Overall health status
        """
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                health = await self.get_health()
                self._checks_performed += 1

                # Check for status change
                if self._last_health is not None:
                    if health.overall_status != self._last_health.overall_status:
                        logger.warning(
                            "health_status_changed",
                            previous=self._last_health.overall_status.value,
                            current=health.overall_status.value,
                        )

                        if self._on_status_change:
                            try:
                                await self._on_status_change(health)
                            except Exception as e:
                                logger.error("health_callback_error", error=str(e))

                self._last_health = health

                # Log periodic health summary
                if self._checks_performed % 6 == 0:  # Every minute at 10s interval
                    logger.info(
                        "health_check_summary",
                        status=health.overall_status.value,
                        unhealthy_count=len(health.unhealthy_components),
                    )

                await asyncio.sleep(self.check_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))
                await asyncio.sleep(self.check_interval_s)

    @property
    def last_health(self) -> Optional[SystemHealth]:
        """Get last health check result."""
        return self._last_health

    @property
    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        if self._last_health is None:
            return False
        return self._last_health.is_healthy
