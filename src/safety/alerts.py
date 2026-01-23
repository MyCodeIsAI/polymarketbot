"""Alert system for trading bot notifications.

This module provides:
- Structured alert types with severity levels
- Multiple notification channels (webhook, logging, file)
- Alert throttling to prevent spam
- Alert history and aggregation
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Alert categories for routing and filtering."""

    CIRCUIT_BREAKER = "circuit_breaker"
    HEALTH = "health"
    TRADE = "trade"
    POSITION = "position"
    BALANCE = "balance"
    SYSTEM = "system"
    API = "api"
    WEBSOCKET = "websocket"


@dataclass
class Alert:
    """Individual alert message."""

    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_id: str = ""
    details: dict = field(default_factory=dict)

    # Metadata
    source: Optional[str] = None
    target_name: Optional[str] = None
    market_id: Optional[str] = None

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.category.value}_{self.timestamp.timestamp():.0f}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "source": self.source,
            "target_name": self.target_name,
            "market_id": self.market_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @property
    def is_critical(self) -> bool:
        """Check if alert is critical severity."""
        return self.severity == AlertSeverity.CRITICAL

    @property
    def is_error_or_above(self) -> bool:
        """Check if alert is error or critical."""
        return self.severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL)


class AlertChannel(ABC):
    """Base class for alert notification channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert through this channel.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def should_send(self, alert: Alert) -> bool:
        """Check if this channel should send the alert.

        Args:
            alert: Alert to check

        Returns:
            True if should send
        """
        pass


class LoggingChannel(AlertChannel):
    """Sends alerts to structured logger."""

    def __init__(
        self,
        min_severity: AlertSeverity = AlertSeverity.INFO,
        categories: Optional[list[AlertCategory]] = None,
    ):
        """Initialize logging channel.

        Args:
            min_severity: Minimum severity to log
            categories: Categories to log (None = all)
        """
        self.min_severity = min_severity
        self.categories = set(categories) if categories else None

        self._severity_order = {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }

    def should_send(self, alert: Alert) -> bool:
        """Check if should log this alert."""
        severity_ok = (
            self._severity_order[alert.severity] >= self._severity_order[self.min_severity]
        )
        category_ok = self.categories is None or alert.category in self.categories
        return severity_ok and category_ok

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        if not self.should_send(alert):
            return False

        log_func = {
            AlertSeverity.DEBUG: logger.debug,
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)

        log_func(
            f"ALERT: {alert.title}",
            alert_id=alert.alert_id,
            category=alert.category.value,
            message=alert.message,
            **alert.details,
        )

        return True


class WebhookChannel(AlertChannel):
    """Sends alerts to a webhook URL (Discord, Slack, etc.)."""

    def __init__(
        self,
        webhook_url: str,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        categories: Optional[list[AlertCategory]] = None,
        timeout_s: float = 10,
        include_details: bool = True,
    ):
        """Initialize webhook channel.

        Args:
            webhook_url: Webhook URL to POST to
            min_severity: Minimum severity to send
            categories: Categories to send (None = all)
            timeout_s: Request timeout
            include_details: Whether to include alert details
        """
        self.webhook_url = webhook_url
        self.min_severity = min_severity
        self.categories = set(categories) if categories else None
        self.timeout_s = timeout_s
        self.include_details = include_details

        self._severity_order = {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }

        self._session = None

    def should_send(self, alert: Alert) -> bool:
        """Check if should send this alert."""
        severity_ok = (
            self._severity_order[alert.severity] >= self._severity_order[self.min_severity]
        )
        category_ok = self.categories is None or alert.category in self.categories
        return severity_ok and category_ok

    def _format_for_discord(self, alert: Alert) -> dict:
        """Format alert as Discord webhook payload."""
        color_map = {
            AlertSeverity.DEBUG: 0x808080,  # Gray
            AlertSeverity.INFO: 0x3498db,  # Blue
            AlertSeverity.WARNING: 0xf39c12,  # Orange
            AlertSeverity.ERROR: 0xe74c3c,  # Red
            AlertSeverity.CRITICAL: 0x9b59b6,  # Purple
        }

        embed = {
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "description": alert.message,
            "color": color_map.get(alert.severity, 0x3498db),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {"name": "Category", "value": alert.category.value, "inline": True},
            ],
        }

        if alert.target_name:
            embed["fields"].append(
                {"name": "Target", "value": alert.target_name, "inline": True}
            )

        if self.include_details and alert.details:
            # Add key details as fields
            for key, value in list(alert.details.items())[:5]:  # Limit fields
                embed["fields"].append(
                    {"name": key, "value": str(value)[:100], "inline": True}
                )

        return {"embeds": [embed]}

    def _format_for_slack(self, alert: Alert) -> dict:
        """Format alert as Slack webhook payload."""
        emoji_map = {
            AlertSeverity.DEBUG: ":grey_question:",
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji_map.get(alert.severity, '')} {alert.title}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.message},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"*Category:* {alert.category.value}"},
                    {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                ],
            },
        ]

        return {"blocks": blocks}

    async def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        if not self.should_send(alert):
            return False

        try:
            import aiohttp

            # Detect webhook type and format appropriately
            if "discord" in self.webhook_url.lower():
                payload = self._format_for_discord(alert)
            elif "slack" in self.webhook_url.lower():
                payload = self._format_for_slack(alert)
            else:
                # Generic JSON payload
                payload = alert.to_dict()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as response:
                    if response.status >= 400:
                        logger.error(
                            "webhook_send_failed",
                            status=response.status,
                            alert_id=alert.alert_id,
                        )
                        return False
                    return True

        except ImportError:
            logger.error("aiohttp_not_installed", alert_id=alert.alert_id)
            return False
        except Exception as e:
            logger.error(
                "webhook_send_error",
                error=str(e),
                alert_id=alert.alert_id,
            )
            return False


class FileChannel(AlertChannel):
    """Writes alerts to a file (JSON lines format)."""

    def __init__(
        self,
        file_path: str,
        min_severity: AlertSeverity = AlertSeverity.INFO,
        categories: Optional[list[AlertCategory]] = None,
    ):
        """Initialize file channel.

        Args:
            file_path: Path to alert file
            min_severity: Minimum severity to write
            categories: Categories to write (None = all)
        """
        self.file_path = file_path
        self.min_severity = min_severity
        self.categories = set(categories) if categories else None

        self._severity_order = {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }

        self._lock = asyncio.Lock()

    def should_send(self, alert: Alert) -> bool:
        """Check if should write this alert."""
        severity_ok = (
            self._severity_order[alert.severity] >= self._severity_order[self.min_severity]
        )
        category_ok = self.categories is None or alert.category in self.categories
        return severity_ok and category_ok

    async def send(self, alert: Alert) -> bool:
        """Write alert to file."""
        if not self.should_send(alert):
            return False

        try:
            async with self._lock:
                with open(self.file_path, "a") as f:
                    f.write(alert.to_json() + "\n")
            return True
        except Exception as e:
            logger.error(
                "file_write_error",
                error=str(e),
                alert_id=alert.alert_id,
            )
            return False


class AlertThrottler:
    """Throttles alerts to prevent spam.

    Groups similar alerts and limits frequency.
    """

    def __init__(
        self,
        window_s: int = 60,
        max_per_window: int = 5,
    ):
        """Initialize throttler.

        Args:
            window_s: Time window in seconds
            max_per_window: Max alerts per category per window
        """
        self.window_s = window_s
        self.max_per_window = max_per_window

        # Track alerts per category: {category: [timestamps]}
        self._alert_times: dict[str, list[datetime]] = {}

    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent (not throttled).

        Args:
            alert: Alert to check

        Returns:
            True if should send, False if throttled
        """
        key = f"{alert.category.value}_{alert.severity.value}"
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_s)

        # Get recent alerts for this key
        if key not in self._alert_times:
            self._alert_times[key] = []

        # Remove old entries
        self._alert_times[key] = [
            t for t in self._alert_times[key] if t > cutoff
        ]

        # Check if under limit
        if len(self._alert_times[key]) >= self.max_per_window:
            return False

        # Record this alert
        self._alert_times[key].append(now)
        return True

    def get_throttle_status(self) -> dict:
        """Get current throttle status.

        Returns:
            Dict with throttle stats per category
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_s)

        status = {}
        for key, times in self._alert_times.items():
            recent = [t for t in times if t > cutoff]
            status[key] = {
                "count": len(recent),
                "limit": self.max_per_window,
                "throttled": len(recent) >= self.max_per_window,
            }

        return status


class AlertManager:
    """Central alert management system.

    Coordinates alert sending across multiple channels with throttling.

    Example:
        manager = AlertManager()

        # Add channels
        manager.add_channel(LoggingChannel())
        manager.add_channel(WebhookChannel(webhook_url))

        # Send alert
        await manager.alert(
            AlertCategory.CIRCUIT_BREAKER,
            AlertSeverity.CRITICAL,
            "Circuit Breaker Tripped",
            "Max daily loss exceeded: $500",
        )

        # Use convenience methods
        await manager.critical("system", "Database connection lost")
    """

    def __init__(
        self,
        throttle_window_s: int = 60,
        throttle_max: int = 5,
        enable_throttling: bool = True,
    ):
        """Initialize alert manager.

        Args:
            throttle_window_s: Throttle window in seconds
            throttle_max: Max alerts per category per window
            enable_throttling: Whether to enable throttling
        """
        self._channels: list[AlertChannel] = []
        self._throttler = AlertThrottler(throttle_window_s, throttle_max)
        self._enable_throttling = enable_throttling

        # Alert history
        self._history: list[Alert] = []
        self._max_history = 1000

        # Stats
        self._alerts_sent = 0
        self._alerts_throttled = 0

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a notification channel.

        Args:
            channel: Channel to add
        """
        self._channels.append(channel)

    async def alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[dict] = None,
        source: Optional[str] = None,
        target_name: Optional[str] = None,
        market_id: Optional[str] = None,
        bypass_throttle: bool = False,
    ) -> Optional[Alert]:
        """Send an alert through all configured channels.

        Args:
            category: Alert category
            severity: Alert severity
            title: Alert title
            message: Alert message
            details: Additional details
            source: Alert source
            target_name: Target account name
            market_id: Related market ID
            bypass_throttle: Skip throttle check

        Returns:
            Alert if sent, None if throttled
        """
        alert = Alert(
            category=category,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            source=source,
            target_name=target_name,
            market_id=market_id,
        )

        # Check throttling
        if self._enable_throttling and not bypass_throttle:
            if not self._throttler.should_send(alert):
                self._alerts_throttled += 1
                logger.debug(
                    "alert_throttled",
                    category=category.value,
                    severity=severity.value,
                )
                return None

        # Send through all channels
        for channel in self._channels:
            try:
                await channel.send(alert)
            except Exception as e:
                logger.error(
                    "channel_send_error",
                    channel=type(channel).__name__,
                    error=str(e),
                )

        # Record in history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        self._alerts_sent += 1

        return alert

    # Convenience methods

    async def debug(
        self,
        category: str,
        message: str,
        **kwargs,
    ) -> Optional[Alert]:
        """Send debug alert."""
        cat = AlertCategory(category) if isinstance(category, str) else category
        return await self.alert(cat, AlertSeverity.DEBUG, "Debug", message, **kwargs)

    async def info(
        self,
        category: str,
        message: str,
        title: str = "Info",
        **kwargs,
    ) -> Optional[Alert]:
        """Send info alert."""
        cat = AlertCategory(category) if isinstance(category, str) else category
        return await self.alert(cat, AlertSeverity.INFO, title, message, **kwargs)

    async def warning(
        self,
        category: str,
        message: str,
        title: str = "Warning",
        **kwargs,
    ) -> Optional[Alert]:
        """Send warning alert."""
        cat = AlertCategory(category) if isinstance(category, str) else category
        return await self.alert(cat, AlertSeverity.WARNING, title, message, **kwargs)

    async def error(
        self,
        category: str,
        message: str,
        title: str = "Error",
        **kwargs,
    ) -> Optional[Alert]:
        """Send error alert."""
        cat = AlertCategory(category) if isinstance(category, str) else category
        return await self.alert(cat, AlertSeverity.ERROR, title, message, **kwargs)

    async def critical(
        self,
        category: str,
        message: str,
        title: str = "Critical",
        **kwargs,
    ) -> Optional[Alert]:
        """Send critical alert (bypasses throttle)."""
        cat = AlertCategory(category) if isinstance(category, str) else category
        return await self.alert(
            cat, AlertSeverity.CRITICAL, title, message,
            bypass_throttle=True, **kwargs
        )

    # Circuit breaker specific

    async def circuit_breaker_tripped(
        self,
        reason: str,
        details: str,
        snapshot: Optional[dict] = None,
    ) -> Optional[Alert]:
        """Send circuit breaker trip alert."""
        return await self.alert(
            AlertCategory.CIRCUIT_BREAKER,
            AlertSeverity.CRITICAL,
            "Circuit Breaker Tripped",
            f"{reason}: {details}",
            details=snapshot or {},
            bypass_throttle=True,
        )

    async def circuit_breaker_reset(self) -> Optional[Alert]:
        """Send circuit breaker reset alert."""
        return await self.alert(
            AlertCategory.CIRCUIT_BREAKER,
            AlertSeverity.INFO,
            "Circuit Breaker Reset",
            "Trading has been re-enabled",
        )

    # Trade specific

    async def trade_executed(
        self,
        side: str,
        size: Decimal,
        price: Decimal,
        market_id: str,
        target_name: str,
        slippage: Optional[Decimal] = None,
    ) -> Optional[Alert]:
        """Send trade execution alert."""
        msg = f"{side} {size} @ {price}"
        if slippage:
            msg += f" (slippage: {slippage*100:.2f}%)"

        return await self.alert(
            AlertCategory.TRADE,
            AlertSeverity.INFO,
            f"Trade Executed: {side}",
            msg,
            details={
                "side": side,
                "size": str(size),
                "price": str(price),
                "slippage": str(slippage) if slippage else None,
            },
            market_id=market_id,
            target_name=target_name,
        )

    async def trade_failed(
        self,
        reason: str,
        market_id: str,
        target_name: str,
        error: Optional[str] = None,
    ) -> Optional[Alert]:
        """Send trade failure alert."""
        return await self.alert(
            AlertCategory.TRADE,
            AlertSeverity.ERROR,
            "Trade Failed",
            reason,
            details={"error": error} if error else {},
            market_id=market_id,
            target_name=target_name,
        )

    # Health specific

    async def health_degraded(
        self,
        component: str,
        message: str,
    ) -> Optional[Alert]:
        """Send health degraded alert."""
        return await self.alert(
            AlertCategory.HEALTH,
            AlertSeverity.WARNING,
            f"Health Degraded: {component}",
            message,
            details={"component": component},
        )

    async def health_unhealthy(
        self,
        component: str,
        message: str,
    ) -> Optional[Alert]:
        """Send health unhealthy alert."""
        return await self.alert(
            AlertCategory.HEALTH,
            AlertSeverity.ERROR,
            f"Health Unhealthy: {component}",
            message,
            details={"component": component},
        )

    # History and stats

    def get_history(
        self,
        limit: int = 100,
        category: Optional[AlertCategory] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """Get alert history.

        Args:
            limit: Max alerts to return
            category: Filter by category
            severity: Filter by severity

        Returns:
            List of alerts
        """
        alerts = self._history

        if category:
            alerts = [a for a in alerts if a.category == category]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def get_stats(self) -> dict:
        """Get alert statistics.

        Returns:
            Stats dictionary
        """
        return {
            "total_sent": self._alerts_sent,
            "total_throttled": self._alerts_throttled,
            "history_size": len(self._history),
            "channels": len(self._channels),
            "throttle_status": self._throttler.get_throttle_status(),
        }
