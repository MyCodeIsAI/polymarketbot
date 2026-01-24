"""Insider Scanner Alert System.

Multi-channel alert delivery for insider detection events.
Builds on the existing safety.alerts infrastructure.

Channels:
- Logging (structured logs)
- Discord webhook (rich embeds)
- Email (SendGrid/SMTP)
- Browser push (Web Push API)
- File (JSON lines for audit)
"""

import asyncio
import json
import hashlib
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class InsiderAlertType(str, Enum):
    """Types of insider detection alerts."""

    # High priority - potential insider detected
    NEW_SUSPICIOUS_WALLET = "new_suspicious_wallet"
    FLAGGED_FUNDING_SOURCE = "flagged_funding_source"
    LARGE_TRADE_WATCHLIST = "large_trade_watchlist"
    POSITION_RESOLVED = "position_resolved"

    # Medium priority - pattern detected
    CLUSTER_DETECTED = "cluster_detected"
    SPLIT_ENTRY_DETECTED = "split_entry_detected"
    TIMING_ANOMALY = "timing_anomaly"

    # Low priority - informational
    NEW_WALLET_CREATED = "new_wallet_created"
    WALLET_STATUS_CHANGE = "wallet_status_change"
    SCAN_COMPLETE = "scan_complete"

    # System
    MONITOR_STARTED = "monitor_started"
    MONITOR_STOPPED = "monitor_stopped"
    MONITOR_ERROR = "monitor_error"


class InsiderAlertPriority(str, Enum):
    """Alert priority levels matching insider scoring."""

    CRITICAL = "critical"  # Score 85+
    HIGH = "high"  # Score 70-84
    MEDIUM = "medium"  # Score 55-69
    LOW = "low"  # Score 40-54
    INFO = "info"  # Informational


@dataclass
class InsiderAlert:
    """An insider detection alert."""

    alert_type: InsiderAlertType
    priority: InsiderAlertPriority
    title: str
    message: str
    wallet_address: Optional[str] = None
    username: Optional[str] = None
    score: Optional[float] = None
    signals: list[str] = field(default_factory=list)
    market_id: Optional[str] = None
    market_name: Optional[str] = None
    position_size_usd: Optional[float] = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_id: str = ""

    def __post_init__(self):
        if not self.alert_id:
            # Generate deterministic alert ID
            content = f"{self.alert_type.value}_{self.wallet_address}_{self.timestamp.timestamp():.0f}"
            self.alert_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "wallet_address": self.wallet_address,
            "username": self.username,
            "score": self.score,
            "signals": self.signals,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "position_size_usd": self.position_size_usd,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class InsiderAlertChannel(ABC):
    """Base class for insider alert channels."""

    @abstractmethod
    async def send(self, alert: InsiderAlert) -> bool:
        """Send an alert through this channel."""
        pass

    @abstractmethod
    def should_send(self, alert: InsiderAlert) -> bool:
        """Check if this channel should send the alert."""
        pass


class LoggingAlertChannel(InsiderAlertChannel):
    """Logs alerts to structured logger."""

    def __init__(
        self,
        min_priority: InsiderAlertPriority = InsiderAlertPriority.INFO,
        alert_types: Optional[set[InsiderAlertType]] = None,
    ):
        self.min_priority = min_priority
        self.alert_types = alert_types
        self._priority_order = {
            InsiderAlertPriority.INFO: 0,
            InsiderAlertPriority.LOW: 1,
            InsiderAlertPriority.MEDIUM: 2,
            InsiderAlertPriority.HIGH: 3,
            InsiderAlertPriority.CRITICAL: 4,
        }

    def should_send(self, alert: InsiderAlert) -> bool:
        priority_ok = (
            self._priority_order[alert.priority]
            >= self._priority_order[self.min_priority]
        )
        type_ok = self.alert_types is None or alert.alert_type in self.alert_types
        return priority_ok and type_ok

    async def send(self, alert: InsiderAlert) -> bool:
        if not self.should_send(alert):
            return False

        log_func = {
            InsiderAlertPriority.INFO: logger.info,
            InsiderAlertPriority.LOW: logger.info,
            InsiderAlertPriority.MEDIUM: logger.warning,
            InsiderAlertPriority.HIGH: logger.warning,
            InsiderAlertPriority.CRITICAL: logger.error,
        }.get(alert.priority, logger.info)

        log_func(
            f"INSIDER_ALERT: {alert.title}",
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            priority=alert.priority.value,
            wallet=alert.wallet_address,
            score=alert.score,
            message=alert.message,
        )
        return True


class DiscordAlertChannel(InsiderAlertChannel):
    """Sends alerts to Discord webhook with rich embeds."""

    def __init__(
        self,
        webhook_url: str,
        min_priority: InsiderAlertPriority = InsiderAlertPriority.MEDIUM,
        alert_types: Optional[set[InsiderAlertType]] = None,
        timeout_s: float = 10,
        mention_role_id: Optional[str] = None,
    ):
        self.webhook_url = webhook_url
        self.min_priority = min_priority
        self.alert_types = alert_types
        self.timeout_s = timeout_s
        self.mention_role_id = mention_role_id
        self._priority_order = {
            InsiderAlertPriority.INFO: 0,
            InsiderAlertPriority.LOW: 1,
            InsiderAlertPriority.MEDIUM: 2,
            InsiderAlertPriority.HIGH: 3,
            InsiderAlertPriority.CRITICAL: 4,
        }

    def should_send(self, alert: InsiderAlert) -> bool:
        priority_ok = (
            self._priority_order[alert.priority]
            >= self._priority_order[self.min_priority]
        )
        type_ok = self.alert_types is None or alert.alert_type in self.alert_types
        return priority_ok and type_ok

    def _format_embed(self, alert: InsiderAlert) -> dict:
        """Format alert as Discord embed."""
        color_map = {
            InsiderAlertPriority.INFO: 0x3498DB,  # Blue
            InsiderAlertPriority.LOW: 0x2ECC71,  # Green
            InsiderAlertPriority.MEDIUM: 0xF39C12,  # Orange
            InsiderAlertPriority.HIGH: 0xE74C3C,  # Red
            InsiderAlertPriority.CRITICAL: 0x9B59B6,  # Purple
        }

        emoji_map = {
            InsiderAlertPriority.INFO: "ℹ️",
            InsiderAlertPriority.LOW: "🟢",
            InsiderAlertPriority.MEDIUM: "🟠",
            InsiderAlertPriority.HIGH: "🔴",
            InsiderAlertPriority.CRITICAL: "🚨",
        }

        embed = {
            "title": f"{emoji_map.get(alert.priority, '')} {alert.title}",
            "description": alert.message,
            "color": color_map.get(alert.priority, 0x3498DB),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [],
            "footer": {"text": f"Alert ID: {alert.alert_id}"},
        }

        # Add wallet info
        if alert.wallet_address:
            wallet_display = alert.wallet_address[:10] + "..." + alert.wallet_address[-6:]
            embed["fields"].append(
                {"name": "Wallet", "value": f"`{wallet_display}`", "inline": True}
            )

        if alert.username:
            embed["fields"].append(
                {"name": "Username", "value": alert.username, "inline": True}
            )

        # Add score
        if alert.score is not None:
            score_bar = self._score_bar(alert.score)
            embed["fields"].append(
                {"name": "Insider Score", "value": f"{alert.score:.1f}/100\n{score_bar}", "inline": True}
            )

        # Add market info
        if alert.market_name:
            embed["fields"].append(
                {"name": "Market", "value": alert.market_name[:50], "inline": True}
            )

        if alert.position_size_usd:
            embed["fields"].append(
                {"name": "Position Size", "value": f"${alert.position_size_usd:,.0f}", "inline": True}
            )

        # Add signals
        if alert.signals:
            signals_text = "\n".join(f"• {s}" for s in alert.signals[:10])
            embed["fields"].append(
                {"name": "Detected Signals", "value": signals_text[:1024], "inline": False}
            )

        return embed

    def _score_bar(self, score: float) -> str:
        """Create visual score bar."""
        filled = int(score / 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty

    async def send(self, alert: InsiderAlert) -> bool:
        if not self.should_send(alert):
            return False

        try:
            import aiohttp

            embed = self._format_embed(alert)
            payload = {"embeds": [embed]}

            # Add mention for critical alerts
            if alert.priority == InsiderAlertPriority.CRITICAL and self.mention_role_id:
                payload["content"] = f"<@&{self.mention_role_id}>"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as response:
                    if response.status >= 400:
                        logger.error(
                            "discord_send_failed",
                            status=response.status,
                            alert_id=alert.alert_id,
                        )
                        return False
                    return True

        except ImportError:
            logger.error("aiohttp_not_installed")
            return False
        except Exception as e:
            logger.error("discord_send_error", error=str(e), alert_id=alert.alert_id)
            return False


class EmailAlertChannel(InsiderAlertChannel):
    """Sends alerts via email (SMTP or SendGrid)."""

    def __init__(
        self,
        to_addresses: list[str],
        from_address: str,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sendgrid_api_key: Optional[str] = None,
        min_priority: InsiderAlertPriority = InsiderAlertPriority.HIGH,
        alert_types: Optional[set[InsiderAlertType]] = None,
        batch_window_s: int = 300,  # Batch emails within 5 minutes
    ):
        self.to_addresses = to_addresses
        self.from_address = from_address
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.sendgrid_api_key = sendgrid_api_key
        self.min_priority = min_priority
        self.alert_types = alert_types
        self.batch_window_s = batch_window_s

        self._priority_order = {
            InsiderAlertPriority.INFO: 0,
            InsiderAlertPriority.LOW: 1,
            InsiderAlertPriority.MEDIUM: 2,
            InsiderAlertPriority.HIGH: 3,
            InsiderAlertPriority.CRITICAL: 4,
        }

        # Batching
        self._pending_alerts: list[InsiderAlert] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def should_send(self, alert: InsiderAlert) -> bool:
        priority_ok = (
            self._priority_order[alert.priority]
            >= self._priority_order[self.min_priority]
        )
        type_ok = self.alert_types is None or alert.alert_type in self.alert_types
        return priority_ok and type_ok

    def _format_html(self, alerts: list[InsiderAlert]) -> str:
        """Format alerts as HTML email."""
        priority_colors = {
            InsiderAlertPriority.INFO: "#3498db",
            InsiderAlertPriority.LOW: "#2ecc71",
            InsiderAlertPriority.MEDIUM: "#f39c12",
            InsiderAlertPriority.HIGH: "#e74c3c",
            InsiderAlertPriority.CRITICAL: "#9b59b6",
        }

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert-critical {{ border-left: 4px solid #9b59b6; background: #f9f2ff; }}
                .alert-high {{ border-left: 4px solid #e74c3c; background: #fff5f5; }}
                .alert-medium {{ border-left: 4px solid #f39c12; background: #fffbf0; }}
                .alert-low {{ border-left: 4px solid #2ecc71; background: #f0fff4; }}
                .alert-info {{ border-left: 4px solid #3498db; background: #f0f8ff; }}
                .title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .meta {{ color: #666; font-size: 12px; margin-bottom: 10px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .signals {{ background: #f5f5f5; padding: 10px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td {{ padding: 5px 10px; border-bottom: 1px solid #eee; }}
                .label {{ color: #666; width: 120px; }}
            </style>
        </head>
        <body>
            <h1>🔍 Insider Scanner Alerts</h1>
            <p>You have {len(alerts)} new alert(s) from the Polymarket Insider Scanner.</p>
        """

        for alert in alerts:
            priority_class = f"alert-{alert.priority.value}"
            color = priority_colors.get(alert.priority, "#3498db")

            html += f"""
            <div class="alert {priority_class}">
                <div class="title" style="color: {color};">{alert.title}</div>
                <div class="meta">
                    {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} |
                    Priority: {alert.priority.value.upper()} |
                    Type: {alert.alert_type.value}
                </div>
                <p>{alert.message}</p>
                <table>
            """

            if alert.wallet_address:
                html += f'<tr><td class="label">Wallet</td><td><code>{alert.wallet_address}</code></td></tr>'

            if alert.username:
                html += f'<tr><td class="label">Username</td><td>{alert.username}</td></tr>'

            if alert.score is not None:
                html += f'<tr><td class="label">Score</td><td class="score">{alert.score:.1f}/100</td></tr>'

            if alert.market_name:
                html += f'<tr><td class="label">Market</td><td>{alert.market_name}</td></tr>'

            if alert.position_size_usd:
                html += f'<tr><td class="label">Position Size</td><td>${alert.position_size_usd:,.0f}</td></tr>'

            html += "</table>"

            if alert.signals:
                html += '<div class="signals"><strong>Detected Signals:</strong><ul>'
                for signal in alert.signals[:10]:
                    html += f"<li>{signal}</li>"
                html += "</ul></div>"

            html += "</div>"

        html += """
            <hr>
            <p style="color: #666; font-size: 12px;">
                This is an automated alert from the Polymarket Insider Scanner.
                <br>Manage your alert preferences in the dashboard.
            </p>
        </body>
        </html>
        """

        return html

    def _format_text(self, alerts: list[InsiderAlert]) -> str:
        """Format alerts as plain text."""
        text = f"INSIDER SCANNER ALERTS ({len(alerts)} alert(s))\n"
        text += "=" * 50 + "\n\n"

        for alert in alerts:
            text += f"[{alert.priority.value.upper()}] {alert.title}\n"
            text += f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            text += f"Type: {alert.alert_type.value}\n"
            text += f"\n{alert.message}\n\n"

            if alert.wallet_address:
                text += f"Wallet: {alert.wallet_address}\n"
            if alert.username:
                text += f"Username: {alert.username}\n"
            if alert.score is not None:
                text += f"Score: {alert.score:.1f}/100\n"
            if alert.market_name:
                text += f"Market: {alert.market_name}\n"
            if alert.position_size_usd:
                text += f"Position Size: ${alert.position_size_usd:,.0f}\n"

            if alert.signals:
                text += "\nDetected Signals:\n"
                for signal in alert.signals[:10]:
                    text += f"  - {signal}\n"

            text += "\n" + "-" * 50 + "\n\n"

        return text

    async def _send_batch(self, alerts: list[InsiderAlert]) -> bool:
        """Send a batch of alerts."""
        if not alerts:
            return True

        subject = f"[Insider Scanner] {len(alerts)} Alert(s) - "
        if any(a.priority == InsiderAlertPriority.CRITICAL for a in alerts):
            subject += "CRITICAL"
        elif any(a.priority == InsiderAlertPriority.HIGH for a in alerts):
            subject += "HIGH PRIORITY"
        else:
            subject += alerts[0].priority.value.upper()

        html_body = self._format_html(alerts)
        text_body = self._format_text(alerts)

        if self.sendgrid_api_key:
            return await self._send_sendgrid(subject, html_body, text_body)
        else:
            return await self._send_smtp(subject, html_body, text_body)

    async def _send_smtp(self, subject: str, html_body: str, text_body: str) -> bool:
        """Send via SMTP."""
        if not self.smtp_host:
            logger.error("smtp_not_configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._smtp_send, msg)

            logger.info("email_sent", to=self.to_addresses, subject=subject)
            return True

        except Exception as e:
            logger.error("smtp_send_error", error=str(e))
            return False

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Blocking SMTP send (run in executor)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)

    async def _send_sendgrid(self, subject: str, html_body: str, text_body: str) -> bool:
        """Send via SendGrid API."""
        try:
            import aiohttp

            payload = {
                "personalizations": [{"to": [{"email": addr} for addr in self.to_addresses]}],
                "from": {"email": self.from_address},
                "subject": subject,
                "content": [
                    {"type": "text/plain", "value": text_body},
                    {"type": "text/html", "value": html_body},
                ],
            }

            headers = {
                "Authorization": f"Bearer {self.sendgrid_api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status >= 400:
                        logger.error("sendgrid_send_failed", status=response.status)
                        return False
                    return True

        except ImportError:
            logger.error("aiohttp_not_installed")
            return False
        except Exception as e:
            logger.error("sendgrid_send_error", error=str(e))
            return False

    async def send(self, alert: InsiderAlert) -> bool:
        """Queue alert for batched sending."""
        if not self.should_send(alert):
            return False

        async with self._lock:
            self._pending_alerts.append(alert)

            # For critical alerts, send immediately
            if alert.priority == InsiderAlertPriority.CRITICAL:
                alerts = self._pending_alerts.copy()
                self._pending_alerts.clear()
                return await self._send_batch(alerts)

            # Start batch timer if not running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._batch_timer())

        return True

    async def _batch_timer(self):
        """Wait for batch window then send."""
        await asyncio.sleep(self.batch_window_s)
        async with self._lock:
            if self._pending_alerts:
                alerts = self._pending_alerts.copy()
                self._pending_alerts.clear()
                await self._send_batch(alerts)


class FileAlertChannel(InsiderAlertChannel):
    """Writes alerts to JSON lines file for audit trail."""

    def __init__(
        self,
        file_path: str,
        min_priority: InsiderAlertPriority = InsiderAlertPriority.INFO,
    ):
        self.file_path = file_path
        self.min_priority = min_priority
        self._priority_order = {
            InsiderAlertPriority.INFO: 0,
            InsiderAlertPriority.LOW: 1,
            InsiderAlertPriority.MEDIUM: 2,
            InsiderAlertPriority.HIGH: 3,
            InsiderAlertPriority.CRITICAL: 4,
        }
        self._lock = asyncio.Lock()

    def should_send(self, alert: InsiderAlert) -> bool:
        return (
            self._priority_order[alert.priority]
            >= self._priority_order[self.min_priority]
        )

    async def send(self, alert: InsiderAlert) -> bool:
        if not self.should_send(alert):
            return False

        try:
            async with self._lock:
                with open(self.file_path, "a") as f:
                    f.write(alert.to_json() + "\n")
            return True
        except Exception as e:
            logger.error("file_write_error", error=str(e), alert_id=alert.alert_id)
            return False


@dataclass
class AlertPreferences:
    """User preferences for alert delivery."""

    # Global settings
    enabled: bool = True
    quiet_hours_start: Optional[int] = None  # 0-23 UTC
    quiet_hours_end: Optional[int] = None

    # Priority thresholds per channel
    discord_min_priority: InsiderAlertPriority = InsiderAlertPriority.MEDIUM
    email_min_priority: InsiderAlertPriority = InsiderAlertPriority.HIGH
    browser_min_priority: InsiderAlertPriority = InsiderAlertPriority.MEDIUM

    # Alert type filters
    disabled_alert_types: set[InsiderAlertType] = field(default_factory=set)

    # Batching
    email_batch_minutes: int = 5

    def is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False

        current_hour = datetime.utcnow().hour
        start = self.quiet_hours_start
        end = self.quiet_hours_end

        if start <= end:
            return start <= current_hour < end
        else:  # Wraps around midnight
            return current_hour >= start or current_hour < end

    def should_alert(self, alert_type: InsiderAlertType, priority: InsiderAlertPriority) -> bool:
        """Check if alert should be sent based on preferences."""
        if not self.enabled:
            return False

        if alert_type in self.disabled_alert_types:
            return False

        # Critical alerts bypass quiet hours
        if priority != InsiderAlertPriority.CRITICAL and self.is_quiet_hours():
            return False

        return True


class InsiderAlertManager:
    """Central manager for insider detection alerts.

    Coordinates alert delivery across multiple channels with:
    - Priority-based routing
    - Throttling to prevent spam
    - User preferences support
    - Alert history and statistics

    Example:
        manager = InsiderAlertManager()

        # Add channels
        manager.add_channel(LoggingAlertChannel())
        manager.add_channel(DiscordAlertChannel(webhook_url))
        manager.add_channel(EmailAlertChannel(
            to_addresses=["alerts@example.com"],
            from_address="scanner@example.com",
            sendgrid_api_key="..."
        ))

        # Send alert
        await manager.suspicious_wallet_detected(
            wallet_address="0x123...",
            username="suspicious_user",
            score=87.5,
            signals=["perfect_win_rate", "fresh_account", "single_market"],
            market_name="Will X happen?",
            position_size_usd=50000
        )
    """

    def __init__(
        self,
        preferences: Optional[AlertPreferences] = None,
        throttle_window_s: int = 60,
        throttle_max_per_wallet: int = 3,
    ):
        self._channels: list[InsiderAlertChannel] = []
        self._preferences = preferences or AlertPreferences()
        self._throttle_window_s = throttle_window_s
        self._throttle_max = throttle_max_per_wallet

        # Throttle tracking: {wallet_address: [timestamps]}
        self._throttle_history: dict[str, list[datetime]] = {}

        # Alert history
        self._history: list[InsiderAlert] = []
        self._max_history = 1000

        # Stats
        self._alerts_sent = 0
        self._alerts_throttled = 0

        # Callbacks
        self._on_alert_callbacks: list[Callable[[InsiderAlert], None]] = []

    def add_channel(self, channel: InsiderAlertChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)

    def add_callback(self, callback: Callable[[InsiderAlert], None]) -> None:
        """Add callback to be called on every alert."""
        self._on_alert_callbacks.append(callback)

    def set_preferences(self, preferences: AlertPreferences) -> None:
        """Update alert preferences."""
        self._preferences = preferences

    def _should_throttle(self, wallet_address: Optional[str]) -> bool:
        """Check if alerts for this wallet should be throttled."""
        if not wallet_address:
            return False

        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self._throttle_window_s)

        if wallet_address not in self._throttle_history:
            self._throttle_history[wallet_address] = []

        # Clean old entries
        self._throttle_history[wallet_address] = [
            t for t in self._throttle_history[wallet_address] if t > cutoff
        ]

        # Check limit
        if len(self._throttle_history[wallet_address]) >= self._throttle_max:
            return True

        # Record this alert
        self._throttle_history[wallet_address].append(now)
        return False

    async def send(self, alert: InsiderAlert, bypass_throttle: bool = False) -> bool:
        """Send an alert through all configured channels.

        Args:
            alert: Alert to send
            bypass_throttle: Skip throttle check

        Returns:
            True if sent to at least one channel
        """
        # Check preferences
        if not self._preferences.should_alert(alert.alert_type, alert.priority):
            return False

        # Check throttle (critical alerts bypass)
        if not bypass_throttle and alert.priority != InsiderAlertPriority.CRITICAL:
            if self._should_throttle(alert.wallet_address):
                self._alerts_throttled += 1
                logger.debug(
                    "alert_throttled",
                    wallet=alert.wallet_address,
                    alert_type=alert.alert_type.value,
                )
                return False

        # Send through all channels
        sent = False
        for channel in self._channels:
            try:
                if await channel.send(alert):
                    sent = True
            except Exception as e:
                logger.error(
                    "channel_send_error",
                    channel=type(channel).__name__,
                    error=str(e),
                )

        # Record in history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        if sent:
            self._alerts_sent += 1

        # Notify callbacks
        for callback in self._on_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("callback_error", error=str(e))

        return sent

    # Convenience methods for specific alert types

    async def suspicious_wallet_detected(
        self,
        wallet_address: str,
        score: float,
        signals: list[str],
        username: Optional[str] = None,
        market_name: Optional[str] = None,
        market_id: Optional[str] = None,
        position_size_usd: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Alert for newly detected suspicious wallet."""
        priority = self._score_to_priority(score)

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=priority,
            title=f"Suspicious Wallet Detected ({priority.value.upper()})",
            message=f"Wallet {wallet_address[:10]}... scored {score:.1f}/100 with {len(signals)} insider signals.",
            wallet_address=wallet_address,
            username=username,
            score=score,
            signals=signals,
            market_name=market_name,
            market_id=market_id,
            position_size_usd=position_size_usd,
            details=details or {},
        )

        return await self.send(alert)

    async def flagged_funding_source(
        self,
        wallet_address: str,
        funding_source: str,
        funding_source_label: Optional[str] = None,
        wallet_score: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Alert for wallet funded by flagged source."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.FLAGGED_FUNDING_SOURCE,
            priority=InsiderAlertPriority.HIGH,
            title="Flagged Funding Source Detected",
            message=f"Wallet {wallet_address[:10]}... received funds from known suspicious source {funding_source[:10]}... ({funding_source_label or 'unknown'}).",
            wallet_address=wallet_address,
            score=wallet_score,
            details={
                "funding_source": funding_source,
                "funding_source_label": funding_source_label,
                **(details or {}),
            },
        )

        return await self.send(alert)

    async def large_trade_watchlist(
        self,
        wallet_address: str,
        trade_size_usd: float,
        market_name: str,
        side: str,
        username: Optional[str] = None,
        wallet_score: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> bool:
        """Alert for large trade from watchlist wallet."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.LARGE_TRADE_WATCHLIST,
            priority=InsiderAlertPriority.HIGH,
            title="Large Trade from Watchlist Wallet",
            message=f"Wallet {wallet_address[:10]}... placed ${trade_size_usd:,.0f} {side} on '{market_name}'.",
            wallet_address=wallet_address,
            username=username,
            score=wallet_score,
            market_name=market_name,
            position_size_usd=trade_size_usd,
            details={"side": side, **(details or {})},
        )

        return await self.send(alert)

    async def position_resolved(
        self,
        wallet_address: str,
        won: bool,
        profit_usd: float,
        market_name: str,
        username: Optional[str] = None,
        wallet_score: Optional[float] = None,
    ) -> bool:
        """Alert when watchlist wallet's position resolves."""
        outcome = "WON" if won else "LOST"
        priority = InsiderAlertPriority.MEDIUM if won else InsiderAlertPriority.LOW

        alert = InsiderAlert(
            alert_type=InsiderAlertType.POSITION_RESOLVED,
            priority=priority,
            title=f"Watchlist Position {outcome}",
            message=f"Wallet {wallet_address[:10]}... {outcome.lower()} ${abs(profit_usd):,.0f} on '{market_name}'.",
            wallet_address=wallet_address,
            username=username,
            score=wallet_score,
            market_name=market_name,
            position_size_usd=abs(profit_usd),
            details={"won": won, "profit_usd": profit_usd},
        )

        return await self.send(alert)

    async def cluster_detected(
        self,
        cluster_id: str,
        wallet_addresses: list[str],
        shared_funding_source: str,
        total_score: float,
    ) -> bool:
        """Alert when a wallet cluster is detected."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.CLUSTER_DETECTED,
            priority=InsiderAlertPriority.HIGH,
            title=f"Wallet Cluster Detected ({len(wallet_addresses)} wallets)",
            message=f"Detected cluster of {len(wallet_addresses)} wallets sharing funding source {shared_funding_source[:10]}...",
            score=total_score,
            details={
                "cluster_id": cluster_id,
                "wallet_count": len(wallet_addresses),
                "wallet_addresses": wallet_addresses[:5],
                "shared_funding_source": shared_funding_source,
            },
        )

        return await self.send(alert)

    async def monitor_started(self, mode: str = "websocket") -> bool:
        """Alert when monitor starts."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.MONITOR_STARTED,
            priority=InsiderAlertPriority.INFO,
            title="Insider Monitor Started",
            message=f"Real-time monitoring started in {mode} mode.",
        )

        return await self.send(alert)

    async def monitor_stopped(self, reason: str = "manual") -> bool:
        """Alert when monitor stops."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.MONITOR_STOPPED,
            priority=InsiderAlertPriority.INFO,
            title="Insider Monitor Stopped",
            message=f"Real-time monitoring stopped. Reason: {reason}",
        )

        return await self.send(alert)

    async def monitor_error(self, error: str, recoverable: bool = True) -> bool:
        """Alert on monitor error."""
        priority = InsiderAlertPriority.MEDIUM if recoverable else InsiderAlertPriority.HIGH

        alert = InsiderAlert(
            alert_type=InsiderAlertType.MONITOR_ERROR,
            priority=priority,
            title="Monitor Error",
            message=f"{'Recoverable' if recoverable else 'Critical'} error in monitor: {error}",
            details={"error": error, "recoverable": recoverable},
        )

        return await self.send(alert, bypass_throttle=not recoverable)

    def _score_to_priority(self, score: float) -> InsiderAlertPriority:
        """Convert insider score to alert priority."""
        if score >= 85:
            return InsiderAlertPriority.CRITICAL
        elif score >= 70:
            return InsiderAlertPriority.HIGH
        elif score >= 55:
            return InsiderAlertPriority.MEDIUM
        elif score >= 40:
            return InsiderAlertPriority.LOW
        else:
            return InsiderAlertPriority.INFO

    # History and stats

    def get_history(
        self,
        limit: int = 100,
        alert_type: Optional[InsiderAlertType] = None,
        priority: Optional[InsiderAlertPriority] = None,
        wallet_address: Optional[str] = None,
    ) -> list[InsiderAlert]:
        """Get alert history with optional filters."""
        alerts = self._history

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        if wallet_address:
            alerts = [a for a in alerts if a.wallet_address == wallet_address]

        return alerts[-limit:]

    def get_stats(self) -> dict:
        """Get alert statistics."""
        return {
            "total_sent": self._alerts_sent,
            "total_throttled": self._alerts_throttled,
            "history_size": len(self._history),
            "channels": len(self._channels),
            "preferences": {
                "enabled": self._preferences.enabled,
                "quiet_hours": f"{self._preferences.quiet_hours_start}-{self._preferences.quiet_hours_end}",
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Flush any pending email batches
        for channel in self._channels:
            if isinstance(channel, EmailAlertChannel):
                if channel._pending_alerts:
                    await channel._send_batch(channel._pending_alerts)
                    channel._pending_alerts.clear()
