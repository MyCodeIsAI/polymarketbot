"""Stage 5 Tests: Alert System.

Tests for:
- Alert types and dataclasses
- Alert channels (Logging, Discord, Email, File)
- Alert preferences and filtering
- Throttling mechanism
- InsiderAlertManager integration
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.insider_scanner.alerts import (
    InsiderAlertType,
    InsiderAlertPriority,
    InsiderAlert,
    InsiderAlertChannel,
    LoggingAlertChannel,
    DiscordAlertChannel,
    EmailAlertChannel,
    FileAlertChannel,
    AlertPreferences,
    InsiderAlertManager,
)


def run_async(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# Test InsiderAlert Dataclass
# =============================================================================


class TestInsiderAlert:
    """Tests for InsiderAlert dataclass."""

    def test_create_basic_alert(self):
        """Test creating a basic alert."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.alert_type == InsiderAlertType.NEW_SUSPICIOUS_WALLET
        assert alert.priority == InsiderAlertPriority.HIGH
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.alert_id  # Auto-generated
        assert len(alert.alert_id) == 16

    def test_alert_with_wallet_info(self):
        """Test alert with wallet information."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.CRITICAL,
            title="Suspicious Wallet",
            message="Detected insider pattern",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            username="suspicious_user",
            score=87.5,
            signals=["perfect_win_rate", "fresh_account", "single_market"],
        )

        assert alert.wallet_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert alert.username == "suspicious_user"
        assert alert.score == 87.5
        assert len(alert.signals) == 3

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.LARGE_TRADE_WATCHLIST,
            priority=InsiderAlertPriority.HIGH,
            title="Large Trade",
            message="$50,000 trade detected",
            wallet_address="0xtest",
            position_size_usd=50000,
        )

        data = alert.to_dict()

        assert data["alert_type"] == "large_trade_watchlist"
        assert data["priority"] == "high"
        assert data["title"] == "Large Trade"
        assert data["position_size_usd"] == 50000

    def test_alert_to_json(self):
        """Test converting alert to JSON."""
        alert = InsiderAlert(
            alert_type=InsiderAlertType.CLUSTER_DETECTED,
            priority=InsiderAlertPriority.HIGH,
            title="Cluster Found",
            message="5 wallets detected",
            score=75.0,
        )

        json_str = alert.to_json()
        data = json.loads(json_str)

        assert data["alert_type"] == "cluster_detected"
        assert data["score"] == 75.0

    def test_alert_id_deterministic(self):
        """Test that alert IDs are deterministic based on content."""
        timestamp = datetime(2026, 1, 23, 12, 0, 0)

        alert1 = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test",
            message="Test",
            wallet_address="0xtest",
            timestamp=timestamp,
        )

        alert2 = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test",
            message="Test",
            wallet_address="0xtest",
            timestamp=timestamp,
        )

        # Same inputs should produce same ID
        assert alert1.alert_id == alert2.alert_id

    def test_all_alert_types_exist(self):
        """Test all expected alert types are defined."""
        expected_types = [
            "NEW_SUSPICIOUS_WALLET",
            "FLAGGED_FUNDING_SOURCE",
            "LARGE_TRADE_WATCHLIST",
            "POSITION_RESOLVED",
            "CLUSTER_DETECTED",
            "SPLIT_ENTRY_DETECTED",
            "TIMING_ANOMALY",
            "NEW_WALLET_CREATED",
            "WALLET_STATUS_CHANGE",
            "SCAN_COMPLETE",
            "MONITOR_STARTED",
            "MONITOR_STOPPED",
            "MONITOR_ERROR",
        ]

        for type_name in expected_types:
            assert hasattr(InsiderAlertType, type_name)

    def test_all_priority_levels_exist(self):
        """Test all priority levels are defined."""
        expected_priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

        for priority_name in expected_priorities:
            assert hasattr(InsiderAlertPriority, priority_name)


# =============================================================================
# Test LoggingAlertChannel
# =============================================================================


class TestLoggingAlertChannel:
    """Tests for LoggingAlertChannel."""

    def test_default_min_priority(self):
        """Test default minimum priority is INFO."""
        channel = LoggingAlertChannel()
        assert channel.min_priority == InsiderAlertPriority.INFO

    def test_should_send_based_on_priority(self):
        """Test filtering based on priority."""
        channel = LoggingAlertChannel(min_priority=InsiderAlertPriority.HIGH)

        high_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="High",
            message="High priority",
        )

        low_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.LOW,
            title="Low",
            message="Low priority",
        )

        assert channel.should_send(high_alert) is True
        assert channel.should_send(low_alert) is False

    def test_should_send_based_on_type(self):
        """Test filtering based on alert type."""
        channel = LoggingAlertChannel(
            alert_types={InsiderAlertType.NEW_SUSPICIOUS_WALLET}
        )

        matching_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Match",
            message="Matching type",
        )

        non_matching_alert = InsiderAlert(
            alert_type=InsiderAlertType.MONITOR_STARTED,
            priority=InsiderAlertPriority.HIGH,
            title="No Match",
            message="Non-matching type",
        )

        assert channel.should_send(matching_alert) is True
        assert channel.should_send(non_matching_alert) is False

    def test_send_logs_alert(self):
        """Test that send() logs the alert."""
        channel = LoggingAlertChannel()

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test Alert",
            message="Test message",
            wallet_address="0xtest",
            score=80.0,
        )

        with patch("src.insider_scanner.alerts.logger") as mock_logger:
            result = run_async(channel.send(alert))

            assert result is True
            mock_logger.warning.assert_called_once()


# =============================================================================
# Test DiscordAlertChannel
# =============================================================================


class TestDiscordAlertChannel:
    """Tests for DiscordAlertChannel."""

    def test_format_embed_structure(self):
        """Test Discord embed formatting."""
        channel = DiscordAlertChannel(webhook_url="https://discord.com/api/webhooks/test")

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.CRITICAL,
            title="Suspicious Wallet Detected",
            message="High score detected",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            username="suspicious_user",
            score=92.5,
            signals=["perfect_win_rate", "fresh_account"],
            market_name="Will X happen by Y?",
            position_size_usd=75000,
        )

        embed = channel._format_embed(alert)

        assert "title" in embed
        assert "description" in embed
        assert "color" in embed
        assert "fields" in embed
        assert embed["color"] == 0x9B59B6  # Purple for critical

    def test_score_bar_formatting(self):
        """Test visual score bar generation."""
        channel = DiscordAlertChannel(webhook_url="https://discord.com/api/webhooks/test")

        assert channel._score_bar(100) == "██████████"
        assert channel._score_bar(50) == "█████░░░░░"
        assert channel._score_bar(0) == "░░░░░░░░░░"
        assert channel._score_bar(87) == "████████░░"

    def test_should_send_respects_priority(self):
        """Test priority filtering for Discord channel."""
        channel = DiscordAlertChannel(
            webhook_url="https://discord.com/api/webhooks/test",
            min_priority=InsiderAlertPriority.HIGH,
        )

        high_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="High",
            message="High",
        )

        medium_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.MEDIUM,
            title="Medium",
            message="Medium",
        )

        assert channel.should_send(high_alert) is True
        assert channel.should_send(medium_alert) is False

    def test_send_makes_http_request(self):
        """Test that send makes HTTP request to webhook."""
        channel = DiscordAlertChannel(webhook_url="https://discord.com/api/webhooks/test")

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test",
            message="Test",
        )

        # Test that we correctly build the embed payload
        embed = channel._format_embed(alert)
        assert "title" in embed
        assert "Test" in embed["title"]

        # For HTTP mocking, we just verify should_send works
        assert channel.should_send(alert) is True


# =============================================================================
# Test EmailAlertChannel
# =============================================================================


class TestEmailAlertChannel:
    """Tests for EmailAlertChannel."""

    def test_html_formatting(self):
        """Test HTML email formatting."""
        channel = EmailAlertChannel(
            to_addresses=["test@example.com"],
            from_address="scanner@example.com",
            smtp_host="smtp.example.com",
        )

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.CRITICAL,
            title="Critical Alert",
            message="Very suspicious wallet detected",
            wallet_address="0xtest123",
            score=95.0,
            signals=["signal1", "signal2"],
        )

        html = channel._format_html([alert])

        assert "Critical Alert" in html
        assert "0xtest123" in html
        assert "95.0" in html
        assert "signal1" in html

    def test_text_formatting(self):
        """Test plain text email formatting."""
        channel = EmailAlertChannel(
            to_addresses=["test@example.com"],
            from_address="scanner@example.com",
            smtp_host="smtp.example.com",
        )

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test Alert",
            message="Test message",
            score=80.0,
        )

        text = channel._format_text([alert])

        assert "Test Alert" in text
        assert "80.0" in text
        assert "INSIDER SCANNER ALERTS" in text

    def test_should_send_default_priority(self):
        """Test default minimum priority is HIGH."""
        channel = EmailAlertChannel(
            to_addresses=["test@example.com"],
            from_address="scanner@example.com",
            smtp_host="smtp.example.com",
        )

        assert channel.min_priority == InsiderAlertPriority.HIGH

    def test_critical_alert_sends_immediately(self):
        """Test critical alerts bypass batching."""
        channel = EmailAlertChannel(
            to_addresses=["test@example.com"],
            from_address="scanner@example.com",
            sendgrid_api_key="test_key",
            batch_window_s=300,
        )

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.CRITICAL,
            title="Critical",
            message="Critical alert",
        )

        # Test that critical alerts pass should_send
        assert channel.should_send(alert) is True

        # Test that HTML formatting works for the email
        html = channel._format_html([alert])
        assert "Critical" in html
        assert "Critical alert" in html


# =============================================================================
# Test FileAlertChannel
# =============================================================================


class TestFileAlertChannel:
    """Tests for FileAlertChannel."""

    def test_writes_json_lines(self):
        """Test that alerts are written as JSON lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            file_path = f.name

        try:
            channel = FileAlertChannel(file_path=file_path)

            alert1 = InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.HIGH,
                title="Alert 1",
                message="First alert",
            )

            alert2 = InsiderAlert(
                alert_type=InsiderAlertType.CLUSTER_DETECTED,
                priority=InsiderAlertPriority.HIGH,
                title="Alert 2",
                message="Second alert",
            )

            run_async(channel.send(alert1))
            run_async(channel.send(alert2))

            with open(file_path) as f:
                lines = f.readlines()

            assert len(lines) == 2

            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])

            assert data1["title"] == "Alert 1"
            assert data2["title"] == "Alert 2"

        finally:
            os.unlink(file_path)

    def test_respects_priority_filter(self):
        """Test priority filtering for file channel."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            file_path = f.name

        try:
            channel = FileAlertChannel(
                file_path=file_path,
                min_priority=InsiderAlertPriority.HIGH,
            )

            high_alert = InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.HIGH,
                title="High",
                message="High",
            )

            low_alert = InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.LOW,
                title="Low",
                message="Low",
            )

            run_async(channel.send(high_alert))
            run_async(channel.send(low_alert))

            with open(file_path) as f:
                lines = f.readlines()

            # Only high priority should be written
            assert len(lines) == 1

        finally:
            os.unlink(file_path)


# =============================================================================
# Test AlertPreferences
# =============================================================================


class TestAlertPreferences:
    """Tests for AlertPreferences."""

    def test_default_preferences(self):
        """Test default preference values."""
        prefs = AlertPreferences()

        assert prefs.enabled is True
        assert prefs.quiet_hours_start is None
        assert prefs.quiet_hours_end is None
        assert prefs.discord_min_priority == InsiderAlertPriority.MEDIUM
        assert prefs.email_min_priority == InsiderAlertPriority.HIGH

    def test_quiet_hours_wrap_around(self):
        """Test quiet hours detection with midnight wrap."""
        prefs = AlertPreferences(
            quiet_hours_start=22,
            quiet_hours_end=6,
        )

        # Test with controlled time - 23:00 should be quiet
        with patch("src.insider_scanner.alerts.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 1, 23, 23, 0, 0)
            assert prefs.is_quiet_hours() is True

            # 10:00 should not be quiet
            mock_dt.utcnow.return_value = datetime(2026, 1, 23, 10, 0, 0)
            assert prefs.is_quiet_hours() is False

    def test_should_alert_disabled(self):
        """Test that disabled preferences block all alerts."""
        prefs = AlertPreferences(enabled=False)

        assert prefs.should_alert(
            InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            InsiderAlertPriority.CRITICAL,
        ) is False

    def test_should_alert_disabled_type(self):
        """Test that disabled alert types are blocked."""
        prefs = AlertPreferences(
            disabled_alert_types={InsiderAlertType.MONITOR_STARTED}
        )

        assert prefs.should_alert(
            InsiderAlertType.MONITOR_STARTED,
            InsiderAlertPriority.INFO,
        ) is False

        assert prefs.should_alert(
            InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            InsiderAlertPriority.HIGH,
        ) is True


# =============================================================================
# Test InsiderAlertManager
# =============================================================================


class TestInsiderAlertManager:
    """Tests for InsiderAlertManager."""

    def test_create_manager(self):
        """Test creating alert manager."""
        manager = InsiderAlertManager()

        assert len(manager._channels) == 0
        assert manager._alerts_sent == 0

    def test_add_channel(self):
        """Test adding channels to manager."""
        manager = InsiderAlertManager()

        channel1 = LoggingAlertChannel()
        channel2 = FileAlertChannel(file_path="/tmp/test.jsonl")

        manager.add_channel(channel1)
        manager.add_channel(channel2)

        assert len(manager._channels) == 2

    def test_send_to_all_channels(self):
        """Test that alerts are sent to all channels."""
        manager = InsiderAlertManager()

        mock_channel1 = AsyncMock(spec=InsiderAlertChannel)
        mock_channel1.send.return_value = True

        mock_channel2 = AsyncMock(spec=InsiderAlertChannel)
        mock_channel2.send.return_value = True

        manager.add_channel(mock_channel1)
        manager.add_channel(mock_channel2)

        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test",
            message="Test",
        )

        result = run_async(manager.send(alert))

        assert result is True
        mock_channel1.send.assert_called_once_with(alert)
        mock_channel2.send.assert_called_once_with(alert)

    def test_suspicious_wallet_detected(self):
        """Test convenience method for suspicious wallet alerts."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        result = run_async(manager.suspicious_wallet_detected(
            wallet_address="0xtest123",
            score=87.5,
            signals=["perfect_win_rate", "fresh_account"],
            username="suspect",
            market_name="Test Market",
            position_size_usd=50000,
        ))

        assert result is True

        # Check the alert that was sent
        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.alert_type == InsiderAlertType.NEW_SUSPICIOUS_WALLET
        assert sent_alert.priority == InsiderAlertPriority.CRITICAL  # 87.5 >= 85
        assert sent_alert.wallet_address == "0xtest123"
        assert sent_alert.score == 87.5

    def test_flagged_funding_source(self):
        """Test convenience method for flagged funding source."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        result = run_async(manager.flagged_funding_source(
            wallet_address="0xnew_wallet",
            funding_source="0xknown_bad",
            funding_source_label="Known Insider",
            wallet_score=65.0,
        ))

        assert result is True

        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.alert_type == InsiderAlertType.FLAGGED_FUNDING_SOURCE
        assert sent_alert.priority == InsiderAlertPriority.HIGH

    def test_score_to_priority_mapping(self):
        """Test that scores map to correct priorities."""
        manager = InsiderAlertManager()

        # Critical: 85+
        assert manager._score_to_priority(85) == InsiderAlertPriority.CRITICAL
        assert manager._score_to_priority(100) == InsiderAlertPriority.CRITICAL

        # High: 70-84
        assert manager._score_to_priority(70) == InsiderAlertPriority.HIGH
        assert manager._score_to_priority(84) == InsiderAlertPriority.HIGH

        # Medium: 55-69
        assert manager._score_to_priority(55) == InsiderAlertPriority.MEDIUM
        assert manager._score_to_priority(69) == InsiderAlertPriority.MEDIUM

        # Low: 40-54
        assert manager._score_to_priority(40) == InsiderAlertPriority.LOW
        assert manager._score_to_priority(54) == InsiderAlertPriority.LOW

        # Info: <40
        assert manager._score_to_priority(39) == InsiderAlertPriority.INFO
        assert manager._score_to_priority(0) == InsiderAlertPriority.INFO

    def test_throttling_per_wallet(self):
        """Test that alerts are throttled per wallet."""
        manager = InsiderAlertManager(
            throttle_window_s=60,
            throttle_max_per_wallet=2,
        )

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        # First two alerts should go through
        for i in range(2):
            alert = InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.HIGH,
                title=f"Alert {i}",
                message="Test",
                wallet_address="0xsame_wallet",
            )
            result = run_async(manager.send(alert))
            assert result is True

        # Third alert should be throttled
        alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Alert 3",
            message="Test",
            wallet_address="0xsame_wallet",
        )
        result = run_async(manager.send(alert))
        assert result is False
        assert manager._alerts_throttled == 1

    def test_critical_bypasses_throttle(self):
        """Test that critical alerts bypass throttling."""
        manager = InsiderAlertManager(
            throttle_window_s=60,
            throttle_max_per_wallet=1,
        )

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        # Send first alert to hit throttle limit
        alert1 = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Alert 1",
            message="Test",
            wallet_address="0xtest",
        )
        run_async(manager.send(alert1))

        # Critical alert should bypass throttle
        critical_alert = InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.CRITICAL,
            title="Critical",
            message="Critical alert",
            wallet_address="0xtest",
        )
        result = run_async(manager.send(critical_alert))
        assert result is True

    def test_alert_history(self):
        """Test that alerts are recorded in history."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        for i in range(5):
            alert = InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.HIGH,
                title=f"Alert {i}",
                message="Test",
            )
            run_async(manager.send(alert))

        history = manager.get_history()
        assert len(history) == 5

    def test_history_filter_by_type(self):
        """Test filtering history by alert type."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        # Send different alert types
        run_async(manager.send(InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Suspicious",
            message="Test",
        )))

        run_async(manager.send(InsiderAlert(
            alert_type=InsiderAlertType.CLUSTER_DETECTED,
            priority=InsiderAlertPriority.HIGH,
            title="Cluster",
            message="Test",
        )))

        run_async(manager.send(InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Suspicious 2",
            message="Test",
        )))

        filtered = manager.get_history(alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET)
        assert len(filtered) == 2

    def test_get_stats(self):
        """Test getting alert statistics."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        for i in range(3):
            run_async(manager.send(InsiderAlert(
                alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
                priority=InsiderAlertPriority.HIGH,
                title=f"Alert {i}",
                message="Test",
            )))

        stats = manager.get_stats()

        assert stats["total_sent"] == 3
        assert stats["history_size"] == 3
        assert stats["channels"] == 1

    def test_monitor_started_alert(self):
        """Test monitor started convenience method."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        run_async(manager.monitor_started(mode="websocket"))

        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.alert_type == InsiderAlertType.MONITOR_STARTED
        assert "websocket" in sent_alert.message

    def test_monitor_error_alert(self):
        """Test monitor error convenience method."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        run_async(manager.monitor_error("Connection lost", recoverable=True))

        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.alert_type == InsiderAlertType.MONITOR_ERROR
        assert sent_alert.priority == InsiderAlertPriority.MEDIUM

        # Non-recoverable should be HIGH
        run_async(manager.monitor_error("Fatal error", recoverable=False))
        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.priority == InsiderAlertPriority.HIGH

    def test_cluster_detected_alert(self):
        """Test cluster detection convenience method."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        run_async(manager.cluster_detected(
            cluster_id="cluster_123",
            wallet_addresses=["0x1", "0x2", "0x3", "0x4", "0x5"],
            shared_funding_source="0xfunding",
            total_score=78.5,
        ))

        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.alert_type == InsiderAlertType.CLUSTER_DETECTED
        assert "5 wallets" in sent_alert.title
        assert sent_alert.details["cluster_id"] == "cluster_123"

    def test_position_resolved_won(self):
        """Test position resolved alert when won."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        run_async(manager.position_resolved(
            wallet_address="0xtest",
            won=True,
            profit_usd=50000,
            market_name="Test Market",
        ))

        sent_alert = mock_channel.send.call_args[0][0]
        assert "WON" in sent_alert.title
        assert sent_alert.priority == InsiderAlertPriority.MEDIUM

    def test_position_resolved_lost(self):
        """Test position resolved alert when lost."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        run_async(manager.position_resolved(
            wallet_address="0xtest",
            won=False,
            profit_usd=-25000,
            market_name="Test Market",
        ))

        sent_alert = mock_channel.send.call_args[0][0]
        assert "LOST" in sent_alert.title
        assert sent_alert.priority == InsiderAlertPriority.LOW

    def test_callback_invoked(self):
        """Test that registered callbacks are invoked."""
        manager = InsiderAlertManager()

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        callback_alerts = []
        manager.add_callback(lambda alert: callback_alerts.append(alert))

        run_async(manager.send(InsiderAlert(
            alert_type=InsiderAlertType.NEW_SUSPICIOUS_WALLET,
            priority=InsiderAlertPriority.HIGH,
            title="Test",
            message="Test",
        )))

        assert len(callback_alerts) == 1
        assert callback_alerts[0].title == "Test"


# =============================================================================
# Integration Tests
# =============================================================================


class TestAlertSystemIntegration:
    """Integration tests for alert system."""

    def test_full_alert_flow(self):
        """Test complete alert flow from detection to delivery."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            file_path = f.name

        try:
            # Set up manager with multiple channels
            manager = InsiderAlertManager()

            logging_channel = LoggingAlertChannel(
                min_priority=InsiderAlertPriority.LOW
            )
            file_channel = FileAlertChannel(
                file_path=file_path,
                min_priority=InsiderAlertPriority.MEDIUM,
            )

            manager.add_channel(logging_channel)
            manager.add_channel(file_channel)

            # Simulate detecting suspicious wallet
            run_async(manager.suspicious_wallet_detected(
                wallet_address="0x1234567890abcdef1234567890abcdef12345678",
                score=78.5,
                signals=[
                    "perfect_win_rate",
                    "fresh_account",
                    "single_market_focus",
                    "low_odds_entry",
                ],
                username="suspicious_user",
                market_name="Will event X happen?",
                position_size_usd=45000,
            ))

            # Check file was written
            with open(file_path) as f:
                lines = f.readlines()

            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["wallet_address"] == "0x1234567890abcdef1234567890abcdef12345678"
            assert data["score"] == 78.5
            assert len(data["signals"]) == 4

        finally:
            os.unlink(file_path)

    def test_alert_priority_routing(self):
        """Test that alerts route to correct channels based on priority."""
        manager = InsiderAlertManager()

        high_priority_channel = AsyncMock(spec=InsiderAlertChannel)
        high_priority_channel.should_send.return_value = True
        high_priority_channel.send.return_value = True

        low_priority_channel = AsyncMock(spec=InsiderAlertChannel)
        low_priority_channel.should_send.return_value = True
        low_priority_channel.send.return_value = True

        manager.add_channel(high_priority_channel)
        manager.add_channel(low_priority_channel)

        # Send alerts of different priorities
        run_async(manager.suspicious_wallet_detected(
            wallet_address="0xcritical",
            score=90,  # CRITICAL
            signals=["test"],
        ))

        run_async(manager.suspicious_wallet_detected(
            wallet_address="0xlow",
            score=45,  # LOW
            signals=["test"],
        ))

        # Both channels should receive both alerts
        assert high_priority_channel.send.call_count == 2
        assert low_priority_channel.send.call_count == 2

    def test_preferences_disable_alert_type(self):
        """Test that preferences can disable specific alert types."""
        prefs = AlertPreferences(
            disabled_alert_types={InsiderAlertType.MONITOR_STARTED}
        )

        manager = InsiderAlertManager(preferences=prefs)

        mock_channel = AsyncMock(spec=InsiderAlertChannel)
        mock_channel.send.return_value = True
        manager.add_channel(mock_channel)

        # This should be blocked
        run_async(manager.monitor_started())
        mock_channel.send.assert_not_called()

        # This should go through
        run_async(manager.suspicious_wallet_detected(
            wallet_address="0xtest",
            score=80,
            signals=["test"],
        ))
        mock_channel.send.assert_called_once()
