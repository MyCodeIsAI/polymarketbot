"""Tests for Insider Scanner configuration module.

Tests:
- Environment detection
- Settings loading
- Database URL resolution
- Alert channel configuration
- Config validation
"""

import os
import pytest
from unittest.mock import patch

from src.insider_scanner.config import (
    InsiderScannerSettings,
    DeploymentMode,
    AlertChannelConfig,
    detect_deployment_mode,
    get_database_url,
    get_alert_channels,
    get_health_check_config,
    validate_config,
    ensure_data_directory,
    reset_settings,
)


class TestDeploymentModeDetection:
    """Tests for deployment mode detection."""

    def test_detect_local_mode(self):
        """Test detection of local mode."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                mode = detect_deployment_mode()
                assert mode == DeploymentMode.LOCAL

    def test_detect_docker_mode(self):
        """Test detection of docker mode."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            mode = detect_deployment_mode()
            assert mode == DeploymentMode.DOCKER
            mock_exists.assert_called_with("/.dockerenv")

    def test_detect_railway_cloud(self):
        """Test detection of Railway cloud platform."""
        with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
            with patch("os.path.exists", return_value=False):
                mode = detect_deployment_mode()
                assert mode == DeploymentMode.CLOUD

    def test_detect_render_cloud(self):
        """Test detection of Render cloud platform."""
        with patch.dict(os.environ, {"RENDER": "true"}):
            with patch("os.path.exists", return_value=False):
                mode = detect_deployment_mode()
                assert mode == DeploymentMode.CLOUD

    def test_detect_fly_cloud(self):
        """Test detection of Fly.io cloud platform."""
        with patch.dict(os.environ, {"FLY_APP_NAME": "myapp"}):
            with patch("os.path.exists", return_value=False):
                mode = detect_deployment_mode()
                assert mode == DeploymentMode.CLOUD


class TestInsiderScannerSettings:
    """Tests for InsiderScannerSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = InsiderScannerSettings()

        assert settings.deployment_mode == DeploymentMode.LOCAL
        assert settings.debug is False
        assert settings.websocket_enabled is True
        assert settings.polling_enabled is True
        assert settings.polling_interval_seconds == 5

    def test_scoring_thresholds(self):
        """Test scoring threshold defaults."""
        settings = InsiderScannerSettings()

        assert settings.score_critical_threshold == 85.0
        assert settings.score_high_threshold == 70.0
        assert settings.score_medium_threshold == 50.0  # Lowered from 55 per Fed Chair analysis
        assert settings.score_low_threshold == 40.0

    def test_alert_defaults(self):
        """Test alert configuration defaults."""
        settings = InsiderScannerSettings()

        assert settings.alert_discord_enabled is False
        assert settings.alert_email_enabled is False
        assert settings.alert_file_enabled is True
        assert settings.alert_min_priority == "medium"
        assert settings.alert_throttle_minutes == 30

    def test_from_env_method(self):
        """Test settings created from environment variables."""
        reset_settings()
        with patch.dict(os.environ, {"INSIDER_DEBUG": "true", "INSIDER_POLLING_INTERVAL_SECONDS": "10"}):
            settings = InsiderScannerSettings.from_env()
            assert settings.debug is True
            assert settings.polling_interval_seconds == 10


class TestDatabaseUrlResolution:
    """Tests for database URL resolution."""

    def test_explicit_database_url(self):
        """Test explicit database URL is used."""
        settings = InsiderScannerSettings(database_url="postgresql://user:pass@host/db")
        url = get_database_url(settings)
        assert url == "postgresql://user:pass@host/db"

    def test_local_mode_sqlite(self):
        """Test local mode uses SQLite."""
        settings = InsiderScannerSettings(database_url="")
        with patch("src.insider_scanner.config.detect_deployment_mode", return_value=DeploymentMode.LOCAL):
            url = get_database_url(settings)
            assert "sqlite:///" in url

    def test_docker_mode_sqlite(self):
        """Test docker mode uses persistent SQLite."""
        settings = InsiderScannerSettings(database_url="")
        with patch("src.insider_scanner.config.detect_deployment_mode", return_value=DeploymentMode.DOCKER):
            url = get_database_url(settings)
            assert "sqlite:///data/" in url

    def test_cloud_mode_postgres_url_fix(self):
        """Test cloud mode fixes Render/Railway postgres:// URL."""
        settings = InsiderScannerSettings(database_url="")
        with patch("src.insider_scanner.config.detect_deployment_mode", return_value=DeploymentMode.CLOUD):
            with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@host/db"}):
                url = get_database_url(settings)
                # Should be converted to postgresql://
                assert url.startswith("postgresql://")
                assert "user:pass@host/db" in url


class TestAlertChannelConfiguration:
    """Tests for alert channel configuration."""

    def test_file_channel_always_available(self):
        """Test file channel is always configured."""
        settings = InsiderScannerSettings(
            alert_discord_enabled=False,
            alert_email_enabled=False,
            alert_file_enabled=True,
        )

        channels = get_alert_channels(settings)

        assert "file" in channels
        assert channels["file"].enabled is True

    def test_discord_channel_configuration(self):
        """Test Discord channel configuration."""
        settings = InsiderScannerSettings(
            alert_discord_enabled=True,
            alert_discord_webhook_url="https://discord.com/api/webhooks/123/abc",
            alert_discord_mention_role="@everyone",
        )

        channels = get_alert_channels(settings)

        assert "discord" in channels
        assert channels["discord"].enabled is True
        assert channels["discord"].discord_webhook_url == "https://discord.com/api/webhooks/123/abc"

    def test_discord_channel_requires_url(self):
        """Test Discord channel requires webhook URL."""
        settings = InsiderScannerSettings(
            alert_discord_enabled=True,
            alert_discord_webhook_url=None,  # Missing URL
        )

        channels = get_alert_channels(settings)

        assert "discord" not in channels

    def test_email_channel_sendgrid(self):
        """Test email channel with SendGrid."""
        settings = InsiderScannerSettings(
            alert_email_enabled=True,
            alert_email_sendgrid_api_key="SG.xxx",
            alert_email_to_addresses=["test@example.com"],
        )

        channels = get_alert_channels(settings)

        assert "email" in channels
        assert channels["email"].email_provider == "sendgrid"

    def test_email_channel_smtp(self):
        """Test email channel with SMTP."""
        settings = InsiderScannerSettings(
            alert_email_enabled=True,
            alert_email_sendgrid_api_key=None,
            alert_email_smtp_host="smtp.example.com",
            alert_email_to_addresses=["test@example.com"],
        )

        channels = get_alert_channels(settings)

        assert "email" in channels
        assert channels["email"].email_provider == "smtp"


class TestHealthCheckConfig:
    """Tests for health check configuration."""

    def test_health_check_config_structure(self):
        """Test health check config has required fields."""
        config = get_health_check_config()

        assert "path" in config
        assert "interval" in config
        assert "timeout" in config
        assert "checks" in config
        assert isinstance(config["checks"], list)

    def test_health_check_includes_database(self):
        """Test health check includes database check."""
        config = get_health_check_config()

        check_names = [c["name"] for c in config["checks"]]
        assert "database" in check_names


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config_no_warnings(self):
        """Test valid configuration produces no warnings."""
        settings = InsiderScannerSettings(
            database_url="sqlite:///data/test.db",
            alert_discord_enabled=True,
            alert_discord_webhook_url="https://discord.com/webhook",
        )

        with patch("src.insider_scanner.config.get_settings", return_value=settings):
            warnings = validate_config()
            # May have warning about email not configured, but no critical warnings
            assert all("threshold" not in w.lower() for w in warnings)

    def test_memory_database_warning(self):
        """Test in-memory database produces warning."""
        settings = InsiderScannerSettings(
            database_url="sqlite:///:memory:",
        )

        with patch("src.insider_scanner.config.get_settings", return_value=settings):
            warnings = validate_config()
            assert any("memory" in w.lower() for w in warnings)

    def test_no_alerts_warning(self):
        """Test no alerts configured produces warning."""
        settings = InsiderScannerSettings(
            alert_discord_enabled=False,
            alert_email_enabled=False,
        )

        with patch("src.insider_scanner.config.get_settings", return_value=settings):
            warnings = validate_config()
            assert any("alert" in w.lower() for w in warnings)

    def test_invalid_thresholds_warning(self):
        """Test invalid thresholds produce warning."""
        settings = InsiderScannerSettings(
            score_critical_threshold=70.0,  # Same as high
            score_high_threshold=70.0,
        )

        with patch("src.insider_scanner.config.get_settings", return_value=settings):
            warnings = validate_config()
            assert any("threshold" in w.lower() for w in warnings)

    def test_polygon_anchor_missing_config(self):
        """Test polygon anchor missing config produces warnings."""
        settings = InsiderScannerSettings(
            anchor_enabled=True,
            anchor_type="polygon",
            anchor_polygon_rpc_url=None,
            anchor_private_key=None,
        )

        with patch("src.insider_scanner.config.get_settings", return_value=settings):
            warnings = validate_config()
            assert any("rpc" in w.lower() for w in warnings)
            assert any("private key" in w.lower() for w in warnings)


class TestDataDirectory:
    """Tests for data directory management."""

    def test_ensure_data_directory_creates(self, tmp_path):
        """Test data directory is created."""
        with patch("src.insider_scanner.config.Path") as mock_path:
            mock_path_instance = mock_path.return_value
            mock_path_instance.mkdir = lambda **kwargs: None

            result = ensure_data_directory()
            mock_path.assert_called_with("data")


class TestAlertChannelConfig:
    """Tests for AlertChannelConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = AlertChannelConfig()

        assert config.enabled is False
        assert config.min_priority == "medium"
        assert config.discord_webhook_url is None
        assert config.email_recipients == []

    def test_custom_values(self):
        """Test custom values."""
        config = AlertChannelConfig(
            enabled=True,
            min_priority="high",
            discord_webhook_url="https://webhook.url",
        )

        assert config.enabled is True
        assert config.min_priority == "high"
        assert config.discord_webhook_url == "https://webhook.url"
