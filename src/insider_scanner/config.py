"""Configuration for Insider Scanner.

Handles:
- Environment detection (local vs cloud)
- Database configuration
- Alert channel configuration
- Performance settings
- Feature flags
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from enum import Enum


class DeploymentMode(str, Enum):
    """Deployment environment mode."""
    LOCAL = "local"
    DOCKER = "docker"
    CLOUD = "cloud"


@dataclass
class InsiderScannerSettings:
    """Settings for the Insider Scanner module.

    Reads from environment variables with INSIDER_ prefix.
    """

    # Deployment
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    debug: bool = False

    # Database
    database_url: str = "sqlite:///data/insider_scanner.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_recycle: int = 3600  # 1 hour

    # Real-time monitoring
    websocket_enabled: bool = True
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polling_enabled: bool = True
    polling_interval_seconds: int = 5
    auto_reconnect: bool = True
    max_reconnect_delay_seconds: int = 30

    # Scoring thresholds
    score_critical_threshold: float = 85.0
    score_high_threshold: float = 70.0
    score_medium_threshold: float = 55.0
    score_low_threshold: float = 40.0

    # Alert configuration
    alert_discord_enabled: bool = False
    alert_discord_webhook_url: Optional[str] = None
    alert_discord_mention_role: Optional[str] = None

    alert_email_enabled: bool = False
    alert_email_sendgrid_api_key: Optional[str] = None
    alert_email_smtp_host: Optional[str] = None
    alert_email_smtp_port: int = 587
    alert_email_smtp_user: Optional[str] = None
    alert_email_smtp_password: Optional[str] = None
    alert_email_from_address: Optional[str] = None
    alert_email_to_addresses: List[str] = field(default_factory=list)

    alert_file_enabled: bool = True
    alert_file_path: str = "data/insider_alerts.jsonl"

    alert_min_priority: str = "medium"
    alert_quiet_hours_start: Optional[int] = None  # 0-23
    alert_quiet_hours_end: Optional[int] = None  # 0-23
    alert_throttle_minutes: int = 30

    # Blockchain anchoring (optional strong proof)
    anchor_enabled: bool = False
    anchor_type: str = "local"  # local, polygon, opentimestamps
    anchor_polygon_rpc_url: Optional[str] = None
    anchor_private_key: Optional[str] = None

    # Performance
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    cache_ttl_seconds: int = 300
    batch_size: int = 100

    # Monitoring
    metrics_enabled: bool = True
    metrics_port: int = 9090

    @classmethod
    def from_env(cls) -> "InsiderScannerSettings":
        """Create settings from environment variables with INSIDER_ prefix."""
        def get_env(key: str, default, type_fn=str):
            val = os.environ.get(f"INSIDER_{key.upper()}", default)
            if val is None:
                return default
            if type_fn == bool:
                return str(val).lower() in ("true", "1", "yes")
            return type_fn(val) if val is not default else default

        return cls(
            deployment_mode=DeploymentMode(get_env("deployment_mode", "local")),
            debug=get_env("debug", False, bool),
            database_url=get_env("database_url", "sqlite:///data/insider_scanner.db"),
            polling_interval_seconds=get_env("polling_interval_seconds", 5, int),
            alert_discord_enabled=get_env("alert_discord_enabled", False, bool),
            alert_discord_webhook_url=get_env("alert_discord_webhook_url", None),
            alert_email_enabled=get_env("alert_email_enabled", False, bool),
            alert_file_enabled=get_env("alert_file_enabled", True, bool),
        )


# Cached settings instance
_settings: Optional[InsiderScannerSettings] = None


def get_settings() -> InsiderScannerSettings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = InsiderScannerSettings.from_env()
    return _settings


def reset_settings():
    """Reset cached settings (for testing)."""
    global _settings
    _settings = None


def detect_deployment_mode() -> DeploymentMode:
    """Auto-detect deployment mode from environment.

    Returns:
        DeploymentMode based on environment detection
    """
    # Docker container detection
    if os.path.exists("/.dockerenv"):
        return DeploymentMode.DOCKER

    # Cloud platform detection (Railway, Render, Fly.io, etc.)
    cloud_vars = ["RAILWAY_ENVIRONMENT", "RENDER", "FLY_APP_NAME", "VERCEL", "AWS_LAMBDA_FUNCTION_NAME"]
    if any(os.environ.get(var) for var in cloud_vars):
        return DeploymentMode.CLOUD

    return DeploymentMode.LOCAL


def get_database_url(settings: Optional[InsiderScannerSettings] = None) -> str:
    """Get the appropriate database URL based on deployment mode.

    Args:
        settings: Optional settings instance

    Returns:
        Database URL string
    """
    if settings is None:
        settings = get_settings()

    # Explicit override
    if settings.database_url:
        return settings.database_url

    mode = detect_deployment_mode()

    if mode == DeploymentMode.DOCKER:
        return "sqlite:///data/insider_scanner.db"
    elif mode == DeploymentMode.CLOUD:
        # Check for cloud database URL
        postgres_url = os.environ.get("DATABASE_URL")
        if postgres_url:
            # Render/Railway use DATABASE_URL
            # Fix for SQLAlchemy 2.0 (postgres:// -> postgresql://)
            if postgres_url.startswith("postgres://"):
                postgres_url = postgres_url.replace("postgres://", "postgresql://", 1)
            return postgres_url
        return "sqlite:///data/insider_scanner.db"
    else:
        return "sqlite:///data/insider_scanner.db"


@dataclass
class AlertChannelConfig:
    """Configuration for a single alert channel."""

    enabled: bool = False
    min_priority: str = "medium"

    # Discord specific
    discord_webhook_url: Optional[str] = None
    discord_mention_role: Optional[str] = None

    # Email specific
    email_provider: str = "smtp"  # smtp or sendgrid
    email_recipients: List[str] = field(default_factory=list)

    # File specific
    file_path: Optional[str] = None


def get_alert_channels(settings: Optional[InsiderScannerSettings] = None) -> dict:
    """Get configured alert channels.

    Args:
        settings: Optional settings instance

    Returns:
        Dict of channel name -> AlertChannelConfig
    """
    if settings is None:
        settings = get_settings()

    channels = {}

    # Discord
    if settings.alert_discord_enabled and settings.alert_discord_webhook_url:
        channels["discord"] = AlertChannelConfig(
            enabled=True,
            min_priority=settings.alert_min_priority,
            discord_webhook_url=settings.alert_discord_webhook_url,
            discord_mention_role=settings.alert_discord_mention_role,
        )

    # Email
    if settings.alert_email_enabled:
        channels["email"] = AlertChannelConfig(
            enabled=True,
            min_priority=settings.alert_min_priority,
            email_provider="sendgrid" if settings.alert_email_sendgrid_api_key else "smtp",
            email_recipients=settings.alert_email_to_addresses,
        )

    # File (always available as audit log)
    channels["file"] = AlertChannelConfig(
        enabled=settings.alert_file_enabled,
        min_priority="low",  # Log all for audit
        file_path=settings.alert_file_path,
    )

    return channels


def ensure_data_directory():
    """Ensure the data directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_health_check_config() -> dict:
    """Get health check configuration.

    Returns:
        Dict with health check settings
    """
    settings = get_settings()

    return {
        "path": "/api/health",
        "interval": 30,
        "timeout": 10,
        "checks": [
            {"name": "database", "type": "database_ping"},
            {"name": "websocket", "type": "websocket_connected", "enabled": settings.websocket_enabled},
            {"name": "alerts", "type": "alert_channel_configured"},
        ],
    }


def validate_config() -> List[str]:
    """Validate current configuration.

    Returns:
        List of warning messages (empty if all valid)
    """
    settings = get_settings()
    warnings = []

    # Check database URL
    if "sqlite:///:memory:" in settings.database_url:
        warnings.append("Using in-memory database - data will not persist")

    # Check alert configuration
    if not settings.alert_discord_enabled and not settings.alert_email_enabled:
        warnings.append("No real-time alerts configured (Discord/Email disabled)")

    # Check scoring thresholds
    if settings.score_critical_threshold <= settings.score_high_threshold:
        warnings.append("Critical threshold should be higher than high threshold")

    # Check anchor configuration
    if settings.anchor_enabled and settings.anchor_type == "polygon":
        if not settings.anchor_polygon_rpc_url:
            warnings.append("Polygon anchoring enabled but RPC URL not configured")
        if not settings.anchor_private_key:
            warnings.append("Polygon anchoring enabled but private key not configured")

    return warnings


# Export commonly used configuration
__all__ = [
    "InsiderScannerSettings",
    "DeploymentMode",
    "AlertChannelConfig",
    "get_settings",
    "get_database_url",
    "get_alert_channels",
    "detect_deployment_mode",
    "ensure_data_directory",
    "get_health_check_config",
    "validate_config",
]
