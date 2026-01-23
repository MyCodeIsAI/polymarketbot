"""Pydantic models for all configuration.

This module defines strongly-typed configuration models with validation.
All sensitive data (private keys, API secrets) are loaded from environment
variables and never stored in config files.
"""

from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class SignatureType(int, Enum):
    """Polymarket wallet signature types."""

    EOA = 0  # Standard Ethereum wallet (MetaMask, hardware wallet)
    POLY_PROXY = 1  # MagicLink / email wallet proxy
    GNOSIS_SAFE = 2  # Browser wallet Gnosis Safe proxy


class SlippageAction(str, Enum):
    """Action to take when slippage exceeds threshold."""

    SKIP = "skip"  # Skip the trade entirely
    LIMIT_ORDER = "limit_order"  # Place passive limit order at target price
    EXECUTE_ANYWAY = "execute_anyway"  # Execute regardless (dangerous)


class TargetAccount(BaseModel):
    """Configuration for an account to copy-trade from.

    Each target account represents a wallet whose positions we want to mirror
    with configurable position sizing and risk parameters.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Friendly name for this target (e.g., 'whale_trader_1')",
    )
    wallet: str = Field(
        ...,
        description="Target's Polymarket proxy wallet address (0x...)",
    )
    enabled: bool = Field(
        default=True,
        description="Whether to actively monitor this account",
    )
    position_ratio: Decimal = Field(
        ...,
        gt=0,
        le=10,
        description="Ratio of target's position to copy (0.01 = 1/100th)",
    )
    max_position_usd: Decimal = Field(
        ...,
        gt=0,
        description="Maximum USD value per position for this target",
    )
    slippage_tolerance: Decimal = Field(
        default=Decimal("0.05"),
        ge=0,
        le=1,
        description="Maximum allowed slippage (0.05 = 5%)",
    )
    slippage_action: SlippageAction = Field(
        default=SlippageAction.SKIP,
        description="What to do when slippage exceeds threshold",
    )
    min_position_usd: Decimal = Field(
        default=Decimal("5"),
        ge=0,
        description="Ignore target positions smaller than this USD value",
    )
    min_copy_size_usd: Decimal = Field(
        default=Decimal("1"),
        ge=0,
        description="Minimum copy order size in USD (positions smaller than this won't be copied)",
    )

    @field_validator("wallet")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Ethereum address format."""
        if not re.match(r"^0x[a-fA-F0-9]{40}$", v):
            raise ValueError(f"Invalid Ethereum address format: {v}")
        return v.lower()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is filesystem and display safe."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v


class YourAccount(BaseModel):
    """Configuration for your trading account.

    This is the account that will execute copy-trades. The private key
    is loaded from an environment variable for security.
    """

    private_key_env: str = Field(
        default="POLYBOT_PRIVATE_KEY",
        description="Environment variable name containing the private key",
    )
    proxy_wallet: str = Field(
        ...,
        description="Your Polymarket proxy wallet address (shown on polymarket.com)",
    )
    signature_type: SignatureType = Field(
        default=SignatureType.GNOSIS_SAFE,
        description="Wallet signature type (most users = GNOSIS_SAFE)",
    )

    @field_validator("proxy_wallet")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Ethereum address format."""
        if not re.match(r"^0x[a-fA-F0-9]{40}$", v):
            raise ValueError(f"Invalid Ethereum address format: {v}")
        return v.lower()


class APICredentials(BaseModel):
    """Polymarket API credentials.

    These are derived from your private key and stored securely.
    The actual values come from environment variables.
    """

    api_key_env: str = Field(
        default="POLYBOT_API_KEY",
        description="Environment variable for API key",
    )
    api_secret_env: str = Field(
        default="POLYBOT_API_SECRET",
        description="Environment variable for API secret (base64)",
    )
    api_passphrase_env: str = Field(
        default="POLYBOT_API_PASSPHRASE",
        description="Environment variable for API passphrase",
    )


class PollingConfig(BaseModel):
    """Configuration for API polling intervals.

    These values are tuned to balance speed vs. rate limit compliance.
    """

    activity_interval_ms: int = Field(
        default=200,
        ge=100,
        le=5000,
        description="Milliseconds between /activity polls per target",
    )
    positions_sync_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds between full position reconciliation",
    )
    balance_check_interval_s: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Seconds between balance checks",
    )
    orderbook_cache_ms: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Order book cache TTL in milliseconds",
    )


class ExecutionConfig(BaseModel):
    """Configuration for order execution."""

    order_timeout_s: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Seconds to wait for order fill before timeout",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum order submission retries",
    )
    retry_delay_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Delay between retries in milliseconds",
    )
    order_expiry_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Minutes until unfilled limit orders expire (GTD)",
    )
    use_fill_or_kill: bool = Field(
        default=False,
        description="Use FOK orders (all-or-nothing execution)",
    )


class SafetyConfig(BaseModel):
    """Safety limits and circuit breakers."""

    max_daily_loss_usd: Decimal = Field(
        default=Decimal("1000"),
        gt=0,
        description="Maximum daily loss before circuit breaker trips",
    )
    max_open_positions: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum concurrent open positions",
    )
    require_liquidity_check: bool = Field(
        default=True,
        description="Check order book depth before executing",
    )
    min_book_depth_usd: Decimal = Field(
        default=Decimal("100"),
        ge=0,
        description="Minimum liquidity in order book to trade",
    )
    max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Consecutive failures before pausing",
    )
    min_balance_usd: Decimal = Field(
        default=Decimal("50"),
        ge=0,
        description="Minimum USDC balance to maintain (pause if below)",
    )
    max_position_drift_percent: Decimal = Field(
        default=Decimal("0.2"),
        ge=0,
        le=1,
        description="Maximum allowed drift from target ratio before alert",
    )


class NetworkConfig(BaseModel):
    """Network and connection settings."""

    api_timeout_s: int = Field(
        default=10,
        ge=1,
        le=60,
        description="HTTP request timeout in seconds",
    )
    connection_pool_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="HTTP connection pool size",
    )
    websocket_ping_interval_s: int = Field(
        default=5,
        ge=1,
        le=30,
        description="WebSocket ping interval in seconds",
    )
    websocket_reconnect_delay_s: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Delay before WebSocket reconnection attempt",
    )
    max_websocket_reconnects: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum consecutive reconnection attempts",
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url_env: str = Field(
        default="POLYBOT_DATABASE_URL",
        description="Environment variable for database URL",
    )
    default_url: str = Field(
        default="sqlite+aiosqlite:///polymarketbot.db",
        description="Default database URL if env var not set",
    )
    echo_sql: bool = Field(
        default=False,
        description="Log SQL queries (debug only)",
    )


class APIEndpoints(BaseModel):
    """Polymarket API endpoints.

    These are the official Polymarket API URLs. Do not modify unless
    Polymarket changes their infrastructure.
    """

    clob_http: str = Field(
        default="https://clob.polymarket.com",
        description="CLOB REST API base URL",
    )
    clob_ws: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/",
        description="CLOB WebSocket base URL",
    )
    data_api: str = Field(
        default="https://data-api.polymarket.com",
        description="Data API base URL",
    )
    gamma_api: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Gamma API base URL",
    )
    polygon_rpc_env: str = Field(
        default="POLYBOT_POLYGON_RPC",
        description="Environment variable for Polygon RPC URL",
    )
    default_polygon_rpc: str = Field(
        default="https://polygon-rpc.com",
        description="Default Polygon RPC (public, rate-limited)",
    )


class AppConfig(BaseModel):
    """Root application configuration.

    This is the main configuration object that contains all settings.
    Load using the `load_config()` function.
    """

    your_account: YourAccount
    api_credentials: APICredentials = Field(default_factory=APICredentials)
    targets: list[TargetAccount] = Field(
        default_factory=list,
        max_length=10,
        description="Target accounts to copy-trade (max 10)",
    )
    polling: PollingConfig = Field(default_factory=PollingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    endpoints: APIEndpoints = Field(default_factory=APIEndpoints)

    @model_validator(mode="after")
    def validate_targets(self) -> "AppConfig":
        """Ensure target names are unique."""
        names = [t.name for t in self.targets]
        if len(names) != len(set(names)):
            raise ValueError("Target account names must be unique")
        return self

    @property
    def enabled_targets(self) -> list[TargetAccount]:
        """Get only enabled target accounts."""
        return [t for t in self.targets if t.enabled]

    @property
    def target_wallets(self) -> list[str]:
        """Get list of all target wallet addresses."""
        return [t.wallet for t in self.targets]

    def get_target_by_name(self, name: str) -> Optional[TargetAccount]:
        """Look up a target account by name."""
        for target in self.targets:
            if target.name == name:
                return target
        return None

    def get_target_by_wallet(self, wallet: str) -> Optional[TargetAccount]:
        """Look up a target account by wallet address."""
        wallet_lower = wallet.lower()
        for target in self.targets:
            if target.wallet == wallet_lower:
                return target
        return None
