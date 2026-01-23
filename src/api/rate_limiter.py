"""Token bucket rate limiter for API requests.

This module implements a token bucket algorithm for rate limiting,
with support for:
- Per-endpoint rate limits
- Burst capacity
- Sustained rate limits
- Automatic token replenishment
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit bucket.

    Attributes:
        requests_per_window: Maximum requests allowed in the window
        window_seconds: Time window in seconds
        burst_limit: Optional burst limit (higher than sustained)
        burst_window_seconds: Burst window in seconds
    """

    requests_per_window: int
    window_seconds: float = 10.0
    burst_limit: Optional[int] = None
    burst_window_seconds: float = 1.0

    @property
    def tokens_per_second(self) -> float:
        """Calculate token replenishment rate."""
        return self.requests_per_window / self.window_seconds


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Uses a sliding window approach with token replenishment.
    """

    config: RateLimitConfig
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize bucket with full tokens."""
        self.tokens = float(self.config.requests_per_window)
        self.last_update = time.monotonic()

    def _replenish(self) -> None:
        """Replenish tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.config.tokens_per_second
        self.tokens = min(
            self.tokens + new_tokens,
            float(self.config.requests_per_window),
        )

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            self._replenish()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.config.tokens_per_second

            return wait_time

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (without locking)."""
        elapsed = time.monotonic() - self.last_update
        new_tokens = elapsed * self.config.tokens_per_second
        return min(
            self.tokens + new_tokens,
            float(self.config.requests_per_window),
        )


# Polymarket API rate limits (from documentation)
POLYMARKET_RATE_LIMITS: dict[str, RateLimitConfig] = {
    # CLOB API limits
    "clob_general": RateLimitConfig(requests_per_window=15000, window_seconds=10),
    "clob_book": RateLimitConfig(requests_per_window=1500, window_seconds=10),
    "clob_books": RateLimitConfig(requests_per_window=500, window_seconds=10),
    "clob_price": RateLimitConfig(requests_per_window=1500, window_seconds=10),
    "clob_prices": RateLimitConfig(requests_per_window=500, window_seconds=10),
    "clob_order_post": RateLimitConfig(
        requests_per_window=36000,
        window_seconds=600,  # 10 minutes
        burst_limit=3500,
        burst_window_seconds=10,
    ),
    "clob_order_delete": RateLimitConfig(
        requests_per_window=30000,
        window_seconds=600,
        burst_limit=3000,
        burst_window_seconds=10,
    ),
    "clob_orders_batch": RateLimitConfig(
        requests_per_window=15000,
        window_seconds=600,
        burst_limit=1000,
        burst_window_seconds=10,
    ),
    "clob_cancel_all": RateLimitConfig(
        requests_per_window=6000,
        window_seconds=600,
        burst_limit=250,
        burst_window_seconds=10,
    ),

    # Data API limits
    "data_general": RateLimitConfig(requests_per_window=1000, window_seconds=10),
    "data_trades": RateLimitConfig(requests_per_window=200, window_seconds=10),
    "data_positions": RateLimitConfig(requests_per_window=150, window_seconds=10),

    # Gamma API limits
    "gamma_general": RateLimitConfig(requests_per_window=4000, window_seconds=10),
    "gamma_events": RateLimitConfig(requests_per_window=500, window_seconds=10),
    "gamma_markets": RateLimitConfig(requests_per_window=300, window_seconds=10),
}


class RateLimiter:
    """Rate limiter with per-endpoint buckets.

    Manages multiple token buckets for different API endpoints,
    automatically applying the correct rate limits.

    Example:
        limiter = RateLimiter()
        await limiter.acquire("/book")  # Uses clob_book limit
        await limiter.acquire("/positions")  # Uses data_positions limit
    """

    def __init__(
        self,
        limits: Optional[dict[str, RateLimitConfig]] = None,
        safety_margin: float = 0.9,
    ):
        """Initialize rate limiter.

        Args:
            limits: Custom rate limit configurations
            safety_margin: Use this fraction of actual limits (0.9 = 90%)
        """
        self.limits = limits or POLYMARKET_RATE_LIMITS
        self.safety_margin = safety_margin
        self._buckets: dict[str, TokenBucket] = {}
        self._path_to_bucket: dict[str, str] = self._build_path_mapping()

    def _build_path_mapping(self) -> dict[str, str]:
        """Build mapping from URL paths to bucket names."""
        return {
            # CLOB API
            "/book": "clob_book",
            "/books": "clob_books",
            "/price": "clob_price",
            "/prices": "clob_prices",
            "/order": "clob_order_post",  # POST
            "/orders": "clob_orders_batch",
            "/cancel-all": "clob_cancel_all",

            # Data API
            "/trades": "data_trades",
            "/positions": "data_positions",
            "/activity": "data_general",

            # Gamma API
            "/events": "gamma_events",
            "/markets": "gamma_markets",
        }

    def _get_bucket_name(self, path: str) -> str:
        """Get the bucket name for a path."""
        # Check exact matches first
        for prefix, bucket_name in self._path_to_bucket.items():
            if path.startswith(prefix) or path.endswith(prefix):
                return bucket_name

        # Determine API type from path patterns
        if "clob" in path.lower() or path in ["/midpoint", "/spread", "/tick-size"]:
            return "clob_general"
        elif any(x in path.lower() for x in ["position", "trade", "activity", "holder"]):
            return "data_general"
        elif any(x in path.lower() for x in ["event", "market", "gamma"]):
            return "gamma_general"

        # Default to most restrictive
        return "data_general"

    def _get_bucket(self, bucket_name: str) -> TokenBucket:
        """Get or create a token bucket."""
        if bucket_name not in self._buckets:
            config = self.limits.get(bucket_name, self.limits["data_general"])

            # Apply safety margin
            adjusted_config = RateLimitConfig(
                requests_per_window=int(config.requests_per_window * self.safety_margin),
                window_seconds=config.window_seconds,
                burst_limit=int(config.burst_limit * self.safety_margin) if config.burst_limit else None,
                burst_window_seconds=config.burst_window_seconds,
            )

            self._buckets[bucket_name] = TokenBucket(config=adjusted_config)

        return self._buckets[bucket_name]

    async def acquire(self, path: str, tokens: int = 1) -> None:
        """Acquire rate limit tokens for a request.

        Blocks if rate limit would be exceeded.

        Args:
            path: API endpoint path
            tokens: Number of tokens to acquire (usually 1)
        """
        bucket_name = self._get_bucket_name(path)
        bucket = self._get_bucket(bucket_name)

        wait_time = await bucket.acquire(tokens)

        if wait_time > 0:
            logger.debug(
                "rate_limit_wait",
                bucket=bucket_name,
                path=path,
                wait_seconds=round(wait_time, 3),
            )
            await asyncio.sleep(wait_time)

            # Try again after waiting
            await bucket.acquire(tokens)

    def get_bucket_status(self, bucket_name: str) -> dict:
        """Get status of a specific bucket."""
        if bucket_name not in self._buckets:
            return {"exists": False}

        bucket = self._buckets[bucket_name]
        return {
            "exists": True,
            "available_tokens": bucket.available_tokens,
            "max_tokens": bucket.config.requests_per_window,
            "tokens_per_second": bucket.config.tokens_per_second,
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all active buckets."""
        return {name: self.get_bucket_status(name) for name in self._buckets}

    def reset(self) -> None:
        """Reset all buckets to full capacity."""
        self._buckets.clear()


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on 429 responses.

    Automatically reduces limits when rate limit errors are encountered
    and gradually increases them when requests succeed.
    """

    def __init__(
        self,
        limits: Optional[dict[str, RateLimitConfig]] = None,
        safety_margin: float = 0.8,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.05,
    ):
        """Initialize adaptive rate limiter.

        Args:
            limits: Custom rate limit configurations
            safety_margin: Initial safety margin
            backoff_factor: Multiply limit by this on 429
            recovery_factor: Multiply limit by this on success (up to original)
        """
        super().__init__(limits, safety_margin)
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self._original_margin = safety_margin
        self._bucket_margins: dict[str, float] = {}

    def report_rate_limit(self, path: str) -> None:
        """Report a rate limit error for adaptive adjustment."""
        bucket_name = self._get_bucket_name(path)

        current_margin = self._bucket_margins.get(bucket_name, self.safety_margin)
        new_margin = current_margin * self.backoff_factor
        self._bucket_margins[bucket_name] = max(new_margin, 0.1)  # Floor at 10%

        logger.warning(
            "rate_limit_backoff",
            bucket=bucket_name,
            new_margin=round(new_margin, 3),
        )

        # Force bucket recreation with new margin
        if bucket_name in self._buckets:
            del self._buckets[bucket_name]

    def report_success(self, path: str) -> None:
        """Report successful request for adaptive recovery."""
        bucket_name = self._get_bucket_name(path)

        if bucket_name in self._bucket_margins:
            current_margin = self._bucket_margins[bucket_name]
            new_margin = min(current_margin * self.recovery_factor, self._original_margin)
            self._bucket_margins[bucket_name] = new_margin
