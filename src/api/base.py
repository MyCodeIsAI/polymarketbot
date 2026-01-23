"""Base async HTTP client with retry logic and connection pooling.

This module provides the foundation for all API clients, handling:
- Connection pooling for performance
- Automatic retries with exponential backoff
- Rate limit handling (429 responses)
- Request/response logging
- Timeout management
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Generic
from urllib.parse import urljoin

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
)

from ..core.exceptions import APIError, RateLimitError, AuthenticationError
from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class APIResponse(Generic[T]):
    """Wrapper for API responses with metadata."""

    data: T
    status_code: int
    latency_ms: float
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


class RetryableError(Exception):
    """Marker exception for errors that should trigger retry."""

    pass


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    if retry_state.attempt_number > 1:
        logger.warning(
            "api_retry",
            attempt=retry_state.attempt_number,
            wait_seconds=retry_state.next_action.sleep if retry_state.next_action else 0,
        )


class BaseAPIClient:
    """Base class for async API clients.

    Provides connection pooling, retries, and common request handling.
    Subclasses should implement endpoint-specific methods.

    Example:
        async with BaseAPIClient("https://api.example.com") as client:
            response = await client.get("/endpoint")
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 10.0,
        max_retries: int = 3,
        pool_size: int = 20,
        rate_limiter: Optional["RateLimiter"] = None,
    ):
        """Initialize the API client.

        Args:
            base_url: Base URL for all requests
            timeout_s: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            pool_size: HTTP connection pool size
            rate_limiter: Optional rate limiter instance
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.pool_size = pool_size
        self.rate_limiter = rate_limiter

        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._error_count = 0

    async def __aenter__(self) -> "BaseAPIClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=self.pool_size,
                max_connections=self.pool_size + 10,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                limits=limits,
                http2=True,  # Enable HTTP/2 for better performance
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith("http"):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for request. Override in subclasses for auth."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_error_response(
        self,
        response: httpx.Response,
        url: str,
    ) -> None:
        """Handle error responses and raise appropriate exceptions."""
        status = response.status_code

        # Rate limit
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            retry_ms = int(float(retry_after) * 1000) if retry_after else 1000
            raise RateLimitError(
                "Rate limit exceeded",
                endpoint=url,
                status_code=status,
                retry_after_ms=retry_ms,
            )

        # Authentication error
        if status in (401, 403):
            raise AuthenticationError(
                f"Authentication failed: {response.text}",
                endpoint=url,
                status_code=status,
                response_body=response.text,
            )

        # Server errors (retryable)
        if status >= 500:
            raise RetryableError(f"Server error {status}: {response.text}")

        # Other client errors
        if status >= 400:
            raise APIError(
                f"API error {status}: {response.text}",
                endpoint=url,
                status_code=status,
                response_body=response.text,
            )

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        skip_rate_limit: bool = False,
    ) -> APIResponse[dict[str, Any]]:
        """Execute an HTTP request with retries.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: URL path (relative to base_url)
            params: Query parameters
            json_data: JSON body data
            headers: Additional headers (merged with default)
            skip_rate_limit: Skip rate limiting for this request

        Returns:
            APIResponse with parsed JSON data

        Raises:
            APIError: For API errors
            RateLimitError: When rate limited
            AuthenticationError: For auth failures
        """
        client = await self._ensure_client()
        url = self._build_url(path)

        # Apply rate limiting
        if self.rate_limiter and not skip_rate_limit:
            await self.rate_limiter.acquire(path)

        # Build headers
        request_headers = await self._get_headers()
        if headers:
            request_headers.update(headers)

        # Clean up params (remove None values)
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        start_time = time.perf_counter()
        self._request_count += 1

        try:
            response = await self._request_with_retry(
                client, method, url, params, json_data, request_headers
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "api_request",
                method=method,
                path=path,
                status=response.status_code,
                latency_ms=round(latency_ms, 2),
            )

            # Handle error responses
            if response.status_code >= 400:
                self._handle_error_response(response, url)

            # Parse response
            try:
                data = response.json() if response.text else {}
            except Exception:
                data = {"raw": response.text}

            return APIResponse(
                data=data,
                status_code=response.status_code,
                latency_ms=latency_ms,
                headers=dict(response.headers),
            )

        except (RateLimitError, AuthenticationError, APIError):
            self._error_count += 1
            raise
        except RetryableError as e:
            self._error_count += 1
            raise APIError(str(e), endpoint=url)
        except httpx.TimeoutException:
            self._error_count += 1
            raise APIError(
                f"Request timed out after {self.timeout_s}s",
                endpoint=url,
            )
        except httpx.ConnectError as e:
            self._error_count += 1
            raise APIError(f"Connection failed: {e}", endpoint=url)
        except Exception as e:
            self._error_count += 1
            raise APIError(f"Unexpected error: {e}", endpoint=url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(RetryableError),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        params: Optional[dict[str, Any]],
        json_data: Optional[dict[str, Any]],
        headers: dict[str, str],
    ) -> httpx.Response:
        """Execute request with retry logic for server errors."""
        response = await client.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=headers,
        )

        # Trigger retry on server errors
        if response.status_code >= 500:
            raise RetryableError(f"Server error {response.status_code}")

        return response

    # Convenience methods

    async def get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> APIResponse[dict[str, Any]]:
        """Execute GET request."""
        return await self._request("GET", path, params=params, **kwargs)

    async def post(
        self,
        path: str,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> APIResponse[dict[str, Any]]:
        """Execute POST request."""
        return await self._request("POST", path, params=params, json_data=json_data, **kwargs)

    async def delete(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> APIResponse[dict[str, Any]]:
        """Execute DELETE request."""
        return await self._request("DELETE", path, params=params, json_data=json_data, **kwargs)

    @property
    def stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
        }


# Import RateLimiter here to avoid circular imports
from .rate_limiter import RateLimiter  # noqa: E402, F401
