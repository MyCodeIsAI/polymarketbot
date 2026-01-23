"""API connectivity testing for Polymarket endpoints.

This module provides functions to test connectivity to all Polymarket
API endpoints, useful for verifying network configuration and VPN setup.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

from ..config.models import APIEndpoints
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EndpointStatus(str, Enum):
    """Status of an API endpoint."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Geo-blocked


@dataclass
class ConnectivityResult:
    """Result of a connectivity test."""

    endpoint_name: str
    url: str
    status: EndpointStatus
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.status == EndpointStatus.OK

    def __str__(self) -> str:
        if self.is_ok:
            return f"{self.endpoint_name}: OK ({self.latency_ms:.0f}ms)"
        return f"{self.endpoint_name}: {self.status.value} - {self.error_message}"


async def test_endpoint(
    name: str,
    url: str,
    path: str = "",
    timeout_s: float = 10.0,
) -> ConnectivityResult:
    """Test connectivity to a single endpoint.

    Args:
        name: Friendly name for the endpoint
        url: Base URL of the endpoint
        path: Optional path to append (for specific endpoint testing)
        timeout_s: Request timeout in seconds

    Returns:
        ConnectivityResult with status and latency
    """
    full_url = f"{url}{path}"

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        start_time = time.perf_counter()

        try:
            response = await client.get(full_url)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Check for geo-blocking (Polymarket returns specific responses)
            if response.status_code == 403:
                # Check if it's a geo-block vs auth issue
                body = response.text.lower()
                if "blocked" in body or "region" in body or "country" in body:
                    return ConnectivityResult(
                        endpoint_name=name,
                        url=full_url,
                        status=EndpointStatus.BLOCKED,
                        latency_ms=latency_ms,
                        status_code=403,
                        error_message="Geo-blocked - check VPN configuration",
                    )

            # 2xx and some 4xx are OK (4xx might just mean we need auth)
            if response.status_code < 500:
                return ConnectivityResult(
                    endpoint_name=name,
                    url=full_url,
                    status=EndpointStatus.OK,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            else:
                return ConnectivityResult(
                    endpoint_name=name,
                    url=full_url,
                    status=EndpointStatus.ERROR,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error_message=f"Server error: {response.status_code}",
                )

        except httpx.TimeoutException:
            return ConnectivityResult(
                endpoint_name=name,
                url=full_url,
                status=EndpointStatus.TIMEOUT,
                error_message=f"Request timed out after {timeout_s}s",
            )
        except httpx.ConnectError as e:
            return ConnectivityResult(
                endpoint_name=name,
                url=full_url,
                status=EndpointStatus.ERROR,
                error_message=f"Connection failed: {e}",
            )
        except Exception as e:
            return ConnectivityResult(
                endpoint_name=name,
                url=full_url,
                status=EndpointStatus.ERROR,
                error_message=str(e),
            )


async def test_websocket_endpoint(
    name: str,
    url: str,
    timeout_s: float = 10.0,
) -> ConnectivityResult:
    """Test connectivity to a WebSocket endpoint.

    This performs a basic connection test without subscribing to channels.

    Args:
        name: Friendly name for the endpoint
        url: WebSocket URL
        timeout_s: Connection timeout in seconds

    Returns:
        ConnectivityResult with status and latency
    """
    try:
        import websockets
    except ImportError:
        return ConnectivityResult(
            endpoint_name=name,
            url=url,
            status=EndpointStatus.ERROR,
            error_message="websockets library not installed",
        )

    start_time = time.perf_counter()

    try:
        async with asyncio.timeout(timeout_s):
            async with websockets.connect(url) as ws:
                latency_ms = (time.perf_counter() - start_time) * 1000
                await ws.close()

                return ConnectivityResult(
                    endpoint_name=name,
                    url=url,
                    status=EndpointStatus.OK,
                    latency_ms=latency_ms,
                )

    except asyncio.TimeoutError:
        return ConnectivityResult(
            endpoint_name=name,
            url=url,
            status=EndpointStatus.TIMEOUT,
            error_message=f"Connection timed out after {timeout_s}s",
        )
    except Exception as e:
        return ConnectivityResult(
            endpoint_name=name,
            url=url,
            status=EndpointStatus.ERROR,
            error_message=str(e),
        )


async def test_all_endpoints(
    endpoints: Optional[APIEndpoints] = None,
    timeout_s: float = 10.0,
) -> list[ConnectivityResult]:
    """Test connectivity to all Polymarket API endpoints.

    Args:
        endpoints: API endpoint configuration. Uses defaults if not provided.
        timeout_s: Request timeout in seconds

    Returns:
        List of ConnectivityResult for each endpoint
    """
    if endpoints is None:
        endpoints = APIEndpoints()

    # Define test cases: (name, url, path)
    http_tests = [
        ("CLOB API", endpoints.clob_http, "/"),
        ("CLOB Markets", endpoints.clob_http, "/markets"),
        ("Data API", endpoints.data_api, "/"),
        ("Gamma API", endpoints.gamma_api, "/events"),
    ]

    ws_tests = [
        ("CLOB WebSocket", endpoints.clob_ws + "market"),
    ]

    # Run HTTP tests in parallel
    http_tasks = [
        test_endpoint(name, url, path, timeout_s)
        for name, url, path in http_tests
    ]

    # Run WebSocket tests
    ws_tasks = [
        test_websocket_endpoint(name, url, timeout_s)
        for name, url in ws_tests
    ]

    results = await asyncio.gather(*http_tasks, *ws_tasks)

    # Log results
    for result in results:
        if result.is_ok:
            logger.info(
                "endpoint_test_passed",
                endpoint=result.endpoint_name,
                latency_ms=round(result.latency_ms or 0, 1),
            )
        else:
            logger.warning(
                "endpoint_test_failed",
                endpoint=result.endpoint_name,
                status=result.status.value,
                error=result.error_message,
            )

    return list(results)


async def test_polygon_rpc(
    rpc_url: str,
    timeout_s: float = 10.0,
) -> ConnectivityResult:
    """Test connectivity to a Polygon RPC endpoint.

    Args:
        rpc_url: The Polygon RPC URL to test
        timeout_s: Request timeout in seconds

    Returns:
        ConnectivityResult with status and latency
    """
    # JSON-RPC request to get chain ID (lightweight test)
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_chainId",
        "params": [],
        "id": 1,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        start_time = time.perf_counter()

        try:
            response = await client.post(rpc_url, json=payload)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                chain_id = data.get("result")

                # Polygon mainnet chain ID is 0x89 (137)
                if chain_id == "0x89":
                    return ConnectivityResult(
                        endpoint_name="Polygon RPC",
                        url=rpc_url,
                        status=EndpointStatus.OK,
                        latency_ms=latency_ms,
                        status_code=200,
                    )
                else:
                    return ConnectivityResult(
                        endpoint_name="Polygon RPC",
                        url=rpc_url,
                        status=EndpointStatus.ERROR,
                        latency_ms=latency_ms,
                        error_message=f"Wrong chain ID: {chain_id} (expected 0x89 for Polygon)",
                    )
            else:
                return ConnectivityResult(
                    endpoint_name="Polygon RPC",
                    url=rpc_url,
                    status=EndpointStatus.ERROR,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error_message=f"HTTP {response.status_code}",
                )

        except httpx.TimeoutException:
            return ConnectivityResult(
                endpoint_name="Polygon RPC",
                url=rpc_url,
                status=EndpointStatus.TIMEOUT,
                error_message=f"Request timed out after {timeout_s}s",
            )
        except Exception as e:
            return ConnectivityResult(
                endpoint_name="Polygon RPC",
                url=rpc_url,
                status=EndpointStatus.ERROR,
                error_message=str(e),
            )


def format_connectivity_results(results: list[ConnectivityResult]) -> str:
    """Format connectivity results for display.

    Args:
        results: List of ConnectivityResult

    Returns:
        Formatted string for display
    """
    lines = ["Polymarket API Connectivity Test Results", "=" * 45, ""]

    ok_count = sum(1 for r in results if r.is_ok)
    total = len(results)

    for result in results:
        if result.is_ok:
            status_icon = "[OK]"
            latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
            lines.append(f"{status_icon} {result.endpoint_name}: {latency}")
        else:
            status_icon = "[!!]" if result.status == EndpointStatus.BLOCKED else "[XX]"
            lines.append(f"{status_icon} {result.endpoint_name}: {result.error_message}")

    lines.append("")
    lines.append(f"Summary: {ok_count}/{total} endpoints reachable")

    if ok_count < total:
        lines.append("")
        lines.append("Troubleshooting tips:")
        if any(r.status == EndpointStatus.BLOCKED for r in results):
            lines.append("  - Geo-blocked: Verify VPN is connected and exit node is in allowed region")
            lines.append("  - Recommended exit nodes: Amsterdam, Frankfurt, or other EU locations")
        if any(r.status == EndpointStatus.TIMEOUT for r in results):
            lines.append("  - Timeout: Check internet connection and VPN stability")
        if any(r.status == EndpointStatus.ERROR for r in results):
            lines.append("  - Error: Check firewall settings and DNS resolution")

    return "\n".join(lines)
