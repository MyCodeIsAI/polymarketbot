"""
Proxy configuration and high-performance HTTP session management.

This module provides:
- SOCKS5/HTTP proxy support for geo-restricted trading
- Connection pooling and keep-alive for low latency
- Latency benchmarking utilities
- Recommended proxy/VPS configuration based on Polymarket infrastructure

POLYMARKET INFRASTRUCTURE (as of 2025):
- Primary servers: AWS eu-west-2 (London, UK)
- Backup servers: AWS eu-west-1 (Ireland)
- CLOB API: https://clob.polymarket.com
- WebSocket: wss://ws-subscriptions-clob.polymarket.com
- Data API: https://data-api.polymarket.com

OPTIMAL VPS LOCATIONS (by latency to Polymarket):
1. Amsterdam, Netherlands (~2-5ms) - RECOMMENDED, not geo-blocked
2. Frankfurt, Germany (~8-12ms) - Good alternative
3. Paris, France (~10-15ms) - Acceptable
4. Dublin, Ireland (~5-8ms) - Close but check geo-restrictions

AVOID:
- London, UK - Geo-blocked despite being closest
- United States - Geo-blocked
- Singapore/Asia - High latency (100ms+)
"""

import os
import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Try to import optional SOCKS support
try:
    import socks
    import socket
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False

try:
    from aiohttp_socks import ProxyConnector, ProxyType
    AIOHTTP_SOCKS_AVAILABLE = True
except ImportError:
    AIOHTTP_SOCKS_AVAILABLE = False


class ProxyType(Enum):
    """Supported proxy types."""
    NONE = "none"
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"
    SOCKS5H = "socks5h"  # DNS resolution on proxy server


@dataclass
class ProxyConfig:
    """Proxy configuration for API requests."""

    enabled: bool = False
    proxy_type: ProxyType = ProxyType.NONE
    host: str = ""
    port: int = 0
    username: Optional[str] = None
    password: Optional[str] = None

    # Connection tuning
    timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_backoff: float = 0.5

    # Connection pooling
    pool_connections: int = 10
    pool_maxsize: int = 20
    keep_alive: bool = True

    def get_proxy_url(self) -> Optional[str]:
        """Get the proxy URL string."""
        if not self.enabled or self.proxy_type == ProxyType.NONE:
            return None

        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"

        return f"{self.proxy_type.value}://{auth}{self.host}:{self.port}"

    def get_requests_proxies(self) -> Optional[Dict[str, str]]:
        """Get proxy dict for requests library."""
        url = self.get_proxy_url()
        if not url:
            return None
        return {"http": url, "https": url}

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "enabled": self.enabled,
            "proxy_type": self.proxy_type.value,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***" if self.password else None,  # Don't expose password
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "pool_connections": self.pool_connections,
            "pool_maxsize": self.pool_maxsize,
            "keep_alive": self.keep_alive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProxyConfig":
        """Deserialize from dict."""
        proxy_type = data.get("proxy_type", "none")
        if isinstance(proxy_type, str):
            proxy_type = ProxyType(proxy_type)

        return cls(
            enabled=data.get("enabled", False),
            proxy_type=proxy_type,
            host=data.get("host", ""),
            port=data.get("port", 0),
            username=data.get("username"),
            password=data.get("password"),
            timeout_seconds=data.get("timeout_seconds", 5.0),
            max_retries=data.get("max_retries", 3),
            pool_connections=data.get("pool_connections", 10),
            pool_maxsize=data.get("pool_maxsize", 20),
            keep_alive=data.get("keep_alive", True),
        )


@dataclass
class LatencyBenchmark:
    """Results from a latency benchmark."""

    endpoint: str
    samples: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else 0

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples) if self.samples else 0

    @property
    def p95_ms(self) -> float:
        if len(self.samples) < 20:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        return sorted_samples[int(len(sorted_samples) * 0.95)]

    @property
    def p99_ms(self) -> float:
        if len(self.samples) < 100:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        return sorted_samples[int(len(sorted_samples) * 0.99)]

    @property
    def success_rate(self) -> float:
        total = len(self.samples) + self.errors
        return (len(self.samples) / total * 100) if total > 0 else 0

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "samples": len(self.samples),
            "errors": self.errors,
            "success_rate": round(self.success_rate, 1),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


# Polymarket endpoints for benchmarking
POLYMARKET_ENDPOINTS = {
    "clob_api": "https://clob.polymarket.com/",
    "data_api": "https://data-api.polymarket.com/markets?limit=1",
    "gamma_api": "https://gamma-api.polymarket.com/markets?limit=1",
    "geoblock": "https://polymarket.com/api/geoblock",
}

# Recommended VPS providers by location
VPS_RECOMMENDATIONS = {
    "amsterdam": {
        "latency_estimate": "2-5ms",
        "geo_blocked": False,
        "recommended": True,
        "providers": [
            {"name": "QuantVPS", "url": "https://quantvps.com", "price": "$29-59/mo", "notes": "Optimized for trading"},
            {"name": "Vultr", "url": "https://vultr.com", "price": "$5-20/mo", "notes": "Good budget option"},
            {"name": "DigitalOcean", "url": "https://digitalocean.com", "price": "$6-24/mo", "notes": "Reliable, easy setup"},
            {"name": "Hetzner", "url": "https://hetzner.com", "price": "$4-20/mo", "notes": "Best value in EU"},
            {"name": "OVH", "url": "https://ovh.com", "price": "$5-15/mo", "notes": "Large EU provider"},
        ],
    },
    "frankfurt": {
        "latency_estimate": "8-12ms",
        "geo_blocked": False,
        "recommended": True,
        "providers": [
            {"name": "Hetzner", "url": "https://hetzner.com", "price": "$4-20/mo", "notes": "Excellent value"},
            {"name": "AWS", "url": "https://aws.amazon.com", "price": "$10-50/mo", "notes": "eu-central-1 region"},
            {"name": "Contabo", "url": "https://contabo.com", "price": "$5-15/mo", "notes": "Budget friendly"},
        ],
    },
    "dublin": {
        "latency_estimate": "5-8ms",
        "geo_blocked": False,
        "recommended": True,
        "providers": [
            {"name": "AWS", "url": "https://aws.amazon.com", "price": "$10-50/mo", "notes": "eu-west-1 region"},
            {"name": "Google Cloud", "url": "https://cloud.google.com", "price": "$10-40/mo", "notes": "europe-west1"},
        ],
    },
    "london": {
        "latency_estimate": "1-2ms",
        "geo_blocked": True,
        "recommended": False,
        "providers": [],
        "warning": "UK is geo-blocked by Polymarket. Do NOT use London servers.",
    },
    "usa": {
        "latency_estimate": "80-120ms",
        "geo_blocked": True,
        "recommended": False,
        "providers": [],
        "warning": "USA is geo-blocked by Polymarket. Must use VPN/proxy from allowed region.",
    },
}


def create_optimized_session(proxy_config: Optional[ProxyConfig] = None) -> requests.Session:
    """
    Create an optimized requests session with connection pooling and retries.

    This session is tuned for low-latency trading:
    - Connection pooling to reuse TCP connections
    - Keep-alive enabled
    - Retry logic with exponential backoff
    - Optional proxy support
    """
    session = requests.Session()

    # Configure retry strategy
    retries = Retry(
        total=proxy_config.max_retries if proxy_config else 3,
        backoff_factor=proxy_config.retry_backoff if proxy_config else 0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
    )

    # Configure connection pooling
    pool_connections = proxy_config.pool_connections if proxy_config else 10
    pool_maxsize = proxy_config.pool_maxsize if proxy_config else 20

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=False,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set proxy if configured
    if proxy_config and proxy_config.enabled:
        proxies = proxy_config.get_requests_proxies()
        if proxies:
            session.proxies.update(proxies)

    # Optimize headers
    session.headers.update({
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "PolymarketBot/1.0",
    })

    return session


def benchmark_endpoint(
    url: str,
    session: Optional[requests.Session] = None,
    samples: int = 10,
    timeout: float = 5.0,
) -> LatencyBenchmark:
    """
    Benchmark latency to a specific endpoint.

    Args:
        url: URL to benchmark
        session: Optional pre-configured session
        samples: Number of requests to make
        timeout: Request timeout in seconds

    Returns:
        LatencyBenchmark with results
    """
    if session is None:
        session = create_optimized_session()

    benchmark = LatencyBenchmark(endpoint=url)

    for i in range(samples):
        try:
            start = time.perf_counter()
            resp = session.get(url, timeout=timeout)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            if resp.status_code < 400:
                benchmark.samples.append(elapsed)
            else:
                benchmark.errors += 1

        except Exception as e:
            benchmark.errors += 1

        # Small delay between samples to avoid rate limiting
        if i < samples - 1:
            time.sleep(0.1)

    return benchmark


def benchmark_all_endpoints(
    proxy_config: Optional[ProxyConfig] = None,
    samples: int = 10,
) -> Dict[str, LatencyBenchmark]:
    """
    Benchmark all Polymarket endpoints.

    Returns dict of endpoint name -> benchmark results.
    """
    session = create_optimized_session(proxy_config)
    results = {}

    for name, url in POLYMARKET_ENDPOINTS.items():
        results[name] = benchmark_endpoint(url, session, samples)

    return results


def check_geo_restriction(proxy_config: Optional[ProxyConfig] = None) -> Dict[str, Any]:
    """
    Check if current IP is geo-restricted by Polymarket.

    Returns:
        Dict with:
        - restricted: bool
        - country: str (if available)
        - ip: str (if available)
        - message: str
    """
    session = create_optimized_session(proxy_config)

    try:
        resp = session.get(
            "https://polymarket.com/api/geoblock",
            timeout=10,
        )
        data = resp.json()

        return {
            "restricted": data.get("restricted", False),
            "country": data.get("country", "Unknown"),
            "ip": data.get("ip", "Unknown"),
            "message": "Geo-blocked" if data.get("restricted") else "Access allowed",
        }

    except Exception as e:
        return {
            "restricted": None,
            "country": "Unknown",
            "ip": "Unknown",
            "message": f"Failed to check: {str(e)}",
        }


def get_infrastructure_info() -> Dict[str, Any]:
    """
    Get comprehensive infrastructure information for optimal setup.
    """
    return {
        "polymarket_servers": {
            "primary": {
                "location": "London, UK",
                "aws_region": "eu-west-2",
                "endpoints": {
                    "clob_api": "https://clob.polymarket.com",
                    "websocket": "wss://ws-subscriptions-clob.polymarket.com",
                },
            },
            "backup": {
                "location": "Dublin, Ireland",
                "aws_region": "eu-west-1",
            },
            "data_api": {
                "location": "Unknown (likely eu-west-2)",
                "endpoint": "https://data-api.polymarket.com",
            },
        },
        "optimal_locations": [
            {
                "city": "Amsterdam",
                "country": "Netherlands",
                "latency": "2-5ms",
                "recommended": True,
                "notes": "Best balance of low latency and legal access",
            },
            {
                "city": "Frankfurt",
                "country": "Germany",
                "latency": "8-12ms",
                "recommended": True,
                "notes": "Good alternative, many VPS providers",
            },
            {
                "city": "Dublin",
                "country": "Ireland",
                "latency": "5-8ms",
                "recommended": True,
                "notes": "Close to backup servers",
            },
            {
                "city": "Paris",
                "country": "France",
                "latency": "10-15ms",
                "recommended": False,
                "notes": "Acceptable but not optimal",
            },
        ],
        "blocked_locations": [
            {"location": "United States", "reason": "Regulatory restrictions"},
            {"location": "United Kingdom", "reason": "Regulatory restrictions"},
            {"location": "France", "reason": "May have restrictions - verify"},
        ],
        "speed_optimization_tips": [
            "Use WebSocket for real-time data instead of polling",
            "Enable HTTP/2 and connection keep-alive",
            "Use connection pooling (10+ connections)",
            "Deploy VPS in Amsterdam for lowest latency",
            "Use SSD/NVMe storage for faster I/O",
            "Minimize DNS lookups by caching",
            "Consider dedicated/bare-metal for lowest jitter",
        ],
        "vps_recommendations": VPS_RECOMMENDATIONS,
    }


# Global proxy configuration (can be updated at runtime)
_global_proxy_config: Optional[ProxyConfig] = None
_proxy_config_file = Path(__file__).parent.parent.parent / "proxy_config.json"


def load_proxy_config() -> ProxyConfig:
    """Load proxy configuration from file."""
    global _global_proxy_config

    if _proxy_config_file.exists():
        try:
            with open(_proxy_config_file, 'r') as f:
                data = json.load(f)
                _global_proxy_config = ProxyConfig.from_dict(data)
        except Exception as e:
            print(f"Failed to load proxy config: {e}")
            _global_proxy_config = ProxyConfig()
    else:
        _global_proxy_config = ProxyConfig()

    return _global_proxy_config


def save_proxy_config(config: ProxyConfig) -> None:
    """Save proxy configuration to file."""
    global _global_proxy_config
    _global_proxy_config = config

    data = {
        "enabled": config.enabled,
        "proxy_type": config.proxy_type.value,
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "password": config.password,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "pool_connections": config.pool_connections,
        "pool_maxsize": config.pool_maxsize,
        "keep_alive": config.keep_alive,
    }

    with open(_proxy_config_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_proxy_config() -> ProxyConfig:
    """Get current proxy configuration."""
    global _global_proxy_config
    if _global_proxy_config is None:
        load_proxy_config()
    return _global_proxy_config
