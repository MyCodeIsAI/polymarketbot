"""
Copy trading core infrastructure.

This module provides the core account configuration and management
used by both ghost mode (simulation) and live trading mode.
"""

from .account import CopyTradeAccount, DEFAULT_SLIPPAGE_TIERS
from .manager import AccountManager
from .proxy import (
    ProxyConfig,
    ProxyType,
    LatencyBenchmark,
    create_optimized_session,
    benchmark_endpoint,
    benchmark_all_endpoints,
    check_geo_restriction,
    get_infrastructure_info,
    load_proxy_config,
    save_proxy_config,
    get_proxy_config,
    VPS_RECOMMENDATIONS,
    POLYMARKET_ENDPOINTS,
)

__all__ = [
    # Account management
    "CopyTradeAccount",
    "DEFAULT_SLIPPAGE_TIERS",
    "AccountManager",
    # Proxy/infrastructure
    "ProxyConfig",
    "ProxyType",
    "LatencyBenchmark",
    "create_optimized_session",
    "benchmark_endpoint",
    "benchmark_all_endpoints",
    "check_geo_restriction",
    "get_infrastructure_info",
    "load_proxy_config",
    "save_proxy_config",
    "get_proxy_config",
    "VPS_RECOMMENDATIONS",
    "POLYMARKET_ENDPOINTS",
]
