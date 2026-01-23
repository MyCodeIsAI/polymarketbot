"""
Polymarket API utilities.

General-purpose functions for interacting with Polymarket APIs.
These are NOT specific to ghost mode - they're used across the application.
"""

import requests
from typing import Optional


def lookup_wallet_from_username(username: str) -> Optional[str]:
    """
    Look up a Polymarket wallet address from a username.
    Uses Polymarket's profile page to resolve username to wallet.

    Args:
        username: The Polymarket username to look up

    Returns:
        The wallet address (lowercase) if found, None otherwise
    """
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        # Primary method: Fetch @username page and extract wallet from HTML
        # Polymarket now uses @username format for profile URLs
        clean_username = username.lstrip('@')
        url = f"https://polymarket.com/@{clean_username}"
        resp = requests.get(url, timeout=10, headers=headers, allow_redirects=True)

        if resp.status_code == 200:
            # Extract wallet address from the page content
            # Look for 0x addresses in the HTML
            wallet_match = re.search(r'0x[a-fA-F0-9]{40}', resp.text)
            if wallet_match:
                return wallet_match.group(0).lower()

            # Also try to find it in JSON data embedded in the page
            wallet_match = re.search(r'"(?:address|proxyWallet|wallet)":\s*"(0x[a-fA-F0-9]{40})"', resp.text)
            if wallet_match:
                return wallet_match.group(1).lower()

        # Fallback: Try fetching by wallet address directly if username looks like an address
        if username.startswith('0x') and len(username) == 42:
            return username.lower()

        return None

    except Exception as e:
        print(f"  [Wallet Lookup Error] {username}: {e}")
        return None


def get_user_activity(wallet: str, activity_type: str = "TRADE", limit: int = 10) -> list:
    """
    Get recent activity for a Polymarket wallet.

    Args:
        wallet: The wallet address to query
        activity_type: Type of activity (TRADE, etc.)
        limit: Maximum number of results

    Returns:
        List of activity records
    """
    try:
        url = f"https://data-api.polymarket.com/activity?user={wallet}&type={activity_type}&limit={limit}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            return resp.json() or []
        return []

    except Exception as e:
        print(f"  [Activity Error] {wallet}: {e}")
        return []


def get_market_info(condition_id: str) -> Optional[dict]:
    """
    Get information about a specific market.

    Args:
        condition_id: The market's condition ID

    Returns:
        Market info dict if found, None otherwise
    """
    try:
        url = f"https://gamma-api.polymarket.com/markets/{condition_id}"
        resp = requests.get(url, timeout=5)

        if resp.status_code == 200:
            return resp.json()
        return None

    except Exception as e:
        print(f"  [Market Info Error] {condition_id}: {e}")
        return None


def get_live_positions(wallet: str) -> list:
    """
    Get live positions for a wallet directly from Polymarket Data API.

    Args:
        wallet: The wallet address to query

    Returns:
        List of position dictionaries with normalized fields
    """
    try:
        url = f"https://data-api.polymarket.com/positions?user={wallet.lower()}&sizeThreshold=0.01"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            positions = data if isinstance(data, list) else data.get("positions", [])

            # Normalize field names for consistency with our internal format
            normalized = []
            for pos in positions:
                normalized.append({
                    "condition_id": pos.get("conditionId", pos.get("condition_id", "")),
                    "token_id": pos.get("assetId", pos.get("asset_id", pos.get("tokenId", ""))),
                    "outcome": pos.get("outcome", ""),
                    "size": float(pos.get("size", 0)),
                    "avg_price": float(pos.get("avgPrice", pos.get("average_price", 0))),
                    "current_price": float(pos.get("curPrice", pos.get("current_price", pos.get("avgPrice", 0)))),
                    "current_value": float(pos.get("currentValue", pos.get("current_value", 0))),
                    "initial_value": float(pos.get("initialValue", pos.get("initial_value", 0))),
                    "realized_pnl": float(pos.get("realizedPnl", pos.get("realized_pnl", 0))),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", pos.get("unrealized_pnl", 0))),
                    "cost_basis": float(pos.get("initialValue", pos.get("initial_value", 0))),
                    "market_slug": pos.get("marketSlug", pos.get("market_slug", "")),
                    "market_title": pos.get("title", pos.get("market_title", "")),
                    "status": "open" if float(pos.get("size", 0)) > 0 else "closed",
                })
            return normalized
        return []

    except Exception as e:
        print(f"  [Live Positions Error] {wallet}: {e}")
        return []


def get_live_portfolio_value(wallet: str) -> dict:
    """
    Get live portfolio value/P&L directly from Polymarket Data API.

    Args:
        wallet: The wallet address to query

    Returns:
        Dict with portfolio value and P&L data
    """
    try:
        url = f"https://data-api.polymarket.com/value?user={wallet.lower()}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "total_value": float(data.get("totalValue", data.get("total_value", 0))),
                "realized_pnl": float(data.get("realizedPnl", data.get("realized_pnl", 0))),
                "unrealized_pnl": float(data.get("unrealizedPnl", data.get("unrealized_pnl", 0))),
                "total_invested": float(data.get("totalInvested", data.get("total_invested", 0))),
                "positions_count": int(data.get("positionsCount", data.get("positions_count", 0))),
            }
        return {
            "total_value": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "total_invested": 0,
            "positions_count": 0,
        }

    except Exception as e:
        print(f"  [Live Portfolio Error] {wallet}: {e}")
        return {
            "total_value": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "total_invested": 0,
            "positions_count": 0,
        }


def get_live_balance(wallet: str) -> float:
    """
    Get live USDC balance for a wallet from Polymarket.

    Note: This requires authenticated API access. For now, returns
    an estimate based on portfolio value.

    Args:
        wallet: The wallet address to query

    Returns:
        USDC balance (or 0 if not available)
    """
    try:
        # The balance endpoint requires authentication
        # For now, we can get an approximation from the positions endpoint
        # In a real implementation with wallet connection, this would use
        # the CLOB API with authentication
        portfolio = get_live_portfolio_value(wallet)
        return portfolio.get("total_value", 0)

    except Exception as e:
        print(f"  [Live Balance Error] {wallet}: {e}")
        return 0
