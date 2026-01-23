"""Historical trade replay for simulation testing.

Fetches and replays historical trades from target wallets
at configurable speeds for testing detection and execution.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, AsyncIterator, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Polymarket/Polygon API endpoints
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_RPC = "https://polygon-rpc.com"


class HistoricalReplay:
    """Replays historical trades from target wallets."""

    def __init__(
        self,
        wallets: List[str],
        speed_multiplier: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        chunk_size: int = 100,
    ):
        self.wallets = [w.lower() for w in wallets]
        self.speed_multiplier = speed_multiplier
        self.start_time = start_time or (datetime.utcnow() - timedelta(days=7))
        self.end_time = end_time or datetime.utcnow()
        self.chunk_size = chunk_size

        self._trades: List[Dict[str, Any]] = []
        self._loaded = False

    async def load_trades(self) -> int:
        """Load historical trades for all wallets.

        Returns:
            Number of trades loaded
        """
        all_trades = []

        async with aiohttp.ClientSession() as session:
            for wallet in self.wallets:
                logger.info("loading_historical_trades", wallet=wallet[:10] + "...")

                try:
                    trades = await self._fetch_trades_for_wallet(session, wallet)
                    all_trades.extend(trades)
                except Exception as e:
                    logger.error("failed_to_load_trades", wallet=wallet, error=str(e))

        # Sort by timestamp
        all_trades.sort(key=lambda t: t.get("timestamp", datetime.min))

        # Filter by time range
        self._trades = [
            t for t in all_trades
            if self.start_time <= t.get("timestamp", datetime.min) <= self.end_time
        ]

        self._loaded = True

        logger.info(
            "historical_trades_loaded",
            total_trades=len(self._trades),
            wallets=len(self.wallets),
        )

        return len(self._trades)

    async def _fetch_trades_for_wallet(
        self,
        session: aiohttp.ClientSession,
        wallet: str,
    ) -> List[Dict[str, Any]]:
        """Fetch trades for a single wallet from Polymarket API.

        Args:
            session: aiohttp session
            wallet: Wallet address

        Returns:
            List of trade dictionaries
        """
        trades = []
        cursor = None

        while True:
            # Build query params
            params = {
                "user": wallet,
                "limit": self.chunk_size,
            }
            if cursor:
                params["cursor"] = cursor

            # Fetch from Polymarket activity API
            url = f"{POLYMARKET_GAMMA_API}/activity"

            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "api_error",
                            status=resp.status,
                            wallet=wallet,
                        )
                        break

                    data = await resp.json()

            except asyncio.TimeoutError:
                logger.warning("api_timeout", wallet=wallet)
                break
            except Exception as e:
                logger.error("api_request_failed", error=str(e))
                break

            # Parse trades from response
            activities = data.get("data", data) if isinstance(data, dict) else data

            if not activities:
                break

            for activity in activities:
                trade = self._parse_activity(activity, wallet)
                if trade:
                    trades.append(trade)

            # Check for pagination
            cursor = data.get("next_cursor") if isinstance(data, dict) else None
            if not cursor or len(activities) < self.chunk_size:
                break

        return trades

    def _parse_activity(
        self,
        activity: Dict[str, Any],
        wallet: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse an activity record into a trade.

        Args:
            activity: Raw activity from API
            wallet: Wallet address

        Returns:
            Parsed trade dict or None
        """
        try:
            # Extract relevant fields based on Polymarket activity format
            activity_type = activity.get("type", "")

            # Only process trades
            if activity_type not in ("trade", "buy", "sell", "TRADE"):
                return None

            # Parse timestamp
            timestamp_str = activity.get("timestamp") or activity.get("created_at")
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromtimestamp(timestamp_str)
            else:
                timestamp = datetime.utcnow()

            # Extract trade details
            trade = {
                "trade_id": activity.get("id") or activity.get("transaction_hash", "unknown"),
                "wallet": wallet,
                "market_id": activity.get("market_id") or activity.get("condition_id", "unknown"),
                "market_name": activity.get("market_title") or activity.get("title", "Unknown"),
                "outcome": activity.get("outcome") or activity.get("outcome_index", "Yes"),
                "side": self._determine_side(activity),
                "size": float(activity.get("size") or activity.get("amount") or 0),
                "price": float(activity.get("price") or activity.get("avg_price") or 0.5),
                "timestamp": timestamp,
            }

            return trade

        except Exception as e:
            logger.debug("parse_activity_failed", error=str(e))
            return None

    def _determine_side(self, activity: Dict[str, Any]) -> str:
        """Determine trade side from activity."""
        side = activity.get("side", "").upper()
        if side in ("BUY", "SELL"):
            return side

        activity_type = activity.get("type", "").lower()
        if "buy" in activity_type:
            return "BUY"
        elif "sell" in activity_type:
            return "SELL"

        # Default based on other indicators
        return "BUY"

    async def trades(self) -> AsyncIterator[Dict[str, Any]]:
        """Replay trades as an async iterator.

        Yields trades with realistic timing based on speed_multiplier.
        """
        if not self._loaded:
            await self.load_trades()

        if not self._trades:
            logger.warning("no_trades_to_replay")
            return

        logger.info(
            "starting_replay",
            trades=len(self._trades),
            speed=self.speed_multiplier,
        )

        replay_start = datetime.utcnow()
        first_trade_time = self._trades[0].get("timestamp", datetime.utcnow())

        for i, trade in enumerate(self._trades):
            trade_time = trade.get("timestamp", datetime.utcnow())

            # Calculate when this trade should be replayed
            time_since_first = (trade_time - first_trade_time).total_seconds()
            replay_time = time_since_first / self.speed_multiplier

            # Wait until it's time to replay this trade
            elapsed = (datetime.utcnow() - replay_start).total_seconds()
            wait_time = replay_time - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Update timestamp to current time for the simulation
            trade["original_timestamp"] = trade["timestamp"]
            trade["timestamp"] = datetime.utcnow()

            yield trade

            if (i + 1) % 10 == 0:
                logger.debug(
                    "replay_progress",
                    completed=i + 1,
                    total=len(self._trades),
                )

        logger.info("replay_complete", total_trades=len(self._trades))

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded trades.

        Returns:
            Summary dictionary
        """
        if not self._trades:
            return {"trades": 0}

        total_volume = sum(t.get("size", 0) * t.get("price", 0) for t in self._trades)
        buy_count = sum(1 for t in self._trades if t.get("side") == "BUY")
        sell_count = len(self._trades) - buy_count

        # Get unique markets
        markets = set(t.get("market_id") for t in self._trades)

        # Time span
        timestamps = [t.get("timestamp") for t in self._trades if t.get("timestamp")]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600 if timestamps else 0

        return {
            "trades": len(self._trades),
            "total_volume": total_volume,
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "unique_markets": len(markets),
            "time_span_hours": time_span,
            "trades_per_hour": len(self._trades) / time_span if time_span > 0 else 0,
        }


class PolygonTradesFetcher:
    """Fetches trades directly from Polygon blockchain."""

    # Polymarket CLOB contract addresses
    EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

    def __init__(self, rpc_url: str = POLYGON_RPC):
        self.rpc_url = rpc_url

    async def get_trades_for_wallet(
        self,
        wallet: str,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get trades from Polygon events.

        This provides more accurate trade data directly from blockchain.
        """
        # Implementation would use web3.py to query OrderFilled events
        # Filtering by maker/taker address

        # For now, return empty - full implementation would require
        # indexing OrderFilled events from the CLOB contracts

        logger.info(
            "polygon_trades_fetch",
            wallet=wallet,
            from_block=from_block,
            to_block=to_block,
        )

        return []
