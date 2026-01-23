"""Simulation module for testing copy trading.

Provides:
- Historical trade replay
- Mock WebSocket feed
- Synthetic trade generation
- Dry-run execution
"""

from .harness import SimulationHarness, SimulationConfig
from .trade_generator import TradeGenerator, TradePattern
from .historical_replay import HistoricalReplay
from .mock_websocket import MockWebSocketFeed

__all__ = [
    "SimulationHarness",
    "SimulationConfig",
    "TradeGenerator",
    "TradePattern",
    "HistoricalReplay",
    "MockWebSocketFeed",
]
