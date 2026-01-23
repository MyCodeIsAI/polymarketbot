"""CLI module for PolymarketBot.

This module provides the command-line interface for:
- Bot lifecycle management (start, stop, status)
- Account management (list, add, pause, resume)
- Position and trade viewing
- P&L reporting
- Configuration validation
- System diagnostics
"""

from .main import app, main
from .formatters import (
    format_table,
    format_status,
    format_positions,
    format_trades,
    format_pnl,
)

__all__ = [
    "app",
    "main",
    "format_table",
    "format_status",
    "format_positions",
    "format_trades",
    "format_pnl",
]
