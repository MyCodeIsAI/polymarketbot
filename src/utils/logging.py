"""Structured logging configuration for PolymarketBot.

This module sets up structured JSON logging using structlog, which provides:
- Consistent log format across the application
- Easy parsing for log aggregation systems
- Context binding for request/operation tracking
- Performance metrics in log entries
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import structlog
from structlog.types import Processor


def _add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _add_service_info(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add service identification to log entries."""
    event_dict["service"] = "polymarketbot"
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON logs. If False, use colored console format.
        log_file: Optional file path to write logs to (in addition to stdout)
    """
    # Get log level from environment or use provided default
    log_level = os.getenv("POLYBOT_LOG_LEVEL", level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Shared processors for all output formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_timestamp,
        _add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON output for production/parsing
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored console output for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """Get a logger instance with optional initial context.

    Args:
        name: Logger name (typically __name__ of the calling module)
        **initial_context: Key-value pairs to bind to all log entries

    Returns:
        A bound structlog logger

    Example:
        logger = get_logger(__name__, target="whale_trader_1")
        logger.info("position_detected", market="BTC > 100k", size=1000)
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LogContext:
    """Context manager for temporary log context binding.

    Use this to add context for a block of code, then automatically
    remove it when the block exits.

    Example:
        with LogContext(order_id="abc123", market="ETH > 5k"):
            logger.info("processing_order")
            # ... do work ...
            logger.info("order_complete")
    """

    def __init__(self, **context: Any):
        self.context = context
        self.token = None

    def __enter__(self) -> "LogContext":
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self.token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_execution_time(logger: structlog.BoundLogger, operation: str):
    """Decorator/context manager to log execution time of operations.

    Example:
        with log_execution_time(logger, "fetch_positions"):
            positions = await api.get_positions()
    """
    from contextlib import contextmanager
    import time

    @contextmanager
    def timer():
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "operation_complete",
                operation=operation,
                elapsed_ms=round(elapsed_ms, 2),
            )

    return timer()


# Pre-configured loggers for common components
def get_api_logger() -> structlog.BoundLogger:
    """Get logger for API operations."""
    return get_logger("polymarketbot.api", component="api")


def get_execution_logger() -> structlog.BoundLogger:
    """Get logger for order execution."""
    return get_logger("polymarketbot.execution", component="execution")


def get_monitor_logger() -> structlog.BoundLogger:
    """Get logger for position monitoring."""
    return get_logger("polymarketbot.monitor", component="monitor")


def get_safety_logger() -> structlog.BoundLogger:
    """Get logger for safety systems."""
    return get_logger("polymarketbot.safety", component="safety")
