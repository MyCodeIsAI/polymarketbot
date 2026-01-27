"""Pytest configuration for insider scanner tests."""

import asyncio
import pytest


@pytest.fixture
def run_async():
    """Helper fixture to run async functions in sync tests."""
    def _run_async(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return _run_async
