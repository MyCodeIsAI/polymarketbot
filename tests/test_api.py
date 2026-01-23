"""Tests for API clients.

These tests include both unit tests (mocked) and integration tests
that hit real Polymarket APIs. Integration tests are marked and can
be skipped with: pytest -m "not integration"
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.base import BaseAPIClient, APIResponse
from src.api.rate_limiter import RateLimiter, TokenBucket, RateLimitConfig
from src.api.auth import L1Authenticator, L2Authenticator, APICredentials, PolymarketAuth
from src.api.data import DataAPIClient, Position, Activity, ActivityType, TradeSide
from src.api.clob import CLOBClient, OrderBook, OrderBookLevel, OrderSide
from src.api.gamma import GammaAPIClient, Event, Market, Token


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        config = RateLimitConfig(
            requests_per_window=100,
            window_seconds=10.0,
        )
        assert config.tokens_per_second == 10.0

    @pytest.mark.asyncio
    async def test_token_bucket_acquire(self):
        """Test token bucket acquisition."""
        config = RateLimitConfig(requests_per_window=10, window_seconds=1.0)
        bucket = TokenBucket(config=config)

        # Should acquire immediately when tokens available
        wait_time = await bucket.acquire(1)
        assert wait_time == 0.0
        assert bucket.tokens == 9.0

    @pytest.mark.asyncio
    async def test_token_bucket_exhaustion(self):
        """Test token bucket when exhausted."""
        config = RateLimitConfig(requests_per_window=2, window_seconds=1.0)
        bucket = TokenBucket(config=config)

        # Exhaust tokens
        await bucket.acquire(2)

        # Next acquire should return wait time
        wait_time = await bucket.acquire(1)
        assert wait_time > 0

    def test_rate_limiter_bucket_selection(self):
        """Test that rate limiter selects correct bucket for path."""
        limiter = RateLimiter()

        # Test path to bucket mapping
        assert limiter._get_bucket_name("/book") == "clob_book"
        assert limiter._get_bucket_name("/positions") == "data_positions"
        assert limiter._get_bucket_name("/events") == "gamma_events"
        assert limiter._get_bucket_name("/unknown") == "data_general"


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication functionality."""

    # Test private key (DO NOT USE IN PRODUCTION - this is a well-known test key)
    TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    TEST_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

    def test_l1_authenticator_address(self):
        """Test L1 authenticator derives correct address."""
        auth = L1Authenticator(self.TEST_PRIVATE_KEY)
        assert auth.address.lower() == self.TEST_ADDRESS.lower()

    def test_l1_sign_message(self):
        """Test L1 message signing."""
        auth = L1Authenticator(self.TEST_PRIVATE_KEY)
        headers = auth.sign_message(timestamp="1234567890")

        assert headers.address.lower() == self.TEST_ADDRESS.lower()
        assert headers.timestamp == "1234567890"
        assert headers.signature  # Should have a signature
        assert len(headers.signature) > 0

    def test_l2_authenticator(self):
        """Test L2 HMAC authentication."""
        creds = APICredentials(
            api_key="test-api-key",
            api_secret="dGVzdC1zZWNyZXQ=",  # "test-secret" base64
            passphrase="test-passphrase",
        )
        auth = L2Authenticator(creds, self.TEST_ADDRESS)

        headers = auth.sign_request("GET", "/orders", "", "1234567890")

        assert headers.api_key == "test-api-key"
        assert headers.passphrase == "test-passphrase"
        assert headers.timestamp == "1234567890"
        assert headers.signature  # HMAC signature

    def test_polymarket_auth_without_credentials(self):
        """Test PolymarketAuth before credentials are set."""
        auth = PolymarketAuth(self.TEST_PRIVATE_KEY)

        assert auth.signer_address.lower() == self.TEST_ADDRESS.lower()
        assert not auth.has_credentials

        # L1 headers should work
        headers = auth.get_l1_headers()
        assert "POLY_ADDRESS" in headers
        assert "POLY_SIGNATURE" in headers

    def test_polymarket_auth_with_credentials(self):
        """Test PolymarketAuth with credentials."""
        creds = APICredentials(
            api_key="test-api-key",
            api_secret="dGVzdC1zZWNyZXQ=",
            passphrase="test-passphrase",
        )
        auth = PolymarketAuth(self.TEST_PRIVATE_KEY, credentials=creds)

        assert auth.has_credentials

        # L2 headers should work
        headers = auth.get_l2_headers("GET", "/orders")
        assert headers["POLY_API_KEY"] == "test-api-key"


# =============================================================================
# Data Models Tests
# =============================================================================


class TestDataModels:
    """Tests for API data models."""

    def test_position_from_api(self):
        """Test Position parsing from API response."""
        data = {
            "conditionId": "0x123",
            "assetId": "0x456",
            "outcome": "Yes",
            "size": "100.5",
            "avgPrice": "0.65",
            "currentValue": "75.0",
            "initialValue": "65.0",
            "realizedPnl": "10.0",
            "unrealizedPnl": "5.0",
        }
        position = Position.from_api(data)

        assert position.condition_id == "0x123"
        assert position.token_id == "0x456"
        assert position.size == Decimal("100.5")
        assert position.average_price == Decimal("0.65")

    def test_activity_from_api(self):
        """Test Activity parsing from API response."""
        data = {
            "id": "abc123",
            "type": "TRADE",
            "timestamp": 1704067200,
            "conditionId": "0x123",
            "assetId": "0x456",
            "outcome": "Yes",
            "side": "BUY",
            "size": "50",
            "price": "0.5",
            "value": "25",
        }
        activity = Activity.from_api(data)

        assert activity.id == "abc123"
        assert activity.type == ActivityType.TRADE
        assert activity.side == TradeSide.BUY
        assert activity.size == Decimal("50")

    def test_order_book_level(self):
        """Test OrderBookLevel parsing."""
        # From list format
        level1 = OrderBookLevel.from_api([0.55, 1000])
        assert level1.price == Decimal("0.55")
        assert level1.size == Decimal("1000")

        # From dict format
        level2 = OrderBookLevel.from_api({"price": "0.55", "size": "1000"})
        assert level2.price == Decimal("0.55")
        assert level2.size == Decimal("1000")

    def test_order_book_properties(self):
        """Test OrderBook calculated properties."""
        book = OrderBook(
            token_id="0x123",
            bids=[
                OrderBookLevel(Decimal("0.55"), Decimal("100")),
                OrderBookLevel(Decimal("0.54"), Decimal("200")),
            ],
            asks=[
                OrderBookLevel(Decimal("0.56"), Decimal("150")),
                OrderBookLevel(Decimal("0.57"), Decimal("250")),
            ],
        )

        assert book.best_bid == Decimal("0.55")
        assert book.best_ask == Decimal("0.56")
        assert book.spread == Decimal("0.01")
        assert book.midpoint == Decimal("0.555")

    def test_market_from_api(self):
        """Test Market parsing from API response."""
        data = {
            "conditionId": "0x123",
            "question": "Will X happen?",
            "slug": "will-x-happen",
            "tokens": [
                {"token_id": "0xyes", "outcome": "Yes"},
                {"token_id": "0xno", "outcome": "No"},
            ],
            "volume": "1000000",
            "active": True,
        }
        market = Market.from_api(data)

        assert market.condition_id == "0x123"
        assert market.question == "Will X happen?"
        assert len(market.tokens) == 2
        assert market.yes_token_id == "0xyes"
        assert market.no_token_id == "0xno"

    def test_event_from_api(self):
        """Test Event parsing from API response."""
        data = {
            "id": "event123",
            "slug": "test-event",
            "title": "Test Event",
            "markets": [
                {
                    "conditionId": "0x123",
                    "question": "Test question?",
                    "slug": "test-market",
                }
            ],
            "volume": "5000000",
        }
        event = Event.from_api(data)

        assert event.id == "event123"
        assert event.title == "Test Event"
        assert len(event.markets) == 1


# =============================================================================
# Integration Tests (hit real APIs)
# =============================================================================


@pytest.mark.integration
class TestDataAPIIntegration:
    """Integration tests for Data API."""

    @pytest.mark.asyncio
    async def test_get_positions_for_known_wallet(self):
        """Test fetching positions for a known active wallet."""
        async with DataAPIClient() as client:
            # Use a known active trader's wallet (public data)
            # This is a random active wallet for testing - replace if needed
            positions = await client.get_positions(
                user="0x0000000000000000000000000000000000000000",
                limit=5,
            )

            # Should return a list (may be empty for zero address)
            assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_get_activity(self):
        """Test fetching activity."""
        async with DataAPIClient() as client:
            activity = await client.get_activity(
                user="0x0000000000000000000000000000000000000000",
                activity_type=ActivityType.TRADE,
                limit=5,
            )

            assert isinstance(activity, list)


@pytest.mark.integration
class TestCLOBIntegration:
    """Integration tests for CLOB API."""

    @pytest.mark.asyncio
    async def test_get_markets(self):
        """Test fetching market list."""
        async with CLOBClient() as client:
            markets = await client.get_markets()

            assert isinstance(markets, list)
            # Should have some active markets
            assert len(markets) > 0

    @pytest.mark.asyncio
    async def test_get_order_book(self):
        """Test fetching order book."""
        async with CLOBClient() as client:
            # First get a market to get a token ID
            markets = await client.get_simplified_markets()

            if markets:
                # Get first market with tokens
                for market in markets:
                    tokens = market.get("tokens", [])
                    if tokens:
                        token_id = tokens[0].get("token_id")
                        if token_id:
                            book = await client.get_order_book(token_id)
                            assert isinstance(book, OrderBook)
                            assert book.token_id == token_id
                            break


@pytest.mark.integration
class TestGammaAPIIntegration:
    """Integration tests for Gamma API."""

    @pytest.mark.asyncio
    async def test_get_events(self):
        """Test fetching events."""
        async with GammaAPIClient() as client:
            events = await client.get_events(limit=5)

            assert isinstance(events, list)
            assert len(events) > 0

            # Check event structure
            event = events[0]
            assert event.id
            assert event.title

    @pytest.mark.asyncio
    async def test_get_markets(self):
        """Test fetching markets."""
        async with GammaAPIClient() as client:
            markets = await client.get_markets(limit=5)

            assert isinstance(markets, list)
            assert len(markets) > 0

    @pytest.mark.asyncio
    async def test_search_markets(self):
        """Test market search."""
        async with GammaAPIClient() as client:
            # Search for a common topic
            markets = await client.search_markets("election", limit=5)

            assert isinstance(markets, list)
            # May or may not find results depending on current markets


# =============================================================================
# Mock Tests for API Clients
# =============================================================================


class TestBaseAPIClient:
    """Tests for BaseAPIClient with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful API request."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '{"data": "test"}'
            mock_response.json.return_value = {"data": "test"}
            mock_response.headers = {}

            mock_instance = AsyncMock()
            mock_instance.request.return_value = mock_response
            mock_instance.is_closed = False
            mock_client.return_value.__aenter__.return_value = mock_instance

            client = BaseAPIClient("https://test.com")
            client._client = mock_instance

            response = await client.get("/test")

            assert response.status_code == 200
            assert response.data == {"data": "test"}

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test that rate limits trigger appropriate behavior."""
        limiter = RateLimiter(safety_margin=1.0)  # No safety margin for testing

        # Acquire many tokens quickly
        for _ in range(10):
            await limiter.acquire("/positions", tokens=1)

        # Check bucket status
        status = limiter.get_bucket_status("data_positions")
        assert status["exists"]
        assert status["available_tokens"] < status["max_tokens"]
