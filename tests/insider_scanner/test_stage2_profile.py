"""Stage 2 Tests: Profile Fetcher and Historical Data Ingestion.

Tests:
- ProfileFetcher initialization
- Position fetching and aggregation
- Activity fetching and processing
- Win rate calculation
- Cumulative position tracking
- Metric derivation

Run with: pytest tests/insider_scanner/test_stage2_profile.py -v
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)

from src.insider_scanner.profile import (
    ProfileFetcher,
    WalletProfile,
    PositionSummary,
    TradeEntry,
)
from src.api.data import Position, Activity, ActivityType, TradeSide
from src.api.gamma import Market, Token


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_data_client():
    """Create a mock Data API client."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_gamma_client():
    """Create a mock Gamma API client."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def sample_positions():
    """Create sample position data."""
    return [
        Position(
            condition_id="0xmarket1",
            token_id="0xtoken1",
            outcome="Yes",
            size=Decimal("100"),
            average_price=Decimal("0.40"),
            current_value=Decimal("50"),
            initial_value=Decimal("40"),
            realized_pnl=Decimal("10"),
            unrealized_pnl=Decimal("0"),
            market_slug="election-market",
            market_title="Will candidate win?",
        ),
        Position(
            condition_id="0xmarket2",
            token_id="0xtoken2",
            outcome="No",
            size=Decimal("200"),
            average_price=Decimal("0.25"),
            current_value=Decimal("100"),
            initial_value=Decimal("50"),
            realized_pnl=Decimal("50"),
            unrealized_pnl=Decimal("0"),
            market_slug="rate-market",
            market_title="Fed rate decision",
        ),
    ]


@pytest.fixture
def sample_activities():
    """Create sample activity data."""
    now = datetime.utcnow()
    return [
        Activity(
            id="act1",
            type=ActivityType.TRADE,
            timestamp=now - timedelta(days=10),
            condition_id="0xmarket1",
            token_id="0xtoken1",
            outcome="Yes",
            side=TradeSide.BUY,
            size=Decimal("50"),
            price=Decimal("0.38"),
            usd_value=Decimal("19"),
            tx_hash="0xtx1",
            user="0xwallet",
            event_slug="election",
            market_title="Will candidate win?",
        ),
        Activity(
            id="act2",
            type=ActivityType.TRADE,
            timestamp=now - timedelta(days=5),
            condition_id="0xmarket1",
            token_id="0xtoken1",
            outcome="Yes",
            side=TradeSide.BUY,
            size=Decimal("50"),
            price=Decimal("0.42"),
            usd_value=Decimal("21"),
            tx_hash="0xtx2",
            user="0xwallet",
            event_slug="election",
            market_title="Will candidate win?",
        ),
        Activity(
            id="act3",
            type=ActivityType.SPLIT,
            timestamp=now - timedelta(days=3),
            condition_id="0xmarket1",
            token_id="0xtoken1",
            outcome="Yes",
            side=None,
            size=Decimal("25"),
            price=Decimal("0"),
            usd_value=Decimal("0"),
        ),
        Activity(
            id="act4",
            type=ActivityType.TRADE,
            timestamp=now - timedelta(days=1),
            condition_id="0xmarket2",
            token_id="0xtoken2",
            outcome="No",
            side=TradeSide.BUY,
            size=Decimal("200"),
            price=Decimal("0.25"),
            usd_value=Decimal("50"),
            tx_hash="0xtx4",
            user="0xwallet",
            event_slug="fed-rate",
            market_title="Fed rate decision",
        ),
    ]


@pytest.fixture
def sample_markets():
    """Create sample market data."""
    return {
        "0xmarket1": Market(
            condition_id="0xmarket1",
            question="Will candidate win?",
            slug="election-market",
            tokens=[
                Token(token_id="0xtoken1", outcome="Yes", price=Decimal("0.60"), winner=True),
                Token(token_id="0xtoken1_no", outcome="No", price=Decimal("0.40"), winner=False),
            ],
            closed=True,
            volume=Decimal("1000000"),
        ),
        "0xmarket2": Market(
            condition_id="0xmarket2",
            question="Fed rate decision",
            slug="rate-market",
            tokens=[
                Token(token_id="0xtoken2_yes", outcome="Yes", price=Decimal("0.30"), winner=False),
                Token(token_id="0xtoken2", outcome="No", price=Decimal("0.70"), winner=True),
            ],
            closed=True,
            volume=Decimal("500000"),
        ),
        "0xmarket3": Market(
            condition_id="0xmarket3",
            question="Active market",
            slug="active-market",
            tokens=[
                Token(token_id="0xtoken3", outcome="Yes", price=Decimal("0.50")),
                Token(token_id="0xtoken3_no", outcome="No", price=Decimal("0.50")),
            ],
            closed=False,
            volume=Decimal("100000"),
        ),
    }


# =============================================================================
# ProfileFetcher Initialization Tests
# =============================================================================

class TestProfileFetcherInit:
    """Test ProfileFetcher initialization."""

    def test_init_with_clients(self, mock_data_client, mock_gamma_client):
        """Test initialization with provided clients."""
        fetcher = ProfileFetcher(
            data_client=mock_data_client,
            gamma_client=mock_gamma_client,
        )

        assert fetcher._data_client is mock_data_client
        assert fetcher._gamma_client is mock_gamma_client
        assert not fetcher._owns_clients  # Didn't create them, so don't own them

    def test_init_without_clients(self):
        """Test initialization creates clients."""
        fetcher = ProfileFetcher()

        assert fetcher._data_client is None
        assert fetcher._gamma_client is None
        assert fetcher._owns_clients  # Will create them in __aenter__


# =============================================================================
# Profile Fetching Tests
# =============================================================================

class TestProfileFetching:
    """Test profile fetching and processing."""

    def test_fetch_profile_basic(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test basic profile fetching."""
        async def _test():
            # Setup mocks
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            # Fetch profile
            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Verify basic info
            assert profile.wallet_address == "0xtest_wallet"
            assert profile.total_positions == 2
            assert profile.transaction_count == 4

        run_async(_test())

    def test_fetch_profile_activity_breakdown(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test activity type counting."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Check activity breakdown
            assert profile.trade_count == 3
            assert profile.split_count == 1
            assert profile.merge_count == 0
            assert profile.redeem_count == 0

        run_async(_test())

    def test_account_age_calculation(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test account age is calculated from activity."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # First activity is 10 days ago
            assert profile.account_age_days == 10

        run_async(_test())


# =============================================================================
# Position Processing Tests
# =============================================================================

class TestPositionProcessing:
    """Test position aggregation and processing."""

    def test_position_aggregation(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_markets,
    ):
        """Test positions are aggregated correctly."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            assert profile.total_positions == 2
            assert profile.unique_markets == 2

        run_async(_test())

    def test_pnl_calculation(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_markets,
    ):
        """Test PnL is calculated correctly."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Position 1: realized=10, unrealized=0
            # Position 2: realized=50, unrealized=0
            assert profile.realized_pnl == Decimal("60")
            assert profile.total_pnl == Decimal("60")

        run_async(_test())

    def test_largest_position_tracking(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_markets,
    ):
        """Test largest position is tracked."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Position 2 has current_value=100, Position 1 has 50
            assert profile.largest_position_usd == Decimal("100")

        run_async(_test())


# =============================================================================
# Win Rate Tests
# =============================================================================

class TestWinRateCalculation:
    """Test win rate calculation."""

    def test_win_rate_all_wins(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_markets,
    ):
        """Test win rate with all winning positions."""
        async def _test():
            # Both positions won (market1 Yes won, market2 No won)
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            assert profile.win_count == 2
            assert profile.loss_count == 0
            assert profile.win_rate == 1.0

        run_async(_test())

    def test_win_rate_mixed(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_markets,
    ):
        """Test win rate with mixed wins/losses."""
        async def _test():
            # Create positions where one wins, one loses
            positions = [
                Position(
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",  # Winner
                    size=Decimal("100"),
                    average_price=Decimal("0.40"),
                    current_value=Decimal("100"),
                    initial_value=Decimal("40"),
                    realized_pnl=Decimal("60"),
                    unrealized_pnl=Decimal("0"),
                ),
                Position(
                    condition_id="0xmarket2",
                    token_id="0xtoken2_yes",
                    outcome="Yes",  # Loser (No won)
                    size=Decimal("100"),
                    average_price=Decimal("0.30"),
                    current_value=Decimal("0"),
                    initial_value=Decimal("30"),
                    realized_pnl=Decimal("-30"),
                    unrealized_pnl=Decimal("0"),
                ),
            ]

            mock_data_client.get_positions.return_value = positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            assert profile.win_count == 1
            assert profile.loss_count == 1
            assert profile.win_rate == 0.5

        run_async(_test())

    def test_win_rate_no_resolved(
        self,
        mock_data_client,
        mock_gamma_client,
    ):
        """Test win rate with no resolved positions."""
        async def _test():
            positions = [
                Position(
                    condition_id="0xmarket3",
                    token_id="0xtoken3",
                    outcome="Yes",
                    size=Decimal("100"),
                    average_price=Decimal("0.50"),
                    current_value=Decimal("50"),
                    initial_value=Decimal("50"),
                    realized_pnl=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                ),
            ]

            # Unresolved market
            unresolved_market = Market(
                condition_id="0xmarket3",
                question="Active market",
                slug="active-market",
                tokens=[
                    Token(token_id="0xtoken3", outcome="Yes", price=Decimal("0.50")),
                    Token(token_id="0xtoken3_no", outcome="No", price=Decimal("0.50")),
                ],
                closed=False,
            )

            mock_data_client.get_positions.return_value = positions
            mock_data_client.get_activity.return_value = []

            async def get_market(condition_id):
                if condition_id == "0xmarket3":
                    return unresolved_market
                return None

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            assert profile.win_count == 0
            assert profile.loss_count == 0
            assert profile.win_rate == 0.0
            assert profile.active_positions == 1

        run_async(_test())


# =============================================================================
# Cumulative Position Tests
# =============================================================================

class TestCumulativePositions:
    """Test cumulative position tracking for split entry detection."""

    def test_cumulative_positions_basic(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test cumulative position calculation."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            cumulative = await fetcher.fetch_cumulative_positions("0xtest_wallet")

            # Should have positions for markets with BUY trades
            assert len(cumulative) > 0

        run_async(_test())

    def test_split_entry_detection(
        self,
        mock_data_client,
        mock_gamma_client,
    ):
        """Test that split entries are detected from multiple trades."""
        async def _test():
            now = datetime.utcnow()

            # Multiple small buys in same market = split entry pattern
            activities = [
                Activity(
                    id="act1",
                    type=ActivityType.TRADE,
                    timestamp=now - timedelta(hours=4),
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",
                    side=TradeSide.BUY,
                    size=Decimal("1000"),
                    price=Decimal("0.40"),
                    usd_value=Decimal("10000"),  # $10K
                ),
                Activity(
                    id="act2",
                    type=ActivityType.TRADE,
                    timestamp=now - timedelta(hours=3),
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",
                    side=TradeSide.BUY,
                    size=Decimal("1000"),
                    price=Decimal("0.41"),
                    usd_value=Decimal("10000"),  # $10K
                ),
                Activity(
                    id="act3",
                    type=ActivityType.TRADE,
                    timestamp=now - timedelta(hours=2),
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",
                    side=TradeSide.BUY,
                    size=Decimal("1000"),
                    price=Decimal("0.42"),
                    usd_value=Decimal("10000"),  # $10K
                ),
                Activity(
                    id="act4",
                    type=ActivityType.TRADE,
                    timestamp=now - timedelta(hours=1),
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",
                    side=TradeSide.BUY,
                    size=Decimal("1000"),
                    price=Decimal("0.43"),
                    usd_value=Decimal("10000"),  # $10K
                ),
            ]

            positions = [
                Position(
                    condition_id="0xmarket1",
                    token_id="0xtoken1",
                    outcome="Yes",
                    size=Decimal("4000"),
                    average_price=Decimal("0.415"),
                    current_value=Decimal("40000"),
                    initial_value=Decimal("40000"),
                    realized_pnl=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                ),
            ]

            mock_data_client.get_positions.return_value = positions
            mock_data_client.get_activity.return_value = activities
            mock_gamma_client.get_market.return_value = None

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            cumulative = await fetcher.fetch_cumulative_positions("0xtest_wallet")

            # Should detect 4 entries totaling $40K
            assert len(cumulative) == 1
            pos = list(cumulative.values())[0]
            assert pos.entry_count == 4
            assert pos.total_usd == Decimal("40000")

        run_async(_test())


# =============================================================================
# Trade Entry Tests
# =============================================================================

class TestTradeEntries:
    """Test trade entry creation from activity."""

    def test_trades_from_activity(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test trades are extracted from activity."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Should have 3 trades (excluding the SPLIT activity)
            assert len(profile.trades) == 3

            # Trades should be sorted by timestamp
            for i in range(len(profile.trades) - 1):
                assert profile.trades[i].timestamp <= profile.trades[i + 1].timestamp

        run_async(_test())


# =============================================================================
# WalletProfile Dataclass Tests
# =============================================================================

class TestWalletProfile:
    """Test WalletProfile dataclass."""

    def test_wallet_profile_defaults(self):
        """Test default values."""
        profile = WalletProfile(
            wallet_address="0xtest",
            fetched_at=datetime.utcnow(),
        )

        assert profile.win_rate == 0.0
        assert profile.total_pnl == Decimal("0")
        assert profile.trades == []
        assert profile.positions == []

    def test_position_summary_creation(self):
        """Test PositionSummary creation."""
        now = datetime.utcnow()
        summary = PositionSummary(
            market_id="0xmarket",
            market_title="Test Market",
            side="Yes",
            total_size=Decimal("100"),
            total_usd=Decimal("5000"),
            entry_count=3,
            avg_price=Decimal("0.50"),
            first_entry=now - timedelta(days=5),
            last_entry=now,
            is_resolved=True,
            won=True,
            pnl=Decimal("2500"),
        )

        assert summary.entry_count == 3
        assert summary.won is True


# =============================================================================
# Integration Test
# =============================================================================

class TestStage2Integration:
    """Integration tests for Stage 2 completion."""

    def test_complete_profile_flow(
        self,
        mock_data_client,
        mock_gamma_client,
        sample_positions,
        sample_activities,
        sample_markets,
    ):
        """Test complete profile fetching flow."""
        async def _test():
            mock_data_client.get_positions.return_value = sample_positions
            mock_data_client.get_activity.return_value = sample_activities

            async def get_market(condition_id):
                return sample_markets.get(condition_id)

            mock_gamma_client.get_market = get_market

            fetcher = ProfileFetcher(
                data_client=mock_data_client,
                gamma_client=mock_gamma_client,
            )
            profile = await fetcher.fetch_profile("0xtest_wallet")

            # Comprehensive assertions
            assert profile.wallet_address == "0xtest_wallet"
            assert profile.total_positions == 2
            assert profile.unique_markets == 2
            assert profile.trade_count == 3
            assert profile.split_count == 1
            assert profile.account_age_days == 10
            assert profile.win_rate == 1.0  # Both positions won
            assert profile.realized_pnl == Decimal("60")
            assert len(profile.trades) == 3
            assert len(profile.activities) == 4

        run_async(_test())

    def test_module_exports(self):
        """Test that Stage 2 classes are exported from module."""
        from src.insider_scanner import (
            ProfileFetcher,
            WalletProfile,
            PositionSummary,
            TradeEntry,
        )

        assert ProfileFetcher is not None
        assert WalletProfile is not None
        assert PositionSummary is not None
        assert TradeEntry is not None
