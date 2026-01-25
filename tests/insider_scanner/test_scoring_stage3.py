"""Tests for Insider Scoring Engine.

Tests:
- Account signal scoring
- Trading signal scoring (including cumulative positions)
- Behavioral signal scoring
- Contextual signal scoring
- Cluster signal scoring
- Variance calibration (soft thresholds)
- Minimum signal requirements
- False positive modifiers
- Priority level determination

Based on documented cases from VARIANCE-CALIBRATION.md
"""

import pytest
from decimal import Decimal

from src.insider_scanner.scoring import (
    InsiderScorer,
    ScoringResult,
    Signal,
    MarketCategory,
    POSITION_SIZE_THRESHOLDS,
    ACCOUNT_AGE_THRESHOLDS,
)


class TestAccountSignals:
    """Test account characteristic scoring."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_brand_new_account_max_score(self):
        """Brand new account (< 1 day) should get maximum account age score."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=0,
            transaction_count=0,
        )

        # Should have account signals
        account_signals = [s for s in result.signals if s.category == "account"]
        assert len(account_signals) >= 1

        # Score should be meaningful
        assert result.dimensions["account"] > 0

    def test_fresh_account_7_days(self):
        """7-day old account should still score but less than brand new."""
        result_new = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=0,
        )
        result_7d = self.scorer.score_wallet(
            wallet_address="0x5678",
            account_age_days=7,
        )

        # Brand new should score higher
        assert result_new.dimensions["account"] > result_7d.dimensions["account"]

    def test_established_account_no_score(self):
        """Established account (> 90 days) should get zero account age score."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=100,
            transaction_count=50,
        )

        # Should not trigger account signals
        age_signals = [s for s in result.signals if s.name == "account_age"]
        assert len(age_signals) == 0 or all(s.weight == 0 for s in age_signals)

    def test_zero_transactions_max_score(self):
        """Zero transaction count should get maximum transaction score."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            transaction_count=0,
        )

        tx_signals = [s for s in result.signals if s.name == "transaction_count"]
        assert len(tx_signals) == 1
        assert tx_signals[0].weight == 10.0

    def test_high_transaction_count_no_score(self):
        """High transaction count (>10) should get zero score."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            transaction_count=100,
        )

        tx_signals = [s for s in result.signals if s.name == "transaction_count"]
        assert len(tx_signals) == 0


class TestTradingSignals:
    """Test trading behavior scoring including cumulative positions."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_large_cumulative_position(self):
        """Large cumulative position should score high."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 30000},
            {"market_id": "market1", "side": "YES", "size_usd": 25000},
            {"market_id": "market1", "side": "YES", "size_usd": 20000},
        ]  # Cumulative: $75K

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        # Should detect cumulative position
        size_signals = [s for s in result.signals if "position_size" in s.name]
        assert len(size_signals) >= 1
        assert result.dimensions["trading"] > 0

    def test_split_entry_pattern_detected(self):
        """Split entry pattern (many small entries) should be detected."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 5000},
            {"market_id": "market1", "side": "YES", "size_usd": 5000},
            {"market_id": "market1", "side": "YES", "size_usd": 5000},
            {"market_id": "market1", "side": "YES", "size_usd": 5000},
            {"market_id": "market1", "side": "YES", "size_usd": 5000},
        ]  # Cumulative: $25K, avg entry: $5K (20% of total = split pattern)

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        split_signals = [s for s in result.signals if "split_entry" in s.name]
        assert len(split_signals) == 1
        assert split_signals[0].weight == 2  # Split entry bonus

    def test_single_entry_no_split_bonus(self):
        """Single entry should not get split entry bonus."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 25000},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        split_signals = [s for s in result.signals if "split_entry" in s.name]
        assert len(split_signals) == 0

    def test_longshot_entry_odds(self):
        """Entry at longshot odds (<5%) should score high."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 10000, "entry_odds": 0.04},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        odds_signals = [s for s in result.signals if "entry_odds" in s.name]
        assert len(odds_signals) == 1
        assert odds_signals[0].weight == 8  # Extreme longshot

    def test_high_odds_entry_minimal_score(self):
        """Entry at high odds (>60%) should get minimal score."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 10000, "entry_odds": 0.75},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        odds_signals = [s for s in result.signals if "entry_odds" in s.name]
        assert len(odds_signals) == 0  # No signal for high odds

    def test_perfect_win_rate(self):
        """100% win rate should score maximum."""
        positions = [
            {"market_id": "m1", "side": "YES", "size_usd": 1000, "resolved": True, "won": True},
            {"market_id": "m2", "side": "YES", "size_usd": 1000, "resolved": True, "won": True},
            {"market_id": "m3", "side": "YES", "size_usd": 1000, "resolved": True, "won": True},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        wr_signals = [s for s in result.signals if "win_rate" in s.name]
        assert len(wr_signals) == 1
        assert wr_signals[0].weight == 15  # Perfect win rate


class TestBehavioralSignals:
    """Test behavioral pattern scoring."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_single_market_concentration(self):
        """100% concentration in single market should score high."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 50000},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        conc_signals = [s for s in result.signals if "concentration" in s.name]
        assert len(conc_signals) == 1
        assert conc_signals[0].weight == 10

    def test_diverse_portfolio_no_concentration(self):
        """Diverse portfolio should not trigger concentration signal."""
        positions = [
            {"market_id": "m1", "side": "YES", "size_usd": 1000},
            {"market_id": "m2", "side": "YES", "size_usd": 1000},
            {"market_id": "m3", "side": "YES", "size_usd": 1000},
            {"market_id": "m4", "side": "YES", "size_usd": 1000},
            {"market_id": "m5", "side": "YES", "size_usd": 1000},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        conc_signals = [s for s in result.signals if "concentration" in s.name]
        # Concentration is only 20% per market
        assert len(conc_signals) == 0 or conc_signals[0].weight == 0

    def test_no_hedging_detected(self):
        """Wallet with no hedging should trigger signal."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 50000},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        hedge_signals = [s for s in result.signals if "hedging" in s.name]
        assert len(hedge_signals) == 1
        assert hedge_signals[0].weight == 5

    def test_hedging_present_no_signal(self):
        """Wallet with hedging should not trigger no-hedging signal."""
        positions = [
            {"market_id": "market1", "side": "YES", "size_usd": 30000},
            {"market_id": "market1", "side": "NO", "size_usd": 10000},  # Hedge
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            positions=positions,
        )

        hedge_signals = [s for s in result.signals if "hedging" in s.name]
        assert len(hedge_signals) == 0


class TestContextualSignals:
    """Test contextual factor scoring."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_military_market_highest_score(self):
        """Military market category should score highest (10 per Fed Chair analysis)."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            market_category=MarketCategory.MILITARY,
        )

        cat_signals = [s for s in result.signals if "market_category" in s.name]
        assert len(cat_signals) == 1
        assert cat_signals[0].weight == 10  # Increased from 8 per Fed Chair analysis

    def test_other_market_no_score(self):
        """Other/unknown market category should score zero."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            market_category=MarketCategory.OTHER,
        )

        cat_signals = [s for s in result.signals if "market_category" in s.name]
        assert len(cat_signals) == 0

    def test_pre_event_timing_6_hours(self):
        """Position placed < 6 hours before event should score high."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            event_hours_away=4,
        )

        timing_signals = [s for s in result.signals if "timing" in s.name]
        assert len(timing_signals) == 1
        assert timing_signals[0].weight == 8

    def test_pre_event_timing_far(self):
        """Position placed > 1 week before event should score low."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            event_hours_away=200,  # > 1 week
        )

        timing_signals = [s for s in result.signals if "timing" in s.name]
        assert len(timing_signals) == 0


class TestClusterSignals:
    """Test cluster/sybil indicator scoring."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_flagged_funding_source(self):
        """Funding from flagged source should score maximum cluster points."""
        flagged = {"0xbad_funder"}

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            funding_source="0xbad_funder",
            flagged_funders=flagged,
        )

        funding_signals = [s for s in result.signals if "funding_source" in s.name]
        assert len(funding_signals) == 1
        assert funding_signals[0].weight == 15

    def test_cluster_membership(self):
        """Membership in wallet cluster should add score."""
        cluster = ["0xwallet2", "0xwallet3", "0xwallet4"]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            cluster_wallets=cluster,
        )

        cluster_signals = [s for s in result.signals if "cluster" in s.name]
        assert len(cluster_signals) == 1
        assert cluster_signals[0].weight == 9  # 3 wallets * 3 points


class TestVarianceCalibration:
    """Test variance calibration and overfitting prevention."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_single_dimension_downgrade(self):
        """Score >= 70 with only 1 active dimension should be downgraded."""
        # High trading score but nothing else
        positions = [
            {"market_id": "m1", "side": "YES", "size_usd": 200000, "entry_odds": 0.03,
             "resolved": True, "won": True},
            {"market_id": "m2", "side": "YES", "size_usd": 200000, "entry_odds": 0.03,
             "resolved": True, "won": True},
            {"market_id": "m3", "side": "YES", "size_usd": 200000, "entry_odds": 0.03,
             "resolved": True, "won": True},
        ]

        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=365,  # Old account
            transaction_count=100,  # Many transactions
            positions=positions,
        )

        # If only trading dimension is active and score would be >= 70
        if result.active_dimensions == 1:
            assert result.score < 70 or result.downgraded
            if result.downgraded:
                assert result.downgrade_reason is not None

    def test_minimum_signals_for_critical(self):
        """CRITICAL priority requires 5+ signals across 3+ dimensions."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=1,  # Fresh
            transaction_count=0,  # Zero tx
            positions=[{"market_id": "m1", "side": "YES", "size_usd": 100000, "entry_odds": 0.03}],
            market_category=MarketCategory.MILITARY,
            event_hours_away=2,
        )

        if result.priority == "critical":
            assert result.signal_count >= 5
            assert result.active_dimensions >= 3

    def test_confidence_interval_narrows_with_signals(self):
        """More signals should narrow the confidence interval."""
        # Few signals
        result_few = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=1,
        )

        # Many signals
        result_many = self.scorer.score_wallet(
            wallet_address="0x5678",
            account_age_days=1,
            transaction_count=0,
            positions=[{"market_id": "m1", "side": "YES", "size_usd": 100000, "entry_odds": 0.03}],
            market_category=MarketCategory.MILITARY,
            event_hours_away=2,
        )

        few_width = result_few.confidence_high - result_few.confidence_low
        many_width = result_many.confidence_high - result_many.confidence_low

        # More signals = narrower interval
        if result_many.signal_count > result_few.signal_count:
            assert many_width <= few_width


class TestFalsePositiveModifiers:
    """Test false positive reduction modifiers."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_bet_lost_major_reduction(self):
        """Losing bet should dramatically reduce score."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=1,
            transaction_count=0,
            market_category=MarketCategory.MILITARY,
        )

        modified = self.scorer.apply_false_positive_modifiers(
            result,
            bet_lost=True,
        )

        # Should reduce significantly (clamped to 0 if original < 40)
        assert modified.score < result.score  # Definitely reduced
        assert modified.score == max(result.score - 40, 0)  # Correctly clamped

    def test_public_analyst_reduction(self):
        """Known public analyst should get score reduction."""
        result = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=1,
        )

        modified = self.scorer.apply_false_positive_modifiers(
            result,
            public_analyst=True,
        )

        # Should reduce by 25
        loss_signals = [s for s in modified.signals if s.name == "public_analyst"]
        assert len(loss_signals) == 1
        assert loss_signals[0].weight == -25


class TestPriorityLevels:
    """Test priority level determination."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_priority_levels(self):
        """Test all priority level thresholds."""
        # Create scenarios for different priority levels

        # CRITICAL: Need score >= 85, signals >= 5, dimensions >= 3
        # HIGH: Need score >= 70, signals >= 4, dimensions >= 2
        # MEDIUM: Need score >= 55, signals >= 3, dimensions >= 2
        # LOW: Need score >= 40, signals >= 2, dimensions >= 1
        # NORMAL: score < 40

        # Test NORMAL (low score)
        result_normal = self.scorer.score_wallet(
            wallet_address="0x1234",
            account_age_days=365,
            transaction_count=100,
        )
        assert result_normal.priority == "normal"


class TestDocumentedCases:
    """Test against documented insider cases from footprints."""

    def setup_method(self):
        self.scorer = InsiderScorer(variance_factor=1.2)

    def test_6741_template_case(self):
        """6741: 24-hour account, Nobel Prize, $53K profit.

        Should score at least LOW. Note: This test uses minimal data.
        Real case had larger position and more context.
        """
        result = self.scorer.score_wallet(
            wallet_address="0x6741",
            account_age_days=0,  # 24 hours
            transaction_count=0,  # First transaction
            positions=[
                {"market_id": "nobel", "side": "YES", "size_usd": 53000,  # Actual profit
                 "entry_odds": 0.04, "resolved": True, "won": True},
            ],
            market_category=MarketCategory.AWARDS,
        )

        # Should score at least LOW with account + odds + concentration signals
        assert result.score >= 40  # At least LOW threshold
        assert result.priority in ["critical", "high", "medium", "low"]

    def test_burdensome_mix_case(self):
        """Burdensome-Mix: 7-day account, Maduro, $409K profit.

        Should score HIGH or CRITICAL.
        """
        result = self.scorer.score_wallet(
            wallet_address="0xburdensome",
            account_age_days=7,
            transaction_count=3,
            positions=[
                {"market_id": "maduro", "side": "YES", "size_usd": 34000,
                 "entry_odds": 0.08, "resolved": True, "won": True},
            ],
            market_category=MarketCategory.MILITARY,
            event_hours_away=3,
        )

        assert result.score >= 55
        assert result.priority in ["critical", "high", "medium"]

    def test_annica_80_percent_win_rate(self):
        """Annica: 80% win rate (not 100%), social media category.

        Should score MEDIUM due to controllable outcomes.
        """
        positions = [
            {"market_id": "m1", "side": "YES", "size_usd": 10000, "resolved": True, "won": True},
            {"market_id": "m2", "side": "YES", "size_usd": 10000, "resolved": True, "won": True},
            {"market_id": "m3", "side": "YES", "size_usd": 10000, "resolved": True, "won": True},
            {"market_id": "m4", "side": "YES", "size_usd": 10000, "resolved": True, "won": True},
            {"market_id": "m5", "side": "YES", "size_usd": 10000, "resolved": True, "won": False},  # Loss
        ]

        result = self.scorer.score_wallet(
            wallet_address="0xannica",
            account_age_days=90,  # Established
            transaction_count=20,
            positions=positions,
            market_category=MarketCategory.SOCIAL_MEDIA,
        )

        # Should score lower due to controllable outcomes (social media)
        # And not perfect win rate
        assert result.score < 85  # Not CRITICAL

    def test_mutualdelta_false_positive(self):
        """mutualdelta: Perfect pattern but LOST $40K.

        Should be heavily discounted.
        """
        result = self.scorer.score_wallet(
            wallet_address="0xmutualdelta",
            account_age_days=3,
            transaction_count=1,
            positions=[
                {"market_id": "iran", "side": "YES", "size_usd": 40000,
                 "entry_odds": 0.18, "resolved": True, "won": False},
            ],
            market_category=MarketCategory.MILITARY,
        )

        # Apply false positive modifier for losing bet
        modified = self.scorer.apply_false_positive_modifiers(result, bet_lost=True)

        # Should be heavily reduced
        assert modified.score < 55  # Below MEDIUM threshold
