"""Database migrations for Insider Scanner.

Adds tables for:
- Flagged wallets with scoring
- Flagged funding sources
- Insider clusters
- Detection signals
- Cumulative positions
- Audit trail (detection records, investment thesis, audit chain)
"""

from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database.migrations import Migration


def get_insider_scanner_migrations() -> list[Migration]:
    """Get all migrations for insider scanner module.

    Returns:
        List of Migration objects
    """
    migrations = []

    # =========================================================================
    # Migration 100: Create insider_flagged_wallets table
    # =========================================================================
    def up_100(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_flagged_wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address VARCHAR(42) NOT NULL UNIQUE,

                -- Scoring
                insider_score NUMERIC(10, 4) NOT NULL DEFAULT 0,
                confidence_low NUMERIC(10, 4),
                confidence_high NUMERIC(10, 4),
                priority VARCHAR(20) DEFAULT 'normal',

                -- Dimension scores
                score_account NUMERIC(10, 4) DEFAULT 0,
                score_trading NUMERIC(10, 4) DEFAULT 0,
                score_behavioral NUMERIC(10, 4) DEFAULT 0,
                score_contextual NUMERIC(10, 4) DEFAULT 0,
                score_cluster NUMERIC(10, 4) DEFAULT 0,

                -- Signal tracking
                signal_count INTEGER DEFAULT 0,
                active_dimensions INTEGER DEFAULT 0,
                signals_json TEXT,

                -- Variance tracking
                downgraded BOOLEAN DEFAULT 0,
                downgrade_reason TEXT,

                -- Status
                status VARCHAR(20) DEFAULT 'new',

                -- Cached account info
                account_age_days INTEGER,
                transaction_count INTEGER,
                first_seen TIMESTAMP,

                -- Cached position info
                total_position_usd NUMERIC(20, 6),
                largest_position_usd NUMERIC(20, 6),
                win_rate NUMERIC(10, 6),

                -- Cluster association
                cluster_id INTEGER,
                funding_source_id INTEGER,

                -- Notes
                notes TEXT,

                -- Timing
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """))

        # Create indexes
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_wallet_score ON insider_flagged_wallets(insider_score)"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_wallet_priority ON insider_flagged_wallets(priority)"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_wallet_status ON insider_flagged_wallets(status)"
        ))

    def down_100(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_flagged_wallets"))

    migrations.append(Migration(
        version=100,
        name="create_insider_flagged_wallets",
        description="Create table for flagged insider trading wallets",
        up=up_100,
        down=down_100,
    ))

    # =========================================================================
    # Migration 101: Create insider_funding_sources table
    # =========================================================================
    def up_101(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_funding_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                funding_address VARCHAR(42) NOT NULL UNIQUE,

                -- Source info
                source_type VARCHAR(50),
                exchange_name VARCHAR(100),

                -- Association
                associated_wallet_count INTEGER DEFAULT 1,
                total_funded_usd NUMERIC(20, 6),

                -- Risk level
                risk_level VARCHAR(20) DEFAULT 'medium',

                -- Evidence
                reason TEXT,
                evidence_json TEXT,

                -- Timing
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Manual flag
                manually_flagged BOOLEAN DEFAULT 0,
                flagged_by VARCHAR(100)
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_funding_address ON insider_funding_sources(funding_address)"
        ))

    def down_101(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_funding_sources"))

    migrations.append(Migration(
        version=101,
        name="create_insider_funding_sources",
        description="Create table for flagged funding sources",
        up=up_101,
        down=down_101,
    ))

    # =========================================================================
    # Migration 102: Create insider_clusters table
    # =========================================================================
    def up_102(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Cluster identification
                cluster_name VARCHAR(100),
                cluster_type VARCHAR(50),

                -- Statistics
                wallet_count INTEGER DEFAULT 0,
                total_position_usd NUMERIC(20, 6),
                avg_score NUMERIC(10, 4),

                -- Detection
                detection_method TEXT,
                confidence NUMERIC(10, 4),

                -- Evidence
                evidence_json TEXT,

                -- Timing
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

    def down_102(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_clusters"))

    migrations.append(Migration(
        version=102,
        name="create_insider_clusters",
        description="Create table for insider wallet clusters",
        up=up_102,
        down=down_102,
    ))

    # =========================================================================
    # Migration 103: Create insider_signals table
    # =========================================================================
    def up_103(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_id INTEGER NOT NULL,

                -- Signal identification
                signal_name VARCHAR(100) NOT NULL,
                category VARCHAR(20) NOT NULL,

                -- Scoring
                weight NUMERIC(10, 4) NOT NULL,
                raw_value TEXT,
                threshold TEXT,

                -- Evidence
                evidence_json TEXT,

                -- Timing
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (wallet_id) REFERENCES insider_flagged_wallets(id)
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_signals_wallet ON insider_signals(wallet_id)"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_signals_category ON insider_signals(category)"
        ))

    def down_103(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_signals"))

    migrations.append(Migration(
        version=103,
        name="create_insider_signals",
        description="Create table for individual detection signals",
        up=up_103,
        down=down_103,
    ))

    # =========================================================================
    # Migration 104: Create insider_cumulative_positions table
    # =========================================================================
    def up_104(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_cumulative_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_id INTEGER NOT NULL,

                -- Position identification
                market_id VARCHAR(66) NOT NULL,
                market_title TEXT,
                side VARCHAR(4) NOT NULL,

                -- Cumulative totals
                cumulative_size NUMERIC(30, 18) NOT NULL DEFAULT 0,
                cumulative_usd NUMERIC(20, 6) NOT NULL DEFAULT 0,
                entry_count INTEGER DEFAULT 0,
                avg_entry_size NUMERIC(20, 6),

                -- Split entry detection
                is_split_entry BOOLEAN DEFAULT 0,
                split_entry_bonus NUMERIC(10, 4) DEFAULT 0,

                -- Entry odds tracking
                first_entry_odds NUMERIC(10, 6),
                avg_entry_odds NUMERIC(10, 6),

                -- Outcome
                outcome_resolved BOOLEAN DEFAULT 0,
                outcome_won BOOLEAN,
                realized_pnl NUMERIC(20, 6),
                roi_percent NUMERIC(10, 4),

                -- Timing
                first_entry_at TIMESTAMP,
                last_entry_at TIMESTAMP,
                resolved_at TIMESTAMP,

                FOREIGN KEY (wallet_id) REFERENCES insider_flagged_wallets(id),
                UNIQUE (wallet_id, market_id, side)
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_position_wallet ON insider_cumulative_positions(wallet_id)"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_insider_position_market ON insider_cumulative_positions(market_id)"
        ))

    def down_104(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_cumulative_positions"))

    migrations.append(Migration(
        version=104,
        name="create_insider_cumulative_positions",
        description="Create table for cumulative position tracking",
        up=up_104,
        down=down_104,
    ))

    # =========================================================================
    # Migration 105: Create insider_detection_records table (Audit Trail)
    # =========================================================================
    def up_105(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_id INTEGER NOT NULL,

                -- Cryptographic hash
                record_hash VARCHAR(64) NOT NULL UNIQUE,

                -- Detection data (immutable snapshot)
                wallet_address VARCHAR(42) NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                insider_score NUMERIC(10, 4) NOT NULL,
                priority VARCHAR(20) NOT NULL,

                -- Full signal breakdown
                signals_snapshot TEXT NOT NULL,

                -- Market context
                market_ids TEXT,
                market_positions TEXT,

                -- Raw API response
                raw_api_snapshot TEXT,

                -- Hash chain link
                previous_record_hash VARCHAR(64),

                -- Blockchain anchoring
                anchor_tx_id VARCHAR(66),
                anchor_type VARCHAR(20),
                anchored_at TIMESTAMP,

                FOREIGN KEY (wallet_id) REFERENCES insider_flagged_wallets(id)
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_detection_wallet ON insider_detection_records(wallet_id)"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_detection_time ON insider_detection_records(detected_at)"
        ))

    def down_105(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_detection_records"))

    migrations.append(Migration(
        version=105,
        name="create_insider_detection_records",
        description="Create immutable detection records for audit trail",
        up=up_105,
        down=down_105,
    ))

    # =========================================================================
    # Migration 106: Create insider_investment_thesis table (Audit Trail)
    # =========================================================================
    def up_106(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_investment_thesis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Cryptographic hash
                thesis_hash VARCHAR(64) NOT NULL UNIQUE,

                -- Timing
                created_at TIMESTAMP NOT NULL,

                -- Linked detections
                detection_record_ids TEXT NOT NULL,

                -- Reasoning
                reasoning TEXT NOT NULL,

                -- Intended action
                intended_action TEXT,
                market_id VARCHAR(66),
                position_side VARCHAR(4),
                position_size NUMERIC(20, 6),

                -- Actual outcome
                action_taken BOOLEAN DEFAULT 0,
                actual_position_size NUMERIC(20, 6),
                actual_outcome TEXT,

                -- Anchoring
                anchor_tx_id VARCHAR(66),
                anchor_type VARCHAR(20),
                anchored_at TIMESTAMP
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_thesis_created ON insider_investment_thesis(created_at)"
        ))

    def down_106(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_investment_thesis"))

    migrations.append(Migration(
        version=106,
        name="create_insider_investment_thesis",
        description="Create investment thesis table for audit trail",
        up=up_106,
        down=down_106,
    ))

    # =========================================================================
    # Migration 107: Create insider_audit_chain table
    # =========================================================================
    def up_107(session: Session) -> None:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS insider_audit_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_number INTEGER NOT NULL,

                -- Entry reference
                entry_type VARCHAR(20) NOT NULL,
                entry_id INTEGER NOT NULL,
                entry_hash VARCHAR(64) NOT NULL,

                -- Chain linkage
                previous_chain_hash VARCHAR(64),
                chain_hash VARCHAR(64) NOT NULL UNIQUE,

                -- Anchoring
                anchored_at TIMESTAMP,
                anchor_proof TEXT,

                -- Timing
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chain_sequence ON insider_audit_chain(sequence_number)"
        ))

    def down_107(session: Session) -> None:
        session.execute(text("DROP TABLE IF EXISTS insider_audit_chain"))

    migrations.append(Migration(
        version=107,
        name="create_insider_audit_chain",
        description="Create audit chain table for tamper-evident record linking",
        up=up_107,
        down=down_107,
    ))

    return migrations


def register_insider_migrations(manager) -> None:
    """Register all insider scanner migrations with a migration manager.

    Args:
        manager: MigrationManager instance
    """
    for migration in get_insider_scanner_migrations():
        manager.register(migration)
