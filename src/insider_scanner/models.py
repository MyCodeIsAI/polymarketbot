"""Database models for Insider Scanner.

Includes:
- Flagged wallets and funding sources
- Detection records with signal breakdowns
- Audit trail for legal protection
- Cumulative position tracking
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
import hashlib
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Numeric,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
    Enum as SQLEnum,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Use same base as main app
from ..database.models import Base


class InsiderPriority(str, Enum):
    """Priority level for flagged wallets."""
    CRITICAL = "critical"  # Score 85+, immediate alert
    HIGH = "high"          # Score 70-84, add to watchlist + alert
    MEDIUM = "medium"      # Score 55-69, add to watchlist
    LOW = "low"            # Score 40-54, monitor passively
    NORMAL = "normal"      # Score < 40, no action


class FlagType(str, Enum):
    """Type of flag for a detected wallet."""
    INSIDER = "insider"    # Suspected insider trading (pattern-based)
    SYBIL = "sybil"        # Known profitable trader's new account (funding match)
    UNKNOWN = "unknown"    # Not yet classified


class WalletStatus(str, Enum):
    """Status of a flagged wallet in the monitoring pipeline."""
    NEW = "new"                    # Just detected
    MONITORING = "monitoring"      # Being actively tracked
    ESCALATED = "escalated"        # Manually escalated for review
    CONFIRMED = "confirmed"        # Confirmed suspicious
    CLEARED = "cleared"            # False positive, cleared
    ARCHIVED = "archived"          # No longer active


class SignalCategory(str, Enum):
    """Categories of detection signals."""
    ACCOUNT = "account"
    TRADING = "trading"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    CLUSTER = "cluster"


class FlaggedWallet(Base):
    """A wallet flagged for suspicious activity.

    This is the primary table for tracking potential insider traders.
    Each wallet has a score, signals breakdown, and status.
    """
    __tablename__ = "insider_flagged_wallets"
    __table_args__ = (
        UniqueConstraint("wallet_address", name="uq_insider_wallet"),
        Index("idx_insider_wallet_score", "insider_score"),
        Index("idx_insider_wallet_priority", "priority"),
        Index("idx_insider_wallet_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(42), nullable=False)

    # Flag type (SYBIL vs INSIDER)
    flag_type = Column(SQLEnum(FlagType), default=FlagType.UNKNOWN)
    linked_trader_id = Column(Integer, ForeignKey("known_profitable_traders.id"), nullable=True)
    funding_match_address = Column(String(42), nullable=True)  # The address that triggered the match

    # Scoring
    insider_score = Column(Numeric(10, 4), nullable=False, default=Decimal("0"))
    confidence_low = Column(Numeric(10, 4), nullable=True)
    confidence_high = Column(Numeric(10, 4), nullable=True)
    priority = Column(SQLEnum(InsiderPriority), default=InsiderPriority.NORMAL)

    # Dimension scores (max: 25, 35, 25, 20, 20)
    score_account = Column(Numeric(10, 4), default=Decimal("0"))
    score_trading = Column(Numeric(10, 4), default=Decimal("0"))
    score_behavioral = Column(Numeric(10, 4), default=Decimal("0"))
    score_contextual = Column(Numeric(10, 4), default=Decimal("0"))
    score_cluster = Column(Numeric(10, 4), default=Decimal("0"))

    # Signal tracking
    signal_count = Column(Integer, default=0)
    active_dimensions = Column(Integer, default=0)
    signals_json = Column(JSON, nullable=True)  # Array of triggered signals

    # Variance tracking
    downgraded = Column(Boolean, default=False)  # Single-dimension downgrade
    downgrade_reason = Column(Text, nullable=True)

    # Status
    status = Column(SQLEnum(WalletStatus), default=WalletStatus.NEW)

    # Account info (cached)
    account_age_days = Column(Integer, nullable=True)
    transaction_count = Column(Integer, nullable=True)
    first_seen = Column(DateTime, nullable=True)

    # Position info (cached)
    total_position_usd = Column(Numeric(20, 6), nullable=True)
    largest_position_usd = Column(Numeric(20, 6), nullable=True)
    win_rate = Column(Numeric(10, 6), nullable=True)

    # Cluster association
    cluster_id = Column(Integer, ForeignKey("insider_clusters.id"), nullable=True)
    funding_source_id = Column(Integer, ForeignKey("insider_funding_sources.id"), nullable=True)

    # User notes
    notes = Column(Text, nullable=True)

    # Timing
    detected_at = Column(DateTime, default=func.now())
    last_scored_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    detection_records = relationship("DetectionRecord", back_populates="wallet", lazy="dynamic")
    positions = relationship("CumulativePosition", back_populates="wallet", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<FlaggedWallet(addr='{self.wallet_address[:10]}...', score={self.insider_score}, priority='{self.priority.value}')>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "wallet_address": self.wallet_address,
            "flag_type": self.flag_type.value if self.flag_type else "unknown",
            "linked_trader_id": self.linked_trader_id,
            "funding_match_address": self.funding_match_address,
            "insider_score": float(self.insider_score) if self.insider_score else 0,
            "confidence_low": float(self.confidence_low) if self.confidence_low else None,
            "confidence_high": float(self.confidence_high) if self.confidence_high else None,
            "priority": self.priority.value,
            "dimensions": {
                "account": float(self.score_account) if self.score_account else 0,
                "trading": float(self.score_trading) if self.score_trading else 0,
                "behavioral": float(self.score_behavioral) if self.score_behavioral else 0,
                "contextual": float(self.score_contextual) if self.score_contextual else 0,
                "cluster": float(self.score_cluster) if self.score_cluster else 0,
            },
            "signal_count": self.signal_count,
            "active_dimensions": self.active_dimensions,
            "signals": self.signals_json,
            "downgraded": self.downgraded,
            "status": self.status.value,
            "account_age_days": self.account_age_days,
            "transaction_count": self.transaction_count,
            "total_position_usd": float(self.total_position_usd) if self.total_position_usd else None,
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "notes": self.notes,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }


class FlaggedFundingSource(Base):
    """A funding source associated with suspicious wallets.

    When multiple wallets share the same funding source, they're likely
    controlled by the same entity. New wallets funded from flagged sources
    get immediately flagged.
    """
    __tablename__ = "insider_funding_sources"
    __table_args__ = (
        UniqueConstraint("funding_address", name="uq_insider_funding"),
        Index("idx_insider_funding_address", "funding_address"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    funding_address = Column(String(42), nullable=False)

    # Source info
    source_type = Column(String(50), nullable=True)  # CEX, DEX, bridge, unknown
    exchange_name = Column(String(100), nullable=True)  # Coinbase, Kraken, etc.

    # Association
    associated_wallet_count = Column(Integer, default=1)
    total_funded_usd = Column(Numeric(20, 6), nullable=True)

    # Risk level
    risk_level = Column(String(20), default="medium")  # low, medium, high, critical

    # Evidence
    reason = Column(Text, nullable=True)
    evidence_json = Column(JSON, nullable=True)  # Links, screenshots, etc.

    # Timing
    first_seen = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())

    # Manual flag
    manually_flagged = Column(Boolean, default=False)
    flagged_by = Column(String(100), nullable=True)

    def __repr__(self) -> str:
        return f"<FlaggedFundingSource(addr='{self.funding_address[:10]}...', count={self.associated_wallet_count})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "funding_address": self.funding_address,
            "source_type": self.source_type,
            "exchange_name": self.exchange_name,
            "associated_wallet_count": self.associated_wallet_count,
            "total_funded_usd": float(self.total_funded_usd) if self.total_funded_usd else None,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "manually_flagged": self.manually_flagged,
        }


class InsiderCluster(Base):
    """A cluster of related wallets (same funding, synchronized trading).

    Clusters indicate coordinated activity - multiple wallets controlled
    by the same entity to evade detection.
    """
    __tablename__ = "insider_clusters"
    __table_args__ = (
        Index("idx_insider_cluster_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Cluster identification
    cluster_name = Column(String(100), nullable=True)
    cluster_type = Column(String(50), nullable=True)  # funding_match, timing_sync, market_overlap

    # Statistics
    wallet_count = Column(Integer, default=0)
    total_position_usd = Column(Numeric(20, 6), nullable=True)
    avg_score = Column(Numeric(10, 4), nullable=True)

    # Detection method
    detection_method = Column(Text, nullable=True)
    confidence = Column(Numeric(10, 4), nullable=True)

    # Evidence
    evidence_json = Column(JSON, nullable=True)

    # Timing
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    wallets = relationship("FlaggedWallet", backref="cluster", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cluster_name": self.cluster_name,
            "cluster_type": self.cluster_type,
            "wallet_count": self.wallet_count,
            "total_position_usd": float(self.total_position_usd) if self.total_position_usd else None,
            "avg_score": float(self.avg_score) if self.avg_score else None,
            "detection_method": self.detection_method,
        }


class InsiderSignal(Base):
    """Individual detection signal triggered for a wallet.

    Each signal has a category, weight, and evidence. Signals are
    aggregated to calculate the composite insider score.
    """
    __tablename__ = "insider_signals"
    __table_args__ = (
        Index("idx_insider_signals_wallet", "wallet_id"),
        Index("idx_insider_signals_category", "category"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(Integer, ForeignKey("insider_flagged_wallets.id"), nullable=False)

    # Signal identification
    signal_name = Column(String(100), nullable=False)
    category = Column(SQLEnum(SignalCategory), nullable=False)

    # Scoring
    weight = Column(Numeric(10, 4), nullable=False)
    raw_value = Column(Text, nullable=True)  # The actual measured value
    threshold = Column(Text, nullable=True)  # What threshold was exceeded

    # Evidence
    evidence_json = Column(JSON, nullable=True)

    # Timing
    detected_at = Column(DateTime, default=func.now())

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "category": self.category.value,
            "weight": float(self.weight),
            "raw_value": self.raw_value,
            "threshold": self.threshold,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }


class CumulativePosition(Base):
    """Cumulative position tracking for a wallet on a specific market.

    Tracks total position across multiple entries to detect split-entry
    evasion patterns. Sum of all YES or NO entries.
    """
    __tablename__ = "insider_cumulative_positions"
    __table_args__ = (
        UniqueConstraint("wallet_id", "market_id", "side", name="uq_insider_position"),
        Index("idx_insider_position_wallet", "wallet_id"),
        Index("idx_insider_position_market", "market_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(Integer, ForeignKey("insider_flagged_wallets.id"), nullable=False)

    # Position identification
    market_id = Column(String(66), nullable=False)
    market_title = Column(Text, nullable=True)
    side = Column(String(4), nullable=False)  # YES or NO

    # Cumulative totals
    cumulative_size = Column(Numeric(30, 18), nullable=False, default=Decimal("0"))
    cumulative_usd = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))
    entry_count = Column(Integer, default=0)
    avg_entry_size = Column(Numeric(20, 6), nullable=True)

    # Split entry detection
    is_split_entry = Column(Boolean, default=False)
    split_entry_bonus = Column(Numeric(10, 4), default=Decimal("0"))

    # Entry odds tracking
    first_entry_odds = Column(Numeric(10, 6), nullable=True)
    avg_entry_odds = Column(Numeric(10, 6), nullable=True)

    # Outcome (if resolved)
    outcome_resolved = Column(Boolean, default=False)
    outcome_won = Column(Boolean, nullable=True)
    realized_pnl = Column(Numeric(20, 6), nullable=True)
    roi_percent = Column(Numeric(10, 4), nullable=True)

    # Timing
    first_entry_at = Column(DateTime, nullable=True)
    last_entry_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Relationship
    wallet = relationship("FlaggedWallet", back_populates="positions")

    def __repr__(self) -> str:
        return f"<CumulativePosition(market='{self.market_id[:16]}...', side='{self.side}', total=${self.cumulative_usd})>"

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "market_title": self.market_title,
            "side": self.side,
            "cumulative_usd": float(self.cumulative_usd) if self.cumulative_usd else 0,
            "entry_count": self.entry_count,
            "avg_entry_size": float(self.avg_entry_size) if self.avg_entry_size else None,
            "is_split_entry": self.is_split_entry,
            "first_entry_odds": float(self.first_entry_odds) if self.first_entry_odds else None,
            "outcome_won": self.outcome_won,
            "roi_percent": float(self.roi_percent) if self.roi_percent else None,
        }


# =============================================================================
# AUDIT TRAIL MODELS (Stage 8 - Legal Protection)
# =============================================================================


class DetectionRecord(Base):
    """Immutable detection record with cryptographic hash.

    Created at the MOMENT of detection, BEFORE any action is taken.
    Provides legal proof of when and how we detected the wallet.
    """
    __tablename__ = "insider_detection_records"
    __table_args__ = (
        UniqueConstraint("record_hash", name="uq_detection_hash"),
        Index("idx_detection_wallet", "wallet_id"),
        Index("idx_detection_time", "detected_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(Integer, ForeignKey("insider_flagged_wallets.id"), nullable=False)

    # Cryptographic hash of all fields (SHA-256)
    record_hash = Column(String(64), nullable=False)

    # Detection data (immutable snapshot)
    wallet_address = Column(String(42), nullable=False)
    detected_at = Column(DateTime, nullable=False)
    insider_score = Column(Numeric(10, 4), nullable=False)
    priority = Column(String(20), nullable=False)

    # Full signal breakdown (JSON)
    signals_snapshot = Column(JSON, nullable=False)

    # Market context at time of detection
    market_ids = Column(JSON, nullable=True)  # Array of market IDs
    market_positions = Column(JSON, nullable=True)  # Position sizes at detection

    # Raw API response (proof of data source)
    raw_api_snapshot = Column(JSON, nullable=True)

    # Hash chain (links to previous record)
    previous_record_hash = Column(String(64), nullable=True)

    # Blockchain anchoring (optional, strongest proof)
    anchor_tx_id = Column(String(66), nullable=True)
    anchor_type = Column(String(20), nullable=True)  # polygon, opentimestamps, local
    anchored_at = Column(DateTime, nullable=True)

    # Relationship
    wallet = relationship("FlaggedWallet", back_populates="detection_records")

    def __repr__(self) -> str:
        return f"<DetectionRecord(wallet='{self.wallet_address[:10]}...', hash='{self.record_hash[:16]}...')>"

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of record data."""
        data = {
            "wallet_address": self.wallet_address,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "insider_score": str(self.insider_score),
            "priority": self.priority,
            "signals_snapshot": self.signals_snapshot,
            "market_ids": self.market_ids,
            "market_positions": self.market_positions,
            "previous_record_hash": self.previous_record_hash,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "record_hash": self.record_hash,
            "wallet_address": self.wallet_address,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "insider_score": float(self.insider_score),
            "priority": self.priority,
            "signals_snapshot": self.signals_snapshot,
            "market_ids": self.market_ids,
            "anchor_tx_id": self.anchor_tx_id,
            "anchor_type": self.anchor_type,
            "anchored_at": self.anchored_at.isoformat() if self.anchored_at else None,
        }


class InvestmentThesis(Base):
    """Documents YOUR investment thesis when acting on a detection.

    Links to detection records to prove you got information from
    the detection system, not from insider knowledge.
    """
    __tablename__ = "insider_investment_thesis"
    __table_args__ = (
        UniqueConstraint("thesis_hash", name="uq_thesis_hash"),
        Index("idx_thesis_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Cryptographic hash
    thesis_hash = Column(String(64), nullable=False)

    # Timing
    created_at = Column(DateTime, nullable=False)

    # Linked detections (JSON array of detection record IDs)
    detection_record_ids = Column(JSON, nullable=False)

    # Your reasoning
    reasoning = Column(Text, nullable=False)

    # Intended action
    intended_action = Column(Text, nullable=True)  # "Place $X on YES for market Y"
    market_id = Column(String(66), nullable=True)
    position_side = Column(String(4), nullable=True)  # YES or NO
    position_size = Column(Numeric(20, 6), nullable=True)

    # Actual outcome (filled in after)
    action_taken = Column(Boolean, default=False)
    actual_position_size = Column(Numeric(20, 6), nullable=True)
    actual_outcome = Column(Text, nullable=True)

    # Anchoring
    anchor_tx_id = Column(String(66), nullable=True)
    anchor_type = Column(String(20), nullable=True)
    anchored_at = Column(DateTime, nullable=True)

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of thesis data."""
        data = {
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "detection_record_ids": self.detection_record_ids,
            "reasoning": self.reasoning,
            "intended_action": self.intended_action,
            "market_id": self.market_id,
            "position_side": self.position_side,
            "position_size": str(self.position_size) if self.position_size else None,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thesis_hash": self.thesis_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "detection_record_ids": self.detection_record_ids,
            "reasoning": self.reasoning,
            "intended_action": self.intended_action,
            "market_id": self.market_id,
            "action_taken": self.action_taken,
            "anchor_tx_id": self.anchor_tx_id,
        }


class AuditChainEntry(Base):
    """Entry in the audit hash chain.

    Links all detection records and investment theses in a tamper-evident
    chain. Any modification to historical records breaks the chain.
    """
    __tablename__ = "insider_audit_chain"
    __table_args__ = (
        UniqueConstraint("chain_hash", name="uq_chain_hash"),
        Index("idx_chain_sequence", "sequence_number"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sequence_number = Column(Integer, nullable=False)

    # Entry reference
    entry_type = Column(String(20), nullable=False)  # detection, thesis
    entry_id = Column(Integer, nullable=False)
    entry_hash = Column(String(64), nullable=False)

    # Chain linkage
    previous_chain_hash = Column(String(64), nullable=True)  # NULL for first entry
    chain_hash = Column(String(64), nullable=False)  # Hash of this entry + previous

    # Anchoring
    anchored_at = Column(DateTime, nullable=True)
    anchor_proof = Column(JSON, nullable=True)

    # Timing
    created_at = Column(DateTime, default=func.now())

    def calculate_chain_hash(self) -> str:
        """Calculate chain hash = SHA-256(entry_hash + previous_chain_hash)."""
        data = f"{self.entry_hash}:{self.previous_chain_hash or 'GENESIS'}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sequence_number": self.sequence_number,
            "entry_type": self.entry_type,
            "entry_id": self.entry_id,
            "entry_hash": self.entry_hash,
            "chain_hash": self.chain_hash,
            "previous_chain_hash": self.previous_chain_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# PROFITABLE TRADER SYBIL DETECTION MODELS
# =============================================================================


class KnownProfitableTrader(Base):
    """A known profitable Polymarket trader from the exported dataset.

    Used to detect sybil accounts - when a new account is funded from
    a known profitable trader's funding source or withdrawal destination.
    """
    __tablename__ = "known_profitable_traders"
    __table_args__ = (
        UniqueConstraint("wallet_address", name="uq_known_trader_wallet"),
        Index("idx_known_trader_wallet", "wallet_address"),
        Index("idx_known_trader_profit", "profit_usd"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(42), nullable=False)

    # Profit data
    profit_usd = Column(Numeric(20, 6), nullable=False)

    # Primary funding source
    funding_source = Column(String(42), nullable=True)
    funding_source_type = Column(String(50), nullable=True)  # eoa, exchange, contract
    funding_timestamp = Column(DateTime, nullable=True)
    funding_amount_matic = Column(Numeric(30, 18), nullable=True)
    funding_count = Column(Integer, default=1)

    # Primary withdrawal destination
    primary_withdrawal_dest = Column(String(42), nullable=True)
    total_withdrawn_matic = Column(Numeric(30, 18), nullable=True)
    withdrawal_count = Column(Integer, default=0)

    # Metadata
    labels = Column(JSON, nullable=True)  # ["whale", "bot", etc.]

    # Import tracking
    imported_at = Column(DateTime, default=func.now())
    data_source = Column(String(100), nullable=True)  # "profitable_wallets_full.json"

    def __repr__(self) -> str:
        return f"<KnownProfitableTrader(wallet='{self.wallet_address[:10]}...', profit=${self.profit_usd})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "wallet_address": self.wallet_address,
            "profit_usd": float(self.profit_usd) if self.profit_usd else 0,
            "funding_source": self.funding_source,
            "funding_source_type": self.funding_source_type,
            "primary_withdrawal_dest": self.primary_withdrawal_dest,
            "labels": self.labels,
        }


class KnownFundingSource(Base):
    """A funding source address linked to profitable traders.

    Used for instant sybil detection - if a new wallet is funded from
    an address in this table, it's flagged as a potential sybil.
    """
    __tablename__ = "known_funding_sources"
    __table_args__ = (
        UniqueConstraint("address", name="uq_known_funding_address"),
        Index("idx_known_funding_address", "address"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), nullable=False)

    # Source classification
    source_type = Column(String(50), nullable=True)  # eoa, exchange, contract, bridge
    labels = Column(JSON, nullable=True)  # ["Binance", "Coinbase", etc.]

    # Linked traders
    funded_trader_wallets = Column(JSON, nullable=False)  # Array of wallet addresses
    funded_trader_count = Column(Integer, default=0)
    total_profit_funded = Column(Numeric(20, 6), nullable=True)  # Combined profit of funded traders

    # Risk assessment
    is_exchange = Column(Boolean, default=False)  # Exchange addresses fund many, less suspicious
    risk_score = Column(Integer, default=50)  # 0-100, higher = more suspicious for sybil

    # Import tracking
    imported_at = Column(DateTime, default=func.now())

    def __repr__(self) -> str:
        return f"<KnownFundingSource(addr='{self.address[:10]}...', traders={self.funded_trader_count})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "address": self.address,
            "source_type": self.source_type,
            "labels": self.labels,
            "funded_trader_count": self.funded_trader_count,
            "total_profit_funded": float(self.total_profit_funded) if self.total_profit_funded else 0,
            "is_exchange": self.is_exchange,
            "risk_score": self.risk_score,
        }


class KnownWithdrawalDest(Base):
    """A withdrawal destination linked to profitable traders.

    Critical for chain detection:
    Trader A withdraws to X → X funds new wallet B → B is flagged as sybil

    The `also_funded_traders` field indicates addresses that received withdrawals
    AND then funded new traders - the key link in the chain.
    """
    __tablename__ = "known_withdrawal_destinations"
    __table_args__ = (
        UniqueConstraint("address", name="uq_known_withdrawal_address"),
        Index("idx_known_withdrawal_address", "address"),
        Index("idx_known_withdrawal_also_funded", "also_funded_traders", postgresql_using="gin"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), nullable=False)

    # Destination classification
    dest_type = Column(String(50), nullable=True)  # eoa, exchange, contract
    labels = Column(JSON, nullable=True)

    # Received from traders
    received_from_traders = Column(JSON, nullable=False)  # Array of trader wallet addresses
    received_from_count = Column(Integer, default=0)
    total_profit_source = Column(Numeric(20, 6), nullable=True)

    # CRITICAL: Did this address also fund new traders? (chain link)
    also_funded_traders = Column(JSON, nullable=True)  # Array of trader wallets funded by this address
    is_bridge_wallet = Column(Boolean, default=False)  # True if withdrawal dest that also funds

    # Import tracking
    imported_at = Column(DateTime, default=func.now())

    def __repr__(self) -> str:
        return f"<KnownWithdrawalDest(addr='{self.address[:10]}...', from={self.received_from_count}, bridge={self.is_bridge_wallet})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "address": self.address,
            "dest_type": self.dest_type,
            "labels": self.labels,
            "received_from_count": self.received_from_count,
            "total_profit_source": float(self.total_profit_source) if self.total_profit_source else 0,
            "also_funded_traders": self.also_funded_traders,
            "is_bridge_wallet": self.is_bridge_wallet,
        }


class FundFlowEdge(Base):
    """An edge in the fund flow graph.

    Stores funding and withdrawal transactions between addresses
    for graph traversal and chain detection.
    """
    __tablename__ = "fund_flow_edges"
    __table_args__ = (
        Index("idx_fund_flow_from", "from_address"),
        Index("idx_fund_flow_to", "to_address"),
        Index("idx_fund_flow_type", "edge_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Edge endpoints
    from_address = Column(String(42), nullable=False)
    to_address = Column(String(42), nullable=False)

    # Transaction data
    value_matic = Column(Numeric(30, 18), nullable=True)
    tx_hash = Column(String(66), nullable=True)
    timestamp = Column(DateTime, nullable=True)

    # Edge type
    edge_type = Column(String(20), nullable=False)  # funding, withdrawal

    # Import tracking
    imported_at = Column(DateTime, default=func.now())

    def to_dict(self) -> dict:
        return {
            "from": self.from_address,
            "to": self.to_address,
            "value_matic": float(self.value_matic) if self.value_matic else 0,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "type": self.edge_type,
        }


class SybilDetection(Base):
    """Record of a sybil detection event.

    Created when a new wallet is funded from a known profitable trader's
    funding source or withdrawal destination.
    """
    __tablename__ = "sybil_detections"
    __table_args__ = (
        Index("idx_sybil_new_wallet", "new_wallet_address"),
        Index("idx_sybil_linked_trader", "linked_trader_wallet"),
        Index("idx_sybil_detected_at", "detected_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # The new wallet that was detected
    new_wallet_address = Column(String(42), nullable=False)
    flagged_wallet_id = Column(Integer, ForeignKey("insider_flagged_wallets.id"), nullable=True)

    # The linked profitable trader
    linked_trader_wallet = Column(String(42), nullable=False)
    linked_trader_id = Column(Integer, ForeignKey("known_profitable_traders.id"), nullable=True)
    linked_trader_profit = Column(Numeric(20, 6), nullable=True)

    # How the match was made
    match_type = Column(String(50), nullable=False)  # funding_source, withdrawal_dest, chain_link
    match_address = Column(String(42), nullable=False)  # The address that matched

    # Chain details (if chain_link match)
    chain_path = Column(JSON, nullable=True)  # Array of addresses in the chain

    # Funding details
    funding_amount_matic = Column(Numeric(30, 18), nullable=True)
    funding_tx_hash = Column(String(66), nullable=True)

    # Timing
    detected_at = Column(DateTime, default=func.now())

    def __repr__(self) -> str:
        return f"<SybilDetection(new='{self.new_wallet_address[:10]}...', linked='{self.linked_trader_wallet[:10]}...', type='{self.match_type}')>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "new_wallet_address": self.new_wallet_address,
            "linked_trader_wallet": self.linked_trader_wallet,
            "linked_trader_profit": float(self.linked_trader_profit) if self.linked_trader_profit else 0,
            "match_type": self.match_type,
            "match_address": self.match_address,
            "chain_path": self.chain_path,
            "funding_amount_matic": float(self.funding_amount_matic) if self.funding_amount_matic else 0,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }
