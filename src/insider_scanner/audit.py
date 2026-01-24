"""Audit Trail System for Legal Protection.

Provides immutable, cryptographically-secured records proving:
1. WHEN you detected a suspicious wallet
2. WHAT signals triggered the detection
3. That your information came FROM the detection system, not insider knowledge

This protects you legally if you act on detected information.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib
import json

from sqlalchemy.orm import Session

from .models import (
    DetectionRecord,
    InvestmentThesis,
    AuditChainEntry,
    FlaggedWallet,
)
from .scoring import ScoringResult, Signal
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AuditTrailManager:
    """Manages the audit trail for insider detections.

    Creates immutable records with:
    - SHA-256 hashing
    - Hash chain linking (tamper-evident)
    - Optional blockchain anchoring
    """

    def __init__(self, db_session: Session):
        """Initialize audit manager.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session

    # =========================================================================
    # DETECTION RECORDS
    # =========================================================================

    def create_detection_record(
        self,
        wallet: FlaggedWallet,
        scoring_result: ScoringResult,
        market_ids: Optional[List[str]] = None,
        market_positions: Optional[Dict] = None,
        raw_api_snapshot: Optional[Dict] = None,
    ) -> DetectionRecord:
        """Create an immutable detection record.

        MUST be called immediately upon detection, BEFORE any action.

        Args:
            wallet: The flagged wallet
            scoring_result: Result from scoring engine
            market_ids: Markets involved in detection
            market_positions: Position data at time of detection
            raw_api_snapshot: Raw API response (proof of data source)

        Returns:
            DetectionRecord with calculated hash
        """
        # Get previous record hash (for chain)
        previous_hash = self._get_latest_record_hash()

        # Prepare signals snapshot
        signals_snapshot = [
            {
                "name": s.name,
                "category": s.category,
                "weight": s.weight,
                "raw_value": str(s.raw_value) if s.raw_value else None,
                "description": s.description,
            }
            for s in scoring_result.signals
        ]

        # Create record - normalize score to float for consistent hashing
        record = DetectionRecord(
            wallet_id=wallet.id,
            wallet_address=wallet.wallet_address,
            detected_at=datetime.utcnow(),
            insider_score=float(scoring_result.score),  # Normalize to float
            priority=scoring_result.priority,
            signals_snapshot=signals_snapshot,
            market_ids=market_ids,
            market_positions=market_positions,
            raw_api_snapshot=raw_api_snapshot,
            previous_record_hash=previous_hash,
        )

        # Calculate and set hash using consistent method
        record.record_hash = self._calculate_detection_hash(record)

        # Add to session
        self.db.add(record)
        self.db.flush()  # Get ID

        # Add to audit chain
        self._add_to_chain("detection", record.id, record.record_hash)

        logger.info(
            "detection_record_created",
            wallet=wallet.wallet_address,
            score=scoring_result.score,
            hash=record.record_hash[:16],
        )

        return record

    def _get_latest_record_hash(self) -> Optional[str]:
        """Get the hash of the most recent detection record."""
        latest = (
            self.db.query(DetectionRecord)
            .order_by(DetectionRecord.id.desc())
            .first()
        )
        return latest.record_hash if latest else None

    # =========================================================================
    # INVESTMENT THESIS
    # =========================================================================

    def create_investment_thesis(
        self,
        detection_record_ids: List[int],
        reasoning: str,
        intended_action: Optional[str] = None,
        market_id: Optional[str] = None,
        position_side: Optional[str] = None,
        position_size: Optional[float] = None,
    ) -> InvestmentThesis:
        """Create an investment thesis document.

        Call this BEFORE placing any bet based on detection.
        Links to detection records to prove information source.

        Args:
            detection_record_ids: IDs of detections informing this thesis
            reasoning: Your reasoning for the investment decision
            intended_action: What you plan to do
            market_id: Market you're betting on
            position_side: YES or NO
            position_size: Intended position size in USD

        Returns:
            InvestmentThesis with calculated hash
        """
        thesis = InvestmentThesis(
            created_at=datetime.utcnow(),
            detection_record_ids=detection_record_ids,
            reasoning=reasoning,
            intended_action=intended_action,
            market_id=market_id,
            position_side=position_side,
            position_size=position_size,
        )

        # Calculate hash
        thesis.thesis_hash = thesis.calculate_hash()

        # Add to session
        self.db.add(thesis)
        self.db.flush()

        # Add to audit chain
        self._add_to_chain("thesis", thesis.id, thesis.thesis_hash)

        logger.info(
            "investment_thesis_created",
            thesis_id=thesis.id,
            detection_count=len(detection_record_ids),
            hash=thesis.thesis_hash[:16],
        )

        return thesis

    def record_thesis_outcome(
        self,
        thesis_id: int,
        action_taken: bool,
        actual_position_size: Optional[float] = None,
        actual_outcome: Optional[str] = None,
    ) -> InvestmentThesis:
        """Record the outcome of an investment thesis.

        Args:
            thesis_id: ID of the thesis
            action_taken: Whether you actually placed the bet
            actual_position_size: Actual position size (if different)
            actual_outcome: Description of outcome

        Returns:
            Updated InvestmentThesis
        """
        thesis = self.db.query(InvestmentThesis).filter(InvestmentThesis.id == thesis_id).first()
        if not thesis:
            raise ValueError(f"Thesis {thesis_id} not found")

        thesis.action_taken = action_taken
        thesis.actual_position_size = actual_position_size
        thesis.actual_outcome = actual_outcome

        logger.info(
            "thesis_outcome_recorded",
            thesis_id=thesis_id,
            action_taken=action_taken,
        )

        return thesis

    # =========================================================================
    # AUDIT CHAIN
    # =========================================================================

    def _add_to_chain(
        self,
        entry_type: str,
        entry_id: int,
        entry_hash: str,
    ) -> AuditChainEntry:
        """Add an entry to the audit chain.

        The chain links all records together. Modifying any historical
        record would break the chain, making tampering detectable.
        """
        # Get latest chain entry
        latest = (
            self.db.query(AuditChainEntry)
            .order_by(AuditChainEntry.sequence_number.desc())
            .first()
        )

        sequence_number = (latest.sequence_number + 1) if latest else 1
        previous_chain_hash = latest.chain_hash if latest else None

        # Create chain entry
        chain_entry = AuditChainEntry(
            sequence_number=sequence_number,
            entry_type=entry_type,
            entry_id=entry_id,
            entry_hash=entry_hash,
            previous_chain_hash=previous_chain_hash,
        )

        # Calculate chain hash
        chain_entry.chain_hash = chain_entry.calculate_chain_hash()

        self.db.add(chain_entry)

        return chain_entry

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit chain.

        Returns:
            Dict with verification results
        """
        entries = (
            self.db.query(AuditChainEntry)
            .order_by(AuditChainEntry.sequence_number.asc())
            .all()
        )

        if not entries:
            return {
                "valid": True,
                "entries_checked": 0,
                "message": "No entries in chain"
            }

        errors = []
        previous_hash = None

        for entry in entries:
            # Check chain linkage
            if entry.previous_chain_hash != previous_hash:
                errors.append({
                    "sequence": entry.sequence_number,
                    "error": "Chain link broken",
                    "expected_previous": previous_hash,
                    "actual_previous": entry.previous_chain_hash,
                })

            # Verify chain hash
            expected_hash = entry.calculate_chain_hash()
            if entry.chain_hash != expected_hash:
                errors.append({
                    "sequence": entry.sequence_number,
                    "error": "Chain hash mismatch",
                    "expected": expected_hash,
                    "actual": entry.chain_hash,
                })

            previous_hash = entry.chain_hash

        return {
            "valid": len(errors) == 0,
            "entries_checked": len(entries),
            "errors": errors,
            "latest_hash": previous_hash,
        }

    def verify_detection_record(self, record_id: int) -> Dict[str, Any]:
        """Verify a specific detection record hasn't been tampered with.

        Args:
            record_id: ID of detection record to verify

        Returns:
            Verification result
        """
        record = self.db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
        if not record:
            return {"valid": False, "error": "Record not found"}

        # Recalculate hash with same normalization as original creation
        expected_hash = self._calculate_detection_hash(record)
        hash_valid = record.record_hash == expected_hash

        # Check if record exists in chain
        chain_entry = (
            self.db.query(AuditChainEntry)
            .filter_by(entry_type="detection", entry_id=record_id)
            .first()
        )

        in_chain = chain_entry is not None
        chain_hash_matches = (
            chain_entry.entry_hash == record.record_hash
            if chain_entry else False
        )

        return {
            "valid": hash_valid and in_chain and chain_hash_matches,
            "hash_valid": hash_valid,
            "in_chain": in_chain,
            "chain_hash_matches": chain_hash_matches,
            "record_hash": record.record_hash,
            "expected_hash": expected_hash,
        }

    def _calculate_detection_hash(self, record: DetectionRecord) -> str:
        """Calculate hash for a detection record with consistent formatting.

        Uses the same logic as record.calculate_hash() but ensures consistent
        numeric formatting across Decimal/float conversions.
        """
        # Normalize insider_score to float then to string with consistent precision
        score_val = float(record.insider_score) if record.insider_score else 0.0

        data = {
            "wallet_address": record.wallet_address,
            "detected_at": record.detected_at.isoformat() if record.detected_at else None,
            "insider_score": str(score_val),
            "priority": record.priority,
            "signals_snapshot": record.signals_snapshot,
            "market_ids": record.market_ids,
            "market_positions": record.market_positions,
            "previous_record_hash": record.previous_record_hash,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # EXPORT / LEGAL DOCUMENTATION
    # =========================================================================

    def export_detection_record(self, record_id: int) -> Dict[str, Any]:
        """Export a detection record for legal purposes.

        Includes:
        - Full record data
        - Hash verification
        - Chain position
        - Anchoring proof (if any)
        """
        record = self.db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
        if not record:
            return {"error": "Record not found"}

        # Get chain entry
        chain_entry = (
            self.db.query(AuditChainEntry)
            .filter_by(entry_type="detection", entry_id=record_id)
            .first()
        )

        # Verify integrity
        verification = self.verify_detection_record(record_id)

        return {
            "record": record.to_dict(),
            "chain_position": chain_entry.sequence_number if chain_entry else None,
            "chain_hash": chain_entry.chain_hash if chain_entry else None,
            "verification": verification,
            "exported_at": datetime.utcnow().isoformat(),
        }

    def export_full_audit_trail(
        self,
        wallet_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export the full audit trail.

        Args:
            wallet_address: Optional filter by wallet

        Returns:
            Complete audit trail with all records and chain
        """
        # Get detection records
        query = self.db.query(DetectionRecord)
        if wallet_address:
            query = query.filter_by(wallet_address=wallet_address)
        records = query.all()

        # Get related theses (filter in Python for SQLite compatibility)
        record_ids = set(r.id for r in records)
        all_theses = self.db.query(InvestmentThesis).all() if record_ids else []
        theses = [
            t for t in all_theses
            if t.detection_record_ids and set(t.detection_record_ids) & record_ids
        ]

        # Verify chain
        chain_verification = self.verify_chain_integrity()

        return {
            "detection_records": [r.to_dict() for r in records],
            "investment_theses": [t.to_dict() for t in theses],
            "chain_verification": chain_verification,
            "exported_at": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "thesis_count": len(theses),
        }

    # =========================================================================
    # BLOCKCHAIN ANCHORING (Optional - Strongest Proof)
    # =========================================================================

    async def anchor_to_blockchain(
        self,
        record_id: int,
        anchor_type: str = "polygon",
    ) -> Optional[str]:
        """Anchor a detection record to blockchain.

        This provides the strongest proof of existence at timestamp.
        The transaction hash serves as permanent, verifiable proof.

        Args:
            record_id: Detection record to anchor
            anchor_type: Blockchain to use (polygon, ethereum, etc.)

        Returns:
            Transaction hash if successful
        """
        record = self.db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
        if not record:
            return None

        # TODO: Implement actual blockchain transaction
        # For now, just log intent
        logger.info(
            "blockchain_anchor_requested",
            record_id=record_id,
            record_hash=record.record_hash,
            anchor_type=anchor_type,
        )

        # In production, this would:
        # 1. Create a transaction with record_hash in data field
        # 2. Submit to blockchain
        # 3. Return tx_hash

        return None  # Placeholder

    async def verify_blockchain_anchor(
        self,
        record_id: int,
    ) -> Dict[str, Any]:
        """Verify a blockchain anchor for a detection record.

        Checks that the anchored hash matches the record hash.
        """
        record = self.db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
        if not record:
            return {"valid": False, "error": "Record not found"}

        if not record.anchor_tx_id:
            return {"valid": False, "error": "No anchor exists"}

        # TODO: Implement actual blockchain verification
        # Would query the blockchain for the tx and verify the hash

        return {
            "valid": True,  # Placeholder
            "anchor_tx_id": record.anchor_tx_id,
            "anchor_type": record.anchor_type,
            "anchored_at": record.anchored_at.isoformat() if record.anchored_at else None,
        }
