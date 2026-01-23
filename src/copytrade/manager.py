"""
Account manager for copy trading.

Handles CRUD operations for tracked accounts and state persistence.
This is core infrastructure shared across ghost mode and live mode.
"""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from .account import CopyTradeAccount, DEFAULT_SLIPPAGE_TIERS


class AccountManager:
    """Manages copy trade accounts with persistence."""

    def __init__(self, state_file: Path, load_from_file: bool = True):
        self.state_file = state_file
        self.accounts: Dict[int, CopyTradeAccount] = {}
        self.seen_trade_hashes: set = set()

        if load_from_file:
            self._load_state()

    def _load_state(self) -> None:
        """Load state from persistence file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                # Load accounts
                for acc_data in data.get('accounts', []):
                    acc = CopyTradeAccount(
                        id=acc_data['id'],
                        name=acc_data['name'],
                        wallet=acc_data.get('wallet', acc_data.get('target_wallet', '')),
                        enabled=acc_data.get('enabled', True),
                        position_ratio=Decimal(str(acc_data.get('position_ratio', '0.01'))),
                        max_position_usd=Decimal(str(acc_data.get('max_position_usd', '500'))),
                        use_tiered_slippage=acc_data.get('use_tiered_slippage', True),
                        flat_slippage_tolerance=Decimal(str(acc_data.get('flat_slippage_tolerance', '0.05'))),
                        keywords=acc_data.get('keywords', []),
                        max_drawdown_percent=Decimal(str(acc_data.get('max_drawdown_percent', '15'))),
                        stoploss_triggered=acc_data.get('stoploss_triggered', False),
                        take_profit_pct=Decimal(str(acc_data.get('take_profit_pct', '0'))),
                        stop_loss_pct=Decimal(str(acc_data.get('stop_loss_pct', '0'))),
                        max_concurrent=int(acc_data.get('max_concurrent', 0)),
                        max_holding_hours=int(acc_data.get('max_holding_hours', 0)),
                        min_liquidity=Decimal(str(acc_data.get('min_liquidity', '0'))),
                        cooldown_seconds=int(acc_data.get('cooldown_seconds', 10)),
                        order_type=acc_data.get('order_type', 'market'),
                    )
                    self.accounts[acc.id] = acc

                # Load seen trade hashes (to prevent re-processing)
                self.seen_trade_hashes = set(data.get('seen_trade_hashes', []))

                print(f"  [State] Loaded {len(self.accounts)} accounts, {len(self.seen_trade_hashes)} seen trades")

        except Exception as e:
            print(f"  [State] Failed to load state: {e}")

    def save_state(self) -> None:
        """Save state to persistence file."""
        try:
            from datetime import datetime

            data = {
                'accounts': [
                    {
                        'id': acc.id,
                        'name': acc.name,
                        'wallet': acc.wallet,
                        'enabled': acc.enabled,
                        'position_ratio': str(acc.position_ratio),
                        'max_position_usd': str(acc.max_position_usd),
                        'use_tiered_slippage': acc.use_tiered_slippage,
                        'flat_slippage_tolerance': str(acc.flat_slippage_tolerance),
                        'keywords': acc.keywords,
                        'max_drawdown_percent': str(acc.max_drawdown_percent),
                        'stoploss_triggered': acc.stoploss_triggered,
                        'take_profit_pct': str(acc.take_profit_pct),
                        'stop_loss_pct': str(acc.stop_loss_pct),
                        'max_concurrent': acc.max_concurrent,
                        'max_holding_hours': acc.max_holding_hours,
                        'min_liquidity': str(acc.min_liquidity),
                        'cooldown_seconds': acc.cooldown_seconds,
                        'order_type': acc.order_type,
                    }
                    for acc in self.accounts.values()
                ],
                'seen_trade_hashes': list(self.seen_trade_hashes)[-1000:],  # Keep last 1000
                'last_shutdown': datetime.utcnow().isoformat(),
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"  [State] Failed to save state: {e}")

    def add_account(
        self,
        name: str,
        wallet: str,
        position_ratio: Decimal = Decimal("0.01"),
        max_position_usd: Decimal = Decimal("500"),
        use_tiered_slippage: bool = True,
        slippage_tiers: List[tuple] = None,
        flat_slippage_tolerance: Decimal = Decimal("0.05"),
        keywords: List[str] = None,
        max_drawdown_percent: Decimal = Decimal("15"),
        take_profit_pct: Decimal = Decimal("0"),
        stop_loss_pct: Decimal = Decimal("0"),
        max_concurrent: int = 0,
        max_holding_hours: int = 0,
        min_liquidity: Decimal = Decimal("0"),
        cooldown_seconds: int = 10,
        order_type: str = "market",
    ) -> CopyTradeAccount:
        """Add an account to track."""
        account_id = max(self.accounts.keys(), default=0) + 1
        account = CopyTradeAccount(
            id=account_id,
            name=name,
            wallet=wallet,
            position_ratio=position_ratio,
            max_position_usd=max_position_usd,
            use_tiered_slippage=use_tiered_slippage,
            slippage_tiers=slippage_tiers if slippage_tiers else DEFAULT_SLIPPAGE_TIERS.copy(),
            flat_slippage_tolerance=flat_slippage_tolerance,
            keywords=keywords or [],
            max_drawdown_percent=max_drawdown_percent,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_concurrent=max_concurrent,
            max_holding_hours=max_holding_hours,
            min_liquidity=min_liquidity,
            cooldown_seconds=cooldown_seconds,
            order_type=order_type,
        )
        self.accounts[account_id] = account
        self.save_state()
        return account

    def update_account(self, account_id: int, **kwargs) -> Optional[CopyTradeAccount]:
        """Update an existing account's settings."""
        if account_id not in self.accounts:
            return None

        account = self.accounts[account_id]

        for key, value in kwargs.items():
            if hasattr(account, key):
                if key in ('position_ratio', 'max_position_usd', 'flat_slippage_tolerance', 'max_drawdown_percent',
                           'take_profit_pct', 'stop_loss_pct', 'min_liquidity'):
                    setattr(account, key, Decimal(str(value)))
                elif key in ('max_concurrent', 'max_holding_hours', 'cooldown_seconds'):
                    setattr(account, key, int(value))
                elif key == 'keywords':
                    if isinstance(value, str):
                        setattr(account, key, [k.strip() for k in value.split(',') if k.strip()])
                    else:
                        setattr(account, key, value)
                elif key == 'use_tiered_slippage':
                    setattr(account, key, bool(value))
                elif key == 'slippage_tiers':
                    if value is not None:
                        setattr(account, key, value)
                else:
                    setattr(account, key, value)

        self.save_state()
        return account

    def delete_account(self, account_id: int) -> bool:
        """Delete an account."""
        if account_id in self.accounts:
            del self.accounts[account_id]
            self.save_state()
            return True
        return False

    def get_account(self, account_id: int) -> Optional[CopyTradeAccount]:
        """Get account by ID."""
        return self.accounts.get(account_id)

    def get_all_accounts(self) -> List[CopyTradeAccount]:
        """Get all accounts."""
        return list(self.accounts.values())

    def get_enabled_accounts(self) -> List[CopyTradeAccount]:
        """Get only enabled accounts."""
        return [a for a in self.accounts.values() if a.enabled]

    def add_seen_trade_hash(self, trade_hash: str) -> None:
        """Mark a trade hash as seen."""
        self.seen_trade_hashes.add(trade_hash)

    def is_trade_seen(self, trade_hash: str) -> bool:
        """Check if a trade hash has been seen."""
        return trade_hash in self.seen_trade_hashes
