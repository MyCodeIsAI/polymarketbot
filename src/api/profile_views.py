"""
Polymarket Profile Views Fetcher.

Profile view counts are not exposed via a public API, but are embedded
in the page's __NEXT_DATA__ script tag. This module extracts that data.
"""

import asyncio
import aiohttp
import re
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ProfileStats:
    """Profile statistics from Polymarket."""
    wallet: str
    views: int
    trades: int
    largest_win: float
    join_date: Optional[str]
    username: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "wallet": self.wallet,
            "views": self.views,
            "trades": self.trades,
            "largest_win": self.largest_win,
            "join_date": self.join_date,
            "username": self.username,
        }


async def get_profile_stats(wallet: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[ProfileStats]:
    """
    Fetch profile statistics including view count from Polymarket.
    
    Args:
        wallet: The wallet address to lookup
        session: Optional aiohttp session for connection reuse
        
    Returns:
        ProfileStats object or None if fetch failed
    """
    url = f"https://polymarket.com/profile/{wallet.lower()}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html',
    }
    
    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
                
            html = await resp.text()
            
            # Extract values using regex (faster than full JSON parse)
            views = 0
            trades = 0
            largest_win = 0
            join_date = None
            username = None
            
            views_match = re.search(r'"views":(\d+)', html)
            if views_match:
                views = int(views_match.group(1))
                
            trades_match = re.search(r'"trades":(\d+)', html)
            if trades_match:
                trades = int(trades_match.group(1))
                
            win_match = re.search(r'"largestWin":(\d+)', html)
            if win_match:
                largest_win = float(win_match.group(1))
                
            date_match = re.search(r'"joinDate":"([^"]*)"', html)
            if date_match and date_match.group(1):
                join_date = date_match.group(1)
                
            name_match = re.search(r'"username":"([^"]*)"', html)
            if name_match and name_match.group(1):
                username = name_match.group(1)
            
            return ProfileStats(
                wallet=wallet.lower(),
                views=views,
                trades=trades,
                largest_win=largest_win,
                join_date=join_date,
                username=username,
            )
            
    except Exception as e:
        print(f"Error fetching profile {wallet[:10]}: {e}")
        return None
        
    finally:
        if close_session:
            await session.close()


async def get_batch_profile_stats(wallets: list[str], max_concurrent: int = 5) -> Dict[str, ProfileStats]:
    """
    Fetch profile stats for multiple wallets.
    
    Args:
        wallets: List of wallet addresses
        max_concurrent: Max concurrent requests (be nice to their servers)
        
    Returns:
        Dict mapping wallet -> ProfileStats
    """
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        async def fetch_one(wallet: str):
            async with semaphore:
                stats = await get_profile_stats(wallet, session)
                if stats:
                    results[wallet.lower()] = stats
                # Small delay between requests
                await asyncio.sleep(0.2)
        
        await asyncio.gather(*[fetch_one(w) for w in wallets])
    
    return results


# Quick test
if __name__ == "__main__":
    async def main():
        # Test single profile
        wallet = "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee"
        stats = await get_profile_stats(wallet)
        if stats:
            print(f"Profile: {stats.username or stats.wallet[:16]}")
            print(f"  Views: {stats.views:,}")
            print(f"  Trades: {stats.trades:,}")
            print(f"  Largest Win: ${stats.largest_win:,.0f}")
            print(f"  Join Date: {stats.join_date}")
        
        # Test batch
        print("\nBatch test:")
        wallets = [
            "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",
            "0xd49ef2c1c59af45d763d81eae69be51c7b8a7b24",
        ]
        batch_stats = await get_batch_profile_stats(wallets)
        for wallet, stats in batch_stats.items():
            print(f"  {stats.username or wallet[:16]}: {stats.views:,} views")
    
    asyncio.run(main())
