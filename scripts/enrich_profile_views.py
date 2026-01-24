#!/usr/bin/env python3
"""
Enrich account data files with profile views from Polymarket.

Usage:
    python3 scripts/enrich_profile_views.py                    # Enrich insider_probe_results.json
    python3 scripts/enrich_profile_views.py --file <path>      # Enrich custom file
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).parent.parent


async def fetch_profile_views(wallet: str, session: aiohttp.ClientSession) -> dict:
    """Fetch profile views for a single wallet."""
    url = f"https://polymarket.com/profile/{wallet.lower()}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html',
    }

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return {"wallet": wallet.lower(), "views": 0, "username": None}

            html = await resp.text()

            views = 0
            username = None

            views_match = re.search(r'"views":(\d+)', html)
            if views_match:
                views = int(views_match.group(1))

            name_match = re.search(r'"username":"([^"]*)"', html)
            if name_match and name_match.group(1):
                username = name_match.group(1)

            return {"wallet": wallet.lower(), "views": views, "username": username}

    except Exception as e:
        return {"wallet": wallet.lower(), "views": 0, "username": None, "error": str(e)}


async def fetch_all_profile_views(
    wallets: list[str],
    max_concurrent: int = 10,
) -> dict[str, dict]:
    """Fetch profile views for all wallets in parallel with rate limiting."""
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = [0]
    total = len(wallets)

    async with aiohttp.ClientSession() as session:
        async def fetch_one(wallet: str):
            async with semaphore:
                stats = await fetch_profile_views(wallet, session)
                results[wallet.lower()] = stats
                completed[0] += 1
                if completed[0] % 25 == 0 or completed[0] == total:
                    print(f"  Progress: {completed[0]}/{total} ({100*completed[0]//total}%)")
                await asyncio.sleep(0.1)  # Rate limiting

        await asyncio.gather(*[fetch_one(w) for w in wallets])

    return results


async def enrich_file(file_path: Path) -> None:
    """Enrich a JSON file with profile views."""
    print(f"\nLoading {file_path}...")

    with open(file_path) as f:
        data = json.load(f)

    # Extract wallets based on file structure
    accounts = data.get("accounts", [])
    if not accounts:
        print("No accounts found in file!")
        return

    wallets = [acc.get("wallet_address") for acc in accounts if acc.get("wallet_address")]
    print(f"Found {len(wallets)} wallets to enrich")

    # Fetch profile views
    print("\nFetching profile views...")
    views_data = await fetch_all_profile_views(wallets)

    # Enrich accounts
    enriched_count = 0
    for acc in accounts:
        wallet = acc.get("wallet_address", "").lower()
        if wallet in views_data:
            acc["profile_views"] = views_data[wallet]["views"]
            acc["profile_username"] = views_data[wallet].get("username")
            enriched_count += 1

    print(f"\nEnriched {enriched_count} accounts with profile views")

    # Save back
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved updated file: {file_path}")

    # Print summary
    views_list = [(acc.get("wallet_address", "")[:12], acc.get("profile_views", 0), acc.get("profile_username", ""))
                  for acc in accounts if acc.get("profile_views", 0) > 0]
    views_list.sort(key=lambda x: x[1], reverse=True)

    if views_list:
        print(f"\nTop 10 by profile views:")
        for i, (wallet, views, username) in enumerate(views_list[:10], 1):
            name_str = f" ({username})" if username else ""
            print(f"  {i}. {wallet}...{name_str}: {views:,} views")


async def main():
    parser = argparse.ArgumentParser(description="Enrich account data with profile views")
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "insider_probe_results.json"),
        help="Path to JSON file to enrich"
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    await enrich_file(file_path)


if __name__ == "__main__":
    asyncio.run(main())
