#!/usr/bin/env python3
"""
Enrich account data files with category information from trade history.

Usage:
    python3 scripts/enrich_categories.py                    # Enrich insider_probe_results.json
    python3 scripts/enrich_categories.py --file <path>      # Enrich custom file
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType


# Category keywords mapping
CATEGORY_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "coin", "token", "defi", "nft", "solana", "sol", "doge", "xrp", "cardano", "ada", "polygon", "matic", "avalanche", "avax", "chainlink", "link", "uniswap", "aave", "maker", "compound", "sushi", "curve", "yearn", "synthetix", "ren", "balancer", "1inch", "pancake", "binance", "coinbase", "kraken", "ftx", "gemini", "bitfinex", "huobi", "okex", "kucoin", "bybit", "stablecoin", "usdt", "usdc", "dai", "frax", "luna", "ust"],
    "politics": ["trump", "biden", "election", "president", "senate", "congress", "democrat", "republican", "gop", "vote", "poll", "governor", "mayor", "primary", "nominee", "cabinet", "secretary", "political", "government", "legislation", "bill", "law", "veto", "impeach", "pardon", "executive", "judicial", "supreme court", "midterm", "electoral", "swing state", "ballot", "recount", "certification", "inauguration", "state of the union", "administration", "white house", "capitol", "campaign", "endorsement", "debate", "running mate", "vice president", "speaker", "majority leader", "minority leader", "caucus", "filibuster", "cloture", "reconciliation", "omnibus", "continuing resolution", "shutdown", "debt ceiling", "maduro", "venezuela", "iran", "khamenei", "regime", "coup", "sanctions", "tariff", "geopolitical"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "tennis", "golf", "ufc", "mma", "boxing", "f1", "formula", "racing", "olympics", "world cup", "super bowl", "playoffs", "finals", "championship", "mvp", "draft", "trade", "free agent", "injury", "roster", "lineup", "score", "win", "lose", "draw", "overtime", "penalty", "foul", "referee", "coach", "manager", "player", "team", "league", "division", "conference", "standings", "record", "streak", "rivalry", "derby", "match", "game", "series", "round", "seed", "bracket", "tournament", "cup", "trophy", "medal", "podium", "qualifying", "preliminary", "semifinal", "quarterfinal", "group stage"],
    "finance": ["fed", "federal reserve", "interest rate", "inflation", "gdp", "unemployment", "stock", "market", "dow", "nasdaq", "s&p", "bond", "treasury", "yield", "recession", "growth", "earnings", "revenue", "profit", "loss", "dividend", "buyback", "ipo", "merger", "acquisition", "bankruptcy", "default", "credit", "debt", "loan", "mortgage", "housing", "real estate", "commodity", "oil", "gold", "silver", "copper", "wheat", "corn", "soybean", "natural gas", "energy", "utility", "bank", "insurance", "hedge fund", "private equity", "venture capital", "fintech", "payment", "remittance", "forex", "currency", "dollar", "euro", "yen", "pound", "yuan", "emerging market", "developed market", "frontier market", "index", "etf", "mutual fund", "401k", "ira", "pension", "annuity", "derivative", "option", "future", "swap", "forward", "cds", "cdo", "mbs", "abs", "repo", "libor", "sofr"],
    "tech": ["apple", "google", "microsoft", "amazon", "meta", "facebook", "twitter", "x.com", "tiktok", "snapchat", "instagram", "youtube", "netflix", "spotify", "uber", "lyft", "airbnb", "doordash", "instacart", "shopify", "stripe", "square", "paypal", "venmo", "robinhood", "coinbase", "openai", "chatgpt", "gpt", "ai", "artificial intelligence", "machine learning", "deep learning", "neural network", "llm", "generative", "autonomous", "robot", "drone", "ev", "electric vehicle", "tesla", "rivian", "lucid", "nio", "byd", "spacex", "starlink", "blue origin", "virgin galactic", "rocket", "satellite", "5g", "6g", "iot", "cloud", "aws", "azure", "gcp", "saas", "paas", "iaas", "cybersecurity", "hack", "breach", "ransomware", "malware", "phishing", "encryption", "privacy", "data", "algorithm", "software", "hardware", "chip", "semiconductor", "nvidia", "amd", "intel", "qualcomm", "broadcom", "tsmc", "asml", "arm"],
    "weather": ["hurricane", "tornado", "earthquake", "flood", "drought", "wildfire", "blizzard", "storm", "typhoon", "cyclone", "monsoon", "el nino", "la nina", "climate", "temperature", "precipitation", "snow", "rain", "wind", "heat wave", "cold snap", "frost", "freeze", "thaw", "ice", "hail", "lightning", "thunder", "fog", "humidity", "pressure", "front", "jet stream", "polar vortex", "atmospheric", "meteorological", "forecast", "prediction", "warning", "watch", "advisory", "emergency", "evacuation", "shelter", "damage", "destruction", "casualty", "fatality", "injury", "rescue", "relief", "recovery", "rebuild", "insurance", "claim", "loss", "estimate", "impact", "aftermath", "season", "record", "historic", "unprecedented", "extreme", "severe", "moderate", "minor", "category"],
    "culture": ["oscar", "emmy", "grammy", "tony", "golden globe", "cannes", "sundance", "venice", "berlin", "toronto", "sxsw", "coachella", "lollapalooza", "bonnaroo", "glastonbury", "burning man", "comic con", "e3", "ces", "mwc", "gdc", "pax", "twitchcon", "vidcon", "playlist live", "beautycon", "fashion week", "met gala", "vmas", "amas", "billboard", "brit", "juno", "aria", "echo", "nrj", "mtv", "bet", "country music", "acm", "cma", "iheartradio", "spotify wrapped", "apple music", "tidal", "deezer", "soundcloud", "bandcamp", "vinyl", "cassette", "cd", "mp3", "streaming", "download", "album", "single", "ep", "mixtape", "playlist", "chart", "billboard hot 100", "uk singles", "global 200", "viral 50", "movie", "film", "tv", "television", "show", "series", "season", "episode", "premiere", "finale", "cliffhanger", "spoiler", "review", "rating", "box office", "opening weekend", "domestic", "international", "worldwide", "gross", "budget", "production", "director", "actor", "actress", "cast", "crew", "script", "screenplay", "adaptation", "sequel", "prequel", "reboot", "remake", "spinoff", "franchise", "universe", "cinematic", "streaming service", "disney+", "hbo max", "paramount+", "peacock", "hulu", "amazon prime", "apple tv+", "discovery+", "espn+", "youtube premium", "twitch", "kick", "rumble", "odysee", "bitchute", "dailymotion", "vimeo", "tiktok", "instagram reels", "youtube shorts", "snapchat spotlight", "elon", "musk", "tweet", "x post"],
}


def categorize_market(market_title: str) -> str:
    """Categorize a market based on its title."""
    if not market_title:
        return "other"

    title_lower = market_title.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category

    return "other"


async def get_account_category(wallet: str, client: DataAPIClient, sample_size: int = 50) -> dict:
    """Get category breakdown for an account by sampling their trades."""
    try:
        # Try TRADE activities first
        activities = await client.get_activity(
            user=wallet.lower(),
            activity_type=ActivityType.TRADE,
            limit=sample_size,
        )

        # If no trades, try all activities (including REDEEM)
        if not activities:
            activities = await client.get_activity(
                user=wallet.lower(),
                limit=sample_size,
            )

        if not activities:
            return {
                "primary_category": "Unknown",
                "category_breakdown": {},
                "category_concentration": 0,
                "trades_sampled": 0,
            }

        # Count categories
        category_counts = defaultdict(int)
        market_cache = {}

        for activity in activities:
            condition_id = activity.condition_id

            # Check cache first
            if condition_id in market_cache:
                category = market_cache[condition_id]
            else:
                # Use market_title directly from activity object (no API call needed)
                market_title = getattr(activity, 'market_title', None) or f"Market {condition_id[:8]}"
                category = categorize_market(market_title)
                market_cache[condition_id] = category

            category_counts[category] += 1

        # Calculate percentages
        total = sum(category_counts.values())
        breakdown = {
            cat: round(count / total * 100, 1)
            for cat, count in category_counts.items()
        }

        # Find primary category
        primary = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "other"
        concentration = breakdown.get(primary, 0)

        # If no single category dominates (>60%), mark as Diversified
        if concentration < 60:
            primary = "Diversified"

        return {
            "primary_category": primary.title() if primary not in ["Diversified", "Unknown"] else primary,
            "category_breakdown": breakdown,
            "category_concentration": concentration,
            "trades_sampled": len(activities),
        }

    except Exception as e:
        return {
            "primary_category": "Unknown",
            "category_breakdown": {},
            "category_concentration": 0,
            "trades_sampled": 0,
            "error": str(e),
        }


async def enrich_file(file_path: Path, sample_size: int = 50) -> None:
    """Enrich a JSON file with category data."""
    print(f"\nLoading {file_path}...")

    with open(file_path) as f:
        data = json.load(f)

    accounts = data.get("accounts", [])
    if not accounts:
        print("No accounts found in file!")
        return

    # Filter accounts that need categorization
    placeholder_categories = ["Other", "other", "Unknown", "Diversified", "", None]
    needs_categorization = [
        acc for acc in accounts
        if acc.get("primary_category") in placeholder_categories
        or not acc.get("category_breakdown")
    ]

    print(f"Found {len(needs_categorization)} accounts needing categorization (out of {len(accounts)} total)")

    if not needs_categorization:
        print("All accounts already have categories!")
        return

    # Create client
    async with DataAPIClient() as client:
        categorized_count = 0
        total = len(needs_categorization)

        for i, acc in enumerate(needs_categorization):
            wallet = acc.get("wallet_address")
            if not wallet:
                continue

            result = await get_account_category(wallet, client, sample_size)

            if result["primary_category"] not in ["Unknown"]:
                acc["primary_category"] = result["primary_category"]
                acc["category_breakdown"] = result["category_breakdown"]
                acc["category_concentration"] = result["category_concentration"]
                categorized_count += 1

            if (i + 1) % 10 == 0 or i + 1 == total:
                print(f"  Progress: {i + 1}/{total} ({100*(i+1)//total}%) - {categorized_count} categorized")

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

    print(f"\nCategorized {categorized_count} accounts")

    # Save back
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved updated file: {file_path}")

    # Print summary
    category_dist = defaultdict(int)
    for acc in accounts:
        cat = acc.get("primary_category", "Unknown")
        category_dist[cat] += 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


async def main():
    parser = argparse.ArgumentParser(description="Enrich account data with category information")
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "insider_probe_results.json"),
        help="Path to JSON file to enrich"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of trades to sample per account"
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    await enrich_file(file_path, args.sample_size)


if __name__ == "__main__":
    asyncio.run(main())
