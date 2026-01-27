#!/usr/bin/env python3
"""Re-extract funding data for insider probe results."""
import asyncio
import json
import aiohttp
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_KEY = os.getenv('ETHERSCAN_API_KEY') or os.getenv('POLYGONSCAN_API_KEY')

async def extract_funding(wallet: str, session: aiohttp.ClientSession) -> dict:
    """Extract funding for a single wallet."""
    url = 'https://api.etherscan.io/v2/api'
    params = {
        'chainid': 137,
        'module': 'account',
        'action': 'tokentx',
        'address': wallet,
        'contractaddress': '0x3c499c542cef5e3811e1192ce70d8cc03d5c3359',
        'page': 1,
        'offset': 100,
        'sort': 'asc',
        'apikey': API_KEY,
    }

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            data = await resp.json()

            if data.get('status') != '1' or not data.get('result'):
                return None

            transfers = data['result']
            funding_info = {}

            for tx in transfers:
                if tx.get('to', '').lower() == wallet.lower():
                    amount = float(tx.get('value', 0)) / 1e6
                    if amount >= 50:
                        funding_info['funding_source'] = tx.get('from', '').lower()
                        funding_info['funding_amount'] = amount
                        funding_info['funding_tx'] = tx.get('hash')
                        break

            for tx in reversed(transfers):
                if tx.get('from', '').lower() == wallet.lower():
                    amount = float(tx.get('value', 0)) / 1e6
                    if amount >= 50:
                        funding_info['withdrawal_dest'] = tx.get('to', '').lower()
                        funding_info['withdrawal_amount'] = amount
                        funding_info['withdrawal_tx'] = tx.get('hash')
                        break

            return funding_info if funding_info else None
    except Exception as e:
        return None

async def main():
    if not API_KEY:
        print("ERROR: Set ETHERSCAN_API_KEY environment variable")
        print("Get a free key at: https://etherscan.io/myapikey")
        return

    print(f'Using API key: {API_KEY[:10]}...')

    input_file = PROJECT_ROOT / 'data' / 'insider_probe_results.json'

    with open(input_file) as f:
        data = json.load(f)

    accounts = data['accounts']
    print(f'Re-extracting funding for {len(accounts)} accounts...')

    funding_sources = {}
    extracted = 0

    async with aiohttp.ClientSession() as session:
        for i, acc in enumerate(accounts):
            wallet = acc['wallet_address']
            if i > 0 and i % 5 == 0:
                await asyncio.sleep(1)

            funding = await extract_funding(wallet, session)

            if funding:
                acc['funding_info'] = funding
                extracted += 1
                src = funding.get('funding_source')
                if src:
                    if src not in funding_sources:
                        funding_sources[src] = []
                    funding_sources[src].append(wallet)

            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(accounts)} - Extracted: {extracted}')

    data['stats']['funding_extracted'] = extracted

    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    funding_data = {
        'generated_at': data['generated_at'],
        'sources': {src: {'funded_wallets': wallets, 'count': len(wallets)} for src, wallets in funding_sources.items()}
    }

    funding_file = PROJECT_ROOT / 'data' / 'insider_funding_sources.json'
    with open(funding_file, 'w') as f:
        json.dump(funding_data, f, indent=2)

    print(f'\nDone! Extracted funding for {extracted}/{len(accounts)} accounts')
    print(f'Unique funding sources: {len(funding_sources)}')

if __name__ == '__main__':
    asyncio.run(main())
