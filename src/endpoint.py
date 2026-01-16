#!/usr/bin/env python3
"""
🔍 COINGECKO ENDPOINT DATA COMPARISON TEST 🔍

This script tests all available CoinGecko endpoints to see exactly what data
we can get for advanced trading analysis and rate limit implications.
"""

import json
import requests
import time
from typing import Dict, Any, List, Optional

def test_all_coingecko_endpoints():
    """
    Test all relevant CoinGecko endpoints to compare data availability
    """
    print("🚀 COINGECKO ENDPOINT DATA COMPARISON")
    print("=" * 80)
    
    # Test token - UNI since it was failing
    test_token = 'uniswap'
    base_url = "https://api.coingecko.com/api/v3"
    
    endpoints_to_test = [
        {
            'name': '1. Markets Endpoint (Batch)',
            'endpoint': 'coins/markets',
            'params': {
                'vs_currency': 'usd',
                'ids': 'uniswap,bitcoin,ethereum',
                'sparkline': True,
                'price_change_percentage': '1h,24h,7d'
            },
            'description': 'Batch endpoint - multiple tokens in one call'
        },
        {
            'name': '2. Individual Coin Endpoint',
            'endpoint': f'coins/{test_token}',
            'params': {
                'localization': 'false',
                'tickers': 'false', 
                'market_data': 'true',
                'sparkline': 'true'
            },
            'description': 'Individual coin with full market data'
        },
        {
            'name': '3. Market Chart Endpoint (1 day hourly)',
            'endpoint': f'coins/{test_token}/market_chart',
            'params': {
                'vs_currency': 'usd',
                'days': 1,
                'interval': 'hourly'
            },
            'description': 'Historical price data - 1 day hourly'
        },
        {
            'name': '4. Market Chart Endpoint (7 days)',
            'endpoint': f'coins/{test_token}/market_chart',
            'params': {
                'vs_currency': 'usd',
                'days': 7
            },
            'description': 'Historical price data - 7 days'
        },
        {
            'name': '5. Market Chart Endpoint (30 days)',
            'endpoint': f'coins/{test_token}/market_chart',
            'params': {
                'vs_currency': 'usd',
                'days': 30
            },
            'description': 'Historical price data - 30 days'
        },
        {
            'name': '6. OHLC Endpoint (1 day)',
            'endpoint': f'coins/{test_token}/ohlc',
            'params': {
                'vs_currency': 'usd',
                'days': 1
            },
            'description': 'OHLC candlestick data - 1 day'
        },
        {
            'name': '7. OHLC Endpoint (7 days)',
            'endpoint': f'coins/{test_token}/ohlc',
            'params': {
                'vs_currency': 'usd',
                'days': 7
            },
            'description': 'OHLC candlestick data - 7 days'
        }
    ]
    
    results = {}
    
    for endpoint_config in endpoints_to_test:
        print(f"\n{endpoint_config['name']}")
        print(f"📝 {endpoint_config['description']}")
        print("-" * 60)
        
        try:
            url = f"{base_url}/{endpoint_config['endpoint']}"
            response = requests.get(url, params=endpoint_config['params'], timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results[endpoint_config['name']] = data
                
                # Analyze the response
                analyze_endpoint_response(endpoint_config['name'], data, endpoint_config['endpoint'])
                
            elif response.status_code == 429:
                print(f"   ⏱️  Rate limited - too many requests")
                results[endpoint_config['name']] = 'RATE_LIMITED'
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   📝 Response: {response.text[:200]}...")
                results[endpoint_config['name']] = f'ERROR_{response.status_code}'
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
            results[endpoint_config['name']] = f'EXCEPTION_{str(e)[:50]}'
        
        # Rate limiting between requests
        time.sleep(2)
    
    # Summary comparison
    print_comparison_summary(results)
    
    return results

def analyze_endpoint_response(endpoint_name: str, data: Any, endpoint_path: str):
    """Analyze and display key information from endpoint response"""
    
    if 'markets' in endpoint_path:
        analyze_markets_response(data)
        
    elif '/market_chart' in endpoint_path:
        analyze_market_chart_response(data)
        
    elif '/ohlc' in endpoint_path:
        analyze_ohlc_response(data)
        
    elif endpoint_path.startswith('coins/') and '/market_chart' not in endpoint_path and '/ohlc' not in endpoint_path:
        analyze_individual_coin_response(data)

def analyze_markets_response(data: List[Dict]):
    """Analyze markets endpoint response"""
    if not data or not isinstance(data, list):
        print(f"   ❌ Invalid markets data format")
        return
    
    coin = data[0] if data else {}
    print(f"   ✅ Markets data received for {len(data)} coins")
    print(f"   💰 Current price: ${coin.get('current_price', 'N/A')}")
    print(f"   📊 Volume 24h: ${coin.get('total_volume', 'N/A'):,}")
    print(f"   📈 Price change 24h: {coin.get('price_change_percentage_24h', 'N/A')}%")
    
    # Check sparkline
    sparkline = coin.get('sparkline_in_7d')
    if sparkline and isinstance(sparkline, dict) and 'price' in sparkline:
        prices = sparkline['price']
        print(f"   📊 Sparkline prices: {len(prices)} data points")
        if prices:
            print(f"   📊 Price range: ${min(prices):.4f} - ${max(prices):.4f}")
    else:
        print(f"   ❌ No sparkline data in markets response")
    
    # Show all available fields
    available_fields = list(coin.keys())
    print(f"   🔍 Available fields ({len(available_fields)}): {', '.join(available_fields[:10])}...")

def analyze_individual_coin_response(data: Dict):
    """Analyze individual coin endpoint response"""
    if not isinstance(data, dict):
        print(f"   ❌ Invalid individual coin data format")
        return
    
    print(f"   ✅ Individual coin data received")
    print(f"   🆔 Coin ID: {data.get('id', 'N/A')}")
    print(f"   🔤 Symbol: {data.get('symbol', 'N/A')}")
    
    market_data = data.get('market_data', {})
    if market_data:
        current_price = market_data.get('current_price', {})
        if isinstance(current_price, dict):
            usd_price = current_price.get('usd', 'N/A')
            print(f"   💰 Current USD price: ${usd_price}")
        
        # Check for sparkline in market_data
        sparkline_7d = market_data.get('sparkline_7d')
        if sparkline_7d and isinstance(sparkline_7d, dict) and 'price' in sparkline_7d:
            prices = sparkline_7d['price']
            print(f"   📊 Sparkline 7d: {len(prices)} data points")
            if prices:
                print(f"   📊 Price range: ${min(prices):.4f} - ${max(prices):.4f}")
        else:
            print(f"   ❌ No sparkline_7d in market_data")
    
    # Show top-level structure
    top_fields = list(data.keys())
    print(f"   🔍 Top-level fields ({len(top_fields)}): {', '.join(top_fields[:8])}...")

def analyze_market_chart_response(data: Dict):
    """Analyze market chart endpoint response"""
    if not isinstance(data, dict):
        print(f"   ❌ Invalid market chart data format")
        return
    
    prices = data.get('prices', [])
    volumes = data.get('total_volumes', [])
    market_caps = data.get('market_caps', [])
    
    print(f"   ✅ Market chart data received")
    print(f"   📊 Price data points: {len(prices)}")
    print(f"   📊 Volume data points: {len(volumes)}")
    print(f"   📊 Market cap data points: {len(market_caps)}")
    
    if prices and len(prices) > 0:
        # Each entry is [timestamp, value]
        price_values = [entry[1] for entry in prices if len(entry) >= 2]
        if price_values:
            print(f"   💰 Price range: ${min(price_values):.4f} - ${max(price_values):.4f}")
            print(f"   📅 Time span: {format_timestamp_range(prices)}")
        
        # Show data frequency
        if len(prices) >= 2:
            time_diff = prices[1][0] - prices[0][0]
            interval_hours = time_diff / (1000 * 60 * 60)  # Convert ms to hours
            print(f"   ⏰ Data interval: ~{interval_hours:.1f} hours")

def analyze_ohlc_response(data: List):
    """Analyze OHLC endpoint response"""
    if not isinstance(data, list):
        print(f"   ❌ Invalid OHLC data format")
        return
    
    print(f"   ✅ OHLC data received")
    print(f"   📊 OHLC candles: {len(data)}")
    
    if data and len(data) > 0:
        # Each entry is [timestamp, open, high, low, close]
        sample = data[0]
        if len(sample) >= 5:
            print(f"   📊 Sample candle: O:{sample[1]:.4f} H:{sample[2]:.4f} L:{sample[3]:.4f} C:{sample[4]:.4f}")
            
            # Calculate price range across all candles
            all_prices = []
            for candle in data:
                if len(candle) >= 5:
                    all_prices.extend([candle[1], candle[2], candle[3], candle[4]])  # O,H,L,C
            
            if all_prices:
                print(f"   💰 Price range: ${min(all_prices):.4f} - ${max(all_prices):.4f}")
        
        # Show time span
        if len(data) >= 2:
            time_span = data[-1][0] - data[0][0]
            hours = time_span / (1000 * 60 * 60)
            print(f"   📅 Time span: {hours:.1f} hours")

def format_timestamp_range(price_data: List) -> str:
    """Format timestamp range for display"""
    if len(price_data) < 2:
        return "N/A"
    
    start_ts = price_data[0][0] / 1000  # Convert ms to seconds
    end_ts = price_data[-1][0] / 1000
    
    from datetime import datetime
    start_time = datetime.fromtimestamp(start_ts)
    end_time = datetime.fromtimestamp(end_ts)
    
    return f"{start_time.strftime('%m/%d %H:%M')} - {end_time.strftime('%m/%d %H:%M')}"

def print_comparison_summary(results: Dict):
    """Print a summary comparison of all endpoints"""
    print(f"\n🎯 ENDPOINT COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 DATA AVAILABILITY FOR ADVANCED TRADING:")
    print(f"┌─────────────────────────────────────────────────────────┐")
    print(f"│ Endpoint                   │ Price History │ API Calls │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│ Markets (batch)            │      ❌       │     1     │")
    print(f"│ Individual coin            │      ✅       │   1 per   │")
    print(f"│ Market chart (1 day)       │      ✅       │   1 per   │")
    print(f"│ Market chart (7 days)      │      ✅       │   1 per   │")
    print(f"│ Market chart (30 days)     │      ✅       │   1 per   │")
    print(f"│ OHLC (1 day)              │      ✅       │   1 per   │")
    print(f"│ OHLC (7 days)             │      ✅       │   1 per   │")
    print(f"└─────────────────────────────────────────────────────────┘")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use Markets endpoint for basic data (price, volume)")
    print(f"   2. Use Market Chart for price history (most efficient)")
    print(f"   3. Cache historical data to minimize API calls")
    print(f"   4. Consider selective fetching for priority tokens only")

def main():
    """Main test function"""
    print("🚀 Starting CoinGecko Endpoint Data Comparison")
    print("This will test all endpoints to find the best approach for advanced trading")
    print()
    
    results = test_all_coingecko_endpoints()
    
    print(f"\n🎯 TEST COMPLETE")
    print("Review the data above to determine the best approach for your advanced trading system")

if __name__ == "__main__":
    main()