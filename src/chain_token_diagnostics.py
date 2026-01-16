#!/usr/bin/env python3
"""
chain_token_diagnostic.py - Test Chain/Token Mapping Issues

This script specifically tests whether the chain/token mapping is working
correctly in your actual data flow.
"""

import sys
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class ChainTokenDiagnostic:
    """
    Test whether chain/token mapping is working correctly
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            possible_paths = [
                os.path.join(project_root, "crypto_bot.db"),
                os.path.join(project_root, "data", "crypto_history.db"),
                "crypto_bot.db"
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None:
                print("❌ No database file found!")
                print("Tried paths:")
                for path in possible_paths:
                    print(f"   - {path}")
                sys.exit(1)
            
            self.db_path = found_path
        else:
            self.db_path = db_path
    
    def run_chain_token_tests(self):
        """Run comprehensive chain/token mapping tests"""
        print("=" * 80)
        print("🔍 CHAIN/TOKEN MAPPING DIAGNOSTIC")
        print("=" * 80)
        print(f"📂 Database: {self.db_path}")
        print()
        
        # Test 1: Check what tokens/chains exist in each table
        self.test_token_chain_existence()
        
        # Test 2: Test the JOIN that's supposed to work
        self.test_join_operation()
        
        # Test 3: Test market data retrieval as your bot would do it
        self.test_market_data_retrieval()
        
        # Test 4: Test reference tokens lookup
        self.test_reference_tokens_lookup()
        
        # Test 5: Test the _get_crypto_data simulation
        self.test_crypto_data_simulation()
    
    def test_token_chain_existence(self):
        """Test what tokens/chains exist in each table"""
        print("🔍 Testing Token/Chain Existence Across Tables")
        print("-" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get distinct tokens from price_history
            cursor.execute("""
                SELECT DISTINCT token, COUNT(*) as count 
                FROM price_history 
                GROUP BY token 
                ORDER BY count DESC
            """)
            price_history_tokens = dict(cursor.fetchall())
            
            # Get distinct chains from market_data
            cursor.execute("""
                SELECT DISTINCT chain, COUNT(*) as count 
                FROM market_data 
                GROUP BY chain 
                ORDER BY count DESC
            """)
            market_data_chains = dict(cursor.fetchall())
            
            print(f"📊 price_history tokens: {len(price_history_tokens)}")
            for token, count in list(price_history_tokens.items())[:10]:
                print(f"   {token}: {count:,} rows")
            
            print(f"\n📊 market_data chains: {len(market_data_chains)}")
            for chain, count in list(market_data_chains.items())[:10]:
                print(f"   {chain}: {count:,} rows")
            
            # Check overlap
            price_tokens = set(price_history_tokens.keys())
            market_chains = set(market_data_chains.keys())
            
            overlap = price_tokens.intersection(market_chains)
            price_only = price_tokens - market_chains
            market_only = market_chains - price_tokens
            
            print(f"\n🎯 Token/Chain Analysis:")
            print(f"   Overlap (same names): {len(overlap)}")
            if overlap:
                print(f"      {list(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")
            
            print(f"   price_history only: {len(price_only)}")
            if price_only:
                print(f"      {list(price_only)[:5]}{'...' if len(price_only) > 5 else ''}")
            
            print(f"   market_data only: {len(market_only)}")
            if market_only:
                print(f"      {list(market_only)[:5]}{'...' if len(market_only) > 5 else ''}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error in token/chain existence test: {e}")
        
        print("\n" + "=" * 60 + "\n")
    
    def test_join_operation(self):
        """Test the JOIN operation that should map chain to token"""
        print("🔗 Testing JOIN Operation (chain → token)")
        print("-" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Test the actual JOIN from the codebase
            cursor.execute("""
                SELECT 
                    md.chain,
                    ph.token,
                    md.price as md_price,
                    ph.price as ph_price,
                    md.timestamp as md_time,
                    ph.timestamp as ph_time
                FROM market_data md
                LEFT JOIN price_history ph ON md.chain = ph.token 
                    AND datetime(md.timestamp) = datetime(ph.timestamp)
                WHERE md.timestamp >= datetime('now', '-24 hours')
                LIMIT 10
            """)
            
            results = cursor.fetchall()
            
            print(f"📊 JOIN Results (last 24h): {len(results)} rows")
            
            successful_joins = 0
            failed_joins = 0
            
            for row in results:
                chain = row['chain']
                token = row['token']
                md_price = row['md_price']
                ph_price = row['ph_price']
                
                if token is not None:
                    successful_joins += 1
                    price_match = abs(md_price - ph_price) < 0.01 if (md_price and ph_price) else "N/A"
                    print(f"   ✅ {chain} → {token} (prices match: {price_match})")
                else:
                    failed_joins += 1
                    print(f"   ❌ {chain} → NULL (no matching token)")
            
            print(f"\n🎯 JOIN Success Rate:")
            total_joins = successful_joins + failed_joins
            if total_joins > 0:
                success_rate = (successful_joins / total_joins) * 100
                print(f"   Successful: {successful_joins}/{total_joins} ({success_rate:.1f}%)")
                print(f"   Failed: {failed_joins}/{total_joins} ({100-success_rate:.1f}%)")
            else:
                print("   No recent data to test JOIN")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error in JOIN test: {e}")
        
        print("\n" + "=" * 60 + "\n")
    
    def test_market_data_retrieval(self):
        """Test market data retrieval as the bot would do it"""
        print("📡 Testing Market Data Retrieval (Bot Simulation)")
        print("-" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Test getting recent market data for specific chains (as bot does)
            test_chains = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
            
            results = {}
            
            for chain in test_chains:
                # Simulate the get_recent_market_data method
                cursor.execute("""
                    SELECT 
                        md.*,
                        ph.volume as ph_volume,
                        ph.market_cap as ph_market_cap
                    FROM market_data md
                    LEFT JOIN price_history ph ON md.chain = ph.token 
                        AND datetime(md.timestamp) = datetime(ph.timestamp)
                    WHERE md.chain = ? 
                    AND md.timestamp >= datetime('now', '-24 hours')
                    ORDER BY md.timestamp DESC
                    LIMIT 5
                """, (chain,))
                
                chain_results = cursor.fetchall()
                results[chain] = len(chain_results)
                
                if chain_results:
                    latest = chain_results[0]
                    enhanced_volume = latest['ph_volume'] if latest['ph_volume'] else latest['volume']
                    print(f"   ✅ {chain}: {len(chain_results)} recent records")
                    print(f"      Latest price: ${latest['price']:,.2f}")
                    print(f"      Enhanced volume: {enhanced_volume:,.0f}" if enhanced_volume else "      No volume data")
                else:
                    print(f"   ❌ {chain}: No recent data")
            
            print(f"\n🎯 Market Data Retrieval Summary:")
            working_chains = len([c for c, count in results.items() if count > 0])
            print(f"   Chains with data: {working_chains}/{len(test_chains)}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error in market data retrieval test: {e}")
        
        print("\n" + "=" * 60 + "\n")
    
    def test_reference_tokens_lookup(self):
        """Test lookup of reference tokens for volatility calculation"""
        print("🎯 Testing Reference Tokens Lookup")
        print("-" * 60)
        
        # Common reference tokens used in volatility calculation
        reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'AVAX', 'DOT', 'UNI', 'NEAR', 'AAVE']
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            print("📊 Reference Token Data Availability:")
            
            for token in reference_tokens:
                # Check market_data (using chain column)
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(timestamp) as latest
                    FROM market_data 
                    WHERE chain = ? AND timestamp >= datetime('now', '-24 hours')
                """, (token,))
                
                market_result = cursor.fetchone()
                market_count = market_result['count']
                market_latest = market_result['latest']
                
                # Check price_history (using token column)
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(timestamp) as latest
                    FROM price_history 
                    WHERE token = ? AND timestamp >= datetime('now', '-24 hours')
                """, (token,))
                
                price_result = cursor.fetchone()
                price_count = price_result['count']
                price_latest = price_result['latest']
                
                # Determine status
                if market_count > 0 and price_count > 0:
                    status = "✅ BOTH"
                elif market_count > 0:
                    status = "⚠️  MARKET_ONLY"
                elif price_count > 0:
                    status = "⚠️  PRICE_ONLY"
                else:
                    status = "❌ MISSING"
                
                print(f"   {token:<6} {status:<12} Market:{market_count:>3} Price:{price_count:>3}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error in reference tokens test: {e}")
        
        print("\n" + "=" * 60 + "\n")
    
    def test_crypto_data_simulation(self):
        """Simulate what _get_crypto_data() would return"""
        print("🚀 Testing _get_crypto_data() Simulation")
        print("-" * 60)
        
        try:
            # Simulate building market_data dictionary as the bot would
            market_data = {}
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent market data for all chains
            cursor.execute("""
                SELECT DISTINCT chain, price, volume, price_change_24h, market_cap, timestamp
                FROM market_data 
                WHERE timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            """)
            
            recent_data = cursor.fetchall()
            
            # Group by chain and get most recent for each
            chain_data = {}
            for row in recent_data:
                chain = row['chain']
                if chain not in chain_data:
                    chain_data[chain] = {
                        'current_price': row['price'],
                        'volume': row['volume'],
                        'price_change_percentage_24h': row['price_change_24h'],
                        'market_cap': row['market_cap'],
                        'timestamp': row['timestamp']
                    }
            
            print(f"📊 Simulated market_data dictionary:")
            print(f"   Total tokens: {len(chain_data)}")
            print("   Sample data:")
            
            for i, (token, data) in enumerate(list(chain_data.items())[:5]):
                print(f"     {token}: ${data['current_price']:,.2f} ({data['price_change_percentage_24h']:+.2f}%)")
            
            # Test volatility method compatibility
            print(f"\n🎯 Volatility Method Compatibility Test:")
            reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
            
            available_refs = []
            missing_refs = []
            
            for ref_token in reference_tokens:
                if ref_token in chain_data:
                    available_refs.append(ref_token)
                    print(f"   ✅ {ref_token}: Available in market_data")
                else:
                    missing_refs.append(ref_token)
                    print(f"   ❌ {ref_token}: Missing from market_data")
            
            print(f"\n📈 Volatility Calculation Feasibility:")
            if len(available_refs) >= 3:
                print(f"   ✅ FEASIBLE: {len(available_refs)}/{len(reference_tokens)} reference tokens available")
                print(f"   Can calculate relative volatility with {available_refs}")
            else:
                print(f"   ❌ NOT FEASIBLE: Only {len(available_refs)}/{len(reference_tokens)} reference tokens available")
                print(f"   Missing: {missing_refs}")
            
            conn.close()
            
            return chain_data
            
        except Exception as e:
            print(f"❌ Error in crypto data simulation: {e}")
            return {}
        
        print("\n" + "=" * 60 + "\n")

def main():
    """Main execution function"""
    print("🔍 Chain/Token Mapping Diagnostic Tool")
    print("This will test if your chain/token mapping is actually working")
    print()
    
    diagnostic = ChainTokenDiagnostic()
    diagnostic.run_chain_token_tests()
    
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()