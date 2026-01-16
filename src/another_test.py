#!/usr/bin/env python3
"""
Test Script for build_sparkline_from_price_history Method
========================================================

This script tests the current and fixed versions of the sparkline method
to identify why it's returning empty arrays.
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List

# Add src directory to path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils.logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Create mock logger wrapper
    class MockLogger:
        def __init__(self, base):
            self.logger = base
        def log_error(self, category, msg):
            self.logger.error(f"{category}: {msg}")
    logger = MockLogger(logger)

class SparklineMethodTester:
    """Test both original and fixed sparkline methods"""
    
    def __init__(self, db_path: str = "data/crypto_history.db"):
        self.db_path = db_path
        self.test_tokens = ['KAITO', 'UNI', 'ETH', 'BTC', 'SOL']
        
    def _get_connection(self):
        """Simulate your database connection method"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()
    
    def test_original_method(self, token: str, hours: int = 168) -> List[float]:
        """
        Test the ORIGINAL build_sparkline_from_price_history method
        This replicates your current implementation
        """
        try:
            conn, cursor = self._get_connection()
            
            print(f"\n🧪 TESTING ORIGINAL METHOD for {token}")
            print("-" * 50)
            
            # Original query with string concatenation (the problematic one)
            original_query = """
                SELECT price, timestamp 
                FROM price_history 
                WHERE token = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """
            
            print(f"📝 Original query: {original_query}")
            print(f"📝 Parameters: token='{token}', hours={hours}")
            
            cursor.execute(original_query, (token, hours))
            results = cursor.fetchall()
            
            print(f"📊 Original method results: {len(results)} records")
            
            if not results:
                print("❌ Original method returned no results")
                return []
            
            # Extract price array
            price_array = [float(row['price']) for row in results]
            
            print(f"✅ Original method extracted {len(price_array)} prices")
            if len(price_array) >= 3:
                print(f"📈 Price range: ${price_array[0]:.6f} → ${price_array[-1]:.6f}")
            
            return price_array
            
        except Exception as e:
            print(f"❌ Original method error: {e}")
            return []
    
    def test_fixed_method(self, token: str, hours: int = 168) -> List[float]:
        """
        Test the FIXED build_sparkline_from_price_history method
        """
        try:
            conn, cursor = self._get_connection()
            
            print(f"\n🔧 TESTING FIXED METHOD for {token}")
            print("-" * 50)
            
            # Calculate cutoff timestamp using Python instead of SQL concatenation
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"📅 Cutoff time calculated: {cutoff_str}")
            
            # FIXED QUERY: Use direct timestamp comparison instead of string concatenation
            fixed_query = """
                SELECT price, timestamp 
                FROM price_history 
                WHERE (token = ? OR token = ?) 
                AND timestamp >= ?
                AND price IS NOT NULL 
                AND price > 0
                ORDER BY timestamp ASC
            """
            
            print(f"📝 Fixed query: {fixed_query}")
            print(f"📝 Parameters: token='{token}', token_upper='{token.upper()}', cutoff='{cutoff_str}'")
            
            cursor.execute(fixed_query, (token, token.upper(), cutoff_str))
            results = cursor.fetchall()
            
            print(f"📊 Fixed method results: {len(results)} records")
            
            if not results:
                print("🔄 No results with standard approach, trying alternatives...")
                
                # Try with lowercase token
                cursor.execute("""
                    SELECT price, timestamp 
                    FROM price_history 
                    WHERE token = ?
                    AND timestamp >= ?
                    AND price IS NOT NULL 
                    AND price > 0
                    ORDER BY timestamp ASC
                """, (token.lower(), cutoff_str))
                
                results = cursor.fetchall()
                print(f"📊 Lowercase attempt: {len(results)} records")
                
                if not results:
                    # Try with longer timeframe
                    longer_cutoff = datetime.now() - timedelta(hours=hours * 2)
                    longer_cutoff_str = longer_cutoff.strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute("""
                        SELECT price, timestamp 
                        FROM price_history 
                        WHERE (token = ? OR token = ? OR token = ?)
                        AND timestamp >= ?
                        AND price IS NOT NULL 
                        AND price > 0
                        ORDER BY timestamp ASC
                    """, (token, token.upper(), token.lower(), longer_cutoff_str))
                    
                    results = cursor.fetchall()
                    print(f"📊 Extended timeframe (2x): {len(results)} records")
            
            if not results:
                print("❌ Fixed method also returned no results")
                return []
            
            # Extract price array with validation
            price_array = []
            invalid_prices = 0
            
            for row in results:
                try:
                    price = float(row['price']) if row['price'] is not None else None
                    if price is not None and price > 0:
                        price_array.append(price)
                    else:
                        invalid_prices += 1
                except (ValueError, TypeError):
                    invalid_prices += 1
                    continue
            
            if invalid_prices > 0:
                print(f"⚠️ Skipped {invalid_prices} invalid price records")
            
            print(f"✅ Fixed method extracted {len(price_array)} valid prices")
            if len(price_array) >= 3:
                print(f"📈 Price range: ${price_array[0]:.6f} → ${price_array[-1]:.6f}")
            
            return price_array
            
        except Exception as e:
            print(f"❌ Fixed method error: {e}")
            return []
    
    def diagnose_token_data(self, token: str):
        """Diagnose what data exists for a specific token"""
        try:
            conn, cursor = self._get_connection()
            
            print(f"\n🔍 DIAGNOSING TOKEN DATA for {token}")
            print("=" * 60)
            
            # Check various case combinations
            for case_variant in [token, token.upper(), token.lower()]:
                cursor.execute("SELECT COUNT(*) as count FROM price_history WHERE token = ?", (case_variant,))
                count = cursor.fetchone()['count']
                print(f"📊 Records for '{case_variant}': {count:,}")
            
            # Check fuzzy matches
            cursor.execute("SELECT COUNT(*) as count FROM price_history WHERE token LIKE ?", (f"%{token}%",))
            fuzzy_count = cursor.fetchone()['count']
            print(f"📊 Fuzzy matches for '{token}': {fuzzy_count:,}")
            
            # Check recent data (last 7 days)
            recent_cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM price_history 
                WHERE (token = ? OR token = ? OR token = ?)
                AND timestamp >= ?
            """, (token, token.upper(), token.lower(), recent_cutoff))
            recent_count = cursor.fetchone()['count']
            print(f"📊 Recent records (7 days): {recent_count:,}")
            
            # Show sample recent records
            cursor.execute("""
                SELECT token, timestamp, price 
                FROM price_history 
                WHERE (token = ? OR token = ? OR token = ?)
                ORDER BY timestamp DESC 
                LIMIT 5
            """, (token, token.upper(), token.lower()))
            samples = cursor.fetchall()
            
            if samples:
                print(f"\n📋 Sample recent records:")
                for sample in samples:
                    print(f"   {sample['token']} | {sample['timestamp']} | ${sample['price']:.6f}")
            else:
                print(f"\n❌ No sample records found for {token}")
            
            # Check date range
            cursor.execute("""
                SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest, COUNT(*) as total
                FROM price_history 
                WHERE (token = ? OR token = ? OR token = ?)
            """, (token, token.upper(), token.lower()))
            date_range = cursor.fetchone()
            
            if date_range and date_range['total'] > 0:
                print(f"\n📅 Date range: {date_range['earliest']} to {date_range['latest']}")
                print(f"📊 Total records: {date_range['total']:,}")
            
        except Exception as e:
            print(f"❌ Diagnosis error: {e}")
    
    def test_all_tokens(self):
        """Test both methods on all test tokens"""
        print("🚀 STARTING COMPREHENSIVE SPARKLINE METHOD TEST")
        print("=" * 80)
        
        # First, show database overview
        try:
            conn, cursor = self._get_connection()
            cursor.execute("SELECT COUNT(*) as total FROM price_history")
            total_records = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(DISTINCT token) as unique_tokens FROM price_history")
            unique_tokens = cursor.fetchone()['unique_tokens']
            
            print(f"📊 Database Overview:")
            print(f"   Total records: {total_records:,}")
            print(f"   Unique tokens: {unique_tokens}")
            
            # Show top tokens by record count
            cursor.execute("""
                SELECT token, COUNT(*) as count 
                FROM price_history 
                GROUP BY token 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_tokens = cursor.fetchall()
            print(f"\n📈 Top tokens by record count:")
            for token_info in top_tokens:
                print(f"   {token_info['token']}: {token_info['count']:,} records")
                
        except Exception as e:
            print(f"❌ Database overview error: {e}")
        
        # Test each token
        for token in self.test_tokens:
            print(f"\n{'='*80}")
            print(f"🎯 TESTING TOKEN: {token}")
            print(f"{'='*80}")
            
            # Diagnose the token first
            self.diagnose_token_data(token)
            
            # Test original method
            original_result = self.test_original_method(token)
            
            # Test fixed method
            fixed_result = self.test_fixed_method(token)
            
            # Compare results
            print(f"\n📊 COMPARISON for {token}:")
            print(f"   Original method: {len(original_result)} points")
            print(f"   Fixed method: {len(fixed_result)} points")
            
            if len(original_result) == 0 and len(fixed_result) > 0:
                print(f"✅ FIXED METHOD SOLVED THE PROBLEM for {token}!")
            elif len(original_result) == 0 and len(fixed_result) == 0:
                print(f"❌ Both methods failed for {token} - data issue")
            elif len(original_result) > 0:
                print(f"ℹ️ Original method already worked for {token}")
    
    def run_focused_test(self, token: str = "KAITO"):
        """Run a focused test on a single problematic token"""
        print(f"🎯 FOCUSED TEST ON {token}")
        print("=" * 50)
        
        self.diagnose_token_data(token)
        original_result = self.test_original_method(token)
        fixed_result = self.test_fixed_method(token)
        
        print(f"\n🏆 FINAL RESULT for {token}:")
        print(f"   Original: {len(original_result)} points")
        print(f"   Fixed: {len(fixed_result)} points")
        
        if len(fixed_result) >= 10:
            print(f"✅ SUCCESS! Fixed method provides {len(fixed_result)} points (>= 10 minimum)")
        else:
            print(f"❌ Still insufficient data: {len(fixed_result)} points (need >= 10)")


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test sparkline method fixes')
    parser.add_argument('--token', '-t', default='KAITO', help='Single token to test (default: KAITO)')
    parser.add_argument('--all', '-a', action='store_true', help='Test all tokens')
    parser.add_argument('--db', '-d', default='data/crypto_history.db', help='Database path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        print("💡 Try: python3 test_sparkline.py --db /path/to/your/crypto_history.db")
        return
    
    tester = SparklineMethodTester(args.db)
    
    if args.all:
        tester.test_all_tokens()
    else:
        tester.run_focused_test(args.token)


if __name__ == "__main__":
    main()