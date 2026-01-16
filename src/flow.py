#!/usr/bin/env python3
"""
🔍 SPARKLINE STORAGE FLOW DIAGNOSTIC 🔍

This script traces the exact flow of sparkline data from API fetch to database storage
to identify where data is getting lost in the pipeline.

Usage:
    python storage_flow_diagnostic.py
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sqlite3

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from coingecko_handler import CoinGeckoHandler
    from database import CryptoDatabase
    from utils.logger import logger
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class StorageFlowDiagnostic:
    """Diagnostic tool to trace sparkline data storage flow"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'storage_flow_tests': {},
            'method_trace': [],
            'database_before': {},
            'database_after': {},
            'errors_found': []
        }
        
        # Initialize handlers
        try:
            self.coingecko = CoinGeckoHandler(
                base_url="https://api.coingecko.com/api/v3",
                cache_duration=60
            )
            self.database = CryptoDatabase()
            print("✅ Initialized handlers for storage flow testing")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            sys.exit(1)
    
    def run_storage_flow_diagnostic(self) -> Dict[str, Any]:
        """Run complete storage flow diagnostic"""
        print("🔍 STARTING STORAGE FLOW DIAGNOSTIC")
        print("=" * 50)
        
        # Test 1: Check database state before
        print("\n📊 CHECKING DATABASE STATE BEFORE")
        self._check_database_state_before()
        
        # Test 2: Trace API fetch and storage
        print("\n🔄 TRACING API FETCH → STORAGE FLOW")
        self._trace_api_to_storage_flow()
        
        # Test 3: Check database state after
        print("\n📊 CHECKING DATABASE STATE AFTER")
        self._check_database_state_after()
        
        # Test 4: Test manual storage
        print("\n🧪 TESTING MANUAL STORAGE")
        self._test_manual_storage()
        
        # Test 5: Test storage method functionality
        print("\n🔧 TESTING STORAGE METHODS")
        self._test_storage_methods()
        
        # Test 6: Check for silent errors
        print("\n⚠️  CHECKING FOR SILENT ERRORS")
        self._check_for_silent_errors()
        
        # Generate analysis
        print("\n💡 ANALYZING RESULTS")
        self._analyze_storage_flow()
        
        # Save results
        self._save_results()
        
        print("\n🎉 STORAGE FLOW DIAGNOSTIC COMPLETE!")
        print("=" * 50)
        return self.results
    
    def _check_database_state_before(self):
        """Check database state before any operations"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Check sparkline_data table
            cursor.execute("SELECT COUNT(*) FROM sparkline_data")
            sparkline_table_count = cursor.fetchone()[0]
            
            # Check price_history table for sparkline_data column
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE sparkline_data IS NOT NULL")
            sparkline_column_count = cursor.fetchone()[0]
            
            # Get recent records
            cursor.execute("""
                SELECT token, COUNT(*) as count, MAX(timestamp) as last_update 
                FROM price_history 
                GROUP BY token 
                ORDER BY last_update DESC 
                LIMIT 5
            """)
            recent_records = cursor.fetchall()
            
            self.results['database_before'] = {
                'sparkline_table_records': sparkline_table_count,
                'sparkline_column_records': sparkline_column_count,
                'recent_price_records': [
                    {'token': r[0], 'count': r[1], 'last_update': r[2]} 
                    for r in recent_records
                ],
                'total_price_records': self._get_total_price_records()
            }
            
            print(f"  📋 Sparkline table records: {sparkline_table_count}")
            print(f"  📋 Sparkline column records: {sparkline_column_count}")
            print(f"  📋 Recent price records: {len(recent_records)}")
            
            conn.close()
            
        except Exception as e:
            print(f"  ❌ Database check failed: {e}")
            self.results['errors_found'].append(f"Database check before: {e}")
    
    def _trace_api_to_storage_flow(self):
        """Trace the complete flow from API to storage"""
        test_token = 'bitcoin'  # Use bitcoin for reliability
        
        print(f"  🔄 Tracing complete flow for {test_token}")
        
        try:
            # Step 1: Fetch from API
            print(f"    Step 1: Fetching API data...")
            
            params = {
                'vs_currency': 'usd',
                'ids': test_token,
                'sparkline': 'true',
                'include_24hr_change': 'true'
            }
            
            start_time = time.time()
            market_data = self.coingecko.get_market_data(
                params=params,
                timeframe="24h",
                priority_tokens=None,
                include_price_history=False
            )
            fetch_time = time.time() - start_time
            
            if market_data and len(market_data) > 0:
                token_data = market_data[0]
                sparkline_in_api = token_data.get('sparkline_in_7d', {})
                
                if isinstance(sparkline_in_api, dict) and 'price' in sparkline_in_api:
                    prices = sparkline_in_api['price']
                    
                    step1_result = {
                        'status': 'success',
                        'fetch_time': fetch_time,
                        'sparkline_found': True,
                        'sparkline_length': len(prices),
                        'sample_prices': prices[:5] if len(prices) > 0 else []
                    }
                    
                    print(f"    ✅ Step 1: Got {len(prices)} sparkline points in {fetch_time:.2f}s")
                    
                    # Step 2: Check if storage was called automatically
                    print(f"    Step 2: Checking automatic storage...")
                    
                    # Look for any storage methods that might have been called
                    # This is tricky without instrumenting the actual code
                    
                    # Step 3: Try manual storage
                    print(f"    Step 3: Testing manual storage...")
                    
                    storage_success = self._test_sparkline_storage(test_token, prices)
                    
                    step1_result['manual_storage_success'] = storage_success
                    
                else:
                    step1_result = {
                        'status': 'api_success_no_sparkline',
                        'fetch_time': fetch_time,
                        'sparkline_found': False,
                        'token_data_keys': list(token_data.keys())
                    }
                    print(f"    ⚠️  Step 1: API success but no sparkline data")
                    
            else:
                step1_result = {
                    'status': 'api_failed',
                    'fetch_time': fetch_time,
                    'market_data_type': type(market_data).__name__,
                    'market_data_length': len(market_data) if isinstance(market_data, list) else 0
                }
                print(f"    ❌ Step 1: API fetch failed")
            
            self.results['storage_flow_tests']['api_to_storage'] = step1_result
            
        except Exception as e:
            print(f"    ❌ Flow trace failed: {e}")
            self.results['errors_found'].append(f"API to storage flow: {e}")
    
    def _test_sparkline_storage(self, token: str, prices: List[float]) -> bool:
        """Test manual sparkline storage"""
        try:
            # Test both storage methods
            
            # Method 1: store_sparkline_data (table approach)
            table_success = self.database.store_sparkline_data(
                token=token,
                sparkline_array=prices,
                timeframe="24h",
                timestamp=datetime.now()
            )
            
            # Method 2: Try storing to price_history with sparkline_data column
            column_success = self._test_column_storage(token, prices)
            
            print(f"      📊 Table storage: {'✅' if table_success else '❌'}")
            print(f"      📊 Column storage: {'✅' if column_success else '❌'}")
            
            return table_success or column_success
            
        except Exception as e:
            print(f"      ❌ Storage test failed: {e}")
            self.results['errors_found'].append(f"Manual storage test: {e}")
            return False
    
    def _test_column_storage(self, token: str, prices: List[float]) -> bool:
        """Test storing sparkline data in price_history column"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Store in price_history table with sparkline_data column
            sparkline_json = json.dumps(prices)
            current_price = prices[-1] if prices else 0
            timestamp = datetime.now()
            
            cursor.execute("""
                INSERT OR REPLACE INTO price_history (
                    token, timestamp, price, sparkline_data
                ) VALUES (?, ?, ?, ?)
            """, (token, timestamp, current_price, sparkline_json))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"      ❌ Column storage failed: {e}")
            return False
    
    def _check_database_state_after(self):
        """Check database state after operations"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Check sparkline_data table
            cursor.execute("SELECT COUNT(*) FROM sparkline_data")
            sparkline_table_count = cursor.fetchone()[0]
            
            # Check price_history table for sparkline_data column
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE sparkline_data IS NOT NULL")
            sparkline_column_count = cursor.fetchone()[0]
            
            # Get most recent sparkline data
            cursor.execute("""
                SELECT token, timestamp, LENGTH(sparkline_data) as data_length
                FROM price_history 
                WHERE sparkline_data IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_sparkline = cursor.fetchall()
            
            self.results['database_after'] = {
                'sparkline_table_records': sparkline_table_count,
                'sparkline_column_records': sparkline_column_count,
                'recent_sparkline_records': [
                    {'token': r[0], 'timestamp': r[1], 'data_length': r[2]} 
                    for r in recent_sparkline
                ]
            }
            
            print(f"  📋 Sparkline table records: {sparkline_table_count}")
            print(f"  📋 Sparkline column records: {sparkline_column_count}")
            print(f"  📋 New sparkline records: {len(recent_sparkline)}")
            
            # Check if data increased
            before_table = self.results['database_before']['sparkline_table_records']
            before_column = self.results['database_before']['sparkline_column_records']
            
            table_increased = sparkline_table_count > before_table
            column_increased = sparkline_column_count > before_column
            
            if table_increased or column_increased:
                print(f"  ✅ Data was stored successfully!")
            else:
                print(f"  ⚠️  No new data stored")
            
            conn.close()
            
        except Exception as e:
            print(f"  ❌ Database check after failed: {e}")
            self.results['errors_found'].append(f"Database check after: {e}")
    
    def _test_manual_storage(self):
        """Test storage methods directly"""
        test_data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        test_token = "TEST_TOKEN"
        
        print(f"  🧪 Testing direct storage methods with {test_token}")
        
        try:
            # Test store_sparkline_data method
            result1 = self.database.store_sparkline_data(
                token=test_token,
                sparkline_array=test_data,
                timeframe="test",
                timestamp=datetime.now()
            )
            
            print(f"    📊 store_sparkline_data: {'✅' if result1 else '❌'}")
            
            # Test enhance_market_data_with_sparkline
            test_market_data = {
                test_token: {
                    'current_price': 1.5,
                    'sparkline_in_7d': {'price': test_data}
                }
            }
            
            enhanced_data = self.database.enhance_market_data_with_sparkline(test_market_data)
            enhancement_worked = enhanced_data and test_token in enhanced_data
            
            print(f"    📊 enhance_market_data: {'✅' if enhancement_worked else '❌'}")
            
            self.results['storage_flow_tests']['manual_tests'] = {
                'store_sparkline_data': result1,
                'enhance_market_data': enhancement_worked
            }
            
        except Exception as e:
            print(f"    ❌ Manual test failed: {e}")
            self.results['errors_found'].append(f"Manual storage test: {e}")
    
    def _test_storage_methods(self):
        """Test individual storage method functionality"""
        print("  🔧 Testing storage method internals")
        
        try:
            # Test database connection
            conn, cursor = self.database._get_connection()
            print(f"    ✅ Database connection: Working")
            
            # Test table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sparkline_data'")
            table_exists = cursor.fetchone() is not None
            print(f"    📋 sparkline_data table: {'✅ Exists' if table_exists else '❌ Missing'}")
            
            # Test column existence
            cursor.execute("PRAGMA table_info(price_history)")
            columns = [row[1] for row in cursor.fetchall()]
            column_exists = 'sparkline_data' in columns
            print(f"    📋 sparkline_data column: {'✅ Exists' if column_exists else '❌ Missing'}")
            
            # Test write permissions
            try:
                cursor.execute("INSERT INTO sparkline_data (token, timeframe, sequence_number, price, data_timestamp) VALUES (?, ?, ?, ?, ?)",
                             ('TEST_WRITE', 'test', 1, 1.0, datetime.now()))
                conn.commit()
                
                cursor.execute("DELETE FROM sparkline_data WHERE token = 'TEST_WRITE'")
                conn.commit()
                print(f"    ✅ Write permissions: Working")
                
            except Exception as e:
                print(f"    ❌ Write permissions: Failed - {e}")
            
        except Exception as e:
            print(f"    ❌ Storage method test failed: {e}")
            self.results['errors_found'].append(f"Storage methods test: {e}")
    
    def _check_for_silent_errors(self):
        """Check for silent errors in the storage flow"""
        print("  ⚠️  Checking for silent errors and issues")
        
        try:
            # Check log files for errors
            log_errors = self._scan_logs_for_errors()
            if log_errors:
                print(f"    ❌ Found {len(log_errors)} errors in logs")
                for error in log_errors[-3:]:  # Show last 3 errors
                    print(f"      - {error}")
            else:
                print(f"    ✅ No recent errors in logs")
            
            # Check for file permissions
            db_writable = os.access(self.database.db_path, os.W_OK)
            print(f"    📁 Database file writable: {'✅' if db_writable else '❌'}")
            
            # Check database integrity
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            conn.close()
            
            print(f"    🔍 Database integrity: {'✅' if integrity_result == 'ok' else '❌'}")
            
        except Exception as e:
            print(f"    ❌ Silent error check failed: {e}")
            self.results['errors_found'].append(f"Silent error check: {e}")
    
    def _scan_logs_for_errors(self) -> List[str]:
        """Scan log files for recent errors"""
        try:
            # Look for common log files
            log_files = [
                'logs/eth_btc_correlation.log',
                'logs/bull.log',
                'logs/coingecko.log'
            ]
            
            errors = []
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Get last 50 lines and look for errors
                        recent_lines = lines[-50:] if len(lines) > 50 else lines
                        for line in recent_lines:
                            if 'ERROR' in line or 'sparkline' in line.lower():
                                errors.append(line.strip())
            
            return errors[-10:]  # Return last 10 errors
            
        except Exception:
            return []
    
    def _get_total_price_records(self) -> int:
        """Get total number of price records"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    
    def _analyze_storage_flow(self):
        """Analyze the storage flow results"""
        analysis = {
            'data_was_stored': False,
            'storage_method_working': False,
            'likely_issues': [],
            'recommendations': []
        }
        
        # Check if data was actually stored
        before_total = (self.results['database_before']['sparkline_table_records'] + 
                       self.results['database_before']['sparkline_column_records'])
        after_total = (self.results['database_after']['sparkline_table_records'] + 
                      self.results['database_after']['sparkline_column_records'])
        
        analysis['data_was_stored'] = after_total > before_total
        
        # Check if storage methods work
        manual_tests = self.results['storage_flow_tests'].get('manual_tests', {})
        analysis['storage_method_working'] = any(manual_tests.values())
        
        # Determine likely issues
        if not analysis['data_was_stored'] and analysis['storage_method_working']:
            analysis['likely_issues'].append("Storage methods work but automatic storage isn't happening")
            analysis['recommendations'].append("Check if API fetch is calling storage methods")
        
        if not analysis['storage_method_working']:
            analysis['likely_issues'].append("Storage methods themselves are failing")
            analysis['recommendations'].append("Check database permissions and table structure")
        
        if len(self.results['errors_found']) > 0:
            analysis['likely_issues'].append("Silent errors detected")
            analysis['recommendations'].append("Review error logs and fix underlying issues")
        
        self.results['analysis'] = analysis
        
        # Print analysis
        print(f"  📊 Data was stored: {'✅' if analysis['data_was_stored'] else '❌'}")
        print(f"  📊 Storage methods work: {'✅' if analysis['storage_method_working'] else '❌'}")
        
        if analysis['likely_issues']:
            print(f"  ⚠️  Likely issues:")
            for issue in analysis['likely_issues']:
                print(f"    - {issue}")
        
        if analysis['recommendations']:
            print(f"  💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")
    
    def _save_results(self):
        """Save diagnostic results"""
        try:
            filename = f"storage_flow_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")


def main():
    """Run the storage flow diagnostic"""
    diagnostic = StorageFlowDiagnostic()
    results = diagnostic.run_storage_flow_diagnostic()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 STORAGE FLOW SUMMARY")
    print("=" * 50)
    
    analysis = results.get('analysis', {})
    
    if analysis.get('data_was_stored'):
        print("🎉 SUCCESS: Data is being stored properly!")
    else:
        print("⚠️  ISSUE: Data is not being stored")
    
    if analysis.get('storage_method_working'):
        print("✅ Storage methods are functional")
    else:
        print("❌ Storage methods have issues")
    
    errors_count = len(results.get('errors_found', []))
    if errors_count > 0:
        print(f"⚠️  Found {errors_count} errors during testing")
    
    issues = analysis.get('likely_issues', [])
    if issues:
        print(f"\n🔍 KEY ISSUES IDENTIFIED:")
        for issue in issues:
            print(f"  • {issue}")
    
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        print(f"\n💡 NEXT STEPS:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    return results


if __name__ == "__main__":
    main()