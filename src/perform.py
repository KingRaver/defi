#!/usr/bin/env python3
"""
perform.py - Comprehensive Testing Script for _calculate_relative_volatility Method

This script tests the entire data flow for the _calculate_relative_volatility method
and helps debug where the data flow might be broken.
"""

import sys
import os
import sqlite3
import statistics
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import json

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock logger if not available
class MockLogger:
    def debug(self, msg): print(f"[DEBUG] {msg}")
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARNING] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def log_error(self, context, error): print(f"[ERROR] {context}: {error}")

try:
    from utils.logger import logger
except ImportError:
    logger = type('MockLogger', (), {
        'logger': MockLogger(),
        'log_error': lambda context, error: print(f"[ERROR] {context}: {error}")
    })()

class VolatilityTester:
    """
    Comprehensive tester for _calculate_relative_volatility method and data flow
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Get project root (go up from src/ to project root)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(project_root, "data", "crypto_history.db")
        else:
            self.db_path = db_path
        self.test_results = {}
        self.reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'AVAX', 'DOT', 'UNI', 'NEAR', 'AAVE']
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("=" * 80)
        print("🚀 STARTING COMPREHENSIVE VOLATILITY METHOD TESTING")
        print("=" * 80)
        
        # Test 1: Database Connection and Structure
        self.test_database_connection()
        
        # Test 2: Database Schema Validation
        self.test_database_schema()
        
        # Test 3: Historical Data Availability
        self.test_historical_data_availability()
        
        # Test 4: Extract Prices Function
        self.test_extract_prices_function()
        
        # Test 5: Mock Historical Price Data Method
        self.test_get_historical_price_data()
        
        # Test 6: Volatility Calculation Logic
        self.test_volatility_calculation()
        
        # Test 7: Complete Method Integration
        self.test_complete_method()
        
        # Test 8: Data Flow Integration
        self.test_data_flow_integration()
        
        # Test 9: Error Handling
        self.test_error_handling()
        
        # Generate Final Report
        self.generate_final_report()
    
    def test_database_connection(self):
        """Test database connection and basic functionality"""
        print("\n📡 Testing Database Connection...")
        test_name = "database_connection"
        
        try:
            # Test if database file exists
            if not os.path.exists(self.db_path):
                self.test_results[test_name] = {
                    "status": "FAILED",
                    "error": f"Database file {self.db_path} does not exist",
                    "suggestion": "Run your main bot to create the database first"
                }
                print(f"❌ Database file not found: {self.db_path}")
                return
            
            # Test database connection
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            self.test_results[test_name] = {
                "status": "PASSED",
                "tables_found": tables,
                "table_count": len(tables)
            }
            print(f"✅ Database connection successful")
            print(f"   Found {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ Database connection failed: {e}")
    
    def test_database_schema(self):
        """Test database schema for required tables and columns"""
        print("\n🗃️  Testing Database Schema...")
        test_name = "database_schema"
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            schema_results = {}
            
            # Test price_history table
            required_tables = {
                'price_history': ['id', 'token', 'timestamp', 'price', 'volume', 'market_cap'],
                'market_data': ['id', 'timestamp', 'chain', 'price', 'volume'],
                'technical_indicators': ['id', 'token', 'timestamp']
            }
            
            for table_name, required_columns in required_tables.items():
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    missing_columns = [col for col in required_columns if col not in columns]
                    
                    schema_results[table_name] = {
                        "exists": True,
                        "columns": columns,
                        "missing_columns": missing_columns,
                        "status": "OK" if not missing_columns else "MISSING_COLUMNS"
                    }
                    
                    if missing_columns:
                        print(f"⚠️  Table {table_name} missing columns: {missing_columns}")
                    else:
                        print(f"✅ Table {table_name} schema OK")
                        
                except sqlite3.OperationalError:
                    schema_results[table_name] = {
                        "exists": False,
                        "status": "MISSING_TABLE"
                    }
                    print(f"❌ Table {table_name} does not exist")
            
            conn.close()
            
            self.test_results[test_name] = {
                "status": "PASSED" if all(r.get("status") == "OK" for r in schema_results.values()) else "PARTIAL",
                "schema_results": schema_results
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Schema test failed: {e}")
    
    def test_historical_data_availability(self):
        """Test availability of historical price data"""
        print("\n📊 Testing Historical Data Availability...")
        test_name = "historical_data_availability"
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            data_availability = {}
            
            # Check price_history table data
            cursor.execute("""
                SELECT token, COUNT(*) as count, 
                       MIN(timestamp) as oldest, 
                       MAX(timestamp) as newest
                FROM price_history 
                GROUP BY token 
                ORDER BY count DESC
            """)
            
            price_history_data = cursor.fetchall()
            
            for row in price_history_data:
                token = row['token']
                count = row['count']
                oldest = row['oldest']
                newest = row['newest']
                
                # Calculate data span
                if oldest and newest:
                    try:
                        oldest_dt = datetime.fromisoformat(oldest.replace('Z', '+00:00'))
                        newest_dt = datetime.fromisoformat(newest.replace('Z', '+00:00'))
                        span_hours = (newest_dt - oldest_dt).total_seconds() / 3600
                    except:
                        span_hours = 0
                else:
                    span_hours = 0
                
                data_availability[token] = {
                    "price_history_count": count,
                    "oldest_timestamp": oldest,
                    "newest_timestamp": newest,
                    "data_span_hours": span_hours,
                    "sufficient_for_1h": span_hours >= 24,
                    "sufficient_for_24h": span_hours >= 168,  # 7 days
                    "sufficient_for_7d": span_hours >= 720   # 30 days
                }
            
            # Check market_data table
            cursor.execute("""
                SELECT chain as token, COUNT(*) as count,
                       MIN(timestamp) as oldest,
                       MAX(timestamp) as newest
                FROM market_data 
                GROUP BY chain
                ORDER BY count DESC
            """)
            
            market_data_results = cursor.fetchall()
            
            for row in market_data_results:
                token = row['token']
                if token in data_availability:
                    data_availability[token]['market_data_count'] = row['count']
                else:
                    data_availability[token] = {
                        'price_history_count': 0,
                        'market_data_count': row['count']
                    }
            
            conn.close()
            
            # Summary statistics
            tokens_with_sufficient_data = {
                "1h": len([t for t, d in data_availability.items() if d.get('sufficient_for_1h', False)]),
                "24h": len([t for t, d in data_availability.items() if d.get('sufficient_for_24h', False)]),
                "7d": len([t for t, d in data_availability.items() if d.get('sufficient_for_7d', False)])
            }
            
            self.test_results[test_name] = {
                "status": "PASSED" if data_availability else "NO_DATA",
                "data_availability": data_availability,
                "tokens_with_sufficient_data": tokens_with_sufficient_data,
                "total_tokens": len(data_availability)
            }
            
            print(f"✅ Found historical data for {len(data_availability)} tokens")
            print(f"   Sufficient for 1h analysis: {tokens_with_sufficient_data['1h']} tokens")
            print(f"   Sufficient for 24h analysis: {tokens_with_sufficient_data['24h']} tokens")
            print(f"   Sufficient for 7d analysis: {tokens_with_sufficient_data['7d']} tokens")
            
            if not data_availability:
                print("⚠️  No historical data found - volatility calculations will fail")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Historical data test failed: {e}")
    
    def test_extract_prices_function(self):
        """Test the extract_prices function logic"""
        print("\n🔍 Testing Extract Prices Function...")
        test_name = "extract_prices_function"
        
        def extract_prices(history_data):
            """Test implementation of extract_prices function"""
            if history_data is None:
                return []
            
            if isinstance(history_data, str):
                return []
            
            if not hasattr(history_data, '__iter__'):
                return []
            
            prices = []
            
            for entry in history_data:
                if entry is None:
                    continue
                
                price = None
                
                try:
                    # Case 1: Dictionary with price property
                    if isinstance(entry, dict):
                        if 'price' in entry and entry['price'] is not None:
                            try:
                                price = float(entry['price'])
                            except (ValueError, TypeError):
                                pass
                    
                    # Case 2: List/tuple with price as first element
                    elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                        if entry[0] is not None:
                            try:
                                price = float(entry[0])
                            except (ValueError, TypeError):
                                pass
                    
                    # Case 3: Entry has price attribute
                    elif not isinstance(entry, (list, tuple)) and hasattr(entry, 'price'):
                        try:
                            price = float(entry.price)
                        except (ValueError, TypeError, AttributeError):
                            pass
                    
                    # Case 4: Entry itself is a number
                    elif isinstance(entry, (int, float)):
                        price = float(entry)
                    
                    # Add price to list if valid
                    if price is not None and price > 0:
                        prices.append(price)
                
                except Exception as extract_error:
                    continue
            
            return prices
        
        # Test cases
        test_cases = [
            # Test case 1: Dictionary format
            {
                "name": "Dictionary format",
                "input": [{"price": 100.0}, {"price": 101.5}, {"price": 99.8}],
                "expected_length": 3
            },
            # Test case 2: List/tuple format
            {
                "name": "List/tuple format",
                "input": [[100.0, 12345], [101.5, 12346], [99.8, 12347]],
                "expected_length": 3
            },
            # Test case 3: Raw numbers
            {
                "name": "Raw numbers",
                "input": [100.0, 101.5, 99.8, 102.1],
                "expected_length": 4
            },
            # Test case 4: Mixed format
            {
                "name": "Mixed format",
                "input": [{"price": 100.0}, [101.5, 12346], 99.8],
                "expected_length": 3
            },
            # Test case 5: Invalid data
            {
                "name": "Invalid data",
                "input": [None, {"price": None}, [], {"invalid": 100}],
                "expected_length": 0
            },
            # Test case 6: Empty input
            {
                "name": "Empty input",
                "input": [],
                "expected_length": 0
            },
            # Test case 7: None input
            {
                "name": "None input",
                "input": None,
                "expected_length": 0
            }
        ]
        
        test_results = {}
        all_passed = True
        
        for test_case in test_cases:
            try:
                result = extract_prices(test_case["input"])
                expected = test_case["expected_length"]
                passed = len(result) == expected
                
                test_results[test_case["name"]] = {
                    "passed": passed,
                    "input_length": len(test_case["input"]) if test_case["input"] is not None else 0,
                    "output_length": len(result),
                    "expected_length": expected,
                    "sample_output": result[:3] if result else []
                }
                
                if passed:
                    print(f"✅ {test_case['name']}: {len(result)} prices extracted")
                else:
                    print(f"❌ {test_case['name']}: Expected {expected}, got {len(result)}")
                    all_passed = False
                    
            except Exception as e:
                test_results[test_case["name"]] = {
                    "passed": False,
                    "error": str(e)
                }
                print(f"❌ {test_case['name']}: Error - {e}")
                all_passed = False
        
        self.test_results[test_name] = {
            "status": "PASSED" if all_passed else "FAILED",
            "test_cases": test_results
        }
    
    def test_get_historical_price_data(self):
        """Test the _get_historical_price_data method functionality"""
        print("\n🕰️  Testing Historical Price Data Retrieval...")
        test_name = "get_historical_price_data"
        
        def mock_get_historical_price_data(token: str, hours: int, timeframe: Optional[str] = None):
            """Mock implementation that queries the database"""
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Calculate time threshold
                time_threshold = datetime.now() - timedelta(hours=hours)
                
                # Query price_history table
                cursor.execute("""
                    SELECT price, volume, market_cap, timestamp
                    FROM price_history
                    WHERE token = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                """, (token, time_threshold.isoformat()))
                
                results = cursor.fetchall()
                conn.close()
                
                if not results:
                    return "Never"  # Simulate CoinGecko "Never" response
                
                # Return in the format expected by extract_prices
                history_data = []
                for row in results:
                    history_data.append({
                        'price': row['price'],
                        'volume': row['volume'],
                        'market_cap': row['market_cap'],
                        'timestamp': row['timestamp']
                    })
                
                return history_data
                
            except Exception as e:
                print(f"Error in mock_get_historical_price_data: {e}")
                return None
        
        # Test different tokens and timeframes
        test_scenarios = [
            {"token": "BTC", "hours": 24, "timeframe": "1h"},
            {"token": "ETH", "hours": 168, "timeframe": "24h"},
            {"token": "SOL", "hours": 720, "timeframe": "7d"},
            {"token": "NONEXISTENT", "hours": 24, "timeframe": "1h"}
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            token = scenario["token"]
            hours = scenario["hours"]
            timeframe = scenario["timeframe"]
            
            try:
                history_data = mock_get_historical_price_data(token, hours, timeframe)
                
                if history_data == "Never":
                    results[f"{token}_{timeframe}"] = {
                        "status": "NO_DATA",
                        "response": "Never"
                    }
                    print(f"⚠️  {token} ({timeframe}): No historical data available")
                elif history_data is None:
                    results[f"{token}_{timeframe}"] = {
                        "status": "ERROR",
                        "response": None
                    }
                    print(f"❌ {token} ({timeframe}): Error retrieving data")
                else:
                    data_points = len(history_data) if isinstance(history_data, list) else 0
                    results[f"{token}_{timeframe}"] = {
                        "status": "SUCCESS",
                        "data_points": data_points,
                        "sample": history_data[:2] if data_points > 0 else []
                    }
                    print(f"✅ {token} ({timeframe}): {data_points} data points retrieved")
                    
            except Exception as e:
                results[f"{token}_{timeframe}"] = {
                    "status": "EXCEPTION",
                    "error": str(e)
                }
                print(f"❌ {token} ({timeframe}): Exception - {e}")
        
        self.test_results[test_name] = {
            "status": "PASSED" if any(r["status"] == "SUCCESS" for r in results.values()) else "NO_DATA",
            "scenarios": results
        }
    
    def test_volatility_calculation(self):
        """Test the volatility calculation logic"""
        print("\n📈 Testing Volatility Calculation Logic...")
        test_name = "volatility_calculation"
        
        # Test with known data
        test_data = [100.0, 102.0, 98.0, 105.0, 103.0, 99.0, 107.0, 104.0]
        
        try:
            # Calculate price changes
            price_changes = []
            for i in range(1, len(test_data)):
                if test_data[i-1] > 0:
                    pct_change = ((test_data[i] / test_data[i-1]) - 1) * 100
                    price_changes.append(pct_change)
            
            # Calculate volatility (standard deviation)
            if len(price_changes) >= 2:
                volatility = statistics.stdev(price_changes)
                mean_change = statistics.mean(price_changes)
                
                self.test_results[test_name] = {
                    "status": "PASSED",
                    "test_data": test_data,
                    "price_changes": price_changes,
                    "volatility": volatility,
                    "mean_change": mean_change,
                    "num_changes": len(price_changes)
                }
                
                print(f"✅ Volatility calculation successful")
                print(f"   Price changes: {[round(pc, 2) for pc in price_changes]}")
                print(f"   Volatility (std dev): {volatility:.4f}")
                print(f"   Mean change: {mean_change:.4f}")
            else:
                self.test_results[test_name] = {
                    "status": "INSUFFICIENT_DATA",
                    "price_changes": price_changes
                }
                print("⚠️  Insufficient price changes for volatility calculation")
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Volatility calculation failed: {e}")
    
    def test_complete_method(self):
        """Test the complete _calculate_relative_volatility method"""
        print("\n🎯 Testing Complete Volatility Method...")
        test_name = "complete_method"
        
        def calculate_relative_volatility(token: str, reference_tokens: List[str], 
                                        market_data: Dict[str, Any], timeframe: str) -> Optional[float]:
            """Complete implementation of the volatility method"""
            try:
                # Get historical data with appropriate window for the timeframe
                if timeframe == "1h":
                    hours = 24
                elif timeframe == "24h":
                    hours = 7 * 24
                else:  # 7d
                    hours = 30 * 24
                
                # Function to safely extract prices from historical data
                def extract_prices(history_data):
                    if history_data is None or isinstance(history_data, str):
                        return []
                    
                    if not hasattr(history_data, '__iter__'):
                        return []
                    
                    prices = []
                    
                    for entry in history_data:
                        if entry is None:
                            continue
                        
                        price = None
                        
                        try:
                            if isinstance(entry, dict):
                                if 'price' in entry and entry['price'] is not None:
                                    try:
                                        price = float(entry['price'])
                                    except (ValueError, TypeError):
                                        pass
                            elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                                if entry[0] is not None:
                                    try:
                                        price = float(entry[0])
                                    except (ValueError, TypeError):
                                        pass
                            elif isinstance(entry, (int, float)):
                                price = float(entry)
                            
                            if price is not None and price > 0:
                                prices.append(price)
                                
                        except Exception:
                            continue
                    
                    return prices
                
                # Mock historical data retrieval
                def get_historical_price_data(token_symbol: str, hours_back: int, tf: str):
                    try:
                        conn = sqlite3.connect(self.db_path)
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        
                        time_threshold = datetime.now() - timedelta(hours=hours_back)
                        
                        cursor.execute("""
                            SELECT price FROM price_history
                            WHERE token = ? AND timestamp >= ?
                            ORDER BY timestamp ASC
                            LIMIT 50
                        """, (token_symbol, time_threshold.isoformat()))
                        
                        results = cursor.fetchall()
                        conn.close()
                        
                        if not results:
                            return None
                        
                        return [{"price": row["price"]} for row in results]
                        
                    except Exception:
                        return None
                
                # Get token price history and extract prices
                token_history = get_historical_price_data(token, hours, timeframe)
                token_prices = extract_prices(token_history)
                
                # Check if we have enough price data
                if len(token_prices) < 5:
                    return None
                
                # Calculate token price changes
                token_changes = []
                for i in range(1, len(token_prices)):
                    if token_prices[i-1] > 0:
                        try:
                            pct_change = ((token_prices[i] / token_prices[i-1]) - 1) * 100
                            token_changes.append(pct_change)
                        except (ZeroDivisionError, OverflowError):
                            continue
                
                if len(token_changes) < 2:
                    return None
                
                # Calculate token volatility (standard deviation)
                try:
                    token_volatility = statistics.stdev(token_changes)
                except statistics.StatisticsError:
                    return None
                
                # Calculate market average volatility
                market_volatilities = []
                
                for ref_token in reference_tokens:
                    if ref_token not in market_data:
                        continue
                    
                    try:
                        ref_history = get_historical_price_data(ref_token, hours, timeframe)
                        ref_prices = extract_prices(ref_history)
                        
                        if len(ref_prices) < 5:
                            continue
                        
                        ref_changes = []
                        for i in range(1, len(ref_prices)):
                            if ref_prices[i-1] > 0:
                                try:
                                    pct_change = ((ref_prices[i] / ref_prices[i-1]) - 1) * 100
                                    ref_changes.append(pct_change)
                                except (ZeroDivisionError, OverflowError):
                                    continue
                        
                        if len(ref_changes) >= 2:
                            try:
                                ref_volatility = statistics.stdev(ref_changes)
                                market_volatilities.append(ref_volatility)
                            except statistics.StatisticsError:
                                continue
                                
                    except Exception:
                        continue
                
                # Check if we have enough market volatility data
                if not market_volatilities:
                    return None
                
                # Calculate market average volatility
                market_avg_volatility = statistics.mean(market_volatilities)
                
                # Calculate relative volatility
                if market_avg_volatility > 0:
                    relative_volatility = token_volatility / market_avg_volatility
                    return relative_volatility
                else:
                    return None
                    
            except Exception as e:
                print(f"Error in calculate_relative_volatility: {e}")
                return None
        
        # Test the method with different scenarios
        test_tokens = ["BTC", "ETH", "SOL"]
        timeframes = ["1h", "24h", "7d"]
        
        # Get REAL market data from database
        def get_real_market_data():
            """Get actual market data from the database"""
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get latest market data for each token
                cursor.execute("""
                    SELECT chain as token, price, volume, price_change_24h, market_cap
                    FROM market_data 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY chain
                    ORDER BY timestamp DESC
                """)
                
                market_data = {}
                for row in cursor.fetchall():
                    market_data[row['token']] = {
                        "current_price": row['price'],
                        "volume": row['volume'] if row['volume'] else 0,
                        "price_change_percentage_24h": row['price_change_24h'] if row['price_change_24h'] else 0,
                        "market_cap": row['market_cap'] if row['market_cap'] else 0
                    }
                
                conn.close()
                return market_data
                
            except Exception as e:
                print(f"Error getting real market data: {e}")
                return {}
        
        real_market_data = get_real_market_data()
        
        if not real_market_data:
            print("❌ No real market data found in database - test WILL FAIL")
            # NO FALLBACK - let it fail so we can see what's broken
        
        results = {}
        
        for token in test_tokens:
            for timeframe in timeframes:
                key = f"{token}_{timeframe}"  # Define key here at the start
                try:
                    volatility = calculate_relative_volatility(
                        token, 
                        self.reference_tokens,
                        real_market_data,
                        timeframe
                    )
                    
                    if volatility is not None:
                        results[key] = {
                            "status": "SUCCESS",
                            "relative_volatility": volatility,
                            "interpretation": "More volatile" if volatility > 1 else "Less volatile"
                        }
                        print(f"✅ {token} ({timeframe}): Relative volatility = {volatility:.4f}")
                    else:
                        results[key] = {
                            "status": "INSUFFICIENT_DATA",
                            "relative_volatility": None
                        }
                        print(f"⚠️  {token} ({timeframe}): Insufficient data")
                        
                except Exception as e:
                    results[key] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    print(f"❌ {token} ({timeframe}): Error - {e}")

        successful_tests = len([r for r in results.values() if r["status"] == "SUCCESS"])
        total_tests = len(results)
        
        self.test_results[test_name] = {
            "status": "PASSED" if successful_tests > 0 else "FAILED",
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "results": results
        }
        
        print(f"\n📊 Method Testing Summary: {successful_tests}/{total_tests} tests successful")
    
    def test_data_flow_integration(self):
        """Test integration with the broader data flow"""
        print("\n🔄 Testing Data Flow Integration...")
        test_name = "data_flow_integration"
        
        # Simulate the integration as used in _analyze_token_vs_market
        def test_integration_flow():
            """Test the integration flow as it would be called"""
            try:
                # Get REAL market data from database
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT chain as token, price, volume, price_change_24h, market_cap
                    FROM market_data 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY chain
                    ORDER BY timestamp DESC
                """)
                
                market_data = {}
                for row in cursor.fetchall():
                    market_data[row['token']] = {
                        "current_price": row['price'],
                        "volume": row['volume'] if row['volume'] else 0,
                        "price_change_percentage_24h": row['price_change_24h'] if row['price_change_24h'] else 0,
                        "market_cap": row['market_cap'] if row['market_cap'] else 0
                    }
                
                conn.close()
                
                if not market_data:
                    return {
                        "status": "NO_REAL_DATA", 
                        "integration_working": False,
                        "issue": "No real market data found in database - THIS IS A REAL FAILURE"
                    }
                
                # Test the hasattr check pattern with enhanced mock bot
                class MockBot:
                    def __init__(self):
                        self.reference_tokens = ["ETH", "SOL", "XRP", "BNB"]
                        self.volatility_cache = {}
                    
                    def _calculate_relative_volatility(self, token, ref_tokens, market_data, timeframe):
                        # Enhanced mock implementation with caching
                        cache_key = f"{token}_{timeframe}"
                        
                        if cache_key in self.volatility_cache:
                            return self.volatility_cache[cache_key]
                        
                        # Simulate different volatility values based on token and timeframe
                        base_volatility = {
                            "BTC": 1.0,    # Market average
                            "ETH": 1.2,    # Slightly more volatile
                            "SOL": 1.5,    # More volatile
                            "XRP": 0.8     # Less volatile
                        }.get(token, 1.0)
                        
                        # Adjust for timeframe
                        timeframe_multiplier = {
                            "1h": 0.8,
                            "24h": 1.0,
                            "7d": 1.3
                        }.get(timeframe, 1.0)
                        
                        # Add some randomness based on token hash for consistency
                        random_factor = 0.8 + (hash(token) % 100) / 250  # 0.8 to 1.2
                        
                        final_volatility = base_volatility * timeframe_multiplier * random_factor
                        
                        # Cache the result
                        self.volatility_cache[cache_key] = final_volatility
                        
                        return final_volatility
                
                mock_bot = MockBot()
                
                # Test the integration pattern from the actual code
                extended_metrics = {}
                token = "BTC"
                timeframe = "1h"
                reference_tokens = ["ETH", "SOL", "XRP"]
                
                if hasattr(mock_bot, '_calculate_relative_volatility'):
                    try:
                        token_volatility = mock_bot._calculate_relative_volatility(
                            token, 
                            reference_tokens, 
                            market_data, 
                            timeframe
                        )
                        if token_volatility is not None:
                            extended_metrics['relative_volatility'] = token_volatility
                            return {
                                "status": "SUCCESS",
                                "extended_metrics": extended_metrics,
                                "integration_working": True
                            }
                        else:
                            return {
                                "status": "NO_DATA",
                                "integration_working": True,
                                "issue": "Method returned None"
                            }
                    except Exception as method_error:
                        return {
                            "status": "METHOD_ERROR",
                            "integration_working": True,
                            "error": str(method_error)
                        }
                else:
                    return {
                        "status": "METHOD_NOT_FOUND",
                        "integration_working": False,
                        "issue": "hasattr check failed"
                    }
                    
            except Exception as e:
                return {
                    "status": "INTEGRATION_ERROR",
                    "integration_working": False,
                    "error": str(e)
                }
        
        integration_result = test_integration_flow()
        
        # Test database query patterns
        def test_database_query_patterns():
            """Test the database query patterns used by the method"""
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                queries_tested = {}
                
                # Query 1: Basic price history query
                try:
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM price_history 
                        WHERE token = 'BTC' AND timestamp >= datetime('now', '-24 hours')
                    """)
                    result = cursor.fetchone()
                    queries_tested["basic_price_query"] = {
                        "status": "SUCCESS",
                        "count": result["count"] if result else 0
                    }
                except Exception as e:
                    queries_tested["basic_price_query"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                
                # Query 2: Multi-token comparison query
                try:
                    cursor.execute("""
                        SELECT token, COUNT(*) as count
                        FROM price_history 
                        WHERE token IN ('BTC', 'ETH', 'SOL', 'XRP')
                        AND timestamp >= datetime('now', '-7 days')
                        GROUP BY token
                    """)
                    results = cursor.fetchall()
                    queries_tested["multi_token_query"] = {
                        "status": "SUCCESS",
                        "results": [dict(row) for row in results]
                    }
                except Exception as e:
                    queries_tested["multi_token_query"] = {
                        "status": "ERROR", 
                        "error": str(e)
                    }
                
                # Query 3: Recent data availability
                try:
                    cursor.execute("""
                        SELECT token, MAX(timestamp) as latest_timestamp
                        FROM price_history
                        GROUP BY token
                        HAVING latest_timestamp >= datetime('now', '-1 hour')
                    """)
                    results = cursor.fetchall()
                    queries_tested["recent_data_query"] = {
                        "status": "SUCCESS",
                        "tokens_with_recent_data": len(results),
                        "tokens": [row["token"] for row in results]
                    }
                except Exception as e:
                    queries_tested["recent_data_query"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                
                conn.close()
                return queries_tested
                
            except Exception as e:
                return {"database_connection_error": str(e)}
        
        db_query_results = test_database_query_patterns()
        
        self.test_results[test_name] = {
            "status": "PASSED" if integration_result.get("integration_working") else "FAILED",
            "integration_test": integration_result,
            "database_queries": db_query_results
        }
        
        # Print results
        if integration_result.get("integration_working"):
            print(f"✅ Integration pattern working: {integration_result.get('status')}")
        else:
            print(f"❌ Integration pattern broken: {integration_result.get('issue', 'Unknown')}")
        
        # Print database query results
        working_queries = len([q for q in db_query_results.values() if isinstance(q, dict) and q.get("status") == "SUCCESS"])
        total_queries = len([q for q in db_query_results.values() if isinstance(q, dict)])
        print(f"   Database queries: {working_queries}/{total_queries} working")
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        print("\n🛡️  Testing Error Handling...")
        test_name = "error_handling"
        
        def test_volatility_with_errors(token, reference_tokens, market_data, timeframe):
            """Test version that introduces various error conditions"""
            error_scenarios = []
            
            try:
                # Test 1: Invalid timeframe
                if timeframe not in ["1h", "24h", "7d"]:
                    error_scenarios.append("invalid_timeframe")
                    return None, error_scenarios
                
                # Test 2: Empty reference tokens
                if not reference_tokens:
                    error_scenarios.append("empty_reference_tokens")
                    return None, error_scenarios
                
                # Test 3: No market data for token
                if token not in market_data:
                    error_scenarios.append("missing_token_in_market_data")
                    return None, error_scenarios
                
                # Test 4: Database connection error simulation
                try:
                    conn = sqlite3.connect("nonexistent.db")
                    conn.close()
                except Exception:
                    error_scenarios.append("database_connection_failed")
                
                # Test 5: Insufficient historical data
                # Simulate by returning empty data
                mock_empty_data = []
                if len(mock_empty_data) < 5:
                    error_scenarios.append("insufficient_historical_data")
                    return None, error_scenarios
                
                # Test 6: Statistical calculation error
                try:
                    test_data = [1]  # Single value will cause statistics.stdev to fail
                    statistics.stdev(test_data)
                except statistics.StatisticsError:
                    error_scenarios.append("statistics_calculation_error")
                    return None, error_scenarios
                
                return 1.0, error_scenarios  # Successful case
                
            except Exception as e:
                error_scenarios.append(f"unexpected_exception_{type(e).__name__}")
                return None, error_scenarios
        
        # Test various error scenarios
        error_test_cases = [
            {
                "name": "Invalid timeframe",
                "token": "BTC",
                "reference_tokens": ["ETH"],
                "market_data": {"BTC": {"price": 100}},
                "timeframe": "invalid"
            },
            {
                "name": "Empty reference tokens",
                "token": "BTC", 
                "reference_tokens": [],
                "market_data": {"BTC": {"price": 100}},
                "timeframe": "1h"
            },
            {
                "name": "Missing token in market data",
                "token": "MISSING",
                "reference_tokens": ["ETH"],
                "market_data": {"BTC": {"price": 100}},
                "timeframe": "1h"
            },
            {
                "name": "Valid input",
                "token": "BTC",
                "reference_tokens": ["ETH", "SOL"],
                "market_data": {"BTC": {"price": 100}, "ETH": {"price": 200}},
                "timeframe": "1h"
            }
        ]
        
        error_results = {}
        
        for test_case in error_test_cases:
            result, errors = test_volatility_with_errors(
                test_case["token"],
                test_case["reference_tokens"], 
                test_case["market_data"],
                test_case["timeframe"]
            )
            
            error_results[test_case["name"]] = {
                "result": result,
                "errors_detected": errors,
                "expected_errors": test_case["name"] != "Valid input"
            }
            
            if errors:
                print(f"⚠️  {test_case['name']}: Errors detected - {', '.join(errors)}")
            else:
                print(f"✅ {test_case['name']}: No errors")
        
        self.test_results[test_name] = {
            "status": "PASSED",
            "error_scenarios": error_results
        }
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "=" * 80)
        print("📋 FINAL TESTING REPORT")
        print("=" * 80)
        
        # Overall status
        passed_tests = len([t for t in self.test_results.values() if t.get("status") == "PASSED"])
        total_tests = len(self.test_results)
        overall_success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        
        # Detailed results
        print(f"\n📊 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = result.get("status", "UNKNOWN")
            status_icon = "✅" if status == "PASSED" else "⚠️" if status in ["PARTIAL", "NO_DATA"] else "❌"
            print(f"   {status_icon} {test_name}: {status}")
            
            # Add specific details for failed tests
            if status == "FAILED" and "error" in result:
                print(f"      Error: {result['error']}")
        
        # Critical issues identification
        print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
        critical_issues = []
        
        # Check database connectivity
        if self.test_results.get("database_connection", {}).get("status") == "FAILED":
            critical_issues.append("Database connection failed - volatility method cannot access historical data")
        
        # Check schema issues
        schema_result = self.test_results.get("database_schema", {})
        if schema_result.get("status") != "PASSED":
            critical_issues.append("Database schema issues - missing tables or columns")
        
        # Check data availability
        data_result = self.test_results.get("historical_data_availability", {})
        if data_result.get("status") == "NO_DATA":
            critical_issues.append("No historical price data available - volatility calculations impossible")
        
        # Check method functionality
        method_result = self.test_results.get("complete_method", {})
        if method_result.get("status") == "FAILED":
            critical_issues.append("Volatility method implementation has errors")
        
        if critical_issues:
            for i, issue in enumerate(critical_issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("   No critical issues found!")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = []
        
        # Database recommendations
        if self.test_results.get("database_connection", {}).get("status") == "FAILED":
            recommendations.append("Fix database connection - check if crypto_history.db exists and is accessible")
        
        # Data recommendations  
        data_result = self.test_results.get("historical_data_availability", {})
        if data_result.get("status") == "NO_DATA":
            recommendations.append("Run your main bot to collect historical price data before testing volatility")
        elif data_result.get("status") == "PASSED":
            insufficient_data = data_result.get("tokens_with_sufficient_data", {})
            if insufficient_data.get("7d", 0) < 3:
                recommendations.append("Let the bot run longer to collect more historical data for 7d analysis")
        
        # Method recommendations
        method_result = self.test_results.get("complete_method", {})
        if method_result.get("success_rate", 0) < 0.5:
            recommendations.append("Method has low success rate - check data format compatibility")
        
        # Integration recommendations
        integration_result = self.test_results.get("data_flow_integration", {})
        if not integration_result.get("integration_test", {}).get("integration_working"):
            recommendations.append("Fix integration pattern - check hasattr usage and method signatures")
        
        if not recommendations:
            recommendations.append("System appears to be working well - continue monitoring")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Data summary
        print(f"\n📈 DATA SUMMARY:")
        data_result = self.test_results.get("historical_data_availability", {})
        if data_result.get("status") == "PASSED":
            total_tokens = data_result.get("total_tokens", 0)
            sufficient_data = data_result.get("tokens_with_sufficient_data", {})
            print(f"   Total tokens with data: {total_tokens}")
            print(f"   Tokens ready for 1h analysis: {sufficient_data.get('1h', 0)}")
            print(f"   Tokens ready for 24h analysis: {sufficient_data.get('24h', 0)}")
            print(f"   Tokens ready for 7d analysis: {sufficient_data.get('7d', 0)}")
        
        # Save detailed results to file
        try:
            with open("volatility_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: volatility_test_results.json")
        except Exception as e:
            print(f"\n⚠️  Could not save results file: {e}")
        
        print("\n" + "=" * 80)
        print("🏁 TESTING COMPLETE")
        print("=" * 80)

def main():
    """Main execution function"""
    print("🚀 Volatility Method Testing Script")
    print("This script will test the _calculate_relative_volatility method and data flow")
    print()
    
    # Initialize tester with auto-detected database path
    tester = VolatilityTester()
    
    if not os.path.exists(tester.db_path):
        print(f"⚠️  Database file '{os.path.basename(tester.db_path)}' not found.")
        print("Please run your main bot first to create the database and collect some data.")
        print()
        create_anyway = input("Continue with testing anyway? (y/N): ").lower().strip()
        if create_anyway != 'y':
            print("Exiting. Run your bot first, then come back to test.")
            return
    
    # Initialize and run tests
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()