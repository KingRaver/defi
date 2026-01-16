#!/usr/bin/env python3
"""
🔍 COINGECKO SPARKLINE DATA DIAGNOSTIC SCRIPT 🔍

This script analyzes your CoinGecko API calls and database storage
to understand why sparkline data extraction is failing.

It will answer all questions about:
1. CoinGecko API response structure
2. Database storage format
3. Data extraction issues
4. Validation failures

Usage:
    python sparkline_diagnostic.py

Requirements:
    - Your existing coingecko_handler.py
    - Your existing database.py
    - Access to your database
    - CoinGecko API access
"""

import sys
import os
import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

# Add the src directory to the path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from coingecko_handler import CoinGeckoHandler
    from database import CryptoDatabase
    from utils.logger import logger
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this script from your project root directory")
    print("and that src/coingecko_handler.py and src/database.py exist")
    sys.exit(1)


class SparklineDiagnostic:
    """Comprehensive diagnostic tool for sparkline data issues"""
    
    def __init__(self):
        """Initialize diagnostic components"""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'api_tests': {},
            'database_tests': {},
            'data_flow_tests': {},
            'extraction_tests': {},
            'recommendations': []
        }
        
        # Initialize handlers
        try:
            self.coingecko = CoinGeckoHandler(
                base_url="https://api.coingecko.com/api/v3",
                cache_duration=60  # Short cache for testing
            )
            self.database = CryptoDatabase()
            print("✅ Successfully initialized CoinGecko handler and database")
        except Exception as e:
            print(f"❌ Failed to initialize handlers: {e}")
            sys.exit(1)
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite"""
        print("🔍 STARTING COMPREHENSIVE SPARKLINE DIAGNOSTIC")
        print("=" * 60)
        
        # Test 1: API Response Structure
        print("\n📡 TESTING API RESPONSE STRUCTURE")
        self._test_api_response_structure()
        
        # Test 2: Database Storage Analysis
        print("\n💾 ANALYZING DATABASE STORAGE")
        self._analyze_database_storage()
        
        # Test 3: Data Flow Analysis
        print("\n🔄 TESTING DATA FLOW")
        self._test_data_flow()
        
        # Test 4: Extraction Logic Testing
        print("\n🧪 TESTING EXTRACTION LOGIC")
        self._test_extraction_logic()
        
        # Test 5: Validation Analysis
        print("\n✅ ANALYZING VALIDATION LOGIC")
        self._analyze_validation_logic()
        
        # Generate recommendations
        print("\n💡 GENERATING RECOMMENDATIONS")
        self._generate_recommendations()
        
        # Save results
        self._save_results()
        
        print("\n🎉 DIAGNOSTIC COMPLETE!")
        print("=" * 60)
        return self.results
    
    def _test_api_response_structure(self):
        """Test actual CoinGecko API responses"""
        test_tokens = ['bitcoin', 'ethereum', 'ripple']  # Using full names for API
        
        for token in test_tokens:
            print(f"  Testing {token}...")
            
            try:
                # Test markets endpoint (your main data source)
                url = f"{self.coingecko.base_url}/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'ids': token,
                    'sparkline': 'true',
                    'include_24hr_change': 'true'
                }
                
                # Simulate your actual API call
                import requests
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        token_data = data[0]
                        
                        # Analyze response structure
                        api_result = {
                            'status': 'success',
                            'available_fields': list(token_data.keys()),
                            'has_sparkline_field': 'sparkline_in_7d' in token_data,
                            'sparkline_structure': None,
                            'sparkline_data_type': None,
                            'sparkline_length': 0,
                            'sample_sparkline_values': []
                        }
                        
                        # Analyze sparkline specifically
                        if 'sparkline_in_7d' in token_data:
                            sparkline = token_data['sparkline_in_7d']
                            api_result['sparkline_structure'] = type(sparkline).__name__
                            
                            if isinstance(sparkline, dict):
                                api_result['sparkline_keys'] = list(sparkline.keys())
                                if 'price' in sparkline:
                                    prices = sparkline['price']
                                    api_result['sparkline_data_type'] = type(prices).__name__
                                    api_result['sparkline_length'] = len(prices) if isinstance(prices, list) else 0
                                    if isinstance(prices, list) and len(prices) > 0:
                                        api_result['sample_sparkline_values'] = prices[:5]  # First 5 values
                            elif isinstance(sparkline, list):
                                api_result['sparkline_data_type'] = 'list'
                                api_result['sparkline_length'] = len(sparkline)
                                api_result['sample_sparkline_values'] = sparkline[:5]
                        
                        self.results['api_tests'][token] = api_result
                        print(f"    ✅ {token}: {api_result['sparkline_length']} price points")
                        
                    else:
                        self.results['api_tests'][token] = {'status': 'no_data', 'error': 'Empty response'}
                        print(f"    ❌ {token}: Empty response")
                        
                else:
                    self.results['api_tests'][token] = {'status': 'api_error', 'code': response.status_code}
                    print(f"    ❌ {token}: API error {response.status_code}")
                    
            except Exception as e:
                self.results['api_tests'][token] = {'status': 'exception', 'error': str(e)}
                print(f"    ❌ {token}: Exception - {e}")
                
            # Rate limiting
            time.sleep(1)
    
    def _analyze_database_storage(self):
        """Analyze how data is stored in database"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Check database schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.results['database_tests']['tables'] = tables
            print(f"  📋 Found tables: {', '.join(tables)}")
            
            # Analyze price_history table if it exists
            if 'price_history' in tables:
                cursor.execute("PRAGMA table_info(price_history)")
                columns = [row[1] for row in cursor.fetchall()]
                self.results['database_tests']['price_history_columns'] = columns
                print(f"  📋 price_history columns: {', '.join(columns)}")
                
                # Get sample data
                cursor.execute("""
                    SELECT token, timestamp, price, sparkline_data, high_price, low_price, volume
                    FROM price_history 
                    WHERE sparkline_data IS NOT NULL 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                sample_rows = cursor.fetchall()
                if sample_rows:
                    print(f"  📊 Found {len(sample_rows)} recent records with sparkline data")
                    
                    for row in sample_rows:
                        token, timestamp, price, sparkline_data, high, low, volume = row
                        
                        # Analyze sparkline data structure
                        sparkline_analysis = {
                            'token': token,
                            'timestamp': timestamp,
                            'price': price,
                            'sparkline_data_length': len(sparkline_data) if sparkline_data else 0,
                            'sparkline_data_type': type(sparkline_data).__name__,
                            'is_json': False,
                            'parsed_structure': None
                        }
                        
                        # Try to parse as JSON
                        if sparkline_data:
                            try:
                                parsed = json.loads(sparkline_data)
                                sparkline_analysis['is_json'] = True
                                sparkline_analysis['parsed_structure'] = type(parsed).__name__
                                
                                if isinstance(parsed, dict):
                                    sparkline_analysis['json_keys'] = list(parsed.keys())
                                elif isinstance(parsed, list):
                                    sparkline_analysis['json_length'] = len(parsed)
                                    sparkline_analysis['sample_values'] = parsed[:3] if len(parsed) > 0 else []
                                    
                            except json.JSONDecodeError:
                                sparkline_analysis['is_json'] = False
                                sparkline_analysis['raw_preview'] = str(sparkline_data)[:100]
                        
                        if 'sample_sparkline_records' not in self.results['database_tests']:
                            self.results['database_tests']['sample_sparkline_records'] = []
                        self.results['database_tests']['sample_sparkline_records'].append(sparkline_analysis)
                        
                        print(f"    🔍 {token}: {sparkline_analysis['sparkline_data_length']} chars, JSON: {sparkline_analysis['is_json']}")
                
                else:
                    print("  ⚠️  No records with sparkline data found")
                    self.results['database_tests']['sparkline_records_found'] = False
                    
            else:
                print("  ❌ price_history table not found")
                self.results['database_tests']['price_history_exists'] = False
                
            conn.close()
            
        except Exception as e:
            print(f"  ❌ Database analysis failed: {e}")
            self.results['database_tests']['error'] = str(e)
    
    def _test_data_flow(self):
        """Test the complete data flow from API to database"""
        test_token = 'XRP'  # The failing token from your error
        
        print(f"  🔄 Testing complete data flow for {test_token}")
        
        try:
            # Step 1: Get fresh market data using correct method signature
            print(f"    Step 1: Fetching market data for {test_token}")
            
            # Use the correct method signature based on project knowledge
            # The get_market_data method expects params dict, not a list of token IDs
            params = {
                'vs_currency': 'usd',
                'ids': 'ripple',  # Use full CoinGecko ID for XRP
                'sparkline': 'true',
                'include_24hr_change': 'true',
                'price_change_percentage': '1h,24h,7d'
            }
            
            market_data = self.coingecko.get_market_data(
                params=params,
                timeframe="1h",
                priority_tokens=None,
                include_price_history=False
            )
            
            if market_data and isinstance(market_data, list) and len(market_data) > 0:
                # Extract the first (and should be only) token data
                token_data = market_data[0]
                
                # Convert to expected format (symbol as key) for validation method
                converted_data = {
                    test_token: token_data  # Use XRP as key, token_data as value
                }
                
                self.results['data_flow_tests']['step1_api_fetch'] = {
                    'status': 'success',
                    'fields_present': list(token_data.keys()),
                    'has_sparkline': 'sparkline_in_7d' in token_data,
                    'sparkline_structure': type(token_data.get('sparkline_in_7d')).__name__ if 'sparkline_in_7d' in token_data else 'not_found',
                    'current_price': token_data.get('current_price'),
                    'volume': token_data.get('total_volume'),
                    'symbol': token_data.get('symbol', '').upper()
                }
                print(f"    ✅ Step 1: Got market data with {len(token_data)} fields")
                
                # Analyze sparkline structure if present
                if 'sparkline_in_7d' in token_data:
                    sparkline = token_data['sparkline_in_7d']
                    sparkline_info = {
                        'type': type(sparkline).__name__,
                        'content': None
                    }
                    
                    if isinstance(sparkline, dict):
                        sparkline_info['keys'] = list(sparkline.keys())
                        if 'price' in sparkline:
                            prices = sparkline['price']
                            sparkline_info['price_array_length'] = len(prices) if isinstance(prices, list) else 0
                            sparkline_info['sample_prices'] = prices[:5] if isinstance(prices, list) and len(prices) > 0 else []
                    elif isinstance(sparkline, list):
                        sparkline_info['array_length'] = len(sparkline)
                        sparkline_info['sample_values'] = sparkline[:5] if len(sparkline) > 0 else []
                    
                    self.results['data_flow_tests']['step1_sparkline_analysis'] = sparkline_info
                    print(f"    🔍 Sparkline data: {sparkline_info}")
                
                # Step 2: Test database enhancement
                print(f"    Step 2: Testing database enhancement")
                try:
                    # Pass the original list format to database enhancement
                    enhanced_data = self.database.enhance_market_data_with_sparkline(token_data)
                    
                    if enhanced_data and len(enhanced_data) > 0:
                        enhanced_token_data = enhanced_data  # Should be the enhanced version of our token
                        
                        self.results['data_flow_tests']['step2_db_enhancement'] = {
                            'status': 'success',
                            'fields_after_enhancement': list(enhanced_token_data.keys()),
                            'sparkline_points': enhanced_token_data.get('_sparkline_points', 0),
                            'sparkline_source': enhanced_token_data.get('_sparkline_source', 'unknown'),
                            'has_real_sparkline': enhanced_token_data.get('_has_real_sparkline', False),
                            'enhancement_fields_added': [
                                field for field in enhanced_token_data.keys() 
                                if field.startswith('_') and field not in token_data
                            ]
                        }
                        print(f"    ✅ Step 2: Enhanced with {enhanced_token_data.get('_sparkline_points', 0)} sparkline points from {enhanced_token_data.get('_sparkline_source', 'unknown')}")
                        
                        # Step 3: Test validation input preparation
                        print(f"    Step 3: Testing validation input preparation")
                        
                        # Prepare validation input in the exact format expected by _validate_trading_data
                        # The method expects: market_data as Dict[str, Any] where key is token symbol
                        validation_input = {test_token: enhanced_token_data}
                        
                        # Analyze what validation would see
                        validation_analysis = {
                            'data_structure': 'dict_with_token_key',
                            'token_key': test_token,
                            'token_fields': list(enhanced_token_data.keys()),
                            'validation_critical_fields': {
                                'current_price': enhanced_token_data.get('current_price'),
                                'total_volume': enhanced_token_data.get('total_volume'),
                                'market_cap': enhanced_token_data.get('market_cap'),
                                'sparkline_in_7d': 'sparkline_in_7d' in enhanced_token_data,
                                'sparkline': 'sparkline' in enhanced_token_data,
                                'prices': 'prices' in enhanced_token_data
                            },
                            'possible_sparkline_fields': [
                                field for field in enhanced_token_data.keys()
                                if any(keyword in field.lower() for keyword in ['sparkline', 'price', 'history'])
                            ]
                        }
                        
                        # Test the exact extraction logic from your validation method
                        print(f"    🧪 Testing sparkline extraction logic")
                        
                        token_data_for_validation = enhanced_token_data
                        price_history = []
                        
                        # Replicate your exact extraction logic
                        sparkline_data = None
                        possible_sparkline_fields = [
                            'sparkline',
                            'sparkline_in_7d', 
                            'sparkline_7d',
                            'price_history',
                            'prices'
                        ]
                        
                        extraction_attempts = {}
                        
                        for field in possible_sparkline_fields:
                            if field in token_data_for_validation:
                                field_data = token_data_for_validation[field]
                                extraction_attempts[field] = {
                                    'found': True,
                                    'type': type(field_data).__name__,
                                    'content_preview': str(field_data)[:100] if field_data else 'empty'
                                }
                                
                                if field_data and not sparkline_data:
                                    sparkline_data = field_data
                                    extraction_attempts[field]['selected_for_extraction'] = True
                                    break
                            else:
                                extraction_attempts[field] = {'found': False}
                        
                        # Extract price array from sparkline data structure
                        extraction_result = {
                            'sparkline_data_found': sparkline_data is not None,
                            'sparkline_data_type': type(sparkline_data).__name__ if sparkline_data else None,
                            'extraction_attempts': extraction_attempts,
                            'final_extraction_successful': False,
                            'extracted_price_count': 0
                        }
                        
                        if sparkline_data:
                            if isinstance(sparkline_data, dict):
                                # CoinGecko format: {'price': [array of prices]}
                                if 'price' in sparkline_data:
                                    price_history = sparkline_data['price']
                                    extraction_result['extraction_method'] = 'dict_price_field'
                                elif 'prices' in sparkline_data:
                                    price_history = sparkline_data['prices']
                                    extraction_result['extraction_method'] = 'dict_prices_field'
                                else:
                                    extraction_result['extraction_method'] = 'dict_no_price_field'
                                    extraction_result['dict_keys'] = list(sparkline_data.keys())
                            elif isinstance(sparkline_data, list):
                                # Direct array format
                                price_history = sparkline_data
                                extraction_result['extraction_method'] = 'direct_list'
                            else:
                                extraction_result['extraction_method'] = 'unknown_format'
                        
                        # Fallback: try to get from 'prices' field directly
                        if not price_history:
                            direct_prices = token_data_for_validation.get('prices', [])
                            if isinstance(direct_prices, list) and direct_prices:
                                price_history = direct_prices
                                extraction_result['extraction_method'] = 'fallback_prices_field'
                        
                        # Validate extracted price history
                        if price_history and isinstance(price_history, list):
                            extraction_result['final_extraction_successful'] = True
                            extraction_result['extracted_price_count'] = len(price_history)
                            extraction_result['sample_prices'] = price_history[:5] if len(price_history) > 0 else []
                            
                            # Test validation requirements
                            min_required_points = 100  # From your validation method
                            extraction_result['meets_min_requirement'] = len(price_history) >= min_required_points
                            extraction_result['min_required_points'] = min_required_points
                        
                        validation_analysis['extraction_result'] = extraction_result
                        
                        self.results['data_flow_tests']['step3_validation_preparation'] = validation_analysis
                        
                        if extraction_result['final_extraction_successful']:
                            print(f"    ✅ Step 3: Extraction successful - {extraction_result['extracted_price_count']} price points via {extraction_result['extraction_method']}")
                            if extraction_result['meets_min_requirement']:
                                print(f"    ✅ Meets validation requirement of {extraction_result['min_required_points']} points")
                            else:
                                print(f"    ⚠️  Does not meet validation requirement of {extraction_result['min_required_points']} points")
                        else:
                            print(f"    ❌ Step 3: Extraction failed - no valid price history found")
                            print(f"         Available fields: {list(enhanced_token_data.keys())}")
                            print(f"         Extraction attempts: {extraction_attempts}")
                    
                    else:
                        print(f"    ❌ Step 2: Enhancement failed or returned empty")
                        self.results['data_flow_tests']['step2_db_enhancement'] = {
                            'status': 'failed',
                            'error': 'Enhancement returned empty or None'
                        }
                        
                except Exception as e:
                    print(f"    ❌ Step 2: Database enhancement failed: {e}")
                    self.results['data_flow_tests']['step2_db_enhancement'] = {
                        'status': 'exception',
                        'error': str(e)
                    }
                    
            else:
                print(f"    ❌ Step 1: Failed to get market data or data was empty")
                self.results['data_flow_tests']['step1_api_fetch'] = {
                    'status': 'failed',
                    'error': 'No market data returned or empty response',
                    'response_type': type(market_data).__name__ if market_data is not None else 'None',
                    'response_length': len(market_data) if isinstance(market_data, list) else 0
                }
                
        except Exception as e:
            print(f"    ❌ Data flow test failed with exception: {e}")
            self.results['data_flow_tests']['error'] = str(e)
            
            # Add more detailed error information
            import traceback
            self.results['data_flow_tests']['traceback'] = traceback.format_exc()
    
    def _test_extraction_logic(self):
        """Test the sparkline extraction logic with real data"""
        print("  🧪 Testing sparkline extraction with various data formats")
        
        # Test different data structures that might be in your database
        test_cases = [
            {
                'name': 'coingecko_dict_format',
                'data': {'sparkline_in_7d': {'price': [1.0, 1.1, 1.2, 1.3, 1.4]}},
                'expected_extraction': 'sparkline_in_7d.price'
            },
            {
                'name': 'direct_list_format',
                'data': {'sparkline': [1.0, 1.1, 1.2, 1.3, 1.4]},
                'expected_extraction': 'sparkline'
            },
            {
                'name': 'prices_field_format',
                'data': {'prices': [1.0, 1.1, 1.2, 1.3, 1.4]},
                'expected_extraction': 'prices'
            },
            {
                'name': 'nested_prices_format',
                'data': {'sparkline_7d': {'prices': [1.0, 1.1, 1.2, 1.3, 1.4]}},
                'expected_extraction': 'sparkline_7d.prices'
            }
        ]
        
        for test_case in test_cases:
            print(f"    Testing {test_case['name']}...")
            
            # Simulate the extraction logic from your validation method
            token_data = test_case['data']
            price_history = []
            
            # Replicate your extraction logic
            sparkline_data = None
            possible_sparkline_fields = [
                'sparkline',
                'sparkline_in_7d',
                'sparkline_7d',
                'price_history',
                'prices'
            ]
            
            for field in possible_sparkline_fields:
                if field in token_data:
                    sparkline_data = token_data[field]
                    if sparkline_data:
                        break
            
            # Extract price array from sparkline data structure
            if sparkline_data:
                if isinstance(sparkline_data, dict):
                    if 'price' in sparkline_data:
                        price_history = sparkline_data['price']
                    elif 'prices' in sparkline_data:
                        price_history = sparkline_data['prices']
                elif isinstance(sparkline_data, list):
                    price_history = sparkline_data
            
            # Fallback: try to get from 'prices' field directly
            if not price_history:
                direct_prices = token_data.get('prices', [])
                if isinstance(direct_prices, list) and direct_prices:
                    price_history = direct_prices
            
            result = {
                'extraction_successful': len(price_history) > 0,
                'extracted_length': len(price_history),
                'extracted_values': price_history[:3] if price_history else [],
                'sparkline_field_found': sparkline_data is not None,
                'sparkline_data_type': type(sparkline_data).__name__ if sparkline_data else None
            }
            
            self.results['extraction_tests'][test_case['name']] = result
            
            if result['extraction_successful']:
                print(f"      ✅ Extracted {result['extracted_length']} values")
            else:
                print(f"      ❌ Extraction failed")
    
    def _analyze_validation_logic(self):
        """Analyze the validation requirements vs actual data"""
        print("  🔍 Analyzing validation requirements")
        
        # Get the actual requirements from your validation method
        validation_requirements = {
            'min_required_points': 100,
            'min_volume_1h': 100_000,
            'min_volume_24h': 1_000_000,
            'min_volume_7d': 5_000_000,
            'min_market_cap': 10_000_000,
            'data_quality_threshold': 0.95  # 95% of data must be valid
        }
        
        self.results['validation_tests'] = {
            'requirements': validation_requirements,
            'typical_coingecko_sparkline_length': 168,  # CoinGecko typically provides 7 days hourly = 168 points
            'requirements_analysis': {}
        }
        
        # Analyze if requirements are realistic
        if validation_requirements['min_required_points'] <= 168:
            self.results['validation_tests']['requirements_analysis']['min_points'] = 'realistic'
            print(f"    ✅ Min points requirement ({validation_requirements['min_required_points']}) is realistic")
        else:
            self.results['validation_tests']['requirements_analysis']['min_points'] = 'too_high'
            print(f"    ⚠️  Min points requirement ({validation_requirements['min_required_points']}) may be too high")
        
        # Check if we're testing with the right tokens
        low_volume_tokens = ['XRP']  # XRP might have volume issues
        if any(token in str(self.results) for token in low_volume_tokens):
            print(f"    ⚠️  Testing with potentially low-volume tokens")
            self.results['validation_tests']['requirements_analysis']['volume_concern'] = True
    
    def _generate_recommendations(self):
        """Generate specific recommendations based on findings"""
        recommendations = []
        
        # Analyze API test results
        api_tests = self.results.get('api_tests', {})
        successful_api_calls = sum(1 for test in api_tests.values() if test.get('status') == 'success')
        
        if successful_api_calls == 0:
            recommendations.append({
                'priority': 'high',
                'category': 'api',
                'issue': 'All API calls failed',
                'solution': 'Check API key, rate limits, and network connectivity'
            })
        elif successful_api_calls < len(api_tests):
            recommendations.append({
                'priority': 'medium',
                'category': 'api',
                'issue': 'Some API calls failed',
                'solution': 'Review failed tokens and API response handling'
            })
        
        # Analyze sparkline data structure
        for token, test_result in api_tests.items():
            if test_result.get('status') == 'success':
                if test_result.get('sparkline_length', 0) == 0:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'data_structure',
                        'issue': f'{token} has no sparkline data in API response',
                        'solution': 'Verify sparkline=true parameter and check CoinGecko API documentation'
                    })
                elif test_result.get('sparkline_structure') not in ['dict', 'list']:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'data_structure',
                        'issue': f'{token} has unexpected sparkline structure: {test_result.get("sparkline_structure")}',
                        'solution': 'Update extraction logic to handle this data format'
                    })
        
        # Analyze database storage
        db_tests = self.results.get('database_tests', {})
        if not db_tests.get('sparkline_records_found', True):
            recommendations.append({
                'priority': 'high',
                'category': 'database',
                'issue': 'No sparkline data found in database',
                'solution': 'Check if data is being stored correctly and sparkline enhancement is working'
            })
        
        # Analyze extraction logic
        extraction_tests = self.results.get('extraction_tests', {})
        failed_extractions = [name for name, test in extraction_tests.items() if not test.get('extraction_successful')]
        
        if failed_extractions:
            recommendations.append({
                'priority': 'high',
                'category': 'extraction',
                'issue': f'Extraction failed for formats: {", ".join(failed_extractions)}',
                'solution': 'Update extraction logic to handle these data formats'
            })
        
        # Analyze validation requirements
        validation_tests = self.results.get('validation_tests', {})
        if validation_tests.get('requirements_analysis', {}).get('min_points') == 'too_high':
            recommendations.append({
                'priority': 'medium',
                'category': 'validation',
                'issue': 'Minimum points requirement may be too high for CoinGecko data',
                'solution': 'Consider reducing min_required_points to 50-100 for CoinGecko sparkline data'
            })
        
        self.results['recommendations'] = recommendations
        
        # Print recommendations
        for rec in recommendations:
            priority_icon = "🔥" if rec['priority'] == 'high' else "⚠️" if rec['priority'] == 'medium' else "💡"
            print(f"  {priority_icon} [{rec['category'].upper()}] {rec['issue']}")
            print(f"     Solution: {rec['solution']}")
    
    def _save_results(self):
        """Save diagnostic results to file"""
        try:
            filename = f"sparkline_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")


def main():
    """Run the diagnostic"""
    diagnostic = SparklineDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    api_success = sum(1 for test in results.get('api_tests', {}).values() if test.get('status') == 'success')
    total_api = len(results.get('api_tests', {}))
    print(f"🌐 API Tests: {api_success}/{total_api} successful")
    
    has_db_data = results.get('database_tests', {}).get('sparkline_records_found', False)
    print(f"💾 Database: {'Has sparkline data' if has_db_data else 'No sparkline data found'}")
    
    extraction_success = sum(1 for test in results.get('extraction_tests', {}).values() if test.get('extraction_successful'))
    total_extraction = len(results.get('extraction_tests', {}))
    print(f"🧪 Extraction: {extraction_success}/{total_extraction} formats working")
    
    high_priority_recs = sum(1 for rec in results.get('recommendations', []) if rec.get('priority') == 'high')
    print(f"🔥 High Priority Issues: {high_priority_recs}")
    
    if high_priority_recs == 0:
        print("\n🎉 No critical issues found! Your sparkline data should be working.")
    else:
        print(f"\n⚠️  Found {high_priority_recs} critical issues that need attention.")
    
    return results


if __name__ == "__main__":
    main()