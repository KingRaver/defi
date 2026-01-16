#!/usr/bin/env python3
"""
🔍 PREDICTION ENGINE DIAGNOSTIC TOOL
====================================
This tool captures EXACTLY what your prediction_engine.py returns
so we can fix the normalization once and for all.

Run this to see the actual data structure being returned.
"""

import sys
import os
import json
import time
import traceback
from typing import Any, Dict, Optional
from datetime import datetime

# Add your project path if needed
# sys.path.append('/path/to/your/project')

def safe_import_prediction_engine():
    """Safely import your prediction engine"""
    try:
        from prediction_engine import EnhancedPredictionEngine
        print("✅ Successfully imported EnhancedPredictionEngine")
        return EnhancedPredictionEngine
    except ImportError as e:
        print(f"❌ Failed to import EnhancedPredictionEngine: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing prediction engine: {e}")
        return None

def safe_import_database():
    """Safely import your database"""
    try:
        from database import CryptoDatabase
        print("✅ Successfully imported Database")
        return CryptoDatabase
    except ImportError as e:
        print(f"❌ Failed to import Database: {e}")
        return None

def safe_import_llm():
    """Safely import your LLM provider"""
    try:
        # Try different possible imports
        try:
            from llm_provider import LLMProvider
            print("✅ Successfully imported LLMProvider")
            return LLMProvider
        except ImportError:
            try:
                from bot import LLMProvider
                print("✅ Successfully imported LLMProvider from bot")
                return LLMProvider
            except ImportError:
                print("❌ Could not find LLMProvider - will use None")
                return None
    except Exception as e:
        print(f"❌ Unexpected error importing LLM provider: {e}")
        return None

def analyze_prediction_output(prediction_data: Any, token: str) -> Dict[str, Any]:
    """Analyze the prediction output in detail"""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'token': token,
        'data_type': str(type(prediction_data)),
        'is_none': prediction_data is None,
        'analysis_results': {}
    }
    
    if prediction_data is None:
        analysis['analysis_results']['error'] = "Prediction returned None"
        return analysis
    
    # Basic type analysis
    if isinstance(prediction_data, dict):
        analysis['analysis_results'].update({
            'is_dict': True,
            'keys': list(prediction_data.keys()),
            'key_count': len(prediction_data.keys()),
            'nested_structure': {}
        })
        
        # Analyze nested structure
        for key, value in prediction_data.items():
            analysis['analysis_results']['nested_structure'][key] = {
                'type': str(type(value)),
                'is_dict': isinstance(value, dict),
                'is_list': isinstance(value, list),
                'value_preview': str(value)[:100] if not isinstance(value, (dict, list)) else None,
                'nested_keys': list(value.keys()) if isinstance(value, dict) else None,
                'list_length': len(value) if isinstance(value, list) else None
            }
        
        # Look for token specifically
        token_locations = []
        
        def find_token_recursive(data, path="root"):
            if isinstance(data, dict):
                for k, v in data.items():
                    current_path = f"{path}.{k}"
                    if k.lower() == 'token' or k.lower() == 'symbol':
                        token_locations.append({
                            'path': current_path,
                            'value': v,
                            'type': str(type(v))
                        })
                    if isinstance(v, dict):
                        find_token_recursive(v, current_path)
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        for i, item in enumerate(v[:3]):  # Check first 3 items
                            find_token_recursive(item, f"{current_path}[{i}]")
        
        find_token_recursive(prediction_data)
        analysis['analysis_results']['token_locations'] = token_locations
        
        # Look for confidence specifically
        confidence_locations = []
        
        def find_confidence_recursive(data, path="root"):
            if isinstance(data, dict):
                for k, v in data.items():
                    current_path = f"{path}.{k}"
                    if 'confidence' in k.lower():
                        confidence_locations.append({
                            'path': current_path,
                            'key': k,
                            'value': v,
                            'type': str(type(v))
                        })
                    if isinstance(v, dict):
                        find_confidence_recursive(v, current_path)
        
        find_confidence_recursive(prediction_data)
        analysis['analysis_results']['confidence_locations'] = confidence_locations
        
    else:
        analysis['analysis_results'].update({
            'is_dict': False,
            'string_representation': str(prediction_data)[:200],
            'length': len(str(prediction_data)) if hasattr(prediction_data, '__len__') else None
        })
    
    return analysis

def test_prediction_engine():
    """Test your prediction engine and capture outputs"""
    print("🔍 PREDICTION ENGINE DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Import components
    EnhancedPredictionEngine = safe_import_prediction_engine()
    Database = safe_import_database()
    LLMProvider = safe_import_llm()
    
    if not EnhancedPredictionEngine:
        print("❌ Cannot proceed without EnhancedPredictionEngine")
        return
    
    # Initialize components
    try:
        print("\n📊 Initializing components...")
        
        # Initialize database
        if Database:
            database = Database()
            print("✅ Database initialized")
        else:
            database = None
            print("⚠️ Database not available - using None")
        
        # Initialize LLM provider
        if LLMProvider:
            try:
                llm_provider = [LLMProvider]
                print("✅ LLM Provider initialized")
            except Exception as e:
                print(f"⚠️ LLM Provider failed to initialize: {e}")
                llm_provider = None
        else:
            llm_provider = None
            print("⚠️ LLM Provider not available - using None")
        
        # Initialize prediction engine
        print("🔮 Initializing Enhanced Prediction Engine...")
        prediction_engine = EnhancedPredictionEngine(
            database=database,
            llm_provider=llm_provider
        )
        print("✅ Prediction engine initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize prediction engine: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return
    
    # Test tokens
    test_tokens = ["BTC", "ETH", "AVAX", "SOL"]
    results = {}
    
    for token in test_tokens:
        print(f"\n🎯 Testing prediction for {token}...")
        print("-" * 30)
        
        try:
            # Test the actual method calls your trading bot uses
            start_time = time.time()
            
            # Try different method names that might exist
            prediction_methods = [
                '-generate_predictions',
                'generate_trading_prediction', 
                'get_prediction',
                'predict',
                'analyze_token'
            ]
            
            prediction_result = None
            method_used = None
            
            from coingecko_handler import CoinGeckoHandler
            coingecko = CoinGeckoHandler(base_url="https://api.coingecko.com/api/v3/", cache_duration=60)
            market_data = coingecko.get_market_data()

            for method_name in prediction_methods:
                if hasattr(prediction_engine, method_name):
                    try:
                        method = getattr(prediction_engine, method_name)
                        print(f"  🔍 Trying method: {method_name}")
            
                        # Try calling with real market data
                        try:
                            if isinstance(market_data, list):
                                market_data_dict = {}
                                for item in market_data:
                                    if isinstance(item, dict) and 'symbol' in item:
                                        symbol = item['symbol'].upper()
                                        market_data_dict[symbol] = item
                            else:
                                market_data_dict = market_data
                            prediction_result = method(token, market_data, "1h")
                            method_used = method_name
                            print(f"  ✅ Success with {method_name}(token, market_data, timeframe)")
                            break
                        except TypeError:
                            # Try with additional parameters
                            try:
                                prediction_result = method(token, timeframe="1h")
                                method_used = f"{method_name}(token, timeframe)"
                                print(f"  ✅ Success with {method_name}(token, timeframe)")
                                break
                            except:
                                continue
                    except Exception as e:
                        print(f"  ❌ {method_name} failed: {e}")
                        continue
            
            execution_time = time.time() - start_time
            
            if prediction_result is None:
                print(f"  ❌ No valid prediction method found for {token}")
                results[token] = {
                    'success': False,
                    'error': 'No valid prediction method found',
                    'execution_time': execution_time
                }
                continue
            
            print(f"  ✅ Prediction generated in {execution_time:.3f}s using {method_used}")
            
            # Analyze the output
            analysis = analyze_prediction_output(prediction_result, token)
            analysis['method_used'] = method_used
            analysis['execution_time'] = execution_time
            analysis['success'] = True
            
            results[token] = analysis
            
            # Print immediate analysis
            print(f"  📊 Data type: {analysis['data_type']}")
            if analysis['analysis_results'].get('is_dict'):
                print(f"  📋 Keys: {analysis['analysis_results']['keys']}")
                if analysis['analysis_results']['token_locations']:
                    print(f"  🎯 Token found at: {[loc['path'] for loc in analysis['analysis_results']['token_locations']]}")
                else:
                    print("  ⚠️ No token field found in response")
                
                if analysis['analysis_results']['confidence_locations']:
                    print(f"  📈 Confidence found at: {[loc['path'] for loc in analysis['analysis_results']['confidence_locations']]}")
                else:
                    print("  ⚠️ No confidence field found in response")
            
        except Exception as e:
            print(f"  ❌ Error testing {token}: {e}")
            results[token] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"prediction_engine_diagnostic_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
    
    # Print summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    successful_tokens = [token for token, result in results.items() if result.get('success')]
    failed_tokens = [token for token, result in results.items() if not result.get('success')]
    
    print(f"✅ Successful predictions: {len(successful_tokens)} - {successful_tokens}")
    print(f"❌ Failed predictions: {len(failed_tokens)} - {failed_tokens}")
    
    if successful_tokens:
        sample_token = successful_tokens[0]
        sample_result = results[sample_token]
        
        print(f"\n🔍 SAMPLE STRUCTURE ANALYSIS ({sample_token}):")
        print(f"Method used: {sample_result.get('method_used')}")
        print(f"Data type: {sample_result['data_type']}")
        
        if sample_result['analysis_results'].get('is_dict'):
            print(f"Root keys: {sample_result['analysis_results']['keys']}")
            
            # Print the actual structure
            print("\n📁 NESTED STRUCTURE:")
            for key, info in sample_result['analysis_results']['nested_structure'].items():
                print(f"  {key}: {info['type']}")
                if info['nested_keys']:
                    print(f"    └─ {info['nested_keys']}")
                elif info['value_preview']:
                    print(f"    └─ {info['value_preview']}")
        
        print(f"\n🎯 TOKEN LOCATIONS:")
        token_locs = sample_result['analysis_results']['token_locations']
        if token_locs:
            for loc in token_locs:
                print(f"  {loc['path']}: {loc['value']} ({loc['type']})")
        else:
            print("  ❌ NO TOKEN FOUND IN PREDICTION OUTPUT")
            print("  💡 This is why normalization fails!")
        
        print(f"\n📈 CONFIDENCE LOCATIONS:")
        conf_locs = sample_result['analysis_results']['confidence_locations']
        if conf_locs:
            for loc in conf_locs:
                print(f"  {loc['path']}: {loc['value']} ({loc['type']})")
        else:
            print("  ⚠️ No confidence field found")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    if successful_tokens:
        sample_result = results[successful_tokens[0]]
        if not sample_result['analysis_results']['token_locations']:
            print("1. ❗ CRITICAL: Token is not included in prediction output")
            print("   - Modify prediction engine to include token in response")
            print("   - OR pass token separately to normalization function")
        
        if not sample_result['analysis_results']['confidence_locations']:
            print("2. ⚠️ WARNING: No confidence field found")
            print("   - Check prediction engine output format")
        
        method_used = sample_result.get('method_used', 'unknown')
        print(f"3. ✅ Use method: {method_used}")
    else:
        print("1. ❗ CRITICAL: No predictions succeeded")
        print("   - Check prediction engine initialization")
        print("   - Check method signatures")
    
    return results

if __name__ == "__main__":
    try:
        results = test_prediction_engine()
        print(f"\n🎉 Diagnostic complete! Use results to fix normalization.")
    except KeyboardInterrupt:
        print(f"\n🛑 Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n💥 Diagnostic failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")