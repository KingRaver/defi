#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 TEST _generate_predictions METHOD 🎯
Focused test for the specific method creating synthetic data
"""

import sys
import os
import time
import logging
import io
from database import CryptoDatabase
from llm_provider import LLMProvider
from config import Config

# Add src directory to path
sys.path.append('src')

def test_generate_predictions_method():
    """
    Test the specific _generate_predictions method that's causing the AVAX synthetic data issue
    """
    
    try:
        # Import your prediction engine
        from prediction_engine import PredictionEngine
        
        print("🎯 Testing _generate_predictions Method")
        print("=" * 50)
        
        # Initialize prediction engine
        config = Config()
        database = CryptoDatabase()
        llm_provider = LLMProvider(config=config)
        prediction_engine = PredictionEngine(database=database, llm_provider=llm_provider)
        
        
        # Test scenarios that trigger synthetic data
        test_scenarios = [
            {
                'name': 'Missing AVAX Token',
                'token': 'AVAX',
                'market_data': {
                    'BTC': {'current_price': 43000, 'market_cap': 800000000000},
                    'ETH': {'current_price': 2500, 'market_cap': 300000000000}
                    # NOTE: AVAX is intentionally missing to trigger the issue
                },
                'timeframe': '24h'
            },
            {
                'name': 'Completely Empty Market Data',
                'token': 'SOL',
                'market_data': {},
                'timeframe': '24h'
            },
            {
                'name': 'Valid AVAX Token',
                'token': 'AVAX',
                'market_data': {
                    'AVAX': {
                        'current_price': 37.89,
                        'market_cap': 15000000000,
                        'total_volume': 500000000,
                        'price_change_percentage_24h': -0.67
                    }
                },
                'timeframe': '24h'
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            print("-" * 30)
            
            # Capture logs to detect synthetic data creation
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            
            # Add handler to the specific logger that shows the warning
            logger = logging.getLogger('ETHBTCCorrelation')
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            
            try:
                start_time = time.time()
                
                # Call the specific method that's causing issues
                prediction = prediction_engine._generate_predictions(
                    token=scenario['token'],
                    market_data=scenario['market_data'],
                    timeframe=scenario['timeframe']
                )
                
                execution_time = time.time() - start_time
                
                # Get captured logs
                log_output = log_capture.getvalue()
                
                # Remove handler
                logger.removeHandler(handler)
                
                # Analyze results
                has_synthetic_warning = 'creating synthetic data' in log_output
                has_prediction = prediction is not None
                
                result = {
                    'success': True,
                    'has_prediction': has_prediction,
                    'has_synthetic_warning': has_synthetic_warning,
                    'execution_time': execution_time,
                    'log_output': log_output,
                    'prediction': prediction
                }
                
                # Print results
                print(f"✅ Method executed successfully")
                print(f"📊 Has prediction: {has_prediction}")
                print(f"🚨 Synthetic data warning: {has_synthetic_warning}")
                print(f"⏱️ Execution time: {execution_time:.3f}s")
                
                if has_synthetic_warning:
                    print(f"🔍 Log output: {log_output.strip()}")
                
                if prediction and isinstance(prediction, dict):
                    print(f"💰 Predicted price: {prediction.get('predicted_price', 'N/A')}")
                    print(f"🎯 Confidence: {prediction.get('confidence', 'N/A')}")
                
                results[scenario['name']] = result
                
            except Exception as method_error:
                logger.removeHandler(handler)
                print(f"❌ Method failed: {str(method_error)}")
                results[scenario['name']] = {
                    'success': False,
                    'error': str(method_error)
                }
        
        # Summary
        print(f"\n📋 SUMMARY")
        print("=" * 50)
        
        synthetic_count = sum(1 for r in results.values() 
                            if r.get('has_synthetic_warning', False))
        total_tests = len(results)
        
        print(f"Total tests: {total_tests}")
        print(f"Synthetic data warnings: {synthetic_count}")
        print(f"Synthetic data rate: {synthetic_count/total_tests*100:.1f}%")
        
        if synthetic_count > 0:
            print(f"\n🚨 CRITICAL FINDINGS:")
            print(f"   - {synthetic_count} out of {total_tests} tests triggered synthetic data")
            print(f"   - This means the method creates fake data when tokens are missing")
            print(f"   - This is dangerous for real wealth generation!")
            
            print(f"\n📍 SPECIFIC ISSUES:")
            for name, result in results.items():
                if result.get('has_synthetic_warning'):
                    print(f"   • {name}: Generated synthetic data")
        
        print(f"\n🎯 RECOMMENDATION:")
        if synthetic_count > 0:
            print(f"   ❌ HALT TRADING: Method generates synthetic data")
            print(f"   🔧 FIX REQUIRED: Replace synthetic data with proper failures")
            print(f"   💰 WEALTH RISK: Trading on fake data could lose money")
        else:
            print(f"   ✅ SAFE: No synthetic data generation detected")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure prediction_engine.py is in the src/ directory")
        return None
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_synthetic_data_location():
    """
    Show exactly where the synthetic data is being generated in the code
    """
    print(f"\n🔍 SYNTHETIC DATA SOURCE LOCATION:")
    print("=" * 50)
    print(f"File: prediction_engine.py")
    print(f"Method: _generate_predictions")
    print(f"Lines: ~47-58 (approximately)")
    print(f"")
    print(f"Problem code:")
    print(f"```python")
    print(f"if not token_data:")
    print(f"    logger.logger.warning(f'No market data found for {{token}}, creating synthetic data')")
    print(f"    # Create minimal synthetic data for fallback")
    print(f"    token_data = {{")
    print(f"        'current_price': 1.0,")
    print(f"        'price_change_percentage_24h': 0.0,")
    print(f"        'volume': 1000000.0,")
    print(f"        'market_cap': 1000000000.0,")
    print(f"        'high_24h': 1.02,")
    print(f"        'low_24h': 0.98")
    print(f"    }}")
    print(f"```")
    print(f"")
    print(f"🚨 This code should be REMOVED and replaced with:")
    print(f"```python")
    print(f"if not token_data:")
    print(f"    logger.logger.error(f'No market data found for {{token}} - HALTING PREDICTION')")
    print(f"    return None  # Fail properly instead of creating fake data")
    print(f"```")

if __name__ == "__main__":
    print("🧪 Prediction Engine _generate_predictions Test")
    print("=" * 60)
    
    # Run the focused test
    results = test_generate_predictions_method()
    
    # Show where the synthetic data is coming from
    show_synthetic_data_location()
    
    print(f"\n✅ Test complete!")
    print(f"🎯 Focus: Fix the synthetic data generation in _generate_predictions method")