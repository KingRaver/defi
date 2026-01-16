#!/usr/bin/env python3
"""
test_prediction_compatibility.py - Dynamic Prediction Method Compatibility Tester

This script dynamically tests your prediction methods with real data to verify
that they produce database-compatible output structures with all required fields.

Author: Compatibility Testing Tool for Generational Wealth System
Created: 2025-07-19
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prediction_tester")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
try:
    from src.prediction_engine import EnhancedPredictionEngine
    from src.database import CryptoDatabase
    from src.coingecko_handler import CoinGeckoHandler
    
    # Mock LLM provider class
    class MockLLMProvider:
        def generate_text(self, prompt, max_tokens=1000):
            # Return a structured JSON response similar to what Claude would return
            return """
            {
              "prediction": {
                "price": 45500.00,
                "confidence": 75.0,
                "lower_bound": 44200.00,
                "upper_bound": 46800.00,
                "percent_change": 1.8,
                "timeframe": "1h"
              },
              "rationale": "Based on technical indicators and recent market movements, Bitcoin shows bullish momentum with strong buying pressure. The MACD is showing a positive crossover and volume is increasing, indicating potential short-term growth.",
              "sentiment": "BULLISH",
              "key_factors": ["Positive MACD crossover", "Increasing volume", "Support level holding"]
            }
            """
    
    logger.info("✅ Successfully imported project modules")
except ImportError as e:
    logger.error(f"❌ Failed to import project modules: {e}")
    logger.error("Make sure you're running this script from the project root")
    sys.exit(1)


@dataclass
class MethodTestResult:
    """Result of testing a prediction method"""
    method_name: str
    success: bool
    has_price: bool
    has_confidence: bool
    has_lower_bound: bool
    has_upper_bound: bool
    output_fields: List[str]
    missing_fields: List[str]
    execution_time: float
    error: Optional[str] = None


class PredictionMethodTester:
    """
    🧪 DYNAMIC PREDICTION METHOD COMPATIBILITY TESTER 🧪
    
    Tests your prediction methods with real data to verify database compatibility.
    """
    
    def __init__(self):
        """Initialize the tester with required dependencies"""
        logger.info("🧪 Initializing Prediction Method Tester")
        
        # Initialize database
        self.db = CryptoDatabase()
        logger.info("✅ Database initialized")
        
        # Mock LLM provider
        self.llm_provider = MockLLMProvider()
        logger.info("✅ Mock LLM provider initialized")
        
        # Initialize prediction engine
        self.prediction_engine = EnhancedPredictionEngine(
            database=self.db,
            llm_provider=self.llm_provider
        )
        logger.info("✅ Prediction engine initialized")
        
        # Initialize CoinGecko handler for market data
        self.coingecko = CoinGeckoHandler(base_url="https://api.coingecko.com/api/v3")
        logger.info("✅ CoinGecko handler initialized")
        
        # Required fields for database compatibility
        self.required_fields = ['price', 'confidence', 'lower_bound', 'upper_bound']
        
        # Test tokens
        self.test_tokens = ['BTC', 'ETH', 'SOL', 'AVAX']
        
        # Test timeframes
        self.timeframes = ['1h', '24h', '7d']
        
        logger.info("🧪 Tester initialized and ready")
    
    def get_test_market_data(self) -> Dict[str, Any]:
        """
        Get market data for testing from CoinGecko or fallback to mock data
        """
        try:
            # Try to get real market data
            logger.info("📈 Fetching real market data from CoinGecko")
            market_data = self.coingecko.get_market_data(
                timeframe="1h",
                include_price_history=True
            )
            
            if market_data and len(market_data) > 0:
                logger.info(f"✅ Successfully fetched market data for {len(market_data)} tokens")
                
                # Ensure dictionary format with symbol as key
                if isinstance(market_data, list):
                    market_dict = {}
                    for item in market_data:
                        symbol = item.get('symbol', '').upper()
                        if symbol:
                            market_dict[symbol] = item
                    market_data = market_dict
                
                return market_data
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch market data: {e}")
        
        # Create mock market data as fallback
        logger.info("📊 Creating mock market data")
        
        mock_data = {
            "BTC": {
                "id": "bitcoin",
                "symbol": "BTC",
                "current_price": 45000.0,
                "market_cap": 850_000_000_000,
                "volume": 25_000_000_000,
                "price_change_percentage_24h": 2.5,
                "sparkline": [44000, 44200, 44500, 45000, 44800, 44900, 45100, 45000, 44900, 
                            45200, 45300, 45500, 45400, 45600, 45700, 45800, 45600, 45500, 
                            45300, 45200, 45000, 44800, 44900, 45000]
            },
            "ETH": {
                "id": "ethereum",
                "symbol": "ETH",
                "current_price": 3200.0,
                "market_cap": 400_000_000_000,
                "volume": 15_000_000_000,
                "price_change_percentage_24h": 1.8,
                "sparkline": [3150, 3180, 3190, 3200, 3220, 3210, 3230, 3250, 3240, 
                            3260, 3280, 3270, 3290, 3300, 3290, 3310, 3300, 3320,
                            3310, 3300, 3290, 3280, 3290, 3200]
            },
            "SOL": {
                "id": "solana",
                "symbol": "SOL",
                "current_price": 180.0,
                "market_cap": 80_000_000_000,
                "volume": 5_000_000_000,
                "price_change_percentage_24h": 3.2,
                "sparkline": [175, 176, 177, 179, 178, 180, 182, 183, 182, 
                            184, 185, 184, 186, 187, 188, 186, 185, 184,
                            182, 180, 178, 179, 180, 180]
            },
            "AVAX": {
                "id": "avalanche-2",
                "symbol": "AVAX",
                "current_price": 42.0,
                "market_cap": 15_000_000_000,
                "volume": 1_000_000_000,
                "price_change_percentage_24h": -1.2,
                "sparkline": [42.5, 42.3, 42.1, 41.9, 41.8, 41.7, 41.6, 41.8, 42.0, 
                            41.9, 41.8, 41.7, 41.6, 41.5, 41.4, 41.5, 41.6, 41.7,
                            41.8, 41.9, 42.0, 42.1, 42.0, 42.0]
            }
        }
        
        return mock_data
    
    def test_prediction_method(self, method_name: str, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h", **kwargs) -> MethodTestResult:
        """
        Test a specific prediction method with real data
        
        Args:
            method_name: Name of the method to test
            token: Token symbol to test with
            market_data: Market data dictionary
            timeframe: Timeframe for prediction
            **kwargs: Additional arguments for the method
            
        Returns:
            Test result with compatibility analysis
        """
        logger.info(f"🧪 Testing method: {method_name} for {token} ({timeframe})")
        
        start_time = time.time()
        result = None
        error = None
        
        try:
            # Get the method from the prediction engine
            method = getattr(self.prediction_engine, method_name)
            
            # Call the method with the appropriate parameters
            if method_name == "_generate_predictions":
                result = method(token, market_data, timeframe)
            elif method_name == "_generate_llm_prediction":
                # Create technical, statistical, and ML predictions first
                tech_prediction = self.prediction_engine._generate_predictions(token, market_data, timeframe)
                stat_prediction = {
                    "price": market_data[token]["current_price"] * 1.01,
                    "confidence": 70.0,
                    "percent_change": 1.0,
                    "lower_bound": market_data[token]["current_price"] * 0.99,
                    "upper_bound": market_data[token]["current_price"] * 1.03,
                    "sentiment": "NEUTRAL"
                }
                ml_prediction = {
                    "price": market_data[token]["current_price"] * 1.02,
                    "confidence": 65.0,
                    "percent_change": 2.0,
                    "lower_bound": market_data[token]["current_price"] * 0.98,
                    "upper_bound": market_data[token]["current_price"] * 1.04,
                    "sentiment": "BULLISH"
                }
                
                # Get prices and volumes for LLM
                prices = market_data[token].get('sparkline', [market_data[token]["current_price"]])
                volumes = [market_data[token]["volume"]] * len(prices)
                
                result = method(
                    token, prices, volumes, market_data[token]["current_price"],
                    tech_prediction, stat_prediction, ml_prediction,
                    "NORMAL", timeframe
                )
            elif method_name == "_parse_llm_response":
                llm_response = """
                {
                  "prediction": {
                    "price": 45500.00,
                    "confidence": 75.0,
                    "lower_bound": 44200.00,
                    "upper_bound": 46800.00,
                    "percent_change": 1.8,
                    "timeframe": "1h"
                  },
                  "rationale": "Technical analysis shows bullish momentum",
                  "sentiment": "BULLISH",
                  "key_factors": ["Positive MACD", "Increasing volume", "Support level holding"]
                }
                """
                
                tech_prediction = {"price": market_data[token]["current_price"] * 1.01, "confidence": 70.0}
                stat_prediction = {"price": market_data[token]["current_price"] * 1.01, "confidence": 70.0}
                ml_prediction = {"price": market_data[token]["current_price"] * 1.02, "confidence": 65.0}
                
                result = method(
                    llm_response, token, market_data[token]["current_price"],
                    tech_prediction, stat_prediction, ml_prediction, timeframe
                )
            elif method_name == "_combine_predictions_without_llm":
                tech_prediction = {"price": market_data[token]["current_price"] * 1.01, "confidence": 70.0}
                stat_prediction = {"price": market_data[token]["current_price"] * 1.01, "confidence": 70.0}
                ml_prediction = {"price": market_data[token]["current_price"] * 1.02, "confidence": 65.0}
                
                result = method(
                    token, market_data[token]["current_price"],
                    tech_prediction, stat_prediction, ml_prediction,
                    "NORMAL", timeframe
                )
            else:
                logger.warning(f"⚠️ Unknown method: {method_name}")
                result = None
                error = f"Unknown method: {method_name}"
            
        except Exception as e:
            logger.error(f"❌ Method execution failed: {e}")
            result = None
            error = str(e)
        
        execution_time = time.time() - start_time
        
        # Analyze the result for database compatibility
        if result is not None:
            has_price = 'price' in result and result['price'] is not None
            has_confidence = 'confidence' in result and result['confidence'] is not None
            has_lower_bound = 'lower_bound' in result and result['lower_bound'] is not None
            has_upper_bound = 'upper_bound' in result and result['upper_bound'] is not None
            
            output_fields = list(result.keys())
            missing_fields = [field for field in self.required_fields if field not in result]
            
            success = all([has_price, has_confidence, has_lower_bound, has_upper_bound])
            
            test_result = MethodTestResult(
                method_name=method_name,
                success=success,
                has_price=has_price,
                has_confidence=has_confidence,
                has_lower_bound=has_lower_bound,
                has_upper_bound=has_upper_bound,
                output_fields=output_fields,
                missing_fields=missing_fields,
                execution_time=execution_time,
                error=error
            )
            
            # Log result
            if success:
                logger.info(f"✅ Method {method_name} passed compatibility test")
                logger.info(f"   Price: ${result.get('price', 0):.4f}")
                logger.info(f"   Confidence: {result.get('confidence', 0):.1f}%")
                logger.info(f"   Bounds: ${result.get('lower_bound', 0):.4f} - ${result.get('upper_bound', 0):.4f}")
            else:
                logger.warning(f"❌ Method {method_name} failed compatibility test")
                logger.warning(f"   Missing fields: {missing_fields}")
            
            return test_result
        else:
            # Method execution failed
            return MethodTestResult(
                method_name=method_name,
                success=False,
                has_price=False,
                has_confidence=False,
                has_lower_bound=False,
                has_upper_bound=False,
                output_fields=[],
                missing_fields=self.required_fields,
                execution_time=execution_time,
                error=error
            )
    
    def run_comprehensive_tests(self) -> Dict[str, List[MethodTestResult]]:
        """
        Run comprehensive tests on all prediction methods
        
        Returns:
            Dictionary of test results by method
        """
        logger.info("\n" + "=" * 50)
        logger.info("🧪 RUNNING COMPREHENSIVE COMPATIBILITY TESTS")
        logger.info("=" * 50)
        
        # Get market data for testing
        market_data = self.get_test_market_data()
        
        # Methods to test
        methods_to_test = [
            "_generate_predictions",
            "_generate_llm_prediction",
            "_parse_llm_response",
            "_combine_predictions_without_llm"
        ]
        
        # Run tests for each method with each token and timeframe
        all_results = {}
        
        for method in methods_to_test:
            method_results = []
            
            # Test with at least one token and timeframe
            token = self.test_tokens[0]
            timeframe = self.timeframes[0]
            
            result = self.test_prediction_method(method, token, market_data, timeframe)
            method_results.append(result)
            
            # If method has issues, test with more combinations
            if not result.success:
                for token in self.test_tokens[1:2]:  # Test with one more token
                    for timeframe in self.timeframes[1:2]:  # Test with one more timeframe
                        result = self.test_prediction_method(method, token, market_data, timeframe)
                        method_results.append(result)
            
            all_results[method] = method_results
        
        return all_results
    
    def print_test_summary(self, results: Dict[str, List[MethodTestResult]]):
        """
        Print a summary of test results
        
        Args:
            results: Dictionary of test results by method
        """
        print("\n" + "=" * 80)
        print("🧪 PREDICTION METHOD COMPATIBILITY TEST RESULTS")
        print("=" * 80)
        
        all_success = True
        
        for method, method_results in results.items():
            success_count = sum(1 for r in method_results if r.success)
            total_count = len(method_results)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            
            if success_count == total_count:
                status = "✅ PASS"
            elif success_count > 0:
                status = "⚠️ PARTIAL"
                all_success = False
            else:
                status = "❌ FAIL"
                all_success = False
            
            print(f"\n{method}: {status} ({success_count}/{total_count}, {success_rate:.1f}%)")
            
            for i, result in enumerate(method_results):
                if result.success:
                    print(f"  ✅ Test {i+1}: All required fields present")
                    print(f"     Price: {result.has_price}, Confidence: {result.has_confidence}, Bounds: {result.has_lower_bound and result.has_upper_bound}")
                else:
                    print(f"  ❌ Test {i+1}: Missing fields: {result.missing_fields}")
                    if result.error:
                        print(f"     Error: {result.error}")
            
            # Show field coverage
            if method_results:
                first_result = method_results[0]
                print(f"  📊 Fields produced: {len(first_result.output_fields)}")
                
                # Show sample of fields
                if first_result.output_fields:
                    sample_fields = first_result.output_fields[:8]
                    print(f"     Sample fields: {', '.join(sample_fields)}")
        
        print("\n" + "=" * 80)
        if all_success:
            print("🎉 ALL METHODS PASSED COMPATIBILITY TESTS!")
            print("Your prediction methods now produce database-compatible output.")
        else:
            print("⚠️ SOME METHODS STILL HAVE COMPATIBILITY ISSUES")
            print("Review the results and fix the remaining issues.")
        print("=" * 80)


def main():
    """Main function"""
    print("\n🧪 STARTING DYNAMIC PREDICTION METHOD COMPATIBILITY TESTS")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = PredictionMethodTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_tests()
        
        # Print summary
        tester.print_test_summary(results)
        
        print("\n✅ TESTING COMPLETE")
        return results
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Run the tests
    main()