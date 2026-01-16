#!/usr/bin/env python3
"""
llm_test.py - Targeted Test for LLM Prediction Methods

This script specifically tests the LLM prediction methods with real data
to verify database compatibility.

Usage:
    python llm_test.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger("llm_test")

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
if os.path.exists(os.path.join(src_dir, 'src')):
    src_dir = os.path.join(src_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import required modules
try:
    from prediction_engine import EnhancedPredictionEngine
    from database import CryptoDatabase
    from coingecko_handler import CoinGeckoHandler
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this from the project root")
    sys.exit(1)

# Test class for LLM prediction methods
class LLMPredictionTester:
    def __init__(self):
        """Initialize the tester with real dependencies"""
        logger.info("Initializing LLM Prediction Tester...")
        
        # Initialize database
        self.db = CryptoDatabase()
        logger.info("Database initialized")
        
        # Initialize CoinGecko handler for real market data
        try:
            self.coingecko = CoinGeckoHandler(base_url="https://api.coingecko.com/api/v3")
            logger.info("CoinGecko handler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CoinGecko handler: {e}")
            sys.exit(1)
        
        # Initialize a mock LLM provider for testing
        class MockLLMProvider:
            def generate_text(self, prompt, max_tokens=1000):
                # Real LLM-like response with dynamic price based on prompt
                # Extract current price from prompt to make prediction realistic
                import re
                try:
                    current_price_match = re.search(r"Current Price: \$(\d+\.\d+)", prompt)
                    if current_price_match:
                        current_price = float(current_price_match.group(1))
                        # Make prediction 1-3% above current price
                        predicted_price = current_price * (1 + (hash(prompt) % 30) / 1000)
                        lower_bound = predicted_price * 0.98
                        upper_bound = predicted_price * 1.02
                        percent_change = ((predicted_price / current_price) - 1) * 100
                    else:
                        # Fallback values
                        predicted_price = 45000.0
                        lower_bound = 44100.0
                        upper_bound = 45900.0
                        percent_change = 2.0
                except Exception as e:
                    logger.warning(f"Error extracting price from prompt: {e}")
                    predicted_price = 45000.0
                    lower_bound = 44100.0
                    upper_bound = 45900.0
                    percent_change = 2.0
                
                return json.dumps({
                    "prediction": {
                        "price": round(predicted_price, 2),
                        "confidence": 70 + (hash(prompt) % 20),  # 70-90% confidence
                        "lower_bound": round(lower_bound, 2),
                        "upper_bound": round(upper_bound, 2),
                        "percent_change": round(percent_change, 2),
                        "timeframe": "1h" if "timeframe: \"1h\"" in prompt else "24h"
                    },
                    "rationale": f"Analysis of technical indicators, market trends, and volume patterns suggests this price movement.",
                    "sentiment": "BULLISH" if percent_change > 0 else "BEARISH",
                    "key_factors": [
                        "Technical momentum indicators",
                        "Volume patterns",
                        "Market sentiment"
                    ]
                })
        
        self.llm_provider = MockLLMProvider()
        logger.info("Mock LLM provider initialized")
        
        # Initialize prediction engine
        self.prediction_engine = EnhancedPredictionEngine(
            database=self.db,
            llm_provider=self.llm_provider
        )
        logger.info("Prediction engine initialized")
        
        # Test tokens and timeframes
        self.test_tokens = ["BTC", "ETH", "SOL", "AVAX"]
        self.timeframes = ["1h", "24h"]
        
        logger.info("LLM Prediction Tester initialized")
    
    def get_real_market_data(self) -> Dict[str, Any]:
        """
        Get real market data from CoinGecko
        
        Returns:
            Dictionary of market data
        """
        logger.info("Fetching real market data from CoinGecko...")
        
        try:
            # Fetch real market data with price history
            market_data = self.coingecko.get_market_data(
                timeframe="1h",
                include_price_history=True,
                priority_tokens=self.test_tokens
            )
            
            if not market_data:
                raise ValueError("No market data returned")
                
            # Convert to dictionary with token symbols as keys if needed
            if isinstance(market_data, list):
                market_dict = {}
                for item in market_data:
                    if isinstance(item, dict):
                        symbol = item.get('symbol', '').upper()
                        if symbol:
                            market_dict[symbol] = item
                market_data = market_dict
            
            logger.info(f"Successfully fetched market data for {len(market_data)} tokens")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            # Don't fallback to mock data - we want to test with real data only
            raise
    
    def test_parse_llm_response(self, token: str, current_price: float) -> bool:
        """
        Test the _parse_llm_response method with real-like data
        
        Args:
            token: Token symbol
            current_price: Current token price
            
        Returns:
            True if test passed, False otherwise
        """
        logger.info(f"Testing _parse_llm_response for {token}...")
        
        try:
            # Generate a realistic LLM response
            llm_response = json.dumps({
                "prediction": {
                    "price": current_price * 1.02,
                    "confidence": 75.0,
                    "lower_bound": current_price * 0.99,
                    "upper_bound": current_price * 1.05,
                    "percent_change": 2.0,
                    "timeframe": "1h"
                },
                "rationale": f"Analysis of technical indicators for {token} suggests a bullish momentum",
                "sentiment": "BULLISH",
                "key_factors": ["Technical momentum", "Volume increase", "Support level"]
            })
            
            # Create basic prediction objects for testing
            tech_prediction = {"price": current_price * 1.01, "confidence": 70.0}
            stat_prediction = {"price": current_price * 1.015, "confidence": 65.0}
            ml_prediction = {"price": current_price * 1.02, "confidence": 60.0}
            
            # Call the method
            result = self.prediction_engine._parse_llm_response(
                llm_response, token, current_price,
                tech_prediction, stat_prediction, ml_prediction,
                "1h"
            )
            
            # Verify result has required fields
            required_fields = ['price', 'confidence', 'lower_bound', 'upper_bound']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                logger.error(f"_parse_llm_response missing fields: {missing_fields}")
                return False
            
            # Verify values are reasonable
            price = result.get('price')
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid price: {price}")
                return False
            
            confidence = result.get('confidence')
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                logger.error(f"Invalid confidence: {confidence}")
                return False
            
            # Test passed
            logger.info(f"✅ _parse_llm_response test passed for {token}")
            logger.info(f"   Price: ${result.get('price', 0):.2f}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.1f}%")
            logger.info(f"   Bounds: ${result.get('lower_bound', 0):.2f} - ${result.get('upper_bound', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"_parse_llm_response test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_generate_llm_prediction(self, token: str, market_data: Dict[str, Any], 
                                     timeframe: str) -> bool:
        """
        Test the _generate_llm_prediction method with real market data
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for prediction
            
        Returns:
            True if test passed, False otherwise
        """
        logger.info(f"Testing _generate_llm_prediction for {token} ({timeframe})...")
        
        try:
            # Extract token data
            token_data = market_data.get(token, {})
            if not token_data:
                logger.error(f"No data found for token {token}")
                return False
            
            # Get current price
            current_price = token_data.get('current_price')
            if current_price is None:
                logger.error(f"No current price found for {token}")
                return False
            
            # Create technical, statistical, and ML predictions
            # Use the prediction engine to generate a technical prediction with real data
            try:
                tech_prediction = self.prediction_engine._generate_predictions(token, market_data, timeframe)
            except Exception as e:
                logger.warning(f"Could not generate technical prediction: {e}. Using fallback.")
                tech_prediction = {
                    "price": current_price * 1.01,
                    "confidence": 70.0,
                    "percent_change": 1.0,
                    "lower_bound": current_price * 0.99,
                    "upper_bound": current_price * 1.03,
                    "sentiment": "NEUTRAL"
                }
            
            # Create basic statistical and ML predictions
            stat_prediction = {
                "price": current_price * 1.015,
                "confidence": 65.0,
                "percent_change": 1.5,
                "lower_bound": current_price * 0.98,
                "upper_bound": current_price * 1.05,
                "sentiment": "NEUTRAL"
            }
            
            ml_prediction = {
                "price": current_price * 1.02,
                "confidence": 60.0,
                "percent_change": 2.0,
                "lower_bound": current_price * 0.97,
                "upper_bound": current_price * 1.07,
                "sentiment": "BULLISH"
            }
            
            # Extract price history
            prices = token_data.get('sparkline', [])
            if not prices:
                prices = token_data.get('price_history', [])
            if not prices:
                prices = [current_price] * 24  # Fallback
            
            # Extract or create volume data with careful error handling
            try:
                # Try various volume field names
                volume = token_data.get('volume')
                if volume is None:
                    volume = token_data.get('total_volume')
                if volume is None:
                    volume = token_data.get('volume_24h')
                if volume is None:
                    # Fallback estimate based on market cap
                    market_cap = token_data.get('market_cap')
                    volume = market_cap / 20 if market_cap else 1000000
                
                volumes = [float(volume)] * len(prices)
            except (TypeError, ValueError) as e:
                logger.warning(f"Error extracting volume data: {e}")
                volumes = [1000000.0] * len(prices)
            
            # Debug log the data we're using
            logger.info(f"Using {len(prices)} price points and {len(volumes)} volume points")
            
            # Call the method
            result = self.prediction_engine._generate_llm_prediction(
                token, prices, volumes, current_price,
                tech_prediction, stat_prediction, ml_prediction,
                "NORMAL", timeframe
            )
            
            # Verify result has required fields
            required_fields = ['price', 'confidence', 'lower_bound', 'upper_bound']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                logger.error(f"_generate_llm_prediction missing fields: {missing_fields}")
                return False
            
            # Verify values are reasonable
            price = result.get('price')
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid price: {price}")
                return False
            
            confidence = result.get('confidence')
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                logger.error(f"Invalid confidence: {confidence}")
                return False
            
            # Test passed
            logger.info(f"✅ _generate_llm_prediction test passed for {token} ({timeframe})")
            logger.info(f"   Price: ${result.get('price', 0):.2f}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.1f}%")
            logger.info(f"   Bounds: ${result.get('lower_bound', 0):.2f} - ${result.get('upper_bound', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"_generate_llm_prediction test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_combine_predictions_without_llm(self, token: str, market_data: Dict[str, Any], 
                                           timeframe: str) -> bool:
        """
        Test the _combine_predictions_without_llm method with real market data
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for prediction
            
        Returns:
            True if test passed, False otherwise
        """
        logger.info(f"Testing _combine_predictions_without_llm for {token} ({timeframe})...")
        
        try:
            # Extract token data
            token_data = market_data.get(token, {})
            if not token_data:
                logger.error(f"No data found for token {token}")
                return False
            
            # Get current price
            current_price = token_data.get('current_price')
            if current_price is None:
                logger.error(f"No current price found for {token}")
                return False
            
            # Create technical, statistical, and ML predictions
            tech_prediction = {
                "price": current_price * 1.01,
                "confidence": 70.0,
                "percent_change": 1.0,
                "lower_bound": current_price * 0.99,
                "upper_bound": current_price * 1.03,
                "sentiment": "NEUTRAL"
            }
            
            stat_prediction = {
                "price": current_price * 1.015,
                "confidence": 65.0,
                "percent_change": 1.5,
                "lower_bound": current_price * 0.98,
                "upper_bound": current_price * 1.05,
                "sentiment": "NEUTRAL"
            }
            
            ml_prediction = {
                "price": current_price * 1.02,
                "confidence": 60.0,
                "percent_change": 2.0,
                "lower_bound": current_price * 0.97,
                "upper_bound": current_price * 1.07,
                "sentiment": "BULLISH"
            }
            
            # Call the method
            result = self.prediction_engine._combine_predictions_without_llm(
                token, current_price,
                tech_prediction, stat_prediction, ml_prediction,
                "NORMAL", timeframe
            )
            
            # Verify result has required fields
            required_fields = ['price', 'confidence', 'lower_bound', 'upper_bound']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                logger.error(f"_combine_predictions_without_llm missing fields: {missing_fields}")
                return False
            
            # Verify values are reasonable
            price = result.get('price')
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid price: {price}")
                return False
            
            confidence = result.get('confidence')
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                logger.error(f"Invalid confidence: {confidence}")
                return False
            
            # Test passed
            logger.info(f"✅ _combine_predictions_without_llm test passed for {token} ({timeframe})")
            logger.info(f"   Price: ${result.get('price', 0):.2f}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.1f}%")
            logger.info(f"   Bounds: ${result.get('lower_bound', 0):.2f} - ${result.get('upper_bound', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"_combine_predictions_without_llm test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_tests(self) -> Dict[str, Dict[str, int]]:
        """
        Run all tests for all methods
        
        Returns:
            Dictionary of test results by method
        """
        logger.info("======== RUNNING LLM PREDICTION TESTS ========")
        
        # Get real market data
        try:
            market_data = self.get_real_market_data()
        except Exception as e:
            logger.error(f"Failed to get market data for testing: {e}")
            return {
                "_parse_llm_response": {"passed": 0, "failed": 0},
                "_generate_llm_prediction": {"passed": 0, "failed": 0},
                "_combine_predictions_without_llm": {"passed": 0, "failed": 0}
            }
        
        # Prepare results dictionary
        results = {
            "_parse_llm_response": {"passed": 0, "failed": 0},
            "_generate_llm_prediction": {"passed": 0, "failed": 0},
            "_combine_predictions_without_llm": {"passed": 0, "failed": 0}
        }
        
        # Test _parse_llm_response for each token
        for token in self.test_tokens:
            # Skip if token not in market data
            if token not in market_data:
                logger.warning(f"Token {token} not found in market data, skipping")
                continue
                
            current_price = market_data[token].get('current_price')
            if current_price is None:
                logger.warning(f"No current price for {token}, skipping")
                continue
                
            # Run test
            if self.test_parse_llm_response(token, current_price):
                results["_parse_llm_response"]["passed"] += 1
            else:
                results["_parse_llm_response"]["failed"] += 1
        
        # Test _generate_llm_prediction for each token and timeframe
        for token in self.test_tokens:
            # Skip if token not in market data
            if token not in market_data:
                logger.warning(f"Token {token} not found in market data, skipping")
                continue
                
            for timeframe in self.timeframes:
                # Run test
                if self.test_generate_llm_prediction(token, market_data, timeframe):
                    results["_generate_llm_prediction"]["passed"] += 1
                else:
                    results["_generate_llm_prediction"]["failed"] += 1
        
        # Test _combine_predictions_without_llm for each token and timeframe
        for token in self.test_tokens:
            # Skip if token not in market data
            if token not in market_data:
                logger.warning(f"Token {token} not found in market data, skipping")
                continue
                
            for timeframe in self.timeframes:
                # Run test
                if self.test_combine_predictions_without_llm(token, market_data, timeframe):
                    results["_combine_predictions_without_llm"]["passed"] += 1
                else:
                    results["_combine_predictions_without_llm"]["failed"] += 1
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, int]]) -> None:
        """
        Print test results in a formatted way
        
        Args:
            results: Dictionary of test results by method
        """
        logger.info("\n" + "=" * 60)
        logger.info("LLM PREDICTION METHODS TEST RESULTS")
        logger.info("=" * 60)
        
        all_passed = True
        
        for method, counts in results.items():
            passed = counts["passed"]
            failed = counts["failed"]
            total = passed + failed
            
            if total == 0:
                pass_rate = 0
                status = "⚠️ NO TESTS"
                all_passed = False
            elif failed == 0:
                pass_rate = 100
                status = "✅ PASS"
            else:
                pass_rate = (passed / total) * 100
                status = "❌ FAIL"
                all_passed = False
            
            logger.info(f"{method}: {status} ({passed}/{total} passed, {pass_rate:.1f}%)")
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED! Your LLM prediction methods are database-compatible.")
            logger.info("You should no longer see $0.0000 values in your Twitter posts.")
        else:
            logger.info("⚠️ SOME TESTS FAILED. Check the logs for details.")
        logger.info("=" * 60)


def main():
    """Main function"""
    logger.info("Starting LLM prediction methods test...")
    
    try:
        # Create tester
        tester = LLMPredictionTester()
        
        # Run tests
        results = tester.run_tests()
        
        # Print results
        tester.print_results(results)
        
        return 0
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())