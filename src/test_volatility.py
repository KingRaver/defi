#!/usr/bin/env python3
"""
test_real_volatility.py - Test Real Bot Volatility Methods

This script imports and tests the actual bot methods with real data
to verify the volatility fix is working in the actual codebase.
"""

import sys
import os
import traceback
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the actual bot and database classes
try:
    from bot import CryptoAnalysisBot
    from database import CryptoDatabase
    from config import config
    from utils.logger import logger
    from utils.browser import browser
    from llm_provider import LLMProvider
    BOT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import bot modules: {e}")
    BOT_AVAILABLE = False

class RealVolatilityTester:
    """
    Test the actual bot volatility methods with real data
    """
    
    def __init__(self):
        if not BOT_AVAILABLE:
            print("❌ Bot modules not available, cannot run real tests")
            sys.exit(1)
        
        self.db = None
        self.bot = None
        self.setup_bot()
    
    def setup_bot(self):
        """Set up the actual bot and database"""
        try:
            print("🔧 Setting up real bot instance...")
            
            # Import directly in this scope to satisfy Pylance
            from database import CryptoDatabase
            from llm_provider import LLMProvider
            from bot import CryptoAnalysisBot
            from utils.browser import browser
            from config import config

            # Initialize database with real connection
            self.db = CryptoDatabase()
            print(f"   ✅ Database initialized: {self.db.db_path}")
            
            # Initialize real LLM provider
            llm_provider = LLMProvider(config)
            
            # Create bot instance with all real components
            self.bot = CryptoAnalysisBot(
                database=self.db,
                llm_provider=llm_provider,
                config=config
            )
            
            # Use the real browser
            self.bot.browser = browser
            
            print(f"   ✅ Bot instance created with real components")
            print(f"   📊 Reference tokens: {self.bot.reference_tokens}")
            
        except Exception as e:
            print(f"❌ Error setting up bot: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)
    
    def test_get_historical_price_data(self, token: str, timeframe: str):
        """Test the actual _get_historical_price_data method"""
        print(f"\n🧪 Testing _get_historical_price_data for {token} ({timeframe})")
        
        try:
            # Map timeframe to hours as the method does
            if timeframe == "1h":
                hours = 24
            elif timeframe == "24h":
                hours = 7 * 24
            else:  # 7d
                hours = 30 * 24
            
            print(f"   📅 Requesting {hours} hours of data")
            
            # Call the actual method
            if self.bot is not None and hasattr(self.bot, '_get_historical_price_data'):
                historical_data = self.bot._get_historical_price_data(token, hours, timeframe)
                
                if historical_data == "Never":
                    print(f"   ❌ Method returned 'Never' - no data found")
                    return None
                elif not historical_data:
                    print(f"   ❌ Method returned empty data")
                    return None
                else:
                    print(f"   ✅ Method returned {len(historical_data)} data points")
                    
                    # Test the data format
                    if isinstance(historical_data, list) and len(historical_data) > 0:
                        first_entry = historical_data[0]
                        if isinstance(first_entry, dict):
                            print(f"   📊 Data format: Dictionary with keys {list(first_entry.keys())}")
                            if 'price' in first_entry:
                                print(f"   💰 Sample price: ${first_entry['price']:.6f}")
                            else:
                                print(f"   ❌ No 'price' key in data!")
                        else:
                            print(f"   ⚠️  Data format: {type(first_entry)} (expected dict)")
                    
                    return historical_data
            else:
                print(f"   ❌ Method _get_historical_price_data not found in bot")
                return None
                
        except Exception as e:
            print(f"   ❌ Error in _get_historical_price_data: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def test_get_crypto_data(self):
        """Test the actual _get_crypto_data method"""
        print(f"\n🚀 Testing _get_crypto_data method")
        
        try:
            # Check if bot is initialized
            if self.bot is None:
                print(f"   ❌ Bot instance is None - cannot test _get_crypto_data")
                return None
                
            # Check if the method exists
            if not hasattr(self.bot, '_get_crypto_data'):
                print(f"   ❌ Method _get_crypto_data not found in bot")
                return None
                
            print(f"   📡 Calling _get_crypto_data...")
            market_data = self.bot._get_crypto_data()
            
            if not market_data:
                print(f"   ❌ _get_crypto_data returned empty data")
                return None
            
            print(f"   ✅ _get_crypto_data returned {len(market_data)} tokens")
            
            # Check if reference_tokens exists
            if not hasattr(self.bot, 'reference_tokens'):
                print(f"   ⚠️ No reference_tokens attribute found in bot")
                return market_data
                
            # Check reference tokens availability
            available_refs = []
            for ref_token in self.bot.reference_tokens:
                if ref_token in market_data:
                    available_refs.append(ref_token)
                    token_data = market_data[ref_token]
                    price = token_data.get('current_price', 0)
                    print(f"   ✅ {ref_token}: ${price:,.2f}")
                else:
                    print(f"   ❌ {ref_token}: Missing from market_data")
            
            print(f"   📊 Available reference tokens: {len(available_refs)}/{len(self.bot.reference_tokens)}")
            return market_data
                
        except Exception as e:
            print(f"   ❌ Error in _get_crypto_data: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def extract_prices(self, historical_data):
        """Extract prices from historical data, handling different formats"""
        prices = []
        
        if not historical_data or historical_data == "Never":
            return prices
            
        for entry in historical_data:
            if isinstance(entry, dict) and 'price' in entry:
                prices.append(entry['price'])
        
        return prices
    
    def test_calculate_relative_volatility(self, token: str, market_data: Dict[str, Any], timeframe: str):
        """Test the actual _calculate_relative_volatility method"""
        print(f"\n🎯 Testing _calculate_relative_volatility for {token} ({timeframe})")
        
        try:
            # Check if bot is initialized
            if self.bot is None:
                print(f"   ❌ Bot instance is None - cannot test _calculate_relative_volatility")
                return None
                
            # Check if the method exists
            if not hasattr(self.bot, '_calculate_relative_volatility'):
                print(f"   ❌ Method _calculate_relative_volatility not found in bot")
                return None
                
            # Check if reference_tokens exists
            if not hasattr(self.bot, 'reference_tokens'):
                print(f"   ⚠️ No reference_tokens attribute found in bot")
                return None
                
            print(f"   🔄 Calling _calculate_relative_volatility...")
            
            result = self.bot._calculate_relative_volatility(
                token=token,
                reference_tokens=self.bot.reference_tokens,
                market_data=market_data,
                timeframe=timeframe
            )
            
            if result is None:
                print(f"   ❌ Method returned None - insufficient data or error")
                return None
            else:
                print(f"   ✅ Method returned relative volatility: {result:.6f}")
                interpretation = "MORE" if result > 1 else "LESS"
                print(f"   📊 Interpretation: {token} is {interpretation} volatile than market average")
                return result
                    
        except Exception as e:
            print(f"   ❌ Error in _calculate_relative_volatility: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def test_analyze_token_vs_market(self, token: str, market_data: Dict[str, Any], timeframe: str):
        """Test the actual _analyze_token_vs_market method"""
        print(f"\n📈 Testing _analyze_token_vs_market for {token} ({timeframe})")
        
        try:
            # Check if bot is initialized
            if self.bot is None:
                print(f"   ❌ Bot instance is None - cannot test _analyze_token_vs_market")
                return None
                
            # Check if the method exists
            if not hasattr(self.bot, '_analyze_token_vs_market'):
                print(f"   ❌ Method _analyze_token_vs_market not found in bot")
                return None
                
            print(f"   🔄 Calling _analyze_token_vs_market...")
            
            result = self.bot._analyze_token_vs_market(
                token=token,
                market_data=market_data,
                timeframe=timeframe
            )
            
            if not result:
                print(f"   ❌ Method returned empty result")
                return None
            else:
                print(f"   ✅ Method returned analysis result")
                
                # Check for relative volatility in extended metrics
                extended_metrics = result.get('extended_metrics', {})
                if 'relative_volatility' in extended_metrics:
                    rel_vol = extended_metrics['relative_volatility']
                    print(f"   🎯 Extended metrics contains relative_volatility: {rel_vol:.6f}")
                else:
                    print(f"   ⚠️  No relative_volatility found in extended_metrics")
                    print(f"   📊 Available extended metrics: {list(extended_metrics.keys())}")
                
                return result
                    
        except Exception as e:
            print(f"   ❌ Error in _analyze_token_vs_market: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def run_comprehensive_real_test(self):
        """Run comprehensive test using actual bot methods"""
        print("=" * 80)
        print("🧪 REAL BOT VOLATILITY METHOD TESTING")
        print("=" * 80)
        print(f"🤖 Testing actual bot instance with real methods and data")
        print()
        
        # Test 1: Get real market data
        print("📊 STEP 1: Testing Real Market Data Retrieval")
        market_data = self.test_get_crypto_data()
        
        if not market_data:
            print("❌ Cannot proceed without market data")
            return
        
        # Test 2: Test historical price data method
        print("\n📅 STEP 2: Testing Historical Price Data Method")
        test_cases = [
            {"token": "AAVE", "timeframe": "1h"},   # Original failing case
            {"token": "BTC", "timeframe": "1h"},
            {"token": "ETH", "timeframe": "24h"},
        ]
        
        historical_results = {}
        for test_case in test_cases:
            token = test_case["token"]
            timeframe = test_case["timeframe"]
            key = f"{token}_{timeframe}"
            
            historical_data = self.test_get_historical_price_data(token, timeframe)
            historical_results[key] = historical_data is not None
        
        # Test 3: Test relative volatility calculation
        print("\n🎯 STEP 3: Testing Relative Volatility Calculation")
        volatility_results = {}
        for test_case in test_cases:
            token = test_case["token"]
            timeframe = test_case["timeframe"]
            key = f"{token}_{timeframe}"
            
            if token in market_data:
                volatility = self.test_calculate_relative_volatility(token, market_data, timeframe)
                volatility_results[key] = volatility
            else:
                print(f"   ⚠️  Skipping {token} - not in market_data")
                volatility_results[key] = None
        
        # Test 4: Test full integration
        print("\n🔗 STEP 4: Testing Full Integration (_analyze_token_vs_market)")
        integration_results = {}
        for test_case in test_cases:
            token = test_case["token"]
            timeframe = test_case["timeframe"]
            key = f"{token}_{timeframe}"
            
            if token in market_data:
                analysis = self.test_analyze_token_vs_market(token, market_data, timeframe)
                integration_results[key] = analysis is not None
            else:
                print(f"   ⚠️  Skipping {token} - not in market_data")
                integration_results[key] = False
        
        # Final Results
        print("\n" + "=" * 80)
        print("📋 REAL BOT TEST RESULTS")
        print("=" * 80)
        
        # Historical data results
        hist_passed = sum(1 for result in historical_results.values() if result)
        hist_total = len(historical_results)
        print(f"📅 Historical Data Method: {hist_passed}/{hist_total} passed")
        
        for key, result in historical_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {key}")
        
        # Volatility calculation results
        vol_passed = sum(1 for result in volatility_results.values() if result is not None)
        vol_total = len(volatility_results)
        print(f"\n🎯 Volatility Calculation: {vol_passed}/{vol_total} passed")
        
        for key, result in volatility_results.items():
            if result is not None:
                print(f"   ✅ {key}: {result:.6f}")
            else:
                print(f"   ❌ {key}: Failed")
        
        # Integration results
        int_passed = sum(1 for result in integration_results.values() if result)
        int_total = len(integration_results)
        print(f"\n🔗 Full Integration: {int_passed}/{int_total} passed")
        
        for key, result in integration_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {key}")
        
        # Overall assessment
        total_passed = hist_passed + vol_passed + int_passed
        total_tests = hist_total + vol_total + int_total
        overall_success = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests Passed: {total_passed}/{total_tests}")
        print(f"   Success Rate: {overall_success:.1f}%")
        
        if overall_success == 100:
            print(f"\n🎉 PERFECT! All real bot methods are working correctly!")
            print(f"   ✅ Your volatility fix is fully functional in the actual bot")
            print(f"   ✅ No more 'Insufficient price data for AAVE: 0 points' errors")
            print(f"   ✅ The bot should now calculate relative volatility successfully")
        elif overall_success >= 80:
            print(f"\n✅ MOSTLY WORKING! Most real bot methods are functional")
            print(f"   Some issues remain but the core fix is working")
        else:
            print(f"\n❌ ISSUES REMAIN: The fix may need more work")
            print(f"   Check the failed tests above for specific problems")
        
        print(f"\n💡 Next Steps:")
        if overall_success == 100:
            print(f"   1. Your bot is ready to run with working volatility calculations")
            print(f"   2. Monitor live logs to confirm no more '0 points' errors")
            print(f"   3. Check that volatility metrics appear in your analysis")
        else:
            print(f"   1. Review failed test cases above")
            print(f"   2. Check method implementations in your bot code")
            print(f"   3. Verify database connectivity and data quality")

def main():
    """Main execution function"""
    print("🤖 Real Bot Volatility Method Tester")
    print("This tests your actual bot methods with real data from your files")
    print()
    
    tester = RealVolatilityTester()
    tester.run_comprehensive_real_test()

if __name__ == "__main__":
    main()