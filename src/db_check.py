#!/usr/bin/env python3
"""
Simple diagnostic to check if prediction engine has proper database access
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_prediction_engine_database_access():
    """Check if prediction engine can access database for sparkline supplementation"""
    
    print("🔍 CHECKING PREDICTION ENGINE DATABASE ACCESS")
    print("=" * 60)
    
    try:
        # Import your prediction engine
        from prediction_engine import EnhancedPredictionEngine
        from database import CryptoDatabase
        from llm_provider import LLMProvider
        
        print("✅ Successfully imported prediction engine modules")
        
        # Initialize database
        db = CryptoDatabase()
        print("✅ Database initialized")
        
        # Initialize LLM provider (needed for prediction engine)
        from config import Config
        config = Config()
        llm = LLMProvider(config)
        print("✅ LLM provider initialized")
        
        # Initialize prediction engine
        prediction_engine = EnhancedPredictionEngine(database=db, llm_provider=llm)
        print("✅ Prediction engine initialized")
        
        # Check if prediction engine has db attribute
        has_db_attr = hasattr(prediction_engine, 'db')
        print(f"📊 prediction_engine.db attribute exists: {has_db_attr}")
        
        if has_db_attr:
            db_is_none = prediction_engine.db is None
            print(f"📊 prediction_engine.db is None: {db_is_none}")
            
            if not db_is_none:
                # Test the actual database method call
                try:
                    test_result = prediction_engine.db.build_sparkline_from_price_history('KAITO', hours=168)
                    print(f"✅ Database method test: {len(test_result)} points returned for KAITO")
                    
                    if len(test_result) >= 10:
                        print("✅ DATABASE ACCESS IS WORKING - This is NOT the problem")
                    else:
                        print("❌ Database method returns insufficient data")
                        
                except Exception as db_test_error:
                    print(f"❌ Database method call failed: {db_test_error}")
            else:
                print("❌ PROBLEM FOUND: prediction_engine.db is None")
        else:
            print("❌ PROBLEM FOUND: prediction_engine has no 'db' attribute")
            
        # Check the specific condition in the code
        condition_check = hasattr(prediction_engine, 'db') and prediction_engine.db
        print(f"📊 Condition 'hasattr(self, 'db') and self.db': {condition_check}")
        
        if not condition_check:
            print("❌ PROBLEM IDENTIFIED: Database supplementation code will be skipped")
            print("💡 This explains why KAITO gets 0 points despite database having data")
        else:
            print("✅ Database supplementation condition passes")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error during check: {e}")


if __name__ == "__main__":
    check_prediction_engine_database_access()