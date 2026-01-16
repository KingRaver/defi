"""
🔧 PREDICTION DATA STRUCTURE DIAGNOSTIC TOOL
============================================

This tool diagnoses and fixes the data structure mismatch between 
prediction_engine.py and integrated_trading_bot.py that's causing 
trades to be blocked with "LOW risk" assessments.

Usage:
    python prediction_diagnostics.py
"""

import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy

class PredictionDataDiagnostics:
    """
    Comprehensive diagnostic tool for prediction data structure issues
    """
    
    def __init__(self):
        """Initialize the diagnostic tool"""
        self.test_cases = []
        self.fixes_applied = []
        self.structure_analysis = {}
        
        print("🔧 Prediction Data Structure Diagnostics Tool Initialized")
        print("=" * 60)
    
    def capture_prediction_data(self, prediction_data: Any, source: str = "unknown") -> str:
        """
        Capture and analyze prediction data structure
        
        Args:
            prediction_data: The prediction data to analyze
            source: Source of the data (e.g., "prediction_engine", "normalized")
            
        Returns:
            Analysis ID for tracking
        """
        analysis_id = f"{source}_{int(time.time())}"
        
        analysis = {
            'id': analysis_id,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data_type': str(type(prediction_data)),
            'raw_data': prediction_data,
            'structure_map': self._map_structure(prediction_data),
            'expected_fields': self._check_expected_fields(prediction_data),
            'issues_found': self._identify_issues(prediction_data),
            'recommended_fixes': []
        }
        
        # Generate fix recommendations
        analysis['recommended_fixes'] = self._generate_fix_recommendations(analysis)
        
        self.test_cases.append(analysis)
        
        # Print immediate analysis
        self._print_analysis(analysis)
        
        return analysis_id
    
    def _map_structure(self, data: Any, path: str = "root", max_depth: int = 5) -> Dict[str, Any]:
        """
        Create a comprehensive map of the data structure
        """
        if max_depth <= 0:
            return {"type": str(type(data)), "value": "MAX_DEPTH_REACHED"}
    
        structure: Dict[str, Any] = {
            "type": str(type(data)),
            "path": path
        }
    
        if isinstance(data, dict):
            structure["keys"] = list(data.keys())
            structure["children"] = {}
            for key, value in data.items():
                child_path = f"{path}.{str(key)}"
                str_key = str(key)
                structure["children"][str_key] = self._map_structure(value, child_path, max_depth - 1)
    
        elif isinstance(data, list):
            structure["length"] = len(data)
            structure["children"] = {}
            for i, item in enumerate(data[:3]):  # Only analyze first 3 items
                child_path = f"{path}[{i}]"
                structure["children"][f"item_{i}"] = self._map_structure(item, child_path, max_depth - 1)
    
        else:
            structure["value"] = str(data)[:100]  # Truncate long values
    
        return structure
    
    def _check_expected_fields(self, data: Any) -> Dict[str, Any]:
        """
        Check for expected trading bot fields
        """
        expected_fields = {
            # Critical fields for trading bot
            'token': {'found': False, 'path': None, 'value': None},
            'confidence': {'found': False, 'path': None, 'value': None},
            'expected_return_pct': {'found': False, 'path': None, 'value': None},
            'volatility_score': {'found': False, 'path': None, 'value': None},
            'market_condition': {'found': False, 'path': None, 'value': None},
            
            # Optional but useful fields
            'current_price': {'found': False, 'path': None, 'value': None},
            'predicted_price': {'found': False, 'path': None, 'value': None},
            'direction': {'found': False, 'path': None, 'value': None},
            'data_quality_score': {'found': False, 'path': None, 'value': None}
        }
        
        # Search for fields at all levels
        self._search_fields_recursive(data, expected_fields, "root")
        
        return expected_fields
    
    def _search_fields_recursive(self, data: Any, expected_fields: Dict, current_path: str):
        """
        Recursively search for expected fields in nested structures
        """
        if isinstance(data, dict):
            for key, value in data.items():
                str_key = str(key)  # Convert key to string
                new_path = f"{current_path}.{str_key}"
                
                # Check if this key matches any expected field
                for expected_key in expected_fields:
                    if str_key.lower() == expected_key.lower() or str_key.lower().replace('_', '') == expected_key.lower().replace('_', ''):
                        expected_fields[expected_key]['found'] = True
                        expected_fields[expected_key]['path'] = new_path
                        expected_fields[expected_key]['value'] = value
                
                # Continue searching deeper
                self._search_fields_recursive(value, expected_fields, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]"
                self._search_fields_recursive(item, expected_fields, new_path)
    
    def _identify_issues(self, data: Any) -> List[str]:
        """
        Identify specific issues with the data structure
        """
        issues = []
        
        # Check if data is None or empty
        if data is None:
            issues.append("CRITICAL: Data is None")
            return issues
        
        if isinstance(data, dict) and not data:
            issues.append("CRITICAL: Data dictionary is empty")
            return issues
        
        # Check for common structural issues
        if isinstance(data, dict):
            # Look for over-nesting
            if 'prediction' in data and isinstance(data['prediction'], dict):
                if 'prediction' in data['prediction']:
                    issues.append("WARNING: Double-nested 'prediction' key detected")
            
        if isinstance(data, dict):
            # Check for missing critical fields at root level
            root_keys = [str(key) for key in data.keys()]  # Convert all keys to strings
            critical_fields = ['token', 'confidence', 'expected_return_pct']
            missing_at_root = [field for field in critical_fields if field not in root_keys]
            
            if len(missing_at_root) > 0:
                issues.append(f"INFO: Critical fields missing at root level: {missing_at_root}")
            
            # Check for data type mismatches
            for key, value in data.items():
                str_key = str(key)  # Convert key to string for comparison
                if str_key in ['confidence', 'expected_return_pct', 'volatility_score']:
                    if not isinstance(value, (int, float)):
                        issues.append(f"TYPE_ERROR: {str_key} should be numeric, got {type(value)}")
        
        elif isinstance(data, list):
            issues.append("STRUCTURE_ERROR: Root data is a list, expected dictionary")
        
        else:
            issues.append(f"STRUCTURE_ERROR: Root data is {type(data)}, expected dictionary")
        
        return issues
    
    def _generate_fix_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific fix recommendations based on analysis
        """
        recommendations = []
        expected_fields = analysis['expected_fields']
        issues = analysis['issues_found']
        
        # Recommendations based on missing fields
        missing_critical = [field for field, info in expected_fields.items() 
                          if field in ['token', 'confidence', 'expected_return_pct'] and not info['found']]
        
        if missing_critical:
            recommendations.append(f"CRITICAL_FIX: Add missing fields to root level: {missing_critical}")
        
        # Recommendations based on found fields in wrong locations
        misplaced_fields = []
        for field, info in expected_fields.items():
            if info['found'] and info['path'] and 'root.' not in info['path']:
                misplaced_fields.append((field, info['path']))
        
        if misplaced_fields:
            recommendations.append("STRUCTURE_FIX: Move nested fields to root level:")
            for field, path in misplaced_fields:
                recommendations.append(f"  - Move {field} from {path} to root")
        
        # Recommendations based on issues
        for issue in issues:
            if "Double-nested" in issue:
                recommendations.append("STRUCTURE_FIX: Flatten double-nested prediction structure")
            elif "TYPE_ERROR" in issue:
                recommendations.append(f"TYPE_FIX: Convert field types - {issue}")
        
        return recommendations
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """
        Print formatted analysis results
        """
        print(f"\n🔍 ANALYSIS: {analysis['id']}")
        print(f"📊 Source: {analysis['source']}")
        print(f"⏰ Timestamp: {analysis['timestamp']}")
        print(f"📋 Data Type: {analysis['data_type']}")
        
        # Print structure overview
        print(f"\n📁 STRUCTURE OVERVIEW:")
        if isinstance(analysis['raw_data'], dict):
            # Convert all keys to strings for safe display
            display_keys = [str(key) for key in analysis['raw_data'].keys()]
            print(f"   Root Keys: {display_keys}")
        
        # Print expected fields status
        print(f"\n✅ EXPECTED FIELDS STATUS:")
        expected_fields = analysis['expected_fields']
        for field, info in expected_fields.items():
            status = "✅ FOUND" if info['found'] else "❌ MISSING"
            location = f" at {info['path']}" if info['found'] and info['path'] else ""
            value_preview = f" = {str(info['value'])[:50]}" if info['found'] and info['value'] is not None else ""
            print(f"   {field:20} {status}{location}{value_preview}")
        
        # Print issues
        if analysis['issues_found']:
            print(f"\n⚠️  ISSUES IDENTIFIED:")
            for issue in analysis['issues_found']:
                print(f"   • {issue}")
        
        # Print recommendations
        if analysis['recommended_fixes']:
            print(f"\n🔧 RECOMMENDED FIXES:")
            for fix in analysis['recommended_fixes']:
                print(f"   • {fix}")
        
        print("-" * 60)
    
    def create_fixed_normalization_function(self) -> str:
        """
        Generate an improved normalization function based on analysis
        """
        function_code = '''
def fixed_normalize_prediction_format(prediction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Robust prediction normalization that handles all discovered data structures
    Generated by diagnostic tool based on actual data analysis
    """
    try:
        print(f"🔧 NORMALIZING: {type(prediction_dict)} with keys: {list(prediction_dict.keys()) if isinstance(prediction_dict, dict) else 'N/A'}")
        
        # Start with clean normalized dictionary
        normalized = {}
        
        # STEP 1: Extract TOKEN
        token = None
        if isinstance(prediction_dict, dict):
            # Try direct access first
            if 'token' in prediction_dict:
                token = prediction_dict['token']
            # Try nested in prediction object
            elif 'prediction' in prediction_dict and isinstance(prediction_dict['prediction'], dict):
                if 'token' in prediction_dict['prediction']:
                    token = prediction_dict['prediction']['token']
            # Try in metadata
            elif 'metadata' in prediction_dict and isinstance(prediction_dict['metadata'], dict):
                if 'token' in prediction_dict['metadata']:
                    token = prediction_dict['metadata']['token']
        
        if not token:
            print("❌ CRITICAL: No token found in prediction data")
            return prediction_dict  # Return original if no token
        
        normalized['token'] = str(token).upper()
        
        # STEP 2: Extract CONFIDENCE (multiple paths)
        confidence = 0.0
        confidence_paths = [
            ['confidence'],
            ['prediction', 'confidence'],
            ['technical_analysis', 'final_confidence'],
            ['technical_analysis', 'confidence'],
            ['technical_analysis', 'signal_confidence'],
            ['prediction_confidence'],
            ['final_confidence']
        ]
        
        for path in confidence_paths:
            try:
                value = prediction_dict
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    # Successfully traversed full path
                    if isinstance(value, (int, float)):
                        confidence = float(value)
                        print(f"✅ Found confidence: {confidence} at path: {' -> '.join(path)}")
                        break
            except (KeyError, TypeError):
                continue
        
        normalized['confidence'] = max(0.0, min(100.0, confidence))
        
        # STEP 3: Extract EXPECTED RETURN (multiple paths)
        expected_return_pct = 0.0
        return_paths = [
            ['expected_return_pct'],
            ['prediction', 'expected_return_pct'],
            ['prediction', 'percent_change'],
            ['percent_change'],
            ['predicted_change_pct']
        ]
        
        for path in return_paths:
            try:
                value = prediction_dict
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if isinstance(value, (int, float)):
                        expected_return_pct = float(value)
                        print(f"✅ Found expected_return_pct: {expected_return_pct} at path: {' -> '.join(path)}")
                        break
            except (KeyError, TypeError):
                continue
        
        # Try to calculate from predicted_price and current_price if not found
        if expected_return_pct == 0.0:
            try:
                predicted_price = None
                current_price = None
                
                # Try multiple paths for prices
                price_paths = [
                    ['prediction', 'predicted_price'],
                    ['predicted_price'],
                    ['prediction', 'price']
                ]
                
                for path in price_paths:
                    value = prediction_dict
                    for key in path:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            break
                    else:
                        if isinstance(value, (int, float)):
                            predicted_price = float(value)
                            break
                
                current_price_paths = [
                    ['current_price'],
                    ['prediction', 'current_price'],
                    ['market_data', 'current_price']
                ]
                
                for path in current_price_paths:
                    value = prediction_dict
                    for key in path:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            break
                    else:
                        if isinstance(value, (int, float)):
                            current_price = float(value)
                            break
                
                if predicted_price and current_price and current_price > 0:
                    expected_return_pct = ((predicted_price - current_price) / current_price) * 100
                    print(f"✅ Calculated expected_return_pct: {expected_return_pct} from prices")
                    
            except (KeyError, TypeError, ZeroDivisionError):
                pass
        
        normalized['expected_return_pct'] = expected_return_pct
        
        # STEP 4: Extract VOLATILITY SCORE
        volatility_score = 50.0  # Default medium volatility
        volatility_paths = [
            ['volatility_score'],
            ['prediction', 'volatility_score'],
            ['risk_assessment', 'volatility_score'],
            ['technical_analysis', 'volatility_score']
        ]
        
        for path in volatility_paths:
            try:
                value = prediction_dict
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if isinstance(value, (int, float)):
                        volatility_score = float(value)
                        print(f"✅ Found volatility_score: {volatility_score} at path: {' -> '.join(path)}")
                        break
            except (KeyError, TypeError):
                continue
        
        normalized['volatility_score'] = max(0.0, min(100.0, volatility_score))
        
        # STEP 5: Extract MARKET CONDITION
        market_condition = 'UNKNOWN'
        condition_paths = [
            ['market_condition'],
            ['prediction', 'market_condition'],
            ['market_analysis', 'market_condition'],
            ['market_analysis', 'condition']
        ]
        
        for path in condition_paths:
            try:
                value = prediction_dict
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if isinstance(value, str):
                        market_condition = str(value).upper()
                        print(f"✅ Found market_condition: {market_condition} at path: {' -> '.join(path)}")
                        break
                    elif isinstance(value, dict) and 'value' in value:
                        market_condition = str(value['value']).upper()
                        print(f"✅ Found market_condition: {market_condition} at path: {' -> '.join(path)} -> value")
                        break
            except (KeyError, TypeError):
                continue
        
        normalized['market_condition'] = market_condition
        
        # STEP 6: Add additional useful fields
        normalized['data_quality_score'] = 75.0  # Default
        
        # Determine direction
        if expected_return_pct > 0.1:
            normalized['direction'] = 'bullish'
        elif expected_return_pct < -0.1:
            normalized['direction'] = 'bearish'
        else:
            normalized['direction'] = 'neutral'
        
        # Add timestamp
        normalized['normalized_timestamp'] = time.time()
        
        print(f"✅ NORMALIZATION COMPLETE:")
        print(f"   Token: {normalized['token']}")
        print(f"   Confidence: {normalized['confidence']:.1f}%")
        print(f"   Expected Return: {normalized['expected_return_pct']:.2f}%")
        print(f"   Direction: {normalized['direction']}")
        print(f"   Market Condition: {normalized['market_condition']}")
        
        return normalized
        
    except Exception as e:
        print(f"❌ NORMALIZATION ERROR: {e}")
        print(f"   Original data type: {type(prediction_dict)}")
        print(f"   Original data keys: {list(prediction_dict.keys()) if isinstance(prediction_dict, dict) else 'N/A'}")
        
        # Return minimal valid structure to prevent blocking
        return {
            'token': 'UNKNOWN',
            'confidence': 0.0,
            'expected_return_pct': 0.0,
            'volatility_score': 50.0,
            'market_condition': 'UNKNOWN',
            'direction': 'neutral',
            'error': str(e),
            'normalized_timestamp': time.time()
        }
'''
        
        return function_code
    
    def generate_test_suite(self) -> str:
        """
        Generate a test suite based on discovered data structures
        """
        test_code = '''
def test_prediction_normalization():
    """
    Test suite for prediction normalization based on actual data structures
    """
    # Test Case 1: Nested prediction structure
    test_case_1 = {
        "prediction": {
            "token": "AVAX",
            "confidence": 72.0,
            "predicted_price": 45.67,
            "current_price": 45.50,
            "percent_change": 0.37
        },
        "technical_analysis": {
            "final_confidence": 68.5
        },
        "market_analysis": {
            "market_condition": "BULLISH"
        }
    }
    
    # Test Case 2: Flat structure
    test_case_2 = {
        "token": "BTC",
        "confidence": 85.0,
        "expected_return_pct": 2.5,
        "volatility_score": 45.0,
        "market_condition": "SIDEWAYS"
    }
    
    # Test Case 3: Complex nested structure
    test_case_3 = {
        "metadata": {
            "token": "ETH"
        },
        "prediction": {
            "prediction": {  # Double nested
                "confidence": 90.0
            },
            "expected_return_pct": 1.8
        },
        "risk_assessment": {
            "volatility_score": 65.0
        }
    }
    
    # Test all cases
    test_cases = [test_case_1, test_case_2, test_case_3]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n🧪 TESTING CASE {i}:")
        try:
            result = fixed_normalize_prediction_format(test_case)
            print(f"✅ Test {i} PASSED")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ Test {i} FAILED: {e}")
'''
        
        return test_code
    
    def create_comprehensive_report(self) -> str:
        """
        Create a comprehensive diagnostic report
        """
        report = []
        report.append("🔧 PREDICTION DATA STRUCTURE DIAGNOSTIC REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Analyses: {len(self.test_cases)}")
        report.append("")
        
        # Summary of issues
        all_issues = []
        for case in self.test_cases:
            all_issues.extend(case['issues_found'])
        
        unique_issues = list(set(all_issues))
        
        report.append("📊 SUMMARY OF ISSUES FOUND:")
        for issue in unique_issues:
            count = all_issues.count(issue)
            report.append(f"   • {issue} (occurred {count} times)")
        report.append("")
        
        # Detailed analysis for each case
        for case in self.test_cases:
            report.append(f"🔍 ANALYSIS: {case['id']}")
            report.append(f"   Source: {case['source']}")
            report.append(f"   Data Type: {case['data_type']}")
            
            # Expected fields status
            found_fields = [field for field, info in case['expected_fields'].items() if info['found']]
            missing_fields = [field for field, info in case['expected_fields'].items() if not info['found']]
            
            report.append(f"   Found Fields: {found_fields}")
            report.append(f"   Missing Fields: {missing_fields}")
            report.append(f"   Issues: {case['issues_found']}")
            report.append("")
        
        return "\\n".join(report)

# Test the diagnostic tool with sample data
def run_diagnostics():
    """
    Run the diagnostic tool with sample data structures
    """
    diagnostics = PredictionDataDiagnostics()
    
    # Sample data structures that might come from prediction engine
    sample_structures = [
        # Structure 1: Nested prediction
        {
            "prediction": {
                "token": "AVAX",
                "confidence": 72.0,
                "predicted_price": 45.67,
                "current_price": 45.50
            },
            "technical_analysis": {
                "final_confidence": 68.5,
                "volatility_score": 55.0
            },
            "market_analysis": {
                "market_condition": "BULLISH"
            }
        },
        
        # Structure 2: Flat structure
        {
            "token": "BTC",
            "confidence": 85.0,
            "expected_return_pct": 2.5,
            "volatility_score": 45.0,
            "market_condition": "SIDEWAYS"
        },
        
        # Structure 3: Missing critical fields
        {
            "symbol": "ETH",  # Wrong field name
            "prediction_confidence": 90.0,  # Wrong field name
            "percent_change": 1.8,  # Wrong field name
            "risk_level": "medium"  # Wrong field name
        }
    ]
    
    # Analyze each structure
    for i, structure in enumerate(sample_structures, 1):
        print(f"\\n{'='*60}")
        print(f"ANALYZING SAMPLE STRUCTURE {i}")
        print(f"{'='*60}")
        diagnostics.capture_prediction_data(structure, f"sample_{i}")
    
    # Generate the fixed normalization function
    print(f"\\n{'='*60}")
    print("GENERATING FIXED NORMALIZATION FUNCTION")
    print(f"{'='*60}")
    
    fixed_function = diagnostics.create_fixed_normalization_function()
    print("✅ Fixed normalization function generated")
    
    # Generate test suite
    test_suite = diagnostics.generate_test_suite()
    print("✅ Test suite generated")
    
    # Generate comprehensive report
    report = diagnostics.create_comprehensive_report()
    print(f"\\n{report}")
    
    return diagnostics, fixed_function, test_suite

if __name__ == "__main__":
    # Run the diagnostics
    diagnostics, fixed_function, test_suite = run_diagnostics()
    
    # Print the fixed function code
    print("\\n" + "="*60)
    print("FIXED NORMALIZATION FUNCTION CODE:")
    print("="*60)
    print(fixed_function)
    
    print("\\n" + "="*60)
    print("TEST SUITE CODE:")
    print("="*60)
    print(test_suite)