#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# 🚀 INTELLIGENT DUAL API MANAGER WITH CENTRALIZED ROUTING 🚀
# ================================================================
# PART 1 OF 3: IMPORTS + CORE CLASS SETUP + PROVIDER INITIALIZATION
# 
# Features:
# - Centralized routing through provider_specialization config (NO hardcoded logic)
# - Recursion prevention with threading-based tracking
# - Comprehensive cache failure diagnostics with root cause analysis
# - Intelligent load balancing based on request type and provider capabilities
# - Rate limit prevention through intelligent request distribution
# - Auto-storage to database with source combination
# - Fail-fast diagnostics for cache issues
# ================================================================

import os
import time
import threading
import json
import statistics
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
import inspect
import logging

# Database and core dependencies
from database import CryptoDatabase

# Logging utilities
from utils.logger import logger

# API Handlers with validation functions
from coingecko_handler import CoinGeckoHandler, validate_trading_readiness as validate_coingecko_readiness
from coinmarketcap_handler import CoinMarketCapHandler, validate_trading_readiness as validate_coinmarketcap_readiness

# Configuration and token mapping
from config import TokenMappingManager

# ================================================================
# 🎯 CENTRALIZED API MANAGER CLASS DEFINITION
# ================================================================

class CryptoAPIManager:
    """
    🚀 INTELLIGENT DUAL API MANAGER WITH CENTRALIZED ROUTING AND RECURSION PREVENTION 🚀
    
    CORE FEATURES:
    - 🎯 CENTRALIZED ROUTING: All decisions through provider_specialization config
    - 🛡️ RECURSION PREVENTION: Threading-based circular call detection
    - 🔍 CACHE FAILURE DIAGNOSTICS: Root cause analysis for cache issues
    - 📊 INTELLIGENT LOAD BALANCING: Request type-based provider selection
    - ⚡ RATE LIMIT PREVENTION: Smart request distribution
    - 🗄️ AUTO-STORAGE: Automatic database storage with source combination
    - 🚀 FAIL-FAST DIAGNOSTICS: Immediate cache issue identification
    
    ROUTING STRATEGY:
    - CoinMarketCap: Bulk data, 7d historical, real-time, market overview
    - CoinGecko: Historical data, individual tokens
    - Automatic fallback with consistent method signatures
    """
    
    def __init__(self):
        """Initialize API manager with centralized routing and recursion prevention"""
        
        # ================================================================
        # 🎯 CENTRALIZED ROUTING CONFIGURATION (SINGLE SOURCE OF TRUTH)
        # ================================================================
        
        # CORE ROUTING TABLE - ALL DECISIONS FLOW THROUGH THIS
        self.provider_specialization = {
            'bulk_data': 'coinmarketcap',        # Bulk operations → CMC
            '7d_historical_data': 'coingecko',  # 7-day historical data → CoinGecko
            'historical_data': 'coingecko',      # Historical data → CoinGecko
            'real_time': 'coingecko',        # Real-time prices → CoinGecko
            'individual_tokens': 'coingecko',    # Single token requests → CoinGecko
            'market_overview': 'coinmarketcap',   # Market overview → CMC 
            'batched': 'coingecko'             # Batched missing data → CoinGecko         
        }
        
        logger.info("🎯 CENTRALIZED ROUTING: Provider specialization configured")
        logger.info(f"📊 Routing Table: {self.provider_specialization}")
        
        # ================================================================
        # 🛡️ RECURSION PREVENTION SYSTEM
        # ================================================================
        
        # Thread-safe recursion tracking
        self._active_requests = set()  # Track ongoing requests by thread+method
        self._request_depth = {}       # Track call depth per thread
        self._lock = threading.Lock()  # Thread safety for recursion tracking
        self.MAX_RECURSION_DEPTH = 20  # Maximum allowed recursion depth
        
        logger.info("🛡️ RECURSION PREVENTION: Thread-safe tracking initialized")
        
        # ================================================================
        # 🔍 CACHE FAILURE DIAGNOSTICS SYSTEM
        # ================================================================
        
        # Comprehensive cache failure tracking
        self.cache_failure_tracker = {
            'failures': [],                # Detailed failure records
            'root_causes': {},            # Root cause frequency tracking
            'exceptions': {},             # Exception pattern tracking
            'provider_patterns': {},      # Provider-specific failure patterns
            'endpoint_patterns': {},      # Endpoint-specific failure patterns
            'time_patterns': {}           # Time-based failure pattern analysis
        }
        
        logger.info("🔍 CACHE DIAGNOSTICS: Comprehensive failure tracking initialized")
        
        # ================================================================
        # 📊 LOAD BALANCING AND PERFORMANCE TRACKING
        # ================================================================
        
        # Core tracking for intelligent routing
        self.providers = {}
        self.provider_status = {}
        self.last_provider = None
        self.last_provider_change = time.time()
        self.min_provider_switch_interval = 300  # 5 minutes minimum between switches
        
        # Request statistics for load balancing analysis
        self.request_stats = {
            'coingecko': {
                'total': 0, 'bulk': 0, 'individual': 0, 
                'historical': 0, 'real_time': 0, '7d_historical_data': 0,
                'market_overview': 0, 'cache_hits': 0, 'cache_misses': 0
            },
            'coinmarketcap': {
                'total': 0, 'bulk': 0, 'individual': 0, 
                'historical': 0, 'real_time': 0, '7d_historical_data': 0,
                'market_overview': 0, 'cache_hits': 0, 'cache_misses': 0
            }
        }
        
        # Performance metrics tracking
        self.performance_metrics = {
            'response_times': {'coingecko': [], 'coinmarketcap': []},
            'success_rates': {'coingecko': 0.0, 'coinmarketcap': 0.0},
            'cache_effectiveness': {'coingecko': 0.0, 'coinmarketcap': 0.0},
            'routing_decisions': {},
            'fallback_usage': {'coingecko': 0, 'coinmarketcap': 0}
        }
        
        logger.info("📊 PERFORMANCE TRACKING: Comprehensive metrics system initialized")
        
        # ================================================================
        # 🗄️ DATABASE CONNECTION FOR AUTO-STORAGE
        # ================================================================
        
        # Initialize database connection for automatic storage
        try:
            self.db = CryptoDatabase()
            logger.info("📊 DATABASE CONNECTION: Auto-storage initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ DATABASE CONNECTION: Auto-storage failed: {e}")
            self.db = None
        
        # ================================================================
        # 🚀 PROVIDER INITIALIZATION WITH ROBUST ERROR HANDLING
        # ================================================================
        
        # Initialize provider preference order for fallback
        self.provider_preference = ['coingecko', 'coinmarketcap']
        
        # Track initialization results for comprehensive logging
        initialization_results = []
        
        # ================================================================
        # 🌟 COINGECKO INITIALIZATION - (FREE TIER)
        # ================================================================
        try:
            coingecko = CoinGeckoHandler(base_url="https://api.coingecko.com/api/v3")
            self.providers['coingecko'] = coingecko
            self.provider_status['coingecko'] = {
                'available': True,
                'last_check': time.time(),
                'readiness': None,
                'initialization_error': None,
                'specializations': ['historical_data', 'individual_tokens'],
                'method_capabilities': {
                    'get_market_data': True,
                    'get_with_cache': True,
                    'get_price_history': True
                },
                'cache_enabled': hasattr(coingecko, '_cache'),
                'quota_tracking': hasattr(coingecko, 'quota_tracker')
            }
            logger.info("✅ COINGECKO PROVIDER: Initialized successfully")
            logger.info("📊 CoinGecko Specializations: Historical data, Individual tokens")
            initialization_results.append(("coingecko", True, None))
        except Exception as e:
            logger.error(f"❌ COINGECKO PROVIDER: Initialization failed: {str(e)}")
            self.provider_status['coingecko'] = {
                'available': False,
                'last_check': time.time(),
                'error': str(e),
                'initialization_error': str(e),
                'specializations': [],
                'method_capabilities': {},
                'cache_enabled': False,
                'quota_tracking': False
            }
            initialization_results.append(("coingecko", False, str(e)))
        
        # ================================================================
        # 💰 COINMARKETCAP INITIALIZATION - (FREE TIER)
        # ================================================================
        try:
            # Try multiple environment variable names for API key discovery
            api_key_candidates = [
                os.getenv('CoinMarketCap_API', ''),           # Original name
                os.getenv('COINMARKETCAP_API_KEY', ''),       # Standard name
                os.getenv('CMC_API_KEY', ''),                 # Short name
                os.getenv('COINMARKETCAP_API', ''),           # Alternative name
                os.getenv('CMC_API', ''),                     # Minimal name
            ]
            
            # Find first non-empty API key
            api_key = None
            for candidate in api_key_candidates:
                if candidate and candidate.strip():
                    api_key = candidate.strip()
                    logger.debug(f"🔑 COINMARKETCAP API KEY: Found in environment")
                    break
            
            if api_key:
                coinmarketcap = CoinMarketCapHandler(api_key=api_key)
                self.providers['coinmarketcap'] = coinmarketcap
                self.provider_status['coinmarketcap'] = {
                    'available': True,
                    'last_check': time.time(),
                    'readiness': None,
                    'initialization_error': None,
                    'specializations': ['bulk_data', '7d_historical_data', 'real_time', 'market_overview'],
                    'method_capabilities': {
                        'get_market_data': True,
                        'get_with_cache': True,
                        'get_historical_data': True
                    },
                    'cache_enabled': hasattr(coinmarketcap, '_cache'),
                    'quota_tracking': hasattr(coinmarketcap, 'quota_tracker'),
                    'api_key_configured': True
                }
                logger.info("✅ COINMARKETCAP PROVIDER: Initialized successfully")
                logger.info("📊 CoinMarketCap Specializations: Bulk data, 7d historical, Real-time, Market overview")
                initialization_results.append(("coinmarketcap", True, None))
            else:
                logger.warning("⚠️ COINMARKETCAP API KEY: Not found in environment variables")
                logger.info("🔍 Checked variables: CoinMarketCap_API, COINMARKETCAP_API_KEY, CMC_API_KEY, COINMARKETCAP_API, CMC_API")
                self.provider_status['coinmarketcap'] = {
                    'available': False,
                    'last_check': time.time(),
                    'error': 'API key not found in environment variables',
                    'initialization_error': 'Missing API key',
                    'specializations': [],
                    'method_capabilities': {},
                    'cache_enabled': False,
                    'quota_tracking': False,
                    'api_key_configured': False
                }
                initialization_results.append(("coinmarketcap", False, "Missing API key"))
                
        except Exception as e:
            logger.error(f"❌ COINMARKETCAP PROVIDER: Initialization failed: {str(e)}")
            self.provider_status['coinmarketcap'] = {
                'available': False,
                'last_check': time.time(),
                'error': str(e),
                'initialization_error': str(e),
                'specializations': [],
                'method_capabilities': {},
                'cache_enabled': False,
                'quota_tracking': False,
                'api_key_configured': False
            }
            initialization_results.append(("coinmarketcap", False, str(e)))
 
        # ================================================================
        # ✅ INITIALIZATION VALIDATION AND SUMMARY
        # ================================================================
        
        # Validate that at least one provider is available
        available_providers = [name for name, status in self.provider_status.items() 
                             if status['available']]
        
        if not available_providers:
            # This should never happen with CoinGecko free tier, but safety first
            logger.error("❌ CRITICAL ERROR: No API providers initialized successfully")
            logger.error("❌ This indicates a serious configuration or network issue")
            raise RuntimeError("Failed to initialize any API providers - check configuration and network connectivity")
        
        # Log comprehensive initialization summary
        logger.info(f"🚀 API MANAGER: Initialized with {len(available_providers)} provider(s)")
        for provider_name, success, error in initialization_results:
            if success:
                capabilities = self.provider_status[provider_name]['method_capabilities']
                specializations = self.provider_status[provider_name]['specializations']
                logger.info(f"  ✅ {provider_name}: Ready")
                logger.info(f"    📋 Specializations: {', '.join(specializations)}")
                logger.info(f"    🔧 Capabilities: {', '.join(capabilities.keys())}")
            else:
                logger.info(f"  ⚠️ {provider_name}: Failed ({error})")
        
        # Set initial provider to first available for backward compatibility
        self.last_provider = available_providers[0]
        logger.info(f"🎯 INITIAL PROVIDER: {self.last_provider}")
        
        # ================================================================
        # 📊 LOAD BALANCING STRATEGY LOGGING
        # ================================================================
        
        # Log comprehensive load balancing strategy
        logger.info("📊 CENTRALIZED ROUTING STRATEGY:")
        for task_type, preferred_provider in self.provider_specialization.items():
            if preferred_provider in available_providers:
                logger.info(f"  📋 {task_type}: {preferred_provider} ✅")
            else:
                fallback = available_providers[0] if available_providers else 'none'
                logger.info(f"  📋 {task_type}: {preferred_provider} (unavailable) → fallback: {fallback}")
        
        # Log load balancing capabilities
        if len(available_providers) > 1:
            logger.info("📊 LOAD BALANCING: ENABLED - Intelligent dual-provider routing active")
            logger.info("📊 RATE LIMIT PREVENTION: Multi-provider distribution available")
            logger.info("📊 REDUNDANCY: Dual-provider fallback protection enabled")
        else:
            logger.info("📊 LOAD BALANCING: LIMITED - Single provider mode")
            logger.info(f"📊 ACTIVE PROVIDER: {available_providers[0]} (solo operation)")
        
        # ================================================================
        # 🔧 ADVANCED CONFIGURATION AND OPTIMIZATION
        # ================================================================
        
        # Cache optimization settings
        self.cache_optimization = {
            'aggressive_caching': len(available_providers) == 1,  # More aggressive if single provider
            'cache_duration_multiplier': 2.0 if len(available_providers) == 1 else 1.0,
            'rate_limit_buffer': 10 if len(available_providers) == 1 else 5,
            'fallback_cache_enabled': True
        }
        
        # Request routing optimization
        self.routing_optimization = {
            'prefer_cached_requests': True,
            'smart_provider_switching': len(available_providers) > 1,
            'automatic_load_balancing': len(available_providers) > 1,
            'quota_aware_routing': True
        }
        
        # Performance monitoring settings
        self.monitoring_config = {
            'track_response_times': True,
            'track_cache_effectiveness': True,
            'track_routing_decisions': True,
            'detailed_failure_analysis': True,
            'performance_alerts': True
        }
        
        logger.info("🔧 OPTIMIZATION CONFIG: Advanced routing and caching configured")
        logger.info(f"⚡ Cache Optimization: {'Aggressive' if self.cache_optimization['aggressive_caching'] else 'Balanced'}")
        logger.info(f"🎯 Routing Strategy: {'Dynamic' if self.routing_optimization['smart_provider_switching'] else 'Static'}")
        
        # ================================================================
        # 🚨 SYSTEM HEALTH VALIDATION
        # ================================================================
        
        # Perform initial health checks
        self._perform_initial_health_checks()
        
        # Log successful initialization
        logger.info("🎉 API MANAGER INITIALIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✅ Providers Available: {len(available_providers)}")
        logger.info(f"✅ Centralized Routing: ACTIVE")
        logger.info(f"✅ Recursion Prevention: ACTIVE")
        logger.info(f"✅ Cache Diagnostics: ACTIVE")
        logger.info(f"✅ Auto-storage: {'ACTIVE' if self.db else 'DISABLED'}")
        logger.info("=" * 70)
    
    def _perform_initial_health_checks(self):
        """Perform initial health checks on all providers"""
        try:
            logger.info("🔍 INITIAL HEALTH CHECKS: Starting provider validation")
            
            for provider_name, provider in self.providers.items():
                if not self.provider_status[provider_name]['available']:
                    continue
                
                try:
                    # Check basic provider health
                    if hasattr(provider, 'quota_tracker') and provider.quota_tracker:
                        quota_status = provider.quota_tracker.get_quota_status()
                        remaining = quota_status.get('daily_remaining', 'unknown')
                        logger.info(f"  📊 {provider_name}: {remaining} requests remaining")
                    
                    # Check cache functionality
                    if hasattr(provider, '_cache') and provider._cache is not None:
                        cache_size = len(provider._cache)
                        logger.info(f"  🗄️ {provider_name}: Cache ready ({cache_size} entries)")
                    
                    # Validate method availability
                    required_methods = ['get_market_data']
                    missing_methods = [method for method in required_methods if not hasattr(provider, method)]
                    if missing_methods:
                        logger.warning(f"  ⚠️ {provider_name}: Missing methods: {missing_methods}")
                    else:
                        logger.info(f"  ✅ {provider_name}: All required methods available")
                        
                except Exception as health_check_error:
                    logger.warning(f"  ⚠️ {provider_name}: Health check warning: {health_check_error}")
            
            logger.info("🔍 INITIAL HEALTH CHECKS: Completed")
            
        except Exception as e:
            logger.warning(f"⚠️ INITIAL HEALTH CHECKS: Error during validation: {e}")

    # ================================================================
    # 🎯 CORE ROUTING METHODS (PART 2 PREVIEW)
    # ================================================================
    # These methods will be implemented in Part 2:
    # - _detect_request_type()
    # - _select_provider_for_request() 
    # - _execute_request_with_provider()
    # - _execute_with_cache_strategy()
    # - _execute_with_direct_strategy()
    # - _execute_fallback_provider()
    # - Cache failure diagnostic methods
    # - Recursion prevention methods

    # ================================================================
    # 🎯 PART 2: CENTRALIZED ROUTING AND CACHE DIAGNOSTICS
    # ================================================================
    # CORE ROUTING METHODS WITH RECURSION PREVENTION AND CACHE FAILURE ANALYSIS
    
    def _detect_request_type(self, params: Optional[Dict[str, Any]] = None, 
                            timeframe: str = "24h",
                            priority_tokens: Optional[List[str]] = None,
                            include_price_history: bool = False) -> str:
        """
        🎯 INTELLIGENT REQUEST TYPE DETECTION FOR CENTRALIZED ROUTING
        
        Analyzes request characteristics to determine optimal provider routing
        
        Args:
            params: Request parameters
            timeframe: Analysis timeframe ("1h", "24h", "7d")
            priority_tokens: List of priority tokens
            include_price_history: Whether price history is requested
            
        Returns:
            Request type for centralized routing decisions
        """
        # Analyze parameters for bulk data indicators FIRST
        if params:
            per_page = params.get('per_page', 0)
            limit = params.get('limit', 0)
            
            # Large requests are bulk operations
            if per_page >= 25 or limit >= 25:
                logger.debug(f"📊 REQUEST TYPE DETECTION: bulk_data (per_page={per_page}, limit={limit})")
                return 'bulk_data'
            
            # Check for specific token IDs (individual requests)
            ids = params.get('ids', '')
            if ids and ',' not in str(ids):
                logger.debug(f"📊 REQUEST TYPE DETECTION: individual_tokens (single id: {ids})")
                return 'individual_tokens'
        
        # 7-day historical data requests (specific routing)
        if include_price_history and timeframe == "7d":
            logger.debug("📊 REQUEST TYPE DETECTION: 7d_historical_data (7-day price history requested)")
            return '7d_historical_data'
        
        # Other historical data requests
        if include_price_history:
            logger.debug("📊 REQUEST TYPE DETECTION: historical_data (price history requested)")
            return 'historical_data'
        
        # Priority tokens indicate focused requests
        if priority_tokens and len(priority_tokens) <= 5:
            logger.debug(f"📊 REQUEST TYPE DETECTION: individual_tokens (priority tokens: {len(priority_tokens)})")
            return 'individual_tokens'
        
        # Default to real-time market overview for standard requests
        logger.debug("📊 REQUEST TYPE DETECTION: real_time (default market overview)")
        return 'real_time'
    
    def _select_provider_for_request(self, request_type: str) -> str:
        """
        🎯 CENTRALIZED PROVIDER SELECTION - SINGLE SOURCE OF TRUTH
        
        All routing decisions flow through provider_specialization configuration
        
        Args:
            request_type: Type of request (bulk_data, historical_data, real_time, etc.)
            
        Returns:
            Provider name to use for this request
        """
        # Get preferred provider from centralized configuration
        preferred_provider = self.provider_specialization.get(request_type, 'coingecko')
        
        # Check if preferred provider is available
        if (preferred_provider in self.providers and 
            self.provider_status.get(preferred_provider, {}).get('available', False)):
            logger.debug(f"📊 CENTRALIZED ROUTING: {request_type} → {preferred_provider} (specialized)")
            return preferred_provider
        
        # Fallback to any available provider
        available_providers = [name for name, status in self.provider_status.items() 
                            if status.get('available', False)]
        
        if available_providers:
            fallback_provider = available_providers[0]
            logger.debug(f"📊 CENTRALIZED ROUTING: {request_type} → {fallback_provider} (fallback)")
            logger.warning(f"⚠️ ROUTING FALLBACK: Preferred provider {preferred_provider} unavailable")
            return fallback_provider
        
        # This should never happen with robust initialization
        logger.error("❌ CRITICAL ROUTING ERROR: No providers available")
        return 'coingecko'  # Ultimate fallback
    
    # ================================================================
    # 🛡️ RECURSION PREVENTION SYSTEM
    # ================================================================
    
    def _check_recursion_prevention(self, method_name: str) -> Tuple[bool, str]:
        """
        🛡️ THREAD-SAFE RECURSION PREVENTION CHECK
        
        Prevents circular calls between methods using thread identification
        
        Args:
            method_name: Name of method being called
            
        Returns:
            Tuple of (can_proceed, request_key)
        """
        thread_id = threading.get_ident()
        request_key = f"{thread_id}_{method_name}"
        
        with self._lock:
            # Check if this exact request is already active
            if request_key in self._active_requests:
                logger.error(f"🚨 RECURSION DETECTED: {method_name} called recursively in thread {thread_id}")
                logger.error(f"🚨 Active requests: {self._active_requests}")
                logger.error("🚨 This indicates a circular call pattern - investigate cache/routing logic")
                return False, request_key
            
            # Check recursion depth for this thread
            if thread_id not in self._request_depth:
                self._request_depth[thread_id] = 0
            
            if self._request_depth[thread_id] >= self.MAX_RECURSION_DEPTH:
                logger.error(f"🚨 RECURSION DEPTH EXCEEDED: Thread {thread_id} depth {self._request_depth[thread_id]}")
                return False, request_key
            
            # Register this request
            self._active_requests.add(request_key)
            self._request_depth[thread_id] += 1
            
            logger.debug(f"🛡️ RECURSION CHECK: {method_name} registered (depth: {self._request_depth[thread_id]})")
            return True, request_key
    
    def _release_recursion_lock(self, request_key: str):
        """
        🛡️ RELEASE RECURSION PREVENTION LOCK
        
        Args:
            request_key: Request key to release
        """
        thread_id = threading.get_ident()
        
        with self._lock:
            self._active_requests.discard(request_key)
            if thread_id in self._request_depth:
                self._request_depth[thread_id] = max(0, self._request_depth[thread_id] - 1)
            
            logger.debug(f"🛡️ RECURSION RELEASE: {request_key} released")
    
    # ================================================================
    # 🔍 COMPREHENSIVE CACHE FAILURE DIAGNOSTICS
    # ================================================================
    
    def _diagnose_cache_failure(self, provider, provider_name: str, endpoint: str, 
                              params: Dict[str, Any], request_type: str):
        """
        🔍 COMPREHENSIVE CACHE FAILURE DIAGNOSTICS WITH ROOT CAUSE ANALYSIS
        
        Performs detailed analysis when cache fails to identify exact cause
        
        Args:
            provider: Provider instance that failed
            provider_name: Name of the provider
            endpoint: API endpoint that was called
            params: Parameters used in the request
            request_type: Type of request that failed
        """
        failure_timestamp = time.time()
        failure_id = f"{provider_name}_{endpoint}_{int(failure_timestamp)}"
        
        logger.error(f"🚨 CACHE FAILURE DETECTED: {failure_id}")
        
        # ================================================================
        # 📊 COLLECT COMPREHENSIVE DIAGNOSTIC DATA
        # ================================================================
        
        diagnostics = {
            'failure_id': failure_id,
            'timestamp': failure_timestamp,
            'provider': provider_name,
            'endpoint': endpoint,
            'request_type': request_type,
            'params': str(params)[:500],  # Truncate for logging
            'provider_available': self.provider_status.get(provider_name, {}).get('available', False),
            'thread_id': threading.get_ident(),
            'active_requests': len(self._active_requests),
            'request_depth': self._request_depth.get(threading.get_ident(), 0)
        }
        
        # ================================================================
        # 🔍 ANALYZE CACHE STATUS
        # ================================================================
        
        try:
            if hasattr(provider, '_cache'):
                if provider._cache is not None:
                    cache_size = len(provider._cache)
                    cache_enabled = True
                    # Check if cache has any entries
                    cache_has_data = cache_size > 0
                    diagnostics['cache_status'] = {
                        'enabled': cache_enabled,
                        'size': cache_size,
                        'has_data': cache_has_data,
                        'type': type(provider._cache).__name__
                    }
                else:
                    diagnostics['cache_status'] = {
                        'enabled': False,
                        'size': 0,
                        'has_data': False,
                        'type': 'None'
                    }
            else:
                diagnostics['cache_status'] = {
                    'enabled': False,
                    'size': 0,
                    'has_data': False,
                    'type': 'missing_attribute'
                }
        except Exception as cache_error:
            diagnostics['cache_status'] = {
                'enabled': False,
                'error': str(cache_error),
                'type': 'error_accessing'
            }
        
        # ================================================================
        # 📊 ANALYZE QUOTA STATUS
        # ================================================================
        
        try:
            if hasattr(provider, 'quota_tracker') and provider.quota_tracker:
                quota_status = provider.quota_tracker.get_quota_status()
                diagnostics['quota_analysis'] = {
                    'daily_remaining': quota_status.get('daily_remaining', 'unknown'),
                    'daily_used': quota_status.get('daily_used', 'unknown'),
                    'daily_limit': quota_status.get('daily_limit', 'unknown'),
                    'success_rate': quota_status.get('success_rate_1h', 'unknown'),
                    'recent_failures': quota_status.get('recent_failures', 'unknown')
                }
            else:
                diagnostics['quota_analysis'] = {
                    'status': 'no_quota_tracker',
                    'tracking_available': False
                }
        except Exception as quota_error:
            diagnostics['quota_analysis'] = {
                'status': 'quota_error',
                'error': str(quota_error)
            }
        
        # ================================================================
        # 🔧 ANALYZE PROVIDER CAPABILITIES
        # ================================================================
        
        diagnostics['provider_analysis'] = {
            'has_get_with_cache': hasattr(provider, 'get_with_cache'),
            'has_get_market_data': hasattr(provider, 'get_market_data'),
            'initialization_error': self.provider_status.get(provider_name, {}).get('initialization_error'),
            'specializations': self.provider_status.get(provider_name, {}).get('specializations', []),
            'method_capabilities': self.provider_status.get(provider_name, {}).get('method_capabilities', {})
        }
        
        # ================================================================
        # 🎯 ROOT CAUSE IDENTIFICATION
        # ================================================================
        
        root_cause = self._identify_cache_root_cause(diagnostics)
        diagnostics['root_cause'] = root_cause
        
        # ================================================================
        # 📈 STORE FAILURE FOR PATTERN ANALYSIS
        # ================================================================
        
        # Store in comprehensive failure tracker
        self.cache_failure_tracker['failures'].append(diagnostics)
        
        # Track root cause frequency
        if root_cause not in self.cache_failure_tracker['root_causes']:
            self.cache_failure_tracker['root_causes'][root_cause] = 0
        self.cache_failure_tracker['root_causes'][root_cause] += 1
        
        # Track provider-specific patterns
        provider_pattern_key = f"{provider_name}_{root_cause}"
        if provider_pattern_key not in self.cache_failure_tracker['provider_patterns']:
            self.cache_failure_tracker['provider_patterns'][provider_pattern_key] = 0
        self.cache_failure_tracker['provider_patterns'][provider_pattern_key] += 1
        
        # Track endpoint-specific patterns
        endpoint_pattern_key = f"{endpoint}_{root_cause}"
        if endpoint_pattern_key not in self.cache_failure_tracker['endpoint_patterns']:
            self.cache_failure_tracker['endpoint_patterns'][endpoint_pattern_key] = 0
        self.cache_failure_tracker['endpoint_patterns'][endpoint_pattern_key] += 1
        
        # Track time-based patterns (hour of day)
        hour_of_day = datetime.fromtimestamp(failure_timestamp).hour
        time_pattern_key = f"hour_{hour_of_day}_{root_cause}"
        if time_pattern_key not in self.cache_failure_tracker['time_patterns']:
            self.cache_failure_tracker['time_patterns'][time_pattern_key] = 0
        self.cache_failure_tracker['time_patterns'][time_pattern_key] += 1
        
        # ================================================================
        # 📋 COMPREHENSIVE FAILURE LOGGING
        # ================================================================
        
        logger.error(f"🔍 CACHE FAILURE ANALYSIS: {failure_id}")
        logger.error(f"   Provider: {provider_name} (available: {diagnostics['provider_available']})")
        logger.error(f"   Endpoint: {endpoint}")
        logger.error(f"   Request Type: {request_type}")
        logger.error(f"   Thread Info: ID={diagnostics['thread_id']}, Depth={diagnostics['request_depth']}")
        logger.error(f"   Cache Status: {diagnostics['cache_status']}")
        logger.error(f"   Quota Analysis: {diagnostics['quota_analysis']}")
        logger.error(f"   Provider Analysis: {diagnostics['provider_analysis']}")
        logger.error(f"   ROOT CAUSE: {root_cause}")
        
        # ================================================================
        # 🚨 IMMEDIATE ACTION RECOMMENDATIONS
        # ================================================================
        
        if root_cause == 'quota_exhausted':
            logger.error("🚨 IMMEDIATE ACTION: Switch provider - quota exhausted")
            logger.error("💡 RECOMMENDATION: Enable aggressive caching or add API key")
        elif root_cause == 'cache_disabled':
            logger.error("🚨 IMMEDIATE ACTION: Cache disabled - check cache initialization")
            logger.error("💡 RECOMMENDATION: Verify provider cache configuration")
        elif root_cause == 'provider_unavailable':
            logger.error("🚨 IMMEDIATE ACTION: Provider unavailable - check API key and initialization")
            logger.error("💡 RECOMMENDATION: Verify environment variables and network connectivity")
        elif root_cause == 'recursion_detected':
            logger.error("🚨 IMMEDIATE ACTION: Recursion detected - circular call pattern")
            logger.error("💡 RECOMMENDATION: Review routing logic and method call patterns")
        elif root_cause == 'method_missing':
            logger.error("🚨 IMMEDIATE ACTION: Required method missing from provider")
            logger.error("💡 RECOMMENDATION: Update provider implementation or use fallback")
        else:
            logger.error("🚨 IMMEDIATE ACTION: Unknown cache failure - needs investigation")
            logger.error("💡 RECOMMENDATION: Enable debug logging and monitor failure patterns")
    
    def _diagnose_cache_exception(self, provider, provider_name: str, endpoint: str, 
                                params: Dict[str, Any], request_type: str, exception: Exception):
        """
        🔍 CACHE EXCEPTION DIAGNOSTICS WITH PATTERN ANALYSIS
        
        Args:
            provider: Provider instance that raised exception
            provider_name: Name of the provider
            endpoint: API endpoint that was called
            params: Parameters used in the request
            request_type: Type of request that failed
            exception: The exception that was raised
        """
        exception_id = f"{provider_name}_{type(exception).__name__}_{int(time.time())}"
        
        logger.error(f"🚨 CACHE EXCEPTION: {exception_id}")
        logger.error(f"   Provider: {provider_name}")
        logger.error(f"   Endpoint: {endpoint}")
        logger.error(f"   Request Type: {request_type}")
        logger.error(f"   Exception Type: {type(exception).__name__}")
        logger.error(f"   Exception Message: {str(exception)}")
        logger.error(f"   Thread ID: {threading.get_ident()}")
        logger.error(f"   Params: {str(params)[:500]}")
        
        # ================================================================
        # 📊 TRACK EXCEPTION PATTERNS
        # ================================================================
        
        exception_key = f"{provider_name}_{type(exception).__name__}"
        if exception_key not in self.cache_failure_tracker['exceptions']:
            self.cache_failure_tracker['exceptions'][exception_key] = []
        
        self.cache_failure_tracker['exceptions'][exception_key].append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'exception_message': str(exception),
            'request_type': request_type,
            'thread_id': threading.get_ident(),
            'exception_traceback': traceback.format_exc()
        })
        
        # ================================================================
        # 🎯 SPECIFIC EXCEPTION HANDLING
        # ================================================================
        
        exception_type_name = type(exception).__name__
        
        if 'RecursionError' in exception_type_name or 'recursion' in str(exception).lower():
            logger.error("🚨 RECURSION ERROR DETECTED - Confirms circular call pattern!")
            logger.error("🚨 Check for calls between get_market_data and get_with_cache")
            logger.error("🚨 Review centralized routing to prevent circular dependencies")
        elif 'ConnectionError' in exception_type_name or 'connection' in str(exception).lower():
            logger.error("🚨 CONNECTION ERROR - Check network/API endpoint availability")
            logger.error("💡 Consider implementing connection retry logic")
        elif 'timeout' in str(exception).lower() or 'TimeoutError' in exception_type_name:
            logger.error("🚨 TIMEOUT ERROR - API response too slow")
            logger.error("💡 Consider increasing timeout values or implementing async requests")
        elif 'KeyError' in exception_type_name or 'AttributeError' in exception_type_name:
            logger.error("🚨 DATA STRUCTURE ERROR - Unexpected API response format")
            logger.error("💡 Implement more robust data validation and error handling")
        elif 'HTTPError' in exception_type_name:
            logger.error("🚨 HTTP ERROR - API endpoint returned error status")
            logger.error("💡 Check API key validity and endpoint availability")
        else:
            logger.error("🚨 UNKNOWN EXCEPTION TYPE - Needs specific handling")
            logger.error("💡 Add specific exception handling for this error type")
        
        # Log full traceback for debugging
        logger.error(f"🔍 FULL TRACEBACK:\n{traceback.format_exc()}")
    
    def _identify_cache_root_cause(self, diagnostics: Dict[str, Any]) -> str:
        """
        🔍 IDENTIFY ROOT CAUSE OF CACHE FAILURE
        
        Analyzes diagnostic data to determine the exact cause of cache failure
        
        Args:
            diagnostics: Comprehensive diagnostic data
            
        Returns:
            Root cause identifier
        """
        # Check provider availability first
        if not diagnostics.get('provider_available', True):
            return 'provider_unavailable'
        
        # Check for recursion issues
        if diagnostics.get('request_depth', 0) >= self.MAX_RECURSION_DEPTH:
            return 'recursion_detected'
        
        # Check quota status
        quota_analysis = diagnostics.get('quota_analysis', {})
        if isinstance(quota_analysis, dict):
            remaining = quota_analysis.get('daily_remaining')
            if isinstance(remaining, (int, float)) and remaining <= 0:
                return 'quota_exhausted'
            elif remaining == 0:  # Handle string '0'
                return 'quota_exhausted'
        
        # Check cache status
        cache_status = diagnostics.get('cache_status', {})
        if isinstance(cache_status, dict):
            if not cache_status.get('enabled', True):
                return 'cache_disabled'
            elif cache_status.get('type') == 'None':
                return 'cache_not_initialized'
            elif 'error' in cache_status:
                return 'cache_access_error'
        
        # Check provider capabilities
        provider_analysis = diagnostics.get('provider_analysis', {})
        if isinstance(provider_analysis, dict):
            if not provider_analysis.get('has_get_with_cache', True):
                return 'method_missing'
            elif provider_analysis.get('initialization_error'):
                return 'provider_initialization_error'
        
        # Check for quota tracker issues
        if quota_analysis.get('status') == 'quota_error':
            return 'quota_tracker_error'
        elif quota_analysis.get('status') == 'no_quota_tracker':
            return 'quota_tracker_missing'
        
        # Default to unknown failure
        return 'unknown_cache_failure'
    
    # ================================================================
    # 🎯 REQUEST EXECUTION METHODS
    # ================================================================
    
    def _execute_request_with_provider(self, provider_name: str, params: Optional[Dict[str, Any]], 
                                     timeframe: str, priority_tokens: Optional[List[str]], 
                                     include_price_history: bool, request_type: str) -> Optional[Any]:
        """
        🎯 EXECUTE REQUEST WITH SPECIFIC PROVIDER USING UNIFIED INTERFACE
        
        Routes to appropriate strategy based on request type
        
        Args:
            provider_name: Name of provider to use
            params: Request parameters
            timeframe: Analysis timeframe
            priority_tokens: Priority tokens list
            include_price_history: Whether to include price history
            request_type: Type of request for strategy selection
            
        Returns:
            Provider response data or None on failure
        """
        provider = self.providers.get(provider_name)
        if not provider:
            logger.error(f"❌ PROVIDER NOT FOUND: {provider_name}")
            return None
            
        if not self.provider_status.get(provider_name, {}).get('available', False):
            logger.error(f"❌ PROVIDER UNAVAILABLE: {provider_name}")
            return None
        
        # Start performance tracking
        start_time = time.time()
        
        # Route to appropriate strategy based on request type
        try:
            if request_type in ['individual_tokens', 'real_time']:
                # High-frequency requests → use cache strategy
                result = self._execute_with_cache_strategy(provider, provider_name, params, request_type)
            else:
                # Bulk/historical requests → use direct method strategy
                result = self._execute_with_direct_strategy(
                    provider, provider_name, params, timeframe, 
                    priority_tokens, include_price_history, request_type
                )
            
            # Track performance metrics
            duration = time.time() - start_time
            success = result is not None
            
            self._record_performance_metrics(provider_name, request_type, duration, success)
            
            return result
            
        except Exception as execution_error:
            duration = time.time() - start_time
            logger.error(f"❌ REQUEST EXECUTION ERROR: {provider_name} → {str(execution_error)}")
            self._record_performance_metrics(provider_name, request_type, duration, False)
            return None
    
    def _execute_with_cache_strategy(self, provider, provider_name: str, 
                                   params: Optional[Dict[str, Any]], request_type: str) -> Optional[Any]:
        """
        🔍 HIGH-FREQUENCY REQUEST STRATEGY: Use cache with fail-fast diagnostics
        
        Args:
            provider: Provider instance
            provider_name: Provider name
            params: Request parameters
            request_type: Type of request
            
        Returns:
            Cached or fresh data, None on failure
        """
        # Determine appropriate endpoint for provider
        if provider_name == 'coingecko':
            endpoint = "coins/markets"
        elif provider_name == 'coinmarketcap':
            endpoint = "cryptocurrency/listings/latest"
        else:
            endpoint = "markets"  # Generic fallback

        if provider_name == 'coinmarketcap' and params:
            params = self._translate_to_coinmarketcap(params, "24h", request_type)    
            
        try:
            logger.debug(f"🔍 CACHE STRATEGY: {provider_name} → {endpoint} for {request_type}")
            
            # Verify provider has cache method
            if not hasattr(provider, 'get_with_cache'):
                logger.error(f"❌ CACHE METHOD MISSING: {provider_name}.get_with_cache not available")
                self._diagnose_cache_failure(provider, provider_name, endpoint, params or {}, request_type)
                return None
            
            # Execute cache request
            result = provider.get_with_cache(endpoint, params)
            
            if result is not None:
                logger.debug(f"✅ CACHE SUCCESS: {provider_name} returned data via cache")
                self._record_cache_hit(provider_name, request_type)
                return result
            else:
                logger.warning(f"⚠️ CACHE MISS: {provider_name} cache returned None")
                self._record_cache_miss(provider_name, request_type)
                # Perform comprehensive cache failure diagnosis
                self._diagnose_cache_failure(provider, provider_name, endpoint, params or {}, request_type)
                return None
                
        except Exception as cache_error:
            logger.error(f"❌ CACHE EXCEPTION: {provider_name} → {str(cache_error)}")
            self._record_cache_miss(provider_name, request_type)
            # Perform comprehensive cache exception diagnosis
            self._diagnose_cache_exception(provider, provider_name, endpoint, params or {}, request_type, cache_error)
            return None
    
    def _execute_with_direct_strategy(self, provider, provider_name: str, 
                                    params: Optional[Dict[str, Any]], timeframe: str,
                                    priority_tokens: Optional[List[str]], 
                                    include_price_history: bool, request_type: str) -> Optional[Any]:
        """
        Enhanced bulk/historical request strategy with parameter translation.
        
        Handles provider-specific parameter translation to ensure API compatibility
        across different cryptocurrency data providers.
        
        Args:
            provider: Provider instance
            provider_name: Provider name ('coinmarketcap' or 'coingecko')
            params: Request parameters (will be translated to provider format)
            timeframe: Analysis timeframe ("1h", "24h", "7d")
            priority_tokens: Optional list of priority tokens
            include_price_history: Whether to include historical price data
            request_type: Type of request for optimization
            
        Returns:
            Provider response data or None on failure
            
        Raises:
            None - All exceptions are caught and logged
        """
        # Initialize variables to prevent unbound errors
        translated_params = None
        
        try:
            logger.debug(f"Executing direct strategy: {provider_name} -> {request_type}")
            
            # Early validation
            if not hasattr(provider, 'get_market_data'):
                logger.error(f"Provider {provider_name} missing required method: get_market_data")
                return None
            
            # ================================================================
            # Parameter Translation Layer
            # ================================================================
            
            translated_params = self._translate_parameters(
                params=params,
                provider_name=provider_name,
                timeframe=timeframe,
                request_type=request_type
            )
            
            if translated_params is None:
                logger.error(f"Parameter translation failed for provider: {provider_name}")
                return None
            
            logger.debug(f"Parameters translated successfully for {provider_name}")
            
            # ================================================================
            # Execute Provider Request
            # ================================================================
            
            result = provider.get_market_data(
                params=translated_params,
                timeframe=timeframe,
                priority_tokens=priority_tokens,
                include_price_history=include_price_history
            )
            
            if result is not None:
                logger.debug(f"Provider {provider_name} returned successful response")
                return result
            else:
                logger.warning(f"Provider {provider_name} returned empty response")
                return None
                
        except Exception as exc:
            # Structured error handling with context
            error_context = {
                'provider': provider_name,
                'request_type': request_type,
                'params_used': translated_params,
                'error_type': type(exc).__name__,
                'error_message': str(exc)
            }
            
            self._handle_provider_error(exc, error_context)
            return None

    def _translate_parameters(self, params: Optional[Dict[str, Any]], 
                            provider_name: str, timeframe: str, 
                            request_type: str) -> Optional[Dict[str, Any]]:
        """
        Translate request parameters to provider-specific format.
        
        Args:
            params: Original parameters (typically CoinGecko format)
            provider_name: Target provider name
            timeframe: Request timeframe
            request_type: Type of request
            
        Returns:
            Translated parameters or None on failure
        """
        try:
            if provider_name == 'coinmarketcap':
                return self._translate_to_coinmarketcap(params, timeframe, request_type)
            elif provider_name == 'coingecko':
                return self._translate_to_coingecko(params, timeframe, request_type)
            else:
                logger.warning(f"Unknown provider: {provider_name}, using parameters as-is")
                return params
                
        except Exception as exc:
            logger.error(f"Parameter translation error for {provider_name}: {exc}")
            return None

    def _translate_to_coinmarketcap(self, params: Optional[Dict[str, Any]], 
                                timeframe: str, request_type: str) -> Dict[str, Any]:
        """
        Translate parameters to CoinMarketCap API format.
        
        Args:
            params: Original parameters
            timeframe: Request timeframe
            request_type: Type of request
            
        Returns:
            CoinMarketCap-compatible parameters
        """
        # Extract key parameters with defaults
        limit = 100
        if params:
            limit = min(params.get('per_page', 100), params.get('limit', 100), 100)
        
        # Base CoinMarketCap parameters
        cmc_params = {
            'start': '1',
            'limit': str(limit),
            'convert': 'USD',
            'sort': 'market_cap',
            'sort_dir': 'desc',
            'cryptocurrency_type': 'all'
        }
        
        # Add auxiliary data based on request type and timeframe
        aux_fields = [
            'num_market_pairs', 'cmc_rank', 'date_added', 'tags', 
            'platform', 'max_supply', 'circulating_supply', 'total_supply'
        ]
        
        if request_type in ['historical_data', '7d_historical_data']:
            cmc_params['aux'] = ','.join(aux_fields)
        
        logger.debug(f"CoinMarketCap parameters: {cmc_params}")
        return cmc_params

    def _translate_to_coingecko(self, params: Optional[Dict[str, Any]], 
                            timeframe: str, request_type: str) -> Dict[str, Any]:
        """
        Validate and enhance parameters for CoinGecko API format.
        
        Args:
            params: Original parameters
            timeframe: Request timeframe  
            request_type: Type of request
            
        Returns:
            CoinGecko-compatible parameters
        """
        if params:
            # Start with provided parameters
            gecko_params = params.copy()
        else:
            # Generate dynamic parameters if none provided
            gecko_params = self._generate_dynamic_params(timeframe)
        
        # Ensure required CoinGecko parameters
        required_defaults = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'sparkline': False,
            'page': 1
        }
        
        for key, default_value in required_defaults.items():
            if key not in gecko_params:
                gecko_params[key] = default_value
        
        logger.debug(f"CoinGecko parameters: {gecko_params}")
        return gecko_params

    def _handle_provider_error(self, exception: Exception, context: Dict[str, Any]) -> None:
        """
        Handle provider errors with structured logging and context.
        
        Args:
            exception: The exception that occurred
            context: Error context dictionary
        """
        provider = context['provider']
        error_msg = str(exception)
        
        # Log basic error information
        logger.error(f"Provider error: {provider} -> {context['error_type']}: {context['error_message']}")
        
        # Categorize and handle specific error types
        if any(code in error_msg for code in ['404', 'Not Found']):
            logger.error(f"Endpoint not found for {provider} - check parameter translation")
            logger.error(f"Parameters used: {context['params_used']}")
            
        elif any(code in error_msg for code in ['400', 'Bad Request']):
            logger.error(f"Parameter validation failed for {provider}")
            logger.error(f"Parameters used: {context['params_used']}")
            
        elif any(code in error_msg for code in ['401', 'Unauthorized']):
            logger.error(f"Authentication failed for {provider} - check API key")
            
        elif any(code in error_msg for code in ['429', 'rate limit']):
            logger.error(f"Rate limit exceeded for {provider}")
            
        elif any(code in error_msg for code in ['500', 'Internal Server Error']):
            logger.error(f"Server error from {provider} - retry may succeed")
            
        else:
            logger.error(f"Unknown error from {provider} - investigate parameter compatibility")
            logger.error(f"Parameters used: {context['params_used']}")
        
        # Log full traceback in debug mode
        logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    # ================================================================
    # 📊 PERFORMANCE TRACKING METHODS
    # ================================================================
    
    def _record_performance_metrics(self, provider_name: str, request_type: str, 
                                   duration: float, success: bool):
        """Record performance metrics for analysis"""
        if provider_name in self.performance_metrics['response_times']:
            self.performance_metrics['response_times'][provider_name].append(duration)
            
            # Keep only last 100 measurements
            if len(self.performance_metrics['response_times'][provider_name]) > 100:
                self.performance_metrics['response_times'][provider_name] = \
                    self.performance_metrics['response_times'][provider_name][-100:]
        
        # Update request statistics
        if provider_name in self.request_stats:
            self.request_stats[provider_name]['total'] += 1
            if request_type in self.request_stats[provider_name]:
                self.request_stats[provider_name][request_type] += 1
        
        # Track routing decisions
        routing_key = f"{request_type}_{provider_name}"
        if routing_key not in self.performance_metrics['routing_decisions']:
            self.performance_metrics['routing_decisions'][routing_key] = 0
        self.performance_metrics['routing_decisions'][routing_key] += 1
    
    def _record_cache_hit(self, provider_name: str, request_type: str):
        """Record cache hit for effectiveness tracking"""
        if provider_name in self.request_stats:
            self.request_stats[provider_name]['cache_hits'] += 1
    
    def _record_cache_miss(self, provider_name: str, request_type: str):
        """Record cache miss for effectiveness tracking"""
        if provider_name in self.request_stats:
            self.request_stats[provider_name]['cache_misses'] += 1

    # ================================================================
    # 🎯 PART 3: MAIN DATA METHODS + UTILITIES + EXPORTS
    # ================================================================
    # MAIN get_market_data METHOD WITH CENTRALIZED ROUTING AND RECURSION PREVENTION
    
    def get_market_data(self, params: Optional[Dict[str, Any]] = None, timeframe: str = "24h", 
                       priority_tokens: Optional[List[str]] = None, include_price_history: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        🚀 MAIN MARKET DATA METHOD WITH CENTRALIZED ROUTING AND RECURSION PREVENTION
        
        Get cryptocurrency market data with sophisticated hybrid approach for advanced trading
        
        🎯 CENTRALIZED ROUTING - ALL REQUESTS FLOW THROUGH provider_specialization
        🛡️ RECURSION PREVENTION - No circular calls between methods
        🔍 CACHE FAILURE DIAGNOSTICS - Root cause analysis for cache issues
        🗄️ AUTO-STORAGE - Automatic database storage with source combination
        
        Args:
            params: Query parameters for the API
            timeframe: Analysis timeframe ("1h", "24h", "7d") - determines price change parameters
            priority_tokens: List of token IDs to fetch detailed price history for
            include_price_history: Whether to fetch price history for priority tokens
        
        Returns:
            List of VALIDATED market data entries with selective price history enhancement
        """
        
        # ================================================================
        # 🛡️ RECURSION PREVENTION CHECK
        # ================================================================
        
        can_proceed, request_key = self._check_recursion_prevention('get_market_data')
        if not can_proceed:
            logger.error("🚨 RECURSION PREVENTED: get_market_data call blocked")
            return None
        
        try:
            return self._get_market_data_internal(params, timeframe, priority_tokens, include_price_history)
        finally:
            self._release_recursion_lock(request_key)
    
    def _get_market_data_internal(self, params: Optional[Dict[str, Any]], timeframe: str, 
                                 priority_tokens: Optional[List[str]], include_price_history: bool) -> Optional[List[Dict[str, Any]]]:
        """
        🎯 INTERNAL MARKET DATA IMPLEMENTATION WITH CENTRALIZED ROUTING
        
        Args:
            params: Query parameters
            timeframe: Analysis timeframe
            priority_tokens: Priority tokens list
            include_price_history: Whether to include price history
            
        Returns:
            Market data list or None on failure
        """
        start_time = time.time()
        
        # Log API Manager coordination
        caller_info = inspect.stack()[2] if len(inspect.stack()) > 2 else None
        is_api_manager_call = caller_info and 'api_manager' in str(caller_info.filename).lower()
        
        if is_api_manager_call:
            logger.info("📊 API MANAGER: Centralized routing request")
        else:
            logger.info("📊 DIRECT: get_market_data called directly")
        
        # ================================================================
        # 🎯 STEP 1: CENTRALIZED ROUTING DECISION (SINGLE SOURCE OF TRUTH)
        # ================================================================
        
        request_type = self._detect_request_type(params, timeframe, priority_tokens, include_price_history)
        selected_provider = self._select_provider_for_request(request_type)
        
        logger.info(f"📊 CENTRALIZED ROUTING: {request_type} → {selected_provider}")
        logger.info(f"📊 Routing Decision: {self.provider_specialization.get(request_type, 'default')} (from config)")
        
        # ================================================================
        # 🎯 STEP 2: DYNAMIC TOKEN SELECTION WITH TOKENMAPPINGMANAGER
        # ================================================================
        
        # Set default params with dynamic token selection if not provided
        if params is None:
            params = self._generate_dynamic_params(timeframe)
        else:
            # Enhance provided params with dynamic tokens if needed
            params = self._enhance_params_with_dynamic_tokens(params, timeframe)
        
        # Log request parameters for coordination
        token_count = len(params.get('ids', '').split(',')) if params.get('ids') else 0
        logger.info(f"📊 Request Parameters: {token_count} tokens, timeframe={timeframe}, per_page={params.get('per_page', 0)}")
        
        # ================================================================
        # 🎯 STEP 3: EXECUTE WITH SELECTED PROVIDER (NO HARDCODED LOGIC)
        # ================================================================
        
        result = self._execute_request_with_provider(
            provider_name=selected_provider,
            params=params,
            timeframe=timeframe,
            priority_tokens=priority_tokens,
            include_price_history=include_price_history,
            request_type=request_type
        )
        
        if result is not None:
            # Success path - add metadata and auto-storage
            enhanced_result = self._enhance_market_data_result(
                result, selected_provider, timeframe, request_type, 
                priority_tokens, include_price_history, start_time
            )
            
            logger.info(f"✅ SUCCESS: {selected_provider} provided {len(enhanced_result)} tokens for {request_type}")
            return enhanced_result
        
        # ================================================================
        # 🎯 STEP 4: FAIL-FAST FALLBACK (NO RECURSION)
        # ================================================================
        
        logger.warning(f"⚠️ PRIMARY PROVIDER FAILED: {selected_provider} for {request_type}")
        return self._execute_fallback_provider(params, timeframe, priority_tokens, include_price_history, request_type, selected_provider, start_time)
    
    def _generate_dynamic_params(self, timeframe: str) -> Dict[str, Any]:
        """
        🎯 GENERATE DYNAMIC PARAMETERS WITH TOKENMAPPINGMANAGER
        
        Args:
            timeframe: Analysis timeframe
            
        Returns:
            Dynamic parameters with current token list
        """
        # Get dynamic token list using TokenMappingManager
        dynamic_token_ids = None
        try:
            token_mapper = TokenMappingManager()
            
            # Get all available tokens from TokenMappingManager (database + hardcoded)
            all_tokens_info = token_mapper.get_all_available_tokens(include_database=True)
            if all_tokens_info and all_tokens_info.get('all_unique_symbols'):
                # Get top 30 symbols by using database lookup for market caps
                available_symbols = all_tokens_info['all_unique_symbols'][:30]  # Limit to 30
                
                # Convert symbols to CoinGecko IDs using TokenMappingManager
                api_ids = []
                for symbol in available_symbols:
                    coingecko_id = token_mapper.symbol_to_coingecko_id(symbol)
                    if coingecko_id:
                        api_ids.append(coingecko_id)
                
                if api_ids:
                    dynamic_token_ids = ",".join(api_ids)
                    logger.info(f"✅ TokenMappingManager: Using {len(api_ids)} dynamic tokens")
                else:
                    logger.warning("⚠️ TokenMappingManager: No CoinGecko IDs found for available tokens")
            else:
                logger.warning("⚠️ TokenMappingManager: No available tokens found")
        except Exception as tmm_error:
            logger.warning(f"⚠️ TokenMappingManager failed: {str(tmm_error)}")
        
        # Fallback to database-driven approach if TokenMappingManager fails
        if not dynamic_token_ids:
            try:
                if self.db:
                    conn, cursor = self.db._get_connection()
                    
                    # Query top 30 tokens by market cap from database with proper timestamp ordering
                    cursor.execute("""
                        SELECT DISTINCT coin_id, MAX(market_cap) as max_market_cap
                        FROM coingecko_market_data 
                        WHERE market_cap > 0 AND coin_id IS NOT NULL
                        GROUP BY coin_id
                        ORDER BY max_market_cap DESC 
                        LIMIT 30
                    """)
                    
                    db_results = cursor.fetchall()
                    if db_results:
                        db_token_ids = [row['coin_id'] for row in db_results if row['coin_id']]
                        if db_token_ids:
                            dynamic_token_ids = ",".join(db_token_ids)
                            logger.info(f"✅ Database fallback: Using {len(db_token_ids)} tokens by market cap")
                
            except Exception as db_error:
                logger.warning(f"⚠️ Database fallback failed: {str(db_error)}")
        
        # Ultimate fallback to hardcoded list if all else fails
        if not dynamic_token_ids:
            fallback_tokens = [
                'bitcoin', 'ethereum', 'binancecoin', 'solana', 'ripple', 
                'polkadot', 'avalanche-2', 'uniswap', 'near', 'aave',
                'filecoin', 'matic-network', 'official-trump', 'kaito'
            ]
            dynamic_token_ids = ",".join(fallback_tokens)
            logger.warning(f"⚠️ ULTIMATE FALLBACK: Using hardcoded token list")
        
        # Determine price change parameters based on timeframe
        if timeframe == "1h":
            price_change_param = "1h,24h"  # Focus on short-term changes
        elif timeframe == "24h":
            price_change_param = "1h,24h,7d"  # Full range of changes
        elif timeframe == "7d":
            price_change_param = "24h,7d,30d"  # Longer-term perspective
        else:
            price_change_param = "1h,24h,7d"  # Default to full range
        
        return {
            "vs_currency": "usd",
            "ids": dynamic_token_ids,
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": price_change_param
        }
    
    def _enhance_params_with_dynamic_tokens(self, params: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """
        🎯 ENHANCE PROVIDED PARAMETERS WITH DYNAMIC TOKENS IF NEEDED
        
        Args:
            params: Existing parameters
            timeframe: Analysis timeframe
            
        Returns:
            Enhanced parameters
        """
        enhanced_params = params.copy()
        
        # If params provided, ensure dynamic tokens if 'ids' not specified
        if 'ids' not in enhanced_params:
            dynamic_params = self._generate_dynamic_params(timeframe)
            enhanced_params['ids'] = dynamic_params['ids']
            logger.info("✅ Applied dynamic token list to provided params")
        
        # CRITICAL FIX: Ensure vs_currency is always present
        if 'vs_currency' not in enhanced_params:
            enhanced_params['vs_currency'] = 'usd'

        # Ensure timeframe-appropriate price change parameters
        if 'price_change_percentage' not in enhanced_params:
            if timeframe == "1h":
                enhanced_params['price_change_percentage'] = "1h,24h"
            elif timeframe == "24h":
                enhanced_params['price_change_percentage'] = "1h,24h,7d"
            elif timeframe == "7d":
                enhanced_params['price_change_percentage'] = "24h,7d,30d"
            else:
                enhanced_params['price_change_percentage'] = "1h,24h,7d"
        
        return enhanced_params
    
    def _execute_fallback_provider(self, params: Optional[Dict[str, Any]], timeframe: str, 
                                  priority_tokens: Optional[List[str]], include_price_history: bool, 
                                  request_type: str, failed_provider: str, start_time: float) -> Optional[List[Dict[str, Any]]]:
        """
        🛡️ FAIL-FAST FALLBACK - NO RECURSION, CONSISTENT INTERFACE
        
        Args:
            params: Request parameters
            timeframe: Analysis timeframe
            priority_tokens: Priority tokens list
            include_price_history: Whether to include price history
            request_type: Type of request
            failed_provider: Provider that failed
            start_time: Original request start time
            
        Returns:
            Fallback result or None
        """
        # Get available providers excluding the failed one
        available_providers = [
            name for name, status in self.provider_status.items() 
            if status.get('available', False) and name != failed_provider
        ]
        
        if not available_providers:
            logger.error("❌ NO FALLBACK PROVIDERS AVAILABLE")
            logger.error(f"❌ Failed provider: {failed_provider}")
            logger.error(f"❌ Available providers: {list(self.provider_status.keys())}")
            return None
        
        fallback_provider = available_providers[0]
        logger.warning(f"🔄 FALLBACK ROUTING: {failed_provider} → {fallback_provider} for {request_type}")
        
        # Track fallback usage
        self.performance_metrics['fallback_usage'][fallback_provider] += 1
        
        # 🎯 USE SAME EXECUTION LOGIC - NO SPECIAL FALLBACK BEHAVIOR
        result = self._execute_request_with_provider(
            provider_name=fallback_provider,
            params=params,
            timeframe=timeframe,
            priority_tokens=priority_tokens,
            include_price_history=include_price_history,
            request_type=request_type
        )
        
        if result is not None:
            enhanced_result = self._enhance_market_data_result(
                result, fallback_provider, timeframe, request_type, 
                priority_tokens, include_price_history, start_time
            )
            logger.info(f"✅ FALLBACK SUCCESS: {fallback_provider} provided {len(enhanced_result)} tokens")
            return enhanced_result
        else:
            logger.error(f"❌ FALLBACK FAILED: {fallback_provider} also returned None")
            logger.error("❌ ALL PROVIDERS EXHAUSTED")
            return None
    
    def _enhance_market_data_result(self, result_data: Any, provider_name: str, timeframe: str, 
                                   request_type: str, priority_tokens: Optional[List[str]], 
                                   include_price_history: bool, start_time: float) -> List[Dict[str, Any]]:
        """
        🎯 ENHANCE MARKET DATA RESULT WITH METADATA AND AUTO-STORAGE
        
        Args:
            result_data: Raw result from provider
            provider_name: Provider that provided the data
            timeframe: Analysis timeframe
            request_type: Type of request
            priority_tokens: Priority tokens list
            include_price_history: Whether price history was included
            start_time: Request start time
            
        Returns:
            Enhanced market data with metadata
        """
        # Ensure result is in list format
        if isinstance(result_data, dict):
            batch_data = [result_data]
        elif isinstance(result_data, list):
            batch_data = result_data
        else:
            logger.error(f"❌ UNEXPECTED RESULT TYPE: {type(result_data)}")
            return []
        
        # ================================================================
        # 🔍 STEP 1: ENHANCE PRIORITY TOKENS WITH DETAILED PRICE HISTORY
        # ================================================================
        
        enhanced_result = []
        
        if include_price_history and priority_tokens:
            logger.info(f"🔍 Enhancing {len(priority_tokens)} priority tokens with detailed price history")
            
            api_calls_made = 0
            for token_data in batch_data:
                if isinstance(token_data, dict):
                    token_id = token_data.get('id', '').lower()
                    
                    # Check if this token is in the priority list
                    if token_id in [t.lower() for t in priority_tokens]:
                        try:
                            logger.debug(f"📊 Priority Enhancement: Processing {token_id}")
                            
                            # Enhanced token with price history (if provider supports it)
                            if hasattr(self.providers.get(provider_name), 'get_price_history'):
                                history_data = self.providers[provider_name].get_price_history(
                                    token_id, days=30, vs_currency="usd"
                                )
                                
                                if history_data and 'prices' in history_data:
                                    token_data['price_history'] = history_data['prices']
                                    token_data['_enhanced_with_history'] = True
                                    api_calls_made += 1
                                    logger.debug(f"📊 Priority Enhancement Success: {token_id} enhanced with {len(history_data['prices'])} price points")
                                else:
                                    logger.warning(f"⚠️ Failed to fetch price history for {token_id}")
                                    token_data['_enhanced_with_history'] = False
                            else:
                                logger.debug(f"📊 Priority Enhancement Skipped: {provider_name} doesn't support price history")
                                token_data['_enhanced_with_history'] = False
                            
                            # Rate limiting between individual calls
                            if api_calls_made < len(priority_tokens):
                                time.sleep(20.0)  # 20 seconds rate limiting
                                
                        except Exception as e:
                            logger.error(f"❌ Error enhancing {token_id}: {str(e)}")
                            token_data['_enhanced_with_history'] = False
                    else:
                        # Not a priority token - no price history enhancement
                        token_data['_enhanced_with_history'] = False
                    
                    enhanced_result.append(token_data)
            
            logger.info(f"🎯 Enhanced {api_calls_made} priority tokens with detailed price history")
        else:
            # No price history enhancement requested
            enhanced_result = batch_data
        
        # ================================================================
        # 🏷️ STEP 2: ADD PROCESSING METADATA
        # ================================================================
        
        duration = time.time() - start_time
        
        # Add metadata to track processing approach and API Manager coordination
        processing_metadata = {
            '_fetch_timeframe': timeframe,
            '_fetch_timestamp': time.time(),
            '_processing_approach': 'centralized_routing',
            '_priority_tokens_enhanced': len(priority_tokens) if priority_tokens else 0,
            '_total_tokens': len(enhanced_result),
            '_includes_price_history': include_price_history,
            '_request_type': request_type,
            '_processing_duration': duration,
            '_routing_method': 'centralized',
            '_data_source': provider_name
        }

        # Add metadata to each token, preserving existing _data_source if present
        for token_data in enhanced_result:
            if isinstance(token_data, dict):
                # Only set _data_source if not already set by handler
                if '_data_source' not in token_data:
                    token_data['_data_source'] = provider_name
                # Add all other metadata
                token_data.update(processing_metadata)

        # ================================================================
        # 🔧 STEP 2.5: COINMARKETCAP PRICE TRANSFORMATION FIX
        # ================================================================
        
        if provider_name == 'coinmarketcap':
            logger.info(f"🔧 FIX: Applying CoinMarketCap price transformation to {len(enhanced_result)} tokens")
            
            for token_data in enhanced_result:
                if isinstance(token_data, dict):
                    # Check if this token needs price transformation
                    if 'current_price' not in token_data or token_data.get('current_price') is None:
                        # Extract price from quote.USD.price structure
                        quote_data = token_data.get('quote', {}).get('USD', {})
                        if quote_data and 'price' in quote_data:
                            extracted_price = quote_data.get('price', 0)
                            
                            # Apply the price transformation
                            token_data['current_price'] = extracted_price
                            
                            # Also add other commonly needed fields for compatibility
                            token_data['market_cap'] = quote_data.get('market_cap', 0)
                            token_data['total_volume'] = quote_data.get('volume_24h', 0)
                            token_data['price_change_percentage_24h'] = quote_data.get('percent_change_24h', 0)
                            token_data['price_change_percentage_1h'] = quote_data.get('percent_change_1h', 0)
                            token_data['price_change_percentage_7d'] = quote_data.get('percent_change_7d', 0)
                            
                            logger.debug(f"🔧 FIX: Transformed {token_data.get('symbol', 'UNKNOWN')}: ${extracted_price}")
                        else:
                            logger.warning(f"🔧 FIX: No quote.USD.price found for {token_data.get('symbol', 'UNKNOWN')}")
            
            logger.info(f"🔧 FIX: CoinMarketCap transformation completed")        
        
        # ================================================================
        # 🗄️ STEP 3: AUTO-STORAGE TO DATABASE
        # ================================================================
        
        if enhanced_result and self.db:
            try:
                if provider_name == 'coinmarketcap':
                    if self.db.store_coinmarketcap_data(enhanced_result):
                        logger.info("✅ CoinMarketCap data stored successfully")
                        if self.db.combine_market_data_sources(hours=24):
                            logger.info("🔄 Market data sources combined successfully")
                elif provider_name == 'coingecko':
                    if self.db.store_coingecko_data(enhanced_result):
                        logger.info("✅ CoinGecko data stored successfully")
                        if self.db.combine_market_data_sources(hours=24):
                            logger.info("🔄 Market data sources combined successfully")
            except Exception as storage_error:
                logger.error(f"❌ Auto-storage error ({provider_name}): {storage_error}")
        
        # ================================================================
        # 📊 STEP 4: LOG COMPREHENSIVE PERFORMANCE METRICS
        # ================================================================

        valid_tokens = [token for token in enhanced_result if isinstance(token, dict) and token.get('current_price')]

        logger.info(f"📊 {provider_name.title()} Performance: ✅ {len(enhanced_result)} tokens in {duration:.3f}s")
        logger.info(f"📊 Data Validation: {len(valid_tokens)}/{len(enhanced_result)} tokens have valid prices")
        logger.info(f"📊 Request Type: {request_type} (routed via centralized configuration)")

        return enhanced_result
    
    # ================================================================
    # 🔍 DATABASE-DRIVEN TOKEN MANAGEMENT
    # ================================================================
    
    def get_tokens_with_recent_data_by_market_cap(self, hours: int = 24, limit: int = 25) -> List[str]:
        """
        Get top tokens by market cap that have recent data in the database
        Enhanced to query CoinGecko and CoinMarketCap tables using TokenMappingManager
        
        Args:
            hours: Number of hours to look back for recent data (default: 24)
            limit: Maximum number of tokens to return (default: 25)
            
        Returns:
            List[str]: Token symbols sorted by market cap (highest first)
        """
        logger.info(f"🔍 Getting top {limit} tokens by market cap with recent data (last {hours}h)")
        
        # Check if database is initialized
        if not self.db:
            try:
                self.db = CryptoDatabase()
                logger.info("✅ Created new database connection")
            except Exception as db_error:
                logger.error(f"❌ Database connection failed: {db_error}")
                return []
        
        try:
            # Try to reconnect if the database is closed
            try:
                conn, cursor = self.db._get_connection()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            except Exception as conn_error:
                if "Cannot operate on a closed database" in str(conn_error) or "database is closed" in str(conn_error):
                    logger.warning(f"⚠️ Database connection closed: {str(conn_error)}")
                    self.db = CryptoDatabase()
                    logger.info("✅ Reconnected to database successfully")
                    conn, cursor = self.db._get_connection()
                else:
                    raise conn_error
            
            # Initialize TokenMappingManager
            token_mapper = None
            try:
                token_mapper = TokenMappingManager()
                logger.debug("✅ TokenMappingManager created for symbol standardization")
            except Exception as token_mapper_error:
                logger.debug(f"⚠️ TokenMappingManager not available: {str(token_mapper_error)}")
                logger.debug("⚠️ Using direct symbol extraction fallback")
            
            # Step 1: Query all three database tables for recent data
            all_tokens_with_market_cap = {}
            tables_queried = 0
            
            # Query CoinGecko table
            try:
                logger.debug(f"Step 1a: Querying coingecko_market_data table...")
                cursor.execute("""
                    SELECT coin_id, symbol, name, current_price, market_cap, timestamp
                    FROM coingecko_market_data 
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                    AND market_cap IS NOT NULL 
                    AND market_cap > 0
                    ORDER BY timestamp DESC
                """, (hours,))
                
                coingecko_results = cursor.fetchall()
                logger.debug(f"📊 CoinGecko table: {len(coingecko_results)} records found")
                
                # Process CoinGecko results
                for row in coingecko_results:
                    # Use TokenMappingManager to get standardized symbol
                    if token_mapper:
                        symbol = token_mapper.coingecko_id_to_symbol(row['coin_id'])
                    else:
                        symbol = row['symbol'].upper() if row['symbol'] else 'UNKNOWN'
                    
                    # Keep the highest market cap for each symbol
                    if symbol not in all_tokens_with_market_cap or row['market_cap'] > all_tokens_with_market_cap[symbol]['market_cap']:
                        all_tokens_with_market_cap[symbol] = {
                            'market_cap': row['market_cap'],
                            'timestamp': row['timestamp'],
                            'source': 'coingecko'
                        }
                
                tables_queried += 1
                logger.debug(f"✅ CoinGecko: {len([k for k, v in all_tokens_with_market_cap.items() if v['source'] == 'coingecko'])} unique tokens")
                
            except Exception as coingecko_error:
                logger.warning(f"⚠️ CoinGecko table query failed: {str(coingecko_error)}")
            
            # Query CoinMarketCap table
            try:
                logger.debug(f"Step 1b: Querying coinmarketcap_market_data table...")
                cursor.execute("""
                    SELECT cmc_id, symbol, slug, name, quote_price, quote_market_cap, timestamp
                    FROM coinmarketcap_market_data 
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                    AND quote_market_cap IS NOT NULL 
                    AND quote_market_cap > 0
                    ORDER BY timestamp DESC
                """, (hours,))
                
                coinmarketcap_results = cursor.fetchall()
                logger.debug(f"📊 CoinMarketCap table: {len(coinmarketcap_results)} records found")
                
                # Process CoinMarketCap results
                for row in coinmarketcap_results:
                    # Use TokenMappingManager to get standardized symbol
                    if token_mapper:
                        symbol = token_mapper.cmc_slug_to_symbol(row['slug'])
                    else:
                        symbol = row['symbol'].upper() if row['symbol'] else 'UNKNOWN'
                    
                    # Keep the highest market cap for each symbol (prefer CoinMarketCap if newer data)
                    if (symbol not in all_tokens_with_market_cap or 
                        row['quote_market_cap'] > all_tokens_with_market_cap[symbol]['market_cap'] or
                        (row['quote_market_cap'] == all_tokens_with_market_cap[symbol]['market_cap'] and 
                        row['timestamp'] > all_tokens_with_market_cap[symbol]['timestamp'])):
                        all_tokens_with_market_cap[symbol] = {
                            'market_cap': row['quote_market_cap'],
                            'timestamp': row['timestamp'],
                            'source': 'coinmarketcap'
                        }
                
                tables_queried += 1
                logger.debug(f"✅ CoinMarketCap: {len([k for k, v in all_tokens_with_market_cap.items() if v['source'] == 'coinmarketcap'])} unique tokens")
                
            except Exception as coinmarketcap_error:
                logger.warning(f"⚠️ CoinMarketCap table query failed: {str(coinmarketcap_error)}")
            
            # Query original market_data table as fallback
            try:
                logger.debug(f"Step 1c: Querying original market_data table...")
                cursor.execute("""
                    SELECT DISTINCT chain
                    FROM market_data 
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                """, (hours,))
                
                recent_tokens = [row['chain'] for row in cursor.fetchall()]
                logger.debug(f"📊 Original market_data table: {len(recent_tokens)} recent tokens")
                
                if recent_tokens:
                    # Get market cap data for these tokens
                    placeholders = ','.join(['?'] * len(recent_tokens))
                    query = f"""
                        SELECT chain, market_cap, timestamp
                        FROM market_data
                        WHERE chain IN ({placeholders})
                        AND timestamp >= datetime('now', '-' || ? || ' hours')
                        AND market_cap > 0
                        ORDER BY timestamp DESC
                    """
                    
                    cursor.execute(query, recent_tokens + [hours])
                    
                    # Process original market_data results
                    for row in cursor.fetchall():
                        # Use TokenMappingManager to standardize symbol
                        if token_mapper:
                            symbol = token_mapper.database_name_to_symbol(row['chain'])
                        else:
                            symbol = row['chain'].upper() if row['chain'] else 'UNKNOWN'
                        
                        # Only add if not already present from newer tables
                        if symbol not in all_tokens_with_market_cap:
                            all_tokens_with_market_cap[symbol] = {
                                'market_cap': row['market_cap'],
                                'timestamp': row['timestamp'],
                                'source': 'market_data'
                            }
                
                tables_queried += 1
                logger.debug(f"✅ Original market_data: {len([k for k, v in all_tokens_with_market_cap.items() if v['source'] == 'market_data'])} unique tokens")
                
            except Exception as market_data_error:
                logger.warning(f"⚠️ Original market_data table query failed: {str(market_data_error)}")
            
            # Step 2: Sort and limit results
            if not all_tokens_with_market_cap:
                logger.warning(f"⚠️ No tokens found with recent data from {tables_queried} tables")
                return []
            
            # Sort by market cap (highest first)
            sorted_tokens = sorted(
                all_tokens_with_market_cap.keys(),
                key=lambda symbol: all_tokens_with_market_cap[symbol]['market_cap'],
                reverse=True
            )
            
            # Limit results
            top_tokens = sorted_tokens[:limit]
            
            # Log source distribution
            source_counts = {}
            for token in top_tokens:
                source = all_tokens_with_market_cap[token]['source']
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logger.info(f"✅ Top {len(top_tokens)} tokens by market cap: {top_tokens[:5]}...")
            logger.info(f"📊 Data sources: {source_counts} from {tables_queried} tables queried")
            
            return top_tokens
            
        except Exception as e:
            logger.error(f"❌ Error getting tokens by market cap: {str(e)}")
            return []
    
    # ================================================================
    # 📊 SYSTEM HEALTH AND MONITORING METHODS
    # ================================================================
    
    def _check_provider_health(self, provider_name: str) -> Dict[str, Any]:
        """Check the health of a specific provider"""
        if provider_name not in self.providers or not self.provider_status[provider_name]['available']:
            return {
                'available': False,
                'ready_for_trading': False,
                'quota_remaining': 0,
                'quota_status': 'UNAVAILABLE'
            }
        
        provider = self.providers[provider_name]
        
        try:
            # Get readiness report
            if provider_name == 'coingecko':
                readiness = validate_coingecko_readiness(provider)
            else:  # coinmarketcap
                readiness = validate_coinmarketcap_readiness(provider)
            
            # Get quota status
            quota_status = provider.quota_tracker.get_quota_status() if hasattr(provider, 'quota_tracker') else {}
            
            # Update provider status
            self.provider_status[provider_name].update({
                'available': True,
                'last_check': time.time(),
                'readiness': readiness,
                'quota_status': quota_status
            })
            
            return {
                'available': True,
                'ready_for_trading': readiness.get('ready_for_trading', False),
                'quota_remaining': quota_status.get('daily_remaining', 0),
                'quota_status': 'HEALTHY' if quota_status.get('daily_remaining', 0) > 50 else 'LIMITED'
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {provider_name}: {str(e)}")
            self.provider_status[provider_name].update({
                'available': False,
                'last_check': time.time(),
                'error': str(e)
            })
            
            return {
                'available': False,
                'ready_for_trading': False,
                'quota_remaining': 0,
                'quota_status': 'ERROR'
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health for all providers"""
        health = {
            'overall_status': 'HEALTHY',
            'providers': {},
            'recommendations': [],
            'active_provider': self.last_provider,
            'available_providers': [],
            'load_balancing_status': 'OPTIMAL',
            'cache_failure_summary': self._get_cache_failure_summary(),
            'performance_summary': self._get_performance_summary()
        }
        
        all_providers_limited = True
        
        for provider_name in self.providers.keys():
            provider_health = self._check_provider_health(provider_name)
            health['providers'][provider_name] = provider_health
            
            if provider_health['available']:
                health['available_providers'].append(provider_name)
                if provider_health['quota_status'] != 'LIMITED':
                    all_providers_limited = False
        
        # Overall status determination
        if not health['available_providers']:
            health['overall_status'] = 'CRITICAL'
            health['load_balancing_status'] = 'DISABLED'
            health['recommendations'].append('All API providers unavailable - check API keys and configuration')
        elif len(health['available_providers']) == 1:
            health['overall_status'] = 'WARNING' 
            health['load_balancing_status'] = 'LIMITED'
            health['recommendations'].append(f'Only {health["available_providers"][0]} available - reduced load balancing capability')
        elif all_providers_limited:
            health['overall_status'] = 'WARNING'
            health['load_balancing_status'] = 'CONSTRAINED'
            health['recommendations'].append('All API providers have limited quota - consider extending cache duration')
        
        return health
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted based on all provider statuses"""
        # Check if any provider is healthy enough for trading
        for provider_name, provider in self.providers.items():
            if not self.provider_status[provider_name]['available']:
                continue
                
            try:
                should_halt, reason = provider.should_halt_trading()
                if not should_halt:
                    # At least one provider is healthy for trading
                    return False, f"Provider {provider_name} is healthy for trading"
            except Exception:
                continue
        
        # If we get here, all providers indicate we should halt trading
        return True, "All API providers indicate trading should be halted"
    
    def force_provider_switch(self, target_provider: Optional[str] = None) -> bool:
        """Force a switch to a different provider"""
        try:
            if target_provider:
                if (target_provider in self.providers and 
                    self.provider_status[target_provider]['available']):
                    self.last_provider = target_provider
                    self.last_provider_change = time.time()
                    logger.info(f"🔄 Forced switch to provider: {target_provider}")
                    return True
                else:
                    logger.warning(f"⚠️ Cannot switch to {target_provider} - not available")
                    return False
            else:
                # Reset to centralized routing system
                self.last_provider = None  # Let centralized routing choose
                self.last_provider_change = 0
                logger.info("🔄 Reset to centralized provider routing")
                return True
        except Exception as e:
            logger.error(f"❌ Error forcing provider switch: {str(e)}")
            return False
    
    # ================================================================
    # 📈 PERFORMANCE AND STATISTICS METHODS
    # ================================================================
    
    def get_request_statistics(self) -> Dict[str, Any]:
        """
        🎯 GET DETAILED STATISTICS ABOUT API USAGE PATTERNS
        
        Returns:
            Dictionary with request statistics and load balancing metrics
        """
        stats = {
            'load_balancing': {
                'strategy': self.provider_specialization,
                'active_provider': self.last_provider,
                'request_distribution': self.request_stats
            },
            'provider_performance': {},
            'recommendations': []
        }
        
        # Calculate provider efficiency
        for provider_name, provider_stats in self.request_stats.items():
            if provider_stats['total'] > 0:
                efficiency = {
                    'total_requests': provider_stats['total'],
                    'request_breakdown': {k: v for k, v in provider_stats.items() if k not in ['total', 'cache_hits', 'cache_misses']},
                    'cache_effectiveness': 0.0,
                    'specialization_usage': 0
                }
                
                # Calculate cache effectiveness
                total_cache_requests = provider_stats['cache_hits'] + provider_stats['cache_misses']
                if total_cache_requests > 0:
                    efficiency['cache_effectiveness'] = provider_stats['cache_hits'] / total_cache_requests
                
                # Calculate how often this provider was used for its specializations
                specializations = self.provider_status.get(provider_name, {}).get('specializations', [])
                specialized_requests = sum(provider_stats.get(spec.replace('_data', ''), 0) for spec in specializations)
                if provider_stats['total'] > 0:
                    efficiency['specialization_usage'] = specialized_requests / provider_stats['total']
                
                stats['provider_performance'][provider_name] = efficiency
        
        # Generate recommendations
        if 'coinmarketcap' not in [p for p, s in self.provider_status.items() if s['available']]:
            stats['recommendations'].append("Add CoinMarketCap API key to enable bulk data optimization")
        
        # Check cache effectiveness
        for provider_name, performance in stats['provider_performance'].items():
            if performance.get('cache_effectiveness', 0) < 0.5:
                stats['recommendations'].append(f"{provider_name} cache effectiveness low - check cache configuration")
        
        return stats
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for all providers"""
        stats = {
            'active_provider': self.last_provider,
            'provider_details': {},
            'system_summary': {},
            'load_balancing_metrics': self.get_request_statistics(),
            'routing_effectiveness': self._analyze_routing_effectiveness()
        }
        
        total_requests = 0
        
        for provider_name, provider in self.providers.items():
            provider_stats = {
                'available': self.provider_status[provider_name]['available'],
                'specializations': self.provider_status[provider_name].get('specializations', []),
                'method_capabilities': self.provider_status[provider_name].get('method_capabilities', {}),
                'cache_enabled': self.provider_status[provider_name].get('cache_enabled', False),
                'quota_tracking': self.provider_status[provider_name].get('quota_tracking', False)
            }
            
            if hasattr(provider, 'quota_tracker') and provider.quota_tracker:
                try:
                    quota_stats = provider.quota_tracker.get_quota_status()
                    provider_stats.update({
                        'daily_requests': quota_stats.get('daily_used', 0),
                        'daily_remaining': quota_stats.get('daily_remaining', 0),
                        'success_rate': quota_stats.get('success_rate_1h', 0),
                        'last_check': self.provider_status[provider_name]['last_check']
                    })
                    total_requests += quota_stats.get('daily_used', 0)
                except Exception as quota_error:
                    provider_stats['quota_error'] = str(quota_error)
            else:
                provider_stats.update({
                    'daily_requests': 0,
                    'daily_remaining': 'unknown',
                    'success_rate': 'unknown',
                    'last_check': self.provider_status[provider_name]['last_check']
                })
            
            stats['provider_details'][provider_name] = provider_stats
        
        stats['system_summary'] = {
            'total_daily_requests': total_requests,
            'available_provider_count': len([p for p in self.provider_status.values() if p['available']]),
            'last_provider_change': self.last_provider_change,
            'load_balancing_active': len([p for p in self.provider_status.values() if p['available']]) > 1,
            'centralized_routing_active': True,
            'recursion_prevention_active': True
        }
        
        return stats
    
    def _analyze_routing_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective the centralized routing is"""
        routing_analysis = {
            'total_routing_decisions': sum(self.performance_metrics['routing_decisions'].values()),
            'provider_distribution': {},
            'request_type_distribution': {},
            'specialization_match_rate': 0.0,
            'fallback_rate': 0.0
        }
        
        # Analyze provider distribution
        for routing_key, count in self.performance_metrics['routing_decisions'].items():
            request_type, provider = routing_key.split('_', 1)
            
            if provider not in routing_analysis['provider_distribution']:
                routing_analysis['provider_distribution'][provider] = 0
            routing_analysis['provider_distribution'][provider] += count
            
            if request_type not in routing_analysis['request_type_distribution']:
                routing_analysis['request_type_distribution'][request_type] = 0
            routing_analysis['request_type_distribution'][request_type] += count
        
        # Calculate specialization match rate
        specialization_matches = 0
        total_decisions = routing_analysis['total_routing_decisions']
        
        for routing_key, count in self.performance_metrics['routing_decisions'].items():
            request_type, provider = routing_key.split('_', 1)
            preferred_provider = self.provider_specialization.get(request_type, 'coingecko')
            
            if provider == preferred_provider:
                specialization_matches += count
        
        if total_decisions > 0:
            routing_analysis['specialization_match_rate'] = specialization_matches / total_decisions
        
        # Calculate fallback rate
        total_fallbacks = sum(self.performance_metrics['fallback_usage'].values())
        if total_decisions > 0:
            routing_analysis['fallback_rate'] = total_fallbacks / total_decisions
        
        return routing_analysis
    
    def _get_cache_failure_summary(self) -> Dict[str, Any]:
        """Get summary of cache failures"""
        if not self.cache_failure_tracker['failures']:
            return {'status': 'no_failures', 'total_failures': 0}
        
        total_failures = len(self.cache_failure_tracker['failures'])
        recent_failures = [
            f for f in self.cache_failure_tracker['failures'] 
            if time.time() - f.get('timestamp', 0) < 3600  # Last hour
        ]
        
        return {
            'status': 'has_failures',
            'total_failures': total_failures,
            'recent_failures_1h': len(recent_failures),
            'most_common_cause': max(self.cache_failure_tracker['root_causes'].items(), 
                                   key=lambda x: x[1])[0] if self.cache_failure_tracker['root_causes'] else 'none',
            'root_cause_distribution': self.cache_failure_tracker['root_causes']
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'average_response_times': {},
            'request_distribution': {},
            'cache_effectiveness': {}
        }
        
        # Calculate average response times
        for provider, times in self.performance_metrics['response_times'].items():
            if times:
                summary['average_response_times'][provider] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'samples': len(times)
                }
        
        # Request distribution
        for provider, stats in self.request_stats.items():
            summary['request_distribution'][provider] = stats['total']
        
        # Cache effectiveness
        for provider, stats in self.request_stats.items():
            total_cache_requests = stats['cache_hits'] + stats['cache_misses']
            if total_cache_requests > 0:
                summary['cache_effectiveness'][provider] = stats['cache_hits'] / total_cache_requests
            else:
                summary['cache_effectiveness'][provider] = 0.0
        
        return summary
    
    # ================================================================
    # 📋 CACHE FAILURE REPORTING METHODS
    # ================================================================
    
    def get_cache_failure_report(self) -> Dict[str, Any]:
        """
        📊 GET COMPREHENSIVE CACHE FAILURE REPORT
        """
        if not self.cache_failure_tracker['failures']:
            return {
                'status': 'no_failures',
                'message': 'No cache failures tracked yet',
                'total_failures': 0,
                'recommendations': ['✅ No cache issues detected - system operating normally']
            }
        
        tracker = self.cache_failure_tracker
        total_failures = len(tracker['failures'])
        recent_failures = [
            f for f in tracker['failures'] 
            if time.time() - f.get('timestamp', 0) < 3600  # Last hour
        ]
        
        report = {
            'status': 'has_failures',
            'total_failures': total_failures,
            'recent_failures_1h': len(recent_failures),
            'root_causes': tracker['root_causes'],
            'exception_patterns': tracker.get('exceptions', {}),
            'provider_patterns': tracker.get('provider_patterns', {}),
            'endpoint_patterns': tracker.get('endpoint_patterns', {}),
            'time_patterns': tracker.get('time_patterns', {}),
            'recommendations': self._generate_cache_recommendations(),
            'most_common_cause': 'none'
        }
        
        if tracker['root_causes']:
            most_common = max(tracker['root_causes'].items(), key=lambda x: x[1])
            report['most_common_cause'] = f"{most_common[0]} ({most_common[1]} occurrences)"
        
        return report
    
    def _generate_cache_recommendations(self) -> List[str]:
        """
        💡 GENERATE ACTIONABLE CACHE RECOMMENDATIONS
        """
        if not self.cache_failure_tracker['failures']:
            return ["✅ No cache failures to analyze"]
        
        recommendations = []
        root_causes = self.cache_failure_tracker['root_causes']
        
        for cause, count in root_causes.items():
            if cause == 'quota_exhausted' and count > 5:
                recommendations.append(f"❌ Quota exhausted {count} times - extend cache duration or add provider")
            elif cause == 'provider_unavailable' and count > 3:
                recommendations.append(f"❌ Provider availability issues ({count} times) - check API keys")
            elif cause == 'cache_disabled' and count > 0:
                recommendations.append(f"❌ Cache disabled ({count} times) - check cache initialization")
            elif cause == 'recursion_detected' and count > 0:
                recommendations.append(f"❌ Recursion detected ({count} times) - review method call patterns")
            elif cause == 'unknown_cache_failure' and count > 10:
                recommendations.append(f"❌ Unknown cache failures ({count} times) - enable debug logging")
        
        # Check exception patterns
        exceptions = self.cache_failure_tracker.get('exceptions', {})
        for exception_key, exception_list in exceptions.items():
            if len(exception_list) > 5:
                recommendations.append(f"❌ Frequent {exception_key} exceptions - investigate underlying cause")
        
        if not recommendations:
            recommendations.append("✅ No critical cache issues detected")
        
        return recommendations
    
    def __str__(self) -> str:
        """String representation of API manager status"""
        available_providers = [name for name, status in self.provider_status.items() 
                             if status['available']]
        return f"CryptoAPIManager(providers={available_providers}, routing=centralized, recursion_protection=active)"


# ============================================================================
# 🚀 UTILITY FUNCTIONS AND MODULE EXPORTS
# ============================================================================

def create_api_manager() -> CryptoAPIManager:
    """
    Create and initialize the intelligent API manager with centralized routing
    
    Returns:
        Fully configured API manager with centralized routing and recursion prevention
    """
    try:
        manager = CryptoAPIManager()
        available_providers = [name for name, status in manager.provider_status.items() 
                             if status['available']]
        
        logger.info(f"🚀 Centralized API Manager created with {len(available_providers)} providers: {available_providers}")
        
        # Log centralized routing capabilities
        if len(available_providers) > 1:
            logger.info("📊 CENTRALIZED ROUTING: ENABLED - Intelligent dual-provider routing active")
            logger.info("📊 RECURSION PREVENTION: ENABLED - Thread-safe circular call protection")
            logger.info("📊 CACHE DIAGNOSTICS: ENABLED - Comprehensive failure analysis")
        else:
            logger.info("📊 CENTRALIZED ROUTING: LIMITED - Single provider mode")
        
        # Log provider status for debugging
        for provider_name, status in manager.provider_status.items():
            if status['available']:
                specializations = status.get('specializations', [])
                logger.debug(f"  ✅ {provider_name}: Available (specializes in: {', '.join(specializations)})")
            else:
                error = status.get('initialization_error', 'Unknown error')
                logger.debug(f"  ⚠️ {provider_name}: Unavailable ({error})")
        
        return manager
        
    except Exception as e:
        logger.error(f"❌ Critical error creating Centralized API Manager: {str(e)}")
        raise RuntimeError(f"Failed to create Centralized API Manager: {str(e)}")

def get_api_manager_diagnostics() -> Dict[str, Any]:
    """
    Get diagnostic information about API manager setup including centralized routing status
    
    Returns:
        Dictionary with comprehensive diagnostic information
    """
    diagnostics = {
        'environment_variables': {},
        'provider_requirements': {},
        'centralized_routing_capabilities': {},
        'recursion_prevention_status': 'ACTIVE',
        'cache_diagnostics_status': 'ACTIVE',
        'recommendations': []
    }
    
    # Check environment variables
    env_vars_to_check = [
        'CoinMarketCap_API',
        'COINMARKETCAP_API_KEY', 
        'CMC_API_KEY',
        'COINMARKETCAP_API'
    ]
    
    for var_name in env_vars_to_check:
        value = os.getenv(var_name, '')
        diagnostics['environment_variables'][var_name] = {
            'present': bool(value),
            'length': len(value) if value else 0
        }
    
    # Provider requirements
    diagnostics['provider_requirements'] = {
        'coingecko': {
            'requires_api_key': False,
            'always_available': True,
            'description': 'Free tier available, no API key required',
            'specializations': ['historical_data', 'individual_tokens'],
            'rate_limits': 'Moderate - 20 calls/minute'
        },
        'coinmarketcap': {
            'requires_api_key': True,
            'api_key_found': any(os.getenv(var, '') for var in env_vars_to_check),
            'description': 'Requires API key from coinmarketcap.com',
            'specializations': ['bulk_data', '7d_historical_data', 'real_time', 'market_overview'],
            'rate_limits': 'Higher limits with API key'
        }
    }
    
    # Centralized routing capabilities
    cmc_available = diagnostics['provider_requirements']['coinmarketcap']['api_key_found']
    diagnostics['centralized_routing_capabilities'] = {
        'intelligent_routing': True,  # Always available with centralized system
        'rate_limit_prevention': cmc_available,
        'bulk_data_optimization': cmc_available,
        'historical_data_optimization': True,  # Available with either provider
        'dual_provider_redundancy': cmc_available,
        'recursion_prevention': True,
        'cache_failure_diagnostics': True,
        'auto_storage': True
    }
    
    # Generate recommendations
    if not cmc_available:
        diagnostics['recommendations'].extend([
            "🎯 Add CoinMarketCap API key for optimal centralized routing",
            "📊 Enable bulk data routing to prevent CoinGecko rate limits",
            "🚀 Unlock dual-provider redundancy for maximum reliability"
        ])
    else:
        diagnostics['recommendations'].append("✅ Optimal dual-API setup detected - all centralized routing features enabled")
    
    diagnostics['recommendations'].extend([
        "✅ Centralized routing prevents hardcoded provider logic",
        "✅ Recursion prevention eliminates circular call patterns",
        "✅ Cache diagnostics provide root cause analysis for failures"
    ])
    
    return diagnostics

def log_api_performance_summary(manager: CryptoAPIManager):
    """
    🎯 LOG COMPREHENSIVE PERFORMANCE SUMMARY WITH CENTRALIZED ROUTING ANALYSIS
    
    Args:
        manager: The API manager instance to analyze
    """
    logger.info("📊 CENTRALIZED API PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    
    # Get statistics
    stats = manager.get_request_statistics()
    
    # Log centralized routing effectiveness
    logger.info(f"🎯 Centralized Routing Strategy: {stats['load_balancing']['strategy']}")
    logger.info(f"🔄 Active Provider: {stats['load_balancing']['active_provider']}")
    
    # Log request distribution
    for provider, distribution in stats['load_balancing']['request_distribution'].items():
        if distribution['total'] > 0:
            logger.info(f"📈 {provider.upper()}: {distribution['total']} total requests")
            for req_type, count in distribution.items():
                if req_type not in ['total', 'cache_hits', 'cache_misses'] and count > 0:
                    logger.info(f"  📋 {req_type}: {count}")
            
            # Log cache effectiveness
            cache_hits = distribution.get('cache_hits', 0)
            cache_misses = distribution.get('cache_misses', 0)
            total_cache = cache_hits + cache_misses
            if total_cache > 0:
                effectiveness = (cache_hits / total_cache) * 100
                logger.info(f"  🗄️ Cache Effectiveness: {effectiveness:.1f}% ({cache_hits}/{total_cache})")
    
    # Log performance metrics
    for provider, performance in stats['provider_performance'].items():
        if performance['total_requests'] > 0:
            specialization_rate = performance['specialization_usage'] * 100
            cache_effectiveness = performance.get('cache_effectiveness', 0) * 100
            logger.info(f"⚡ {provider.upper()} Efficiency:")
            logger.info(f"  📊 Specialization Match: {specialization_rate:.1f}%")
            logger.info(f"  🗄️ Cache Effectiveness: {cache_effectiveness:.1f}%")
    
    # Log system health
    health = manager.get_system_health()
    logger.info(f"🏥 System Health: {health['overall_status']}")
    logger.info(f"🔄 Load Balancing: {health['load_balancing_status']}")
    
    # Log cache failure summary
    cache_summary = health.get('cache_failure_summary', {})
    if cache_summary.get('total_failures', 0) > 0:
        logger.info(f"🚨 Cache Failures: {cache_summary['total_failures']} total, {cache_summary.get('recent_failures_1h', 0)} recent")
        logger.info(f"🔍 Most Common Cause: {cache_summary.get('most_common_cause', 'unknown')}")
    else:
        logger.info("✅ Cache Performance: No failures detected")
    
    # Log recommendations
    if stats['recommendations']:
        logger.info("💡 Recommendations:")
        for rec in stats['recommendations']:
            logger.info(f"  • {rec}")
    
    logger.info("=" * 70)

# ============================================================================
# 🎯 MODULE EXPORTS AND METADATA
# ============================================================================

__all__ = [
    'CryptoAPIManager',
    'create_api_manager',
    'get_api_manager_diagnostics',
    'log_api_performance_summary'
]

__version__ = "4.0.0"
__author__ = "Generational Wealth Trading System"
__description__ = "Centralized API manager with intelligent routing, recursion prevention, and comprehensive cache diagnostics"
