#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Timeline Scraper - Part 1: Core Infrastructure & Configuration
Award-winning modular design with advanced anti-detection and resilience
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import time
import random
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from collections import deque, defaultdict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException,
    WebDriverException, ElementClickInterceptedException
)

from datetime_utils import ensure_naive_datetimes, strip_timezone, safe_datetime_diff
from utils.logger import logger


class ScrapingMode(Enum):
    """Scraping operation modes for different strategies"""
    STEALTH = "stealth"          # Maximum anti-detection
    BALANCED = "balanced"        # Default mode
    AGGRESSIVE = "aggressive"    # Fast but higher detection risk
    RECOVERY = "recovery"        # Minimal operations during errors


class ContentType(Enum):
    """Types of content we're interested in"""
    MARKET_DISCUSSION = "market_discussion"
    CRYPTO_NEWS = "crypto_news"
    TECH_DISCUSSION = "tech_discussion"
    QUESTION_POST = "question_post"
    PROMOTIONAL_AD = "promotional_ad"
    HONEYPOT_CONTENT = "honeypot_content"


@dataclass
class SelectorConfig:
    """Configuration for CSS/XPath selectors with fallbacks"""
    primary: str
    fallbacks: List[str] = field(default_factory=list)
    weight: float = 1.0
    last_success: Optional[datetime] = None
    success_rate: float = 0.0
    total_attempts: int = 0
    successful_attempts: int = 0
    
    def record_attempt(self, success: bool) -> None:
        """Record selector usage statistics"""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
            self.last_success = datetime.now()
        self.success_rate = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0.0
    
    def get_all_selectors(self) -> List[str]:
        """Get all selectors in priority order"""
        return [self.primary] + self.fallbacks
    
    def is_reliable(self) -> bool:
        """Check if this selector configuration is reliable"""
        return self.success_rate > 0.5 and self.total_attempts >= 3


@dataclass
class ScrapingConfig:
    """Comprehensive configuration for scraping operations"""
    # Core settings
    mode: ScrapingMode = ScrapingMode.BALANCED
    max_posts_target: int = 50
    max_scroll_attempts: int = 20
    scroll_pause_range: Tuple[float, float] = (1.5, 3.0)
    
    # Timing and delays
    element_wait_timeout: int = 15
    page_load_timeout: int = 30
    recovery_wait_time: float = 5.0
    anti_detection_delay_range: Tuple[float, float] = (0.3, 1.2)
    
    # Retry and recovery
    max_retries: int = 5
    max_consecutive_failures: int = 3
    circuit_breaker_threshold: int = 10
    recovery_mode_duration: int = 300  # 5 minutes
    
    # Content filtering
    min_post_text_length: int = 10
    max_post_text_length: int = 2000
    skip_promotional_content: bool = True
    skip_honeypot_detection: bool = True
    
    # Performance
    enable_parallel_processing: bool = True
    max_worker_threads: int = 3
    cache_extracted_data: bool = True
    
    # Debugging
    enable_debug_screenshots: bool = False
    verbose_logging: bool = False
    save_failed_extractions: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker pattern"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN - too many recent failures")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful operation"""
        with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """Handle failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'


class AdaptiveSelectorManager:
    """Manages CSS selectors with adaptive fallback and learning"""
    
    def __init__(self):
        self.selectors: Dict[str, SelectorConfig] = {}
        self._initialize_default_selectors()
        self._lock = threading.Lock()
        
    def _initialize_default_selectors(self) -> None:
        """Initialize default selector configurations"""
        # Timeline post selectors (primary content containers)
        self.selectors['timeline_posts'] = SelectorConfig(
            primary='div[data-testid="cellInnerDiv"]',
            fallbacks=[
                'article[data-testid="tweet"]',
                'div[role="article"]',
                'div[data-testid="tweetWrapperOuter"]',
                'article[role="article"]'
            ]
        )
        
        # Post text content selectors
        self.selectors['post_text'] = SelectorConfig(
            primary='div[data-testid="tweetText"]',
            fallbacks=[
                'div[lang]',
                'div.css-901oao',
                'span[data-testid="tweetText"]',
                'div[dir="auto"]'
            ]
        )
        
        # Author information selectors
        self.selectors['post_author'] = SelectorConfig(
            primary='div[data-testid="User-Name"]',
            fallbacks=[
                'div[data-testid="User-Names"]',
                'a[role="link"] span',
                'div[dir="auto"] > span'
            ]
        )
        
        # Timestamp selectors
        self.selectors['timestamp'] = SelectorConfig(
            primary='time',
            fallbacks=[
                'a[href*="/status/"] time',
                'div[data-testid="User-Name"] time',
                'span[data-testid="Time"]'
            ]
        )
        
        # Engagement metrics selectors
        self.selectors['engagement'] = SelectorConfig(
            primary='div[role="group"] div[role="button"]',
            fallbacks=[
                'div[data-testid="reply"]',
                'div[data-testid="retweet"]',
                'div[data-testid="like"]',
                'div[data-testid="bookmark"]'
            ]
        )
        
        # Ad/promotional content indicators
        self.selectors['ad_indicators'] = SelectorConfig(
            primary='div[data-testid="promotedIndicator"]',
            fallbacks=[
                'span:contains("Ad")',
                'span:contains("Promoted")',
                'div[aria-label*="Promoted"]',
                'div[data-testid="placementTracking"]'
            ]
        )
        
        # Media content selectors
        self.selectors['media_content'] = SelectorConfig(
            primary='div[data-testid="tweetPhoto"]',
            fallbacks=[
                'div[data-testid="videoPlayer"]',
                'div[data-testid="card.wrapper"]',
                'img[src*="pbs.twimg.com"]',
                'video'
            ]
        )
    
    def get_selector_config(self, selector_name: str) -> Optional[SelectorConfig]:
        """Get selector configuration by name"""
        return self.selectors.get(selector_name)
    
    def get_best_selector(self, selector_name: str) -> Optional[str]:
        """Get the most reliable selector for a given purpose"""
        config = self.selectors.get(selector_name)
        if not config:
            return None
        
        # If we have enough data, prefer the most successful selector
        if config.total_attempts >= 5:
            all_selectors = config.get_all_selectors()
            # For now, return primary unless it's consistently failing
            if config.success_rate > 0.3:
                return config.primary
            elif config.fallbacks:
                return config.fallbacks[0]
        
        return config.primary
    
    def record_selector_result(self, selector_name: str, selector_used: str, success: bool) -> None:
        """Record the result of using a selector"""
        with self._lock:
            config = self.selectors.get(selector_name)
            if config and selector_used in config.get_all_selectors():
                config.record_attempt(success)
    
    def update_selector_priority(self, selector_name: str, successful_selector: str) -> None:
        """Update selector priority based on successful usage"""
        with self._lock:
            config = self.selectors.get(selector_name)
            if not config:
                return
            
            # If a fallback selector was more successful, promote it
            if successful_selector != config.primary and successful_selector in config.fallbacks:
                config.fallbacks.remove(successful_selector)
                config.fallbacks.insert(0, successful_selector)
                logger.logger.debug(f"Promoted selector '{successful_selector}' for {selector_name}")


class AntiDetectionManager:
    """Manages anti-detection measures and human-like behavior"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.last_action_time = time.time()
        self.action_history = deque(maxlen=50)
        self.suspicious_activity_score = 0.0
        self._lock = threading.Lock()
    
    def add_human_delay(self, action_type: str = "default") -> None:
        """Add human-like delay between actions"""
        delay_ranges = {
            "scroll": (1.0, 2.5),
            "click": (0.8, 1.8),
            "type": (0.1, 0.3),
            "page_load": (2.0, 4.0),
            "default": self.config.anti_detection_delay_range
        }
        
        min_delay, max_delay = delay_ranges.get(action_type, delay_ranges["default"])
        delay = random.uniform(min_delay, max_delay)
        
        # Add slight variation based on recent activity
        if len(self.action_history) > 5:
            recent_speed = len([a for a in self.action_history if time.time() - a < 10])
            if recent_speed > 8:  # Too many actions recently
                delay *= 1.5  # Slow down
        
        time.sleep(delay)
        
        with self._lock:
            self.action_history.append(time.time())
            self.last_action_time = time.time()
    
    def should_add_extra_delay(self) -> bool:
        """Determine if we should add extra delay to appear more human"""
        current_time = time.time()
        recent_actions = [a for a in self.action_history if current_time - a < 30]
        
        # If too many actions in short time, add delay
        if len(recent_actions) > 15:
            return True
        
        # Random chance to add delay for unpredictability
        return random.random() < 0.1
    
    def get_random_scroll_amount(self) -> int:
        """Get randomized scroll amount to appear human-like"""
        base_amounts = [600, 700, 800, 900, 1000]
        base = random.choice(base_amounts)
        variation = random.randint(-100, 150)
        return max(400, base + variation)
    
    def should_take_break(self) -> bool:
        """Determine if we should take a longer break"""
        # Take break after extended activity
        if len(self.action_history) > 30:
            time_span = self.action_history[-1] - self.action_history[0]
            if time_span < 300:  # 30 actions in less than 5 minutes
                return True
        
        # Random break chance
        return random.random() < 0.05
    
    def take_human_break(self) -> None:
        """Take a human-like break"""
        break_duration = random.uniform(10, 30)
        logger.logger.info(f"Taking human-like break for {break_duration:.1f} seconds")
        time.sleep(break_duration)
        
        # Clear some action history after break
        with self._lock:
            if len(self.action_history) > 10:
                for _ in range(5):
                    self.action_history.popleft()


class BrowserInteractionManager:
    """Manages safe browser interactions with retry logic"""
    
    def __init__(self, browser, config: ScrapingConfig, anti_detection: AntiDetectionManager):
        logger.logger.info("🔧 Initializing BrowserInteractionManager")
        self.browser = browser
        self.config = config
        self.anti_detection = anti_detection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.recovery_mode_duration
        )
        logger.logger.info(f"✅ BrowserInteractionManager initialized with timeout: {config.element_wait_timeout}")
    
    def safe_find_elements(self, css_selector: str, timeout: Optional[int] = None) -> List[Any]:
        """
        Safely find elements using CSS selectors with timeout and error handling
        
        FIXED: Now returns elements as they become available, doesn't wait for ALL
        
        Args:
            css_selector: CSS selector string
            timeout: Maximum time to wait for elements (uses config default if None)
            
        Returns:
            List of valid WebElement objects, empty list if none found or error occurs
        """
        timeout = timeout or self.config.element_wait_timeout
        logger.logger.info(f"🔍 safe_find_elements called: selector='{css_selector}', timeout={timeout}s")
        
        try:
            # Check browser health before attempting element search
            if not self.browser.is_browser_healthy():
                logger.logger.warning("Browser health check failed, cannot search for elements")
                return []
            
            # Add small random delay for anti-detection
            if self.anti_detection.should_add_extra_delay():
                logger.logger.debug("Adding anti-detection delay")
                self.anti_detection.add_human_delay("search")
            
            logger.logger.debug(f"🔍 Searching for elements: {css_selector} (timeout: {timeout}s)")
            
            # FIXED: Try immediate search first (most elements are already loaded)
            try:
                logger.logger.debug("Attempting immediate element search")
                immediate_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, css_selector)
                logger.logger.info(f"⚡ Immediate search result: {len(immediate_elements)} elements found")
                if immediate_elements:
                    logger.logger.debug(f"✅ Found {len(immediate_elements)} elements immediately")
                    validated = self._validate_elements(immediate_elements)
                    logger.logger.info(f"🎯 Returning {len(validated)} validated elements from immediate search")
                    return validated
            except Exception as e:
                logger.logger.warning(f"Immediate search failed: {str(e)}")
            
            # FIXED: If no immediate elements, wait for at least ONE element
            logger.logger.debug("No immediate elements found, starting WebDriverWait")
            wait = WebDriverWait(self.browser.driver, timeout)
            
            # Wait for at least one element to be present
            logger.logger.debug(f"Waiting for presence of element: {css_selector}")
            wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            )
            logger.logger.debug("Element presence confirmed by WebDriverWait")
            
            # Now find all available elements
            logger.logger.debug("Finding all elements after wait")
            elements = self.browser.driver.find_elements(By.CSS_SELECTOR, css_selector)
            logger.logger.info(f"✅ Found {len(elements)} elements after wait")
            
            # Filter out stale elements
            logger.logger.debug("Validating elements")
            valid_elements = self._validate_elements(elements)
            logger.logger.info(f"📋 Validated {len(valid_elements)} elements")
            
            return valid_elements
                
        except TimeoutException:
            logger.logger.warning(f"⏰ Timeout waiting for elements: {css_selector}")
            return []
        except Exception as e:
            logger.logger.error(f"💥 Error finding elements '{css_selector}': {str(e)}")
            return []
        
    def _validate_elements(self, elements: List[Any]) -> List[Any]:
        """Validate that elements are not stale and are usable"""
        logger.logger.debug(f"🔬 Validating {len(elements)} elements")
        valid_elements = []
        
        for i, element in enumerate(elements):
            try:
                # Test if element is still attached to DOM
                logger.logger.debug(f"Testing element {i+1} display status")
                _ = element.is_displayed()  # This will raise exception if stale
                valid_elements.append(element)
                logger.logger.debug(f"✅ Element {i+1} is valid")
                
            except StaleElementReferenceException:
                logger.logger.warning(f"❌ Element {i+1} is stale, skipping")
                continue
            except Exception as e:
                # Other errors (element not visible, etc.) - still include element
                logger.logger.debug(f"⚠️ Element {i+1} validation warning: {str(e)}")
                valid_elements.append(element)
        
        logger.logger.debug(f"🎯 Validation complete: {len(valid_elements)}/{len(elements)} elements valid")
        return valid_elements    
    
    def safe_get_text(self, element) -> str:
        """Safely extract text from an element"""
        logger.logger.debug("🔤 safe_get_text called")
        try:
            # Add tiny delay for anti-detection
            if random.random() < 0.1:
                logger.logger.debug("Adding tiny anti-detection delay")
                time.sleep(random.uniform(0.05, 0.15))
            
            text = element.text.strip()
            logger.logger.debug(f"📝 Extracted text: '{text[:50]}...' ({len(text)} chars)")
            return text
        except (StaleElementReferenceException, NoSuchElementException):
            logger.logger.warning("❌ Element is stale or not found when getting text")
            return ""
        except Exception as e:
            logger.logger.warning(f"💥 Error getting text: {str(e)}")
            return ""
    
    def safe_get_attribute(self, element, attribute: str) -> Optional[str]:
        """Safely get attribute from an element"""
        logger.logger.debug(f"🏷️ safe_get_attribute called for: {attribute}")
        try:
            attr_value = element.get_attribute(attribute)
            logger.logger.debug(f"✅ Attribute '{attribute}': {attr_value}")
            return attr_value
        except (StaleElementReferenceException, NoSuchElementException):
            logger.logger.warning(f"❌ Element is stale or not found when getting attribute: {attribute}")
            return None
        except Exception as e:
            logger.logger.warning(f"💥 Error getting attribute {attribute}: {str(e)}")
            return None
    
    def safe_execute_script(self, script: str, *args) -> Any:
        """Safely execute JavaScript"""
        logger.logger.debug("safe_execute_script called")
        try:
            # Check browser health before attempting JavaScript execution
            if not self.browser.is_browser_healthy():
                logger.logger.warning("Browser health check failed, cannot execute JavaScript")
                return None
            
            # Add human-like delay before JS execution
            logger.logger.debug("Adding human-like delay before JS execution")
            self.anti_detection.add_human_delay("script")
            
            logger.logger.debug(f"Executing JS script: {script[:100]}...")
            result = self.browser.driver.execute_script(script, *args)
            logger.logger.debug("JavaScript execution successful")
            return result
        except Exception as e:
            logger.logger.error(f"JavaScript execution failed: {str(e)}")
            return None
    
    def safe_scroll(self, amount: Optional[int] = None) -> bool:
        """Perform safe, human-like scrolling"""
        logger.logger.debug("📜 safe_scroll called")
        try:
            scroll_amount = amount or self.anti_detection.get_random_scroll_amount()
            logger.logger.debug(f"Scrolling by {scroll_amount} pixels")
            
            # Use smooth scrolling for better anti-detection
            script = f"""
                window.scrollBy({{
                    top: {scroll_amount},
                    left: 0,
                    behavior: 'smooth'
                }});
            """
            
            logger.logger.debug("Executing scroll script")
            result = self.safe_execute_script(script)
            
            # Add human-like pause after scrolling
            logger.logger.debug("Adding human-like pause after scrolling")
            self.anti_detection.add_human_delay("scroll")
            
            success = result is not None
            logger.logger.debug(f"Scroll result: {'success' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.logger.error(f"💥 Scrolling failed: {str(e)}")
            return False


class ExtractionCache:
    """Cache for extracted content to avoid re-processing"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def _generate_key(self, element_html: str) -> str:
        """Generate cache key from element HTML"""
        return hashlib.md5(element_html.encode()).hexdigest()
    
    def get(self, element_html: str) -> Optional[Dict]:
        """Get cached extraction result"""
        key = self._generate_key(element_html)
        
        with self._lock:
            if key in self.cache:
                # Check if entry is still valid
                if time.time() - self.timestamps[key] < self.ttl_seconds:
                    return self.cache[key].copy()
                else:
                    # Entry expired, remove it
                    del self.cache[key]
                    del self.timestamps[key]
        
        return None
    
    def set(self, element_html: str, data: Dict) -> None:
        """Cache extraction result"""
        key = self._generate_key(element_html)
        
        with self._lock:
            # Implement simple LRU by removing oldest entries
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = data.copy()
            self.timestamps[key] = time.time()
    
    def clear_expired(self) -> None:
        """Clear expired cache entries"""
        current_time = time.time()
        
        with self._lock:
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp >= self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]


class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback behaviors"""
    
    def __init__(self, config: ScrapingConfig, browser_instance=None):
        self.config = config
        self.browser_instance = browser_instance  # FIXED: Store browser instance directly
        self.error_counts = defaultdict(int)
        self.last_error_time = defaultdict(float)
        self.extraction_cache = None  # Will be set by the scraper if available
        self.recovery_strategies = {
            'page_refresh': self._refresh_page,
            'clear_cache': self._clear_cache,
            'restart_browser': self._restart_browser,
            'wait_and_retry': self._wait_and_retry
        }
    
    def set_browser_instance(self, browser_instance):
        """Set the browser instance for error recovery operations"""
        self.browser_instance = browser_instance
    
    def set_extraction_cache(self, extraction_cache):
        """Set the extraction cache for cache clearing operations"""
        self.extraction_cache = extraction_cache
    
    def handle_error(self, error_type: str, error: Exception, context: str = "") -> bool:
        """Handle error with appropriate recovery strategy"""
        self.error_counts[error_type] += 1
        self.last_error_time[error_type] = time.time()
        
        logger.logger.warning(f"Error {error_type} in {context}: {str(error)}")
        
        # Choose recovery strategy based on error type and frequency
        if error_type in ['TimeoutException', 'NoSuchElementException']:
            if self.error_counts[error_type] < 3:
                return self._wait_and_retry()
            else:
                return self._refresh_page()
        
        elif error_type in ['StaleElementReferenceException']:
            return True  # Usually resolves itself on retry
        
        elif error_type in ['WebDriverException']:
            if self.error_counts[error_type] < 2:
                return self._clear_cache()
            else:
                return self._restart_browser()
        
        else:
            # Generic error handling
            return self._wait_and_retry()
    
    def _refresh_page(self) -> bool:
        """Refresh the current page using browser's refresh capability"""
        try:
            logger.logger.info("Attempting page refresh for error recovery")
            
            # FIXED: Use direct browser instance access from ErrorRecoveryManager
            # The ErrorRecoveryManager needs to be initialized with browser access
            if not hasattr(self, 'browser_instance') or not self.browser_instance:
                logger.logger.error("Browser instance not available for page refresh")
                return False
                
            # Check if browser is healthy before attempting refresh
            if not self.browser_instance.is_browser_healthy():
                logger.logger.warning("Browser not healthy, refresh may fail")
                
            # Use the browser's wait_and_refresh method with config timeout
            timeout = getattr(self.config, 'recovery_wait_time', 5)
            self.browser_instance.wait_and_refresh(timeout)
            
            # Verify refresh was successful by checking if browser is still responsive
            if self.browser_instance.is_browser_healthy():
                logger.logger.info("Page refresh completed successfully")
                return True
            else:
                logger.logger.warning("Page refresh completed but browser health check failed")
                return False
                
        except Exception as e:
            logger.logger.error(f"Page refresh failed: {str(e)}")
            return False

    def _clear_cache(self) -> bool:
        """Clear browser cache and extraction cache for error recovery"""
        try:
            logger.logger.info("Clearing cache for error recovery")
            
            # FIXED: Use direct browser instance access from ErrorRecoveryManager
            if not hasattr(self, 'browser_instance') or not self.browser_instance:
                logger.logger.error("Browser instance not available for cache clearing")
                return False
                
            success = True
            
            # Clear browser cookies using browser's clear_cookies method
            try:
                self.browser_instance.clear_cookies()
                logger.logger.debug("Browser cookies cleared successfully")
            except Exception as cookie_error:
                logger.logger.warning(f"Failed to clear browser cookies: {str(cookie_error)}")
                success = False
                
            # Clear extraction cache if it exists
            if hasattr(self, 'extraction_cache') and self.extraction_cache:
                try:
                    self.extraction_cache.clear_expired()
                    logger.logger.debug("Extraction cache cleared successfully")
                except Exception as cache_error:
                    logger.logger.warning(f"Failed to clear extraction cache: {str(cache_error)}")
                    success = False
            
            if success:
                logger.logger.info("Cache clearing completed successfully")
                return True
            else:
                logger.logger.warning("Cache clearing completed with some failures")
                return False
                
        except Exception as e:
            logger.logger.error(f"Cache clear failed: {str(e)}")
            return False

    def _restart_browser(self) -> bool:
        """Restart browser session using enhanced browser restart capability"""
        try:
            logger.logger.info("Restarting browser for error recovery")
            
            # FIXED: Use direct browser instance access from ErrorRecoveryManager
            if not hasattr(self, 'browser_instance') or not self.browser_instance:
                logger.logger.error("Browser instance not available for restart")
                return False
                
            # Use the restart_browser_session method from the browser instance
            restart_success = self.browser_instance.restart_browser_session()
            
            if restart_success:
                logger.logger.info("Browser restart completed successfully")
                # Reset error counts after successful restart
                if hasattr(self, 'error_counts'):
                    self.error_counts.clear()
                return True
            else:
                logger.logger.error("Browser restart failed")
                return False
                
        except Exception as e:
            logger.logger.error(f"Browser restart failed: {str(e)}")
            return False
    
    def _wait_and_retry(self) -> bool:
        """Simple wait and retry strategy"""
        wait_time = min(self.config.recovery_wait_time * 2, 15.0)
        logger.logger.info(f"Waiting {wait_time}s before retry")
        time.sleep(wait_time)
        return True
    
    def should_abort_operation(self, error_type: str) -> bool:
        """Determine if operation should be aborted due to too many errors"""
        return self.error_counts[error_type] > self.config.max_consecutive_failures
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Timeline Scraper - Part 2: Advanced Post Detection & Extraction
Sophisticated content detection with honeypot filtering and robust data extraction
"""

import re
import time
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)

from datetime_utils import strip_timezone, safe_datetime_diff
from utils.logger import logger


@dataclass
class PostMetrics:
    """Structured post engagement metrics"""
    replies: int = 0
    retweets: int = 0
    likes: int = 0
    bookmarks: int = 0
    views: int = 0
    quote_tweets: int = 0
    
    def total_engagement(self) -> int:
        """Calculate total engagement score"""
        return self.replies + self.retweets + self.likes + self.bookmarks
    
    def engagement_score(self) -> float:
        """Calculate weighted engagement score"""
        # Replies are most valuable, then retweets, then likes
        return (self.replies * 3.0 + self.retweets * 2.0 + 
                self.likes * 1.0 + self.bookmarks * 1.5)


@dataclass
class PostMedia:
    """Structured post media information"""
    has_image: bool = False
    has_video: bool = False
    has_gif: bool = False
    has_poll: bool = False
    has_external_link: bool = False
    has_quote_tweet: bool = False
    media_count: int = 0
    external_domains: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.external_domains is None:
            self.external_domains = []
    
    def has_any_media(self) -> bool:
        """Check if post has any media content"""
        return (self.has_image or self.has_video or self.has_gif or 
                self.has_poll or self.has_external_link or self.has_quote_tweet)


@dataclass
class PostMetadata:
    """Comprehensive post metadata"""
    post_id: str
    post_url: Optional[str]
    author_name: str
    author_handle: str
    author_verified: bool
    author_profile_url: Optional[str]
    timestamp: Optional[datetime]
    timestamp_text: str
    is_reply: bool = False
    is_retweet: bool = False
    is_quote_tweet: bool = False
    is_thread: bool = False
    thread_position: Optional[int] = None


class HoneypotDetector:
    """Advanced detection system for promotional content and honeypot accounts"""
    
    def __init__(self):
        # Known honeypot domains (updated based on your examples)
        self.suspicious_domains = {
            'allai.com', 'tandfonline.com', 'aitools.com', 'techguru.ai',
            'cryptomaster.io', 'tradingbot.com', 'cointrader.app',
            'blockchainacademy.com', 'defitools.org', 'nftcreator.app',
            'crypto-revolution.com', 'ai-breakthrough.net', 'futuretech.org'
        }
        
        # MUCH MORE TARGETED promotional language patterns
        # Only catch obvious spam/ads, not legitimate tech discussions
        self.ad_language_patterns = [
            # Explicit promotional CTAs
            r'\b(get started now|learn more now|click here now|join today|sign up today|try (?:it )?now)\b',
            r'\b(limited time offer|special offer|exclusive deal|act now|hurry|don\'t miss)\b',
            r'\b(free trial|no cost|100% free|risk free|money back guarantee)\b',
            
            # Very specific AI/crypto spam patterns (much more targeted)
            r'\balli? ai changes everything\b',  # Very specific to your examples
            r'\bunlimited code generation\b',    # Very specific to your examples
            r'\bharness the full power of ai\b', # Very specific to your examples
            
            # Obvious testimonial spam (much more specific)
            r'\b\d+\.\d+\s*rating from \d+\+?\s*users?\b',  # "4.9 rating from 1000+ users"
            r'\btrusted by \d+k?\+?\s*(?:users?|companies|customers)\b',  # "trusted by 10k+ users"
            
            # Obvious promotional domains in text
            r'\b(?:visit|check out|go to) [a-z]+\.(?:com|ai|io)\b'
        ]
        
        # MUCH MORE TARGETED username patterns - only obvious spam accounts
        self.suspicious_username_patterns = [
            r'.*(?:kicksnpromos|dealsandoffers|promocode).*',  # Very obvious promo accounts
            r'^[a-z]+\d{4,}$',  # generic names with 4+ numbers (john1234, mary5678)
            r'.*(?:official).*(?:ai|crypto).*(?:bot|tool|system).*'  # "OfficialAIBot", "CryptoSystemOfficial"
        ]
        
        # MUCH MORE TARGETED promotional indicators - only obvious ads
        self.promotional_indicators = [
            'sponsored', 'partnership', 'affiliate link',
            'discount code', 'promo code', 'limited time',
            'special offer', 'exclusive access', 'beta access'
        ]
    
    def is_honeypot_content(self, post_element, post_text: str, author_handle: str = "", 
                          post_url: str = "") -> Tuple[bool, str]:
        """
        MUCH MORE CONSERVATIVE honeypot detection - only catch obvious spam
        
        Returns:
            Tuple of (is_honeypot, reason)
        """
        # Check 1: Look for explicit ad indicators (most reliable)
        ad_indicators = self._check_ad_indicators(post_element)
        if ad_indicators:
            return True, f"Explicit ad indicator: {ad_indicators}"
        
        # Check 2: Known suspicious domains (very reliable)
        suspicious_domains = self._extract_suspicious_domains(post_element, post_text)
        if suspicious_domains:
            return True, f"Known suspicious domain: {', '.join(suspicious_domains)}"
        
        # Check 3: Very specific promotional language (much more conservative)
        promotional_score = self._analyze_promotional_language(post_text)
        if promotional_score >= 2:  # Raised threshold - need multiple very specific patterns
            return True, f"High promotional language score: {promotional_score}"
        
        # Check 4: Very obvious username patterns (much more conservative)
        if self._is_obvious_spam_username(author_handle):
            return True, f"Obvious spam username pattern: {author_handle}"
        
        # Check 5: Multiple explicit promotional indicators
        explicit_promo_count = sum(1 for indicator in self.promotional_indicators if indicator in post_text.lower())
        if explicit_promo_count >= 2:  # Need multiple explicit promotional terms
            return True, f"Multiple explicit promotional indicators: {explicit_promo_count}"
        
        # Check 6: Very specific fake testimonial patterns
        if self._has_obvious_fake_testimonials(post_text):
            return True, "Obvious fake testimonials/ratings pattern"
        
        # If none of the very specific patterns match, it's likely legitimate
        return False, ""
    
    def _check_ad_indicators(self, post_element) -> Optional[str]:
        """Check for explicit ad/promoted indicators"""
        try:
            # Look for explicit "Ad" or "Promoted" text
            ad_text_patterns = [
                r'\bAd\b',              # Standalone "Ad" 
                r'\bPromoted\b',        # Standalone "Promoted"
                r'\bSponsored\b'        # Standalone "Sponsored"
            ]
            
            post_html = post_element.get_attribute('innerHTML') or ""
            post_text = post_element.text or ""
            
            for pattern in ad_text_patterns:
                if re.search(pattern, post_html) or re.search(pattern, post_text):
                    return f"Found explicit ad text: {pattern}"
            
            # Look for ad-related data attributes (more reliable than text)
            ad_attributes = ['data-testid*="promoted"', 'data-testid*="ad"', 'aria-label*="Promoted"']
            for attr in ad_attributes:
                try:
                    elements = post_element.find_elements(By.CSS_SELECTOR, f'[{attr}]')
                    if elements:
                        return f"Found ad attribute: {attr}"
                except:
                    continue
            
            return None
        except Exception as e:
            logger.logger.debug(f"Error checking ad indicators: {str(e)}")
            return None
    
    def _analyze_promotional_language(self, text: str) -> int:
        """Analyze text for VERY SPECIFIC promotional language patterns"""
        score = 0
        text_lower = text.lower()
        
        # Only count very specific promotional patterns
        for pattern in self.ad_language_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            score += len(matches)
        
        return score
    
    def _extract_suspicious_domains(self, post_element, post_text: str) -> List[str]:
        """Extract and check domains against KNOWN suspicious list only"""
        suspicious_found = []
        
        try:
            # Get all links in the post
            links = post_element.find_elements(By.TAG_NAME, 'a')
            
            for link in links:
                href = link.get_attribute('href')
                if href:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(href).netloc.lower()
                        # Only flag if it's in our KNOWN suspicious domains list
                        if domain in self.suspicious_domains:
                            suspicious_found.append(domain)
                    except:
                        continue
            
            # Check for domains mentioned in text (only known suspicious ones)
            for domain in self.suspicious_domains:
                if domain in post_text.lower():
                    suspicious_found.append(domain)
        
        except Exception as e:
            logger.logger.debug(f"Error extracting domains: {str(e)}")
        
        return list(set(suspicious_found))  # Remove duplicates
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if a domain has suspicious patterns"""
        suspicious_patterns = [
            r'.*ai.*tools?.*',
            r'.*crypto.*(?:master|pro|guru).*',
            r'.*trading.*(?:bot|system|auto).*',
            r'.*blockchain.*(?:academy|university|course).*',
            r'.*(?:free|easy|quick).*(?:crypto|trading|ai).*'
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, domain):
                return True
        
        return False
    
    def _is_suspicious_username(self, username: str) -> bool:
        """Check if username follows suspicious patterns"""
        username_lower = username.lower().replace('@', '')
        
        for pattern in self.suspicious_username_patterns:
            if re.match(pattern, username_lower):
                return True
        
        return False
    
    def _is_obvious_spam_username(self, username: str) -> bool:
        """Check for VERY OBVIOUS spam username patterns only"""
        if not username:
            return False
            
        username_lower = username.lower().replace('@', '')
        
        # Only check for very obvious spam patterns
        for pattern in self.suspicious_username_patterns:
            if re.match(pattern, username_lower):
                return True
        
        return False
    
    def _is_generic_ai_crypto_promo(self, text: str) -> bool:
        """Detect generic AI/crypto promotional content"""
        text_lower = text.lower()
        
        # Look for combinations of AI/crypto terms with promotional language
        ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'algorithm']
        crypto_terms = ['crypto', 'bitcoin', 'blockchain', 'trading', 'defi']
        promo_terms = ['powerful', 'revolutionary', 'cutting-edge', 'breakthrough', 'unlimited']
        
        has_tech_term = any(term in text_lower for term in ai_terms + crypto_terms)
        has_promo_term = any(term in text_lower for term in promo_terms)
        
        return has_tech_term and has_promo_term
    
    def _has_promotional_cta(self, post_element) -> bool:
        """Check for promotional call-to-action elements"""
        try:
            cta_selectors = [
                'a:contains("Get Started")',
                'a:contains("Learn More")',
                'button:contains("Try Now")',
                'a:contains("Sign Up")',
                'a:contains("Join Now")'
            ]
            
            for selector in cta_selectors:
                try:
                    elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def _has_obvious_fake_testimonials(self, text: str) -> bool:
        """Detect VERY OBVIOUS fake testimonials only"""
        text_lower = text.lower()
        
        # Very specific fake testimonial patterns
        obvious_fake_patterns = [
            r'\b\d+\.\d+\s*rating from \d+(?:k|\+|,\d+)?\+?\s*(?:users?|customers?)\b',  # "4.9 rating from 1000+ users"
            r'\btrusted by \d+(?:k|\+|,\d+)?\+?\s*(?:users?|customers?|companies)\b',    # "trusted by 10k+ users"
            r'\b#1\s*(?:rated|app|tool|platform|solution)\b',                           # "#1 rated app"
            r'\bover \d+(?:k|\+|,\d+)?\+?\s*(?:satisfied|happy)\s*(?:users?|customers?)\b' # "over 1000+ satisfied users"
        ]
        
        for pattern in obvious_fake_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False


class AdvancedPostDetector:
    """Advanced post detection with multiple strategies and fallbacks"""
    
    def __init__(self, browser_manager, selector_manager, config):
        self.browser_manager = browser_manager
        self.selector_manager = selector_manager
        self.config = config
        self.honeypot_detector = HoneypotDetector()
        
        # Post detection strategies in order of preference
        self.detection_strategies = [
            self._detect_posts_primary,
            self._detect_posts_article_fallback,
            self._detect_posts_text_based,
            self._detect_posts_link_based,
            self._detect_posts_emergency
        ]
    
    def find_all_posts(self, max_attempts: int = 3) -> List[Any]:
        """Find all posts using multiple detection strategies"""
        logger.logger.info("Starting advanced post detection")
        
        all_posts = []
        successful_strategy = None
        
        for attempt in range(max_attempts):
            for strategy_name, strategy_func in zip(
                ['primary', 'article', 'text_based', 'link_based', 'emergency'],
                self.detection_strategies
            ):
                try:
                    logger.logger.debug(f"Attempt {attempt + 1}: Trying {strategy_name} strategy")
                    posts = strategy_func()
                    
                    if posts and len(posts) > 0:
                        logger.logger.info(f"{strategy_name} strategy found {len(posts)} posts")
                        all_posts.extend(posts)
                        successful_strategy = strategy_name
                        break
                        
                except Exception as e:
                    logger.logger.debug(f"{strategy_name} strategy failed: {str(e)}")
                    continue
            
            if all_posts:
                break
            
            # Wait before retry
            if attempt < max_attempts - 1:
                logger.logger.info(f"No posts found, waiting before retry {attempt + 2}")
                time.sleep(2)
        
        # Remove duplicates and filter honeypots
        unique_posts = self._deduplicate_posts(all_posts)
        filtered_posts = self._filter_honeypot_posts(unique_posts)
        
        logger.logger.info(f"Final result: {len(filtered_posts)} posts after deduplication and filtering")
        if successful_strategy:
            logger.logger.info(f"Most successful strategy: {successful_strategy}")
        
        return filtered_posts
    
    def _detect_posts_primary(self) -> List[Any]:
        """Primary post detection using best known selectors"""
        logger.logger.info("🔍 Starting primary post detection")
        
        selector_config = self.selector_manager.get_selector_config('timeline_posts')
        if not selector_config:
            logger.logger.warning("No selector config for timeline_posts")
            return []
        
        best_selector = self.selector_manager.get_best_selector('timeline_posts')
        if not best_selector:
            logger.logger.warning("No best selector for timeline_posts")
            return []
        
        logger.logger.info(f"Using primary selector: {best_selector}")
        posts = self.browser_manager.safe_find_elements(best_selector)
        logger.logger.info(f"Primary selector '{best_selector}' found {len(posts)} elements")
        
        self.selector_manager.record_selector_result('timeline_posts', best_selector, len(posts) > 0)
        
        return posts

    def _detect_posts_article_fallback(self) -> List[Any]:
        """Fallback using article elements"""
        logger.logger.info("Starting article fallback detection")
        
        article_selectors = [
            'article[data-testid="tweet"]',
            'article[role="article"]', 
            'div[role="article"]'
        ]
        
        for selector in article_selectors:
            logger.logger.info(f"Trying article selector: {selector}")
            posts = self.browser_manager.safe_find_elements(selector)
            logger.logger.info(f"Article selector '{selector}' found {len(posts)} elements")
            
            if posts:
                return posts
        
        return []
    
    def _filter_valid_tweet_elements(self, elements: List[Any]) -> List[Any]:
        """Filter cellInnerDiv elements to only include actual tweets"""
        valid_elements = []
        
        for i, element in enumerate(elements):
            try:
                # Check if this cellInnerDiv contains a tweet article
                tweet_article = element.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                
                if tweet_article:
                    # Get text content to verify it's substantial
                    text = self.browser_manager.safe_get_text(element)
                    
                    if text and len(text.strip()) > 20:  # Must have substantial text
                        logger.logger.debug(f"✅ Valid tweet element {i+1}: {len(text)} chars")
                        valid_elements.append(element)
                    else:
                        logger.logger.debug(f"❌ Element {i+1} has insufficient text: {len(text) if text else 0} chars")
                else:
                    logger.logger.debug(f"❌ Element {i+1} has no tweet article")
                    
            except Exception as e:
                logger.logger.debug(f"💥 Error validating element {i+1}: {str(e)}")
                continue
        
        logger.logger.debug(f"📊 Filtered {len(valid_elements)}/{len(elements)} valid tweet elements")
        return valid_elements
    
    def _validate_tweet_articles(self, articles: List[Any]) -> List[Any]:
        """Validate that article elements are actual tweets with content"""
        valid_articles = []
        
        for i, article in enumerate(articles):
            try:
                # Check for tweet text content
                tweet_text_elements = article.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                
                if tweet_text_elements:
                    # Get full article text
                    text = self.browser_manager.safe_get_text(article)
                    
                    if text and len(text.strip()) > 30:  # Tweets should have substantial content
                        logger.logger.debug(f"✅ Valid tweet article {i+1}: {len(text)} chars")
                        valid_articles.append(article)
                    else:
                        logger.logger.debug(f"❌ Article {i+1} has insufficient content: {len(text) if text else 0} chars")
                else:
                    logger.logger.debug(f"❌ Article {i+1} has no tweet text element")
                    
            except Exception as e:
                logger.logger.debug(f"💥 Error validating article {i+1}: {str(e)}")
                continue
        
        logger.logger.debug(f"📊 Validated {len(valid_articles)}/{len(articles)} tweet articles")
        return valid_articles
    
    def _detect_posts_text_based(self) -> List[Any]:
        """Text-based detection by finding tweet text and getting parents"""
        text_selectors = ['div[data-testid="tweetText"]', 'div[lang]', 'span[data-testid="tweetText"]']
        
        for text_selector in text_selectors:
            text_elements = self.browser_manager.safe_find_elements(text_selector)
            if text_elements:
                # Get parent containers that likely represent posts
                post_containers = []
                for text_elem in text_elements:
                    try:
                        # Navigate up the DOM to find post container
                        parent = text_elem
                        for _ in range(5):  # Go up max 5 levels
                            parent = self.browser_manager.browser.driver.execute_script(
                                "return arguments[0].parentElement;", parent
                            )
                            if parent and self._looks_like_post_container(parent):
                                post_containers.append(parent)
                                break
                    except:
                        continue
                
                if post_containers:
                    return post_containers
        
        return []
    
    def _detect_posts_link_based(self) -> List[Any]:
        """Detection based on status links"""
        status_links = self.browser_manager.safe_find_elements('a[href*="/status/"]')
        
        post_containers = []
        for link in status_links:
            try:
                # Navigate up to find the post container
                parent = link
                for _ in range(6):  # Go up max 6 levels
                    parent = self.browser_manager.browser.driver.execute_script(
                        "return arguments[0].parentElement;", parent
                    )
                    if parent and self._looks_like_post_container(parent):
                        post_containers.append(parent)
                        break
            except:
                continue
        
        return post_containers
    
    def _detect_posts_emergency(self) -> List[Any]:
        """Emergency detection using broad selectors"""
        logger.logger.warning("Using emergency post detection")
        
        emergency_selectors = [
            'div[class*="css-1dbjc4n"]',  # Common Twitter class
            'div[dir="auto"]',           # Content direction attribute
            'div[data-testid]'           # Any element with testid
        ]
        
        for selector in emergency_selectors:
            elements = self.browser_manager.safe_find_elements(selector)
            
            # Filter elements that might be posts
            potential_posts = []
            for elem in elements:
                try:
                    # Check if element might be a post
                    if self._might_be_post_element(elem):
                        potential_posts.append(elem)
                except:
                    continue
            
            if len(potential_posts) > 5:  # Found reasonable number
                return potential_posts[:20]  # Limit to prevent noise
        
        return []
    
    def _looks_like_post_container(self, element) -> bool:
        """Check if element looks like a post container"""
        try:
            # Check element size
            size = element.size
            if size['height'] < 50 or size['width'] < 200:
                return False
            
            # Check for typical post content
            inner_html = element.get_attribute('innerHTML') or ""
            has_text_content = len(inner_html.strip()) > 100
            has_status_link = '/status/' in inner_html
            has_user_mention = '@' in inner_html or 'User-Name' in inner_html
            
            return has_text_content and (has_status_link or has_user_mention)
            
        except:
            return False
    
    def _might_be_post_element(self, element) -> bool:
        """Check if element might be a post (loose criteria)"""
        try:
            # Check for minimum size
            size = element.size
            if size['height'] < 30:
                return False
            
            # Check for text content
            text = self.browser_manager.safe_get_text(element)
            if len(text) < 75:
                return False
            
            # Check for tweet-like characteristics
            inner_html = element.get_attribute('innerHTML') or ""
            tweet_indicators = [
                'status/', '@', 'reply', 'retweet', 'like',
                'User-Name', 'tweetText', 'data-testid'
            ]
            
            indicator_count = sum(1 for indicator in tweet_indicators if indicator in inner_html)
            return indicator_count >= 1
            
        except:
            return False
    
    def _deduplicate_posts(self, posts: List[Any]) -> List[Any]:
        """Remove duplicate posts based on normalized content similarity"""
        unique_posts = []
        seen_signatures = set()
        
        logger.logger.debug(f"Starting deduplication of {len(posts)} posts")
        
        for i, post in enumerate(posts):
            try:
                # Extract text content
                text = self.browser_manager.safe_get_text(post)
                
                # Skip posts with no text content
                if not text or len(text.strip()) == 0:
                    logger.logger.debug(f"Post {i+1}: Skipping post with no text content")
                    continue
                
                # Normalize text for better deduplication
                # Remove extra whitespace, convert to lowercase, strip
                normalized_text = ' '.join(text.strip().lower().split())
                
                # Use normalized full text as signature (no truncation)
                signature = normalized_text
                signature_hash = hashlib.md5(signature.encode('utf-8')).hexdigest()
                
                if signature_hash not in seen_signatures:
                    seen_signatures.add(signature_hash)
                    unique_posts.append(post)
                    logger.logger.debug(f"Post {i+1}: Added unique post - '{text[:50]}...'")
                else:
                    logger.logger.debug(f"Post {i+1}: Filtered duplicate post - '{text[:50]}...'")
                    
            except Exception as e:
                # If we can't create signature, include it to be safe
                logger.logger.debug(f"Post {i+1}: Error creating signature ({str(e)}), including post to be safe")
                unique_posts.append(post)
        
        removed_count = len(posts) - len(unique_posts)
        if removed_count > 0:
            logger.logger.info(f"Deduplication complete: Removed {removed_count} duplicate posts, {len(unique_posts)} unique posts remain")
        else:
            logger.logger.debug(f"Deduplication complete: No duplicates found, all {len(unique_posts)} posts are unique")
        
        return unique_posts
    
    def _filter_honeypot_posts(self, posts: List[Any]) -> List[Any]:
        """Filter out honeypot and promotional content"""
        # FIXED: Corrected the backwards logic
        if self.config.skip_honeypot_detection:
            # If skipping honeypot detection, return all posts without filtering
            logger.logger.debug("Skipping honeypot detection as configured")
            return posts
        
        # Proceed with honeypot filtering
        logger.logger.info(f"Filtering {len(posts)} posts for honeypot/promotional content")
        filtered_posts = []
        honeypot_count = 0
        filter_details = {
            'explicit_ads': 0,
            'promotional_language': 0,
            'suspicious_domains': 0,
            'suspicious_usernames': 0,
            'generic_ai_promo': 0,
            'errors': 0
        }
        
        for post in posts:
            try:
                # Extract basic info for honeypot detection
                post_text = self.browser_manager.safe_get_text(post)
                
                # Skip posts with no text content
                if not post_text or len(post_text.strip()) == 0:
                    logger.logger.debug("Skipping post with no text content")
                    continue
                
                # Try to get author handle
                author_handle = ""
                try:
                    author_elements = post.find_elements(By.CSS_SELECTOR, 'div[data-testid="User-Name"] a')
                    if author_elements:
                        author_handle = author_elements[0].get_attribute('href') or ""
                        author_handle = author_handle.split('/')[-1] if '/' in author_handle else ""
                except Exception as author_error:
                    logger.logger.debug(f"Could not extract author handle: {str(author_error)}")
                    pass
                
                # Check if it's honeypot content
                is_honeypot, reason = self.honeypot_detector.is_honeypot_content(
                    post, post_text, author_handle
                )
                
                if is_honeypot:
                    honeypot_count += 1
                    # Track the specific reason for filtering statistics
                    if "ad indicator" in reason.lower():
                        filter_details['explicit_ads'] += 1
                    elif "promotional language" in reason.lower():
                        filter_details['promotional_language'] += 1
                    elif "suspicious domain" in reason.lower():
                        filter_details['suspicious_domains'] += 1
                    elif "suspicious username" in reason.lower():
                        filter_details['suspicious_usernames'] += 1
                    elif "generic ai" in reason.lower() or "generic crypto" in reason.lower():
                        filter_details['generic_ai_promo'] += 1
                    
                    logger.logger.debug(f"Filtered honeypot post: {reason} | Text preview: '{post_text[:50]}...'")
                else:
                    # Post passed honeypot detection, include it
                    filtered_posts.append(post)
                    
            except Exception as e:
                filter_details['errors'] += 1
                logger.logger.debug(f"Error filtering post: {str(e)}")
                # If we can't determine due to error, include it to be safe
                # This prevents legitimate posts from being lost due to technical issues
                filtered_posts.append(post)
        
        # Log comprehensive filtering results
        if honeypot_count > 0:
            logger.logger.info(f"🛡️ Honeypot filtering complete:")
            logger.logger.info(f"   • Input posts: {len(posts)}")
            logger.logger.info(f"   • Filtered out: {honeypot_count} honeypots/ads")
            logger.logger.info(f"   • Remaining posts: {len(filtered_posts)}")
            logger.logger.info(f"   • Filter breakdown: {filter_details}")
        else:
            logger.logger.info(f"✅ No honeypot content detected in {len(posts)} posts")
        
        return filtered_posts


class PostDataExtractor:
    """Advanced post data extraction with CURRENT Twitter/X selectors"""
    
    def __init__(self, browser_manager, selector_manager, config, extraction_cache=None):
        self.browser_manager = browser_manager
        self.selector_manager = selector_manager
        self.config = config
        self.cache = extraction_cache
        
        # Updated metric parsing patterns based on real Twitter/X aria-labels
        self.metric_patterns = {
            'replies': [
                r'(\d+(?:,\d+)*)\s+replies?',  # "9 replies"
                r'(\d+(?:,\d+)*)\s+Replies?',  # "9 Replies"
                r'(\d+(?:,\d+)*)\s+repl',      # Fallback
            ],
            'retweets': [
                r'(\d+(?:,\d+)*)\s+reposts?',  # "5 reposts"
                r'(\d+(?:,\d+)*)\s+Reposts?',  # "5 Reposts"
                r'(\d+(?:,\d+)*)\s+retweets?', # Fallback
            ],
            'likes': [
                r'(\d+(?:,\d+)*)\s+[Ll]ikes?',  # "29 Likes"
                r'(\d+(?:,\d+)*)\s+[Ll]ike',    # "29 Like"
            ],
            'views': [
                r'(\d+(?:,\d+)*(?:\.\d+)?[KkMm]?)\s+views?',  # "5.3K views"
                r'(\d+(?:,\d+)*)\s+views?',                   # "5376 views"
            ]
        }

    def extract_post_data(self, post_element) -> Optional[Dict[str, Any]]:
        """Extract comprehensive post data using WORKING selectors"""
        try:
            # Extract all components with simplified, working methods
            post_id = self._extract_post_id_simple(post_element)
            post_url = self._extract_post_url_simple(post_element)
            text_content = self._extract_text_content_simple(post_element)
            author_info = self._extract_author_info_simple(post_element)
            timestamp_info = self._extract_timestamp_simple(post_element)
            metrics = self._extract_metrics_simple(post_element)
            media_info = self._extract_media_simple(post_element)
            
            # Skip posts with no text content (likely ads or broken posts)
            if not text_content or len(text_content.strip()) < 2:
                logger.logger.debug("Skipping post with no text content")
                return None
            
            # Compile final post data
            post_data = {
                'post_id': post_id,
                'post_url': post_url,
                'text': text_content,
                'author_name': author_info.get('name', 'Unknown'),
                'author_handle': author_info.get('handle', 'Unknown'),
                'author_verified': author_info.get('verified', False),
                'author_profile_url': author_info.get('profile_url'),
                'timestamp': timestamp_info.get('datetime'),
                'timestamp_text': timestamp_info.get('text'),
                'metrics': {
                    'replies': metrics.get('replies', 0),
                    'retweets': metrics.get('retweets', 0), 
                    'likes': metrics.get('likes', 0),
                    'views': metrics.get('views', 0),
                    'engagement_score': metrics.get('replies', 0) + metrics.get('retweets', 0) + metrics.get('likes', 0)
                },
                'media': media_info,
                'is_reply': self._detect_reply(post_element),
                'is_retweet': self._detect_retweet(post_element),
                'scraped_at': strip_timezone(datetime.now())
            }
            
            # Cache the result if enabled
            if self.cache:
                self._cache_extracted_data(post_element, post_data)
            
            logger.logger.debug(f"✅ Successfully extracted post: {text_content[:50]}...")
            return post_data
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract post data: {str(e)}")
            logger.logger.debug(f"Post element HTML preview: {post_element.get_attribute('outerHTML')[:200]}...")
            return None
    
    def _extract_post_id_simple(self, post_element) -> str:
        """Extract post ID using WORKING methods"""
        try:
            # Strategy 1: Look for status links (THIS WORKS)
            status_links = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            for link in status_links:
                href = link.get_attribute('href')
                if href and '/photo/' not in href and '/analytics' not in href:
                    # Extract ID from URL: https://x.com/username/status/1955275580636536902
                    match = re.search(r'/status/(\d+)$', href)
                    if match:
                        return match.group(1)
            
            # Strategy 2: Use first status link if no clean one found
            if status_links:
                href = status_links[0].get_attribute('href')
                if href:
                    match = re.search(r'/status/(\d+)', href)
                    if match:
                        return match.group(1)
            
            # Strategy 3: Generate from content hash
            text = self._extract_text_content_simple(post_element)
            if text:
                content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                return f"hash_{content_hash}_{int(time.time())}"
            
            # Fallback: Use timestamp
            return f"post_{int(time.time() * 1000)}"
            
        except Exception as e:
            logger.logger.debug(f"Error extracting post ID: {str(e)}")
            return f"error_{int(time.time() * 1000)}"

    def _extract_post_url_simple(self, post_element) -> Optional[str]:
        """Extract post URL using WORKING methods"""
        try:
            # Look for clean status links (avoid /photo/ and /analytics)
            status_links = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            for link in status_links:
                href = link.get_attribute('href')
                if href and '/photo/' not in href and '/analytics' not in href:
                    return href
            
            # Return first status link if no clean one found
            if status_links:
                href = status_links[0].get_attribute('href')
                if href:
                    # Clean up the URL to main status
                    clean_url = re.sub(r'/(?:photo|analytics).*$', '', href)
                    return clean_url
                    
            return None
            
        except Exception as e:
            logger.logger.debug(f"Error extracting post URL: {str(e)}")
            return None

    def _extract_text_content_simple(self, post_element) -> str:
        """Extract text content using WORKING selectors"""
        try:
            # Primary selector (WORKS PERFECTLY)
            text_elem = post_element.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            if text_elem:
                text = text_elem.get_attribute('textContent') or text_elem.text
                if text and len(text.strip()) > 0:
                    return text.strip()
            
            # Fallback: Try alternative selectors
            fallback_selectors = [
                '[data-testid="tweetText"]',
                'div[lang]',
                'div[dir="auto"]'
            ]
            
            for selector in fallback_selectors:
                try:
                    elem = post_element.find_element(By.CSS_SELECTOR, selector)
                    if elem:
                        text = elem.get_attribute('textContent') or elem.text
                        if text and len(text.strip()) > 5:  # Avoid getting author names
                            return text.strip()
                except:
                    continue
            
            return ""
            
        except Exception as e:
            logger.logger.debug(f"Error extracting text content: {str(e)}")
            return ""

    def _extract_author_info_simple(self, post_element) -> Dict[str, Any]:
        """Extract author info using WORKING selectors"""
        try:
            author_info = {
                'name': 'Unknown',
                'handle': 'Unknown', 
                'verified': False,
                'profile_url': None
            }
            
            # Get User-Name element (WORKS)
            user_name_elem = post_element.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"]')
            if user_name_elem:
                # Extract name from first link
                name_links = user_name_elem.find_elements(By.CSS_SELECTOR, 'a[role="link"] span')
                if name_links:
                    author_info['name'] = name_links[0].text.strip()
                
                # Extract handle and profile URL from links
                profile_links = user_name_elem.find_elements(By.CSS_SELECTOR, 'a[href*="/"]')
                for link in profile_links:
                    href = link.get_attribute('href')
                    if href and '/status/' not in href:
                        author_info['profile_url'] = href
                        # Extract handle from URL
                        handle_match = re.search(r'x\.com/([^/?]+)', href)
                        if handle_match:
                            author_info['handle'] = f"@{handle_match.group(1)}"
                        break
                
                # Check for verification badge
                verified_elem = user_name_elem.find_elements(By.CSS_SELECTOR, 'svg[data-testid="icon-verified"]')
                author_info['verified'] = len(verified_elem) > 0
            
            return author_info
            
        except Exception as e:
            logger.logger.debug(f"Error extracting author info: {str(e)}")
            return {
                'name': 'Unknown',
                'handle': 'Unknown',
                'verified': False,
                'profile_url': None
            }

    def _extract_timestamp_simple(self, post_element) -> Dict[str, Any]:
        """Extract timestamp using WORKING selectors"""
        try:
            # Find time element (WORKS)
            time_elem = post_element.find_element(By.CSS_SELECTOR, 'time')
            if time_elem:
                datetime_str = time_elem.get_attribute('datetime')
                text_content = time_elem.text.strip()
                
                timestamp = None
                if datetime_str:
                    try:
                        # Parse ISO datetime string
                        timestamp = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        timestamp = strip_timezone(timestamp)  # Convert to naive datetime
                    except:
                        pass
                
                return {
                    'datetime': timestamp,
                    'text': text_content,
                    'raw_datetime': datetime_str
                }
            
            return {'datetime': None, 'text': '', 'raw_datetime': None}
            
        except Exception as e:
            logger.logger.debug(f"Error extracting timestamp: {str(e)}")
            return {'datetime': None, 'text': '', 'raw_datetime': None}

    def _extract_metrics_simple(self, post_element) -> Dict[str, int]:
        """Extract engagement metrics using WORKING selectors and aria-labels"""
        try:
            metrics = {
                'replies': 0,
                'retweets': 0, 
                'likes': 0,
                'views': 0
            }
            
            # Strategy 1: Use individual metric buttons (WORKS BEST)
            metric_buttons = {
                'replies': 'button[data-testid="reply"]',
                'retweets': 'button[data-testid="retweet"]', 
                'likes': 'button[data-testid="like"]'
            }
            
            for metric_name, selector in metric_buttons.items():
                try:
                    button = post_element.find_element(By.CSS_SELECTOR, selector)
                    if button:
                        # Get count from aria-label
                        aria_label = button.get_attribute('aria-label') or ""
                        
                        # Parse from aria-label patterns
                        for pattern in self.metric_patterns[metric_name]:
                            match = re.search(pattern, aria_label, re.IGNORECASE)
                            if match:
                                count_str = match.group(1).replace(',', '')
                                try:
                                    metrics[metric_name] = int(count_str)
                                    break
                                except:
                                    pass
                        
                        # Fallback: get text content if aria-label failed
                        if metrics[metric_name] == 0:
                            text_content = button.text.strip()
                            if text_content and text_content.isdigit():
                                metrics[metric_name] = int(text_content)
                                
                except:
                    continue
            
            # Strategy 2: Extract views from analytics link or group aria-label
            try:
                # Look for analytics link with views
                analytics_links = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/analytics"]')
                for link in analytics_links:
                    aria_label = link.get_attribute('aria-label') or ""
                    view_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?[KkMm]?)\s+views?', aria_label, re.IGNORECASE)
                    if view_match:
                        view_str = view_match.group(1)
                        # Convert K/M notation
                        if 'k' in view_str.lower():
                            metrics['views'] = int(float(view_str.lower().replace('k', '')) * 1000)
                        elif 'm' in view_str.lower():
                            metrics['views'] = int(float(view_str.lower().replace('m', '')) * 1000000)
                        else:
                            metrics['views'] = int(view_str.replace(',', ''))
                        break
                
                # Fallback: look in group aria-label
                if metrics['views'] == 0:
                    group_elem = post_element.find_element(By.CSS_SELECTOR, 'div[role="group"]')
                    if group_elem:
                        aria_label = group_elem.get_attribute('aria-label') or ""
                        view_match = re.search(r'(\d+(?:,\d+)*)\s+views?', aria_label, re.IGNORECASE)
                        if view_match:
                            metrics['views'] = int(view_match.group(1).replace(',', ''))
                        
            except:
                pass
            
            logger.logger.debug(f"Extracted metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.logger.debug(f"Error extracting metrics: {str(e)}")
            return {'replies': 0, 'retweets': 0, 'likes': 0, 'views': 0}

    def _extract_media_simple(self, post_element) -> Dict[str, Any]:
        """Extract media info using WORKING selectors"""
        try:
            media_info = {
                'has_image': False,
                'has_video': False,
                'has_external_link': False,
                'external_domains': []
            }
            
            # Check for images (WORKS)
            image_elems = post_element.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetPhoto"]')
            media_info['has_image'] = len(image_elems) > 0
            
            # Check for videos 
            video_elems = post_element.find_elements(By.CSS_SELECTOR, 'div[data-testid="videoPlayer"]')
            media_info['has_video'] = len(video_elems) > 0
            
            # Check for external links
            links = post_element.find_elements(By.CSS_SELECTOR, 'a[href]')
            for link in links:
                href = link.get_attribute('href')
                if href and not href.startswith('https://x.com/') and not href.startswith('https://twitter.com/'):
                    media_info['has_external_link'] = True
                    try:
                        domain = urlparse(href).netloc
                        if domain and domain not in media_info['external_domains']:
                            media_info['external_domains'].append(domain)
                    except:
                        pass
            
            return media_info
            
        except Exception as e:
            logger.logger.debug(f"Error extracting media info: {str(e)}")
            return {
                'has_image': False,
                'has_video': False, 
                'has_external_link': False,
                'external_domains': []
            }

    def _detect_reply(self, post_element) -> bool:
        """Detect if post is a reply"""
        try:
            # Look for reply indicators
            reply_indicators = [
                'div[data-testid="reply"]',
                '[aria-label*="Replying to"]',
                'span:contains("Replying to")'
            ]
            
            for selector in reply_indicators:
                if post_element.find_elements(By.CSS_SELECTOR, selector):
                    return True
            
            return False
        except:
            return False

    def _detect_retweet(self, post_element) -> bool:
        """Detect if post is a retweet"""
        try:
            # Look for retweet indicators
            retweet_indicators = [
                '[data-testid*="retweet"]',
                'span:contains("retweeted")',
                'span:contains("Retweeted")'
            ]
            
            for selector in retweet_indicators:
                if post_element.find_elements(By.CSS_SELECTOR, selector):
                    return True
            
            return False
        except:
            return False

    def _cache_extracted_data(self, post_element, post_data):
        """Cache extracted data if caching is enabled"""
        try:
            if self.cache and post_data:
                cache_key = post_data.get('post_id', f"post_{int(time.time())}")
                self.cache.set(cache_key, post_data)
        except Exception as e:
            logger.logger.debug(f"Failed to cache data: {str(e)}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Timeline Scraper - Part 3: Intelligent Content Analysis & Filtering
Advanced content analysis, market relevance detection, and opportunity scoring
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from datetime_utils import strip_timezone, safe_datetime_diff
from utils.logger import logger


class ContentCategory(Enum):
    """Content categories for classification"""
    CRYPTO_DISCUSSION = "crypto_discussion"
    MARKET_ANALYSIS = "market_analysis"
    TECH_NEWS = "tech_news"
    QUESTION_SEEKING = "question_seeking"
    EDUCATIONAL = "educational"
    PRICE_DISCUSSION = "price_discussion"
    TRADING_SIGNALS = "trading_signals"
    PROJECT_UPDATES = "project_updates"
    GENERAL_TECH = "general_tech"
    RECEPTIVE_USER = "receptive_user"
    LOW_VALUE = "low_value"


class SentimentType(Enum):
    """Sentiment classifications"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    EXCITED = "excited"
    CONCERNED = "concerned"


@dataclass
class ContentAnalysis:
    """Comprehensive content analysis results"""
    categories: List[ContentCategory] = field(default_factory=list)
    primary_category: Optional[ContentCategory] = None
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_confidence: float = 0.0
    
    # Keywords and entities
    crypto_keywords: List[str] = field(default_factory=list)
    tech_keywords: List[str] = field(default_factory=list)
    finance_keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    
    # Content characteristics
    has_question: bool = False
    question_type: Optional[str] = None
    has_price_mention: bool = False
    has_percentage: bool = False
    word_count: int = 0
    readability_score: float = 0.0
    
    # Opportunity indicators
    engagement_potential: float = 0.0
    reply_opportunity_score: float = 0.0
    conversation_starter_potential: float = 0.0
    educational_opportunity: float = 0.0


@dataclass
class MarketRelevanceScore:
    """Market relevance scoring breakdown"""
    base_score: float = 0.0
    keyword_bonus: float = 0.0
    question_bonus: float = 0.0
    engagement_bonus: float = 0.0
    recency_bonus: float = 0.0
    author_bonus: float = 0.0
    final_score: float = 0.0
    relevance_tier: str = "low"  # low, medium, high, premium


class AdvancedContentAnalyzer:
    """Sophisticated content analysis engine"""
    
    def __init__(self):
        self._initialize_keyword_databases()
        self._initialize_pattern_databases()
        self._initialize_scoring_weights()
    
    def _initialize_keyword_databases(self):
        """Initialize comprehensive keyword databases"""
        # Crypto keywords by category
        self.crypto_keywords = {
            'major_coins': [
                'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'cardano', 'ada',
                'polkadot', 'dot', 'avalanche', 'avax', 'chainlink', 'link', 'polygon', 'matic',
                'binance', 'bnb', 'ripple', 'xrp', 'dogecoin', 'doge', 'shiba', 'shib',
                'litecoin', 'ltc', 'monero', 'xmr', 'near', 'cosmos', 'atom', 'algorand', 'algo'
            ],
            'defi_terms': [
                'defi', 'decentralized finance', 'yield farming', 'liquidity mining', 'staking',
                'uniswap', 'pancakeswap', 'compound', 'aave', 'makerdao', 'dai', 'usdc', 'usdt',
                'liquidity pool', 'amm', 'automated market maker', 'tvl', 'total value locked',
                'governance token', 'dex', 'decentralized exchange', 'lending protocol'
            ],
            'nft_terms': [
                'nft', 'non fungible token', 'opensea', 'rarible', 'foundation', 'superrare',
                'collectible', 'digital art', 'pfp', 'profile picture', 'mint', 'minting',
                'floor price', 'collection', 'metadata', 'royalties', 'gas fees'
            ],
            'technical_terms': [
                'blockchain', 'smart contract', 'consensus', 'proof of stake', 'proof of work',
                'mining', 'validator', 'node', 'hash', 'merkle tree', 'fork', 'hardfork',
                'layer 2', 'scaling', 'rollup', 'sidechain', 'cross chain', 'interoperability',
                'oracle', 'api', 'sdk', 'protocol', 'mainnet', 'testnet', 'genesis block'
            ],
            'trading_terms': [
                'trading', 'hodl', 'fomo', 'fud', 'dyor', 'ath', 'all time high', 'dip',
                'moon', 'lambo', 'diamond hands', 'paper hands', 'whale', 'retail',
                'support', 'resistance', 'bull run', 'bear market', 'correction', 'pump',
                'dump', 'rug pull', 'exit scam', 'market cap', 'volume', 'volatility'
            ]
        }
        
        # Finance and trading keywords
        self.finance_keywords = {
            'general_finance': [
                'investment', 'portfolio', 'asset', 'equity', 'bond', 'stock', 'share',
                'dividend', 'yield', 'return', 'profit', 'loss', 'gain', 'capital',
                'market', 'exchange', 'broker', 'fund', 'etf', 'mutual fund', 'hedge fund'
            ],
            'economic_terms': [
                'inflation', 'deflation', 'recession', 'economy', 'gdp', 'fed', 'interest rate',
                'monetary policy', 'fiscal policy', 'stimulus', 'quantitative easing',
                'central bank', 'currency', 'forex', 'commodities', 'gold', 'silver', 'oil'
            ],
            'analysis_terms': [
                'technical analysis', 'fundamental analysis', 'chart', 'pattern', 'trend',
                'indicator', 'rsi', 'macd', 'moving average', 'fibonacci', 'candlestick',
                'volume', 'momentum', 'oscillator', 'breakout', 'reversal', 'continuation'
            ]
        }
        
        # Tech keywords
        self.tech_keywords = {
            'general_tech': [
                'technology', 'innovation', 'software', 'hardware', 'application', 'platform',
                'system', 'network', 'internet', 'web', 'mobile', 'cloud', 'server',
                'database', 'api', 'algorithm', 'programming', 'code', 'development'
            ],
            'emerging_tech': [
                'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
                'neural network', 'automation', 'robotics', 'iot', 'internet of things',
                'virtual reality', 'vr', 'augmented reality', 'ar', 'quantum computing',
                'edge computing', '5g', 'cybersecurity', 'big data', 'data science'
            ],
            'web3_tech': [
                'web3', 'metaverse', 'dao', 'decentralized autonomous organization',
                'ipfs', 'filecoin', 'arweave', 'ceramic', 'lens protocol', 'ens',
                'ethereum name service', 'identity', 'self sovereign', 'zero knowledge'
            ]
        }
    
    def _initialize_pattern_databases(self):
        """Initialize regex patterns for content analysis"""
        # Question patterns
        self.question_patterns = {
            'seeking_advice': [
                r'\b(?:what|which|how) (?:should|would|do) (?:i|you|we)\b',
                r'\b(?:any|anyone) (?:recommend|suggest|know)\b',
                r'\b(?:best|good) (?:way|method|approach|strategy)\b',
                r'\b(?:help|advice|guidance|recommendation)\b',
                r'\bneed (?:help|advice|suggestions?)\b'
            ],
            'information_seeking': [
                r'\b(?:what|where|when|why|how) (?:is|are|do|does|can|will)\b',
                r'\b(?:explain|understand|learn|know) (?:about|how|what)\b',
                r'\b(?:difference between|compare|vs|versus)\b',
                r'\b(?:beginner|new to|just started|first time)\b'
            ],
            'opinion_seeking': [
                r'\b(?:thoughts|opinion|take|view) on\b',
                r'\bwhat do you (?:think|believe|feel)\b',
                r'\b(?:agree|disagree) with\b',
                r'\byour (?:thoughts|opinion|experience)\b'
            ],
            'prediction_seeking': [
                r'\b(?:predict|forecast|expect|think) (?:will|might|could)\b',
                r'\bwhere (?:will|do you think) (?:it|this|price) (?:go|head)\b',
                r'\b(?:bullish|bearish) on\b',
                r'\bprice (?:target|prediction|forecast)\b'
            ]
        }
        
        # Price and percentage patterns
        self.price_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?[kKmMbB]?',  # $1,000, $1.5K, $1M
            r'\d+(?:,\d{3})*\s*(?:usd|usdt|usdc|dollars?)',  # 1000 USD
            r'\d+(?:\.\d+)?\s*(?:btc|eth|sol|ada|dot|avax)',  # 0.1 BTC
            r'[+-]?\d+(?:\.\d+)?%',  # +5%, -10.5%
            r'\d+x\s*(?:gains?|returns?|profit)',  # 10x gains
            r'(?:up|down|gained|lost)\s+\d+(?:\.\d+)?%'  # up 15%
        ]
        
        # Sentiment indicators
        self.sentiment_patterns = {
            'bullish': [
                r'\b(?:bullish|bull|moon|rocket|pump|surge|rally|breakout)\b',
                r'\b(?:hodl|diamond hands|buy the dip|to the moon)\b',
                r'\b(?:excited|optimistic|confident|strong|solid)\b',
                r'\b(?:breakthrough|innovation|adoption|partnership)\b'
            ],
            'bearish': [
                r'\b(?:bearish|bear|crash|dump|correction|pullback)\b',
                r'\b(?:sell|exit|concern|worried|doubt|uncertain)\b',
                r'\b(?:overvalued|bubble|manipulation|scam|rug)\b',
                r'\b(?:paper hands|panic|fear|fud)\b'
            ],
            'neutral': [
                r'\b(?:stable|sideways|consolidation|range)\b',
                r'\b(?:observation|analysis|data|fact)\b',
                r'\b(?:monitoring|watching|tracking)\b'
            ]
        }
        
        # Conversation starters
        self.conversation_patterns = [
            r'\b(?:what do you|anyone else|does anyone|has anyone)\b',
            r'\b(?:unpopular opinion|hot take|controversial)\b',
            r'\b(?:change my mind|convince me|prove me wrong)\b',
            r'\b(?:let\'s discuss|open discussion|thoughts)\b',
            r'\b(?:team|community|together|collective)\b'
        ]
    
    def _initialize_scoring_weights(self):
        """Initialize scoring weights for different factors"""
        self.scoring_weights = {
            'crypto_keywords': 2.0,
            'finance_keywords': 1.5,
            'tech_keywords': 1.0,
            'question_bonus': 3.0,
            'price_mention': 2.0,
            'engagement_metrics': 1.5,
            'recency': 2.0,
            'conversation_starter': 2.5,
            'educational_value': 1.8,
            'author_reputation': 1.2
        }
    
    def analyze_content(self, post_data: Dict[str, Any]) -> ContentAnalysis:
        """Perform comprehensive content analysis"""
        text = post_data.get('text', '').lower()
        
        analysis = ContentAnalysis()
        analysis.word_count = len(text.split())
        
        # Extract keywords and entities
        analysis.crypto_keywords = self._extract_crypto_keywords(text)
        analysis.finance_keywords = self._extract_finance_keywords(text)
        analysis.tech_keywords = self._extract_tech_keywords(text)
        analysis.hashtags = self._extract_hashtags(post_data.get('text', ''))
        analysis.mentions = self._extract_mentions(post_data.get('text', ''))
        
        # Analyze content characteristics
        analysis.has_question, analysis.question_type = self._analyze_questions(text)
        analysis.has_price_mention = self._detect_price_mentions(text)
        analysis.has_percentage = self._detect_percentages(text)
        
        # Categorize content
        analysis.categories = self._categorize_content(text, analysis)
        analysis.primary_category = analysis.categories[0] if analysis.categories else ContentCategory.LOW_VALUE
        
        # Analyze sentiment
        analysis.sentiment, analysis.sentiment_confidence = self._analyze_sentiment(text)
        
        # Calculate opportunity scores
        analysis.engagement_potential = self._calculate_engagement_potential(post_data, analysis)
        analysis.reply_opportunity_score = self._calculate_reply_opportunity(post_data, analysis)
        analysis.conversation_starter_potential = self._calculate_conversation_potential(text, analysis)
        analysis.educational_opportunity = self._calculate_educational_opportunity(text, analysis)
        
        return analysis
    
    def _extract_crypto_keywords(self, text: str) -> List[str]:
        """Extract cryptocurrency-related keywords"""
        found_keywords = []
        
        for category, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_finance_keywords(self, text: str) -> List[str]:
        """Extract finance-related keywords"""
        found_keywords = []
        
        for category, keywords in self.finance_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_tech_keywords(self, text: str) -> List[str]:
        """Extract technology-related keywords"""
        found_keywords = []
        
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        return re.findall(r'#(\w+)', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        return re.findall(r'@(\w+)', text)
    
    def _analyze_questions(self, text: str) -> Tuple[bool, Optional[str]]:
        """Analyze if text contains questions and classify them"""
        has_question = '?' in text
        question_type = None
        
        if has_question:
            for q_type, patterns in self.question_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        question_type = q_type
                        break
                if question_type:
                    break
        
        return has_question, question_type
    
    def _detect_price_mentions(self, text: str) -> bool:
        """Detect price mentions in text"""
        for pattern in self.price_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_percentages(self, text: str) -> bool:
        """Detect percentage mentions in text"""
        return bool(re.search(r'\d+(?:\.\d+)?%', text))
    
    def _categorize_content(self, text: str, analysis: ContentAnalysis) -> List[ContentCategory]:
        """Categorize content based on keywords and characteristics"""
        categories = []
        
        # Crypto discussion
        if len(analysis.crypto_keywords) >= 2 or any(coin in text for coin in ['bitcoin', 'ethereum', 'btc', 'eth']):
            categories.append(ContentCategory.CRYPTO_DISCUSSION)
        
        # Market analysis
        if (analysis.has_price_mention or analysis.has_percentage or 
            any(term in text for term in ['chart', 'analysis', 'technical', 'support', 'resistance'])):
            categories.append(ContentCategory.MARKET_ANALYSIS)
        
        # Price discussion
        if analysis.has_price_mention and any(term in text for term in ['price', 'value', 'worth', 'cost']):
            categories.append(ContentCategory.PRICE_DISCUSSION)
        
        # Question seeking
        if analysis.has_question:
            categories.append(ContentCategory.QUESTION_SEEKING)
        
        # Educational content
        if any(term in text for term in ['learn', 'explain', 'tutorial', 'guide', 'how to', 'what is']):
            categories.append(ContentCategory.EDUCATIONAL)
        
        # Tech news
        if (len(analysis.tech_keywords) >= 2 or 
            any(term in text for term in ['update', 'release', 'launch', 'announcement'])):
            categories.append(ContentCategory.TECH_NEWS)
        
        # Trading signals
        if any(term in text for term in ['buy', 'sell', 'long', 'short', 'entry', 'exit', 'target']):
            categories.append(ContentCategory.TRADING_SIGNALS)
        
        # General tech
        if analysis.tech_keywords and not categories:
            categories.append(ContentCategory.GENERAL_TECH)
        
        # Receptive user (looking for conversation)
        if (analysis.has_question or 
            any(pattern in text for pattern in ['looking for', 'need help', 'new to', 'beginner'])):
            categories.append(ContentCategory.RECEPTIVE_USER)
        
        # Default to low value if no categories
        if not categories:
            categories.append(ContentCategory.LOW_VALUE)
        
        return categories
    
    def _analyze_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """Analyze sentiment with confidence score"""
        sentiment_scores = defaultdict(int)
        
        for sentiment, patterns in self.sentiment_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                sentiment_scores[sentiment] += matches
        
        if not sentiment_scores:
            return SentimentType.NEUTRAL, 0.5
        
        # Find dominant sentiment
        max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        sentiment_name = max_sentiment[0]
        confidence = min(max_sentiment[1] / len(text.split()), 1.0)
        
        sentiment_map = {
            'bullish': SentimentType.BULLISH,
            'bearish': SentimentType.BEARISH,
            'neutral': SentimentType.NEUTRAL
        }
        
        return sentiment_map.get(sentiment_name, SentimentType.NEUTRAL), confidence
    
    def _calculate_engagement_potential(self, post_data: Dict[str, Any], analysis: ContentAnalysis) -> float:
        """Calculate potential for engagement based on content and metrics"""
        score = 0.0
        
        # Base score from existing engagement
        metrics = post_data.get('metrics', {})
        current_engagement = metrics.get('likes', 0) + metrics.get('retweets', 0) + metrics.get('replies', 0)
        score += min(current_engagement / 100.0, 5.0)  # Cap at 5.0
        
        # Content factor bonuses
        if analysis.has_question:
            score += 3.0
        if analysis.crypto_keywords:
            score += len(analysis.crypto_keywords) * 0.5
        if analysis.has_price_mention:
            score += 2.0
        if ContentCategory.QUESTION_SEEKING in analysis.categories:
            score += 2.5
        
        # Recency bonus
        timestamp = post_data.get('timestamp')
        if timestamp:
            hours_old = safe_datetime_diff(datetime.now(), timestamp) / 3600
            if hours_old < 1:
                score += 2.0
            elif hours_old < 6:
                score += 1.0
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _calculate_reply_opportunity(self, post_data: Dict[str, Any], analysis: ContentAnalysis) -> float:
        """Calculate opportunity score for replying to this post"""
        score = 0.0
        
        # Question bonus - questions are great reply opportunities
        if analysis.has_question:
            score += 4.0
            if analysis.question_type in ['seeking_advice', 'information_seeking']:
                score += 2.0
        
        # Low reply count bonus - easier to get noticed
        reply_count = post_data.get('metrics', {}).get('replies', 0)
        if reply_count == 0:
            score += 3.0
        elif reply_count <= 5:
            score += 2.0
        elif reply_count <= 15:
            score += 1.0
        
        # Category bonuses
        if ContentCategory.EDUCATIONAL in analysis.categories:
            score += 2.0
        if ContentCategory.CRYPTO_DISCUSSION in analysis.categories:
            score += 1.5
        if ContentCategory.RECEPTIVE_USER in analysis.categories:
            score += 2.5
        
        # Author verification penalty (harder to get noticed)
        if post_data.get('author_verified', False):
            score *= 0.7
        
        return min(score, 10.0)
    
    def _calculate_conversation_potential(self, text: str, analysis: ContentAnalysis) -> float:
        """Calculate potential for starting conversations"""
        score = 0.0
        
        # Conversation starter patterns
        for pattern in self.conversation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1.5
        
        # Opinion/debate potential
        if any(term in text for term in ['opinion', 'think', 'believe', 'controversial', 'unpopular']):
            score += 2.0
        
        # Question bonus
        if analysis.has_question:
            score += 1.5
        
        # Topic relevance
        if analysis.crypto_keywords or analysis.finance_keywords:
            score += 1.0
        
        return min(score, 10.0)
    
    def _calculate_educational_opportunity(self, text: str, analysis: ContentAnalysis) -> float:
        """Calculate opportunity to provide educational value"""
        score = 0.0
        
        # Learning indicators
        learning_terms = ['learn', 'understand', 'explain', 'how', 'what', 'why', 'beginner', 'new to']
        for term in learning_terms:
            if term in text:
                score += 1.0
        
        # Question types that indicate learning opportunity
        if analysis.question_type in ['information_seeking', 'seeking_advice']:
            score += 3.0
        
        # Complex topics that benefit from explanation
        if analysis.crypto_keywords or analysis.tech_keywords:
            score += 1.5
        
        return min(score, 10.0)


class MarketRelevanceAnalyzer:
    """Analyzes market relevance and scores posts for crypto/finance relevance"""
    
    def __init__(self, content_analyzer: AdvancedContentAnalyzer):
        self.content_analyzer = content_analyzer
        self.relevance_thresholds = {
            'premium': 8.0,
            'high': 6.0,
            'medium': 4.0,
            'low': 2.0
        }
    
    def calculate_market_relevance(self, post_data: Dict[str, Any], 
                                 content_analysis: ContentAnalysis) -> MarketRelevanceScore:
        """Calculate comprehensive market relevance score"""
        score = MarketRelevanceScore()
        
        # Base score from keywords
        score.base_score = self._calculate_base_score(content_analysis)
        
        # Keyword density bonus
        score.keyword_bonus = self._calculate_keyword_bonus(post_data, content_analysis)
        
        # Question bonus - questions are valuable
        score.question_bonus = self._calculate_question_bonus(content_analysis)
        
        # Engagement bonus
        score.engagement_bonus = self._calculate_engagement_bonus(post_data)
        
        # Recency bonus
        score.recency_bonus = self._calculate_recency_bonus(post_data)
        
        # Author bonus
        score.author_bonus = self._calculate_author_bonus(post_data)
        
        # Calculate final score
        score.final_score = (score.base_score + score.keyword_bonus + score.question_bonus + 
                           score.engagement_bonus + score.recency_bonus + score.author_bonus)
        
        # Determine relevance tier
        score.relevance_tier = self._determine_relevance_tier(score.final_score)
        
        return score
    
    def _calculate_base_score(self, analysis: ContentAnalysis) -> float:
        """Calculate base relevance score from content categories"""
        category_scores = {
            ContentCategory.CRYPTO_DISCUSSION: 4.0,
            ContentCategory.MARKET_ANALYSIS: 3.5,
            ContentCategory.PRICE_DISCUSSION: 3.0,
            ContentCategory.TRADING_SIGNALS: 3.5,
            ContentCategory.TECH_NEWS: 2.0,
            ContentCategory.EDUCATIONAL: 2.5,
            ContentCategory.QUESTION_SEEKING: 2.0,
            ContentCategory.RECEPTIVE_USER: 1.5,
            ContentCategory.GENERAL_TECH: 1.0,
            ContentCategory.LOW_VALUE: 0.0
        }
        
        max_score = 0.0
        for category in analysis.categories:
            max_score = max(max_score, category_scores.get(category, 0.0))
        
        return max_score
    
    def _calculate_keyword_bonus(self, post_data: Dict[str, Any], analysis: ContentAnalysis) -> float:
        """Calculate bonus based on keyword density and quality"""
        bonus = 0.0
        
        # Crypto keyword bonus
        crypto_count = len(analysis.crypto_keywords)
        if crypto_count >= 1:
            bonus += min(crypto_count * 0.5, 2.0)
        
        # Finance keyword bonus
        finance_count = len(analysis.finance_keywords)
        if finance_count >= 1:
            bonus += min(finance_count * 0.3, 1.5)
        
        # Price mention bonus
        if analysis.has_price_mention:
            bonus += 1.0
        
        # Percentage mention bonus
        if analysis.has_percentage:
            bonus += 0.5
        
        # Major coin mention bonus
        text = post_data.get('text', '').lower()
        major_coins = ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol']
        for coin in major_coins:
            if coin in text:
                bonus += 0.5
                break
        
        return bonus
    
    def _calculate_question_bonus(self, analysis: ContentAnalysis) -> float:
        """Calculate bonus for question content"""
        if not analysis.has_question:
            return 0.0
        
        question_bonuses = {
            'seeking_advice': 2.0,
            'information_seeking': 1.5,
            'opinion_seeking': 1.0,
            'prediction_seeking': 1.8
        }
        
        # Handle the case where question_type might be None
        if analysis.question_type is None:
            return 1.0
        
        return question_bonuses.get(analysis.question_type, 1.0)
    
    def _calculate_engagement_bonus(self, post_data: Dict[str, Any]) -> float:
        """Calculate bonus based on existing engagement"""
        metrics = post_data.get('metrics', {})
        
        # Low reply count is good for reply opportunities
        reply_count = metrics.get('replies', 0)
        if reply_count == 0:
            bonus = 1.5
        elif reply_count <= 5:
            bonus = 1.0
        elif reply_count <= 15:
            bonus = 0.5
        else:
            bonus = 0.0
        
        # Some engagement shows it's interesting
        likes = metrics.get('likes', 0)
        retweets = metrics.get('retweets', 0)
        
        if likes > 0 or retweets > 0:
            bonus += 0.5
        
        return bonus
    
    def _calculate_recency_bonus(self, post_data: Dict[str, Any]) -> float:
        """Calculate bonus based on post recency"""
        timestamp = post_data.get('timestamp')
        if not timestamp:
            return 0.0
        
        hours_old = safe_datetime_diff(datetime.now(), timestamp) / 3600
        
        if hours_old < 1:
            return 2.0
        elif hours_old < 6:
            return 1.5
        elif hours_old < 24:
            return 1.0
        elif hours_old < 72:
            return 0.5
        else:
            return 0.0
    
    def _calculate_author_bonus(self, post_data: Dict[str, Any]) -> float:
        """Calculate bonus based on author characteristics"""
        bonus = 0.0
        
        # Verified authors get slight penalty (harder to get noticed)
        if post_data.get('author_verified', False):
            bonus -= 0.5
        
        # Popular account posts get slight bonus (more visibility)
        if post_data.get('from_popular_account', False):
            bonus += 0.3
        
        return bonus
    
    def _determine_relevance_tier(self, final_score: float) -> str:
        """Determine relevance tier based on final score"""
        for tier, threshold in self.relevance_thresholds.items():
            if final_score >= threshold:
                return tier
        return 'low'


class ContentPrioritizer:
    """Prioritizes content based on multiple scoring factors"""
    
    def __init__(self, content_analyzer: AdvancedContentAnalyzer, 
                 relevance_analyzer: MarketRelevanceAnalyzer):
        self.content_analyzer = content_analyzer
        self.relevance_analyzer = relevance_analyzer
        
        # Priority weights for different factors
        self.priority_weights = {
            'market_relevance': 0.35,
            'reply_opportunity': 0.25,
            'engagement_potential': 0.20,
            'conversation_potential': 0.10,
            'educational_value': 0.10
        }
    
    def prioritize_posts(self, posts_data: List[Dict[str, Any]], 
                        max_posts: int = 50) -> List[Dict[str, Any]]:
        """Prioritize posts based on comprehensive scoring"""
        logger.logger.info(f"Prioritizing {len(posts_data)} posts with advanced scoring")
        
        scored_posts = []
        
        for post_data in posts_data:
            try:
                # Perform content analysis
                content_analysis = self.content_analyzer.analyze_content(post_data)
                
                # Calculate market relevance
                relevance_score = self.relevance_analyzer.calculate_market_relevance(
                    post_data, content_analysis
                )
                
                # Calculate final priority score
                priority_score = self._calculate_priority_score(
                    post_data, content_analysis, relevance_score
                )
                
                # Add analysis results to post data
                post_data['content_analysis'] = content_analysis
                post_data['market_relevance'] = relevance_score
                post_data['priority_score'] = priority_score
                
                scored_posts.append(post_data)
                
            except Exception as e:
                logger.logger.warning(f"Error analyzing post: {str(e)}")
                # Add with minimal scoring
                post_data['priority_score'] = 0.0
                scored_posts.append(post_data)
        
        # Sort by priority score
        scored_posts.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)
        
        # Log top scoring posts
        self._log_top_posts(scored_posts[:10])
        
        return scored_posts[:max_posts]
    
    def _calculate_priority_score(self, post_data: Dict[str, Any], 
                                content_analysis: ContentAnalysis,
                                relevance_score: MarketRelevanceScore) -> float:
        """Calculate final priority score using weighted factors"""
        
        # Normalize scores to 0-10 range
        market_score = min(relevance_score.final_score, 10.0)
        reply_score = content_analysis.reply_opportunity_score
        engagement_score = content_analysis.engagement_potential
        conversation_score = content_analysis.conversation_starter_potential
        educational_score = content_analysis.educational_opportunity
        
        # Calculate weighted priority score
        priority_score = (
            market_score * self.priority_weights['market_relevance'] +
            reply_score * self.priority_weights['reply_opportunity'] +
            engagement_score * self.priority_weights['engagement_potential'] +
            conversation_score * self.priority_weights['conversation_potential'] +
            educational_score * self.priority_weights['educational_value']
        )
        
        # Apply tier multipliers
        tier_multipliers = {
            'premium': 1.3,
            'high': 1.1,
            'medium': 1.0,
            'low': 0.8
        }
        
        multiplier = tier_multipliers.get(relevance_score.relevance_tier, 1.0)
        priority_score *= multiplier
        
        # Apply category bonuses
        category_bonuses = {
            ContentCategory.CRYPTO_DISCUSSION: 1.2,
            ContentCategory.QUESTION_SEEKING: 1.15,
            ContentCategory.MARKET_ANALYSIS: 1.1,
            ContentCategory.RECEPTIVE_USER: 1.1,
            ContentCategory.EDUCATIONAL: 1.05
        }
        
        primary_category = content_analysis.primary_category
        if primary_category in category_bonuses:
            priority_score *= category_bonuses[primary_category]
        
        return min(priority_score, 10.0)  # Cap at 10.0
    
    def _log_top_posts(self, top_posts: List[Dict[str, Any]]) -> None:
        """Log details of top scoring posts for debugging"""
        logger.logger.info("Top priority posts:")
        
        for i, post in enumerate(top_posts[:5]):
            try:
                score = post.get('priority_score', 0.0)
                relevance = post.get('market_relevance')
                analysis = post.get('content_analysis')
                
                text_preview = post.get('text', '')[:60]
                if len(post.get('text', '')) > 60:
                    text_preview += "..."
                
                author = post.get('author_handle', 'Unknown')
                tier = relevance.relevance_tier if relevance else 'unknown'
                category = analysis.primary_category.value if analysis else 'unknown'
                
                logger.logger.info(
                    f"  {i+1}. Score: {score:.2f} | Tier: {tier} | "
                    f"Category: {category} | {author}: '{text_preview}'"
                )
                
            except Exception as e:
                logger.logger.debug(f"Error logging post {i+1}: {str(e)}")


class SmartContentFilter:
    """Intelligent content filtering with multiple criteria"""
    
    def __init__(self):
        self.filter_rules = {
            'min_word_count': 1,
            'max_word_count': 500,
            'min_relevance_score': 0.3,
            'blocked_categories': [ContentCategory.LOW_VALUE],
            'preferred_categories': [
                ContentCategory.CRYPTO_DISCUSSION,
                ContentCategory.MARKET_ANALYSIS,
                ContentCategory.TECH_NEWS,
                ContentCategory.QUESTION_SEEKING,
                ContentCategory.EDUCATIONAL,
                ContentCategory.PRICE_DISCUSSION,
                ContentCategory.TRADING_SIGNALS,
                ContentCategory.PROJECT_UPDATES,
                ContentCategory.GENERAL_TECH,
                ContentCategory.RECEPTIVE_USER
            ]
        }
    
    def filter_posts(self, posts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply intelligent filtering rules"""
        logger.logger.info(f"Filtering {len(posts_data)} posts with smart criteria")
        
        filtered_posts = []
        filter_stats = defaultdict(int)
        
        for post_data in posts_data:
            try:
                # Apply all filter rules
                if self._passes_basic_filters(post_data, filter_stats):
                    if self._passes_content_filters(post_data, filter_stats):
                        if self._passes_quality_filters(post_data, filter_stats):
                            filtered_posts.append(post_data)
                        
            except Exception as e:
                logger.logger.debug(f"Error filtering post: {str(e)}")
                # Include on error to be safe
                filtered_posts.append(post_data)
        
        # Log filter statistics
        self._log_filter_stats(len(posts_data), len(filtered_posts), filter_stats)
        
        return filtered_posts
    
    def _passes_basic_filters(self, post_data: Dict[str, Any], 
                            filter_stats: Dict[str, int]) -> bool:
        """Apply basic filtering rules"""
        
        # Word count filter
        text = post_data.get('text', '')
        word_count = len(text.split())
        
        if word_count < self.filter_rules['min_word_count']:
            filter_stats['too_short'] += 1
            return False
        
        if word_count > self.filter_rules['max_word_count']:
            filter_stats['too_long'] += 1
            return False
        
        # Empty text filter
        if not text.strip():
            filter_stats['empty_text'] += 1
            return False
        
        return True
    
    def _passes_content_filters(self, post_data: Dict[str, Any], 
                              filter_stats: Dict[str, int]) -> bool:
        """Apply content-based filtering rules"""
        
        analysis = post_data.get('content_analysis')
        if not analysis:
            return True  # Pass if no analysis available
        
        # Category filter
        if analysis.primary_category in self.filter_rules['blocked_categories']:
            filter_stats['blocked_category'] += 1
            return False
        
        # Market relevance filter
        relevance = post_data.get('market_relevance')
        if relevance and relevance.final_score < self.filter_rules['min_relevance_score']:
            filter_stats['low_relevance'] += 1
            return False
        
        return True
    
    def _passes_quality_filters(self, post_data: Dict[str, Any], 
                              filter_stats: Dict[str, int]) -> bool:
        """Apply quality-based filtering rules"""
        
        # Filter out posts that are clearly spam or low quality
        text = post_data.get('text', '').lower()
        
        # Spam indicators
        spam_patterns = [
            r'\b(?:follow|dm|link in bio)\b',
            r'\b(?:click here|visit|check out)\b.*http',
            r'\b(?:free|giveaway|contest)\b.*(?:follow|retweet)',
            r'(\w)\1{3,}',  # Repeated characters
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text):
                filter_stats['spam_detected'] += 1
                return False
        
        # Excessive emoji filter
        emoji_count = len(re.findall(r'[😀-🿿]', text))
        if emoji_count > len(text.split()) / 2:  # More emojis than words
            filter_stats['excessive_emojis'] += 1
            return False
        
        return True
    
    def _log_filter_stats(self, original_count: int, filtered_count: int, 
                         filter_stats: Dict[str, int]) -> None:
        """Log filtering statistics"""
        removed_count = original_count - filtered_count
        
        logger.logger.info(
            f"Filtering complete: {filtered_count}/{original_count} posts passed "
            f"({removed_count} removed)"
        )
        
        if filter_stats:
            logger.logger.debug("Filter breakdown:")
            for reason, count in filter_stats.items():
                logger.logger.debug(f"  {reason}: {count}")


class OpportunityDetector:
    """Detects specific opportunities for engagement and education"""
    
    def __init__(self):
        # Opportunity patterns for different types of engagement (200+ patterns)
        self.opportunity_patterns = {
            'beginner_questions': [
                # Basic beginner indicators
                r'\b(?:new to|just started|beginner|first time|noob|newbie)\b',
                r'\b(?:how do i|where do i|what is|explain|help me)\b.*(?:crypto|bitcoin|blockchain|defi|nft)',
                r'\b(?:can someone explain|help me understand|eli5|explain like)\b',
                r'\b(?:complete beginner|total newbie|just getting started)\b',
                r'\b(?:never used|never tried|first attempt|starting out)\b',
                r'\b(?:learning|trying to learn|want to understand)\b.*(?:crypto|blockchain|defi)',
                r'\b(?:confused about|don\'t understand|makes no sense)\b',
                r'\b(?:basic question|simple question|dumb question|stupid question)\b',
                r'\b(?:where to start|how to begin|getting into crypto)\b',
                r'\b(?:what does.*mean|definition of|meaning of)\b',
                
                # Specific beginner topics
                r'\b(?:what is|how does).*(?:bitcoin|ethereum|blockchain|mining|staking|defi|yield farming)\b',
                r'\b(?:difference between|vs|compared to).*(?:bitcoin|ethereum|crypto|blockchain)\b',
                r'\b(?:how to buy|where to buy|best place to buy)\b.*(?:crypto|bitcoin|ethereum)',
                r'\b(?:which wallet|best wallet|wallet recommendation)\b',
                r'\b(?:safe to|is it safe|security|secure)\b.*(?:crypto|bitcoin|exchange)',
                r'\b(?:getting started|first steps|beginner guide)\b.*(?:crypto|blockchain|defi)',
                r'\b(?:recommended|suggest|advice for).*(?:beginners|newbies|new users)\b',
            ],
            
            'technical_help': [
                # Technical problems
                r'\b(?:error|problem|issue|bug|trouble|broken|failing)\b',
                r'\b(?:not working|failed|stuck|confused|can\'t get)\b',
                r'\b(?:help|assistance|support|guidance|solution)\b',
                r'\b(?:wallet|transaction|mining|staking|node|sync).*(?:help|issue|problem|error)\b',
                r'\b(?:gas fees|high fees|fee calculation|transaction stuck)\b',
                r'\b(?:metamask|trust wallet|ledger|hardware wallet).*(?:help|issue|problem)\b',
                r'\b(?:smart contract|solidity|web3|dapp).*(?:help|error|debug)\b',
                r'\b(?:connection|network|rpc|endpoint).*(?:issue|problem|error)\b',
                
                # Technical setup questions
                r'\b(?:how to setup|configure|install|connect)\b.*(?:wallet|node|mining|staking)',
                r'\b(?:troubleshooting|debug|fix|resolve)\b',
                r'\b(?:won\'t connect|connection failed|network error)\b',
                r'\b(?:transaction pending|stuck transaction|failed tx)\b',
                r'\b(?:lost|missing|can\'t find).*(?:coins|tokens|funds|wallet)\b',
                r'\b(?:recover|restore|backup).*(?:wallet|seed|private key)\b',
                r'\b(?:upgrade|update|migration).*(?:help|issue|problem)\b',
                r'\b(?:api|integration|development).*(?:help|issue|question)\b',
                r'\b(?:slippage|mev|sandwich attack).*(?:help|avoid|prevent)\b',
                r'\b(?:bridge|cross chain|multichain).*(?:help|issue|stuck)\b',
            ],
            
            'investment_advice': [
                # Investment decisions
                r'\b(?:should i|worth|good investment|buy|sell|hold)\b',
                r'\b(?:portfolio|allocation|strategy|advice|recommendation)\b',
                r'\b(?:hodl|hold|exit|take profit|stop loss)\b',
                r'\b(?:when to buy|when to sell|timing|entry point)\b',
                r'\b(?:dca|dollar cost average|averaging down)\b',
                r'\b(?:diversify|diversification|spread risk)\b',
                r'\b(?:long term|short term|swing trading|day trading)\b',
                r'\b(?:risk management|position sizing|allocation)\b',
                
                # Specific investment questions
                r'\b(?:altcoin|shitcoin|memecoin|gem).*(?:advice|recommendation|thoughts)\b',
                r'\b(?:all in|yolo|moon|lambo).*(?:worth it|good idea|advice)\b',
                r'\b(?:cut losses|take profits|rebalance)\b',
                r'\b(?:staking rewards|yield|apy|apr).*(?:worth|good|advice)\b',
                r'\b(?:defi|yield farming|liquidity mining).*(?:safe|worth|advice)\b',
                r'\b(?:nft|pfp|collection).*(?:investment|worth|buy)\b',
                r'\b(?:bear market|bull market|dip|correction).*(?:strategy|advice|what to do)\b',
                r'\b(?:market crash|recession|inflation).*(?:crypto|bitcoin|strategy)\b',
                r'\b(?:tax|taxes|capital gains).*(?:crypto|advice|help)\b',
            ],
            
            'price_prediction': [
                # Price predictions and forecasts
                r'\b(?:price prediction|target|forecast|projection)\b',
                r'\b(?:bullish|bearish|moon|crash|dump|pump)\b',
                r'\b(?:where (?:will|do you think)).*(?:go|price|head|end up)\b',
                r'\b(?:next target|resistance|support|levels)\b',
                r'\b(?:ath|all time high|new highs|break out)\b',
                r'\b(?:bottom|floor|consolidation|sideways)\b',
                r'\b(?:technical analysis|ta|charts|indicators)\b.*(?:prediction|forecast)',
                r'\b(?:fibonacci|elliott wave|wyckoff).*(?:analysis|prediction)\b',
                
                # Specific price discussions
                r'\b(?:100k|1m|million|trillion).*(?:bitcoin|btc|possible|realistic)\b',
                r'\b(?:flip|flippening).*(?:ethereum|bitcoin|market cap)\b',
                r'\b(?:cycle|halving|bull run|bear market).*(?:prediction|when|timeline)\b',
                r'\b(?:eoy|end of year|2024|2025|2030).*(?:price|target|prediction)\b',
                r'\b(?:realistic|conservative|optimistic).*(?:target|price|prediction)\b',
                r'\b(?:market cap|valuation|fair value)\b.*(?:prediction|estimate)\b',
                r'\b(?:adoption|mass adoption).*(?:price|impact|when)\b',
                r'\b(?:regulation|etf|institutional).*(?:price|impact|bullish)\b',
            ],
            
            'news_discussion': [
                # News and current events
                r'\b(?:thoughts on|opinion|what do you think).*(?:news|announcement|update)\b',
                r'\b(?:breaking|just announced|update|release|launched)\b',
                r'\b(?:partnership|adoption|regulation|sec|cftc)\b',
                r'\b(?:hack|exploit|rug pull|scam).*(?:thoughts|opinion|impact)\b',
                r'\b(?:upgrade|hard fork|soft fork|update).*(?:thoughts|impact)\b',
                r'\b(?:earnings|revenue|profit|loss).*(?:crypto|bitcoin|company)\b',
                r'\b(?:lawsuit|legal|court|judge).*(?:crypto|bitcoin|impact)\b',
                r'\b(?:ban|banned|restriction|illegal).*(?:crypto|bitcoin|country)\b',
                
                # Specific news topics
                r'\b(?:fed|federal reserve|interest rates|inflation).*(?:crypto|bitcoin|impact)\b',
                r'\b(?:etf|spot etf|approval|rejection).*(?:thoughts|impact|news)\b',
                r'\b(?:tesla|microstrategy|saylor|musk).*(?:bitcoin|crypto|news)\b',
                r'\b(?:binance|coinbase|exchange).*(?:news|announcement|problem)\b',
                r'\b(?:tether|usdt|usdc|stablecoin).*(?:news|concern|depeg)\b',
                r'\b(?:merge|ethereum 2.0|pos|pow).*(?:thoughts|impact|successful)\b',
                r'\b(?:cbdc|digital dollar|digital yuan).*(?:thoughts|threat|impact)\b',
                r'\b(?:web3|metaverse|gaming|nft).*(?:news|adoption|thoughts)\b',
            ],
            
            'market_sentiment': [
                # Sentiment and emotions
                r'\b(?:bullish|bearish|optimistic|pessimistic|hopeful|worried)\b',
                r'\b(?:fud|fear|doubt|uncertainty|panic|euphoria)\b',
                r'\b(?:fomo|fear of missing out|buying the dip)\b',
                r'\b(?:diamond hands|paper hands|hodl|weak hands)\b',
                r'\b(?:moon|lambo|rekt|ngmi|wagmi|lfg)\b',
                r'\b(?:cope|copium|hopium|depression|anxiety)\b',
                r'\b(?:confident|uncertain|scared|excited|nervous)\b',
                r'\b(?:market sentiment|crowd psychology|herd mentality)\b',
                r'\b(?:capitulation|despair|greed|fear index)\b',
                r'\b(?:retail|institutions|whales|manipulation)\b',
            ],
            
            'trading_discussion': [
                # Trading strategies and techniques
                r'\b(?:trading|trader|trade setup|entry|exit)\b',
                r'\b(?:scalping|swing trading|position trading|day trading)\b',
                r'\b(?:leverage|margin|futures|options|derivatives)\b',
                r'\b(?:stop loss|take profit|risk reward|r:r)\b',
                r'\b(?:breakout|breakdown|reversal|continuation)\b',
                r'\b(?:volume|orderbook|whale alert|large transaction)\b',
                r'\b(?:arbitrage|spread|premium|discount)\b',
                r'\b(?:liquidation|squeeze|gamma|delta)\b',
                r'\b(?:bot|algorithm|automated trading|strategy)\b',
                r'\b(?:backtest|forward test|paper trading|demo)\b',
                
                # Technical analysis
                r'\b(?:rsi|macd|bollinger|moving average|ema|sma)\b',
                r'\b(?:fibonacci|golden ratio|0.618|retracement)\b',
                r'\b(?:support|resistance|trendline|channel)\b',
                r'\b(?:triangle|wedge|flag|pennant|head and shoulders)\b',
                r'\b(?:divergence|convergence|momentum|oscillator)\b',
            ],
            
            'community_engagement': [
                # Community and social aspects
                r'\b(?:community|fam|family|apes|diamond hands|hodlers)\b',
                r'\b(?:together|unity|strength|support|helping)\b',
                r'\b(?:newcomer|welcome|joining|glad to be here)\b',
                r'\b(?:thank you|thanks|appreciate|grateful|helpful)\b',
                r'\b(?:discussion|debate|conversation|chat|talk)\b',
                r'\b(?:share|sharing|experience|story|journey)\b',
                r'\b(?:meetup|conference|event|gathering|irl)\b',
                r'\b(?:twitter|reddit|discord|telegram|social)\b',
                r'\b(?:influencer|kol|thought leader|expert|guru)\b',
                r'\b(?:shill|promote|advertise|marketing|awareness)\b',
            ],
            
            'educational_content': [
                # Educational and informational
                r'\b(?:learn|learning|education|knowledge|understanding)\b',
                r'\b(?:tutorial|guide|walkthrough|step by step|how to)\b',
                r'\b(?:course|book|podcast|video|resource)\b',
                r'\b(?:whitepaper|documentation|research|study)\b',
                r'\b(?:fundamental|technical|analysis|due diligence)\b',
                r'\b(?:blockchain|cryptocurrency|decentralized|distributed)\b',
                r'\b(?:consensus|mining|validation|node|network)\b',
                r'\b(?:smart contract|dapp|defi|dao|nft)\b',
                r'\b(?:tokenomics|economics|game theory|incentives)\b',
                r'\b(?:security|privacy|anonymity|pseudonymous)\b',
            ],
            
            'general_questions': [
                # General question patterns
                r'\b(?:what|how|why|when|where|who|which)\b',
                r'\b(?:question|ask|asking|wondering|curious)\b',
                r'\b(?:anyone|someone|everybody|community)\b.*(?:know|help|explain)',
                r'\b(?:thoughts|opinions|views|perspective|take)\b',
                r'\b(?:experience|tried|using|used)\b.*(?:thoughts|review)',
                r'\b(?:recommend|suggestion|advice|guidance)\b',
                r'\b(?:best|worst|top|bottom|favorite|preferred)\b',
                r'\b(?:comparison|compare|vs|versus|difference)\b',
                r'\b(?:worth it|good idea|bad idea|smart|stupid)\b',
                r'\b(?:alternatives|options|choices|possibilities)\b',
                
                # Question indicators
                r'\?',  # Any question mark
                r'\b(?:is it|are they|can i|should i|will it|would it)\b',
                r'\b(?:does anyone|has anyone|is anyone|can anyone)\b',
                r'\b(?:looking for|searching for|need|want|seeking)\b',
                r'\b(?:help me|assist me|guide me|show me)\b',
            ],
            
            'project_specific': [
                # Major cryptocurrency projects
                r'\b(?:bitcoin|btc|ethereum|eth|solana|sol|cardano|ada)\b',
                r'\b(?:binance|bnb|polygon|matic|avalanche|avax|polkadot|dot)\b',
                r'\b(?:chainlink|link|uniswap|uni|aave|compound|maker|mkr)\b',
                r'\b(?:cosmos|atom|terra|luna|algorand|algo|tezos|xtz)\b',
                r'\b(?:near|protocol|fantom|ftm|harmony|one|elrond|egld)\b',
                
                # DeFi protocols
                r'\b(?:defi|decentralized finance|yield|liquidity|farming)\b',
                r'\b(?:pancakeswap|sushiswap|curve|balancer|1inch)\b',
                r'\b(?:lending|borrowing|collateral|liquidation)\b',
                r'\b(?:amm|automated market maker|swap|exchange)\b',
                
                # NFT and gaming
                r'\b(?:nft|non fungible|opensea|pfp|collection|mint)\b',
                r'\b(?:gaming|play to earn|p2e|metaverse|virtual)\b',
                r'\b(?:axie|sandbox|decentraland|enjin|flow)\b',
            ],
            
            'meme_and_culture': [
                # Crypto memes and culture
                r'\b(?:moon|lambo|rekt|ngmi|wagmi|lfg|gm|gn)\b',
                r'\b(?:ape|aping|degen|chad|virgin|wojak|pepe)\b',
                r'\b(?:diamond hands|paper hands|hodl|buy the dip)\b',
                r'\b(?:pump|dump|pamp|pampit|bogdanoff|sminem)\b',
                r'\b(?:cope|seethe|dilate|based|cringe|kek)\b',
                r'\b(?:fren|anon|ser|fud|number go up|ngu)\b',
                r'\b(?:line go up|green candle|red candle|crab market)\b',
                r'\b(?:this is the way|wen moon|wen lambo|soon)\b',
                r'\b(?:bullish|bearish|crabish|sideways|chop)\b',
                r'\b(?:hopium|copium|rope|sui|make it|not gonna make it)\b',
            ],
            
            'regulatory_and_legal': [
                # Regulatory discussions
                r'\b(?:regulation|regulatory|sec|cftc|irs|government)\b',
                r'\b(?:legal|illegal|law|legislation|bill|act)\b',
                r'\b(?:compliance|kyc|aml|sanctions|blacklist)\b',
                r'\b(?:tax|taxes|taxation|capital gains|income)\b',
                r'\b(?:banned|ban|restriction|prohibited|allowed)\b',
                r'\b(?:etf|spot etf|futures etf|approval|rejection)\b',
                r'\b(?:custody|custodian|institutional|retail)\b',
                r'\b(?:cbdc|digital currency|federal reserve|central bank)\b',
                r'\b(?:lawsuit|court|judge|settlement|fine)\b',
                r'\b(?:policy|framework|guidance|clarity|uncertainty)\b',
            ]
        }
    
    def detect_opportunities(self, posts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and categorize engagement opportunities"""
        
        opportunities = []
        
        for post_data in posts_data:
            try:
                text = post_data.get('text', '').lower()
                analysis = post_data.get('content_analysis')
                
                # Detect opportunity types
                opportunity_types = []
                for opp_type, patterns in self.opportunity_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            opportunity_types.append(opp_type)
                            break
                
                if opportunity_types:
                    # Calculate opportunity score - only if we have valid analysis
                    if analysis is not None:
                        opp_score = self._calculate_opportunity_score(
                            post_data, analysis, opportunity_types
                        )
                    else:
                        # Fallback score calculation without analysis
                        opp_score = len(opportunity_types) * 2.0  # Basic scoring
                    
                    # Add opportunity metadata
                    post_data['opportunity_types'] = opportunity_types
                    post_data['opportunity_score'] = opp_score
                    
                    opportunities.append(post_data)
                    
            except Exception as e:
                logger.logger.debug(f"Error detecting opportunities: {str(e)}")
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x.get('opportunity_score', 0.0), reverse=True)
        
        logger.logger.info(f"Detected {len(opportunities)} engagement opportunities")
        
        return opportunities
    
    def _calculate_opportunity_score(self, post_data: Dict[str, Any], 
                                   analysis: ContentAnalysis,
                                   opportunity_types: List[str]) -> float:
        """Calculate opportunity score based on multiple factors"""
        
        base_score = len(opportunity_types) * 2.0  # Base score from opportunity count
        
        # Type-specific bonuses
        type_bonuses = {
            'beginner_questions': 3.0,  # High value - educational opportunity
            'technical_help': 2.5,     # Good value - show expertise
            'investment_advice': 2.0,   # Medium value - be careful with advice
            'price_prediction': 1.5,   # Lower value - speculative
            'news_discussion': 2.0      # Good value - thought leadership
        }
        
        type_bonus = sum(type_bonuses.get(opp_type, 1.0) for opp_type in opportunity_types)
        
        # Question bonus
        question_bonus = 2.0 if analysis and analysis.has_question else 0.0
        
        # Engagement bonus (low replies = good opportunity)
        reply_count = post_data.get('metrics', {}).get('replies', 0)
        engagement_bonus = max(3.0 - (reply_count * 0.3), 0.0)
        
        # Recency bonus
        timestamp = post_data.get('timestamp')
        recency_bonus = 0.0
        if timestamp:
            hours_old = safe_datetime_diff(datetime.now(), timestamp) / 3600
            if hours_old < 2:
                recency_bonus = 2.0
            elif hours_old < 12:
                recency_bonus = 1.0
        
        total_score = base_score + type_bonus + question_bonus + engagement_bonus + recency_bonus
        
        return min(total_score, 15.0)  # Cap at 15.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Timeline Scraper - Part 4: Main Scraper Orchestration
Complete scraping workflow with all components integrated
"""

import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)

from datetime_utils import ensure_naive_datetimes, strip_timezone, safe_datetime_diff
from utils.logger import logger

# Import all components from previous parts
# from .scraper_part1 import (ScrapingConfig, AdaptiveSelectorManager, AntiDetectionManager,
#                             BrowserInteractionManager, ExtractionCache, ErrorRecoveryManager)
# from .scraper_part2 import (HoneypotDetector, AdvancedPostDetector, PostDataExtractor)
# from .scraper_part3 import (AdvancedContentAnalyzer, MarketRelevanceAnalyzer,
#                             ContentPrioritizer, SmartContentFilter, OpportunityDetector)


@dataclass
class ScrapingSession:
    """Track scraping session state and statistics"""
    session_id: str
    start_time: datetime
    posts_found: int = 0
    posts_processed: int = 0
    posts_filtered: int = 0
    honeypots_detected: int = 0
    errors_encountered: int = 0
    scroll_attempts: int = 0
    navigation_attempts: int = 0
    cache_hits: int = 0
    performance_metrics: Dict[str, float] | None = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class ScrapingResult:
    """Complete scraping operation results"""
    session: ScrapingSession
    posts: List[Dict[str, Any]]
    high_priority_posts: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    filter_statistics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class EnhancedTimelineScraper:
    """
    Main timeline scraper orchestrating all components for award-winning performance
    """
    
    def __init__(self, browser, config=None, db=None):
        """Initialize the enhanced timeline scraper"""
        self.browser = browser
        self.config = config or self._get_default_config()
        self.db = db
        
        # Initialize all core components
        self._initialize_components()
        
        # State tracking
        self.current_session: Optional[ScrapingSession] = None
        self.navigation_urls = [
            "https://twitter.com/home",
            "https://x.com/home"
        ]
        
        logger.logger.info("Enhanced Timeline Scraper initialized with all components")
    
    def _get_default_config(self):
        """Get default scraping configuration"""
        return ScrapingConfig(
            mode=ScrapingMode.BALANCED,
            max_posts_target=50,
            skip_honeypot_detection=True,
            enable_debug_screenshots=False
        )

    def _initialize_components(self):
        """Initialize all scraping components"""
        try:
            self.selector_manager = AdaptiveSelectorManager()
            self.anti_detection = AntiDetectionManager(self.config)
            self.browser_manager = BrowserInteractionManager(
                self.browser, self.config, self.anti_detection
            )
            self.extraction_cache = ExtractionCache() if self.config.cache_extracted_data else None
            self.error_recovery = ErrorRecoveryManager(self.config)
            
            self.honeypot_detector = HoneypotDetector()
            self.post_detector = AdvancedPostDetector(
                self.browser_manager, self.selector_manager, self.config
            )
            self.data_extractor = PostDataExtractor(
                self.browser_manager, self.selector_manager, self.config, self.extraction_cache
            )
            
            self.content_analyzer = AdvancedContentAnalyzer()
            self.relevance_analyzer = MarketRelevanceAnalyzer(self.content_analyzer)
            self.content_prioritizer = ContentPrioritizer(
                self.content_analyzer, self.relevance_analyzer
            )
            self.content_filter = SmartContentFilter()
            self.opportunity_detector = OpportunityDetector()
            
            logger.logger.info("✅ All scraping components initialized successfully")
            
        except Exception as e:
            logger.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def scrape_timeline(self, count: int = 10) -> ScrapingResult:
        """
        Main scraping method with complete workflow orchestration
        
        Args:
            count: Target number of high-quality posts to return
            
        Returns:
            ScrapingResult with complete operation details
        """
        session_id = f"session_{int(time.time())}"
        session = ScrapingSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.current_session = session
        
        logger.logger.info(f"🚀 Starting enhanced timeline scraping session: {session_id}")
        logger.logger.info(f"Target: {count} high-quality posts")
        
        try:
            # Phase 1: Navigation and Setup
            if not self._navigate_to_timeline(session):
                return self._create_failure_result(session, "Navigation failed")
            
            # Phase 2: Post Detection and Collection
            raw_posts = self._collect_posts(session, count * 3)  # Collect 3x for filtering
            if not raw_posts:
                return self._create_failure_result(session, "No posts found")
            
            # Phase 3: Data Extraction
            extracted_posts = self._extract_post_data(session, raw_posts)
            if not extracted_posts:
                return self._create_failure_result(session, "Data extraction failed")
            
            # Phase 4: Content Analysis and Filtering
            analyzed_posts = self._analyze_and_filter_content(session, extracted_posts)
            
            # Phase 5: Prioritization and Final Selection
            prioritized_posts = self._prioritize_posts(session, analyzed_posts, count)
            
            # Phase 6: Opportunity Detection
            opportunities = self._detect_opportunities(session, prioritized_posts)
            
            # Create success result
            result = self._create_success_result(
                session, prioritized_posts, opportunities
            )
            
            logger.logger.info(f"✅ Scraping session {session_id} completed successfully")
            logger.logger.info(f"📊 Results: {len(prioritized_posts)} posts, {len(opportunities)} opportunities")
            
            return result
            
        except Exception as e:
            logger.logger.error(f"❌ Scraping session {session_id} failed: {str(e)}")
            return self._create_failure_result(session, str(e))
        
        finally:
            self._cleanup_session(session)
    
    def _navigate_to_timeline(self, session: ScrapingSession) -> bool:
        """Navigate to Twitter/X timeline with enhanced reliability"""
        logger.logger.info("Navigating to timeline...")
        
        # Check browser health before attempting navigation
        if not self.browser.is_browser_healthy():
            logger.logger.error("Browser health check failed, cannot navigate to timeline")
            return False
        
        max_attempts = 3
        for attempt in range(max_attempts):
            session.navigation_attempts += 1
            
            try:
                url = random.choice(self.navigation_urls)  # Randomize URL
                logger.logger.debug(f"Navigation attempt {attempt + 1}: {url}")
                
                # Navigate to URL
                self.browser.driver.get(url)
                
                # Add human-like delay
                self.anti_detection.add_human_delay("page_load")
                
                # Verify we're on the correct page
                if self._verify_timeline_loaded():
                    logger.logger.info("Successfully navigated to timeline")
                    return True
                
                # If verification failed, try recovery
                if attempt < max_attempts - 1:
                    logger.logger.warning(f"Timeline verification failed, attempt {attempt + 1}")
                    self._attempt_navigation_recovery()
                
            except Exception as e:
                session.errors_encountered += 1
                error_type = type(e).__name__
                
                if self.error_recovery.handle_error(error_type, e, "navigation"):
                    continue
                else:
                    logger.logger.error(f"Navigation recovery failed: {str(e)}")
                    break
        
        logger.logger.error("Failed to navigate to timeline after all attempts")
        return False
    
    def _verify_timeline_loaded(self) -> bool:
        """Verify that the timeline has loaded properly"""
        try:
            # Use multiple verification strategies
            verification_methods = [
                self._check_timeline_posts,
                self._check_ui_elements,
                self._check_page_title
            ]
            
            for method in verification_methods:
                try:
                    if method():
                        return True
                except Exception as e:
                    logger.logger.debug(f"Verification method failed: {str(e)}")
                    continue
            
            return False
            
        except Exception as e:
            logger.logger.warning(f"Timeline verification error: {str(e)}")
            return False
    
    def _check_timeline_posts(self) -> bool:
        """Check if timeline posts are visible"""
        try:
            post_selectors = ['div[data-testid="cellInnerDiv"]', 'article[data-testid="tweet"]']
            
            for selector in post_selectors:
                elements = self.browser_manager.safe_find_elements(selector, timeout=10)
                if len(elements) >= 3:  # At least 3 posts visible
                    return True
            
            return False
        except:
            return False
    
    def _check_ui_elements(self) -> bool:
        """Check for Twitter/X UI elements"""
        try:
            ui_selectors = [
                'div[data-testid="primaryColumn"]',
                'nav[role="navigation"]',
                'header[role="banner"]'
            ]
            
            for selector in ui_selectors:
                elements = self.browser_manager.safe_find_elements(selector, timeout=5)
                if elements:
                    return True
            
            return False
        except:
            return False
    
    def _check_page_title(self) -> bool:
        """Check page title for Twitter/X indicators"""
        try:
            title = self.browser.driver.title.lower()
            return any(indicator in title for indicator in ['twitter', 'x.com', 'home'])
        except:
            return False
    
    def _attempt_navigation_recovery(self) -> None:
        """Attempt to recover from navigation issues"""
        try:
            logger.logger.info("Attempting navigation recovery...")
            
            # Strategy 1: Refresh page
            self.browser.driver.refresh()
            time.sleep(3)
            
            # Strategy 2: Clear cookies if refresh doesn't work
            if not self._verify_timeline_loaded():
                try:
                    self.browser.driver.delete_all_cookies()
                    time.sleep(2)
                except:
                    pass
            
        except Exception as e:
            logger.logger.debug(f"Navigation recovery error: {str(e)}")
    
    def _collect_posts(self, session: ScrapingSession, target_count: int) -> List[Any]:
        """Collect raw post elements using advanced detection"""
        logger.logger.info(f"🔍 Collecting posts (target: {target_count})...")
        
        all_posts = []
        scroll_attempts = 0
        max_scrolls = self.config.max_scroll_attempts
        consecutive_failures = 0
        
        while (len(all_posts) < target_count and 
               scroll_attempts < max_scrolls and 
               consecutive_failures < 3):
            
            try:
                # Detect posts on current view
                current_posts = self.post_detector.find_all_posts()
                
                if current_posts:
                    # Add new posts (detector handles deduplication)
                    new_posts = [p for p in current_posts if p not in all_posts]
                    all_posts.extend(new_posts)
                    session.posts_found += len(new_posts)
                    
                    logger.logger.debug(f"Found {len(new_posts)} new posts (total: {len(all_posts)})")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.logger.debug(f"No posts found in current view (failures: {consecutive_failures})")
                
                # Check if we have enough posts
                if len(all_posts) >= target_count:
                    logger.logger.info(f"✅ Target reached: {len(all_posts)} posts collected")
                    break
                
                # Scroll down for more content
                scroll_success = self._smart_scroll(session)
                scroll_attempts += 1
                session.scroll_attempts += 1
                
                if not scroll_success:
                    consecutive_failures += 1
                    logger.logger.warning(f"Scroll failed (attempt {scroll_attempts})")
                
                # Take break if needed for anti-detection
                if self.anti_detection.should_take_break():
                    self.anti_detection.take_human_break()
                
            except Exception as e:
                session.errors_encountered += 1
                consecutive_failures += 1
                
                error_type = type(e).__name__
                if not self.error_recovery.handle_error(error_type, e, "post_collection"):
                    logger.logger.error(f"Post collection error: {str(e)}")
                    break
        
        logger.logger.info(f"📊 Post collection complete: {len(all_posts)} posts found")
        return all_posts
    
    def _smart_scroll(self, session: ScrapingSession) -> bool:
        """Perform intelligent scrolling with anti-detection"""
        try:
            # Check current scroll position
            current_height = self.browser_manager.safe_execute_script(
                "return document.documentElement.scrollHeight"
            )
            
            # Perform scroll with human-like behavior
            scroll_success = self.browser_manager.safe_scroll()
            
            if scroll_success:
                # Verify scroll actually happened
                new_height = self.browser_manager.safe_execute_script(
                    "return document.documentElement.scrollHeight"
                )
                
                # Add random pause for content loading
                pause_time = random.uniform(1.5, 3.0)
                time.sleep(pause_time)
                
                return new_height != current_height or scroll_success
            
            return False
            
        except Exception as e:
            logger.logger.debug(f"Smart scroll error: {str(e)}")
            return False
    
    def _extract_post_data(self, session: ScrapingSession, post_elements: List[Any]) -> List[Dict[str, Any]]:
        """Extract data from post elements"""
        logger.logger.info(f"📤 Extracting data from {len(post_elements)} posts...")
        
        extracted_posts = []
        extraction_errors = 0
        
        for i, post_element in enumerate(post_elements):
            try:
                # Check cache first
                if self.extraction_cache:
                    cached_data = self.extraction_cache.get(
                        post_element.get_attribute('outerHTML') or f"post_{i}"
                    )
                    if cached_data:
                        session.cache_hits += 1
                        extracted_posts.append(cached_data)
                        continue
                
                # Extract post data
                post_data = self.data_extractor.extract_post_data(post_element)
                
                if post_data:
                    extracted_posts.append(post_data)
                    session.posts_processed += 1
                else:
                    extraction_errors += 1
                
                # Add small delay for anti-detection
                if random.random() < 0.1:  # 10% chance
                    self.anti_detection.add_human_delay("extraction")
                
            except Exception as e:
                extraction_errors += 1
                session.errors_encountered += 1
                logger.logger.debug(f"Post extraction error: {str(e)}")
        
        logger.logger.info(f"✅ Extracted {len(extracted_posts)} posts ({extraction_errors} errors)")
        return extracted_posts
    
    def _analyze_and_filter_content(self, session: ScrapingSession, 
                                  posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze content and apply intelligent filtering"""
        logger.logger.info(f"🧠 Analyzing and filtering {len(posts)} posts...")
        
        # Apply smart content filtering first
        filtered_posts = self.content_filter.filter_posts(posts)
        session.posts_filtered = len(posts) - len(filtered_posts)
        
        logger.logger.info(f"📊 Content analysis complete: {len(filtered_posts)} posts passed filtering")
        return filtered_posts
    
    def _prioritize_posts(self, session: ScrapingSession, posts: List[Dict[str, Any]], 
                         target_count: int) -> List[Dict[str, Any]]:
        """Prioritize posts using advanced scoring"""
        logger.logger.info(f"🎯 Prioritizing {len(posts)} posts...")
        
        # Use content prioritizer for advanced scoring
        prioritized_posts = self.content_prioritizer.prioritize_posts(posts, target_count)
        
        logger.logger.info(f"✅ Prioritization complete: {len(prioritized_posts)} top posts selected")
        return prioritized_posts
    
    def _detect_opportunities(self, session: ScrapingSession, 
                            posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect engagement opportunities"""
        logger.logger.info(f"🎪 Detecting opportunities in {len(posts)} posts...")
        
        opportunities = self.opportunity_detector.detect_opportunities(posts)
        
        logger.logger.info(f"✅ Found {len(opportunities)} engagement opportunities")
        return opportunities
    
    def _create_success_result(self, session: ScrapingSession, posts: List[Dict[str, Any]], 
                             opportunities: List[Dict[str, Any]]) -> ScrapingResult:
        """Create successful scraping result"""
        
        # Separate high priority posts (top 60% by score)
        high_priority_posts = []
        if posts:
            sorted_posts = sorted(posts, key=lambda x: x.get('priority_score', 0.0), reverse=True)
            cutoff_index = max(1, int(len(sorted_posts) * 0.6))
            high_priority_posts = sorted_posts[:cutoff_index]
        
        # Calculate performance metrics
        duration = safe_datetime_diff(datetime.now(), session.start_time)
        performance_summary = {
            'session_duration_seconds': duration,
            'posts_per_minute': (session.posts_found / (duration / 60)) if duration > 0 else 0,
            'extraction_success_rate': (session.posts_processed / session.posts_found) if session.posts_found > 0 else 0,
            'cache_hit_rate': (session.cache_hits / session.posts_processed) if session.posts_processed > 0 else 0,
            'error_rate': (session.errors_encountered / session.posts_found) if session.posts_found > 0 else 0
        }
        
        # Filter statistics
        filter_statistics = {
            'total_found': session.posts_found,
            'successfully_processed': session.posts_processed,
            'filtered_out': session.posts_filtered,
            'final_count': len(posts),
            'high_priority_count': len(high_priority_posts),
            'opportunities_count': len(opportunities)
        }
        
        return ScrapingResult(
            session=session,
            posts=posts,
            high_priority_posts=high_priority_posts,
            opportunities=opportunities,
            performance_summary=performance_summary,
            filter_statistics=filter_statistics,
            success=True
        )
    
    def _create_failure_result(self, session: ScrapingSession, error_message: str) -> ScrapingResult:
        """Create failure scraping result"""
        duration = safe_datetime_diff(datetime.now(), session.start_time)
        
        performance_summary = {
            'session_duration_seconds': duration,
            'posts_found': session.posts_found,
            'errors_encountered': session.errors_encountered,
            'scroll_attempts': session.scroll_attempts,
            'navigation_attempts': session.navigation_attempts
        }
        
        filter_statistics = {
            'total_found': session.posts_found,
            'error_message': error_message
        }
        
        return ScrapingResult(
            session=session,
            posts=[],
            high_priority_posts=[],
            opportunities=[],
            performance_summary=performance_summary,
            filter_statistics=filter_statistics,
            success=False,
            error_message=error_message
        )
    
    def _cleanup_session(self, session: ScrapingSession) -> None:
        """Clean up after scraping session"""
        try:
            # Clear extraction cache if it's getting too large
            if self.extraction_cache:
                self.extraction_cache.clear_expired()
            
            # Check if browser session should be restarted based on uptime
            if self.browser.should_restart_session():
                logger.logger.info(f"Browser session uptime exceeded limit, initiating restart")
                restart_success = self.browser.restart_browser_session()
                if restart_success:
                    logger.logger.info("Browser session restarted successfully during cleanup")
                else:
                    logger.logger.warning("Browser session restart failed during cleanup")
            
            # Log session summary
            duration = safe_datetime_diff(datetime.now(), session.start_time)
            logger.logger.info(f"Session {session.session_id} summary:")
            logger.logger.info(f"   Duration: {duration:.1f}s")
            logger.logger.info(f"   Posts found: {session.posts_found}")
            logger.logger.info(f"   Posts processed: {session.posts_processed}")
            logger.logger.info(f"   Errors: {session.errors_encountered}")
            logger.logger.info(f"   Scrolls: {session.scroll_attempts}")
            
        except Exception as e:
            logger.logger.debug(f"Cleanup error: {str(e)}")
    
    # Legacy compatibility methods for existing bot integration
    
    def scrape_timeline_legacy(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility with existing bot
        Returns just the posts list as the original scraper did
        """
        logger.logger.info("🔄 Running in legacy compatibility mode")
        
        result = self.scrape_timeline(count)
        
        if result.success:
            return result.posts
        else:
            logger.logger.warning(f"Scraping failed: {result.error_message}")
            return []
    
    def find_market_related_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy method - now handled automatically in the main workflow"""
        logger.logger.debug("Legacy find_market_related_posts called - using modern filtering")
        
        # Apply modern content analysis and filtering
        filtered_posts = self.content_filter.filter_posts(posts)
        analyzed_posts = []
        
        for post in filtered_posts:
            # Add legacy market_related flag for compatibility
            analysis = self.content_analyzer.analyze_content(post)
            relevance = self.relevance_analyzer.calculate_market_relevance(post, analysis)
            
            if relevance.final_score >= 2.0:  # Minimum relevance threshold
                post['market_related'] = True
                post['market_keywords'] = analysis.crypto_keywords + analysis.finance_keywords
                post['market_relevance_score'] = relevance.final_score
                analyzed_posts.append(post)
        
        return analyzed_posts
    
    def filter_already_replied_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy method - filter posts we've already replied to"""
        if not self.db:
            logger.logger.warning("No database available for reply filtering")
            return posts
        
        filtered_posts = []
        
        for post in posts:
            post_id = post.get('post_id')
            post_url = post.get('post_url')
            
            try:
                if post_id or post_url:
                    already_replied = self.db.check_if_post_replied(post_id, post_url)
                    if not already_replied:
                        filtered_posts.append(post)
                else:
                    # Include posts without IDs to be safe
                    filtered_posts.append(post)
            except Exception as e:
                logger.logger.debug(f"Error checking reply status: {str(e)}")
                filtered_posts.append(post)
        
        logger.logger.info(f"Reply filtering: {len(filtered_posts)}/{len(posts)} posts haven't been replied to")
        return filtered_posts
    
    @ensure_naive_datetimes
    def prioritize_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy method - now uses advanced prioritization"""
        logger.logger.debug("Legacy prioritize_posts called - using modern prioritization")
        
        return self.content_prioritizer.prioritize_posts(posts, len(posts))
    
    def navigate_to_home_timeline(self) -> bool:
        """Legacy method for timeline navigation"""
        session = ScrapingSession(
            session_id=f"legacy_{int(time.time())}",
            start_time=datetime.now()
        )
        
        return self._navigate_to_timeline(session)                        
