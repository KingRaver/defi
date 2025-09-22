#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import inspect
import time
import re
from typing import Any, Callable, Dict, List, Optional, Set
from utils.logger import logger


class BotMethodMonitor:
    """
    Comprehensive monitoring class for bot.py methods to track posting pipeline,
    detect Claude enhancement bypasses, and log all method calls.
    """
    
    def __init__(self):
        self.call_stack: List[Dict[str, Any]] = []
        self.posting_methods: Set[str] = {
            'post_tweet_with_claude_enhancement',
            'post_tweet',
            'send_tweet',
            'publish_tweet',
            'create_tweet'
        }
        self.enhancement_indicators = {
            'structure_patterns': [
                r'Here\'s an analysis',
                r'Here is an analysis', 
                r'Analysis of.*:',
                r'\d+\.\s+\w+',  # Numbered lists
                r'Technology Impact:',
                r'Key findings:'
            ],
            'hashtags': r'#[A-Z]{2,}',  # Uppercase hashtags like #AI #ETH
            'engagement_language': [
                r'Here\'s',
                r'Let\'s dive into',
                r'Breaking down',
                r'Key takeaways'
            ]
        }
        
    def monitor_bot_class(self, bot_class):
        """
        Wrap all methods in the bot class with monitoring
        
        Args:
            bot_class: The bot class to monitor
            
        Returns:
            The wrapped bot class
        """
        logger.logger.info("🔍 Starting comprehensive bot method monitoring")
        
        # Get all methods from the bot class
        for attr_name in dir(bot_class):
            if not attr_name.startswith('_'):  # Skip private methods
                attr = getattr(bot_class, attr_name)
                if callable(attr):
                    # Wrap the method with monitoring
                    wrapped_method = self._wrap_method(attr, attr_name)
                    setattr(bot_class, attr_name, wrapped_method)
                    logger.logger.debug(f"📊 Wrapped method: {attr_name}")
        
        return bot_class
    
    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """
        Wrap individual method with comprehensive monitoring
        
        Args:
            method: Method to wrap
            method_name: Name of the method
            
        Returns:
            Wrapped method with monitoring
        """
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            call_id = f"{method_name}_{int(time.time() * 1000)}"
            start_time = time.time()
            
            # Log method entry
            logger.logger.info(f"🎯 METHOD CALL: {method_name}")
            logger.logger.debug(f"   📝 Call ID: {call_id}")
            logger.logger.debug(f"   📊 Args count: {len(args)}")
            logger.logger.debug(f"   🔧 Kwargs: {list(kwargs.keys())}")
            
            # Track call in stack
            call_info = {
                'call_id': call_id,
                'method_name': method_name,
                'start_time': start_time,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'is_posting_method': method_name in self.posting_methods
            }
            self.call_stack.append(call_info)
            
            try:
                # Check if this is a posting method and analyze content
                if self._is_posting_method(method_name):
                    self._analyze_posting_call(method_name, args, kwargs, call_id)
                
                # Execute the original method
                result = method(*args, **kwargs)
                
                # Log successful completion
                execution_time = time.time() - start_time
                logger.logger.info(f"✅ METHOD SUCCESS: {method_name} ({execution_time:.3f}s)")
                
                # Update call info
                call_info.update({
                    'success': True,
                    'execution_time': execution_time,
                    'result_type': type(result).__name__
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.logger.error(f"❌ METHOD ERROR: {method_name} ({execution_time:.3f}s) - {str(e)}")
                
                # Update call info
                call_info.update({
                    'success': False,
                    'execution_time': execution_time,
                    'error': str(e)
                })
                
                raise
            
            finally:
                # Always log the call summary
                self._log_call_summary(call_info)
        
        return wrapper
    
    def _is_posting_method(self, method_name: str) -> bool:
        """Check if method is related to posting content"""
        posting_keywords = ['post', 'tweet', 'send', 'publish', 'create_tweet']
        return (method_name in self.posting_methods or 
                any(keyword in method_name.lower() for keyword in posting_keywords))
    
    def _analyze_posting_call(self, method_name: str, args: tuple, kwargs: dict, call_id: str):
        """
        Analyze posting method calls to detect Claude enhancement bypasses
        
        Args:
            method_name: Name of the posting method
            args: Method arguments
            kwargs: Method keyword arguments
            call_id: Unique call identifier
        """
        logger.logger.warning(f"🚨 POSTING METHOD DETECTED: {method_name}")
        
        # Extract content from arguments
        content = self._extract_content_from_args(args, kwargs)
        
        if content:
            logger.logger.info(f"📄 Content found in {method_name}: {content[:100]}...")
            
            # Check if content has been Claude enhanced
            enhancement_status = self._check_claude_enhancement(content)
            
            if enhancement_status['is_enhanced']:
                logger.logger.info(f"✅ CONTENT ENHANCED: {method_name}")
                logger.logger.debug(f"   🎯 Enhancement indicators: {enhancement_status['indicators']}")
            else:
                logger.logger.error(f"🚫 ENHANCEMENT BYPASS DETECTED: {method_name}")
                logger.logger.error(f"   📝 Raw content: {content}")
                logger.logger.error(f"   🔍 Missing indicators: structure, hashtags, engagement language")
                
                # Log detailed bypass information
                self._log_enhancement_bypass(method_name, content, call_id)
        else:
            logger.logger.warning(f"⚠️  No content found in posting method: {method_name}")
    
    def _extract_content_from_args(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Extract content from method arguments"""
        # Check common argument names for content
        content_keys = ['content', 'text', 'tweet_text', 'message', 'post_content']
        
        # Check kwargs first
        for key in content_keys:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]
        
        # Check positional args (usually first or second argument)
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10:  # Reasonable content length
                return arg
        
        return None
    
    def _check_claude_enhancement(self, content: str) -> Dict[str, Any]:
        """
        Check if content shows signs of Claude enhancement
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with enhancement status and indicators found
        """
        enhancement_status = {
            'is_enhanced': False,
            'indicators': [],
            'score': 0
        }
        
        # Check for structure patterns
        structure_score = 0
        for pattern in self.enhancement_indicators['structure_patterns']:
            if re.search(pattern, content, re.IGNORECASE):
                enhancement_status['indicators'].append(f"Structure: {pattern}")
                structure_score += 1
        
        # Check for hashtags (uppercase)
        hashtag_matches = re.findall(self.enhancement_indicators['hashtags'], content)
        if hashtag_matches:
            enhancement_status['indicators'].append(f"Hashtags: {hashtag_matches}")
            structure_score += len(hashtag_matches)
        
        # Check for engagement language
        engagement_score = 0
        for pattern in self.enhancement_indicators['engagement_language']:
            if re.search(pattern, content, re.IGNORECASE):
                enhancement_status['indicators'].append(f"Engagement: {pattern}")
                engagement_score += 1
        
        # Calculate final score and determine if enhanced
        total_score = structure_score + engagement_score
        enhancement_status['score'] = total_score
        
        # Content is considered enhanced if it has at least 2 indicators
        enhancement_status['is_enhanced'] = total_score >= 2
        
        return enhancement_status
    
    def _log_enhancement_bypass(self, method_name: str, content: str, call_id: str):
        """Log detailed information about enhancement bypass"""
        logger.logger.error("=" * 80)
        logger.logger.error("🚨 CLAUDE ENHANCEMENT BYPASS DETECTED 🚨")
        logger.logger.error("=" * 80)
        logger.logger.error(f"Method: {method_name}")
        logger.logger.error(f"Call ID: {call_id}")
        logger.logger.error(f"Content Length: {len(content)} characters")
        logger.logger.error("Content Preview:")
        logger.logger.error(f"'{content[:200]}{'...' if len(content) > 200 else ''}'")
        logger.logger.error("Expected Enhancement Indicators Missing:")
        logger.logger.error("  - Structured format (numbered lists, sections)")
        logger.logger.error("  - Engagement language ('Here's an analysis', etc.)")
        logger.logger.error("  - Proper hashtags (#AI, #ETH, etc.)")
        logger.logger.error("=" * 80)
    
    def _log_call_summary(self, call_info: Dict[str, Any]):
        """Log summary of method call"""
        status = "SUCCESS" if call_info.get('success', False) else "FAILED"
        execution_time = call_info.get('execution_time', 0)
        
        logger.logger.debug(f"📊 CALL SUMMARY [{call_info['call_id']}]:")
        logger.logger.debug(f"   Method: {call_info['method_name']}")
        logger.logger.debug(f"   Status: {status}")
        logger.logger.debug(f"   Time: {execution_time:.3f}s")
        logger.logger.debug(f"   Posting Method: {call_info['is_posting_method']}")
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get statistics about monitored calls"""
        total_calls = len(self.call_stack)
        posting_calls = len([call for call in self.call_stack if call['is_posting_method']])
        failed_calls = len([call for call in self.call_stack if not call.get('success', True)])
        
        return {
            'total_calls': total_calls,
            'posting_calls': posting_calls,
            'failed_calls': failed_calls,
            'success_rate': (total_calls - failed_calls) / total_calls if total_calls > 0 else 0,
            'recent_calls': self.call_stack[-10:] if self.call_stack else []
        }
    
    def clear_call_history(self):
        """Clear the call stack history"""
        logger.logger.info(f"🧹 Clearing call history ({len(self.call_stack)} calls)")
        self.call_stack.clear()


# Usage example:
def setup_bot_monitoring(bot_instance):
    """
    Setup monitoring for a bot instance
    
    Args:
        bot_instance: Instance of your bot class
        
    Returns:
        Monitored bot instance
    """
    monitor = BotMethodMonitor()
    
    # Wrap the bot class
    monitored_bot_class = monitor.monitor_bot_class(bot_instance.__class__)
    
    # Create new instance with monitoring
    monitored_bot = monitored_bot_class.__new__(monitored_bot_class)
    monitored_bot.__dict__.update(bot_instance.__dict__)
    
    logger.logger.info("🚀 Bot monitoring setup complete")
    
    return monitored_bot, monitor
