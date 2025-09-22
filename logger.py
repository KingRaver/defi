#!/usr/bin/env python3# -*- coding: utf-8 -*-

from typing import Optional, Union
import os
import logging
import time
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import logging.config

class CorrelationLogger:
    def __init__(self) -> None:
        # Setup unified logging first (before any other logger setup)
        self.logger = self.setup_unified_logging()
        
        # Get references to specific sub-loggers (these are already configured by setup_unified_logging)
        self.coingecko_logger = logging.getLogger('ETHBTCCorrelation.coingecko')
        self.client_logger = logging.getLogger('ETHBTCCorrelation.claude')
        self.sheets_logger = logging.getLogger('ETHBTCCorrelation.google_sheets')
        self.analysis_logger = logging.getLogger('ETHBTCCorrelation.analysis')
        
        # Create a reference to the M4TechnicalFoundation logger
        self.m4_logger = logging.getLogger('M4TechnicalFoundation')

    def info(self, message):
        """Log an info message (compatibility method)"""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug message (compatibility method)"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log a warning message (compatibility method)"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log an error message (compatibility method)"""
        self.logger.error(message, exc_info=exc_info)    

    def _setup_api_logger(self, api_name: str) -> logging.Logger:
        """Setup specific logger for each API with its own file"""
        # The loggers are already configured by setup_unified_logging
        # Just return the appropriate logger based on the API name
        return logging.getLogger(f'ETHBTCCorrelation.{api_name}')

    def _setup_analysis_logger(self) -> logging.Logger:
        """Setup specific logger for market analysis"""
        # The logger is already configured by setup_unified_logging
        return logging.getLogger('ETHBTCCorrelation.analysis')

    def log_coingecko_request(self, endpoint: str, success: bool = True) -> None:
        """Log Coingecko API interactions"""
        msg = f"CoinGecko API Request - Endpoint: {endpoint}"
        if success:
            self.coingecko_logger.info(msg)
        else:
            self.coingecko_logger.error(msg)

    def log_claude_analysis(
        self, 
        btc_price: float, 
        eth_price: float, 
        status: bool = True
    ) -> None:
        """Log Claude analysis details"""
        msg = (
            "Claude Analysis - "
            f"BTC Price: ${btc_price:,.2f} - "
            f"ETH Price: ${eth_price:,.2f}"
        )
        
        if status:
            self.client_logger.info(msg)
        else:
            self.client_logger.error(msg)

    def log_sheets_update(
        self, 
        data_type: str, 
        status: bool = True
    ) -> None:
        """Log Google Sheets interactions"""
        msg = f"Google Sheets Update - Data Type: {data_type}"
       
        if status:
            self.sheets_logger.info(msg)
        else:
            self.sheets_logger.error(msg)

    def log_market_correlation(
        self, 
        correlation_coefficient: float, 
        price_movement: float
    ) -> None:
        """Log market correlation details"""
        self.logger.info(
            "Market Correlation - "
            f"Correlation Coefficient: {correlation_coefficient:.2f} - "
            f"Price Movement: {price_movement:.2f}%"
        )

    def log_error(
        self, 
        error_type: str, 
        message: str, 
        exc_info: Union[bool, Exception, None] = None
    ) -> None:
        """Log errors with stack trace option"""
        self.logger.error(
            f"Error - Type: {error_type} - Message: {message}",
            exc_info=exc_info if exc_info else False
        )

    def log_twitter_action(self, action_type: str, status: str) -> None:
        """Log Twitter related actions"""
        self.logger.info(f"Twitter Action - Type: {action_type} - Status: {status}")

    def log_startup(self) -> None:
        """Log application startup"""
        self.logger.info("=" * 50)
        self.logger.info(f"ETH-BTC Correlation Bot Starting - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)

    def log_shutdown(self) -> None:
        """Log application shutdown"""
        self.logger.info("=" * 50)
        self.logger.info(f"ETH-BTC Correlation Bot Shutting Down - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)

    def setup_unified_logging(self):
        """
        Configure a unified logging system to prevent duplicate logs.
        This resets any existing logging configuration to ensure consistency.
        """
        # Reset all existing loggers to prevent duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Clear any existing loggers
        logging.Logger.manager.loggerDict.clear()
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('logs/analysis'):
            os.makedirs('logs/analysis')
        if not os.path.exists('logs/technical'):
            os.makedirs('logs/technical')
        
        # Configure the logging system
        logging_config = {
            'version': 1,
            'disable_existing_loggers': True,  # This disables existing loggers
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'm4_formatter': {
                    'format': '%(asctime)s | 🚀 %(name)s | %(levelname)s | %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
            },
            'handlers': {
                'console': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/eth_btc_correlation.log',
                    'maxBytes': 10*1024*1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
                'coingecko_file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/coingecko_api.log',
                    'maxBytes': 5*1024*1024,  # 5MB
                    'backupCount': 3,
                    'encoding': 'utf-8',
                },
                'claude_file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/claude_api.log',
                    'maxBytes': 5*1024*1024,  # 5MB
                    'backupCount': 3,
                    'encoding': 'utf-8',
                },
                'sheets_file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/google_sheets_api.log',
                    'maxBytes': 5*1024*1024,  # 5MB
                    'backupCount': 3,
                    'encoding': 'utf-8',
                },
                'analysis_file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/analysis/market_analysis.log',
                    'maxBytes': 10*1024*1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
                'm4_console': {
                    'level': 'INFO',
                    'formatter': 'm4_formatter',
                    'class': 'logging.StreamHandler',
                },
                'm4_file': {
                    'level': 'INFO',
                    'formatter': 'm4_formatter',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/technical/m4_foundation.log',
                    'maxBytes': 10*1024*1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'ETHBTCCorrelation': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False,
                },
                'ETHBTCCorrelation.coingecko': {
                    'handlers': ['coingecko_file'],
                    'level': 'INFO',
                    'propagate': True,  # Propagate to parent logger for console output
                },
                'ETHBTCCorrelation.claude': {
                    'handlers': ['claude_file'],
                    'level': 'INFO',
                    'propagate': True,  # Propagate to parent logger for console output
                },
                'ETHBTCCorrelation.google_sheets': {
                    'handlers': ['sheets_file'],
                    'level': 'INFO',
                    'propagate': True,  # Propagate to parent logger for console output
                },
                'ETHBTCCorrelation.analysis': {
                    'handlers': ['analysis_file'],
                    'level': 'INFO',
                    'propagate': True,  # Propagate to parent logger for console output
                },
                'M4TechnicalFoundation': {
                    'handlers': ['m4_console', 'm4_file'],
                    'level': 'INFO',
                    'propagate': False,  # Don't propagate to avoid duplicate logs
                },
            }
        }
        
        # Apply the configuration
        logging.config.dictConfig(logging_config)
        
        # Let's inform that logging has been set up
        logger = logging.getLogger('ETHBTCCorrelation')
        logger.info("🔧 Unified logging system initialized")
        
        # Also ensure M4TechnicalFoundation logger is initialized once
        m4_logger = logging.getLogger('M4TechnicalFoundation')
        m4_logger.info("🔧 M4TechnicalFoundation logger initialized")
        
        return logger

    def log_sheets_operation(
        self, 
        operation_type: str, 
        status: bool, 
        details: Optional[str] = None
    ) -> None:
        """Log Google Sheets operations"""
        msg = f"Google Sheets Operation - Type: {operation_type} - Status: {'Success' if status else 'Failed'}"
        if details:
            msg += f" - Details: {details}"
        
        if status:
            self.sheets_logger.info(msg)
        else:
            self.sheets_logger.error(msg)

# Singleton instance
logger = CorrelationLogger()
