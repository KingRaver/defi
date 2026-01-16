import sqlite3
import threading
import traceback
from datetime import datetime, timedelta, timezone
import json
from typing import Dict, List, Optional, Union, Any, Tuple, cast
from dataclasses import asdict
import os
from utils.logger import logger

class CryptoDatabase:
    """Database handler for cryptocurrency market data, analysis and predictions"""
    
    def __init__(self, db_path: str = "data/crypto_history.db"):
        """Initialize database connection and create tables if they don't exist"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage
        self._initialize_database()
        self.add_ichimoku_column()
        self.add_missing_columns()
        self.add_replied_posts_table()
        self.add_price_history_table()
        self.add_sparkline_table()
        self.add_sparkline_column()
        self.add_price_range_columns()
    
    def strip_timezone(self, dt):
        """
        Ensure datetime is timezone-naive by converting to UTC and removing tzinfo
    
        Args:
            dt: Datetime object that might have timezone info
        
        Returns:
            Timezone-naive datetime object in UTC
        """
        from datetime import datetime, timezone
    
        # If it's not a datetime, try to convert
        if not isinstance(dt, datetime):
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except ValueError:
                    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(dt, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError(f"Unable to parse datetime string: {dt}")
            else:
                raise TypeError(f"Expected datetime or string, got {type(dt)}")

        # Handle timezone - dt is guaranteed to be datetime here
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        return dt

    def standardize_timestamp_for_storage(self, timestamp):
        """
        Prepare timestamp for consistent database storage
        - Converts to UTC if timezone-aware
        - Strips timezone info 
        - Ensures consistent ISO format
    
        Args:
            timestamp: Datetime object or string to standardize
        
        Returns:
            Timestamp string formatted for storage
        """
        # First ensure we have a standardized datetime object
        timestamp = self.strip_timezone(timestamp)
    
        # Format with microsecond precision for accurate ordering
        return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')

    def standardize_timestamp_for_query(self, target_time):
        """
        Prepare timestamp for database queries
        - Handles various input formats
        - Ensures consistent output format for SQLite
    
        Args:
            target_time: Datetime object or string to standardize
        
        Returns:
            Timestamp in format suitable for SQLite queries
        """
        # Use our comprehensive standardization method
        target_time = self.strip_timezone(target_time)
    
        # Return formatted for SQLite query
        return target_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    

    def _standardize_timestamp(self, timestamp):
        """
        Ensure timestamp is in a consistent format for database operations
    
        Args:
            timestamp: Datetime object or string
    
        Returns:
            Standardized datetime object (timezone-naive)
        """
        # Use our strip_timezone method which handles all the conversion logic
        return self.strip_timezone(timestamp)
    
    def add_price_range_columns(self):
        """Add high_price and low_price columns to price_history table"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("PRAGMA table_info(price_history)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'high_price' not in columns:
                cursor.execute("ALTER TABLE price_history ADD COLUMN high_price REAL")
                
            if 'low_price' not in columns:
                cursor.execute("ALTER TABLE price_history ADD COLUMN low_price REAL")
                
            conn.commit()
            return True
        except Exception as e:
            logger.log_error("Add Price Range Columns", str(e))
            conn.rollback()
            return False

    def _get_closest_historical_price(self, token, target_time, max_time_difference_hours=None):
        """
        Get the closest historical price to the target time

        Args:
            token: Token symbol
            target_time: Target datetime
            max_time_difference_hours: Maximum allowed time difference in hours (optional)
    
        Returns:
            dict with price, timestamp and time_difference or None if no suitable record found
        """
        from datetime import datetime

        conn, cursor = self._get_connection()

        try:
            # Standardize target time for querying
            target_time_std = self.standardize_timestamp_for_query(target_time)
        
            # Convert max_time_difference to seconds if provided
            max_time_difference = None
            if max_time_difference_hours is not None:
                max_time_difference = max_time_difference_hours * 3600
    
            # First try to find a price point BEFORE the target time
            cursor.execute("""
                SELECT price, timestamp
                FROM price_history
                WHERE token = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (token, target_time_std))
    
            before_result = cursor.fetchone()
    
            # Then try to find a price point AFTER the target time
            cursor.execute("""
                SELECT price, timestamp
                FROM price_history
                WHERE token = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT 1
            """, (token, target_time_std))
    
            after_result = cursor.fetchone()
    
            # No data available at all
            if not before_result and not after_result:
                return None
        
            # Calculate time differences and find the closest point
            best_result = None
            smallest_diff = float('inf')
    
            # Parse target_time to datetime object for comparison
            target_time_dt = self.strip_timezone(target_time)
    
            if before_result:
                before_time = self.strip_timezone(before_result['timestamp'])
                before_diff = abs((target_time_dt - before_time).total_seconds())
        
                if max_time_difference is None or before_diff <= max_time_difference:
                    best_result = before_result
                    smallest_diff = before_diff
    
            if after_result:
                after_time = self.strip_timezone(after_result['timestamp'])
                after_diff = abs((target_time_dt - after_time).total_seconds())
        
                if (max_time_difference is None or after_diff <= max_time_difference) and after_diff < smallest_diff:
                    best_result = after_result
                    smallest_diff = after_diff
    
            if best_result:
                actual_time = self.strip_timezone(best_result['timestamp'])
                time_diff_hours = smallest_diff / 3600
        
                return {
                    'price': best_result['price'],
                    'timestamp': actual_time,
                    'time_difference_hours': time_diff_hours
                }
            else:
                return None
    
        except Exception as e:
            # Add appropriate logging
            return None
        
    def create_billionaire_tracking_tables(self):
        """
        Create tables for billionaire wealth tracking system
    
        This method adds the necessary tables to track:
        - Portfolio performance and milestones
        - Trade history for wealth generation
        - Risk management and position sizing
        - Performance metrics and analytics
        """
        conn, cursor = self._get_connection()
    
        try:
            # Billionaire Portfolio Tracking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_portfolio_value REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    total_return_pct REAL,
                    daily_return_pct REAL,
                    max_drawdown_pct REAL,
                    positions_count INTEGER,
                    risk_level TEXT,
                    wealth_milestone TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Trade History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    action TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    portfolio_allocation_pct REAL,
                    position_size_pct REAL,
                    risk_score REAL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    trade_reason TEXT,
                    technical_signals TEXT,  -- JSON string of signals
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Milestones Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_milestones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    milestone_type TEXT NOT NULL,  -- 'first_million', 'ten_million', etc.
                    milestone_value REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    time_to_achieve INTEGER,  -- days from start
                    strategy_used TEXT,
                    performance_metrics TEXT,  -- JSON string
                    celebration_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Performance Metrics Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_return REAL,
                    cumulative_return REAL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_consecutive_wins INTEGER,
                    max_consecutive_losses INTEGER,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win REAL,
                    avg_loss REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
        
            # Billionaire Risk Management Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_risk_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    portfolio_value REAL NOT NULL,
                    total_risk_exposure REAL,
                    max_position_size_pct REAL,
                    risk_per_trade_pct REAL,
                    correlation_risk REAL,
                    leverage_used REAL,
                    var_95 REAL,  -- Value at Risk 95%
                    expected_shortfall REAL,
                    risk_adjusted_return REAL,
                    risk_level TEXT,  -- 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Wealth Targets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_wealth_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_name TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    current_progress REAL,
                    progress_pct REAL,
                    estimated_time_to_achieve INTEGER,  -- days
                    strategy_focus TEXT,
                    priority_level INTEGER,
                    is_achieved BOOLEAN DEFAULT FALSE,
                    achieved_date DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Insert default wealth targets if table is empty
            cursor.execute("SELECT COUNT(*) FROM billionaire_wealth_targets")
            if cursor.fetchone()[0] == 0:
                default_targets = [
                    ('First Million', 1_000_000, 1),
                    ('Ten Million', 10_000_000, 2),
                    ('Hundred Million', 100_000_000, 3),
                    ('Quarter Billion', 250_000_000, 4),
                    ('Half Billion', 500_000_000, 5),
                    ('First Billion', 1_000_000_000, 6),
                    ('Five Billion', 5_000_000_000, 7),
                    ('Ten Billion', 10_000_000_000, 8),
                    ('Ultimate Target', 50_000_000_000, 9)
                ]
            
                for target_name, target_value, priority in default_targets:
                    cursor.execute("""
                        INSERT INTO billionaire_wealth_targets 
                        (target_name, target_value, current_progress, progress_pct, priority_level)
                        VALUES (?, ?, 0, 0, ?)
                    """, (target_name, target_value, priority))
        
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_portfolio_timestamp 
                ON billionaire_portfolio(timestamp)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_trades_timestamp_token 
                ON billionaire_trades(timestamp, token)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_performance_date 
                ON billionaire_performance(date)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_milestones_type 
                ON billionaire_milestones(milestone_type)
            """)
        
            conn.commit()
            logger.logger.info("✅ Billionaire tracking tables created successfully")
        
            # Log the tables created
            tables_created = [
                "billionaire_portfolio",
                "billionaire_trades", 
                "billionaire_milestones",
                "billionaire_performance",
                "billionaire_risk_management",
                "billionaire_wealth_targets"
            ]
        
            logger.logger.info(f"💰 Created {len(tables_created)} billionaire tracking tables:")
            for table in tables_created:
                logger.logger.info(f"   📊 {table}")
        
            return True
        
        except Exception as e:
            logger.log_error("Create Billionaire Tracking Tables", str(e))
            conn.rollback()
            return False


    # Additional helper methods for the billionaire system

    def store_billionaire_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Store a billionaire trade in the database"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_trades (
                    timestamp, token, action, quantity, price, total_value,
                    portfolio_allocation_pct, position_size_pct, risk_score,
                    profit_loss, profit_loss_pct, trade_reason, technical_signals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('token'),
                trade_data.get('action'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('total_value'),
                trade_data.get('portfolio_allocation_pct'),
                trade_data.get('position_size_pct'),
                trade_data.get('risk_score'),
                trade_data.get('profit_loss'),
                trade_data.get('profit_loss_pct'),
                trade_data.get('trade_reason'),
                json.dumps(trade_data.get('technical_signals', {}))
            ))
        
            trade_id = cursor.lastrowid
            conn.commit()
            logger.logger.info(f"💰 Billionaire trade stored: {trade_data.get('action')} {trade_data.get('token')}")
            return trade_id
        
        except Exception as e:
            logger.log_error("Store Billionaire Trade", str(e))
            conn.rollback()
            return None


    def update_billionaire_portfolio(self, portfolio_data: Dict[str, Any]) -> bool:
        """Update billionaire portfolio tracking"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_portfolio (
                    timestamp, total_portfolio_value, initial_capital,
                    total_return_pct, daily_return_pct, max_drawdown_pct,
                    positions_count, risk_level, wealth_milestone, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_data.get('timestamp', datetime.now()),
                portfolio_data.get('total_portfolio_value'),
                portfolio_data.get('initial_capital'),
                portfolio_data.get('total_return_pct'),
                portfolio_data.get('daily_return_pct'),
                portfolio_data.get('max_drawdown_pct'),
                portfolio_data.get('positions_count'),
                portfolio_data.get('risk_level'),
                portfolio_data.get('wealth_milestone'),
                portfolio_data.get('notes')
            ))
        
            conn.commit()
            logger.logger.info(f"💰 Portfolio updated: ${portfolio_data.get('total_portfolio_value', 0):,.2f}")
            return True
        
        except Exception as e:
            logger.log_error("Update Billionaire Portfolio", str(e))
            conn.rollback()
            return False


    def record_billionaire_milestone(self, milestone_data: Dict[str, Any]) -> bool:
        """Record a billionaire wealth milestone achievement"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_milestones (
                    timestamp, milestone_type, milestone_value, portfolio_value,
                    time_to_achieve, strategy_used, performance_metrics, celebration_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                milestone_data.get('timestamp', datetime.now()),
                milestone_data.get('milestone_type'),
                milestone_data.get('milestone_value'),
                milestone_data.get('portfolio_value'),
                milestone_data.get('time_to_achieve'),
                milestone_data.get('strategy_used'),
                json.dumps(milestone_data.get('performance_metrics', {})),
                milestone_data.get('celebration_notes')
            ))
        
            # Update the wealth targets table
            cursor.execute("""
                UPDATE billionaire_wealth_targets 
                SET is_achieved = TRUE, achieved_date = ?, current_progress = ?, progress_pct = 100
                WHERE target_name = ?
            """, (
                datetime.now(),
                milestone_data.get('milestone_value'),
                milestone_data.get('milestone_type')
            ))
        
            conn.commit()
            logger.logger.info(f"🎉 MILESTONE ACHIEVED: {milestone_data.get('milestone_type')} - ${milestone_data.get('milestone_value'):,.2f}")
            return True
        
        except Exception as e:
            logger.log_error("Record Billionaire Milestone", str(e))
            conn.rollback()
            return False    

    def add_price_history_table(self):
        """
        Add price_history table for tracking historical price data
        This allows us to calculate price changes ourselves instead of relying on external APIs
        """
        conn, cursor = self._get_connection()
        try:
            # Create price_history table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume REAL,
                    market_cap REAL,
                    total_supply REAL,
                    circulating_supply REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token, timestamp)
                )
            """)
        
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token ON price_history(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token_timestamp ON price_history(token, timestamp)")
        
            conn.commit()
            logger.logger.info("Added price_history table to database")
            return True
        except Exception as e:
            logger.log_error("Add Price History Table", str(e))
            conn.rollback()
            return False

    def store_price_history(self, token: str, price: float, volume: Optional[float] = None, 
                            market_cap: Optional[float] = None, total_supply: Optional[float] = None, 
                            circulating_supply: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Store price data in the price_history table with enhanced timestamp handling
    
        Args:
            token: Token symbol
            price: Current price
            volume: Trading volume (optional)
            market_cap: Market capitalization (optional)
            total_supply: Total supply (optional)
            circulating_supply: Circulating supply (optional)
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Input validation with logging
            logger.logger.debug(f"store_price_history called for token: {token}")
        
            if not token:
                logger.logger.error("store_price_history called with empty token")
                return False
        
            if not isinstance(price, (int, float)) or price <= 0:
                logger.logger.error(f"store_price_history called with invalid price for {token}: {price}")
                return False
        
            logger.logger.debug(f"Price value for {token}: {price}")
        
            # Handle timestamp with standardization
            if timestamp is None:
                timestamp = datetime.now()
                logger.logger.debug(f"Using current time for {token}: {timestamp}")
        
            # Standardize timestamp for consistent storage
            std_timestamp = self._standardize_timestamp(timestamp)
            logger.logger.debug(f"Standardized timestamp for {token}: {std_timestamp}")
        
            conn, cursor = self._get_connection()
        
            # Make sure the table exists
            logger.logger.debug(f"Ensuring price_history table exists for {token}")
            self._ensure_price_history_table_exists()
        
            # Check if we already have data for this token at this timestamp
            cursor.execute("""
                SELECT id, price FROM price_history
                WHERE token = ? AND timestamp = ?
            """, (token, std_timestamp))
        
            existing_record = cursor.fetchone()
            if existing_record:
                logger.logger.debug(f"Found existing record for {token} at {std_timestamp}: id={existing_record['id']}, price={existing_record['price']}")
            else:
                logger.logger.debug(f"No existing record for {token} at {std_timestamp}")
        
            # Insert or replace price data
            logger.logger.debug(f"Inserting/replacing price data for {token}")
            cursor.execute("""
                INSERT OR REPLACE INTO price_history (
                    token, timestamp, price, volume, market_cap, 
                    total_supply, circulating_supply
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                token,
                std_timestamp,
                price,
                volume,
                market_cap,
                total_supply,
                circulating_supply
            ))
        
            conn.commit()
            logger.logger.debug(f"Successfully committed price history for {token}")
            return True
        
        except Exception as e:
            conn = None
            try:
                conn, cursor = self._get_connection()
            except:
                pass
            
            logger.log_error(f"Store Price History - {token}", str(e))
            logger.logger.error(f"Error in store_price_history for {token}: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            if conn:
                conn.rollback()
            return False

    def _ensure_price_history_table_exists(self):
        """Ensure price_history table exists in the database with enhanced logging"""
        conn, cursor = self._get_connection()
        try:
            logger.logger.debug("_ensure_price_history_table_exists called")
        
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
            table_exists = cursor.fetchone() is not None
        
            if table_exists:
                logger.logger.debug("price_history table already exists")
            
                # Additional validation: check table structure
                cursor.execute("PRAGMA table_info(price_history)")
                columns = cursor.fetchall()
                column_names = [column[1] for column in columns]
            
                logger.logger.debug(f"price_history table has columns: {column_names}")
            
                # Check for required columns
                required_columns = ['id', 'token', 'timestamp', 'price', 'volume', 'market_cap', 
                                  'total_supply', 'circulating_supply', 'created_at']
            
                missing_columns = [col for col in required_columns if col not in column_names]
            
                if missing_columns:
                    logger.logger.warning(f"price_history table is missing columns: {missing_columns}")
                    # We could add code here to alter the table and add missing columns if needed
                else:
                    logger.logger.debug("price_history table structure is valid")
                
                # Check table indices
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='price_history'")
                indices = [idx[0] for idx in cursor.fetchall()]
            
                logger.logger.debug(f"price_history table has indices: {indices}")
            
                required_indices = ['idx_price_history_token', 'idx_price_history_timestamp', 
                              'idx_price_history_token_timestamp']
            
                missing_indices = [idx for idx in required_indices if idx not in indices]
            
                if missing_indices:
                    logger.logger.warning(f"price_history table is missing indices: {missing_indices}")
                
                    # Create missing indices
                    for idx in missing_indices:
                        try:
                            if idx == 'idx_price_history_token':
                                logger.logger.debug("Creating index: idx_price_history_token")
                                cursor.execute("CREATE INDEX idx_price_history_token ON price_history(token)")
                            elif idx == 'idx_price_history_timestamp':
                                logger.logger.debug("Creating index: idx_price_history_timestamp")
                                cursor.execute("CREATE INDEX idx_price_history_timestamp ON price_history(timestamp)")
                            elif idx == 'idx_price_history_token_timestamp':
                                logger.logger.debug("Creating index: idx_price_history_token_timestamp")
                                cursor.execute("CREATE INDEX idx_price_history_token_timestamp ON price_history(token, timestamp)")
                        except Exception as idx_error:
                            logger.logger.error(f"Error creating index {idx}: {str(idx_error)}")
                            # Continue with other indices even if one fails
                
                    conn.commit()
                    logger.logger.debug("Created missing indices for price_history table")
                else:
                    logger.logger.debug("All required indices exist for price_history table")
                
                # Query table statistics for diagnostics
                cursor.execute("SELECT COUNT(*) as count FROM price_history")
                row_count = cursor.fetchone()['count']
            
                logger.logger.debug(f"price_history table has {row_count} rows")
            
                if row_count > 0:
                    # Query time range of data
                    cursor.execute("""
                        SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
                        FROM price_history
                    """)
                
                    time_result = cursor.fetchone()
                    if time_result and time_result['min_time'] and time_result['max_time']:
                        min_time = time_result['min_time']
                        max_time = time_result['max_time']
                        logger.logger.debug(f"price_history table has data from {min_time} to {max_time}")
                    
                    # Query distinct tokens
                    cursor.execute("""
                        SELECT token, COUNT(*) as count
                        FROM price_history
                        GROUP BY token
                        ORDER BY count DESC
                    """)
                
                    token_counts = cursor.fetchall()
                    if token_counts:
                        token_info = ", ".join([f"{row['token']}({row['count']})" for row in token_counts[:10]])
                        if len(token_counts) > 10:
                            token_info += f", ... and {len(token_counts)-10} more"
                        logger.logger.debug(f"price_history tokens: {token_info}")
            else:
                logger.logger.info("price_history table does not exist, creating it")
            
                # Create price_history table
                cursor.execute("""
                    CREATE TABLE price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        token TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        price REAL NOT NULL,
                        volume REAL,
                        market_cap REAL,
                        total_supply REAL,
                        circulating_supply REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(token, timestamp)
                    )
                """)
        
                # Create indexes for better query performance
                logger.logger.debug("Creating indices for new price_history table")
                cursor.execute("CREATE INDEX idx_price_history_token ON price_history(token)")
                cursor.execute("CREATE INDEX idx_price_history_timestamp ON price_history(timestamp)")
                cursor.execute("CREATE INDEX idx_price_history_token_timestamp ON price_history(token, timestamp)")
        
                conn.commit()
                logger.logger.info("Successfully created price_history table")
            
            return True
        except Exception as e:
            logger.log_error("Ensure Price History Table Exists", str(e))
            logger.logger.error(f"Error in _ensure_price_history_table_exists: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            conn.rollback()
            return False

    def analyze_price_history_coverage(self, token: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """
        Analyze price history data coverage
    
        Args:
            token: Token symbol (optional, analyzes all tokens if None)
            days: Number of days to analyze (default: 7)
        
        Returns:
            Dictionary with coverage analysis
        """
        conn, cursor = self._get_connection()
    
        try:
            results = {}
        
            # Get list of tokens to analyze
            if token:
                tokens = [token]
            else:
                cursor.execute("""
                    SELECT DISTINCT token FROM price_history
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                """, (days,))
                tokens = [row['token'] for row in cursor.fetchall()]
        
            # Analyze each token
            for t in tokens:
                # Get time range
                cursor.execute("""
                    SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time, COUNT(*) as count
                    FROM price_history
                    WHERE token = ? AND timestamp >= datetime('now', '-' || ? || ' days')
                """, (t, days))
            
                data = cursor.fetchone()
                if not data or data['count'] == 0:
                    results[t] = {"status": "no_data"}
                    continue
            
                # Get all timestamps to analyze gaps
                cursor.execute("""
                    SELECT timestamp
                    FROM price_history
                    WHERE token = ? AND timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                """, (t, days))
            
                timestamps = []
                for row in cursor.fetchall():
                    ts = row['timestamp']
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                try:
                                    ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                                except ValueError:
                                    continue
                    timestamps.append(ts)
            
                # Calculate gaps between timestamps
                gaps = []
                if len(timestamps) > 1:
                    for i in range(1, len(timestamps)):
                        gap_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
                        gaps.append(gap_seconds / 3600)  # Convert to hours
            
                # Calculate coverage for different timeframes
                min_time = self._standardize_timestamp(data['min_time'])
                max_time = self._standardize_timestamp(data['max_time'])
                time_span_hours = ((max_time - min_time).total_seconds() / 3600) if min_time and max_time else 0
            
                coverage = {
                    "token": t,
                    "record_count": data['count'],
                    "time_span_hours": time_span_hours,
                    "time_span_days": time_span_hours / 24,
                    "min_time": min_time,
                    "max_time": max_time,
                    "avg_interval_hours": time_span_hours / (data['count'] - 1) if data['count'] > 1 else None,
                    "gaps": {
                        "count": len(gaps),
                        "min_gap_hours": min(gaps) if gaps else None,
                        "max_gap_hours": max(gaps) if gaps else None,
                        "avg_gap_hours": sum(gaps) / len(gaps) if gaps else None,
                        "gaps_over_1h": sum(1 for g in gaps if g > 1),
                        "gaps_over_6h": sum(1 for g in gaps if g > 6),
                        "gaps_over_24h": sum(1 for g in gaps if g > 24),
                    }
                }
            
                # Add coverage assessment
                if time_span_hours >= 24*7:
                    coverage["7d_coverage"] = "complete"
                elif time_span_hours >= 24:
                    coverage["7d_coverage"] = "partial"
                else:
                    coverage["7d_coverage"] = "insufficient"
                
                if time_span_hours >= 24:
                    coverage["24h_coverage"] = "complete"
                elif time_span_hours >= 1:
                    coverage["24h_coverage"] = "partial"
                else:
                    coverage["24h_coverage"] = "insufficient"
                
                if time_span_hours >= 1:
                    coverage["1h_coverage"] = "complete"
                else:
                    coverage["1h_coverage"] = "insufficient"
            
                results[t] = coverage
        
            return results
        
        except Exception as e:
            logger.log_error("Analyze Price History Coverage", str(e))
            logger.logger.error(f"Error in analyze_price_history_coverage: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def calculate_price_changes(self, token: str, current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate price changes for different periods with enhanced handling of intermittent data
    
        Args:
            token: Token symbol
            current_price: Optional current price (fetches latest if not provided)
    
        Returns:
            Dictionary with price changes for different periods
        """
        try:
            conn, cursor = self._get_connection()
        
            logger.logger.debug(f"calculate_price_changes called for token: {token}, current_price: {current_price}")
        
            # Get current price and timestamp if not provided
            current_time = datetime.now()
            if current_price is None:
                logger.logger.debug(f"No current price provided for {token}, querying database")
                cursor.execute("""
                    SELECT price, timestamp FROM price_history
                    WHERE token = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token,))
            
                result = cursor.fetchone()
                if result:
                    current_price = result["price"]
                    current_time = self._standardize_timestamp(result["timestamp"])
                    logger.logger.debug(f"Found latest price for {token} in database: {current_price} at {current_time}")
                else:
                    logger.logger.warning(f"No price history found for {token}")
                    return {}  # No data available
        
            # Dictionary to store results
            price_changes = {}
        
            # Define time periods to calculate (in hours)
            periods = {
                'price_change_percentage_1h': 1,
                'price_change_percentage_24h': 24,
                'price_change_percentage_7d': 24 * 7,
                'price_change_percentage_30d': 24 * 30,
            }
        
            # Define maximum time difference for each period (more flexibility for longer periods)
            max_time_differences = {
                'price_change_percentage_1h': 0.5,      # 30 minutes for 1h
                'price_change_percentage_24h': 6,       # 6 hours for 24h
                'price_change_percentage_7d': 24,       # 24 hours for 7d
                'price_change_percentage_30d': 48,      # 48 hours for 30d
            }
        
            logger.logger.debug(f"Calculating price changes for {token} over periods: {list(periods.keys())}")
        
            # Calculate change for each period
            for period_name, hours in periods.items():
                try:
                    # Get target time for historical price
                    target_time = current_time - timedelta(hours=hours)
                    logger.logger.debug(f"Target time for {token} {period_name}: {target_time}")
                
                    # Get closest historical price with appropriate flexibility
                    historical_data = self._get_closest_historical_price(
                        token, 
                        target_time, 
                        max_time_difference_hours=max_time_differences.get(period_name)
                    )
                
                    if historical_data and historical_data['price'] > 0:
                        previous_price = historical_data['price']
                        actual_time = historical_data['timestamp']
                        time_diff_hours = historical_data['time_difference_hours']
                    
                        logger.logger.debug(
                            f"Found historical price for {token} for {period_name}: {previous_price} "
                            f"(target: {target_time}, actual: {actual_time}, diff: {time_diff_hours:.2f} hours)"
                        )
                    
                        # Calculate percentage change
                        percent_change = ((current_price / previous_price) - 1) * 100
                        logger.logger.debug(
                            f"Calculated {period_name} for {token}: {percent_change:.2f}% "
                            f"(from {previous_price} to {current_price})"
                        )
                    
                        # Store in results
                        price_changes[period_name] = percent_change
                    
                        # Also add shorter keys for compatibility
                        if period_name == 'price_change_percentage_24h':
                            price_changes['price_change_24h'] = percent_change
                            logger.logger.debug(f"Added compatibility key 'price_change_24h' for {token}: {percent_change:.2f}%")
                        elif period_name == 'price_change_percentage_7d':
                            price_changes['price_change_7d'] = percent_change
                            logger.logger.debug(f"Added compatibility key 'price_change_7d' for {token}: {percent_change:.2f}%")
                    else:
                        logger.logger.warning(f"No valid historical price found for {token} at {target_time} for {period_name}")
                    
                except Exception as period_error:
                    logger.log_error(f"Calculate {period_name} - {token}", str(period_error))
                    logger.logger.error(f"Error calculating {period_name} for {token}: {str(period_error)}")
                    logger.logger.debug(f"Traceback: {traceback.format_exc()}")
        
            # If we didn't calculate any changes, log it clearly
            if not price_changes:
                logger.logger.warning(f"No price changes calculated for {token} - no market change data available for comparison")
            else:
                logger.logger.debug(f"Successfully calculated price changes for {token}: {price_changes}")
        
            return price_changes
        
        except Exception as e:
            logger.log_error(f"Calculate Price Changes - {token}", str(e))
            logger.logger.error(f"Error in calculate_price_changes for {token}: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def add_replied_posts_table(self):
        """Add the replied_posts table if it doesn't exist"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replied_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    reply_content TEXT,
                    replied_at DATETIME NOT NULL,
                    UNIQUE(post_id)
                )
            """)
        
            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_id ON replied_posts(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_url ON replied_posts(post_url)")
        
            conn.commit()
            logger.logger.info("Added replied_posts table to database")
            return True
        except Exception as e:
            logger.log_error("Add Replied Posts Table", str(e))
            conn.rollback()
            return False 

    def store_reply(self, post_id: str, post_url: Optional[str] = None, post_author: Optional[str] = None,
               post_text: Optional[str] = None, reply_text: Optional[str] = None, reply_time: Optional[datetime] = None):
        """
        Store a reply to a post in the database

        Args:
            post_id: The ID of the post being replied to
            post_url: URL of the post (optional)
            post_author: Author of the original post (optional)
            post_text: The content of the original post (optional)
            reply_text: The content of your reply (optional)
            reply_time: Optional timestamp (defaults to current time)

        Returns:
            bool: True if stored successfully, False otherwise
        """
        conn = None
        try:
            if reply_time is None:
                reply_time = datetime.now()

            conn, cursor = self._get_connection()

            # First check if we need to create the table
            self._ensure_replied_posts_table_exists()

            # Store the reply
            cursor.execute("""
                INSERT INTO replied_posts (
                    post_id, post_url, reply_content, replied_at
                ) VALUES (?, ?, ?, ?)
            """, (
                post_id,
                post_url,
                reply_text,
                reply_time
            ))

            conn.commit()
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Store Reply", str(e))
            return False
        
    def store_content_analysis(self, post_id, content=None, analysis_data=None, 
                               reply_worthy=False, reply_score=0.0, features=None,
                               engagement_scores=None, response_focus=None, 
                               author_handle=None, post_url=None, timestamp=None):
        """
        Store content analysis results for a post
    
        Args:
            post_id: Unique identifier for the post
            content: Original post text content (optional)
            analysis_data: Dictionary containing analysis results
            reply_worthy: Whether the post is worth replying to
            reply_score: Score indicating reply priority
            features: Post features extracted during analysis
            engagement_scores: Engagement metrics for the post
            response_focus: Recommended response approach
            author_handle: Twitter handle of the post author (optional)
            post_url: URL to the original post (optional)
            timestamp: Timestamp for the analysis (defaults to current time)
    
        Returns:
            bool: True if successfully stored, False otherwise
        """
        conn, cursor = self._get_connection()
        try:
            # Check if content_analysis table exists, create if it doesn't
            self._ensure_content_analysis_table_exists()
    
            # Prepare timestamp
            if timestamp is None:
                timestamp = datetime.now()
    
            # Build analysis data if not provided
            if analysis_data is None:
                analysis_data = {
                    "reply_worthy": reply_worthy,
                    "reply_score": reply_score,
                    "features": features,
                    "engagement_scores": engagement_scores,
                    "response_focus": response_focus
                }
    
            # Convert analysis_data to JSON string
            analysis_json = json.dumps(analysis_data)
    
            cursor.execute("""
                INSERT INTO content_analysis (
                    post_id, content, analysis_data, author_handle,
                    post_url, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                content,
                analysis_json,
                author_handle,
                post_url,
                timestamp
            ))
    
            conn.commit()
            return True
    
        except Exception as e:
            logger.log_error("Store Content Analysis", str(e))
            if conn:
                conn.rollback()
            return False
        
    def add_sparkline_column(self):
        """
        Add sparkline column to price_history table for storing historical price arrays
        This allows us to store and retrieve sparkline data for all tokens and timeframes
        """
        conn, cursor = self._get_connection()
        try:
            # Check if sparkline column already exists
            cursor.execute("PRAGMA table_info(price_history)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'sparkline_data' not in columns:
                # Add the sparkline_data column as TEXT to store JSON arrays
                cursor.execute("ALTER TABLE price_history ADD COLUMN sparkline_data TEXT")
                logger.logger.info("Added sparkline_data column to price_history table")
                
                # Add index for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_sparkline ON price_history(token, timestamp) WHERE sparkline_data IS NOT NULL")
                
                conn.commit()
                return True
            else:
                logger.logger.debug("sparkline_data column already exists in price_history table")
                return True
                
        except Exception as e:
            logger.log_error("Add Sparkline Column", str(e))
            conn.rollback()
            return False

    def add_sparkline_table(self):
        """
        Create sparkline_data table for storing historical price arrays in proper SQL format
        Each sparkline point gets its own row with timestamp and sequence
        """
        conn, cursor = self._get_connection()
        try:
            # Check if table exists first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sparkline_data'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Create sparkline_data table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE sparkline_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        token TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        sequence_number INTEGER NOT NULL,
                        price REAL NOT NULL,
                        data_timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(token, timeframe, sequence_number, data_timestamp)
                    )
                """)
                logger.logger.info("Created sparkline_data table")
            else:
                logger.logger.debug("sparkline_data table already exists")
            
            # Create indexes for better query performance (these are safe to run multiple times)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sparkline_token_timeframe ON sparkline_data(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sparkline_data_timestamp ON sparkline_data(data_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sparkline_token_timeframe_timestamp ON sparkline_data(token, timeframe, data_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sparkline_sequence ON sparkline_data(token, timeframe, sequence_number)")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.log_error("Add Sparkline Table", str(e))
            conn.rollback()
            return False

    def store_sparkline_data(self, token: str, sparkline_array: List[float], timeframe: str = "7d", 
                            timestamp: Optional[datetime] = None) -> bool:
        """
        Store sparkline data array for a token with timeframe
        
        Args:
            token: Token symbol
            sparkline_array: Array of price points
            timeframe: Timeframe for this sparkline ("1h", "24h", "7d")
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            bool: True if stored successfully
        """
        conn = None
        try:
            if not sparkline_array or not isinstance(sparkline_array, list):
                logger.logger.warning(f"Invalid sparkline data for {token}: {type(sparkline_array)}")
                return False
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # Standardize timestamp
            std_timestamp = self._standardize_timestamp(timestamp)
            
            conn, cursor = self._get_connection()
            
            # Ensure table and column exist
            self._ensure_price_history_table_exists()
            self.add_sparkline_column()
            
            # Store sparkline data as JSON
            sparkline_json = json.dumps(sparkline_array)
            
            # Insert or update with sparkline data
            # Use current price from sparkline array (last price)
            current_price = float(sparkline_array[-1]) if sparkline_array else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO price_history (
                    token, timestamp, price, sparkline_data
                ) VALUES (?, ?, ?, ?)
            """, (
                token,
                std_timestamp,
                current_price,
                sparkline_json
            ))
            
            conn.commit()
            logger.logger.debug(f"Stored sparkline data for {token} ({len(sparkline_array)} points, {timeframe})")
            return True
            
        except Exception as e:
            logger.log_error(f"Store Sparkline Data - {token}", str(e))
            if conn:
                conn.rollback()
            return False

    def get_sparkline_data(self, token: str, hours: int = 168, timeframe: str = "7d") -> Optional[List[float]]:
        """
        Retrieve sparkline data for a token
        
        Args:
            token: Token symbol
            hours: Number of hours to look back (default: 168 = 7 days)
            timeframe: Requested timeframe
        
        Returns:
            List of price points or None if not available
        """
        try:
            conn, cursor = self._get_connection()
            
            # Get the most recent sparkline data for this token
            cursor.execute("""
                SELECT sparkline_data, timestamp 
                FROM price_history 
                WHERE token = ? 
                AND sparkline_data IS NOT NULL
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (token, hours))
            
            result = cursor.fetchone()
            
            if result and result['sparkline_data']:
                try:
                    sparkline_array = json.loads(result['sparkline_data'])
                    if isinstance(sparkline_array, list) and len(sparkline_array) > 0:
                        logger.logger.debug(f"Retrieved sparkline data for {token}: {len(sparkline_array)} points")
                        return sparkline_array
                except json.JSONDecodeError as json_error:
                    logger.logger.warning(f"Failed to parse sparkline JSON for {token}: {str(json_error)}")
            
            logger.logger.debug(f"No sparkline data found for {token} in last {hours} hours")
            return None
            
        except Exception as e:
            logger.log_error(f"Get Sparkline Data - {token}", str(e))
            return None

    def build_sparkline_from_price_history(self, token: str, hours: int = 168) -> List[float]:
        """
        Build sparkline array from individual price history records
        Fallback when no stored sparkline data is available
        
        Args:
            token: Token symbol
            hours: Number of hours to look back (default 168 = 7 days)
        
        Returns:
            List of price points built from price history
        """
        try:
            conn, cursor = self._get_connection()
            
            # Get individual price records to build sparkline
            cursor.execute("""
                SELECT price, timestamp 
                FROM price_history 
                WHERE token = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """, (token, hours))
            
            results = cursor.fetchall()
            
            if not results:
                logger.logger.debug(f"No price history found for {token} to build sparkline")
                return []
            
            # Extract price array
            price_array = [float(row['price']) for row in results]
            
            logger.logger.debug(f"Built sparkline from price history for {token}: {len(price_array)} points")
            return price_array
            
        except Exception as e:
            logger.log_error(f"Build Sparkline from Price History - {token}", str(e))
            return []
  
    def force_database_supplementation(self, result, timeframe, token_id_map):
        """
        Supplement market data with database sparkline data
        
        Args:
            result: Market data dictionary
            timeframe: Timeframe for data lookup  
            token_id_map: Mapping of token IDs
        
        Returns:
            Enhanced result dictionary with database sparklines
        """
        
        hours = {"1h": 48, "24h": 168, "7d": 720}.get(timeframe, 168)
        print(f"🔧 Force supplementing {len(result)} tokens from database")
        
        # Import and create database connection if needed
        try:
            # Check if we already have a database connection
            if not hasattr(self, 'database') or self.database is None:
                # Use the database from config if available
                try:
                    from config import config
                    self.database = config.db
                except ImportError:
                    from database import CryptoDatabase
                    self.database = CryptoDatabase()
            
            db_instance = self.database
            
        except ImportError as e:
            print(f"❌ Cannot import database: {e}")
            return result
        
        for token_symbol in result.keys():
            if isinstance(token_symbol, str) and token_symbol.isupper():
                try:
                    # Get price array from database (returns List[float])
                    price_array = db_instance.build_sparkline_from_price_history(token_symbol, hours=hours)
                    
                    # Check if we got valid data
                    if price_array and len(price_array) > 0:
                        result[token_symbol]['sparkline'] = price_array
                        result[token_symbol]['_sparkline_source'] = 'database'
                        result[token_symbol]['_sparkline_points'] = len(price_array)
                        print(f"✅ {token_symbol}: {len(price_array)} points from database")
                    else:
                        print(f"❌ {token_symbol}: No database data")
                        # Ensure sparkline key exists even if empty
                        if 'sparkline' not in result[token_symbol]:
                            result[token_symbol]['sparkline'] = []
                            
                except Exception as e:
                    print(f"❌ {token_symbol}: Error - {e}")
                    # Ensure sparkline key exists even on error
                    if 'sparkline' not in result[token_symbol]:
                        result[token_symbol]['sparkline'] = []
        
        return result

    def cleanup_old_sparkline_data(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old sparkline data to manage database size
        
        Args:
            days_to_keep: Number of days of sparkline data to keep
        
        Returns:
            bool: Success indicator
        """
        try:
            conn, cursor = self._get_connection()
            
            # Remove old sparkline data (set to NULL, keep price records)
            cursor.execute("""
                UPDATE price_history 
                SET sparkline_data = NULL 
                WHERE sparkline_data IS NOT NULL 
                AND timestamp < datetime('now', '-' || ? || ' days')
            """, (days_to_keep,))
            
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                logger.logger.info(f"Cleaned up sparkline data from {rows_updated} old records (kept last {days_to_keep} days)")
            
            return True
            
        except Exception as e:
            logger.log_error("Cleanup Old Sparkline Data", str(e))
            return False    

    def _ensure_content_analysis_table_exists(self) -> None:
        """Ensure the content_analysis table exists in the database"""
        conn, cursor = self._get_connection()
        try:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analysis'")
            if cursor.fetchone() is None:
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE content_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        post_id TEXT NOT NULL,
                        content TEXT,
                        analysis_data TEXT NOT NULL,
                        author_handle TEXT,
                        post_url TEXT,
                        timestamp DATETIME NOT NULL,
                        UNIQUE(post_id)
                    )
                """)
            
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_post_id ON content_analysis(post_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_author ON content_analysis(author_handle)")
            
                conn.commit()
                logger.logger.info("Created content_analysis table in database")
        
        except Exception as e:
            logger.log_error("Ensure Content Analysis Table", str(e))
            if conn:
                conn.rollback()    

    def _ensure_replied_posts_table_exists(self):
        """Ensure the replied_posts table exists in the database"""
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replied_posts'")
            if cursor.fetchone() is None:
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE replied_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        post_id TEXT NOT NULL,
                        original_content TEXT,
                        reply_content TEXT,
                        UNIQUE(post_id)
                    )
                """)
                
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Ensure Replied Posts Table", str(e))

    def mark_post_as_replied(self, post_id: str, post_url: Optional[str] = None, reply_content: Optional[str] = None) -> bool:
        """
        Mark a post as replied to

        Args:
            post_id: The ID of the post
            post_url: The URL of the post (optional)
            reply_content: The content of the reply (optional)

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
        
            # First check if this post has already been marked as replied
            cursor.execute("""
                SELECT COUNT(*) FROM replied_posts WHERE post_id = ?
            """, (post_id,))
        
            # If it's already in the database, we can consider this a success
            if cursor.fetchone()[0] > 0:
                logger.logger.info(f"Post {post_id} was already marked as replied to")
                return True
        
            # If not already in the database, add it
            cursor.execute("""
                INSERT INTO replied_posts (post_id, post_url, reply_content, replied_at)
                VALUES (?, ?, ?, ?)
            """, (post_id, post_url, reply_content, datetime.now()))
        
            conn.commit()
            return True
        except Exception as e:
            logger.log_error("Mark Post As Replied", str(e))
            if conn:
                conn.rollback()
            return False
        
    def _ensure_tech_columns_exist(self) -> None:
        """Ensure tech-related columns exist in the posted_content table"""
        conn, cursor = self._get_connection()
        try:
            # Check if columns exist
            cursor.execute("PRAGMA table_info(posted_content)")
            columns = [column[1] for column in cursor.fetchall()]
        
            # Add tech_category column if it doesn't exist
            if 'tech_category' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN tech_category TEXT")
                logger.logger.info("Added missing tech_category column to posted_content table")
        
            # Add tech_metadata column if it doesn't exist
            if 'tech_metadata' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN tech_metadata TEXT")
                logger.logger.info("Added missing tech_metadata column to posted_content table")
            
                # Initialize tech_metadata as empty JSON for existing rows
                cursor.execute("UPDATE posted_content SET tech_metadata = '{}' WHERE tech_metadata IS NULL")
                logger.logger.info("Initialized tech_metadata as empty JSON for existing rows")
        
            # Add is_educational column if it doesn't exist
            if 'is_educational' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN is_educational BOOLEAN DEFAULT 0")
                logger.logger.info("Added missing is_educational column to posted_content table")
        
            # We'll ensure these operations are committed even if later operations fail
            conn.commit()
        
            # Backward compatibility: If there are existing rows with market_context or vs_market_change
            # in other columns, we could migrate them, but that's typically handled in a separate migration method
        
            conn.commit()
        except Exception as e:
            logger.log_error("Ensure Tech Columns Exist", str(e))
            conn.rollback()    

    def add_missing_columns(self):
        """Add missing columns to technical_indicators table if they don't exist"""
        conn, cursor = self._get_connection()
        changes_made = False
    
        try:
            # Check if columns exist
            cursor.execute("PRAGMA table_info(technical_indicators)")
            columns = [column[1] for column in cursor.fetchall()]
    
            # Add the ichimoku_data column if it doesn't exist
            if 'ichimoku_data' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN ichimoku_data TEXT")
                logger.logger.info("Added missing ichimoku_data column to technical_indicators table")
                changes_made = True
            
            # Add the pivot_points column if it doesn't exist
            if 'pivot_points' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN pivot_points TEXT")
                logger.logger.info("Added missing pivot_points column to technical_indicators table")
                changes_made = True
            
            conn.commit()
            return changes_made
        except Exception as e:
            logger.log_error("Add Missing Columns", str(e))
            conn.rollback()
            return False
    
    def add_ichimoku_column(self):
        """Add the missing ichimoku_data column to technical_indicators table if it doesn't exist"""
        conn, cursor = self._get_connection()
        try:
            # Check if column exists
            cursor.execute("PRAGMA table_info(technical_indicators)")
            columns = [column[1] for column in cursor.fetchall()]
        
            # Add the column if it doesn't exist
            if 'ichimoku_data' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN ichimoku_data TEXT")
                conn.commit()
                logger.logger.info("Added missing ichimoku_data column to technical_indicators table")
                return True
            return False
        except Exception as e:
            logger.log_error("Add Ichimoku Column", str(e))
            conn.rollback()
            return False    

    @property
    def conn(self):
        """Thread-safe connection property - returns the connection for current thread"""
        conn, _ = self._get_connection()
        return conn
        
    @property
    def cursor(self):
        """Thread-safe cursor property - returns the cursor for current thread"""
        _, cursor = self._get_connection()
        return cursor

    def _get_connection(self):
        """Get database connection, creating it if necessary - thread-safe version"""
        # Check if this thread has a connection
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            # Create a new connection for this thread
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.row_factory = sqlite3.Row
            self.local.cursor = self.local.conn.cursor()
        
        return self.local.conn, self.local.cursor

    def _initialize_database(self):
        """Create necessary tables if they don't exist"""
        conn, cursor = self._get_connection()
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    obv REAL,
                    adx REAL,
                    ichimoku_data TEXT,
                    pivot_points TEXT,
                    overall_trend TEXT,
                    trend_strength REAL,
                    volatility REAL,
                    raw_data JSON
                )
            """)

        try:
            # Market Data Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    price_change_24h REAL,
                    market_cap REAL,
                    ath REAL,
                    ath_change_percentage REAL
                )
            """)

            # Posted Content Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posted_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    content TEXT NOT NULL,
                    sentiment JSON NOT NULL,
                    trigger_type TEXT NOT NULL,
                    price_data JSON NOT NULL,
                    meme_phrases JSON NOT NULL,
                    is_prediction BOOLEAN DEFAULT 0,
                    prediction_data JSON,
                    timeframe TEXT DEFAULT '1h'
                )
            """)

            # Chain Mood History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    indicators JSON NOT NULL
                )
            """)
        
            # Smart Money Indicators Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_money_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    volume_z_score REAL,
                    price_volume_divergence BOOLEAN,
                    stealth_accumulation BOOLEAN,
                    abnormal_volume BOOLEAN,
                    volume_vs_hourly_avg REAL,
                    volume_vs_daily_avg REAL,
                    volume_cluster_detected BOOLEAN,
                    unusual_trading_hours JSON,
                    raw_data JSON
                )
            """)
        
            # Token Market Comparison Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_market_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    vs_market_avg_change REAL,
                    vs_market_volume_growth REAL,
                    outperforming_market BOOLEAN,
                    correlations JSON
                )
            """)
        
            # Token Correlations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    avg_price_correlation REAL NOT NULL,
                    avg_volume_correlation REAL NOT NULL,
                    full_data JSON NOT NULL
                )
            """)

            # Token Equity Tracking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_equity_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    data_type TEXT NOT NULL DEFAULT 'token_equity',
                    data JSON NOT NULL
                )
            """)
        
            # Generic JSON Data Table for flexible storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generic_json_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    data_type TEXT NOT NULL,
                    data JSON NOT NULL
                )
            """)
        
            # PREDICTION TABLES
        
            # Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    prediction_rationale TEXT,
                    method_weights JSON,
                    model_inputs JSON,
                    technical_signals JSON,
                    expiration_time DATETIME NOT NULL
                )
            """)

            # Prediction Outcomes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    actual_outcome REAL NOT NULL,
                    accuracy_percentage REAL NOT NULL,
                    was_correct BOOLEAN NOT NULL,
                    evaluation_time DATETIME NOT NULL,
                    deviation_from_prediction REAL NOT NULL,
                    market_conditions JSON,
                    FOREIGN KEY (prediction_id) REFERENCES price_predictions(id)
                )
            """)

            # Prediction Performance Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    avg_deviation REAL NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """)
        
            # REMOVED THE DUPLICATE technical_indicators TABLE CREATION HERE
        
            # Statistical Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistical_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    model_parameters JSON,
                    input_data_summary JSON
                )
            """)
        
            # Machine Learning Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    feature_importance JSON,
                    model_parameters JSON
                )
            """)
        
            # Claude AI Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claude_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    claude_model TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL,
                    sentiment TEXT,
                    rationale TEXT,
                    key_factors JSON,
                    input_data JSON
                )
            """)
        
            # Timeframe Metrics Table - New table to track metrics by timeframe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeframe_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    avg_accuracy REAL,
                    total_count INTEGER,
                    correct_count INTEGER,
                    model_weights JSON,
                    best_model TEXT,
                    last_updated DATETIME NOT NULL
                )
            """)
        
            # Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_chain ON market_data(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timestamp ON posted_content(timestamp)")
        
            # HERE'S THE FIX: Check if timeframe column exists in posted_content before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(posted_content)")
                columns = [column[1] for column in cursor.fetchall()]
            
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE posted_content ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to posted_content table")
            
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timeframe ON posted_content(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check", str(e))
        
            # Check if timeframe column exists in technical_indicators before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(technical_indicators)")
                columns = [column[1] for column in cursor.fetchall()]
            
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE technical_indicators ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to technical_indicators table")
            
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timeframe ON technical_indicators(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check for technical_indicators", str(e))
        
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_timestamp ON mood_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_chain ON mood_history(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_timestamp ON smart_money_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_chain ON smart_money_indicators(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_timestamp ON generic_json_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_type ON generic_json_data(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_timestamp ON token_market_comparison(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_token ON token_market_comparison(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_timestamp ON token_correlations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_token ON token_correlations(token)")
        
            # Prediction indices - Enhanced for timeframe support
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token ON price_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timeframe ON price_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timestamp ON price_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_expiration ON price_predictions(expiration_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token_timeframe ON price_predictions(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_prediction_id ON prediction_outcomes(prediction_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token ON prediction_performance(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_timeframe ON prediction_performance(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token_timeframe ON prediction_performance(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_token ON technical_indicators(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timestamp ON technical_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timeframe ON technical_indicators(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_token ON statistical_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_timeframe ON statistical_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_token ON ml_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_timeframe ON ml_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_token ON claude_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_timeframe ON claude_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_token ON timeframe_metrics(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_timeframe ON timeframe_metrics(timeframe)")

            conn.commit()
            logger.logger.info("Database initialized successfully")
    
        except Exception as e:
            logger.log_error("Database Initialization", str(e))
            raise

    #########################
    # CORE DATA STORAGE METHODS
    #########################
    def get_tech_content(self, tech_category: Optional[str] = None, hours: int = 24, limit: int = 10) -> List[Dict]:
        """
        Get recent tech content posts
        Can filter by tech category
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT * FROM posted_content 
                WHERE tech_category IS NOT NULL
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params: List[Union[int, str]] = [hours]
        
            if tech_category:
                query += " AND tech_category = ?"
                params.append(tech_category)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
        
            cursor.execute(query, params)
        
            results = [dict(row) for row in cursor.fetchall()]
        
            # Parse JSON fields
            for result in results:
                result["sentiment"] = json.loads(result["sentiment"]) if result["sentiment"] else {}
                result["price_data"] = json.loads(result["price_data"]) if result["price_data"] else {}
                result["meme_phrases"] = json.loads(result["meme_phrases"]) if result["meme_phrases"] else {}
                result["prediction_data"] = json.loads(result["prediction_data"]) if result["prediction_data"] else None
                result["tech_metadata"] = json.loads(result["tech_metadata"]) if result["tech_metadata"] else None
            
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Get Tech Content", str(e))
            return []        
    
    def calculate_market_comparison_data(self, token: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate market comparison data for a specific token with robust token identifier handling
        Compares token performance against the overall market with enhanced error handling
        
        Args:
            token: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            market_data: Dictionary with market data for all tokens (supports both symbol and ID keys)
        
        Returns:
            Dictionary with comprehensive comparison metrics
        """
        try:
            # Define symbol to CoinGecko ID mapping
            symbol_to_coingecko = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'XRP': 'ripple',
                'BNB': 'binancecoin', 'AVAX': 'avalanche-2', 'DOT': 'polkadot', 'UNI': 'uniswap',
                'NEAR': 'near', 'AAVE': 'aave', 'FIL': 'filecoin', 'POL': 'matic-network',
                'TRUMP': 'official-trump', 'KAITO': 'kaito'
            }
            
            # Validate parameters
            if not market_data or not isinstance(market_data, dict):
                logger.logger.warning(f"Invalid market data for {token}: {type(market_data)}")
                return {"error": "Invalid market data"}
                
            # Try to get token data - check both symbol and CoinGecko ID
            token_data = market_data.get(token)
            if not token_data:
                coingecko_id = symbol_to_coingecko.get(token.upper())
                if coingecko_id:
                    token_data = market_data.get(coingecko_id)
                    
            if not token_data or not isinstance(token_data, dict):
                logger.logger.warning(f"Token {token} not found in market data keys: {list(market_data.keys())}")
                return {"error": f"Token {token} not found in market data"}
                
            logger.logger.debug(f"calculate_market_comparison_data called for token: {token}")
            
            # Initialize variables
            token_changes = {}
            market_changes = {}
            
            # Get token's 24h price change
            token_price_change_24h = None
            
            # Try different possible keys for 24h price change
            change_keys = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
            for change_key in change_keys:
                if change_key in token_data and token_data[change_key] is not None:
                    try:
                        token_price_change_24h = float(token_data[change_key])
                        logger.logger.debug(f"Found existing {change_key} for {token}: {token_price_change_24h}")
                        break
                    except (ValueError, TypeError):
                        continue
                        
            # If no existing change found, calculate it
            if token_price_change_24h is None:
                token_price = token_data.get('current_price') or token_data.get('price')
                if token_price is not None:
                    try:
                        token_price = float(token_price)
                        if token_price > 0:
                            calc_changes = self.calculate_price_changes(token, token_price)
                            if calc_changes and 'price_change_percentage_24h' in calc_changes:
                                token_price_change_24h = calc_changes['price_change_percentage_24h']
                                logger.logger.debug(f"Calculated 24h change for {token}: {token_price_change_24h}")
                    except (ValueError, TypeError):
                        pass
                        
            if token_price_change_24h is None:
                token_price_change_24h = 0.0
                logger.logger.warning(f"Could not determine 24h price change for {token}, using 0.0")
                
            token_changes['24h'] = token_price_change_24h
            
            # Calculate market average 24h change
            sum_24h_change = 0
            valid_tokens = 0
            
            for other_token, other_data in market_data.items():
                if other_token == token or not isinstance(other_data, dict):
                    continue
                    
                other_change = None
                
                # Try to get existing price change for other token
                for change_key in change_keys:
                    if change_key in other_data and other_data[change_key] is not None:
                        try:
                            other_change = float(other_data[change_key])
                            logger.logger.debug(f"Found existing {change_key} for {other_token}: {other_change}")
                            break
                        except (ValueError, TypeError):
                            continue
                            
                # If no existing change, try to calculate it
                if other_change is None:
                    other_price = other_data.get('current_price') or other_data.get('price')
                    if other_price is not None:
                        try:
                            other_price = float(other_price)
                            if other_price > 0:
                                calc_changes = self.calculate_price_changes(other_token, other_price)
                                logger.logger.debug(f"Calculated price changes for {other_token}: {calc_changes}")
                                
                                if calc_changes and 'price_change_percentage_24h' in calc_changes:
                                    other_change = calc_changes['price_change_percentage_24h']
                                    logger.logger.debug(f"Using calculated price change for {other_token}: {other_change}")
                        except (ValueError, TypeError):
                            pass
                            
                # Add to market average if we have valid data
                if other_change is not None:
                    sum_24h_change += other_change
                    valid_tokens += 1
                    logger.logger.debug(f"Added {other_token} to market average with change: {other_change}")
                    
            # Calculate market average
            market_avg_24h_change = sum_24h_change / valid_tokens if valid_tokens > 0 else 0
            logger.logger.debug(f"Market average 24h change: {market_avg_24h_change} (from {valid_tokens} tokens)")
            market_changes['24h'] = market_avg_24h_change
            
            # Calculate comparison metrics
            vs_market_avg_change = token_changes['24h'] - market_changes['24h']
            logger.logger.debug(f"{token} vs market change: {vs_market_avg_change} ({token_changes['24h']} vs {market_changes['24h']})")
            
            outperforming_market = token_changes['24h'] > market_changes['24h']
            logger.logger.debug(f"{token} outperforming market: {outperforming_market}")
            
            # Calculate volume comparison if data available
            token_volume = token_data.get('volume', 0) or token_data.get('total_volume', 0)
            logger.logger.debug(f"{token} volume: {token_volume}")
            
            market_volume_sum = 0
            market_volume_tokens = 0
            
            for other_token, other_data in market_data.items():
                if other_token == token or not isinstance(other_data, dict):
                    continue
                    
                other_volume = other_data.get('volume', None) or other_data.get('total_volume', None)
                if other_volume is not None and other_volume > 0:
                    market_volume_sum += other_volume
                    market_volume_tokens += 1
                    
            market_avg_volume = market_volume_sum / market_volume_tokens if market_volume_tokens > 0 else 0
            logger.logger.debug(f"Market average volume: {market_avg_volume} (from {market_volume_tokens} tokens)")
            
            vs_market_volume_ratio = token_volume / market_avg_volume if market_avg_volume > 0 else 1
            logger.logger.debug(f"{token} vs market volume ratio: {vs_market_volume_ratio}")
            
            # For volume growth, we need historical data
            # For now, just set a neutral value
            vs_market_volume_growth = 0
            logger.logger.debug(f"{token} vs market volume growth: {vs_market_volume_growth} (default value)")
            
            # Calculate correlations with other tokens
            correlations = {}
            
            # Top tokens to calculate correlations with
            top_tokens = ["BTC", "ETH"]
            logger.logger.debug(f"Calculating correlations with top tokens: {top_tokens}")
            
            # Add any other tokens in the market data, up to a limit
            other_tokens = [t for t in market_data.keys() if t != token and t not in top_tokens]
            top_tokens.extend(other_tokens[:3])  # Limit to 3 additional tokens
            logger.logger.debug(f"Extended correlation token list: {top_tokens}")
            
            for other_token in top_tokens:
                if other_token == token:
                    continue
                    
                # Try to get other token data with robust lookup
                other_data = market_data.get(other_token)
                if not other_data:
                    coingecko_id = symbol_to_coingecko.get(other_token.upper())
                    if coingecko_id:
                        other_data = market_data.get(coingecko_id)
                        
                if not other_data or not isinstance(other_data, dict):
                    continue
                    
                # Calculate simple price correlation
                # This is a very basic correlation - in a real implementation, 
                # you would use historical price data for a proper correlation
                other_price_change = None
                
                # Try to get 24h change for other token
                for change_key in change_keys:
                    if change_key in other_data and other_data[change_key] is not None:
                        try:
                            other_price_change = float(other_data[change_key])
                            logger.logger.debug(f"Found existing {change_key} for {other_token}: {other_price_change}")
                            break
                        except (ValueError, TypeError):
                            continue
                            
                if other_price_change is None:
                    other_price = other_data.get('current_price') or other_data.get('price')
                    if other_price is not None:
                        try:
                            other_price = float(other_price)
                            if other_price > 0:
                                calc_changes = self.calculate_price_changes(other_token, other_price)
                                if calc_changes and 'price_change_percentage_24h' in calc_changes:
                                    other_price_change = calc_changes['price_change_percentage_24h']
                        except (ValueError, TypeError):
                            pass
                            
                if other_price_change is not None:
                    # Simple direction correlation
                    if (token_changes['24h'] > 0 and other_price_change > 0) or \
                    (token_changes['24h'] < 0 and other_price_change < 0):
                        direction_correlation = 1.0  # Same direction
                        logger.logger.debug(f"Positive correlation between {token} and {other_token}")
                    else:
                        direction_correlation = -1.0  # Opposite directions
                        logger.logger.debug(f"Negative correlation between {token} and {other_token}")
                        
                    # Add to correlations
                    correlations[other_token] = {
                        "price_change_correlation": direction_correlation,
                        "other_token_change": other_price_change
                    }
                    
            # Prepare result
            result = {
                "token": token,
                "token_change_24h": token_changes['24h'],
                "market_avg_change_24h": market_changes['24h'],
                "vs_market_avg_change": vs_market_avg_change,
                "outperforming_market": outperforming_market,
                "vs_market_volume_ratio": vs_market_volume_ratio,
                "vs_market_volume_growth": vs_market_volume_growth,
                "correlations": correlations
            }
            
            # Store the result in database for future reference
            try:
                logger.logger.debug(f"Storing market comparison data for {token} in database")
                self.store_token_market_comparison(
                    token=token,
                    vs_market_avg_change=vs_market_avg_change,
                    vs_market_volume_growth=vs_market_volume_growth,
                    outperforming_market=outperforming_market,
                    correlations=correlations
                )
                logger.logger.debug(f"Successfully stored market comparison data for {token}")
            except Exception as store_error:
                logger.logger.error(f"Error storing market comparison data for {token}: {str(store_error)}")
                
            return result
            
        except Exception as e:
            logger.log_error(f"Calculate Market Comparison - {token}", str(e))
            logger.logger.error(f"Error in calculate_market_comparison_data for {token}: {str(e)}")
            return {"error": str(e)}
    
    def store_market_data(self, chain: str, data: Dict[str, Any]) -> None:
        """
        Store market data for a specific chain
        Enhanced to also store in price_history table for change calculations
        NOW SUPPORTS: Market chart arrays, sparkline data, and standard market data

        Args:
            chain: Token symbol
            data: Market data dictionary - supports multiple CoinGecko endpoint formats:
                - Standard market data (existing functionality)
                - Market chart data with 'prices', 'volumes', 'market_caps' arrays
                - Individual coin data with 'market_data' and 'sparkline_in_7d'
        """
        conn, cursor = self._get_connection()
        try:
            current_time = datetime.now()

            # Helper function to safely extract numeric value with fallback
            def safe_extract(data_dict, key, default=0):
                try:
                    value = data_dict.get(key, default)
                    if value is None:
                        return default
                    if isinstance(value, (int, float)):
                        return value
                    return float(value)  # Try to convert strings or other types
                except (ValueError, TypeError):
                    return default

            # ================================================================
            # 🆕 NEW: DETECT AND HANDLE MARKET CHART DATA FORMAT
            # ================================================================
            if 'prices' in data and isinstance(data['prices'], list) and len(data['prices']) > 0:
                logger.logger.debug(f"Detected market chart data format for {chain}")
                
                # Market chart format: {'prices': [[timestamp, price], ...], 'volumes': [[timestamp, volume], ...]}
                prices_array = data['prices']
                volumes_array = data.get('volumes', [])
                market_caps_array = data.get('market_caps', [])
                
                # Store each historical point
                stored_points = 0
                for i, price_point in enumerate(prices_array):
                    try:
                        if len(price_point) >= 2:
                            timestamp_ms = price_point[0]
                            price_value = price_point[1]
                            
                            # Convert timestamp from milliseconds to datetime
                            point_timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                            
                            # Get corresponding volume if available
                            volume_value = None
                            if i < len(volumes_array) and len(volumes_array[i]) >= 2:
                                volume_value = volumes_array[i][1]
                            
                            # Get corresponding market cap if available
                            market_cap_value = None
                            if i < len(market_caps_array) and len(market_caps_array[i]) >= 2:
                                market_cap_value = market_caps_array[i][1]
                            
                            # Store this historical point
                            if self.store_price_history(
                                token=chain,
                                price=float(price_value),
                                volume=float(volume_value) if volume_value else None,
                                market_cap=float(market_cap_value) if market_cap_value else None,
                                timestamp=point_timestamp
                            ):
                                stored_points += 1
                                
                    except Exception as point_error:
                        logger.logger.warning(f"Error storing historical point {i} for {chain}: {str(point_error)}")
                        continue
                
                logger.logger.info(f"Stored {stored_points} historical data points for {chain} from market chart")
                return  # Market chart data handled, exit method

            # ================================================================
            # 🆕 NEW: DETECT AND HANDLE INDIVIDUAL COIN DATA WITH SPARKLINE
            # ================================================================
            if 'market_data' in data and 'sparkline_in_7d' in data:
                logger.logger.debug(f"Detected individual coin data format for {chain}")
                
                # Extract current market data
                market_data_section = data['market_data']
                current_price = safe_extract(market_data_section.get('current_price', {}), 'usd')
                volume_24h = safe_extract(market_data_section.get('total_volume', {}), 'usd')
                market_cap = safe_extract(market_data_section.get('market_cap', {}), 'usd')
                
                # Store current snapshot in original market_data table
                if current_price > 0:
                    cursor.execute("""
                        INSERT INTO market_data (
                            timestamp, chain, price, volume, price_change_24h, 
                            market_cap, ath, ath_change_percentage
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        current_time, chain, current_price, volume_24h, 0, 
                        market_cap, 0, 0
                    ))
                
                # Process sparkline data
                sparkline_data = data.get('sparkline_in_7d', {})
                if isinstance(sparkline_data, dict) and 'price' in sparkline_data:
                    price_array = sparkline_data['price']
                    if isinstance(price_array, list) and len(price_array) > 0:
                        
                        # Calculate time intervals (assuming 7 days / number of points)
                        total_hours = 7 * 24  # 7 days in hours
                        interval_hours = total_hours / len(price_array)
                        
                        # Store each sparkline point as historical data
                        stored_sparkline = 0
                        for i, price_val in enumerate(price_array):
                            try:
                                # Calculate timestamp for this point (working backwards from now)
                                point_timestamp = current_time - timedelta(hours=(len(price_array) - i - 1) * interval_hours)
                                
                                if self.store_price_history(
                                    token=chain,
                                    price=float(price_val),
                                    timestamp=point_timestamp
                                ):
                                    stored_sparkline += 1
                                    
                            except Exception as sparkline_error:
                                logger.logger.warning(f"Error storing sparkline point {i} for {chain}: {str(sparkline_error)}")
                                continue
                        
                        logger.logger.info(f"Stored {stored_sparkline} sparkline data points for {chain}")
                
                # Store current price in price_history table
                if current_price > 0:
                    self.store_price_history(
                        token=chain,
                        price=current_price,
                        volume=volume_24h,
                        market_cap=market_cap
                    )
                
                conn.commit()
                logger.logger.debug(f"Stored individual coin data for {chain}")
                return  # Individual coin data handled, exit method

            # ================================================================
            # ✅ EXISTING: STANDARD MARKET DATA FORMAT (UNCHANGED)
            # ================================================================
            
            # Extract data with safety checks and fallbacks
            current_price = safe_extract(data, 'current_price')
            volume = safe_extract(data, 'volume')

            # Try multiple keys for price change
            price_change_24h = None
            for change_key in ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']:
                if change_key in data:
                    price_change_24h = safe_extract(data, change_key)
                    if price_change_24h != 0:  # Found a non-zero value
                        break

            # If no value found, use calculated value or default
            if price_change_24h is None:
                # Try to calculate from price history
                price_changes = self.calculate_price_changes(chain, current_price)
                price_change_24h = price_changes.get('price_change_percentage_24h', 0)

            # Extract other fields with fallbacks
            market_cap = safe_extract(data, 'market_cap')
            ath = safe_extract(data, 'ath')
            ath_change_percentage = safe_extract(data, 'ath_change_percentage')
            total_supply = safe_extract(data, 'total_supply')
            circulating_supply = safe_extract(data, 'circulating_supply')

            # Handle additional fields that might be in the data
            # Convert any datetime objects to ISO format
            additional_data = {}
            for key, value in data.items():
                if key not in ['current_price', 'volume', 'price_change_percentage_24h', 'price_change_24h', 
                            'market_cap', 'ath', 'ath_change_percentage', 'total_supply', 'circulating_supply']:
                    if isinstance(value, datetime):
                        additional_data[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        additional_data[key] = json.dumps(value)
                    else:
                        additional_data[key] = value

            # Only store in market_data if price is valid
            if current_price > 0:
                # Insert into original market_data table
                cursor.execute("""
                    INSERT INTO market_data (
                        timestamp, chain, price, volume, price_change_24h, 
                        market_cap, ath, ath_change_percentage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_time, chain, current_price, volume, price_change_24h, 
                    market_cap, ath, ath_change_percentage
                ))

                # Also store in price_history table for our own calculations
                logger.logger.debug(f"Storing {chain} in price_history: price={current_price}, volume={volume}")
                self.store_price_history(
                    token=chain,
                    price=current_price,
                    volume=volume,
                    market_cap=market_cap,
                    total_supply=total_supply,
                    circulating_supply=circulating_supply
                )

                # Store additional data in generic_json_data if there's extra info
                if additional_data:
                    try:
                        cursor.execute("""
                            INSERT INTO generic_json_data (timestamp, data_type, data)
                            VALUES (?, ?, ?)
                        """, (
                            current_time,
                            f"market_data_extended_{chain}",
                            json.dumps(additional_data)
                        ))
                    except Exception as json_error:
                        # Log but continue - this is supplementary data
                        logger.log_error(f"Generic JSON Data - {chain}", str(json_error))

            conn.commit()
            logger.logger.debug(f"Stored market data for {chain} at {current_time.isoformat()}")

        except Exception as e:
            logger.log_error(f"Store Market Data - {chain}", str(e))
            if 'conn' in locals() and conn:
                conn.rollback()

    def enhance_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance market data with calculated price changes
        Handles various input formats and ensures backward compatibility
        NOW ENHANCED: Supports multiple CoinGecko endpoint formats including market chart and sparkline data

        Args:
            market_data: Market data dictionary or list - supports:
                    - Standard market data (existing functionality)
                    - Market chart data with price/volume arrays
                    - Individual coin data with sparkline
                    - Mixed format combinations
        
        Returns:
            Enhanced market data dictionary (same format as before)
        """
        try:
            logger.logger.debug(f"enhance_market_data called with data type: {type(market_data)}")
        
            # Log input data structure overview
            if isinstance(market_data, dict):
                logger.logger.debug(f"Input is dictionary with {len(market_data)} keys")
                sample_keys = list(market_data.keys())[:5]
                logger.logger.debug(f"Sample keys: {sample_keys}")
            
                # Log a sample value structure if available
                if sample_keys and sample_keys[0] in market_data:
                    sample_value = market_data[sample_keys[0]]
                    logger.logger.debug(f"Sample value type for key '{sample_keys[0]}': {type(sample_value)}")
                
                    if isinstance(sample_value, dict):
                        logger.logger.debug(f"Sample value keys: {list(sample_value.keys())[:5]}")
            elif isinstance(market_data, list):
                logger.logger.debug(f"Input is list with {len(market_data)} items")
            
                # Log a sample item structure if available
                if market_data and len(market_data) > 0:
                    sample_item = market_data[0]
                    logger.logger.debug(f"Sample item type: {type(sample_item)}")
                
                    if isinstance(sample_item, dict):
                        logger.logger.debug(f"Sample item keys: {list(sample_item.keys())[:5]}")
            else:
                logger.logger.warning(f"Unexpected market_data type: {type(market_data)}")
                return market_data

            # ================================================================
            # 🆕 ENHANCED: PRE-PROCESS DIFFERENT COINGECKO ENDPOINT FORMATS
            # ================================================================
            
            # Convert list to dictionary format (existing functionality preserved)
            market_dict = {}
            if isinstance(market_data, list):
                logger.logger.debug("Converting list to dictionary format")
                for item in market_data:
                    if isinstance(item, dict):
                        # Use symbol as primary key, fallback to id
                        symbol = item.get('symbol')
                        if symbol:
                            market_dict[symbol.upper()] = item
                        elif item.get('id'):
                            # Try to map CoinGecko ID to symbol
                            coingecko_id = item.get('id')
                            if coingecko_id and isinstance(coingecko_id, str):
                                symbol = self._map_coingecko_id_to_symbol(coingecko_id)
                                if symbol:
                                    market_dict[symbol] = item
            else:
                market_dict = market_data.copy()

            # ================================================================
            # 🆕 NEW: DETECT AND HANDLE MARKET CHART DATA IN MARKET_DICT
            # ================================================================
            for token, data in list(market_dict.items()):
                if isinstance(data, dict):
                    # Check if this is market chart format data
                    if self._is_market_chart_format(data):
                        logger.logger.debug(f"Detected market chart format for {token}")
                        # Store the market chart data using our enhanced store_market_data
                        self.store_market_data(token, data)
                        # Convert to standard format for processing
                        market_dict[token] = self._convert_chart_to_standard_format(token, data)
                    
                    # Check if this is individual coin format with sparkline
                    elif self._is_individual_coin_format(data):
                        logger.logger.debug(f"Detected individual coin format for {token}")
                        # Store the individual coin data
                        self.store_market_data(token, data)
                        # Convert to standard format for processing
                        market_dict[token] = self._convert_individual_to_standard_format(token, data)

            # ================================================================
            # ✅ EXISTING FUNCTIONALITY (UNCHANGED)
            # ================================================================
            
            # Initialize counters for processing summary
            processed_tokens = 0
            price_history_stored = 0
            price_changes_calculated = 0
            errors_encountered = 0

            # Process each token in the market data
            for token, data in market_dict.items():
                if not isinstance(data, dict):
                    logger.logger.warning(f"Skipping non-dict data for token {token}: {type(data)}")
                    continue

                try:
                    processed_tokens += 1
                    logger.logger.debug(f"Processing token: {token}")

                    # Extract current price safely
                    current_price = None
                    for price_key in ['current_price', 'price', 'last_price']:
                        if price_key in data and data[price_key] is not None:
                            try:
                                current_price = float(data[price_key])
                                if current_price > 0:
                                    break
                            except (ValueError, TypeError):
                                continue

                    if not current_price or current_price <= 0:
                        logger.logger.warning(f"No valid current price found for {token}")
                        continue

                    # Extract other market data safely
                    volume = None
                    for volume_key in ['volume', 'total_volume', 'volume_24h']:
                        if volume_key in data and data[volume_key] is not None:
                            try:
                                volume = float(data[volume_key])
                                if volume >= 0:
                                    break
                            except (ValueError, TypeError):
                                continue

                    market_cap = None
                    for cap_key in ['market_cap', 'market_cap_usd']:
                        if cap_key in data and data[cap_key] is not None:
                            try:
                                market_cap = float(data[cap_key])
                                if market_cap > 0:
                                    break
                            except (ValueError, TypeError):
                                continue

                    # Extract supply data
                    total_supply = data.get('total_supply')
                    circulating_supply = data.get('circulating_supply')

                    # Store in price_history table for our own calculations
                    try:
                        logger.logger.debug(f"Storing price history for {token}")
                        stored = self.store_price_history(
                            token=token,
                            price=current_price,
                            volume=volume,
                            market_cap=market_cap,
                            total_supply=total_supply,
                            circulating_supply=circulating_supply
                        )
                    
                        if stored:
                            logger.logger.debug(f"Successfully stored price history for {token}")
                            price_history_stored += 1
                        else:
                            logger.logger.warning(f"Failed to store price history for {token}")
                    except Exception as store_error:
                        logger.log_error(f"Store Price History - {token}", str(store_error))
                        logger.logger.error(f"Error storing price history for {token}: {str(store_error)}")
                        # Continue processing other tokens even if one fails
                        errors_encountered += 1
                        continue
            
                    # Calculate price changes
                    try:
                        logger.logger.debug(f"Calculating price changes for {token}")
                        price_changes = self.calculate_price_changes(token, current_price)
                    
                        if price_changes:
                            logger.logger.debug(f"Successfully calculated price changes for {token}: {price_changes}")
                        
                            # Update data with calculated changes
                            for change_key, change_value in price_changes.items():
                                if change_value is not None:
                                    data[change_key] = change_value
                                    logger.logger.debug(f"Added calculated change to {token} data: {change_key}={change_value}")
                            
                                # Also set the original key if it exists but is None/0
                                if change_key == 'price_change_percentage_24h':
                                    # For backward compatibility with existing code
                                    if 'price_change_24h' not in data or data['price_change_24h'] is None or data['price_change_24h'] == 0:
                                        data['price_change_24h'] = change_value
                                        logger.logger.debug(f"Added calculated change to {token} data for compatibility: price_change_24h={change_value}")
                        
                            price_changes_calculated += 1
                        else:
                            logger.logger.warning(f"No price changes calculated for {token}")
                    except Exception as calc_error:
                        logger.log_error(f"Calculate Price Changes - {token}", str(calc_error))
                        logger.logger.error(f"Error calculating price changes for {token}: {str(calc_error)}")
                        errors_encountered += 1
            
                except Exception as token_error:
                    logger.log_error(f"Enhance Market Data - {token}", str(token_error))
                    logger.logger.error(f"Error processing token {token}: {str(token_error)}")
                    errors_encountered += 1

            # Log summary statistics
            logger.logger.info(f"enhance_market_data summary: processed {processed_tokens} tokens, "
                            f"stored {price_history_stored} to price history, "
                            f"calculated changes for {price_changes_calculated}, "
                            f"encountered {errors_encountered} errors")

            # Return in the same format that was provided
            if isinstance(market_data, list):
                # Convert back to list
                logger.logger.debug("Converting enhanced market_dict back to list")
                result = list(market_dict.values())
            
                # Remove duplicates (tokens added by both symbol and id)
                seen_items = set()
                unique_result = []

                for item in result:
                    # Use a unique identifier for deduplication
                    item_id = item.get('id') or item.get('symbol', '')
                    if item_id and item_id not in seen_items:
                        seen_items.add(item_id)
                        unique_result.append(item)

                logger.logger.debug(f"Returning enhanced market data list with {len(unique_result)} unique entries")
                return market_dict  # Return the dictionary, not the list

            logger.logger.debug(f"Returning enhanced market data dictionary with {len(market_dict)} entries")
            return market_dict

        except Exception as e:
            logger.log_error("Enhance Market Data", str(e))
            logger.logger.error(f"Error in enhance_market_data: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Return original to avoid breaking anything
            return market_data

    # ================================================================
    # 🆕 NEW HELPER METHODS FOR DIFFERENT COINGECKO FORMATS
    # ================================================================

    def _is_market_chart_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is from market chart endpoint"""
        return ('prices' in data and isinstance(data['prices'], list) and 
                len(data['prices']) > 0 and isinstance(data['prices'][0], list))

    def _is_individual_coin_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is from individual coin endpoint"""
        return ('market_data' in data and 'sparkline_in_7d' in data)

    def _map_coingecko_id_to_symbol(self, coingecko_id: str) -> str:
        """Map CoinGecko ID to trading symbol"""
        symbol_to_id = {
            'BTC': 'bitcoin', 
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'BNB': 'binancecoin',
            'AVAX': 'avalanche-2',
            'DOT': 'polkadot',
            'UNI': 'uniswap',
            'NEAR': 'near',
            'AAVE': 'aave',
            'FIL': 'filecoin',
            'POL': 'matic-network',
            'TRUMP': 'official-trump',
            'KAITO': 'kaito'
        }
        return symbol_to_id.get(coingecko_id, coingecko_id.upper())

    def _convert_chart_to_standard_format(self, token: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert market chart format to standard market data format"""
        try:
            prices_array = chart_data.get('prices', [])
            volumes_array = chart_data.get('volumes', [])
            
            if not prices_array:
                return {'symbol': token, 'current_price': 0, 'volume': 0}
            
            # Get latest price and volume
            latest_price = prices_array[-1][1] if len(prices_array[-1]) > 1 else 0
            latest_volume = volumes_array[-1][1] if volumes_array and len(volumes_array[-1]) > 1 else 0
            
            # Calculate 24h change if we have enough data
            price_change_24h = 0
            if len(prices_array) > 24:  # Assuming hourly data
                price_24h_ago = prices_array[-25][1] if len(prices_array[-25]) > 1 else latest_price
                if price_24h_ago > 0:
                    price_change_24h = ((latest_price / price_24h_ago) - 1) * 100
            
            return {
                'symbol': token,
                'current_price': latest_price,
                'volume': latest_volume,
                'price_change_percentage_24h': price_change_24h,
                'data_source': 'market_chart'
            }
            
        except Exception as e:
            logger.logger.error(f"Error converting chart data for {token}: {str(e)}")
            return {'symbol': token, 'current_price': 0, 'volume': 0}

    def _convert_individual_to_standard_format(self, token: str, coin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert individual coin format to standard market data format"""
        try:
            market_data_section = coin_data.get('market_data', {})
            
            # Extract current price
            current_price_data = market_data_section.get('current_price', {})
            current_price = current_price_data.get('usd', 0) if isinstance(current_price_data, dict) else 0
            
            # Extract volume
            volume_data = market_data_section.get('total_volume', {})
            volume = volume_data.get('usd', 0) if isinstance(volume_data, dict) else 0
            
            # Extract market cap
            market_cap_data = market_data_section.get('market_cap', {})
            market_cap = market_cap_data.get('usd', 0) if isinstance(market_cap_data, dict) else 0
            
            # Extract price changes
            price_change_24h = market_data_section.get('price_change_percentage_24h', 0)
            price_change_7d = market_data_section.get('price_change_percentage_7d', 0)
            
            return {
                'symbol': token,
                'current_price': current_price,
                'volume': volume,
                'market_cap': market_cap,
                'price_change_percentage_24h': price_change_24h,
                'price_change_percentage_7d': price_change_7d,
                'data_source': 'individual_coin'
            }
            
        except Exception as e:
            logger.logger.error(f"Error converting individual coin data for {token}: {str(e)}")
            return {'symbol': token, 'current_price': 0, 'volume': 0}

    def store_token_correlations(self, token: str, correlations: Dict[str, Any]) -> None:
        """Store token-specific correlation data"""
        conn, cursor = self._get_connection()
        try:
            # Extract average correlations
            avg_price_corr = correlations.get('avg_price_correlation', 0)
            avg_volume_corr = correlations.get('avg_volume_correlation', 0)
            
            cursor.execute("""
                INSERT INTO token_correlations (
                    timestamp, token, avg_price_correlation, avg_volume_correlation, full_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                avg_price_corr,
                avg_volume_corr,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"Stored correlation data for {token}")
        except Exception as e:
            logger.log_error(f"Store Token Correlations - {token}", str(e))
            conn.rollback()
            
    def store_token_market_comparison(self, token: str, vs_market_avg_change: float,
                                    vs_market_volume_growth: float, outperforming_market: bool,
                                    correlations: Dict[str, Any]) -> None:
        """Store token vs market comparison data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO token_market_comparison (
                    timestamp, token, vs_market_avg_change, vs_market_volume_growth,
                    outperforming_market, correlations
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                vs_market_avg_change,
                vs_market_volume_growth,
                1 if outperforming_market else 0,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"Stored market comparison data for {token}")
        except Exception as e:
            logger.log_error(f"Store Token Market Comparison - {token}", str(e))
            conn.rollback()

    def store_posted_content(self, content: str, sentiment: Dict,
                            trigger_type: str, price_data: Optional[Dict] = None,
                            meme_phrases: Optional[Dict] = None, is_prediction: bool = False,
                            prediction_data: Optional[Dict] = None, timeframe: str = "1h",
                            tech_category: Optional[str] = None, tech_metadata: Optional[Dict] = None,
                            is_educational: bool = False, market_context: Optional[Dict] = None,
                            vs_market_change: Optional[float] = None, market_sentiment: Optional[str] = None,
                            timestamp: Optional[datetime] = None):
        """Store posted content with metadata, timeframe and tech-related fields"""
        conn, cursor = self._get_connection()
        try:
            # Validate that content is not None or empty
            if content is None or content.strip() == "":
                logger.logger.warning("Attempt to store empty content in posted_content table")
                content = "[Empty Content]"  # Provide a default value to avoid NOT NULL constraint
        
            # First check if tech columns exist, add them if they don't
            self._ensure_tech_columns_exist()
        
            # Set defaults for mutable parameters
            if price_data is None:
                price_data = {}
            if meme_phrases is None:
                meme_phrases = {}
            
            # Use provided timestamp or default to current time
            current_time = timestamp if timestamp is not None else datetime.now()

            # Helper function to convert datetime objects in a dictionary to ISO format strings
            def datetime_to_iso(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: datetime_to_iso(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [datetime_to_iso(item) for item in obj]
                return obj

            # Process JSON data to handle datetime objects
            sentiment_json = json.dumps(datetime_to_iso(sentiment))
    
            # Include vs_market_change in price_data if provided
            if vs_market_change is not None:
                price_data_copy = price_data.copy() if price_data else {}
                price_data_copy['vs_market_change'] = vs_market_change
                price_data_json = json.dumps(datetime_to_iso(price_data_copy))
            else:
                price_data_json = json.dumps(datetime_to_iso(price_data))
        
            meme_phrases_json = json.dumps(datetime_to_iso(meme_phrases))
            prediction_data_json = json.dumps(datetime_to_iso(prediction_data)) if prediction_data else None
            tech_metadata_json = json.dumps(datetime_to_iso(tech_metadata)) if tech_metadata else None

            cursor.execute("""
                INSERT INTO posted_content (
                    timestamp, content, sentiment, trigger_type, 
                    price_data, meme_phrases, is_prediction, prediction_data, timeframe,
                    tech_category, tech_metadata, is_educational
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                current_time,
                content,
                sentiment_json,
                trigger_type,
                price_data_json,
                meme_phrases_json,
                1 if is_prediction else 0,
                prediction_data_json,
                timeframe,
                tech_category,
                tech_metadata_json,
                1 if is_educational else 0
            ))
            conn.commit()
        except Exception as e:
            logger.log_error("Store Posted Content", str(e))
            conn.rollback()
    
    def check_if_post_replied(self, post_id: str, post_url: Optional[str] = None) -> bool:
        """
        Check if we've already replied to a post
    
        Args:
            post_id: The ID of the post
            post_url: The URL of the post (optional)
        
        Returns:
            True if we've already replied to this post, False otherwise
        """
        try:
            conn, cursor = self._get_connection()
        
            # Check for post_id first
            if post_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM replied_posts
                    WHERE post_id = ?
                """, (post_id,))
                count = cursor.fetchone()[0]
                if count > 0:
                    return True
                
            # If post_url is provided and post_id check failed, try with URL
            if post_url:
                cursor.execute("""
                    SELECT COUNT(*) FROM replied_posts
                    WHERE post_url = ?
                """, (post_url,))
                count = cursor.fetchone()[0]
                if count > 0:
                    return True
                
            # No record found
            return False
        
        except Exception as e:
            logger.log_error("Check If Post Replied", str(e))
            return False

    def store_mood(self, chain: str, mood: str, indicators: Dict) -> None:
        """Store mood data for a specific chain"""
        conn = None
        try:
            conn, cursor = self._get_connection()
            cursor.execute("""
                INSERT INTO mood_history (
                    timestamp, chain, mood, indicators
                ) VALUES (?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                mood,
                json.dumps(indicators)  # indicators is already a Dict, no need for asdict()
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Mood - {chain}", str(e))
            if conn:
                conn.rollback()
            
    def store_smart_money_indicators(self, chain: str, indicators: Dict[str, Any]) -> None:
        """Store smart money indicators for a chain"""
        conn, cursor = self._get_connection()
        try:
            # Extract values with defaults for potential missing keys
            volume_z_score = indicators.get('volume_z_score', 0.0)
            price_volume_divergence = 1 if indicators.get('price_volume_divergence', False) else 0
            stealth_accumulation = 1 if indicators.get('stealth_accumulation', False) else 0
            abnormal_volume = 1 if indicators.get('abnormal_volume', False) else 0
            volume_vs_hourly_avg = indicators.get('volume_vs_hourly_avg', 0.0)
            volume_vs_daily_avg = indicators.get('volume_vs_daily_avg', 0.0)
            volume_cluster_detected = 1 if indicators.get('volume_cluster_detected', False) else 0
            
            # Convert unusual_trading_hours to JSON if present
            unusual_hours = json.dumps(indicators.get('unusual_trading_hours', []))
            
            # Store all raw data for future reference
            raw_data = json.dumps(indicators)
            
            cursor.execute("""
                INSERT INTO smart_money_indicators (
                    timestamp, chain, volume_z_score, price_volume_divergence,
                    stealth_accumulation, abnormal_volume, volume_vs_hourly_avg,
                    volume_vs_daily_avg, volume_cluster_detected, unusual_trading_hours,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                volume_z_score,
                price_volume_divergence,
                stealth_accumulation,
                abnormal_volume,
                volume_vs_hourly_avg,
                volume_vs_daily_avg,
                volume_cluster_detected,
                unusual_hours,
                raw_data
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Smart Money Indicators - {chain}", str(e))
            conn.rollback()
            
    def _store_json_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Generic method to store JSON data in a generic_json_data table"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO generic_json_data (
                    timestamp, data_type, data
                ) VALUES (?, ?, ?)
            """, (
                datetime.now(),
                data_type,
                json.dumps(data)
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store JSON Data - {data_type}", str(e))
            conn.rollback()

    #########################
    # DATA RETRIEVAL METHODS
    #########################

    def get_recent_market_data(self, chain: str, hours: int = 24) -> List[Dict]:
        """
        Get recent market data for a specific chain with robust token identifier handling
        Enhanced to pull from both market_data AND price_history tables for comprehensive data
        
        Args:
            chain: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            hours: Number of hours to look back (default: 24)
        
        Returns:
            List[Dict]: Market data entries with enhanced fields from price_history when available
        """
        try:
            # Normalize token identifier to handle both symbol and CoinGecko ID inputs
            normalized_token = self._normalize_token_for_database(chain)
            
            # Validate parameters
            if not normalized_token:
                logger.logger.warning(f"Invalid token identifier: {chain}")
                return []
                
            if hours <= 0:
                logger.logger.warning(f"Invalid hours parameter: {hours}")
                hours = 24
                
            logger.logger.debug(f"Getting recent market data for {normalized_token} (original: {chain}), hours: {hours}")
            
            conn, cursor = self._get_connection()
            
            # Enhanced query: get data from market_data table with additional price_history data
            cursor.execute("""
                SELECT 
                    md.*,
                    ph.volume as ph_volume,
                    ph.market_cap as ph_market_cap,
                    ph.total_supply,
                    ph.circulating_supply
                FROM market_data md
                LEFT JOIN price_history ph ON md.chain = ph.token 
                    AND datetime(md.timestamp) = datetime(ph.timestamp)
                WHERE md.chain = ? 
                AND md.timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY md.timestamp DESC
            """, (normalized_token, hours))
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                
                # Enhance the row with additional data from price_history if available
                if row_dict.get('ph_volume') is not None:
                    # Use price_history volume if it's more recent/accurate
                    row_dict['volume_enhanced'] = row_dict['ph_volume']
                
                if row_dict.get('ph_market_cap') is not None:
                    # Use price_history market_cap if available
                    row_dict['market_cap_enhanced'] = row_dict['ph_market_cap']
                
                # Add supply data if available from price_history
                if row_dict.get('total_supply') is not None:
                    row_dict['total_supply'] = row_dict['total_supply']
                
                if row_dict.get('circulating_supply') is not None:
                    row_dict['circulating_supply'] = row_dict['circulating_supply']
                
                # Clean up the temporary fields
                for temp_field in ['ph_volume', 'ph_market_cap']:
                    if temp_field in row_dict:
                        del row_dict[temp_field]
                
                results.append(row_dict)
            
            logger.logger.debug(f"Retrieved {len(results)} market data entries for {normalized_token}")
            return results

        except Exception as e:
            logger.log_error(f"Get Recent Market Data - {chain}", str(e))
            return []

    def _normalize_token_for_database(self, token_input: str) -> str:
        """
        Normalize token identifier for database operations
        Converts both symbols and CoinGecko IDs to the format used in database storage
        
        Args:
            token_input: Token identifier (symbol or CoinGecko ID)
        
        Returns:
            Normalized token identifier for database queries
        """
        try:
            if not token_input or not isinstance(token_input, str):
                return ""
                
            token_input = token_input.strip()
            
            # If it's already a symbol (uppercase, short), return as-is
            if token_input.isupper() and len(token_input) <= 6:
                return token_input
                
            # Check if it's a CoinGecko ID and convert to symbol
            coingecko_to_symbol = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH', 
                'solana': 'SOL',
                'ripple': 'XRP',
                'binancecoin': 'BNB',
                'avalanche-2': 'AVAX',
                'polkadot': 'DOT',
                'uniswap': 'UNI',
                'near': 'NEAR',
                'aave': 'AAVE',
                'filecoin': 'FIL',
                'matic-network': 'POL',
                'official-trump': 'TRUMP',
                'kaito': 'KAITO'
            }
            
            if token_input.lower() in coingecko_to_symbol:
                return coingecko_to_symbol[token_input.lower()]
                
            # If not found in mapping, try uppercase conversion
            return token_input.upper()
            
        except Exception as e:
            logger.logger.debug(f"Error normalizing token {token_input}: {str(e)}")
            return token_input.upper() if token_input else ""
            
    def get_token_correlations(self, token: str, hours: int = 24) -> List[Dict]:
        """
        Get token-specific correlation data with robust token identifier handling
        Retrieves correlation analysis from database with enhanced error handling
        
        Args:
            token: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            hours: Number of hours to look back (default: 24)
        
        Returns:
            List[Dict]: Correlation data entries with parsed JSON fields
        """
        try:
            # Define symbol to CoinGecko ID mapping for normalization
            symbol_to_coingecko = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'XRP': 'ripple',
                'BNB': 'binancecoin', 'AVAX': 'avalanche-2', 'DOT': 'polkadot', 'UNI': 'uniswap',
                'NEAR': 'near', 'AAVE': 'aave', 'FIL': 'filecoin', 'POL': 'matic-network',
                'TRUMP': 'official-trump', 'KAITO': 'kaito'
            }
            
            coingecko_to_symbol = {v: k for k, v in symbol_to_coingecko.items()}
            
            # Validate parameters
            if not token or not isinstance(token, str):
                logger.logger.warning("Invalid token parameter for correlation lookup")
                return []
                
            if hours <= 0:
                logger.logger.warning(f"Invalid hours parameter: {hours}")
                hours = 24
                
            # Normalize token to symbol format for database storage consistency
            normalized_token = token.upper()
            if token.lower() in coingecko_to_symbol:
                normalized_token = coingecko_to_symbol[token.lower()]
                
            logger.logger.debug(f"Getting token correlations for {normalized_token} (original: {token}), hours: {hours}")
            
            conn, cursor = self._get_connection()
            
            # Try primary token lookup first
            cursor.execute("""
                SELECT * FROM token_correlations 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (normalized_token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # If no results with normalized token, try original token
            if not results and normalized_token != token:
                logger.logger.debug(f"No results for {normalized_token}, trying original token {token}")
                cursor.execute("""
                    SELECT * FROM token_correlations 
                    WHERE token = ?
                    AND timestamp >= datetime('now', '-' || ? || ' hours')
                    ORDER BY timestamp DESC
                """, (token, hours))
                
                results = [dict(row) for row in cursor.fetchall()]
                
            # If still no results, try CoinGecko ID format
            if not results:
                coingecko_id = symbol_to_coingecko.get(token.upper())
                if coingecko_id:
                    logger.logger.debug(f"No results for symbol, trying CoinGecko ID {coingecko_id}")
                    cursor.execute("""
                        SELECT * FROM token_correlations 
                        WHERE token = ?
                        AND timestamp >= datetime('now', '-' || ? || ' hours')
                        ORDER BY timestamp DESC
                    """, (coingecko_id, hours))
                    
                    results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field in results
            for result in results:
                try:
                    result["full_data"] = json.loads(result["full_data"]) if result["full_data"] else {}
                except (json.JSONDecodeError, TypeError) as json_error:
                    logger.logger.debug(f"Error parsing JSON for correlation data: {str(json_error)}")
                    result["full_data"] = {}
                    
            logger.logger.debug(f"Retrieved {len(results)} correlation entries for {token}")
            return results
            
        except Exception as e:
            logger.log_error(f"Get Token Correlations - {token}", str(e))
            return []
            
    def get_token_market_comparison(self, token: str, hours: int = 24) -> List[Dict]:
        """Get token vs market comparison data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM token_market_comparison 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field
            for result in results:
                result["correlations"] = json.loads(result["correlations"]) if result["correlations"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Token Market Comparison - {token}", str(e))
            return []
        
    def get_recent_posts(self, hours: int = 24, timeframe: Optional[str] = None) -> List[Dict]:
        """
        Get recent posted content
        Can filter by timeframe
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT * FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params: List[Union[int, str]] = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["sentiment"] = json.loads(result["sentiment"]) if result["sentiment"] else {}
                result["price_data"] = json.loads(result["price_data"]) if result["price_data"] else {}
                result["meme_phrases"] = json.loads(result["meme_phrases"]) if result["meme_phrases"] else {}
                result["prediction_data"] = json.loads(result["prediction_data"]) if result["prediction_data"] else None
                
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Get Recent Posts", str(e))
            return []

    def get_chain_stats(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(price) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume,
                    AVG(price_change_24h) as avg_price_change
                FROM market_data 
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (chain, hours))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return {}
        except Exception as e:
            logger.log_error(f"Get Chain Stats - {chain}", str(e))
            return {}
            
    def get_smart_money_indicators(self, chain: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent smart money indicators for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (chain, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["unusual_trading_hours"] = json.loads(result["unusual_trading_hours"]) if result["unusual_trading_hours"] else []
                result["raw_data"] = json.loads(result["raw_data"]) if result["raw_data"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Smart Money Indicators - {chain}", str(e))
            return []
            
    def get_token_market_stats(self, token: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary of token vs market performance"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(vs_market_avg_change) as avg_performance_diff,
                    AVG(vs_market_volume_growth) as avg_volume_growth_diff,
                    SUM(CASE WHEN outperforming_market = 1 THEN 1 ELSE 0 END) as outperforming_count,
                    COUNT(*) as total_records
                FROM token_market_comparison
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (token, hours))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Calculate percentage of time outperforming
                if result_dict['total_records'] > 0:
                    result_dict['outperforming_percentage'] = (result_dict['outperforming_count'] / result_dict['total_records']) * 100
                else:
                    result_dict['outperforming_percentage'] = 0
                    
                return result_dict
            return {}
        except Exception as e:
            logger.log_error(f"Get Token Market Stats - {token}", str(e))
            return {}

    def get_latest_smart_money_alert(self, chain: str) -> Optional[Dict[str, Any]]:
        """Get the most recent smart money alert for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND (abnormal_volume = 1 OR stealth_accumulation = 1 OR volume_cluster_detected = 1)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (chain,))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Parse JSON fields
                result_dict["unusual_trading_hours"] = json.loads(result_dict["unusual_trading_hours"]) if result_dict["unusual_trading_hours"] else []
                result_dict["raw_data"] = json.loads(result_dict["raw_data"]) if result_dict["raw_data"] else {}
                
                return result_dict
            return None
        except Exception as e:
            logger.log_error(f"Get Latest Smart Money Alert - {chain}", str(e))
            return None
    
    def get_volume_trend(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get volume trend analysis for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    timestamp,
                    volume
                FROM market_data
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """, (chain, hours))
            
            results = cursor.fetchall()
            if not results:
                return {'trend': 'insufficient_data', 'change': 0}
                
            # Calculate trend
            volumes = [row['volume'] for row in results]
            earliest_volume = volumes[0] if volumes else 0
            latest_volume = volumes[-1] if volumes else 0
            
            if earliest_volume > 0:
                change_pct = ((latest_volume - earliest_volume) / earliest_volume) * 100
            else:
                change_pct = 0
                
            # Determine trend description
            if change_pct >= 15:
                trend = "significant_increase"
            elif change_pct <= -15:
                trend = "significant_decrease"
            elif change_pct >= 5:
                trend = "moderate_increase"
            elif change_pct <= -5:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            return {
                'trend': trend,
                'change': change_pct,
                'earliest_volume': earliest_volume,
                'latest_volume': latest_volume,
                'data_points': len(volumes)
            }
            
        except Exception as e:
            logger.log_error(f"Get Volume Trend - {chain}", str(e))
            return {'trend': 'error', 'change': 0}
            
    def get_top_performing_tokens(self, hours: int = 24, limit: int = 5) -> List[Dict[str, Any]]:
        """Get list of top performing tokens based on price change"""
        conn, cursor = self._get_connection()
        try:
            # Get unique tokens in database
            cursor.execute("""
                SELECT DISTINCT chain
                FROM market_data
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))
            tokens = [row['chain'] for row in cursor.fetchall()]
            
            results = []
            for token in tokens:
                # Get latest price and 24h change
                cursor.execute("""
                    SELECT price, price_change_24h
                    FROM market_data
                    WHERE chain = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token,))
                data = cursor.fetchone()
                
                if data:
                    results.append({
                        'token': token,
                        'price': data['price'],
                        'price_change_24h': data['price_change_24h']
                    })
            
            # Sort by price change (descending)
            results.sort(key=lambda x: x.get('price_change_24h', 0), reverse=True)
            
            # Return top N tokens
            return results[:limit]
            
        except Exception as e:
            logger.log_error("Get Top Performing Tokens", str(e))
            return []

    def get_tokens_by_prediction_accuracy(self, timeframe: str = "1h", min_predictions: int = 5) -> List[Dict[str, Any]]:
        """
        Get tokens sorted by prediction accuracy for a specific timeframe
        Only includes tokens with at least min_predictions number of predictions
        """
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT token, accuracy_rate, total_predictions, correct_predictions
                FROM prediction_performance
                WHERE timeframe = ? AND total_predictions >= ?
                ORDER BY accuracy_rate DESC
            """, (timeframe, min_predictions))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error(f"Get Tokens By Prediction Accuracy - {timeframe}", str(e))
            return []

    #########################
    # DUPLICATE DETECTION METHODS
    #########################
    
    def check_content_similarity(self, content: str, timeframe: Optional[str] = None) -> bool:
        """
        Check if similar content was recently posted
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-1 hour')
            """
            
            params = []
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Simple similarity check - can be enhanced later
            return any(content.strip() == post.strip() for post in recent_posts)
        except Exception as e:
            logger.log_error("Check Content Similarity", str(e))
            return False
            
    def check_exact_content_match(self, content: str, timeframe: Optional[str] = None) -> bool:
        """
        Check for exact match of content within recent posts
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT COUNT(*) as count FROM posted_content 
                WHERE content = ? 
                AND timestamp >= datetime('now', '-3 hours')
            """
            
            params = [content]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result['count'] > 0 if result else False
        except Exception as e:
            logger.log_error("Check Exact Content Match", str(e))
            return False
            
    def check_content_similarity_with_timeframe(self, content: str, hours: int = 1, timeframe: Optional[str] = None) -> bool:
        """
        Check if similar content was posted within a specified timeframe
        Can filter by prediction timeframe
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            
            params: List[Union[int, str]] = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Split content into main text and hashtags
            content_main = content.split("\n\n#")[0].lower() if "\n\n#" in content else content.lower()
            
            for post in recent_posts:
                post_main = post.split("\n\n#")[0].lower() if "\n\n#" in post else post.lower()
                
                # Calculate similarity based on word overlap
                content_words = set(content_main.split())
                post_words = set(post_main.split())
                
                if content_words and post_words:
                    overlap = len(content_words.intersection(post_words))
                    similarity = overlap / max(len(content_words), len(post_words))
                    
                    # Consider similar if 70% or more words overlap
                    if similarity > 0.7:
                        return True
            
            return False
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Check Content Similarity With Timeframe", str(e))
            return False

    #########################
    # PREDICTION METHODS
    #########################
    
    def store_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> Optional[int]:
        """
        Store a prediction in the database with enhanced compatibility for multiple prediction formats
        
        Enhanced to handle both flat and nested prediction structures:
        - Flat structure: prediction_data["price"], prediction_data["confidence"]
        - Nested structure: prediction_data["prediction"]["price"], prediction_data["prediction"]["confidence"]
        
        Returns the ID of the inserted prediction, or None if failed
        """
        conn = None
        prediction_id = None
        
        try:
            conn, cursor = self._get_connection()
            
            # ================================================================
            # 🔧 ENHANCED PREDICTION DATA EXTRACTION 🔧
            # ================================================================
            
            # Helper function to safely extract prediction fields from multiple possible locations
            def extract_prediction_field(field_name: str, default_value=None):
                """Extract a field from either flat or nested prediction structure"""
                # Try flat structure first (prediction_data[field])
                if field_name in prediction_data and prediction_data[field_name] is not None:
                    return prediction_data[field_name]
                
                # Try nested structure (prediction_data["prediction"][field])
                if "prediction" in prediction_data and isinstance(prediction_data["prediction"], dict):
                    nested_prediction = prediction_data["prediction"]
                    if field_name in nested_prediction and nested_prediction[field_name] is not None:
                        return nested_prediction[field_name]
                
                # Return default if not found in either location
                return default_value
            
            # Extract prediction details with compatibility for both structures
            prediction_price = extract_prediction_field("price", 0)
            prediction_confidence = extract_prediction_field("confidence", 70)
            prediction_lower_bound = extract_prediction_field("lower_bound", 0)
            prediction_upper_bound = extract_prediction_field("upper_bound", 0)
            
            # Extract other fields with fallbacks
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            model_weights = json.dumps(prediction_data.get("model_weights", {}))
            model_inputs = json.dumps(prediction_data.get("inputs", {}))
            
            # ================================================================
            # 🔧 DATA VALIDATION AND LOGGING 🔧
            # ================================================================
            
            # Log the extraction results for debugging
            logger.logger.debug(f"Extracted prediction fields for {token}:")
            logger.logger.debug(f"  price: {prediction_price}")
            logger.logger.debug(f"  confidence: {prediction_confidence}")
            logger.logger.debug(f"  lower_bound: {prediction_lower_bound}")
            logger.logger.debug(f"  upper_bound: {prediction_upper_bound}")
            
            # Validate critical fields with proper None checks
            if prediction_price is None or prediction_price <= 0:
                # Try to find alternative price fields
                alt_price = extract_prediction_field("predicted_price", None)
                if alt_price is None:
                    alt_price = extract_prediction_field("take_profit", None)
                if alt_price is None:
                    alt_price = extract_prediction_field("target_price", None)
                
                if alt_price is not None and alt_price > 0:
                    prediction_price = alt_price
                    logger.logger.debug(f"Used alternative price field for {token}: {prediction_price}")
                else:
                    raise ValueError(f"No valid price found for {token} prediction")
            
            # Ensure confidence is within valid range with None check
            if prediction_confidence is None:
                prediction_confidence = 70  # Default confidence
            if not (0 <= prediction_confidence <= 100):
                prediction_confidence = max(0, min(100, prediction_confidence))
                logger.logger.debug(f"Clamped confidence for {token} to: {prediction_confidence}")
            
            # ================================================================
            # 🔧 BOUNDS VALIDATION AND FALLBACK CALCULATION 🔧
            # ================================================================
            
            # If bounds are missing or invalid, calculate them (with None checks)
            if (prediction_lower_bound is None or prediction_lower_bound <= 0 or 
                prediction_upper_bound is None or prediction_upper_bound <= 0):
                logger.logger.debug(f"Calculating missing bounds for {token}")
                
                # Get expected return for bound calculation with None check
                expected_return_pct = extract_prediction_field("expected_return_pct", 0)
                if expected_return_pct is None or expected_return_pct == 0:
                    expected_return_pct = extract_prediction_field("percent_change", 0)
                if expected_return_pct is None:
                    expected_return_pct = 0
                
                # Get current price for calculations with None check
                current_price = extract_prediction_field("current_price", prediction_price)
                if current_price is None or current_price <= 0:
                    current_price = extract_prediction_field("entry_price", prediction_price)
                if current_price is None or current_price <= 0:
                    current_price = prediction_price
                
                # Calculate bounds based on confidence and expected return
                confidence_factor = prediction_confidence / 100.0
                
                if expected_return_pct >= 0:  # Positive prediction
                    range_factor = abs(expected_return_pct) * (1 - confidence_factor) * 0.5
                    upper_bound_pct = expected_return_pct + range_factor
                    lower_bound_pct = expected_return_pct - range_factor
                else:  # Negative prediction
                    range_factor = abs(expected_return_pct) * (1 - confidence_factor) * 0.5
                    upper_bound_pct = expected_return_pct + range_factor  # Less negative
                    lower_bound_pct = expected_return_pct - range_factor  # More negative
                
                # Calculate actual price bounds with safety checks
                prediction_upper_bound = float(current_price) * (1 + upper_bound_pct / 100)
                prediction_lower_bound = float(current_price) * (1 + lower_bound_pct / 100)
                
                logger.logger.debug(f"Calculated bounds for {token}: [{prediction_lower_bound:.6f}, {prediction_upper_bound:.6f}]")
            
            # Final validation - ensure bounds are not None
            if prediction_lower_bound is None:
                prediction_lower_bound = float(prediction_price) * 0.95  # 5% below prediction
            if prediction_upper_bound is None:
                prediction_upper_bound = float(prediction_price) * 1.05  # 5% above prediction
            
            # ================================================================
            # 🔧 CALCULATE EXPIRATION TIME 🔧
            # ================================================================
            
            # Calculate expiration time based on timeframe
            if timeframe == "1h":
                expiration_time = datetime.now() + timedelta(hours=1)
            elif timeframe == "24h":
                expiration_time = datetime.now() + timedelta(hours=24)
            elif timeframe == "7d":
                expiration_time = datetime.now() + timedelta(days=7)
            else:
                expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
            
            # ================================================================
            # 🔧 DATABASE INSERTION 🔧
            # ================================================================
            
            cursor.execute("""
                INSERT INTO price_predictions (
                    timestamp, token, timeframe, prediction_type,
                    prediction_value, confidence_level, lower_bound, upper_bound,
                    prediction_rationale, method_weights, model_inputs, technical_signals,
                    expiration_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    "price",
                    float(prediction_price),
                    float(prediction_confidence),
                    float(prediction_lower_bound),
                    float(prediction_upper_bound),
                    rationale,
                    model_weights,
                    model_inputs,
                    key_factors,
                    expiration_time
                ))
                
            conn.commit()
            prediction_id = cursor.lastrowid
            logger.logger.debug(f"Stored {timeframe} prediction for {token} with ID {prediction_id}")
            
            # ================================================================
            # 🔧 STORE IN SPECIALIZED TABLES 🔧
            # ================================================================
            
            # Store Claude prediction if it was used
            if prediction_data.get("model_weights", {}).get("claude_enhanced", 0) > 0:
                self._store_claude_prediction(token, prediction_data, timeframe)
                
            # Store technical analysis if available
            if "inputs" in prediction_data and "technical_analysis" in prediction_data["inputs"]:
                self._store_technical_indicators(token, prediction_data["inputs"]["technical_analysis"], timeframe)
                
            # Store statistical forecast if available
            if "inputs" in prediction_data and "statistical_forecast" in prediction_data["inputs"]:
                self._store_statistical_forecast(token, prediction_data["inputs"]["statistical_forecast"], timeframe)
                
            # Store ML forecast if available
            if "inputs" in prediction_data and "ml_forecast" in prediction_data["inputs"]:
                self._store_ml_forecast(token, prediction_data["inputs"]["ml_forecast"], timeframe)
                
            # Update timeframe metrics
            self._update_timeframe_metrics(token, timeframe, prediction_data)
            
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            if conn:
                conn.rollback()
            
        return prediction_id
    
    def _store_technical_indicators(self, token: str, technical_analysis: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store technical indicator data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract indicator values
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            signals = technical_analysis.get("signals", {})
            
            # Extract individual indicators if available
            indicators = technical_analysis.get("indicators", {})
            
            # Get RSI
            rsi = indicators.get("rsi", None)
            
            # Get MACD
            macd = indicators.get("macd", {})
            macd_line = macd.get("macd_line", None)
            signal_line = macd.get("signal_line", None)
            histogram = macd.get("histogram", None)
            
            # Get Bollinger Bands
            bb = indicators.get("bollinger_bands", {})
            bb_upper = bb.get("upper", None)
            bb_middle = bb.get("middle", None)
            bb_lower = bb.get("lower", None)
            
            # Get Stochastic
            stoch = indicators.get("stochastic", {})
            stoch_k = stoch.get("k", None)
            stoch_d = stoch.get("d", None)
            
            # Get OBV
            obv = indicators.get("obv", None)
            
            # Get ADX
            adx = indicators.get("adx", None)
            
            # Get additional timeframe-specific indicators
            ichimoku_data = json.dumps(indicators.get("ichimoku", {}))
            pivot_points = json.dumps(indicators.get("pivot_points", {}))
            
            # Get volatility
            volatility = technical_analysis.get("volatility", None)
            
            # Store in database
            cursor.execute("""
                INSERT INTO technical_indicators (
                    timestamp, token, timeframe, rsi, macd_line, macd_signal, 
                    macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, obv, adx, ichimoku_data, pivot_points,
                    overall_trend, trend_strength, volatility, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                rsi,
                macd_line,
                signal_line,
                histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                stoch_k,
                stoch_d,
                obv,
                adx,
                ichimoku_data,
                pivot_points,
                overall_trend,
                trend_strength,
                volatility,
                json.dumps(technical_analysis)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Technical Indicators - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_statistical_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store statistical forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type from model_info if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "ARIMA")
            
            # Extract model parameters if available
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO statistical_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    model_parameters, input_data_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                model_parameters,
                "{}"   # Input data summary (empty for now)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Statistical Forecast - {token} ({timeframe})", str(e))
            conn.rollback()

    def _store_ml_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store machine learning forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type and parameters if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "RandomForest")
            
            # Extract feature importance if available
            feature_importance = json.dumps(forecast_data.get("feature_importance", {}))
            
            # Store model parameters
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO ml_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    feature_importance, model_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                feature_importance,
                model_parameters
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store ML Forecast - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_claude_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store Claude AI prediction data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            
            # Default Claude model
            claude_model = "claude-3-5-sonnet-20240620"
            
            # Store inputs if available
            input_data = json.dumps(prediction_data.get("inputs", {}))
            
            # Store in database
            cursor.execute("""
                INSERT INTO claude_predictions (
                    timestamp, token, timeframe, claude_model,
                    prediction_value, confidence_level, sentiment,
                    rationale, key_factors, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                claude_model,
                prediction.get("price", 0),
                prediction.get("confidence", 70),
                sentiment,
                rationale,
                key_factors,
                input_data
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Claude Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _update_timeframe_metrics(self, token: str, timeframe: str, prediction_data: Dict[str, Any]) -> None:
        """Update timeframe metrics based on new prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get current metrics for this token and timeframe
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            # Get prediction performance
            performance = self.get_prediction_performance(token=token, timeframe=timeframe)
            
            if performance:
                avg_accuracy = performance[0]["accuracy_rate"]
                total_count = performance[0]["total_predictions"]
                correct_count = performance[0]["correct_predictions"]
            else:
                avg_accuracy = 0
                total_count = 0
                correct_count = 0
            
            # Extract model weights
            model_weights = prediction_data.get("model_weights", {})
            
            # Determine best model
            if model_weights:
                best_model = max(model_weights.items(), key=lambda x: x[1])[0]
            else:
                best_model = "unknown"
            
            if metrics:
                # Update existing metrics
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        model_weights = ?,
                        best_model = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Insert new metrics
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_active_predictions(self, token: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active (non-expired) predictions
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM price_predictions
                WHERE expiration_time > datetime('now')
            """
            params = []
            
            if token:
                query += " AND token = ?"
                params.append(token)
                
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Active Predictions", str(e))
            return []

    def get_all_timeframe_predictions(self, token: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get active predictions for a token across all timeframes
        Returns a dictionary of predictions keyed by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get all supported timeframes
            timeframes = ["1h", "24h", "7d"]
            
            result = {}
            
            for tf in timeframes:
                query = """
                    SELECT * FROM price_predictions
                    WHERE token = ? AND timeframe = ? AND expiration_time > datetime('now')
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                
                cursor.execute(query, (token, tf))
                prediction = cursor.fetchone()
                
                if prediction:
                    # Convert to dict and parse JSON fields
                    pred_dict = dict(prediction)
                    pred_dict["method_weights"] = json.loads(pred_dict["method_weights"]) if pred_dict["method_weights"] else {}
                    pred_dict["model_inputs"] = json.loads(pred_dict["model_inputs"]) if pred_dict["model_inputs"] else {}
                    pred_dict["technical_signals"] = json.loads(pred_dict["technical_signals"]) if pred_dict["technical_signals"] else []
                    
                    result[tf] = pred_dict
                
            return result
            
        except Exception as e:
            logger.log_error(f"Get All Timeframe Predictions - {token}", str(e))
            return {}

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get a prediction by its ID"""
        conn, cursor = self._get_connection()
        
        try:
            cursor.execute("""
                SELECT * FROM price_predictions
                WHERE id = ?
            """, (prediction_id,))
            
            prediction = cursor.fetchone()
            if not prediction:
                return None
                
            # Convert to dict and parse JSON fields
            result = dict(prediction)
            result["method_weights"] = json.loads(result["method_weights"]) if result["method_weights"] else {}
            result["model_inputs"] = json.loads(result["model_inputs"]) if result["model_inputs"] else {}
            result["technical_signals"] = json.loads(result["technical_signals"]) if result["technical_signals"] else []
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Prediction By ID - {prediction_id}", str(e))
            return None

    def get_expired_unevaluated_predictions(self, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all expired predictions that haven't been evaluated yet
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT p.* FROM price_predictions p
                LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.expiration_time <= datetime('now')
                AND o.id IS NULL
            """
            
            params = []
            
            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY p.timeframe ASC, p.expiration_time ASC"
            
            cursor.execute(query, params)
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Expired Unevaluated Predictions", str(e))
            return []

    def record_prediction_outcome(self, prediction_id: int, actual_price: float) -> bool:
        """Record the outcome of a prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get the prediction details
            prediction = self.get_prediction_by_id(prediction_id)
            if not prediction:
                return False
                
            # Calculate accuracy metrics
            prediction_value = prediction["prediction_value"]
            lower_bound = prediction["lower_bound"]
            upper_bound = prediction["upper_bound"]
            timeframe = prediction["timeframe"]
            
            # Percentage accuracy (how close the prediction was)
            price_diff = abs(actual_price - prediction_value)
            accuracy_percentage = (1 - (price_diff / prediction_value)) * 100 if prediction_value > 0 else 0
            
            # Whether the actual price fell within the predicted range
            was_correct = lower_bound <= actual_price <= upper_bound
            
            # Deviation from prediction (for tracking bias)
            deviation = ((actual_price / prediction_value) - 1) * 100 if prediction_value > 0 else 0
            
            # Get market conditions at evaluation time
            market_data = self.get_recent_market_data(prediction["token"], 1)
            market_conditions = json.dumps({
                "evaluation_time": datetime.now().isoformat(),
                "token": prediction["token"],
                "market_data": market_data[:1] if market_data else []
            })
            
            # Store the outcome
            cursor.execute("""
                INSERT INTO prediction_outcomes (
                    prediction_id, actual_outcome, accuracy_percentage,
                    was_correct, evaluation_time, deviation_from_prediction,
                    market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                actual_price,
                accuracy_percentage,
                1 if was_correct else 0,
                datetime.now(),
                deviation,
                market_conditions
            ))
            
            # Update the performance summary
            token = prediction["token"]
            prediction_type = prediction["prediction_type"]
            
            self._update_prediction_performance(token, timeframe, prediction_type, was_correct, abs(deviation))
            
            # Update timeframe metrics
            self._update_timeframe_outcome_metrics(token, timeframe, was_correct, accuracy_percentage)
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.log_error(f"Record Prediction Outcome - {prediction_id}", str(e))
            conn.rollback()
            return False

    def _update_prediction_performance(self, token: str, timeframe: str, prediction_type: str, was_correct: bool, deviation: float) -> None:
        """Update prediction performance summary"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if performance record exists
            cursor.execute("""
                SELECT * FROM prediction_performance
                WHERE token = ? AND timeframe = ? AND prediction_type = ?
            """, (token, timeframe, prediction_type))
            
            performance = cursor.fetchone()
            
            if performance:
                # Update existing record
                performance_dict = dict(performance)
                total_predictions = performance_dict["total_predictions"] + 1
                correct_predictions = performance_dict["correct_predictions"] + (1 if was_correct else 0)
                accuracy_rate = (correct_predictions / total_predictions) * 100
                
                # Update average deviation (weighted average)
                avg_deviation = (performance_dict["avg_deviation"] * performance_dict["total_predictions"] + deviation) / total_predictions
                
                cursor.execute("""
                    UPDATE prediction_performance
                    SET total_predictions = ?,
                        correct_predictions = ?,
                        accuracy_rate = ?,
                        avg_deviation = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    total_predictions,
                    correct_predictions,
                    accuracy_rate,
                    avg_deviation,
                    datetime.now(),
                    performance_dict["id"]
                ))
                
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO prediction_performance (
                        token, timeframe, prediction_type, total_predictions,
                        correct_predictions, accuracy_rate, avg_deviation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token,
                    timeframe,
                    prediction_type,
                    1,
                    1 if was_correct else 0,
                    100 if was_correct else 0,
                    deviation,
                    datetime.now()
                ))
                
        except Exception as e:
            logger.log_error(f"Update Prediction Performance - {token}", str(e))
            raise

    def _update_timeframe_outcome_metrics(self, token: str, timeframe: str, was_correct: bool, accuracy_percentage: float) -> None:
        """Update timeframe metrics with outcome data"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if metrics record exists
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            if metrics:
                # Update existing metrics
                metrics_dict = dict(metrics)
                total_count = metrics_dict["total_count"] + 1
                correct_count = metrics_dict["correct_count"] + (1 if was_correct else 0)
                
                # Recalculate average accuracy with new data point
                # Use weighted average based on number of predictions
                old_weight = (total_count - 1) / total_count
                new_weight = 1 / total_count
                avg_accuracy = (metrics_dict["avg_accuracy"] * old_weight) + (accuracy_percentage * new_weight)
                
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Should not happen normally, but create metrics if missing
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    accuracy_percentage,
                    1,
                    1 if was_correct else 0,
                    "{}",
                    "unknown",
                    datetime.now()
                ))
                
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Outcome Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_prediction_performance(self, token: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get prediction performance statistics
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = "SELECT * FROM prediction_performance"
            params = []
            
            if token or timeframe:
                query += " WHERE "
                
            if token:
                query += "token = ?"
                params.append(token)
                
            if token and timeframe:
                query += " AND "
                
            if timeframe:
                query += "timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error("Get Prediction Performance", str(e))
            return []

    def get_timeframe_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summary across all timeframes
        Returns a dictionary with metrics for each timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get performance for each timeframe across all tokens
            timeframes = ["1h", "24h", "7d"]
            result = {}
            
            for tf in timeframes:
                cursor.execute("""
                    SELECT 
                        AVG(accuracy_rate) as avg_accuracy,
                        SUM(total_predictions) as total_predictions,
                        SUM(correct_predictions) as correct_predictions,
                        AVG(avg_deviation) as avg_deviation
                    FROM prediction_performance
                    WHERE timeframe = ?
                """, (tf,))
                
                stats = cursor.fetchone()
                
                if stats:
                    stats_dict = dict(stats)
                    
                    # Calculate overall accuracy
                    total = stats_dict["total_predictions"] or 0
                    correct = stats_dict["correct_predictions"] or 0
                    accuracy = (correct / total * 100) if total > 0 else 0
                    
                    result[tf] = {
                        "accuracy": accuracy,
                        "total_predictions": total,
                        "correct_predictions": correct,
                        "avg_deviation": stats_dict["avg_deviation"] or 0
                    }
                    
                    # Get best performing token for this timeframe
                    cursor.execute("""
                        SELECT token, accuracy_rate, total_predictions
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate DESC
                        LIMIT 1
                    """, (tf,))
                    
                    best_token = cursor.fetchone()
                    if best_token:
                        result[tf]["best_token"] = {
                            "token": best_token["token"],
                            "accuracy": best_token["accuracy_rate"],
                            "predictions": best_token["total_predictions"]
                        }
                    
                    # Get worst performing token for this timeframe
                    cursor.execute("""
                        SELECT token, accuracy_rate, total_predictions
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate ASC
                        LIMIT 1
                    """, (tf,))
                    
                    worst_token = cursor.fetchone()
                    if worst_token:
                        result[tf]["worst_token"] = {
                            "token": worst_token["token"],
                            "accuracy": worst_token["accuracy_rate"],
                            "predictions": worst_token["total_predictions"]
                        }
            
            return result
            
        except Exception as e:
            logger.log_error("Get Timeframe Performance Summary", str(e))
            return {}

    def get_recent_prediction_outcomes(self, token: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent prediction outcomes with their original predictions
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT p.*, o.actual_outcome, o.accuracy_percentage, o.was_correct, 
                       o.evaluation_time, o.deviation_from_prediction
                FROM prediction_outcomes o
                JOIN price_predictions p ON o.prediction_id = p.id
                WHERE 1=1
            """
            params = []
            
            if token:
                query += " AND p.token = ?"
                params.append(token)
                
            query += " ORDER BY o.evaluation_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            outcomes = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for outcome in outcomes:
                outcome["method_weights"] = json.loads(outcome["method_weights"]) if outcome["method_weights"] else {}
                outcome["model_inputs"] = json.loads(outcome["model_inputs"]) if outcome["model_inputs"] else {}
                outcome["technical_signals"] = json.loads(outcome["technical_signals"]) if outcome["technical_signals"] else []
                
            return outcomes
            
        except Exception as e:
            logger.log_error("Get Recent Prediction Outcomes", str(e))
            return []

    def get_timeframe_metrics(self, token: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for different timeframes
        Returns a dictionary with metrics for each timeframe, optionally filtered by token
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM timeframe_metrics
                WHERE 1=1
            """
            params = []
            
            if token:
                query += " AND token = ?"
                params.append(token)
                
            query += " ORDER BY token, timeframe"
            
            cursor.execute(query, params)
            metrics = cursor.fetchall()
            
            result = {}
            
            for metric in metrics:
                metric_dict = dict(metric)
                timeframe = metric_dict["timeframe"]
                
                # Parse JSON fields
                metric_dict["model_weights"] = json.loads(metric_dict["model_weights"]) if metric_dict["model_weights"] else {}
                
                if token:
                    # If filtering by token, return metrics keyed by timeframe
                    result[timeframe] = metric_dict
                else:
                    # If not filtering by token, organize by token then timeframe
                    token_name = metric_dict["token"]
                    if token_name not in result:
                        result[token_name] = {}
                        
                    result[token_name][timeframe] = metric_dict
            
            return result
            
        except Exception as e:
            logger.log_error("Get Timeframe Metrics", str(e))
            return {}
            
    def get_technical_indicators(self, token: str, timeframe: str = "1h", hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent technical indicators for a token and timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            cursor.execute("""
                SELECT * FROM technical_indicators
                WHERE token = ? AND timeframe = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, timeframe, hours))
            
            indicators = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for indicator in indicators:
                indicator["raw_data"] = json.loads(indicator["raw_data"]) if indicator["raw_data"] else {}
                indicator["ichimoku_data"] = json.loads(indicator["ichimoku_data"]) if indicator["ichimoku_data"] else {}
                indicator["pivot_points"] = json.loads(indicator["pivot_points"]) if indicator["pivot_points"] else {}
                
            return indicators
            
        except Exception as e:
            logger.log_error(f"Get Technical Indicators - {token} ({timeframe})", str(e))
            return []
            
    def get_statistical_forecasts(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent statistical forecasts for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM statistical_forecasts
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            forecasts = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for forecast in forecasts:
                forecast["model_parameters"] = json.loads(forecast["model_parameters"]) if forecast["model_parameters"] else {}
                forecast["input_data_summary"] = json.loads(forecast["input_data_summary"]) if forecast["input_data_summary"] else {}
                
            return forecasts
            
        except Exception as e:
            logger.log_error(f"Get Statistical Forecasts - {token}", str(e))
            return []
            
    def get_ml_forecasts(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent machine learning forecasts for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM ml_forecasts
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            forecasts = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for forecast in forecasts:
                forecast["feature_importance"] = json.loads(forecast["feature_importance"]) if forecast["feature_importance"] else {}
                forecast["model_parameters"] = json.loads(forecast["model_parameters"]) if forecast["model_parameters"] else {}
                
            return forecasts
            
        except Exception as e:
            logger.log_error(f"Get ML Forecasts - {token}", str(e))
            return []
            
    def get_claude_predictions(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent Claude AI predictions for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM claude_predictions
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["key_factors"] = json.loads(prediction["key_factors"]) if prediction["key_factors"] else []
                prediction["input_data"] = json.loads(prediction["input_data"]) if prediction["input_data"] else {}
                
            return predictions
            
        except Exception as e:
            logger.log_error(f"Get Claude Predictions - {token}", str(e))
            return []

    def get_prediction_accuracy_by_model(self, timeframe: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Calculate prediction accuracy statistics by model type
        Returns accuracy metrics for different prediction approaches
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Base query for predictions and outcomes
            query = """
                SELECT p.id, p.token, p.timeframe, p.method_weights, 
                    o.was_correct, o.deviation_from_prediction
                FROM price_predictions p
                JOIN prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.timestamp >= datetime('now', '-' || ? || ' days')
            """
            
            # Start with days parameter (convert to string for SQL concatenation)
            params = [str(days)]
            
            # Add timeframe condition if specified
            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Initialize counters for each model type
            model_stats = {
                "technical_analysis": {"correct": 0, "total": 0, "deviation_sum": 0},
                "statistical_models": {"correct": 0, "total": 0, "deviation_sum": 0},
                "machine_learning": {"correct": 0, "total": 0, "deviation_sum": 0},
                "claude_enhanced": {"correct": 0, "total": 0, "deviation_sum": 0},
                "combined": {"correct": 0, "total": 0, "deviation_sum": 0}
            }
            
            # Add timeframe-specific counters
            timeframe_stats = {}
            
            # Process results
            for row in results:
                # Parse model weights
                weights = json.loads(row["method_weights"]) if row["method_weights"] else {}
                was_correct = row["was_correct"] == 1
                deviation = abs(row["deviation_from_prediction"])
                row_timeframe = row["timeframe"]
                
                # Update combined stats
                model_stats["combined"]["total"] += 1
                if was_correct:
                    model_stats["combined"]["correct"] += 1
                model_stats["combined"]["deviation_sum"] += deviation
                
                # Update timeframe stats
                if row_timeframe not in timeframe_stats:
                    timeframe_stats[row_timeframe] = {"correct": 0, "total": 0, "deviation_sum": 0}
                
                timeframe_stats[row_timeframe]["total"] += 1
                if was_correct:
                    timeframe_stats[row_timeframe]["correct"] += 1
                timeframe_stats[row_timeframe]["deviation_sum"] += deviation
                
                # Determine primary model based on weights
                if weights:
                    primary_model = max(weights.items(), key=lambda x: x[1])[0]
                    
                    # Update model-specific stats
                    if primary_model in model_stats:
                        model_stats[primary_model]["total"] += 1
                        if was_correct:
                            model_stats[primary_model]["correct"] += 1
                        model_stats[primary_model]["deviation_sum"] += deviation
                    
                    # Update stats for all models used in this prediction
                    for model, weight in weights.items():
                        if model in model_stats and weight > 0:
                            # Add fractional count based on weight
                            model_stats[model]["total"] += weight
                            if was_correct:
                                model_stats[model]["correct"] += weight
                            model_stats[model]["deviation_sum"] += deviation * weight
            
            # Calculate accuracy rates and average deviations
            model_results = {}
            for model, stats in model_stats.items():
                if stats["total"] > 0:
                    accuracy = (stats["correct"] / stats["total"]) * 100
                    avg_deviation = stats["deviation_sum"] / stats["total"]
                    
                    model_results[model] = {
                        "accuracy_rate": accuracy,
                        "avg_deviation": avg_deviation,
                        "total_predictions": stats["total"]
                    }
            
            # Calculate timeframe statistics
            tf_results = {}
            for tf, stats in timeframe_stats.items():
                if stats["total"] > 0:
                    accuracy = (stats["correct"] / stats["total"]) * 100
                    avg_deviation = stats["deviation_sum"] / stats["total"]
                    
                    tf_results[tf] = {
                        "accuracy_rate": accuracy,
                        "avg_deviation": avg_deviation,
                        "total_predictions": stats["total"]
                    }
            
            # Combine results
            return {
                "models": model_results,
                "timeframes": tf_results,
                "total_predictions": model_stats["combined"]["total"],
                "overall_accuracy": (model_stats["combined"]["correct"] / model_stats["combined"]["total"] * 100) 
                                if model_stats["combined"]["total"] > 0 else 0
            }
            
        except Exception as e:
            logger.log_error("Get Prediction Accuracy By Model", str(e))
            return {}
    
    def get_prediction_comparison_across_timeframes(self, token: str, limit: int = 5) -> Dict[str, Any]:
        """
        Compare prediction performance across different timeframes for a specific token
        Returns latest predictions and their outcomes for each timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get performance summary for each timeframe
            timeframes = ["1h", "24h", "7d"]
            result = {
                "summary": {},
                "recent_predictions": {}
            }
            
            # Get performance stats for each timeframe
            for tf in timeframes:
                cursor.execute("""
                    SELECT * FROM prediction_performance
                    WHERE token = ? AND timeframe = ?
                """, (token, tf))
                
                performance = cursor.fetchone()
                
                if performance:
                    perf_dict = dict(performance)
                    result["summary"][tf] = {
                        "accuracy": perf_dict["accuracy_rate"],
                        "total_predictions": perf_dict["total_predictions"],
                        "correct_predictions": perf_dict["correct_predictions"],
                        "avg_deviation": perf_dict["avg_deviation"]
                    }
                
                # Get recent predictions for this timeframe
                cursor.execute("""
                    SELECT p.*, o.actual_outcome, o.was_correct, o.deviation_from_prediction
                    FROM price_predictions p
                    LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.token = ? AND p.timeframe = ?
                    ORDER BY p.timestamp DESC
                    LIMIT ?
                """, (token, tf, limit))
                
                predictions = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for pred in predictions:
                    pred["method_weights"] = json.loads(pred["method_weights"]) if pred["method_weights"] else {}
                    pred["technical_signals"] = json.loads(pred["technical_signals"]) if pred["technical_signals"] else []
                
                result["recent_predictions"][tf] = predictions
            
            # Add overall statistics
            if result["summary"]:
                total_correct = sum(tf_stats.get("correct_predictions", 0) for tf_stats in result["summary"].values())
                total_predictions = sum(tf_stats.get("total_predictions", 0) for tf_stats in result["summary"].values())
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                else:
                    overall_accuracy = 0
                    
                result["overall"] = {
                    "accuracy": overall_accuracy,
                    "total_predictions": total_predictions,
                    "total_correct": total_correct
                }
                
                # Find best timeframe for this token
                best_timeframe = max(result["summary"].items(), key=lambda x: x[1]["accuracy"])
                result["best_timeframe"] = {
                    "timeframe": best_timeframe[0],
                    "accuracy": best_timeframe[1]["accuracy"]
                }
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Prediction Comparison Across Timeframes - {token}", str(e))
            return {}

    #########################
    # DATABASE MAINTENANCE METHODS
    #########################
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data to prevent database bloat
        Returns count of deleted records by table
        """
        conn, cursor = self._get_connection()
        
        tables_to_clean = [
            "market_data",
            "posted_content",
            "mood_history",
            "smart_money_indicators",
            "token_market_comparison",
            "token_correlations",
            "generic_json_data",
            "technical_indicators",
            "statistical_forecasts",
            "ml_forecasts",
            "claude_predictions"
        ]
        
        deleted_counts = {}
        
        try:
            for table in tables_to_clean:
                # Keep prediction-related tables longer
                retention_days = days_to_keep * 2 if table in ["price_predictions", "prediction_outcomes"] else days_to_keep
                
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (retention_days,))
                
                deleted_counts[table] = cursor.rowcount
                
            # Special handling for evaluated predictions
            cursor.execute("""
                DELETE FROM price_predictions
                WHERE id IN (
                    SELECT p.id
                    FROM price_predictions p
                    JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.timestamp < datetime('now', '-' || ? || ' days')
                )
            """, (days_to_keep * 2,))
            
            deleted_counts["price_predictions"] = cursor.rowcount
            
            conn.commit()
            logger.logger.info(f"Database cleanup completed: {deleted_counts}")
            
            return deleted_counts
            
        except Exception as e:
            logger.log_error("Database Cleanup", str(e))
            conn.rollback()
            return {}
            
    def optimize_database(self) -> bool:
        """
        Optimize database performance by running VACUUM and ANALYZE
        """
        conn, cursor = self._get_connection()
        
        try:
            # Backup current connection settings
            old_isolation_level = conn.isolation_level
            
            # Set isolation level to None for VACUUM
            conn.isolation_level = None
            
            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            # Restore original isolation level
            conn.isolation_level = old_isolation_level
            
            logger.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.log_error("Database Optimization", str(e))
            return False
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including table sizes and row counts
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row["name"] for row in cursor.fetchall()]
            
            stats = {
                "tables": {},
                "total_rows": 0,
                "last_optimized": None
            }
            
            # Get row count for each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                row_count = cursor.fetchone()["count"]
                
                # Get most recent timestamp if available
                try:
                    cursor.execute(f"SELECT MAX(timestamp) as last_update FROM {table}")
                    last_update = cursor.fetchone()["last_update"]
                except:
                    last_update = None
                
                stats["tables"][table] = {
                    "rows": row_count,
                    "last_update": last_update
                }
                
                stats["total_rows"] += row_count
                
            # Get database size (approximate)
            stats["database_size_kb"] = os.path.getsize(self.db_path) / 1024
            
            # Get last VACUUM time (if available in generic_json_data)
            cursor.execute("""
                SELECT timestamp FROM generic_json_data
                WHERE data_type = 'database_maintenance'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            last_maintenance = cursor.fetchone()
            if last_maintenance:
                stats["last_optimized"] = last_maintenance["timestamp"]
                
            return stats
            
        except Exception as e:
            logger.log_error("Get Database Stats", str(e))
            return {"error": str(e)}
            
    def get_timeframe_prediction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of predictions and accuracy across all timeframes
        """
        conn, cursor = self._get_connection()
        
        try:
            summary = {
                "timeframes": {},
                "total": {
                    "predictions": 0,
                    "correct": 0,
                    "accuracy": 0
                }
            }
            
            # Get stats for each timeframe
            for timeframe in ["1h", "24h", "7d"]:
                # Get overall stats
                cursor.execute("""
                    SELECT 
                        SUM(total_predictions) as total,
                        SUM(correct_predictions) as correct
                    FROM prediction_performance
                    WHERE timeframe = ?
                """, (timeframe,))
                
                stats = cursor.fetchone()
                
                if stats and stats["total"]:
                    total = stats["total"]
                    correct = stats["correct"]
                    accuracy = (correct / total * 100) if total > 0 else 0
                    
                    summary["timeframes"][timeframe] = {
                        "predictions": total,
                        "correct": correct,
                        "accuracy": accuracy
                    }
                    
                    # Get top performing token
                    cursor.execute("""
                        SELECT token, accuracy_rate
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate DESC
                        LIMIT 1
                    """, (timeframe,))
                    
                    best = cursor.fetchone()
                    if best:
                        summary["timeframes"][timeframe]["best_token"] = {
                            "token": best["token"],
                            "accuracy": best["accuracy_rate"]
                        }
                        
                    # Update totals
                    summary["total"]["predictions"] += total
                    summary["total"]["correct"] += correct
            
            # Calculate overall accuracy
            if summary["total"]["predictions"] > 0:
                summary["total"]["accuracy"] = (summary["total"]["correct"] / summary["total"]["predictions"]) * 100
                
            # Add prediction counts by timeframe
            cursor.execute("""
                SELECT timeframe, COUNT(*) as count
                FROM price_predictions
                GROUP BY timeframe
            """)
            
            counts = cursor.fetchall()
            for row in counts:
                tf = row["timeframe"]
                if tf in summary["timeframes"]:
                    summary["timeframes"][tf]["total_stored"] = row["count"]
                    
            # Add active prediction counts
            cursor.execute("""
                SELECT timeframe, COUNT(*) as count
                FROM price_predictions
                WHERE expiration_time > datetime('now')
                GROUP BY timeframe
            """)
            
            active_counts = cursor.fetchall()
            for row in active_counts:
                tf = row["timeframe"]
                if tf in summary["timeframes"]:
                    summary["timeframes"][tf]["active"] = row["count"]
                    
            return summary
            
        except Exception as e:
            logger.log_error("Get Timeframe Prediction Summary", str(e))
            return {}
    