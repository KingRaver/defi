# Tokenetics Architecture Documentation

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Technical Stack](#technical-stack)
- [Module Details](#module-details)
- [Database Schema](#database-schema)
- [API Integrations](#api-integrations)
- [Trading System Architecture](#trading-system-architecture)
- [Security Architecture](#security-architecture)
- [Scalability & Performance](#scalability--performance)

---

## Overview

Tokenetics is a production-ready autonomous DeFi trading platform with advanced AI capabilities. The system consists of ~70,000 lines of Python code organized into specialized modules handling market analysis, prediction, social intelligence, and blockchain integration.

**Current Status:** Final testing and prediction refinement phase before live trading deployment.

### Architecture Principles

1. **Modular Design** - Separation of concerns across distinct functional modules
2. **Fail-Safe First** - Multiple safety mechanisms and emergency protocols
3. **Data-Driven** - All decisions backed by multi-source data and validation
4. **API Abstraction** - Clean interfaces for swappable service providers
5. **Production Ready** - Complete logging, error handling, monitoring

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
│                  (Social Media / Manual Controls)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │   bot.py         │  │ integrated_      │                     │
│  │   (Social/       │  │ trading_bot.py   │                     │
│  │    Analysis)     │  │ (Trading)        │                     │
│  └──────────────────┘  └──────────────────┘                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  AI/ML LAYER   │  │ ANALYSIS LAYER │  │ SOCIAL LAYER   │
│                │  │                │  │                │
│ • Claude AI    │  │ • Technical    │  │ • Sentiment    │
│ • LSTM/ARIMA   │  │   Indicators   │  │   Analysis     │
│ • Ensemble     │  │ • Pattern      │  │ • Trending     │
│   Models       │  │   Recognition  │  │   Topics       │
│ • Confidence   │  │ • Signal Gen   │  │ • Community    │
│   Scoring      │  │ • Multi-TF     │  │   Engagement   │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ BLOCKCHAIN     │  │  DATA       │  │  EXTERNAL APIs  │
│ LAYER (Ready)  │  │  LAYER      │  │                 │
│                │  │             │  │                 │
│ • Multi-chain  │  │ • SQLite DB │  │ • CoinGecko     │
│ • Web3.py      │  │ • Google    │  │ • Claude API    │
│ • DEX Integ.   │  │   Sheets    │  │ • Twitter/X     │
│ • Wallet Mgmt  │  │ • Logging   │  │ • RPC Nodes     │
│ • Gas Optim.   │  │ • Cache     │  │ • DEX Protocols │
└────────────────┘  └─────────────┘  └─────────────────┘
```

### Component Interaction Flow

```
Market Data → Technical Analysis → Prediction Engine → AI Decision → [Trading Execution]
     ↓              ↓                      ↓                ↓              ↓
   Cache     Signal Generation      Confidence Score   Content Gen    Position Mgmt
     ↓              ↓                      ↓                ↓              ↓
  Database    Multi-Indicator        Ensemble Vote    Social Post    Risk Control
               Confluence
```

---

## Directory Structure

```
defi/
├── .env                              # Environment configuration (secrets)
├── .gitignore                        # Git ignore patterns
├── README.md                         # Project overview & quick start
├── ARCHITECTURE.md                   # This file - system architecture
├── STRUCTURE.md                      # Directory structure details
├── requirements.txt                  # Python dependencies
├── __init__.py                       # Root package initialization
│
├── src/                              # Source code (70,000+ lines)
│   │
│   ├── # Main Orchestration
│   ├── bot.py                        # Main bot (social media + analysis)
│   ├── config.py                     # Configuration management
│   ├── integrated_trading_bot.py     # Autonomous trading system (8,750 lines)
│   │
│   ├── # AI & Machine Learning
│   ├── llm_provider.py               # LLM abstraction (Claude/OpenAI/Mistral)
│   ├── prediction_engine.py          # ARIMA/LSTM/Ensemble predictions (15,924 lines)
│   ├── mood_config.py                # Market sentiment determination
│   ├── meme_phrases.py               # Content generation library (2,136 lines)
│   │
│   ├── # Technical Analysis Stack
│   ├── technical_system.py           # System orchestration (5,228 lines)
│   ├── technical_indicators.py       # 50+ indicators (2,506 lines)
│   ├── technical_signals.py          # Signal generation (5,747 lines)
│   ├── technical_portfolio.py        # Portfolio management (2,321 lines)
│   ├── technical_core.py             # Core TA logic (1,266 lines)
│   ├── technical_calculations.py     # Math engines (1,695 lines)
│   ├── technical_foundation.py       # M4 optimization (995 lines)
│   ├── technical_integration.py      # Compatibility layer (1,887 lines)
│   │
│   ├── # Social Media Intelligence
│   ├── timeline_scraper.py           # X/Twitter scraping (2,596 lines)
│   ├── reply_handler.py              # Intelligent replies (2,252 lines)
│   ├── content_analyzer.py           # Sentiment analysis (2,401 lines)
│   ├── warpcast_handler.py           # Warpcast integration (placeholder)
│   │
│   ├── # Data & APIs
│   ├── database.py                   # SQLite ORM (5,420 lines)
│   ├── coingecko_handler.py          # CoinGecko API (2,016 lines)
│   ├── multi_chain_manager.py        # Web3 multi-chain (1,266 lines)
│   │
│   ├── # Utilities
│   ├── datetime_utils.py             # Timezone-aware datetime handling
│   │
│   └── utils/
│       ├── __init__.py
│       ├── browser.py                # Selenium WebDriver setup
│       ├── logger.py                 # Centralized logging
│       └── sheets_handler.py         # Google Sheets API
│
├── data/                             # Data storage
│   ├── crypto_history.db             # Main SQLite database
│   └── backup/                       # Database backups
│       └── crypto_history.db.bak     # Automated backups
│
├── logs/                             # Application logs
│   ├── analysis/
│   │   └── market_analysis.log       # Market analysis logs
│   ├── billion_dollar_system/        # Legacy logs
│   ├── claude.log                    # Claude API interactions
│   ├── claude_api.log                # Claude API details
│   ├── coingecko.log                 # CoinGecko data logs
│   ├── coingecko_api.log             # CoinGecko API calls
│   ├── google_sheets_api.log         # Google Sheets operations
│   ├── eth_btc_correlation.log       # Correlation analysis
│   └── trading_system_*.log          # Trading system logs
│
└── venv/                             # Python virtual environment
```

---

## Core Components

### 1. Orchestration Layer

#### **bot.py** - Main Social Media & Analysis Bot
- **Purpose**: Primary orchestration for analysis and social engagement
- **Responsibilities**:
  - 5-minute analysis cycles
  - Market data fetching (CoinGecko)
  - Technical analysis execution
  - Prediction generation
  - Social media posting (X/Twitter)
  - Timeline scraping and reply generation
  - Database persistence
- **Key Classes**: `TradingBot`
- **Dependencies**: All major subsystems

#### **integrated_trading_bot.py** - Autonomous Trading System
- **Purpose**: Complete DeFi trading automation (ready for deployment)
- **Responsibilities**:
  - Trade signal generation
  - Position management (entry/exit)
  - Risk management and position sizing
  - Multi-chain trade execution
  - Portfolio tracking
  - Performance analytics
  - Emergency protocols
- **Key Classes**:
  - `IntegratedTradingBot`
  - `RiskManager`
  - `TradingStrategy`
  - `MarketAnalyzer`
  - `SecureWalletManager`
  - `PerformanceTracker`
- **Status**: Implemented but not activated (awaiting prediction refinement)

#### **config.py** - Configuration Management
- **Purpose**: Centralized configuration and constants
- **Contains**:
  - Token tracking (13 cryptocurrencies)
  - API endpoints and keys
  - Trading parameters
  - Tweet constraints
  - Prediction timeframes
  - Database paths
  - Network configurations

---

### 2. AI & Machine Learning Layer

#### **llm_provider.py** - LLM Abstraction
- **Purpose**: Unified interface for multiple LLM providers
- **Supported Providers**:
  - Anthropic Claude (primary - claude-3-5-sonnet)
  - OpenAI (GPT-4, commented out)
  - Mistral AI (commented out)
  - Groq (commented out)
- **Capabilities**:
  - Market analysis generation
  - Content creation for social media
  - Reply generation
  - Context-aware responses

#### **prediction_engine.py** - ML Prediction System (15,924 lines)
- **Purpose**: Multi-model price prediction with confidence scoring
- **Models Implemented**:
  - **ARIMA**: AutoRegressive Integrated Moving Average
    - Stationarity testing (ADF, KPSS)
    - Auto parameter tuning (p, d, q)
    - Confidence intervals
  - **LSTM**: Long Short-Term Memory networks
    - Multi-layer architecture (64-128-64 units)
    - Dropout regularization
    - 60-period lookback window
  - **Ensemble**: Multi-model voting
    - Weighted averaging
    - Outlier rejection
    - Confidence calibration
- **Features**:
  - Prediction timeframes: 1h, 24h, 7d
  - Confidence scoring
  - Accuracy tracking and validation
  - Walk-forward testing
  - Continuous learning

#### **mood_config.py** - Market Sentiment Analysis
- **Purpose**: Determine market mood states
- **Mood States**:
  - Bullish
  - Bearish
  - Neutral
  - Volatile
  - Recovering
- **Analysis Factors**:
  - Price trends
  - Technical indicators
  - Volume patterns
  - Volatility measures
  - Social sentiment

#### **meme_phrases.py** - Content Generation Library (2,136 lines)
- **Purpose**: Contextual phrase library for authentic engagement
- **Contains**:
  - 2,100+ phrases categorized by mood and context
  - Crypto slang and terminology
  - Technical analysis commentary
  - Market condition descriptions
  - Community engagement phrases

---

### 3. Technical Analysis Layer

#### **technical_system.py** - System Orchestration (5,228 lines)
- **Purpose**: High-level coordination of technical analysis
- **Responsibilities**:
  - Multi-timeframe coordination
  - Indicator aggregation
  - Signal consolidation
  - Security level management
  - Performance optimization

#### **technical_indicators.py** - Indicator Engine (2,506 lines)
- **Purpose**: Modular technical indicator calculations
- **Indicators** (50+):

  **Momentum:**
  - RSI (Relative Strength Index)
  - Stochastic Oscillator
  - Williams %R
  - CCI (Commodity Channel Index)
  - ROC (Rate of Change)
  - MFI (Money Flow Index)

  **Trend:**
  - MACD (Moving Average Convergence Divergence)
  - ADX (Average Directional Index)
  - Ichimoku Cloud
  - Parabolic SAR
  - DMI (Directional Movement Index)
  - TRIX

  **Volatility:**
  - Bollinger Bands
  - ATR (Average True Range)
  - Keltner Channels
  - Donchian Channels
  - Standard Deviation

  **Volume:**
  - OBV (On-Balance Volume)
  - VWAP (Volume-Weighted Average Price)
  - Accumulation/Distribution
  - Chaikin Money Flow
  - Volume Oscillator

#### **technical_signals.py** - Signal Generation (5,747 lines)
- **Purpose**: Generate trading signals from indicators
- **Signal Types**:
  - Buy/Sell signals
  - Strength scores (0-100)
  - Confluence detection (3+ indicators)
  - Multi-timeframe confirmation
  - Risk level assessment

#### **technical_portfolio.py** - Portfolio Management (2,321 lines)
- **Purpose**: Portfolio tracking and wealth targets
- **Features**:
  - Position tracking
  - P&L calculation
  - Allocation optimization
  - Performance metrics
  - Risk exposure monitoring

#### **technical_core.py** - Core TA Logic (1,266 lines)
- **Purpose**: Fundamental TA calculations and classes
- **Provides**:
  - Base calculation methods
  - Data structure definitions
  - Common TA utilities

#### **technical_calculations.py** - Math Engines (1,695 lines)
- **Purpose**: Low-level mathematical calculations
- **Features**:
  - Numba JIT-compiled functions
  - Array operations
  - Statistical computations
  - Performance-critical calculations

#### **technical_foundation.py** - M4 Optimization (995 lines)
- **Purpose**: Hardware acceleration and optimization
- **Capabilities**:
  - M4 Ultra chip detection
  - Numba JIT compilation
  - Memory optimization
  - Parallel processing
  - Cache management

#### **technical_integration.py** - Compatibility Layer (1,887 lines)
- **Purpose**: Integration between TA modules
- **Provides**:
  - Cross-module communication
  - Data format conversion
  - API standardization
  - Backwards compatibility

---

### 4. Social Intelligence Layer

#### **timeline_scraper.py** - X/Twitter Scraping (2,596 lines)
- **Purpose**: Automated timeline data collection
- **Capabilities**:
  - Login automation (Selenium)
  - Timeline post extraction
  - Engagement metrics collection
  - Trending topic detection
  - Rate limit handling
  - Session management

#### **reply_handler.py** - Intelligent Reply System (2,252 lines)
- **Purpose**: Generate context-aware replies
- **Features**:
  - Post analysis and classification
  - Reply opportunity scoring
  - Context-appropriate response generation
  - Duplicate prevention
  - Engagement tracking

#### **content_analyzer.py** - Sentiment Analysis (2,401 lines)
- **Purpose**: Analyze social media content for sentiment
- **Analysis**:
  - Sentiment classification (bullish/bearish/neutral)
  - Trending topic extraction
  - Keyword frequency analysis
  - Influencer detection
  - Conversation threading
  - Engagement prediction

---

### 5. Data & API Layer

#### **database.py** - SQLite ORM (5,420 lines)
- **Purpose**: Database management and schema
- **Tables**:
  - `market_data` - Price, volume, market cap
  - `predictions` - Model predictions and outcomes
  - `price_history` - OHLC historical data
  - `technical_indicators` - Indicator values
  - `sparkline_data` - Price sparklines
  - `replied_posts` - Social media reply tracking
  - `trades` - Trade history (for future use)
  - `positions` - Open positions (for future use)
- **Features**:
  - Connection pooling
  - Transaction management
  - Query optimization
  - Data validation
  - Automated backups

#### **coingecko_handler.py** - CoinGecko API (2,016 lines)
- **Purpose**: Cryptocurrency market data integration
- **Capabilities**:
  - Price data fetching
  - Market cap and volume data
  - Sparkline data (24h price charts)
  - Rate limiting and quota tracking
  - Data validation and normalization
  - Caching (60-second cache)
  - Retry logic with exponential backoff

#### **multi_chain_manager.py** - Web3 Integration (1,266 lines)
- **Purpose**: Multi-chain blockchain connectivity
- **Supported Networks**:
  - Ethereum Mainnet
  - Polygon
  - Binance Smart Chain (BSC)
  - Arbitrum
  - Optimism
- **Features**:
  - Web3 provider management
  - Gas price fetching and optimization
  - Balance checking
  - Smart contract interaction
  - Price data aggregation
  - Transaction building
  - Network switching
- **Status**: Fully implemented, ready for trading mode

---

### 6. Utilities Layer

#### **utils/logger.py** - Centralized Logging
- **Purpose**: Unified logging across all modules
- **Features**:
  - Multiple log levels (DEBUG, INFO, WARNING, ERROR)
  - File and console output
  - Rotation and retention policies
  - Module-specific loggers
  - Performance logging

#### **utils/browser.py** - Selenium Automation
- **Purpose**: Chrome WebDriver setup and management
- **Capabilities**:
  - Headless browser configuration
  - Chrome options management
  - WebDriver lifecycle
  - Screenshot capture
  - Error recovery

#### **utils/sheets_handler.py** - Google Sheets API
- **Purpose**: Data export to Google Sheets
- **Features**:
  - OAuth2 authentication
  - Sheet creation and management
  - Data appending and updating
  - Batch operations
  - Error handling

#### **datetime_utils.py** - DateTime Handling
- **Purpose**: Timezone-aware datetime operations
- **Utilities**:
  - Timezone stripping/conversion
  - Naive datetime enforcement (SQLite compatibility)
  - Safe datetime differences
  - Timestamp formatting

---

## Data Flow

### Analysis Cycle (Current - Every 5 Minutes)

```
1. Timer Trigger (5min)
        ↓
2. Fetch Market Data
   └─→ CoinGecko API
   └─→ Cache Check (60s TTL)
        ↓
3. Store Raw Data
   └─→ SQLite Database
   └─→ market_data table
        ↓
4. Technical Analysis
   ├─→ Calculate 50+ Indicators
   ├─→ Generate Signals
   ├─→ Multi-TF Confluence
   └─→ Store technical_indicators
        ↓
5. Prediction Engine
   ├─→ ARIMA Model
   ├─→ LSTM Model
   ├─→ Ensemble Vote
   ├─→ Confidence Score
   └─→ Store predictions
        ↓
6. Social Sentiment
   ├─→ Scrape X/Twitter Timeline
   ├─→ Analyze Sentiment
   ├─→ Detect Trending Topics
   └─→ Score Reply Opportunities
        ↓
7. Market Mood Determination
   ├─→ Technical State
   ├─→ Social Sentiment
   ├─→ Volatility Score
   └─→ Mood Classification
        ↓
8. AI Content Generation
   ├─→ Claude API Call
   ├─→ Market Context
   ├─→ Mood-based Phrasing
   ├─→ Meme Phrase Selection
   └─→ Format for Twitter
        ↓
9. Social Media Post
   ├─→ Selenium Browser
   ├─→ Post Tweet
   └─→ Log Success
        ↓
10. Reply Generation
    ├─→ Find Reply Opportunities
    ├─→ Generate Context Replies
    ├─→ Post Replies
    └─→ Track replied_posts
        ↓
11. Data Logging
    ├─→ Google Sheets Update
    ├─→ Log Files
    └─→ Performance Metrics
        ↓
12. Accuracy Tracking
    └─→ Compare Predictions vs Actual
    └─→ Update Model Weights
        ↓
13. Sleep Until Next Cycle
```

### Trading Flow (Ready - Not Active)

```
1. Continuous Market Monitoring
   └─→ Multi-chain price feeds
        ↓
2. Signal Generation
   ├─→ Technical Confluence (3+ indicators)
   ├─→ Prediction Confidence (>75%)
   └─→ Social Sentiment Support
        ↓
3. Trade Opportunity
   └─→ If conditions met
        ↓
4. Risk Assessment
   ├─→ Position Sizing (2% risk)
   ├─→ Correlation Check
   ├─→ Volatility Analysis
   └─→ Account Health
        ↓
5. Execute Trade
   ├─→ Multi-chain Manager
   ├─→ DEX Selection (best price)
   ├─→ Gas Optimization
   ├─→ Slippage Protection
   └─→ Transaction Submission
        ↓
6. Position Management
   ├─→ Track Entry Price
   ├─→ Set Stop Loss (5%)
   ├─→ Set Take Profit (10%)
   ├─→ Enable Trailing Stop
   └─→ Monitor Continuously
        ↓
7. Exit Conditions
   ├─→ Take Profit Hit
   ├─→ Stop Loss Triggered
   ├─→ Trailing Stop
   ├─→ Partial Profit (50% @ 5%)
   ├─→ Time-based Exit
   └─→ Emergency Protocol
        ↓
8. Trade Completion
   ├─→ Execute Exit
   ├─→ Calculate P&L
   ├─→ Update Portfolio
   ├─→ Log Trade Journal
   └─→ Update ML Models
        ↓
9. Performance Analysis
   └─→ Win Rate, Sharpe, Drawdown
```

---

## Database Schema

### Primary Tables

#### **market_data**
```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    token_id TEXT NOT NULL,
    price REAL NOT NULL,
    volume_24h REAL,
    market_cap REAL,
    price_change_24h REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timestamp, token_id)
);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_token ON market_data(token_id);
```

#### **predictions**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    token_id TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- '1h', '24h', '7d'
    model_type TEXT NOT NULL,  -- 'arima', 'lstm', 'ensemble'
    predicted_price REAL NOT NULL,
    confidence REAL NOT NULL,  -- 0.0 to 1.0
    actual_price REAL,
    prediction_error REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    validated_at TEXT
);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_token ON predictions(token_id);
```

#### **technical_indicators**
```sql
CREATE TABLE technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    token_id TEXT NOT NULL,
    rsi REAL,
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    stoch_k REAL,
    stoch_d REAL,
    adx REAL,
    atr REAL,
    obv REAL,
    vwap REAL,
    ichimoku_conversion REAL,
    ichimoku_base REAL,
    ichimoku_span_a REAL,
    ichimoku_span_b REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timestamp, token_id)
);
```

#### **replied_posts**
```sql
CREATE TABLE replied_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT UNIQUE NOT NULL,
    post_author TEXT,
    post_content TEXT,
    reply_content TEXT,
    sentiment TEXT,
    replied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_replied_posts_id ON replied_posts(post_id);
```

#### **trades** (Future use)
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    token_id TEXT NOT NULL,
    trade_type TEXT NOT NULL,  -- 'BUY', 'SELL'
    entry_price REAL NOT NULL,
    exit_price REAL,
    amount_usd REAL NOT NULL,
    network TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl_usd REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    prediction_confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Integrations

### 1. CoinGecko API
- **Purpose**: Real-time cryptocurrency market data
- **Endpoints Used**:
  - `/coins/markets` - Current prices, volumes, market caps
  - `/coins/{id}/market_chart` - Historical price data
- **Rate Limits**: Free tier - 10-30 calls/minute
- **Quota Tracking**: Built-in with warnings
- **Caching**: 60-second TTL to minimize calls
- **Error Handling**: Exponential backoff, fallback prices

### 2. Claude AI (Anthropic)
- **Purpose**: Advanced market analysis and content generation
- **Model**: claude-3-5-sonnet-20240620
- **Max Tokens**: 1000 per request
- **Temperature**: 0.7 (balanced creativity)
- **Use Cases**:
  - Market analysis prose
  - Tweet content generation
  - Reply generation
  - Sentiment interpretation
- **Rate Limits**: Based on API tier
- **Cost**: ~$0.003 per analysis (~$0.036/hour)

### 3. Google Sheets API
- **Purpose**: Data export and reporting
- **Authentication**: Service Account OAuth2
- **Scopes**: `spreadsheets` (read/write)
- **Operations**:
  - Append rows (market data, predictions)
  - Update ranges (performance metrics)
  - Create sheets (new reports)
- **Batch Operations**: Grouped for efficiency

### 4. X/Twitter (via Selenium)
- **Purpose**: Social media posting and scraping
- **Method**: Browser automation (not official API)
- **Capabilities**:
  - Post tweets
  - Scrape timeline
  - Reply to posts
  - Extract engagement metrics
- **Rate Limiting**: Manual timing to avoid blocks
- **Session Management**: Persistent login

### 5. Blockchain RPC Endpoints (Ready)
- **Providers**: Alchemy, Infura, public nodes
- **Networks**:
  - Ethereum: `eth-mainnet.g.alchemy.com`
  - Polygon: `polygon-mainnet.g.alchemy.com`
  - Arbitrum: `arb-mainnet.g.alchemy.com`
  - BSC: `bsc-dataseed.binance.org`
  - Optimism: `mainnet.optimism.io`
- **Methods Used**:
  - `eth_gasPrice` - Gas price estimation
  - `eth_getBalance` - Wallet balance
  - `eth_call` - Smart contract interaction
  - `eth_sendRawTransaction` - Submit trades
  - `eth_getTransactionReceipt` - Confirm trades

### 6. DEX Protocol APIs (Ready)
- **Uniswap V3**: Smart contract calls for swaps
- **SushiSwap**: Multi-chain routing
- **1inch**: DEX aggregation for best prices
- **Methods**: Token swaps, liquidity checks, price quotes

---

## Trading System Architecture

### Position Management

```python
@dataclass
class Position:
    position_id: str
    token: str
    trade_type: TradeType  # LONG, SHORT, BUY, SELL
    entry_price: float
    amount_usd: float
    entry_time: datetime
    network: str
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: Optional[float]
    status: TradeStatus  # OPEN, CLOSED, PARTIAL
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    prediction_confidence: float
```

### Risk Manager

**Position Sizing:**
```python
max_risk_per_trade = account_balance * 0.02  # 2% risk
position_size = max_risk_per_trade / stop_loss_pct
```

**Risk Checks:**
- Maximum 3-5 simultaneous positions
- Correlation limits (avoid all Layer 1s)
- Daily loss limit: 5% of account
- Drawdown limit: 10% triggers pause
- Volatility-adjusted sizing

### Trade Execution Flow

1. **Signal Generation**
   - 3+ indicator confluence
   - Prediction confidence >75%
   - Social sentiment alignment

2. **Risk Assessment**
   - Calculate position size
   - Check correlation
   - Verify available capital

3. **Pre-Trade Validation**
   - Liquidity check on DEX
   - Gas price acceptable
   - Slippage within limits

4. **Execution**
   - Select best DEX (1inch aggregation)
   - Build transaction
   - Sign with wallet
   - Submit to blockchain

5. **Confirmation**
   - Wait for transaction receipt
   - Verify execution price
   - Log trade details

6. **Position Monitoring**
   - Real-time P&L tracking
   - Trailing stop updates
   - Exit signal monitoring

---

## Security Architecture

### Wallet Security

**Encryption:**
- AES-256 encryption for private keys
- Keys stored encrypted in .env
- Decryption only in memory during signing
- No plaintext keys on disk

**Key Management:**
```python
from cryptography.fernet import Fernet
import keyring

# Encryption key stored in OS keyring
encryption_key = keyring.get_password("tokenetics", "encryption_key")
cipher = Fernet(encryption_key)

# Encrypt private key
encrypted_key = cipher.encrypt(private_key.encode())

# Decrypt only when needed
private_key = cipher.decrypt(encrypted_key).decode()
```

**Multi-Sig Support** (Optional):
- Multiple approvers required
- Threshold signatures (e.g., 2-of-3)
- Enhanced security for large capital

### API Security

**API Key Protection:**
- All keys in .env (gitignored)
- Environment variable loading
- Never logged or exposed
- Rotation policy recommended

**Rate Limiting:**
- Request queuing
- Quota tracking
- Exponential backoff
- Graceful degradation

### Trading Safety

**Pre-Flight Checks:**
```python
def validate_trade(trade):
    assert trade.position_size <= max_position_size
    assert trade.prediction_confidence >= 0.75
    assert trade.network in SUPPORTED_NETWORKS
    assert has_sufficient_balance(trade)
    assert not exceeds_correlation_limit(trade)
    assert not exceeds_daily_loss_limit()
```

**Emergency Protocols:**
- Instant position closure button
- Automatic stop on critical errors
- Drawdown limits
- Manual override capability

---

## Scalability & Performance

### Optimization Strategies

**1. M4 Ultra Acceleration**
- Numba JIT compilation for critical paths
- Hardware detection and optimization
- Parallel array operations
- Memory-efficient calculations

**2. Caching**
- 60-second price data cache
- Indicator calculation cache
- API response caching
- Database query optimization

**3. Database Performance**
- Indexed queries (timestamp, token_id)
- Batch inserts
- Connection pooling
- Vacuum and optimize schedule

**4. Async Operations**
- Parallel API calls where possible
- Non-blocking I/O for network requests
- Background logging
- Concurrent indicator calculations

### Performance Metrics

**Current Performance:**
- Analysis cycle: 5 minutes
- Technical indicators: ~50ms per token
- Prediction generation: ~2-5 seconds
- Database insert: <10ms
- API calls: 100-500ms (network dependent)

**Throughput:**
- 13 tokens analyzed per cycle
- ~3,700 data points stored per hour
- ~720 API calls per day
- 99%+ uptime

### Scaling Considerations

**Horizontal Scaling:**
- Multiple bots for different token sets
- Separate trading and analysis instances
- Load-balanced API requests

**Vertical Scaling:**
- More CPU cores for parallel processing
- Additional RAM for larger ML models
- SSD for database performance

---

## Development & Deployment

### Environment Setup

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Initialize database
python src/database.py

# 5. Run in analysis mode
python src/bot.py

# 6. Run in trading mode (when ready)
python src/integrated_trading_bot.py --mode=live
```

### Configuration Files

**.env Structure:**
```bash
# AI/LLM
ANTHROPIC_API_KEY=
LLM_PROVIDER=anthropic

# Market Data
COINGECKO_API_KEY=

# Social Media
TWITTER_USERNAME=
TWITTER_PASSWORD=
CHROME_DRIVER_PATH=

# Blockchain (for trading)
ETHEREUM_RPC_URL=
POLYGON_RPC_URL=
ARBITRUM_RPC_URL=
BSC_RPC_URL=
OPTIMISM_RPC_URL=

# Wallet (encrypted)
WALLET_PRIVATE_KEY_ENCRYPTED=
WALLET_ENCRYPTION_PASSWORD=
WALLET_ADDRESS=

# Trading Parameters
MAX_POSITION_SIZE_USD=1000
RISK_PER_TRADE_PCT=2.0
STOP_LOSS_PCT=5.0
TAKE_PROFIT_PCT=10.0
MIN_PREDICTION_CONFIDENCE=0.75
TRADING_MODE=CONSERVATIVE
```

### Logging Configuration

**Log Levels:**
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Warning messages for potential issues
- ERROR: Error messages for failures

**Log Files:**
- `logs/claude.log` - Claude API interactions
- `logs/coingecko.log` - CoinGecko data fetching
- `logs/google_sheets_api.log` - Sheets operations
- `logs/analysis/market_analysis.log` - Analysis results
- `logs/trading_system_*.log` - Trading operations

---

## Module Dependencies

### Dependency Graph

```
bot.py
├── config.py
├── database.py
├── llm_provider.py
├── coingecko_handler.py
├── prediction_engine.py
│   ├── technical_system.py
│   │   ├── technical_indicators.py
│   │   ├── technical_signals.py
│   │   ├── technical_core.py
│   │   ├── technical_calculations.py
│   │   └── technical_foundation.py
│   └── mood_config.py
├── timeline_scraper.py
│   └── utils/browser.py
├── reply_handler.py
│   └── content_analyzer.py
├── meme_phrases.py
└── utils/
    ├── logger.py
    └── sheets_handler.py

integrated_trading_bot.py
├── config.py
├── database.py
├── multi_chain_manager.py
├── prediction_engine.py
├── technical_portfolio.py
├── llm_provider.py
└── coingecko_handler.py
```

### External Dependencies

**Core:**
- Python 3.11+
- anthropic (Claude SDK)
- web3 (blockchain)
- selenium (browser automation)

**Data Science:**
- pandas, numpy, polars
- scikit-learn
- statsmodels
- tensorflow, keras
- numba

**APIs:**
- requests, aiohttp
- google-auth, gspread
- python-dotenv

**Database:**
- sqlalchemy

See [requirements.txt](requirements.txt) for complete list.

---

## Testing Strategy

### Current Testing Approach

**Prediction Validation:**
- Walk-forward testing on historical data
- Real-time prediction vs actual tracking
- Confidence calibration analysis
- Model performance dashboards

**System Testing:**
- Unit tests for critical functions
- Integration tests for API interactions
- End-to-end analysis cycle tests
- Database integrity checks

**Social Media Testing:**
- Manual review of generated content
- Engagement metric tracking
- Reply quality assessment
- Sentiment analysis validation

### Future Testing (Before Live Trading)

**Paper Trading:**
- Simulated trades with real data
- P&L tracking without capital risk
- Strategy refinement
- Risk parameter validation

**Backtesting:**
- Historical data replay
- Strategy performance evaluation
- Drawdown analysis
- Win rate optimization

---

## Monitoring & Observability

### Key Metrics Tracked

**Performance:**
- Prediction accuracy by timeframe
- Indicator calculation time
- API response times
- Database query performance

**Trading (When Active):**
- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Average hold time
- Position count

**System Health:**
- CPU and memory usage
- Database size
- API quota consumption
- Error rates
- Uptime percentage

### Alerting (Planned)

- Critical errors (email/SMS)
- API quota warnings
- Trading losses exceeding limits
- System downtime alerts
- Unusual market conditions

---

## Future Architecture Enhancements

### Phase 3-5 Additions (Planned)

**Advanced Features:**
- Options trading integration
- Perpetual futures support
- Advanced MEV protection (Flashbots)
- DAO governance participation
- Mobile app + push notifications

**Infrastructure:**
- Redis caching layer
- PostgreSQL migration (from SQLite)
- Docker containerization
- Kubernetes orchestration
- Load balancers

**Monitoring:**
- Grafana dashboards
- Prometheus metrics
- ELK stack for log aggregation
- APM (Application Performance Monitoring)

**Security:**
- Hardware wallet integration (Ledger/Trezor)
- 2FA for critical operations
- Audit logging
- Compliance reporting

---

## Conclusion

Tokenetics represents a production-ready autonomous DeFi trading platform with comprehensive market analysis, AI-powered predictions, social intelligence, and multi-chain trading capabilities. The architecture is modular, scalable, and secure, with all trading infrastructure complete and awaiting final prediction refinement before live deployment.

**Current Focus:** Validating and optimizing prediction models to ensure consistent profitability before activating autonomous trading with real capital.

**Total Codebase:** ~70,000 lines of production Python code across 40+ modules, representing a sophisticated, enterprise-grade trading system.

---

*Last Updated: 2026-01-16*
*Version: 2.0 - Complete Architecture Documentation*
