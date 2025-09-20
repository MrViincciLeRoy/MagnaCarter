# magnaCarter Trading System
## Market-Adaptive Global Neural Architecture for Crypto Automation, Reinforcement Training, and Equity Returns

A comprehensive algorithmic trading system that integrates with the Alpaca API for both backtesting and live trading of cryptocurrencies and stocks.

## Features

- **Backtesting Framework**: Test strategies on historical data with detailed performance metrics
- **Live Trading**: Automated trading with configurable risk management
- **Multi-Asset Support**: Trade both cryptocurrencies (BTC/USD, ETH/USD) and stocks (AAPL, etc.)
- **Technical Indicators**: Built-in technical analysis using pandas-ta
- **Risk Management**: Stop losses, take profits, position sizing, and daily trade limits
- **Paper Trading**: Safe testing environment before risking real capital
- **Multi-Symbol Trading**: Run multiple trading strategies simultaneously

## Strategy Overview

The trading strategy uses:
- **Core Indicators**: Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Hull Moving Average (HMA)
- **Support/Resistance**: Dynamic support and resistance level detection
- **Higher Timeframe Analysis**: Multi-timeframe signal generation
- **Trend Change Detection**: Identifies trend reversals for entry/exit signals

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd magna-carter-trading-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Alpaca API credentials:
   - Get your API keys from [Alpaca Markets](https://alpaca.markets/)
   - Set environment variables:
```bash
export APCA_API_KEY_ID="your_api_key_here"
export APCA_API_SECRET_KEY="your_secret_key_here"
```

## Quick Start

### 1. Backtesting

Run a backtest on historical data:

```python
from magna_carter_trading_system import MagnaCarterBacktester

# Initialize the backtester
backtester = MagnaCarterBacktester(
    api_key="your_api_key",
    secret_key="your_secret_key",
    paper_trading=True,
    # Strategy parameters
    len_cbrc=30,
    signal_period='480min'
)

# Run backtest
stats, data = backtester.run_backtest(
    symbol="BTC/USD",
    start_date="2024-01-01",
    end_date="2024-06-30",
    initial_cash=100000
)
```

### 2. Live Trading (Paper Trading)

Start live paper trading:

```python
from magna_carter_trading_system import MagnaCarterBacktester, LiveTradingConfig

# Initialize system
system = MagnaCarterBacktester(
    api_key="your_api_key",
    secret_key="your_secret_key",
    paper_trading=True  # ALWAYS start with paper trading!
)

# Configure trading parameters
config = LiveTradingConfig(
    max_position_size=0.5,      # Use 50% of portfolio
    max_daily_trades=5,         # Max 5 trades per day
    min_trade_interval=1800,    # 30 min between trades
    check_interval=1800,        # Check every 30 min
    risk_per_trade=0.02         # Risk 2% per trade
)

# Start live trading
trader = system.start_live_trading(
    symbol="BTC/USD",
    config=config
)
```

### 3. Multi-Symbol Trading

Trade multiple assets simultaneously:

```python
# Different configs for different asset types
crypto_config = LiveTradingConfig(
    max_position_size=0.4,
    max_daily_trades=8,
    check_interval=1800
)

stock_config = LiveTradingConfig(
    max_position_size=0.6,
    max_daily_trades=5,
    check_interval=3600
)

# Start multiple traders
symbols_configs = [
    ("BTC/USD", crypto_config),
    ("ETH/USD", crypto_config),
    ("AAPL", stock_config),
]

for symbol, config in symbols_configs:
    system.start_live_trading(
        symbol=symbol,
        config=config,
        background=True
    )
```

## Command Line Interface

Run demos from the command line:

```bash
# Backtest demo
python magna_carter_trading_system.py --mode backtest --symbol BTC/USD

# Live trading demo
python magna_carter_trading_system.py --mode live --symbol BTC/USD --paper

# Multi-symbol trading demo
python magna_carter_trading_system.py --mode multi --paper
```

## Configuration

### LiveTradingConfig Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_position_size` | Maximum % of portfolio per trade | 0.8 (80%) |
| `stop_loss_pct` | Stop loss percentage | 0.05 (5%) |
| `take_profit_pct` | Take profit percentage | 0.15 (15%) |
| `max_daily_trades` | Maximum trades per day | 10 |
| `min_trade_interval` | Minimum seconds between trades | 3600 (1 hour) |
| `check_interval` | Signal check interval in seconds | 1800 (30 min) |
| `risk_per_trade` | Risk percentage per trade | 0.02 (2%) |

### Strategy Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `len_cbrc` | SMA length for core indicator | 34 |
| `len_cbrc_high_low` | High/Low period for support/resistance | 13 |
| `rst_len` | Resistance/Support calculation period | 10 |
| `ema1_len` | First EMA length | 13 |
| `ema2_len` | Second EMA length | 21 |
| `hma_base_length` | Hull MA base length | 8 |
| `hma_length_scalar` | Hull MA length multiplier | 5 |
| `signal_period` | Higher timeframe for signals | '720min' |

## Safety Features

### Risk Management
- **Position Sizing**: Automatic calculation based on portfolio size and risk tolerance
- **Stop Losses**: Configurable percentage-based stop losses
- **Daily Limits**: Maximum number of trades per day
- **Time Intervals**: Minimum time between trades to avoid overtrading

### Paper Trading
- Always start with paper trading to test your strategy
- No real money at risk while learning and optimizing
- Full simulation of real trading conditions

### Error Handling
- Robust error handling for API failures
- Automatic fallback to sample data when API is unavailable
- Comprehensive logging for debugging

## Monitoring and Performance

### Account Information
```python
# Get current account status
account_info = trader.get_account_info()
print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
print(f"Buying Power: ${account_info['buying_power']:,.2f}")
```

### Position Tracking
```python
# Check current positions
position = trader.get_current_position()
if position:
    print(f"Current Position: {position['side']} {position['qty']} shares")
    print(f"Unrealized P&L: {position['unrealized_plpc']:.2f}%")
```

### Performance Summary
```python
# Get comprehensive performance summary
summary = system.get_live_trading_summary()
print(f"Active Traders: {summary['active_traders']}")
```

## Important Warnings

⚠️ **ALWAYS START WITH PAPER TRADING** - Never risk real money until you've thoroughly tested your strategy

⚠️ **Cryptocurrency Trading Risks**:
- High volatility can lead to significant losses
- 24/7 markets mean constant exposure
- Regulatory risks in various jurisdictions

⚠️ **Algorithmic Trading Risks**:
- Technical failures can lead to unintended trades
- Market conditions can change rapidly
- Past performance does not guarantee future results

⚠️ **API Rate Limits**: Be aware of Alpaca's API rate limits to avoid service interruptions

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API keys are correct
   - Check internet connection
   - Ensure Alpaca account is funded (for live trading)

2. **No Trading Signals**
   - Check if there's sufficient historical data
   - Verify strategy parameters aren't too restrictive
   - Review market conditions (low volatility periods may generate fewer signals)

3. **Order Rejection**
   - Verify account has sufficient buying power
   - Check if symbol is supported for trading
   - Ensure position sizes meet minimum requirements

### Debugging

Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All changes include appropriate tests
- Documentation is updated for new features

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always conduct thorough testing with paper trading before risking real capital.

The authors and contributors are not responsible for any financial losses incurred through the use of this software.


---

**Remember: Never risk more than you can afford to lose, and always start with paper trading!**
