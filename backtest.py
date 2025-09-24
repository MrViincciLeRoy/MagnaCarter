import os
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from backtesting import Backtest, Strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading"""
    max_position_size: float = 0.8      # Max % of portfolio per trade
    stop_loss_pct: float = 0.05         # 5% stop loss
    take_profit_pct: float = 0.15       # 15% take profit
    max_daily_trades: int = 10          # Max trades per day
    min_trade_interval: int = 3600      # Min seconds between trades (1 hour)
    check_interval: int = 1800          # Check signals every 30 minutes
    risk_per_trade: float = 0.02        # Risk 2% of portfolio per trade


@dataclass
class TradeSignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    signal: int  # 1 = buy, -1 = sell, 0 = hold
    price: float
    confidence: float = 1.0


class AlpacaMegaCryptoBotFixed:
    """
    Alpaca-wrapped version of the Mega crypto bot strategy with modern API integration
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper_trading: bool = True,
        **strategy_params
    ):
        """
        Initialize the Alpaca-wrapped strategy

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Whether to use paper trading credentials
            **strategy_params: Strategy parameters to override defaults
        """
        # API Configuration
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper_trading = paper_trading

        # Initialize Alpaca clients
        if self.api_key and self.secret_key:
            try:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=paper_trading
                )

                self.crypto_data_client = CryptoHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )

                self.stock_data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                
                # Test connection
                account = self.trading_client.get_account()
                logger.info(f"Connected to Alpaca account: {account.status}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca clients: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                self.trading_client = None
                self.crypto_data_client = None
                self.stock_data_client = None
        else:
            logger.warning("No Alpaca API keys provided - will use sample data")
            self.trading_client = None
            self.crypto_data_client = None
            self.stock_data_client = None

        # Strategy parameters
        default_params = {
            'len_cbrc': 34,
            'len_cbrc_high_low': 13,
            'rst_len': 10,
            'ema1_len': 13,
            'ema2_len': 21,
            'hma_base_length': 8,
            'hma_length_scalar': 5,
            'signal_period': '720min'
        }
        self.params = {**default_params, **strategy_params}

        logger.info("AlpacaMegaCryptoBotFixed initialized")
        logger.info(f"Paper trading: {paper_trading}")

    def get_crypto_data(
        self,
        symbol: str = "BTC/USD",
        start_date: str = "2024-01-01",
        end_date: str = None,
        timeframe: TimeFrame = TimeFrame.Hour
    ) -> pd.DataFrame:
        """
        Fetch crypto data from Alpaca

        Args:
            symbol: Crypto symbol (e.g., "BTC/USD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            timeframe: Data timeframe

        Returns:
            DataFrame with OHLCV data
        """
        if not self.crypto_data_client:
            logger.warning("No crypto data client available - generating sample data")
            return self._generate_sample_data(symbol, start_date, end_date)

        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d")
            )

            bars = self.crypto_data_client.get_crypto_bars(request)
            df = bars.df

            if len(df) > 0:
                # Clean up the dataframe
                df = df.reset_index()
                df = df.set_index('timestamp')
                df = df.drop('symbol', axis=1, errors='ignore')

                # Standardize column names
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trade Count', 'VWAP']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                # Remove timezone info for compatibility
                df.index = df.index.tz_localize(None)

                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning("No data returned from Alpaca")
                return self._generate_sample_data(symbol, start_date, end_date)

        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.info("Falling back to sample data")
            return self._generate_sample_data(symbol, start_date, end_date)

    def get_stock_data(
        self,
        symbol: str = "AAPL",
        start_date: str = "2024-01-01",
        end_date: str = None,
        timeframe: TimeFrame = TimeFrame.Hour
    ) -> pd.DataFrame:
        """
        Fetch stock data from Alpaca

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            timeframe: Data timeframe

        Returns:
            DataFrame with OHLCV data
        """
        if not self.stock_data_client:
            logger.warning("No stock data client available - generating sample data")
            return self._generate_sample_data(symbol, start_date, end_date, asset_type="stock")

        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d")
            )

            bars = self.stock_data_client.get_stock_bars(request)
            df = bars.df

            if len(df) > 0:
                # Clean up the dataframe
                df = df.reset_index()
                df = df.set_index('timestamp')
                df = df.drop('symbol', axis=1, errors='ignore')

                # Standardize column names
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trade Count', 'VWAP']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                # Remove timezone info for compatibility
                df.index = df.index.tz_localize(None)

                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning("No data returned from Alpaca")
                return self._generate_sample_data(symbol, start_date, end_date, asset_type="stock")

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.info("Falling back to sample data")
            return self._generate_sample_data(symbol, start_date, end_date, asset_type="stock")

    def _generate_sample_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str = None,
        asset_type: str = "crypto"
    ) -> pd.DataFrame:
        """
        Generate realistic sample data when API is not available
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start_dt, end_dt, freq='1H')

        # Set random seed for reproducible results
        np.random.seed(hash(symbol) % 2**32)

        # Generate price data based on asset type
        n_points = len(dates)

        if asset_type == "crypto" or "BTC" in symbol or "ETH" in symbol:
            # Crypto-like volatility and trend
            base_price = 45000 if "BTC" in symbol else 2500
            trend = np.linspace(base_price * 0.7, base_price * 1.5, n_points)
            volatility = 0.03
        else:
            # Stock-like behavior
            base_price = 150
            trend = np.linspace(base_price * 0.9, base_price * 1.3, n_points)
            volatility = 0.02

        # Add realistic noise
        noise = np.random.normal(0, volatility, n_points).cumsum()
        prices = trend * (1 + noise * 0.1)

        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(prices[0])

        # Generate realistic high/low
        bar_volatility = np.random.uniform(0.005, 0.025, n_points)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + bar_volatility)
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - bar_volatility)
        df['Volume'] = np.random.randint(100, 2000, n_points)

        logger.info(f"Generated {len(df)} sample bars for {symbol}")
        return df

    def _valuewhen(self, condition: pd.Series, source: pd.Series) -> pd.Series:
        """Helper function for valuewhen logic"""
        assert isinstance(condition, pd.Series) and condition.dtype == 'bool'
        assert isinstance(source, pd.Series)
        result = pd.Series(np.nan, index=source.index)
        result.loc[condition] = source[condition]
        return result.ffill()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all strategy indicators

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with indicators and signals
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        strategy_df = df.copy()
        p = self.params

        logger.info("Calculating technical indicators...")

        # --- Core Indicators ---
        strategy_df['cbrc_sma'] = ta.sma(strategy_df['Close'], length=p['len_cbrc'])
        strategy_df['cbrc_ul'] = strategy_df['High'].rolling(window=p['len_cbrc_high_low']).max()
        strategy_df['cbrc_ll'] = strategy_df['Low'].rolling(window=p['len_cbrc_high_low']).min()

        # Support/Resistance levels
        rst_highest = strategy_df['High'].rolling(window=p['rst_len']).max()
        rst_lowest = strategy_df['Low'].rolling(window=p['rst_len']).min()
        strategy_df['res'] = self._valuewhen(strategy_df['High'] >= rst_highest, strategy_df['High'])
        strategy_df['sup'] = self._valuewhen(strategy_df['Low'] <= rst_lowest, strategy_df['Low'])

        # EMAs
        strategy_df['ema1'] = ta.ema(strategy_df['Close'], length=p['ema1_len'])
        strategy_df['ema2'] = ta.ema(strategy_df['Close'], length=p['ema2_len'])

        # Hull Moving Average
        hma_length = p['hma_base_length'] + p['hma_length_scalar'] * 6
        strategy_df['hma'] = ta.hma(strategy_df['Close'], length=hma_length)

        logger.info("Technical indicators calculated")

        # --- Signal Generation ---
        logger.info("Generating trading signals...")

        # Create higher timeframe data
        ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        df_higher_tf = strategy_df.resample(p['signal_period']).agg(ohlc_dict).dropna()

        df_higher_tf = df_higher_tf.rename(columns={
            'Open': 'Open_htf', 'High': 'High_htf',
            'Low': 'Low_htf', 'Close': 'Close_htf'
        })

        # Merge higher timeframe data
        strategy_df = pd.merge_asof(
            strategy_df, df_higher_tf,
            left_index=True, right_index=True,
            direction='backward'
        )
        strategy_df = strategy_df.ffill()

        # Signal logic - trend changes on higher timeframe
        htf_close = strategy_df['Close_htf']
        htf_open = strategy_df['Open_htf']

        current_green = htf_close > htf_open
        current_red = htf_close < htf_open
        prev_green = htf_close.shift(1) > htf_open.shift(1)
        prev_red = htf_close.shift(1) < htf_open.shift(1)

        # Generate signals on trend changes
        long_cond = current_green & prev_red   # Green after red
        short_cond = current_red & prev_green  # Red after green

        strategy_df['signal'] = 0
        strategy_df.loc[long_cond, 'signal'] = 1
        strategy_df.loc[short_cond, 'signal'] = -1

        logger.info("Trading signals generated")

        return strategy_df

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal distribution and trading opportunities"""
        if 'signal' not in df.columns:
            return {}

        signal_counts = df['signal'].value_counts().sort_index()
        total_signals = len(df)

        analysis = {
            'total_bars': total_signals,
            'signal_distribution': {}
        }

        for signal, count in signal_counts.items():
            signal_name = {-1: 'Short', 0: 'Hold', 1: 'Long'}.get(signal, 'Unknown')
            percentage = (count / total_signals) * 100
            analysis['signal_distribution'][signal_name] = {
                'count': count,
                'percentage': percentage
            }

        # Trading opportunities
        signal_changes = (df['signal'] != df['signal'].shift(1)).sum()
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()

        analysis.update({
            'signal_changes': signal_changes,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        })

        return analysis


class AlpacaLiveTrader:
    """
    Live trading implementation using the Alpaca API
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        symbol: str = "BTC/USD",
        paper_trading: bool = True,
        config: LiveTradingConfig = None,
        **strategy_params
    ):
        """
        Initialize live trader

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            symbol: Trading symbol
            paper_trading: Use paper trading
            config: Live trading configuration
            **strategy_params: Strategy parameters
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_API_SECRET')
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.config = config or LiveTradingConfig()

        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key must be provided for live trading")

        # Initialize clients with detailed error handling
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=paper_trading
            )
            
            # Test connection
            account = self.trading_client.get_account()
            logger.info(f"Connected to trading client: {account.status}")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading client: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"API Key present: {bool(self.api_key)}")
            logger.error(f"Secret Key present: {bool(self.secret_key)}")
            raise

        # Initialize strategy
        self.strategy_bot = AlpacaMegaCryptoBotFixed(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper_trading=paper_trading,
            **strategy_params
        )

        # Trading state
        self.last_signal = 0
        self.last_trade_time = datetime.min
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.is_trading = False
        self.stop_trading_flag = False

        # Determine if crypto or stock
        self.is_crypto = "/" in symbol or any(crypto in symbol.upper()
                                            for crypto in ["BTC", "ETH", "LTC", "USD"])

        logger.info(f"AlpacaLiveTrader initialized for {symbol}")
        logger.info(f"Paper trading: {paper_trading}")
        logger.info(f"Asset type: {'Crypto' if self.is_crypto else 'Stock'}")

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': getattr(account, 'daytrade_buying_power', 0),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return {}

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current position for the symbol"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return {
                        'symbol': position.symbol,
                        'qty': float(position.qty),
                        'side': position.side,
                        'market_value': float(position.market_value),
                        'avg_entry_price': float(position.avg_entry_price),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate appropriate position size with detailed error handling"""
        try:
            logger.info(f"Calculating position size for price: ${current_price}")
            
            account_info = self.get_account_info()
            if not account_info:
                logger.error("No account info available for position sizing")
                return 0.0

            portfolio_value = account_info.get('portfolio_value', 0)
            buying_power = account_info.get('buying_power', 0)
            
            logger.info(f"Account values - Portfolio: ${portfolio_value}, Buying Power: ${buying_power}")

            if portfolio_value <= 0:
                logger.error("Invalid portfolio value")
                return 0.0

            # Use risk-based position sizing
            risk_amount = portfolio_value * self.config.risk_per_trade
            max_position_value = portfolio_value * self.config.max_position_size

            # Choose the smaller of risk-based or max position
            position_value = min(risk_amount, max_position_value)
            
            # Also limit by buying power
            position_value = min(position_value, buying_power * 0.95)  # Leave 5% margin
            
            logger.info(f"Position sizing - Risk: ${risk_amount}, Max: ${max_position_value}, Available: ${buying_power}")
            logger.info(f"Selected position value: ${position_value}")

            if position_value <= 0:
                logger.warning("Position value is zero or negative")
                return 0.0

            if self.is_crypto:
                # Crypto supports fractional trading
                quantity = position_value / current_price
                quantity = round(quantity, 6)  # 6 decimal precision
                
                # Check minimum order size for crypto (typically $1-$10)
                min_order_value = 10.0  # $10 minimum
                if quantity * current_price < min_order_value:
                    logger.warning(f"Order value ${quantity * current_price} below minimum ${min_order_value}")
                    return 0.0
                    
            else:
                # Stocks - whole shares only
                quantity = int(position_value / current_price)
                
                # Ensure at least 1 share
                if quantity < 1:
                    logger.warning(f"Position size less than 1 share: {quantity}")
                    return 0.0

            logger.info(f"Calculated position size: {quantity} units at ${current_price:.2f} = ${quantity * current_price:.2f}")
            return quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Current price: {current_price}")
            return 0.0

    def place_market_order(self, side: OrderSide, quantity: float) -> bool:
        """Place a market order with enhanced error handling"""
        try:
            if quantity <= 0:
                logger.error(f"Invalid quantity for order: {quantity}")
                return False

            logger.info(f"Attempting to place order: {side.value} {quantity} {self.symbol}")
            
            market_order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )

            order = self.trading_client.submit_order(order_data=market_order_data)

            logger.info(f"Order submitted successfully:")
            logger.info(f"  Order ID: {order.id}")
            logger.info(f"  Symbol: {order.symbol}")
            logger.info(f"  Side: {order.side}")
            logger.info(f"  Quantity: {order.qty}")
            logger.info(f"  Status: {order.status}")

            # Update trading state
            self.last_trade_time = datetime.now()
            self.daily_trade_count += 1

            return True

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Order details - Side: {side}, Quantity: {quantity}, Symbol: {self.symbol}")
            
            # Try to get more specific error information
            if hasattr(e, 'response'):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            
            return False

    def close_position(self) -> bool:
        """Close current position with enhanced error handling"""
        try:
            position = self.get_current_position()
            if not position:
                logger.info("No position to close")
                return True

            logger.info(f"Closing position: {position}")

            # Determine opposite side
            if position['side'] == 'long':
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY

            quantity = abs(float(position['qty']))
            return self.place_market_order(side, quantity)

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False

    def should_trade(self, signal: int) -> bool:
        """Check if trading conditions are met"""
        # Check if trading is stopped
        if self.stop_trading_flag:
            logger.debug("Trading stopped by flag")
            return False

        # Reset daily trade count
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            logger.info(f"Reset daily trade count for {current_date}")

        # Check daily trade limit
        if self.daily_trade_count >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trade_count}/{self.config.max_daily_trades}")
            return False

        # Check minimum time interval
        time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
        if time_since_last < self.config.min_trade_interval:
            remaining = self.config.min_trade_interval - time_since_last
            logger.debug(f"Waiting {remaining:.0f}s before next trade")
            return False

        # Check if signal changed
        if signal == self.last_signal:
            logger.debug(f"Signal unchanged: {signal}")
            return False

        logger.info(f"Trading conditions met. Signal: {signal}, Last: {self.last_signal}")
        return True

    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute trade based on signal with comprehensive error handling"""
        try:
            logger.info(f"Executing trade - Signal: {signal.signal}, Price: ${signal.price}, Symbol: {signal.symbol}")
            
            if not self.should_trade(signal.signal):
                logger.info("Trade conditions not met")
                return False

            current_position = self.get_current_position()
            logger.info(f"Current position: {current_position}")
            
            success = False

            if signal.signal == 1:  # Buy signal
                logger.info("Processing BUY signal")
                
                if current_position and current_position['side'] == 'short':
                    # Close short first
                    logger.info("Closing short position before going long")
                    self.close_position()
                    time.sleep(2)  # Wait for order to process

                if not current_position or current_position['side'] != 'long':
                    # Open long position
                    quantity = self.calculate_position_size(signal.price)
                    logger.info(f"Calculated position size: {quantity}")
                    
                    if quantity > 0:
                        success = self.place_market_order(OrderSide.BUY, quantity)
                        if success:
                            logger.info(f"LONG position opened at ${signal.price:.2f}")
                    else:
                        logger.error("Cannot open position - invalid quantity")
                        
            elif signal.signal == -1:  # Sell signal
                logger.info("Processing SELL signal")
                
                if current_position and current_position['side'] == 'long':
                    # Close long position
                    success = self.close_position()
                    if success:
                        logger.info(f"LONG position closed at ${signal.price:.2f}")
                elif current_position and current_position['side'] == 'short':
                    logger.info("Already in short position")
                    success = True
                else:
                    logger.info("No position to close for sell signal")
                    success = True

            if success:
                self.last_signal = signal.signal
                logger.info(f"Trade executed successfully. New signal state: {signal.signal}")
            else:
                logger.error("Trade execution failed")

            return success

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Signal details: {signal}")
            return False

    def generate_signal(self) -> Optional[TradeSignal]:
        """Generate trading signal from current market data with enhanced error handling"""
        try:
            # Get recent data for analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Get last week

            logger.info(f"Generating signal for {self.symbol}")

            if self.is_crypto:
                data = self.strategy_bot.get_crypto_data(
                    symbol=self.symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
            else:
                data = self.strategy_bot.get_stock_data(
                    symbol=self.symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )

            if data.empty:
                logger.error("No data available for signal generation")
                return None
                
            if len(data) < 100:
                logger.warning(f"Insufficient data for signal generation: {len(data)} bars")
                return None

            # Calculate indicators and signals
            strategy_data = self.strategy_bot.calculate_indicators(data)

            # Get latest signal
            latest_signal = int(strategy_data['signal'].iloc[-1])
            latest_price = float(strategy_data['Close'].iloc[-1])
            
            logger.info(f"Generated signal: {latest_signal} at ${latest_price:.2f}")

            return TradeSignal(
                timestamp=datetime.now(),
                symbol=self.symbol,
                signal=latest_signal,
                price=latest_price
            )

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None

    def run_single_check(self) -> bool:
        """Run a single trading check cycle with detailed logging"""
        try:
            logger.info("Running trading check...")

            # Generate signal
            signal = self.generate_signal()
            if signal is None:
                logger.error("Failed to generate signal")
                return False

            logger.info(f"Current signal: {signal.signal} at ${signal.price:.2f}")
            logger.info(f"Last signal: {self.last_signal}")

            # Execute trade if signal changed
            if signal.signal != self.last_signal:
                logger.info(f"Signal changed from {self.last_signal} to {signal.signal}")
                success = self.execute_trade(signal)
                logger.info(f"Trade execution: {'SUCCESS' if success else 'FAILED'}")
                return success
            else:
                logger.info("No signal change - no action taken")
                return True

        except Exception as e:
            logger.error(f"Error in trading check: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False

    def start_live_trading(self):
        """Start live trading loop"""
        logger.info("Starting live trading...")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Paper trading: {self.paper_trading}")
        logger.info(f"Check interval: {self.config.check_interval}s")

        # Display account info
        account_info = self.get_account_info()
        if account_info:
            logger.info(f"Portfolio value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"Buying power: ${account_info.get('buying_power', 0):,.2f}")
            logger.info(f"Cash: ${account_info.get('cash', 0):,.2f}")

        self.is_trading = True
        self.stop_trading_flag = False

        try:
            while not self.stop_trading_flag:
                self.run_single_check()

                # Sleep with interrupt checking
                for i in range(self.config.check_interval):
                    if self.stop_trading_flag:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            logger.error(f"Error type: {type(e).__name__}")
        finally:
            self.is_trading = False
            logger.info("Live trading stopped")

    def stop_live_trading(self):
        """Stop live trading"""
        logger.info("Stopping live trading...")
        self.stop_trading_flag = True

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            account_info = self.get_account_info()
            position = self.get_current_position()

            # Get recent orders
            try:
                orders = self.trading_client.get_orders()
                recent_orders = []
                for order in orders[:10]:  # Last 10 orders
                    recent_orders.append({
                        'id': str(order.id),
                        'symbol': order.symbol,
                        'side': order.side,
                        'qty': float(order.qty),
                        'status': order.status,
                        'filled_at': str(order.filled_at) if hasattr(order, 'filled_at') else None
                    })
            except Exception as e:
                logger.error(f"Error getting orders: {e}")
                recent_orders = []

            return {
                'account_info': account_info,
                'current_position': position,
                'recent_orders': recent_orders,
                'daily_trades': self.daily_trade_count,
                'last_signal': self.last_signal,
                'trading_active': self.is_trading,
                'symbol': self.symbol
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return {}

    def test_api_connection(self) -> bool:
        """Test API connection and trading capabilities"""
        try:
            logger.info("Testing API connection...")
            
            # Test account access
            account_info = self.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Account test passed - Portfolio: ${account_info.get('portfolio_value', 0):,.2f}")
            
            # Test position retrieval
            position = self.get_current_position()
            logger.info(f"Position test passed - Current position: {position}")
            
            # Test signal generation
            signal = self.generate_signal()
            if signal is None:
                logger.error("Failed to generate signal")
                return False
                
            logger.info(f"Signal test passed - Generated: {signal.signal} at ${signal.price:.2f}")
            
            logger.info("All API tests passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False


class AlpacaStrategyWrapper(Strategy):
    """
    Backtesting strategy wrapper for the Alpaca crypto bot
    """

    def init(self):
        """Initialize strategy variables"""
        self.signal = self.I(lambda x: x, self.data.signal)
        self.last_signal = 0

        # Track performance metrics
        self.trade_count = 0

    def next(self):
        """Execute trading logic on each bar"""
        current_signal = self.signal[-1]

        # Only act if signal has changed
        if current_signal != self.last_signal:
            current_price = self.data.Close[-1]

            # Buy signal: Enter long position
            if current_signal == 1 and not self.position:
                self.buy()
                self.trade_count += 1
                logger.info(f"BUY #{self.trade_count} at ${current_price:.2f}")

            # Sell signal: Close long position
            elif current_signal == -1 and self.position.is_long:
                self.position.close()
                self.trade_count += 1
                logger.info(f"SELL #{self.trade_count} at ${current_price:.2f}")

            self.last_signal = current_signal


class AlpacaBacktester:
    """
    Complete backtesting and live trading system with Alpaca integration
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper_trading: bool = True,
        **strategy_params
    ):
        """Initialize the system"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        self.strategy_params = strategy_params

        self.strategy_bot = AlpacaMegaCryptoBotFixed(
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=paper_trading,
            **strategy_params
        )

        # Live trading instances
        self.live_traders: Dict[str, AlpacaLiveTrader] = {}

    def create_live_trader(
        self,
        symbol: str,
        config: LiveTradingConfig = None
    ) -> AlpacaLiveTrader:
        """Create a live trader for a specific symbol"""
        trader = AlpacaLiveTrader(
            api_key=self.api_key,
            secret_key=self.secret_key,
            symbol=symbol,
            paper_trading=self.paper_trading,
            config=config or LiveTradingConfig(),
            **self.strategy_params
        )

        self.live_traders[symbol] = trader
        return trader

    def start_live_trading(
        self,
        symbol: str,
        config: LiveTradingConfig = None,
        background: bool = False
    ) -> Optional[AlpacaLiveTrader]:
        """
        Start live trading for a symbol

        Args:
            symbol: Trading symbol
            config: Live trading configuration
            background: Run in background thread

        Returns:
            AlpacaLiveTrader instance
        """
        if symbol not in self.live_traders:
            trader = self.create_live_trader(symbol, config)
        else:
            trader = self.live_traders[symbol]

        # Test connection before starting
        if not trader.test_api_connection():
            logger.error(f"API connection test failed for {symbol}")
            return None

        if background:
            # Run in background thread
            thread = threading.Thread(
                target=trader.start_live_trading,
                name=f"LiveTrader-{symbol}",
                daemon=True
            )
            thread.start()
            logger.info(f"Live trading started in background for {symbol}")
        else:
            # Run in main thread (blocking)
            trader.start_live_trading()

        return trader

    def stop_live_trading(self, symbol: str = None):
        """
        Stop live trading for symbol(s)

        Args:
            symbol: Specific symbol to stop, or None for all
        """
        if symbol:
            if symbol in self.live_traders:
                self.live_traders[symbol].stop_live_trading()
                logger.info(f"Stopped live trading for {symbol}")
        else:
            # Stop all live traders
            for sym, trader in self.live_traders.items():
                trader.stop_live_trading()
                logger.info(f"Stopped live trading for {sym}")

    def get_live_trading_summary(self) -> Dict[str, Any]:
        """Get summary of all live trading activities"""
        summary = {
            'active_traders': len(self.live_traders),
            'symbols': list(self.live_traders.keys()),
            'trader_summaries': {}
        }

        for symbol, trader in self.live_traders.items():
            summary['trader_summaries'][symbol] = trader.get_performance_summary()

        return summary

    def run_backtest(
        self,
        symbol: str = "BTC/USD",
        start_date: str = "2024-01-01",
        end_date: str = None,
        initial_cash: float = None,
        commission: float = 0.002
    ) -> Tuple[Any, pd.DataFrame]:
        """
        Run complete backtest

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_cash: Starting capital
            commission: Trading commission rate

        Returns:
            Tuple of (backtest_stats, strategy_data)
        """
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Period: {start_date} to {end_date or 'today'}")

        try:
            # --- 1. Get Data ---
            if "/" in symbol or "USD" in symbol.upper():
                data = self.strategy_bot.get_crypto_data(symbol, start_date, end_date)
            else:
                data = self.strategy_bot.get_stock_data(symbol, start_date, end_date)

            if data.empty:
                logger.error("No data available for backtesting")
                return None, pd.DataFrame()

            logger.info(f"Data loaded: {len(data)} bars")

            # --- 2. Calculate Indicators ---
            strategy_data = self.strategy_bot.calculate_indicators(data)

            # --- 3. Analyze Signals ---
            signal_analysis = self.strategy_bot.analyze_signals(strategy_data)

            if signal_analysis:
                logger.info("Signal Analysis:")
                for signal_name, stats in signal_analysis['signal_distribution'].items():
                    logger.info(f"   {signal_name}: {stats['count']:,} ({stats['percentage']:.2f}%)")

                logger.info("Trading Opportunities:")
                logger.info(f"   Signal changes: {signal_analysis['signal_changes']}")
                logger.info(f"   Buy signals: {signal_analysis['buy_signals']}")
                logger.info(f"   Sell signals: {signal_analysis['sell_signals']}")

                if signal_analysis['signal_changes'] == 0:
                    logger.warning("No trading signals generated!")
                    return None, strategy_data

            # --- 4. Setup Backtest ---
            if initial_cash is None:
                max_price = strategy_data['Close'].max()
                if "/" in symbol or "BTC" in symbol or "ETH" in symbol:
                    initial_cash = max(100000, max_price * 3)  # Crypto
                else:
                    initial_cash = max(50000, max_price * 100)  # Stocks

            logger.info(f"Initial capital: ${initial_cash:,.2f}")
            logger.info(f"Max {symbol} price: ${strategy_data['Close'].max():,.2f}")

            # --- 5. Run Backtest ---
            bt = Backtest(
                strategy_data,
                AlpacaStrategyWrapper,
                cash=initial_cash,
                commission=commission,
                trade_on_close=True
            )

            stats = bt.run()

            # --- 6. Performance Analysis ---
            self._print_results(stats, strategy_data, symbol)

            # --- 7. Generate Plot ---
            logger.info("Generating performance plot...")
            bt.plot()

            return stats, strategy_data

        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None, pd.DataFrame()

    def _print_results(self, stats: Any, data: pd.DataFrame, symbol: str):
        """Print formatted backtest results"""
        logger.info("="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)

        # Basic performance metrics
        logger.info(f"Return: {stats['Return [%]']:.2f}%")
        logger.info(f"Buy & Hold: {self._calculate_buy_hold_return(data):.2f}%")
        logger.info(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")

        # Trading statistics
        logger.info("Trading Stats:")
        logger.info(f"   Total Trades: {stats['# Trades']}")

        if stats['# Trades'] > 0:
            logger.info(f"   Win Rate: {stats['Win Rate [%]']:.2f}%")
            logger.info(f"   Avg Trade: {stats['Avg. Trade [%]']:.2f}%")
            logger.info(f"   Best Trade: {stats['Best Trade [%]']:.2f}%")
            logger.info(f"   Worst Trade: {stats['Worst Trade [%]']:.2f}%")
            logger.info(f"   Avg Duration: {stats['Avg. Trade Duration']}")
        else:
            logger.warning("   No trades executed!")

        logger.info("="*50)

    def _calculate_buy_hold_return(self, data: pd.DataFrame) -> float:
        """Calculate buy and hold return for comparison"""
        try:
            initial_price = data['Close'].iloc[0]
            final_price = data['Close'].iloc[-1]
            return ((final_price - initial_price) / initial_price) * 100
        except:
            return 0.0


# ==============================================================================
# EXAMPLE USAGE AND DEMOS
# ==============================================================================
def demo_backtest():
    """Demo: Run a backtest"""
    logger.info("Running backtest demo...")

    backtester = AlpacaBacktester(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper_trading=True,
        # Strategy parameters
        len_cbrc=30,
        signal_period='480min'
    )

    stats, data = backtester.run_backtest(
        symbol="BTC/USD",
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_cash=100000
    )

    return stats, data


def demo_live_trading():
    """Demo: Live trading setup"""
    logger.info("Setting up live trading demo...")

    # Initialize system
    system = AlpacaBacktester(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper_trading=True,
        # Strategy parameters
        len_cbrc=25,
        signal_period='360min'
    )

    # Configure live trading
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
        config=config,
        background=False  # Run in main thread
    )

    return system


def demo_multi_symbol_trading():
    """Demo: Multiple symbol live trading"""
    logger.info("Setting up multi-symbol trading demo...")

    system = AlpacaBacktester(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper_trading=True
    )

    # Different configs for different assets
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
            background=True  # Run all in background
        )

    # Monitor all traders
    try:
        while True:
            summary = system.get_live_trading_summary()
            logger.info(f"Active traders: {summary['active_traders']}")
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Stopping all traders...")
        system.stop_live_trading()

    return system


def main():
    """Main function with usage examples"""
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca Trading System")
    parser.add_argument("--mode", choices=["backtest", "live", "multi", "test"],
                       default="backtest", help="Operation mode")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--paper", action="store_true", default=True,
                       help="Use paper trading")

    args = parser.parse_args()

    if args.mode == "test":
        # Test API connection
        logger.info("Testing API connection...")
        system = AlpacaBacktester(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper_trading=True
        )
        
        trader = system.create_live_trader(args.symbol)
        success = trader.test_api_connection()
        logger.info(f"API test result: {'PASSED' if success else 'FAILED'}")

    elif args.mode == "backtest":
        # Run backtest
        stats, data = demo_backtest()
        if stats:
            logger.info("Backtest completed!")

    elif args.mode == "live":
        # Single symbol live trading
        logger.info("Starting live trading...")
        logger.warning("Make sure you're using paper trading first!")
        system = demo_live_trading()

    elif args.mode == "multi":
        # Multi-symbol live trading
        logger.info("Starting multi-symbol trading...")
        logger.warning("Make sure you're using paper trading first!")
        system = demo_multi_symbol_trading()


if __name__ == "__main__":
    main()
