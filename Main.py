"""
Unified Trading Integration System
Combines NotebookTradingSystem and AlpacaMegaCryptoBotFixed with live, backtest, and scan modes

WARNING: This system shows unrealistic profit projections in Monte Carlo testing.
The 100% profit rate and 35%+ returns are not achievable in real trading.
Use only for educational purposes and paper trading.
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required components (with fallbacks)
try:
    # Import original Alpaca system
    from backtest import AlpacaMegaCryptoBotFixed, AlpacaLiveTrader, LiveTradingConfig
    HAS_ORIGINAL_SYSTEM = True
except ImportError:
    HAS_ORIGINAL_SYSTEM = False
    logger.warning("Original Alpaca system not found - using fallback implementations")

try:
    # Import notebook system
    from magnaNotebook import NotebookTradingSystem, NotebookTradingConfig
    HAS_NOTEBOOK_SYSTEM = True
except ImportError:
    HAS_NOTEBOOK_SYSTEM = False
    logger.warning("Notebook system not found - using fallback implementations")

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    logger.warning("Alpaca SDK not found - install with: pip install alpaca-py")


@dataclass
class UnifiedTradingConfig:
    """Unified configuration for all trading modes"""
    # Capital settings
    initial_capital: float = 100.0
    max_position_size: float = 0.8
    risk_per_trade: float = 0.10
    
    # Trading parameters
    take_profit_pct: float = 0.08
    stop_loss_pct: float = 0.04
    max_daily_trades: int = 20
    min_trade_interval: int = 300  # 5 minutes
    
    # Target symbols
    target_symbols: List[str] = None
    
    # API credentials
    api_key: str = None
    secret_key: str = None
    paper_trading: bool = True
    
    # System selection
    use_notebook_system: bool = True
    use_original_system: bool = False
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        
        if self.api_key is None:
            self.api_key = os.getenv('ALPACA_API_KEY')
        
        if self.secret_key is None:
            self.secret_key = os.getenv('ALPACA_SECRET_KEY')


class UnifiedTradingSystem:
    """
    Unified system that integrates both NotebookTradingSystem and AlpacaMegaCryptoBotFixed
    Supports live trading, backtesting, and scanning modes
    """
    
    def __init__(self, config: UnifiedTradingConfig = None):
        self.config = config or UnifiedTradingConfig()
        
        # Initialize systems based on configuration
        self.notebook_system = None
        self.original_system = None
        self.alpaca_client = None
        
        # Trading state
        self.is_running = False
        self.mode = "scan"  # Default mode
        self.last_signals = {}
        self.performance_history = []
        
        # Initialize systems
        self._initialize_systems()
        
        logger.info("Unified Trading System initialized")
        logger.info(f"Capital: ${self.config.initial_capital}")
        logger.info(f"Paper trading: {self.config.paper_trading}")
        logger.info(f"Target symbols: {self.config.target_symbols}")
    
    def _initialize_systems(self):
        """Initialize the trading systems"""
        # Initialize notebook system
        if self.config.use_notebook_system and HAS_NOTEBOOK_SYSTEM:
            notebook_config = NotebookTradingConfig(
                initial_capital=self.config.initial_capital,
                max_position_size=self.config.max_position_size,
                risk_per_trade=self.config.risk_per_trade,
                target_symbols=self.config.target_symbols
            )
            self.notebook_system = NotebookTradingSystem(notebook_config)
            logger.info("Notebook system initialized")
        
        # Initialize original system
        if self.config.use_original_system and HAS_ORIGINAL_SYSTEM:
            self.original_system = AlpacaMegaCryptoBotFixed(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper_trading=self.config.paper_trading
            )
            logger.info("Original Alpaca system initialized")
        
        # Initialize Alpaca client for live trading
        if self.config.api_key and self.config.secret_key and HAS_ALPACA:
            self.alpaca_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper_trading
            )
            logger.info("Alpaca client initialized")
    
    def get_market_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get market data for analysis"""
        try:
            if self.original_system:
                # Use original system's data fetching
                if "/" in symbol:
                    return self.original_system.get_crypto_data(
                        symbol=symbol,
                        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                    )
                else:
                    return self.original_system.get_stock_data(
                        symbol=symbol,
                        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                    )
            elif self.notebook_system:
                # Use notebook system's data generation
                return self.notebook_system.generate_sample_data(symbol, days)
            else:
                # Fallback data generation
                return self._generate_fallback_data(symbol, days)
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return self._generate_fallback_data(symbol, days)
    
    def _generate_fallback_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate fallback market data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start_date, end_date, freq='H')
        
        # Basic price simulation
        base_price = 45000 if "BTC" in symbol else 2500 if "ETH" in symbol else 100
        n_periods = len(dates)
        
        np.random.seed(hash(symbol) % 2**32)
        prices = base_price * (1 + np.random.normal(0, 0.02, n_periods).cumsum() * 0.1)
        prices = np.maximum(prices, base_price * 0.5)
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(prices[0])
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.01
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.99
        df['Volume'] = np.random.lognormal(8, 1, n_periods)
        
        return df
    
    def generate_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate unified trading signal"""
        try:
            # Get market data
            data = self.get_market_data(symbol, days=7)
            
            if data.empty or len(data) < 50:
                return {
                    'symbol': symbol,
                    'signal': 0,
                    'confidence': 0.5,
                    'price': 0,
                    'source': 'insufficient_data'
                }
            
            current_price = data['Close'].iloc[-1]
            signals = {}
            
            # Get notebook system signal
            if self.notebook_system:
                try:
                    tech_data = self.notebook_system.calculate_technical_indicators(data)
                    nb_signal, nb_confidence = self.notebook_system.generate_ai_signal(tech_data)
                    signals['notebook'] = {
                        'signal': nb_signal,
                        'confidence': nb_confidence,
                        'weight': 0.6
                    }
                except Exception as e:
                    logger.warning(f"Notebook signal error for {symbol}: {e}")
            
            # Get original system signal
            if self.original_system:
                try:
                    strategy_data = self.original_system.calculate_indicators(data)
                    if 'signal' in strategy_data.columns:
                        orig_signal = int(strategy_data['signal'].iloc[-1])
                        # Estimate confidence based on signal consistency
                        recent_signals = strategy_data['signal'].tail(5).values
                        consistency = np.mean(recent_signals == orig_signal)
                        signals['original'] = {
                            'signal': orig_signal,
                            'confidence': consistency,
                            'weight': 0.4
                        }
                except Exception as e:
                    logger.warning(f"Original signal error for {symbol}: {e}")
            
            # Combine signals
            if not signals:
                return {
                    'symbol': symbol,
                    'signal': 0,
                    'confidence': 0.5,
                    'price': current_price,
                    'source': 'no_signals'
                }
            
            # Weighted signal combination
            weighted_signal = 0
            total_weight = 0
            combined_confidence = 0
            
            for source, sig_data in signals.items():
                weight = sig_data['weight'] * sig_data['confidence']
                weighted_signal += sig_data['signal'] * weight
                total_weight += weight
                combined_confidence += sig_data['confidence'] * sig_data['weight']
            
            if total_weight > 0:
                final_signal = int(np.round(weighted_signal / total_weight))
                final_confidence = combined_confidence / sum(s['weight'] for s in signals.values())
            else:
                final_signal = 0
                final_confidence = 0.5
            
            return {
                'symbol': symbol,
                'signal': final_signal,
                'confidence': final_confidence,
                'price': current_price,
                'source': 'combined',
                'individual_signals': signals,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 0,
                'confidence': 0.5,
                'price': 0,
                'source': 'error',
                'error': str(e)
            }
    
    def scan_mode(self) -> Dict[str, Any]:
        """Scan mode - analyze symbols without trading"""
        logger.info("Running scan mode...")
        
        scan_results = {
            'timestamp': datetime.now(),
            'mode': 'scan',
            'signals': {},
            'summary': {}
        }
        
        # Generate signals for all symbols
        for symbol in self.config.target_symbols:
            signal_data = self.generate_trading_signal(symbol)
            scan_results['signals'][symbol] = signal_data
            
            logger.info(f"Scan {symbol}: Signal={signal_data['signal']}, "
                       f"Confidence={signal_data['confidence']:.2f}, "
                       f"Price=${signal_data['price']:.2f}")
        
        # Generate summary
        signals = [s['signal'] for s in scan_results['signals'].values()]
        confidences = [s['confidence'] for s in scan_results['signals'].values()]
        
        scan_results['summary'] = {
            'total_symbols': len(self.config.target_symbols),
            'buy_signals': sum(1 for s in signals if s == 1),
            'sell_signals': sum(1 for s in signals if s == -1),
            'hold_signals': sum(1 for s in signals if s == 0),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'high_confidence_signals': sum(1 for c in confidences if c > 0.7)
        }
        
        logger.info(f"Scan summary: {scan_results['summary']['buy_signals']} buy, "
                   f"{scan_results['summary']['sell_signals']} sell, "
                   f"avg confidence: {scan_results['summary']['avg_confidence']:.2f}")
        
        return scan_results
    
    def backtest_mode(self, days: int = 30, symbol: str = "BTC/USD") -> Dict[str, Any]:
        """Backtest mode - historical performance testing"""
        logger.info(f"Running backtest for {symbol} over {days} days...")
        
        # Get historical data
        data = self.get_market_data(symbol, days)
        
        if data.empty or len(data) < 100:
            logger.error("Insufficient data for backtesting")
            return {'error': 'insufficient_data'}
        
        # Initialize backtest variables
        initial_capital = self.config.initial_capital
        portfolio_value = initial_capital
        positions = {}
        trades = []
        daily_values = [initial_capital]
        
        # Run backtest simulation
        for i in range(50, len(data)):  # Start after enough data for indicators
            current_data = data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            
            # Generate signal using current system
            if self.notebook_system:
                tech_data = self.notebook_system.calculate_technical_indicators(current_data)
                signal, confidence = self.notebook_system.generate_ai_signal(tech_data)
            else:
                # Fallback signal generation
                signal = 0
                confidence = 0.5
            
            # Execute trades based on signals
            if signal != 0 and confidence > 0.6:
                position_size = portfolio_value * self.config.max_position_size
                quantity = position_size / current_price
                
                if signal == 1:  # Buy
                    if quantity * current_price <= portfolio_value:
                        portfolio_value -= quantity * current_price
                        positions[symbol] = positions.get(symbol, 0) + quantity
                        trades.append({
                            'timestamp': current_data.index[-1],
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price,
                            'value': quantity * current_price
                        })
                
                elif signal == -1:  # Sell
                    if symbol in positions and positions[symbol] > 0:
                        sell_quantity = min(quantity, positions[symbol])
                        portfolio_value += sell_quantity * current_price
                        positions[symbol] -= sell_quantity
                        trades.append({
                            'timestamp': current_data.index[-1],
                            'action': 'SELL',
                            'quantity': sell_quantity,
                            'price': current_price,
                            'value': sell_quantity * current_price
                        })
            
            # Calculate current portfolio value
            position_value = sum(qty * current_price for qty in positions.values())
            total_value = portfolio_value + position_value
            daily_values.append(total_value)
        
        # Calculate performance metrics
        final_value = daily_values[-1]
        total_return = final_value - initial_capital
        return_pct = (total_return / initial_capital) * 100
        max_value = max(daily_values)
        max_drawdown = (max_value - min(daily_values)) / max_value * 100
        
        backtest_results = {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'trades': trades,
            'daily_values': daily_values,
            'positions': positions
        }
        
        logger.info(f"Backtest results: {return_pct:+.1f}% return, "
                   f"{len(trades)} trades, {max_drawdown:.1f}% max drawdown")
        
        return backtest_results
    
    def live_mode(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Live trading mode - execute real trades"""
        logger.warning("LIVE MODE ACTIVATED - This will execute real trades!")
        logger.warning(f"Paper trading: {self.config.paper_trading}")
        
        if not self.config.paper_trading:
            confirm = input("You are about to trade with REAL MONEY. Type 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                logger.info("Live trading cancelled")
                return {'status': 'cancelled'}
        
        self.is_running = True
        self.mode = "live"
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        live_results = {
            'start_time': start_time,
            'end_time': end_time,
            'initial_capital': self.config.initial_capital,
            'trades': [],
            'signals': [],
            'errors': []
        }
        
        trade_count = 0
        last_trade_time = datetime.min
        
        try:
            while self.is_running and datetime.now() < end_time:
                cycle_start = datetime.now()
                
                # Check trading limits
                if trade_count >= self.config.max_daily_trades:
                    logger.warning("Daily trade limit reached")
                    break
                
                # Check minimum time interval
                if (datetime.now() - last_trade_time).seconds < self.config.min_trade_interval:
                    time.sleep(10)
                    continue
                
                # Scan for signals
                scan_results = self.scan_mode()
                live_results['signals'].append(scan_results)
                
                # Execute trades for high-confidence signals
                for symbol, signal_data in scan_results['signals'].items():
                    if (signal_data['signal'] != 0 and 
                        signal_data['confidence'] > 0.7 and
                        signal_data['signal'] != self.last_signals.get(symbol, 0)):
                        
                        # Execute trade
                        trade_result = self._execute_live_trade(symbol, signal_data)
                        
                        if trade_result.get('success'):
                            live_results['trades'].append(trade_result)
                            self.last_signals[symbol] = signal_data['signal']
                            trade_count += 1
                            last_trade_time = datetime.now()
                            logger.info(f"Trade executed: {trade_result}")
                        else:
                            live_results['errors'].append(trade_result)
                
                # Sleep until next check
                sleep_time = max(30, 60 - (datetime.now() - cycle_start).seconds)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Live trading interrupted by user")
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            live_results['errors'].append(str(e))
        finally:
            self.is_running = False
            logger.info("Live trading session ended")
        
        return live_results
    
    def _execute_live_trade(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """Execute a live trade"""
        try:
            if not self.alpaca_client:
                # Simulation mode
                return {
                    'symbol': symbol,
                    'action': 'BUY' if signal_data['signal'] == 1 else 'SELL',
                    'price': signal_data['price'],
                    'simulation': True,
                    'success': True,
                    'timestamp': datetime.now()
                }
            
            # Calculate position size
            account = self.alpaca_client.get_account()
            portfolio_value = float(account.portfolio_value)
            position_value = portfolio_value * self.config.max_position_size * signal_data['confidence']
            quantity = position_value / signal_data['price']
            
            # Round quantity appropriately
            if "/" in symbol:  # Crypto
                quantity = round(quantity, 6)
            else:  # Stock
                quantity = int(quantity)
            
            if quantity <= 0:
                return {'success': False, 'reason': 'Invalid quantity'}
            
            # Create order
            side = OrderSide.BUY if signal_data['signal'] == 1 else OrderSide.SELL
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order
            order = self.alpaca_client.submit_order(order_data=order_request)
            
            return {
                'symbol': symbol,
                'action': side.value,
                'quantity': quantity,
                'price': signal_data['price'],
                'order_id': str(order.id),
                'success': True,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing live trade for {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def stop_trading(self):
        """Stop live trading"""
        self.is_running = False
        logger.info("Stopping trading...")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            report = {
                'timestamp': datetime.now(),
                'config': {
                    'initial_capital': self.config.initial_capital,
                    'max_position_size': self.config.max_position_size,
                    'risk_per_trade': self.config.risk_per_trade,
                    'target_symbols': self.config.target_symbols
                },
                'systems': {
                    'notebook_available': self.notebook_system is not None,
                    'original_available': self.original_system is not None,
                    'alpaca_available': self.alpaca_client is not None
                },
                'mode': self.mode,
                'is_running': self.is_running
            }
            
            # Add account info if available
            if self.alpaca_client:
                try:
                    account = self.alpaca_client.get_account()
                    report['account'] = {
                        'portfolio_value': float(account.portfolio_value),
                        'buying_power': float(account.buying_power),
                        'cash': float(account.cash)
                    }
                except:
                    pass
            
            return report
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions for different modes
def run_scan(symbols: List[str] = None, config: UnifiedTradingConfig = None) -> Dict[str, Any]:
    """Quick scan function"""
    if symbols:
        if config is None:
            config = UnifiedTradingConfig()
        config.target_symbols = symbols
    
    system = UnifiedTradingSystem(config)
    return system.scan_mode()


def run_backtest(symbol: str = "BTC/USD", days: int = 30, capital: float = 100.0) -> Dict[str, Any]:
    """Quick backtest function"""
    config = UnifiedTradingConfig(
        initial_capital=capital,
        target_symbols=[symbol]
    )
    
    system = UnifiedTradingSystem(config)
    return system.backtest_mode(days, symbol)


def run_live_trading(duration_hours: int = 1, config: UnifiedTradingConfig = None) -> Dict[str, Any]:
    """Quick live trading function"""
    if config is None:
        config = UnifiedTradingConfig()
    
    system = UnifiedTradingSystem(config)
    return system.live_mode(duration_hours)


def main():
    """Main function with mode selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Trading System")
    parser.add_argument("--mode", choices=["scan", "backtest", "live"], 
                       default="scan", help="Trading mode")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=100.0, help="Initial capital")
    parser.add_argument("--days", type=int, default=7, help="Days for backtest")
    parser.add_argument("--duration", type=int, default=1, help="Live trading duration (hours)")
    
    args = parser.parse_args()
    
    # Show critical warning
    print("\n" + "="*60)
    print("CRITICAL WARNING: UNREALISTIC PROFIT EXPECTATIONS")
    print("="*60)
    print("Your Monte Carlo results showing 100% profit probability")
    print("and 35%+ returns are NOT realistic for actual trading.")
    print("This system has significant simulation biases that create")
    print("false confidence. Real trading typically results in losses.")
    print("Use ONLY for educational purposes and paper trading.")
    print("="*60)
    
    #user_confirm = input("\nDo you understand these risks and want to continue? (yes/no): ")
    user_confirm = 'yes' 
    if user_confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Configure system
    config = UnifiedTradingConfig(
        initial_capital=args.capital,
        target_symbols=[args.symbol] if args.mode == "backtest" else ["BTC/USD", "ETH/USD", "SOL/USD"],
        paper_trading=True  # ALWAYS default to paper trading
    )
    
    system = UnifiedTradingSystem(config)
    
    # Run selected mode
    if args.mode == "scan":
        results = system.scan_mode()
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == "backtest":
        results = system.backtest_mode(args.days, args.symbol)
        if 'error' not in results:
            print(f"\nBacktest Results for {args.symbol}:")
            print(f"Return: {results['return_pct']:+.1f}%")
            print(f"Trades: {results['total_trades']}")
            print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
        else:
            print(f"Backtest failed: {results['error']}")
    
    elif args.mode == "live":
        print(f"Starting live trading for {args.duration} hour(s)...")
        print("Press Ctrl+C to stop trading early")
        results = system.live_mode(args.duration)
        print(f"\nLive trading completed:")
        print(f"Trades executed: {len(results.get('trades', []))}")
        print(f"Errors: {len(results.get('errors', []))}")


if __name__ == "__main__":
    main()
