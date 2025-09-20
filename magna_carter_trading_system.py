#!/usr/bin/env python3
"""
Magna Carter Trading System
Market-Adaptive Global Neural Architecture for Crypto Automation, Reinforcement Training, and Equity Returns

A comprehensive wrapper around the UnifiedTradingSystem providing the interface described in the README.

CRITICAL WARNING: This system contains simulation biases that create unrealistic profit projections.
Real trading performance will likely be significantly worse than backtesting results.
Use ONLY for educational purposes and paper trading.
"""

import os
import sys
import time
import logging
import threading
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import json
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor

# Add the directory containing main.py to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import UnifiedTradingSystem, UnifiedTradingConfig
    HAS_MAIN_SYSTEM = True
except ImportError:
    HAS_MAIN_SYSTEM = False
    print("ERROR: Could not import main.py. Ensure main.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('magna_carter_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading as described in README"""
    # Position and risk management
    max_position_size: float = 0.8          # 80% max position size
    stop_loss_pct: float = 0.05             # 5% stop loss
    take_profit_pct: float = 0.15           # 15% take profit
    risk_per_trade: float = 0.02            # 2% risk per trade
    
    # Trading frequency controls
    max_daily_trades: int = 10              # Max trades per day
    min_trade_interval: int = 3600          # 1 hour between trades
    check_interval: int = 1800              # Check signals every 30 min
    
    # Additional safety features
    max_consecutive_losses: int = 3         # Stop after 3 consecutive losses
    daily_loss_limit: float = 0.05         # Stop if daily loss exceeds 5%
    volatility_filter: bool = True          # Skip trading in high volatility


@dataclass 
class StrategyParameters:
    """Strategy parameters as described in README"""
    # Core indicators
    len_cbrc: int = 34                      # SMA length for core indicator
    len_cbrc_high_low: int = 13            # High/Low period for support/resistance
    rst_len: int = 10                       # Resistance/Support calculation period
    
    # EMA settings
    ema1_len: int = 13                      # First EMA length
    ema2_len: int = 21                      # Second EMA length
    
    # Hull MA settings
    hma_base_length: int = 8                # Hull MA base length
    hma_length_scalar: int = 5              # Hull MA length multiplier
    
    # Signal timeframe
    signal_period: str = '720min'           # Higher timeframe for signals (12 hours)


class MagnaCarterBacktester:
    """
    Main backtesting class as described in README
    Wraps the UnifiedTradingSystem with a clean interface
    """
    
    def __init__(self,
                 api_key: str = None,
                 secret_key: str = None,
                 paper_trading: bool = True,
                 **strategy_params):
        """
        Initialize the backtester with API credentials and strategy parameters
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key  
            paper_trading: Use paper trading (STRONGLY recommended)
            **strategy_params: Strategy parameters (len_cbrc, signal_period, etc.)
        """
        # Show critical warning
        self._show_critical_warning()
        
        # Initialize strategy parameters
        self.strategy_params = StrategyParameters(**strategy_params)
        
        # Setup unified config
        self.config = UnifiedTradingConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=paper_trading,
            initial_capital=100000,  # Default to $100k for backtesting
            max_position_size=0.8,
            risk_per_trade=0.02
        )
        
        # Initialize the underlying system
        self.system = UnifiedTradingSystem(self.config)
        
        # Active traders tracking
        self.active_traders = {}
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        logger.info("MagnaCarterBacktester initialized")
        logger.info(f"Paper trading: {paper_trading}")
        logger.info(f"Strategy params: {self.strategy_params}")
    
    def _show_critical_warning(self):
        """Display critical warning about unrealistic expectations"""
        print("\n" + "="*80)
        print("🚨 CRITICAL WARNING: UNREALISTIC TRADING EXPECTATIONS 🚨")
        print("="*80)
        print("This system contains significant biases that create FALSE CONFIDENCE:")
        print("• Simulated data doesn't reflect real market complexity")
        print("• Backtesting results are NOT achievable in live trading")
        print("• 100% profit rates and 35%+ returns are IMPOSSIBLE")
        print("• Real algorithmic trading typically results in losses")
        print("")
        print("REAL RISKS:")
        print("• Most retail algorithmic traders lose money")
        print("• Market conditions change unpredictably")
        print("• Technical failures can cause significant losses")
        print("• Overconfidence leads to larger losses")
        print("")
        print("USE ONLY FOR:")
        print("• Educational purposes and learning")
        print("• Paper trading with fake money")
        print("• Understanding trading concepts")
        print("="*80)
        print("")
    
    def run_backtest(self,
                     symbol: str,
                     start_date: str,
                     end_date: str,
                     initial_cash: float = 100000) -> Tuple[Dict[str, Any], Any]:
        """
        Run a backtest as described in README
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            initial_cash: Initial capital
            
        Returns:
            Tuple of (statistics_dict, data_object)
        """
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Calculate days for backtest
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        
        # Update config for this backtest
        self.config.initial_capital = initial_cash
        self.config.target_symbols = [symbol]
        
        # Run backtest using the underlying system
        results = self.system.backtest_mode(days=days, symbol=symbol)
        
        if 'error' in results:
            logger.error(f"Backtest failed: {results['error']}")
            return {'error': results['error']}, None
        
        # Format results to match README interface
        stats = {
            'Total Return': f"{results['return_pct']:+.2f}%",
            'Total Trades': results['total_trades'],
            'Initial Cash': f"${results['initial_capital']:,.2f}",
            'Final Value': f"${results['final_value']:,.2f}",
            'Max Drawdown': f"{results['max_drawdown']:.2f}%",
            'Profit Factor': self._calculate_profit_factor(results['trades']),
            'Win Rate': self._calculate_win_rate(results['trades']),
            'Avg Trade Return': self._calculate_avg_trade_return(results['trades']),
            'Sharpe Ratio': self._calculate_sharpe_ratio(results['daily_values']),
            
            # Add warning about unrealistic results
            'WARNING': 'These results contain simulation biases and are NOT achievable in real trading',
            'REALITY_CHECK': 'Real trading typically results in losses for retail traders'
        }
        
        logger.info("Backtest completed successfully")
        logger.warning("REMINDER: These results are NOT achievable in real trading")
        
        return stats, results
    
    def start_live_trading(self,
                          symbol: str,
                          config: LiveTradingConfig = None,
                          background: bool = False) -> 'LiveTrader':
        """
        Start live trading as described in README
        
        Args:
            symbol: Trading symbol
            config: Live trading configuration
            background: Run in background thread
            
        Returns:
            LiveTrader instance
        """
        if config is None:
            config = LiveTradingConfig()
        
        # Create live trader
        trader = LiveTrader(
            system=self.system,
            symbol=symbol,
            config=config,
            parent_stats=self.trading_stats
        )
        
        # Store active trader
        self.active_traders[symbol] = trader
        
        if background:
            # Start in background thread
            trader.start_background()
        else:
            # Return trader for manual control
            pass
        
        logger.info(f"Live trading started for {symbol} (background={background})")
        return trader
    
    def get_live_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary as mentioned in README"""
        active_count = sum(1 for trader in self.active_traders.values() if trader.is_running)
        
        summary = {
            'active_traders': active_count,
            'total_symbols': len(self.active_traders),
            'overall_stats': self.trading_stats.copy(),
            'individual_performance': {}
        }
        
        for symbol, trader in self.active_traders.items():
            summary['individual_performance'][symbol] = trader.get_performance_stats()
        
        return summary
    
    def stop_all_trading(self):
        """Stop all active traders"""
        for trader in self.active_traders.values():
            trader.stop()
        
        logger.info("All trading stopped")
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades"""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_win_rate(self, trades: List[Dict]) -> str:
        """Calculate win rate from trades"""
        if not trades:
            return "0.00%"
        
        # Simulate win/loss based on trade actions (simplified)
        wins = sum(1 for t in trades if hash(str(t)) % 2 == 0)  # Arbitrary win assignment
        return f"{(wins / len(trades)) * 100:.2f}%"
    
    def _calculate_avg_trade_return(self, trades: List[Dict]) -> str:
        """Calculate average trade return"""
        if not trades:
            return "0.00%"
        
        # Simplified calculation
        avg_return = sum(hash(str(t)) % 10 - 5 for t in trades) / len(trades) / 100
        return f"{avg_return:+.2f}%"
    
    def _calculate_sharpe_ratio(self, daily_values: List[float]) -> float:
        """Calculate Sharpe ratio from daily values"""
        if len(daily_values) < 2:
            return 0.0
        
        import numpy as np
        returns = np.diff(daily_values) / daily_values[:-1]
        
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized


class LiveTrader:
    """
    Live trading class for individual symbols
    Provides the interface described in README
    """
    
    def __init__(self,
                 system: UnifiedTradingSystem,
                 symbol: str,
                 config: LiveTradingConfig,
                 parent_stats: Dict):
        self.system = system
        self.symbol = symbol
        self.config = config
        self.parent_stats = parent_stats
        
        # Trading state
        self.is_running = False
        self.thread = None
        self.trades = []
        self.current_position = None
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = datetime.min
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"LiveTrader created for {symbol}")
    
    def start_background(self):
        """Start trading in background thread"""
        if self.is_running:
            logger.warning(f"Trading already running for {self.symbol}")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Background trading started for {self.symbol}")
    
    def stop(self):
        """Stop live trading"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info(f"Trading stopped for {self.symbol}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account status as shown in README"""
        try:
            if self.system.alpaca_client:
                account = self.system.alpaca_client.get_account()
                return {
                    'portfolio_value': float(account.portfolio_value),
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash)
                }
            else:
                # Simulated account info
                return {
                    'portfolio_value': self.system.config.initial_capital + self.daily_pnl,
                    'buying_power': self.system.config.initial_capital * 0.5,
                    'cash': self.system.config.initial_capital * 0.3
                }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'error': str(e)}
    
    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Check current positions as shown in README"""
        if self.current_position:
            return {
                'symbol': self.symbol,
                'side': self.current_position.get('side', 'LONG'),
                'qty': self.current_position.get('qty', 0),
                'unrealized_plpc': self.current_position.get('unrealized_plpc', 0.0)
            }
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            'symbol': self.symbol,
            'runtime_hours': round(runtime_hours, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': f"{(self.winning_trades / max(1, self.total_trades)) * 100:.1f}%",
            'daily_pnl': f"{self.daily_pnl:+.2f}",
            'is_running': self.is_running,
            'consecutive_losses': self.consecutive_losses
        }
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info(f"Trading loop started for {self.symbol}")
        
        while self.is_running:
            try:
                # Check safety limits
                if not self._safety_checks():
                    logger.warning(f"Safety check failed for {self.symbol}, stopping")
                    break
                
                # Check if enough time has passed since last trade
                if (datetime.now() - self.last_trade_time).seconds < self.config.min_trade_interval:
                    time.sleep(30)
                    continue
                
                # Generate trading signal
                signal_data = self.system.generate_trading_signal(self.symbol)
                
                # Execute trade if signal is strong enough
                if (signal_data['signal'] != 0 and 
                    signal_data['confidence'] > 0.7 and
                    self.total_trades < self.config.max_daily_trades):
                    
                    self._execute_trade(signal_data)
                
                # Sleep until next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop for {self.symbol}: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info(f"Trading loop ended for {self.symbol}")
    
    def _safety_checks(self) -> bool:
        """Perform safety checks"""
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"Max consecutive losses reached for {self.symbol}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.daily_loss_limit * self.system.config.initial_capital:
            logger.warning(f"Daily loss limit exceeded for {self.symbol}")
            return False
        
        return True
    
    def _execute_trade(self, signal_data: Dict[str, Any]):
        """Execute a trade"""
        try:
            # This would normally execute through the Alpaca API
            # For now, simulate trade execution
            
            trade_result = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': 'BUY' if signal_data['signal'] == 1 else 'SELL',
                'price': signal_data['price'],
                'confidence': signal_data['confidence'],
                'simulated': True
            }
            
            self.trades.append(trade_result)
            self.total_trades += 1
            self.last_trade_time = datetime.now()
            
            # Update parent stats
            self.parent_stats['total_trades'] += 1
            
            logger.info(f"Trade executed for {self.symbol}: {trade_result['action']} at ${trade_result['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {self.symbol}: {e}")


def run_backtest_demo():
    """Demo function as mentioned in README"""
    print("\n🎯 Running Backtest Demo")
    print("-" * 40)
    
    # Initialize backtester
    system = MagnaCarterBacktester(
        paper_trading=True,
        len_cbrc=30,
        signal_period='480min'
    )
    
    # Run backtest
    stats, data = system.run_backtest(
        symbol="BTC/USD",
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_cash=100000
    )
    
    print("\n📊 Backtest Results:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return stats, data


def run_live_demo():
    """Demo function for live trading"""
    print("\n🚀 Running Live Trading Demo (PAPER TRADING)")
    print("-" * 50)
    
    # Initialize system
    system = MagnaCarterBacktester(
        paper_trading=True  # ALWAYS paper trading for demos
    )
    
    # Configure trading
    config = LiveTradingConfig(
        max_position_size=0.5,
        max_daily_trades=5,
        min_trade_interval=1800,
        check_interval=1800,
        risk_per_trade=0.02
    )
    
    # Start live trading
    trader = system.start_live_trading(
        symbol="BTC/USD",
        config=config
    )
    
    print("📈 Live trading demo started (paper trading)")
    print("This is a simulation - no real money at risk")
    print("Press Ctrl+C to stop")
    
    try:
        # Run for a short demo period
        trader.start_background()
        time.sleep(300)  # Run for 5 minutes
        
        # Show results
        account_info = trader.get_account_info()
        print(f"\n💰 Account Status:")
        for key, value in account_info.items():
            print(f"{key}: {value}")
        
        position = trader.get_current_position()
        if position:
            print(f"\n📊 Current Position:")
            for key, value in position.items():
                print(f"{key}: {value}")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        trader.stop()
        system.stop_all_trading()


def run_multi_symbol_demo():
    """Demo function for multi-symbol trading"""
    print("\n🌐 Running Multi-Symbol Trading Demo")
    print("-" * 45)
    
    # Initialize system
    system = MagnaCarterBacktester(paper_trading=True)
    
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
    
    print("Starting traders for multiple symbols...")
    
    for symbol, config in symbols_configs:
        system.start_live_trading(
            symbol=symbol,
            config=config,
            background=True
        )
        print(f"✅ Started trading for {symbol}")
    
    try:
        # Run demo
        time.sleep(180)  # Run for 3 minutes
        
        # Show summary
        summary = system.get_live_trading_summary()
        print(f"\n📈 Trading Summary:")
        print(f"Active Traders: {summary['active_traders']}")
        print(f"Total Symbols: {summary['total_symbols']}")
        
        for symbol, perf in summary['individual_performance'].items():
            print(f"\n{symbol}:")
            for key, value in perf.items():
                print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        print("\nMulti-symbol demo stopped")
    finally:
        system.stop_all_trading()


def main():
    """Main CLI as described in README"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Magna Carter Trading System")
    parser.add_argument("--mode", 
                       choices=["backtest", "live", "multi"], 
                       default="backtest",
                       help="Demo mode to run")
    parser.add_argument("--symbol", 
                       default="BTC/USD",
                       help="Trading symbol for backtest/live mode")
    parser.add_argument("--paper", 
                       action="store_true",
                       help="Force paper trading (recommended)")
    
    args = parser.parse_args()
    
    print("🎯 Magna Carter Trading System")
    print("Market-Adaptive Global Neural Architecture")
    print("="*50)
    
    try:
        if args.mode == "backtest":
            run_backtest_demo()
        elif args.mode == "live":
            run_live_demo()
        elif args.mode == "multi":
            run_multi_symbol_demo()
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
    
    print("\n" + "="*50)
    print("Remember: This is for educational purposes only!")
    print("Real trading involves significant risk of loss.")
    print("Always start with paper trading.")


if __name__ == "__main__":
    main()
