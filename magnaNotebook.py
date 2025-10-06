
"""
Magna AI Trading System - Notebook Friendly Version
Easy-to-use functions for Jupyter notebooks and interactive development
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Import configurations
from dataclasses import dataclass, field


@dataclass
class NotebookTradingConfig:
    """Simplified configuration for notebook usage"""
    initial_capital: float = 100.0
    max_position_size: float = 0.8
    risk_per_trade: float = 0.10
    take_profit_pct: float = 0.08
    stop_loss_pct: float = 0.04
    target_symbols: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD"])
    simulation_mode: bool = True


class NotebookTradingSystem:
    """Simplified trading system optimized for notebook usage"""
    
    def __init__(self, config: NotebookTradingConfig = None):
        self.config = config or NotebookTradingConfig()
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        
        print(f"Notebook Trading System Initialized")
        print(f"Initial Capital: ${self.config.initial_capital:.2f}")
        print(f"Target Symbols: {self.config.target_symbols}")
        print(f"Simulation Mode: {self.config.simulation_mode}")
    
    def generate_sample_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate realistic sample market data"""
        # Set parameters based on symbol
        if "BTC" in symbol:
            base_price, volatility = 45000, 0.03
        elif "ETH" in symbol:
            base_price, volatility = 2500, 0.04
        elif "SOL" in symbol:
            base_price, volatility = 100, 0.06
        else:
            base_price, volatility = 50, 0.05
        
        # Generate time series
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start_date, end_date, freq='H')
        
        # Create price movement with trend and noise
        np.random.seed(hash(symbol) % 2**32)
        n_periods = len(dates)
        
        trend = np.linspace(base_price * 0.8, base_price * 1.3, n_periods)
        noise = np.random.normal(0, volatility, n_periods).cumsum()
        prices = trend * (1 + noise * 0.1)
        prices = np.maximum(prices, base_price * 0.1)  # Ensure positive prices
        
        # Create OHLCV DataFrame
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(prices[0])
        
        bar_volatility = np.random.uniform(0.005, 0.03, n_periods)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + bar_volatility)
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - bar_volatility)
        df['Volume'] = np.random.lognormal(mean=8, sigma=1, size=n_periods)
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if len(df) < 20:
            return df
        
        # Moving averages
        df['SMA_9'] = df['Close'].rolling(window=9).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        return df
    
    def generate_ai_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate AI-like trading signal (simplified)"""
        if len(df) < 50:
            return 0, 0.5
        
        # Multiple signal components
        signals = []
        weights = []
        
        # Trend following signal
        sma_9 = df['SMA_9'].iloc[-1]
        sma_21 = df['SMA_21'].iloc[-1]
        if pd.notna(sma_9) and pd.notna(sma_21):
            trend_signal = 1 if sma_9 > sma_21 else -1
            trend_strength = abs(sma_9 - sma_21) / sma_21
            signals.append(trend_signal)
            weights.append(min(trend_strength * 10, 1.0))
        
        # Momentum signal (RSI)
        rsi = df['RSI'].iloc[-1]
        if pd.notna(rsi):
            if rsi < 30:
                momentum_signal = 1  # Oversold
                momentum_confidence = (30 - rsi) / 30
            elif rsi > 70:
                momentum_signal = -1  # Overbought
                momentum_confidence = (rsi - 70) / 30
            else:
                momentum_signal = 0
                momentum_confidence = 0.3
            
            if momentum_signal != 0:
                signals.append(momentum_signal)
                weights.append(momentum_confidence)
        
        # MACD signal
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]
        if pd.notna(macd) and pd.notna(macd_signal):
            macd_cross = 1 if macd > macd_signal else -1
            macd_strength = abs(macd - macd_signal) / df['Close'].iloc[-1]
            signals.append(macd_cross)
            weights.append(min(macd_strength * 1000, 1.0))
        
        # Volatility breakout signal
        recent_prices = df['Close'].tail(10)
        if len(recent_prices) >= 10:
            volatility = recent_prices.std() / recent_prices.mean()
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            if abs(price_change) > volatility * 0.5:  # Breakout detected
                breakout_signal = 1 if price_change > 0 else -1
                signals.append(breakout_signal)
                weights.append(0.7)
        
        # Combine signals
        if not signals:
            return 0, 0.5
        
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
        confidence = min(sum(weights) / len(signals), 0.95)
        
        # Convert to discrete signal
        if weighted_signal > 0.3:
            return 1, confidence
        elif weighted_signal < -0.3:
            return -1, confidence
        else:
            return 0, confidence
    
    def calculate_position_size(self, signal_confidence: float, current_price: float) -> float:
        """Calculate optimal position size"""
        # Base position size
        base_size = self.portfolio_value * self.config.max_position_size
        
        # Adjust by confidence
        adjusted_size = base_size * signal_confidence
        
        # Risk-based adjustment
        risk_amount = self.portfolio_value * self.config.risk_per_trade
        position_size = min(adjusted_size, risk_amount)
        
        # Convert to quantity (fractional for crypto)
        quantity = position_size / current_price
        return round(quantity, 6)
    
    def simulate_trade(self, symbol: str, signal: int, quantity: float, price: float) -> Dict:
        """Simulate a trade execution"""
        trade_value = quantity * price
        
        if signal == 1:  # Buy
            if trade_value <= self.portfolio_value:
                self.portfolio_value -= trade_value
                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                
                pos = self.positions[symbol]
                total_cost = pos['quantity'] * pos['avg_price'] + trade_value
                pos['quantity'] += quantity
                pos['avg_price'] = total_cost / pos['quantity']
                
                trade_result = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'success': True
                }
            else:
                trade_result = {'success': False, 'reason': 'Insufficient funds'}
        
        elif signal == -1:  # Sell
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                self.portfolio_value += trade_value
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
                
                trade_result = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'success': True
                }
            else:
                trade_result = {'success': False, 'reason': 'Insufficient position'}
        
        else:
            trade_result = {'success': False, 'reason': 'No signal'}
        
        if trade_result.get('success'):
            self.trade_history.append(trade_result)
        
        return trade_result
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current positions"""
        cash = self.portfolio_value
        position_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position['quantity'] * current_prices[symbol]
        
        total_value = cash + position_value
        self.performance_history.append({
            'timestamp': datetime.now(),
            'cash': cash,
            'positions': position_value,
            'total': total_value
        })
        
        return total_value


# Notebook-friendly functions
def quick_test_system(capital: float = 100.0, days: int = 7) -> Dict:
    """Quick test of the trading system"""
    print(f"Running quick test with ${capital:.2f} over {days} days")
    
    config = NotebookTradingConfig(initial_capital=capital)
    system = NotebookTradingSystem(config)
    
    results = {}
    
    for symbol in config.target_symbols:
        print(f"\nTesting {symbol}...")
        
        # Generate data and signals
        data = system.generate_sample_data(symbol, days)
        data = system.calculate_technical_indicators(data)
        
        # Simulate trading
        trades = []
        for i in range(len(data)):
            if i < 50:  # Need enough data for indicators
                continue
                
            current_data = data.iloc[:i+1]
            signal, confidence = system.generate_ai_signal(current_data)
            
            if signal != 0 and confidence > 0.6:
                current_price = current_data['Close'].iloc[-1]
                quantity = system.calculate_position_size(confidence, current_price)
                
                if quantity > 0:
                    trade_result = system.simulate_trade(symbol, signal, quantity, current_price)
                    if trade_result.get('success'):
                        trades.append(trade_result)
        
        results[symbol] = {
            'trades': len(trades),
            'data_points': len(data),
            'final_price': data['Close'].iloc[-1]
        }
    
    # Calculate performance
    current_prices = {symbol: results[symbol]['final_price']
                     for symbol in results.keys()}
    final_value = system.update_portfolio_value(current_prices)
    
    total_trades = sum(results[symbol]['trades'] for symbol in results.keys())
    profit_loss = final_value - capital
    profit_pct = (profit_loss / capital) * 100
    
    print(f"\n{'='*50}")
    print(f"QUICK TEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ${capital:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Profit/Loss: ${profit_loss:.2f} ({profit_pct:+.1f}%)")
    print(f"Total Trades: {total_trades}")
    print(f"Trade History: {len(system.trade_history)} executed trades")
    
    return {
        'initial_capital': capital,
        'final_value': final_value,
        'profit_loss': profit_loss,
        'profit_pct': profit_pct,
        'total_trades': total_trades,
        'system': system,
        'results_by_symbol': results
    }


def analyze_symbol_performance(symbol: str = "BTC/USD", days: int = 30) -> Dict:
    """Analyze performance for a specific symbol"""
    print(f"Analyzing {symbol} performance over {days} days")
    
    system = NotebookTradingSystem()
    data = system.generate_sample_data(symbol, days)
    data = system.calculate_technical_indicators(data)
    
    # Generate signals for each point
    signals = []
    confidences = []
    
    for i in range(50, len(data)):  # Start after enough data for indicators
        current_data = data.iloc[:i+1]
        signal, confidence = system.generate_ai_signal(current_data)
        signals.append(signal)
        confidences.append(confidence)
    
    # Add to dataframe
    signal_data = data.iloc[50:].copy()
    signal_data['Signal'] = signals
    signal_data['Confidence'] = confidences
    
    # Analysis
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    hold_signals = sum(1 for s in signals if s == 0)
    avg_confidence = np.mean(confidences)
    
    print(f"\nSignal Analysis:")
    print(f"Buy Signals: {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    print(f"Hold Signals: {hold_signals}")
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    return {
        'symbol': symbol,
        'data': signal_data,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'avg_confidence': avg_confidence,
        'price_change': (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
    }


def plot_trading_signals(symbol: str = "BTC/USD", days: int = 14):
    """Plot price data with trading signals"""
    analysis = analyze_symbol_performance(symbol, days)
    data = analysis['data']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{symbol} Trading Analysis', fontsize=16)
    
    # Price and signals
    ax1.plot(data.index, data['Close'], label='Price', linewidth=2)
    ax1.plot(data.index, data['SMA_9'], label='SMA 9', alpha=0.7)
    ax1.plot(data.index, data['SMA_21'], label='SMA 21', alpha=0.7)
    
    # Mark buy/sell signals
    buy_points = data[data['Signal'] == 1]
    sell_points = data[data['Signal'] == -1]
    
    if not buy_points.empty:
        ax1.scatter(buy_points.index, buy_points['Close'],
                   color='green', marker='^', s=100, label='Buy Signal')
    if not sell_points.empty:
        ax1.scatter(sell_points.index, sell_points['Close'],
                   color='red', marker='v', s=100, label='Sell Signal')
    
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Signal confidence
    ax3.plot(data.index, data['Confidence'], label='Signal Confidence', color='orange')
    ax3.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Confidence Threshold')
    ax3.set_ylabel('Confidence')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def run_portfolio_simulation(
    capital: float = 100.0, 
    days: int = 30, 
    symbols: List[str] = None
) -> Dict:
    """Run a comprehensive portfolio simulation"""
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]
    
    print(f"Running portfolio simulation with ${capital:.2f} over {days} days")
    print(f"Symbols: {symbols}")
    
    config = NotebookTradingConfig(
        initial_capital=capital,
        target_symbols=symbols,
        max_position_size=0.8,
        risk_per_trade=0.15
    )
    
    system = NotebookTradingSystem(config)
    
    # Generate data for all symbols
    all_data = {}
    for symbol in symbols:
        data = system.generate_sample_data(symbol, days)
        all_data[symbol] = system.calculate_technical_indicators(data)
    
    # Simulate day by day
    daily_values = [capital]
    daily_trades = []
    
    # Get common time index
    min_length = min(len(data) for data in all_data.values())
    
    for day in range(50, min_length):  # Start after indicators are calculated
        current_prices = {}
        trades_today = 0
        
        # Check each symbol for signals
        for symbol in symbols:
            current_data = all_data[symbol].iloc[:day+1]
            signal, confidence = system.generate_ai_signal(current_data)
            current_price = current_data['Close'].iloc[-1]
            current_prices[symbol] = current_price
            
            # Execute trades based on signals
            if signal != 0 and confidence > 0.6 and trades_today < 3:  # Limit trades per day
                quantity = system.calculate_position_size(confidence, current_price)
                
                if quantity > 0:
                    trade_result = system.simulate_trade(symbol, signal, quantity, current_price)
                    if trade_result.get('success'):
                        trades_today += 1
        
        # Update portfolio value
        total_value = system.update_portfolio_value(current_prices)
        daily_values.append(total_value)
        daily_trades.append(trades_today)
    
    # Calculate final results
    final_value = daily_values[-1]
    total_return = final_value - capital
    return_pct = (total_return / capital) * 100
    total_trades = len(system.trade_history)
    max_value = max(daily_values)
    max_drawdown = (max_value - min(daily_values)) / max_value * 100
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PORTFOLIO SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital: ${capital:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: ${total_return:.2f} ({return_pct:+.1f}%)")
    print(f"Max Value: ${max_value:.2f}")
    print(f"Max Drawdown: {max_drawdown:.1f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades/Day: {np.mean(daily_trades):.1f}")
    
    return {
        'initial_capital': capital,
        'final_value': final_value,
        'total_return': total_return,
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'daily_values': daily_values,
        'daily_trades': daily_trades,
        'system': system,
        'trade_history': system.trade_history
    }


def plot_portfolio_performance(simulation_results: Dict):
    """Plot portfolio performance over time"""
    daily_values = simulation_results['daily_values']
    daily_trades = simulation_results['daily_trades']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value over time
    ax1.plot(range(len(daily_values)), daily_values, linewidth=2, color='blue')
    ax1.axhline(y=simulation_results['initial_capital'],
                color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Daily trades
    ax2.bar(range(len(daily_trades)), daily_trades, alpha=0.7, color='orange')
    ax2.set_title('Daily Trading Activity')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Number of Trades')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compare_strategies(
    capital: float = 100.0, 
    days: int = 30
) -> Dict:
    """Compare different trading strategies"""
    print(f"Comparing trading strategies with ${capital:.2f} over {days} days")
    
    strategies = {
        'Conservative': NotebookTradingConfig(
            initial_capital=capital,
            max_position_size=0.5,
            risk_per_trade=0.05,
            take_profit_pct=0.06,
            stop_loss_pct=0.03
        ),
        'Balanced': NotebookTradingConfig(
            initial_capital=capital,
            max_position_size=0.7,
            risk_per_trade=0.10,
            take_profit_pct=0.08,
            stop_loss_pct=0.04
        ),
        'Aggressive': NotebookTradingConfig(
            initial_capital=capital,
            max_position_size=0.9,
            risk_per_trade=0.20,
            take_profit_pct=0.12,
            stop_loss_pct=0.06
        )
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"\nTesting {strategy_name} strategy...")
        
        system = NotebookTradingSystem(config)
        
        # Run simulation for BTC/USD
        data = system.generate_sample_data("BTC/USD", days)
        data = system.calculate_technical_indicators(data)
        
        trades = 0
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            signal, confidence = system.generate_ai_signal(current_data)
            
            if signal != 0 and confidence > 0.6:
                current_price = current_data['Close'].iloc[-1]
                quantity = system.calculate_position_size(confidence, current_price)
                
                if quantity > 0:
                    trade_result = system.simulate_trade("BTC/USD", signal, quantity, current_price)
                    if trade_result.get('success'):
                        trades += 1
        
        final_value = system.update_portfolio_value({"BTC/USD": data['Close'].iloc[-1]})
        
        results[strategy_name] = {
            'final_value': final_value,
            'return_pct': (final_value - capital) / capital * 100,
            'trades': trades,
            'system': system
        }
        
        print(f"{strategy_name}: ${final_value:.2f} ({results[strategy_name]['return_pct']:+.1f}%) - {trades} trades")
    
    return results


def stress_test_system(capital: float = 100.0, crash_scenario: bool = True) -> Dict:
    """Stress test the system under adverse conditions"""
    print(f"Stress testing system with ${capital:.2f}")
    print(f"Crash scenario: {crash_scenario}")
    
    system = NotebookTradingSystem(NotebookTradingConfig(initial_capital=capital))
    
    # Generate stressful market data
    dates = pd.date_range(datetime.now() - timedelta(days=7), datetime.now(), freq='H')
    
    if crash_scenario:
        # Simulate a market crash
        initial_price = 45000
        crash_factor = 0.6  # 40% crash
        recovery_factor = 0.8  # Partial recovery
        
        # Create crash pattern
        n_periods = len(dates)
        crash_point = n_periods // 3
        recovery_point = crash_point + n_periods // 3
        
        prices = []
        for i in range(n_periods):
            if i < crash_point:
                # Normal volatility
                price = initial_price * (1 + np.random.normal(0, 0.01))
            elif i < recovery_point:
                # Crash phase
                crash_progress = (i - crash_point) / (recovery_point - crash_point)
                price = initial_price * (1 - crash_progress * (1 - crash_factor)) * (1 + np.random.normal(0, 0.03))
            else:
                # Recovery phase
                recovery_progress = (i - recovery_point) / (n_periods - recovery_point)
                target_price = initial_price * crash_factor * recovery_factor
                price = target_price * (1 + np.random.normal(0, 0.02))
            
            prices.append(max(price, initial_price * 0.1))  # Floor price
    
    # Create stress test data
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1).fillna(prices[0])
    df['High'] = df[['Open', 'Close']].max(axis=1) * 1.02
    df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.98
    df['Volume'] = np.random.lognormal(mean=10, sigma=1, size=len(dates))
    
    df = system.calculate_technical_indicators(df)
    
    # Simulate trading during stress
    trades = []
    portfolio_values = [capital]
    
    for i in range(50, len(df)):
        current_data = df.iloc[:i+1]
        signal, confidence = system.generate_ai_signal(current_data)
        current_price = current_data['Close'].iloc[-1]
        
        if signal != 0 and confidence > 0.5:  # Lower threshold during stress
            quantity = system.calculate_position_size(confidence, current_price)
            
            if quantity > 0:
                trade_result = system.simulate_trade("BTC/USD", signal, quantity, current_price)
                if trade_result.get('success'):
                    trades.append(trade_result)
        
        # Update portfolio value
        portfolio_value = system.update_portfolio_value({"BTC/USD": current_price})
        portfolio_values.append(portfolio_value)
    
    final_value = portfolio_values[-1]
    min_value = min(portfolio_values)
    max_drawdown = (capital - min_value) / capital * 100
    
    print(f"\n{'='*50}")
    print(f"STRESS TEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ${capital:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Minimum Value: ${min_value:.2f}")
    print(f"Max Drawdown: {max_drawdown:.1f}%")
    print(f"Total Trades: {len(trades)}")
    print(f"Return: {((final_value - capital) / capital * 100):+.1f}%")
    
    return {
        'initial_capital': capital,
        'final_value': final_value,
        'min_value': min_value,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'portfolio_values': portfolio_values,
        'price_data': df,
        'trades': trades
    }


# Easy-to-use notebook functions
def demo_basic_trading():
    """Basic trading demo - perfect for notebooks"""
    print("Running Basic Trading Demo")
    print("=" * 40)
    
    # Quick test
    result = quick_test_system(capital=100.0, days=7)
    
    # Show some trades
    if result['system'].trade_history:
        print(f"\nFirst 3 trades:")
        for i, trade in enumerate(result['system'].trade_history[:3]):
            print(f"{i+1}. {trade['action']} {trade['quantity']:.4f} {trade['symbol']} at ${trade['price']:.2f}")
    
    return result


def demo_ai_signals():
    """Demonstrate AI signal generation"""
    print("AI Signal Generation Demo")
    print("=" * 40)
    
    analysis = analyze_symbol_performance("BTC/USD", days=14)
    plot_trading_signals("BTC/USD", days=14)
    
    return analysis


def demo_portfolio_management():
    """Demonstrate portfolio management"""
    print("Portfolio Management Demo")
    print("=" * 40)
    
    simulation = run_portfolio_simulation(
        capital=100.0,
        days=21,
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"]
    )
    
    plot_portfolio_performance(simulation)
    
    return simulation


def demo_strategy_comparison():
    """Compare different trading strategies"""
    print("Strategy Comparison Demo")
    print("=" * 40)
    
    comparison = compare_strategies(capital=100.0, days=21)
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    strategies = list(comparison.keys())
    returns = [comparison[s]['return_pct'] for s in strategies]
    trades = [comparison[s]['trades'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, returns, width, label='Return %', alpha=0.8)
    ax.bar(x + width/2, [t/5 for t in trades], width, label='Trades (÷5)', alpha=0.8)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Performance')
    ax.set_title('Strategy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison


def demo_stress_testing():
    """Demonstrate stress testing"""
    print("Stress Testing Demo")
    print("=" * 40)
    
    stress_result = stress_test_system(capital=100.0, crash_scenario=True)
    
    # Plot stress test results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value during stress
    ax1.plot(stress_result['portfolio_values'], linewidth=2, color='red')
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Value During Market Stress')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Price movement
    price_data = stress_result['price_data']
    ax2.plot(price_data['Close'], linewidth=2, color='blue')
    ax2.set_title('Market Price During Stress Test')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return stress_result


# Utility functions for notebook usage
def create_trading_dashboard(symbols: List[str] = None):
    """Create a comprehensive trading dashboard"""
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    print("Creating Trading Dashboard")
    print("=" * 50)
    
    fig, axes = plt.subplots(len(symbols), 2, figsize=(15, 5*len(symbols)))
    if len(symbols) == 1:
        axes = axes.reshape(1, -1)
    
    results = {}
    
    for i, symbol in enumerate(symbols):
        print(f"\nAnalyzing {symbol}...")
        
        # Generate data
        system = NotebookTradingSystem()
        data = system.generate_sample_data(symbol, days=21)
        data = system.calculate_technical_indicators(data)
        
        # Generate signals
        signals = []
        for j in range(50, len(data)):
            current_data = data.iloc[:j+1]
            signal, confidence = system.generate_ai_signal(current_data)
            signals.append((signal, confidence))
        
        signal_data = data.iloc[50:].copy()
        signal_data['Signal'] = [s[0] for s in signals]
        signal_data['Confidence'] = [s[1] for s in signals]
        
        # Plot price and signals
        ax1 = axes[i, 0]
        ax1.plot(signal_data.index, signal_data['Close'], label='Price', linewidth=2)
        ax1.plot(signal_data.index, signal_data['SMA_21'], label='SMA 21', alpha=0.7)
        
        buy_points = signal_data[signal_data['Signal'] == 1]
        sell_points = signal_data[signal_data['Signal'] == -1]
        
        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['Close'],
                       color='green', marker='^', s=60, alpha=0.7)
        if not sell_points.empty:
            ax1.scatter(sell_points.index, sell_points['Close'],
                       color='red', marker='v', s=60, alpha=0.7)
        
        ax1.set_title(f'{symbol} Price & Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot RSI and confidence
        ax2 = axes[i, 1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(signal_data.index, signal_data['RSI'], color='purple', label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI', color='purple')
        ax2.set_ylim(0, 100)
        
        ax2_twin.plot(signal_data.index, signal_data['Confidence'],
                     color='orange', alpha=0.7, label='Confidence')
        ax2_twin.set_ylabel('Confidence', color='orange')
        ax2_twin.set_ylim(0, 1)
        
        ax2.set_title(f'{symbol} RSI & Signal Confidence')
        ax2.grid(True, alpha=0.3)
        
        # Store results
        results[symbol] = {
            'buy_signals': len(buy_points),
            'sell_signals': len(sell_points),
            'avg_confidence': signal_data['Confidence'].mean(),
            'price_change': (signal_data['Close'].iloc[-1] - signal_data['Close'].iloc[0]) / signal_data['Close'].iloc[0] * 100
        }
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print("DASHBOARD SUMMARY")
    print(f"{'='*50}")
    for symbol, result in results.items():
        print(f"{symbol}:")
        print(f"  Buy Signals: {result['buy_signals']}")
        print(f"  Sell Signals: {result['sell_signals']}")
        print(f"  Avg Confidence: {result['avg_confidence']:.2f}")
        print(f"  Price Change: {result['price_change']:+.1f}%")
    
    return results


def run_monte_carlo_simulation(
    capital: float = 100.0, 
    num_simulations: int = 100, 
    days: int = 30
) -> Dict:
    """Run Monte Carlo simulation for risk assessment"""
    print(f"Running Monte Carlo Simulation")
    print(f"Simulations: {num_simulations}, Days: {days}, Capital: ${capital:.2f}")
    
    results = []
    
    for sim in range(num_simulations):
        if sim % 20 == 0:
            print(f"Progress: {sim}/{num_simulations}")
        
        # Run simulation with random seed
        np.random.seed(sim)
        
        config = NotebookTradingConfig(
            initial_capital=capital,
            max_position_size=np.random.uniform(0.5, 0.9),
            risk_per_trade=np.random.uniform(0.05, 0.20)
        )
        
        system = NotebookTradingSystem(config)
        
        # Generate random market data
        symbol = "BTC/USD"
        data = system.generate_sample_data(symbol, days)
        data = system.calculate_technical_indicators(data)
        
        # Simulate trading
        trades = 0
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            signal, confidence = system.generate_ai_signal(current_data)
            
            if signal != 0 and confidence > 0.6:
                current_price = current_data['Close'].iloc[-1]
                quantity = system.calculate_position_size(confidence, current_price)
                
                if quantity > 0:
                    trade_result = system.simulate_trade(symbol, signal, quantity, current_price)
                    if trade_result.get('success'):
                        trades += 1
        
        final_value = system.update_portfolio_value({symbol: data['Close'].iloc[-1]})
        return_pct = (final_value - capital) / capital * 100
        
        results.append({
            'final_value': final_value,
            'return_pct': return_pct,
            'trades': trades
        })
    
    # Analyze results
    returns = [r['return_pct'] for r in results]
    final_values = [r['final_value'] for r in results]
    
    analysis = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'median_return': np.median(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'prob_profit': sum(1 for r in returns if r > 0) / len(returns),
        'prob_loss_over_10': sum(1 for r in returns if r < -10) / len(returns),
        'percentile_5': np.percentile(returns, 5),
        'percentile_95': np.percentile(returns, 95)
    }
    
    print(f"\n{'='*50}")
    print("MONTE CARLO RESULTS")
    print(f"{'='*50}")
    print(f"Mean Return: {analysis['mean_return']:.1f}%")
    print(f"Std Deviation: {analysis['std_return']:.1f}%")
    print(f"Median Return: {analysis['median_return']:.1f}%")
    print(f"Min Return: {analysis['min_return']:.1f}%")
    print(f"Max Return: {analysis['max_return']:.1f}%")
    print(f"Probability of Profit: {analysis['prob_profit']*100:.1f}%")
    print(f"Probability of >10% Loss: {analysis['prob_loss_over_10']*100:.1f}%")
    print(f"5th Percentile: {analysis['percentile_5']:.1f}%")
    print(f"95th Percentile: {analysis['percentile_95']:.1f}%")
    
    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of returns
    ax1.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='Break-even')
    ax1.axvline(x=analysis['mean_return'], color='green', linestyle='-', label='Mean')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(returns)
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Return Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'analysis': analysis,
        'raw_results': results,
        'returns': returns
    }


# Notebook test functions - easy to run individually
def test_quick():
    """Quick 1-minute test"""
    return quick_test_system(100, 7)

def test_signals():
    """Test signal generation"""
    return demo_ai_signals()

def test_portfolio():
    """Test portfolio simulation"""
    return demo_portfolio_management()

def test_strategies():
    """Test strategy comparison"""
    return demo_strategy_comparison()

def test_stress():
    """Test stress scenarios"""
    return demo_stress_testing()

def test_dashboard():
    """Test trading dashboard"""
    return create_trading_dashboard()

def test_monte_carlo():
    """Test Monte Carlo simulation"""
    return run_monte_carlo_simulation(num_simulations=50)

def run_all_tests():
    """Run all tests in sequence"""
    print("Running All Tests - This may take a few minutes")
    print("=" * 60)
    
    tests = [
        ("Quick Test", test_quick),
        ("Signal Generation", test_signals),
        ("Portfolio Management", test_portfolio),
        ("Strategy Comparison", test_strategies),
        ("Stress Testing", test_stress),
        ("Trading Dashboard", test_dashboard),
        ("Monte Carlo", test_monte_carlo)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        try:
            results[test_name] = test_func()
            print(f"✓ {test_name} completed successfully")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = None
    
    return results


# Risk warning function
def show_risk_warnings():
    """Display important risk warnings"""
    warning_text = """
    ⚠️  IMPORTANT RISK WARNINGS ⚠️
    ================================
    
    1. FINANCIAL RISK:
       • Trading cryptocurrencies is extremely risky
       • You can lose your entire investment
       • Past performance does not predict future results
       • AI predictions are not guarantees
    
    2. PAPER TRADING FIRST:
       • Always test strategies with paper trading
       • Never use real money until thoroughly tested
       • Understand the system completely before trading
    
    3. RESPONSIBLE TRADING:
       • Only trade with money you can afford to lose
       • Start with very small amounts
       • Never trade borrowed money
       • Set strict loss limits
    
    4. SYSTEM LIMITATIONS:
       • This is educational/experimental code
       • Not professional trading software
       • May contain bugs or errors
       • Requires continuous monitoring
    
    5. LEGAL CONSIDERATIONS:
       • Check local trading regulations
       • Understand tax implications
       • Ensure compliance with financial laws
    
    Remember: The goal of maximizing $100 is aggressive and
    extremely high-risk. Most attempts will likely result in losses.
    """
    
    print(warning_text)


# Example usage for notebooks:
"""
# Import the functions
from magna_notebook import *

# Show warnings first
show_risk_warnings()

# Quick test
result = test_quick()

# Test individual components
signals = test_signals()
portfolio = test_portfolio() 
strategies = test_strategies()

# More comprehensive tests
dashboard = test_dashboard()
monte_carlo = test_monte_carlo()

# Run everything
all_results = run_all_tests()
"""
