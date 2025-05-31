import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

class PerformanceCalculator:
    """
    Performance Calculator for computing trading performance metrics.
    
    Responsibilities:
    1. Calculating win rate, profit/loss, ROI
    2. Computing drawdown and risk metrics
    3. Generating performance reports
    4. Calculating daily, weekly, and monthly returns
    """
    
    def __init__(self, database_service=None):
        """
        Initialize the Performance Calculator.
        
        Args:
            database_service: Database service for retrieving trade data
        """
        self.database_service = database_service
        self.logger = logging.getLogger(__name__)
        
    def calculate_performance_metrics(self, trades: List[Dict] = None) -> Dict:
        """
        Calculate key performance metrics from trade data.
        
        Args:
            trades: List of trade dictionaries (if None, fetches from database)
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get trades if not provided
            if trades is None and self.database_service:
                trades = self.database_service.get_trades()
                
            if not trades:
                return self._get_empty_metrics()
                
            # Filter completed trades
            completed_trades = [trade for trade in trades if trade.get('status') == 'Closed']
            
            if not completed_trades:
                return self._get_empty_metrics()
                
            # Calculate win rate
            winning_trades = [trade for trade in completed_trades if trade.get('pnl', 0) > 0]
            win_rate = (len(winning_trades) / len(completed_trades)) * 100
            
            # Calculate total profit
            total_profit = sum(trade.get('pnl', 0) for trade in completed_trades)
            
            # Calculate average profit per trade
            avg_profit = total_profit / len(completed_trades)
            
            # Calculate profit factor
            gross_profit = sum(trade.get('pnl', 0) for trade in winning_trades)
            losing_trades = [trade for trade in completed_trades if trade.get('pnl', 0) <= 0]
            gross_loss = abs(sum(trade.get('pnl', 0) for trade in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate maximum drawdown
            drawdown = self.calculate_max_drawdown(trades)
            
            # Calculate ROI
            initial_balance = 1000  # Default initial balance
            roi = (total_profit / initial_balance) * 100
            
            # Calculate daily returns
            daily_returns = self.calculate_daily_returns(trades)
            
            # Average daily return
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = 0
            if daily_returns and len(daily_returns) > 1:
                daily_returns_std = np.std(daily_returns)
                if daily_returns_std > 0:
                    sharpe_ratio = (avg_daily_return / daily_returns_std) * np.sqrt(252)  # Annualized
            
            # Calculate average trade duration
            durations = []
            for trade in completed_trades:
                if 'entry_time' in trade and 'exit_time' in trade:
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
                    durations.append(duration)
            
            avg_trade_duration = np.mean(durations) if durations else 0
            
            metrics = {
                'win_rate': win_rate,
                'profit': total_profit,
                'avg_profit': avg_profit,
                'profit_factor': profit_factor,
                'drawdown': drawdown,
                'roi': roi,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade_duration': avg_trade_duration,
                'total_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'daily_returns': daily_returns,
                'avg_daily_return': avg_daily_return,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._get_empty_metrics()
        
    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """
        Calculate maximum drawdown from trade history.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Maximum drawdown percentage
        """
        try:
            if not trades:
                return 0.0
                
            # Sort trades by exit time
            completed_trades = [t for t in trades if t.get('status') == 'Closed' and 'exit_time' in t]
            sorted_trades = sorted(completed_trades, key=lambda x: x['exit_time'])
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            balance = 1000.0  # Initial balance
            
            for trade in sorted_trades:
                pnl = trade.get('pnl', 0)
                balance += pnl
                cumulative_pnl.append(balance)
            
            if not cumulative_pnl:
                return 0.0
                
            # Calculate drawdown
            max_drawdown = 0.0
            peak_balance = cumulative_pnl[0]
            
            for balance in cumulative_pnl:
                if balance > peak_balance:
                    peak_balance = balance
                else:
                    drawdown = (peak_balance - balance) / peak_balance * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return 0.0
        
    def calculate_daily_returns(self, trades: List[Dict]) -> List[float]:
        """
        Calculate daily returns from trade history.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            List of daily return percentages
        """
        try:
            if not trades:
                return []
                
            # Filter completed trades
            completed_trades = [t for t in trades if t.get('status') == 'Closed' and 'exit_time' in t]
            
            if not completed_trades:
                return []
                
            # Convert exit times to dates and group trades by date
            trade_df = pd.DataFrame(completed_trades)
            trade_df['exit_date'] = pd.to_datetime(trade_df['exit_time']).dt.date
            
            # Group by date and calculate daily P&L
            daily_pnl = trade_df.groupby('exit_date')['pnl'].sum()
            
            # Convert to daily returns based on initial balance + cumulative P&L
            initial_balance = 1000.0
            daily_returns = []
            
            # Get list of all dates in range
            date_range = pd.date_range(
                start=daily_pnl.index.min(),
                end=daily_pnl.index.max()
            )
            
            balance = initial_balance
            
            for date in date_range:
                date_key = date.date()
                if date_key in daily_pnl.index:
                    daily_return = daily_pnl[date_key] / balance * 100
                    balance += daily_pnl[date_key]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0.0)  # No trades for this day
            
            return daily_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating daily returns: {str(e)}")
            return []
        
    def generate_performance_report(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            start_date: Start date for the report period (ISO format)
            end_date: End date for the report period (ISO format)
            
        Returns:
            Dictionary containing report data
        """
        try:
            if self.database_service is None:
                return {}
                
            filters = {}
            if start_date:
                filters['start_time'] = start_date
            if end_date:
                filters['end_time'] = end_date
                
            trades = self.database_service.get_trades(filters)
            
            # Calculate overall performance metrics
            metrics = self.calculate_performance_metrics(trades)
            
            # Calculate performance by symbol
            symbols = set(trade.get('symbol', '') for trade in trades)
            symbol_performance = {}
            
            for symbol in symbols:
                if not symbol:
                    continue
                symbol_trades = [trade for trade in trades if trade.get('symbol') == symbol]
                symbol_metrics = self.calculate_performance_metrics(symbol_trades)
                symbol_performance[symbol] = symbol_metrics
            
            # Calculate performance by strategy
            strategies = set(trade.get('strategy', '') for trade in trades)
            strategy_performance = {}
            
            for strategy in strategies:
                if not strategy:
                    continue
                strategy_trades = [trade for trade in trades if trade.get('strategy') == strategy]
                strategy_metrics = self.calculate_performance_metrics(strategy_trades)
                strategy_performance[strategy] = strategy_metrics
            
            # Calculate monthly performance
            monthly_performance = self.calculate_monthly_performance(trades)
            
            report = {
                'overall_metrics': metrics,
                'symbol_performance': symbol_performance,
                'strategy_performance': strategy_performance,
                'monthly_performance': monthly_performance,
                'start_date': start_date,
                'end_date': end_date,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {}
        
    def calculate_monthly_performance(self, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics by month.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with monthly performance data
        """
        try:
            if not trades:
                return {}
                
            # Filter completed trades
            completed_trades = [t for t in trades if t.get('status') == 'Closed' and 'exit_time' in t]
            
            if not completed_trades:
                return {}
                
            # Convert to DataFrame for easier manipulation
            trade_df = pd.DataFrame(completed_trades)
            trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time'])
            trade_df['month'] = trade_df['exit_time'].dt.strftime('%Y-%m')
            
            # Group by month
            monthly_groups = trade_df.groupby('month')
            monthly_performance = {}
            
            for month, group in monthly_groups:
                month_trades = group.to_dict('records')
                monthly_metrics = self.calculate_performance_metrics(month_trades)
                monthly_performance[month] = {
                    'profit': monthly_metrics['profit'],
                    'win_rate': monthly_metrics['win_rate'],
                    'trades': len(month_trades),
                    'avg_profit': monthly_metrics['avg_profit']
                }
            
            return monthly_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly performance: {str(e)}")
            return {}
        
    def _get_empty_metrics(self) -> Dict:
        """
        Return an empty metrics dictionary with default values.
        
        Returns:
            Default metrics dictionary
        """
        return {
            'win_rate': 0.0,
            'profit': 0.0,
            'avg_profit': 0.0,
            'profit_factor': 0.0,
            'drawdown': 0.0,
            'roi': 0.0,
            'sharpe_ratio': 0.0,
            'avg_trade_duration': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'daily_returns': [],
            'avg_daily_return': 0.0,
            'timestamp': datetime.now().isoformat()
        }
