import json
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd

class DatabaseService:
    """
    Database Service for handling data storage and retrieval.
    
    Responsibilities:
    1. Storing trade data
    2. Retrieving trade history
    3. Storing and retrieving performance metrics
    4. Managing data persistence
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Database Service.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.metrics_file = os.path.join(data_dir, "metrics.json")
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data structures
        self.trades = self._load_trades()
        self.metrics = self._load_metrics()
        
    def save_trade_data(self, trade_data: Dict) -> bool:
        """
        Save trade data to storage.
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Add trade ID if not present
            if 'id' not in trade_data:
                trade_data['id'] = len(self.trades) + 1
                
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
                
            # Add trade to memory
            self.trades.append(trade_data)
            
            # Save to file
            with open(self.trades_file, "w") as f:
                json.dump(self.trades, f, indent=4)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving trade data: {str(e)}")
            return False
            
    def get_trades(self, filters: Dict = None) -> List[Dict]:
        """
        Retrieve trade data with optional filtering.
        
        Args:
            filters: Filter criteria
            
        Returns:
            List of trade data dictionaries
        """
        if not filters:
            return self.trades
            
        filtered_trades = self.trades
        
        # Apply filters
        if 'symbol' in filters:
            filtered_trades = [trade for trade in filtered_trades if trade.get('symbol') == filters['symbol']]
            
        if 'start_time' in filters:
            filtered_trades = [trade for trade in filtered_trades 
                                if trade.get('entry_time', '') >= filters['start_time']]
            
        if 'end_time' in filters:
            filtered_trades = [trade for trade in filtered_trades 
                                if trade.get('entry_time', '') <= filters['end_time']]
            
        if 'status' in filters:
            filtered_trades = [trade for trade in filtered_trades 
                                if trade.get('status') == filters['status']]
            
        if 'strategy' in filters:
            filtered_trades = [trade for trade in filtered_trades 
                                if trade.get('strategy') == filters['strategy']]
            
        return filtered_trades
        
    def update_trade(self, trade_id: int, update_data: Dict) -> bool:
        """
        Update an existing trade record.
        
        Args:
            trade_id: ID of the trade to update
            update_data: Data to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Find the trade by ID
            for i, trade in enumerate(self.trades):
                if trade.get('id') == trade_id:
                    # Update the trade data
                    self.trades[i].update(update_data)
                    
                    # Save to file
                    with open(self.trades_file, "w") as f:
                        json.dump(self.trades, f, indent=4)
                        
                    return True
                    
            self.logger.warning(f"Trade ID not found: {trade_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            return False
            
    def save_performance_metrics(self, metrics: Dict) -> bool:
        """
        Save performance metrics to storage.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Update metrics in memory
            self.metrics.update(metrics)
            
            # Add timestamp if not present
            if 'timestamp' not in self.metrics:
                self.metrics['timestamp'] = datetime.now().isoformat()
            else:
                self.metrics['timestamp'] = datetime.now().isoformat()
                
            # Save to file
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=4)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
            return False
            
    def get_performance_metrics(self) -> Dict:
        """
        Retrieve performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return self.metrics
        
    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """
        Get portfolio value history.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of portfolio value entries
        """
        try:
            # If portfolio history exists in metrics, return it
            if 'portfolio_history' in self.metrics:
                history = self.metrics['portfolio_history']
                # Limit to the requested number of days
                if len(history) > days:
                    return history[-days:]
                return history
                
            # Otherwise, generate it from trade data
            return self._generate_portfolio_history(days)
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            return []
            
    def clear_data(self) -> bool:
        """
        Clear all stored data.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self.trades = []
            self.metrics = {}
            
            # Remove data files
            if os.path.exists(self.trades_file):
                os.remove(self.trades_file)
                
            if os.path.exists(self.metrics_file):
                os.remove(self.metrics_file)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            return False
            
    def _load_trades(self) -> List[Dict]:
        """
        Load trades from file.
        
        Returns:
            List of trade dictionaries
        """
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading trades file: {str(e)}")
                return []
        else:
            return []
            
    def _load_metrics(self) -> Dict:
        """
        Load metrics from file.
        
        Returns:
            Metrics dictionary
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metrics file: {str(e)}")
                return {}
        else:
            return {}
            
    def _generate_portfolio_history(self, days: int = 30) -> List[Dict]:
        """
        Generate portfolio history from trade data.
        
        Args:
            days: Number of days of history
            
        Returns:
            List of portfolio value entries
        """
        try:
            # Start with an initial balance
            initial_balance = 1000.0  # Default initial balance
            
            # Get relevant trades
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Convert trades to DataFrame for easier manipulation
            if not self.trades:
                # Generate mock data if no trades exist
                dates = pd.date_range(end=end_date, periods=days)
                values = [initial_balance]
                
                for i in range(1, days):
                    # Random daily change between -1% and +2%
                    daily_change = pd.np.random.uniform(-0.01, 0.02)
                    values.append(values[-1] * (1 + daily_change))
                
                return [{'timestamp': date.isoformat(), 'value': value} 
                        for date, value in zip(dates, values)]
            
            # Process actual trade data
            trade_df = pd.DataFrame(self.trades)
            
            # Convert timestamps to datetime
            trade_df['entry_time'] = pd.to_datetime(trade_df['entry_time'])
            trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time'])
            
            # Filter trades within the date range
            relevant_trades = trade_df[(trade_df['entry_time'] >= start_date) | 
                                     (trade_df['exit_time'] >= start_date)]
            
            # Generate daily portfolio values
            dates = pd.date_range(start=start_date, end=end_date)
            portfolio_values = []
            
            # Start with initial balance
            current_value = initial_balance
            
            for date in dates:
                # Add profits from trades closed on this date
                closed_trades = relevant_trades[relevant_trades['exit_time'].dt.date == date.date()]
                
                for _, trade in closed_trades.iterrows():
                    if 'pnl' in trade:
                        current_value += trade['pnl']
                
                portfolio_values.append(current_value)
            
            history = [{'timestamp': date.isoformat(), 'value': value} 
                      for date, value in zip(dates, portfolio_values)]
            
            # Save to metrics for future reference
            self.metrics['portfolio_history'] = history
            self.save_performance_metrics(self.metrics)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio history: {str(e)}")
            return []
