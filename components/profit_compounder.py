from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import time

class ProfitCompounder:
    """
    Profit Compounder for managing profit reinvestment.
    
    Responsibilities:
    1. Calculating compounding amounts based on trading performance
    2. Executing compounding strategy at the configured frequency
    3. Tracking compounding history and performance
    """
    
    def __init__(self, target_compound_rate: float = 100.0, compound_frequency: str = "Daily", order_manager=None):
        """
        Initialize the Profit Compounder.
        
        Args:
            target_compound_rate: Target compounding rate as percentage (e.g., 100 = reinvest all profits)
            compound_frequency: Frequency of compounding (Daily, Weekly, Monthly)
            order_manager: Order manager for executing compounding orders
        """
        self.target_compound_rate = target_compound_rate
        self.compound_frequency = compound_frequency
        self.order_manager = order_manager
        self.last_compound_time = None
        self.compound_history = []
        self.logger = logging.getLogger(__name__)
        self.initial_balance = 0.0
        self.current_balance = 0.0
        
    def initialize(self, config: Dict) -> bool:
        """
        Initialize the Profit Compounder with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if 'target_compound_rate' in config:
                self.target_compound_rate = float(config['target_compound_rate'])
                
            if 'compound_frequency' in config:
                self.compound_frequency = config['compound_frequency']
                
            # Initialize last compound time based on frequency
            self.last_compound_time = self._get_last_period_start()
            
            self.logger.info(f"Profit Compounder initialized with rate={self.target_compound_rate}%, "
                             f"frequency={self.compound_frequency}")
                             
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Profit Compounder: {str(e)}")
            return False
    
    def set_initial_balance(self, balance: float) -> None:
        """
        Set the initial account balance.
        
        Args:
            balance: Initial account balance
        """
        self.initial_balance = balance
        self.current_balance = balance
        
    def update_current_balance(self, balance: float) -> None:
        """
        Update the current account balance.
        
        Args:
            balance: Current account balance
        """
        self.current_balance = balance
        
    def execute_compounding(self, account_balance: Dict) -> Dict:
        """
        Execute the compounding strategy if it's time to compound.
        
        Args:
            account_balance: Account balance information
            
        Returns:
            Dictionary with compounding results
        """
        if not self._is_time_to_compound():
            return {'success': False, 'reason': 'Not time to compound yet'}
            
        if not account_balance:
            return {'success': False, 'reason': 'No account balance data'}
            
        try:
            # Extract relevant balance data
            total_equity = float(account_balance.get('totalEq', 0))
            
            # Find the trading asset (e.g., USDT)
            available_balance = 0.0
            trading_asset = None
            
            for asset in account_balance.get('details', []):
                currency = asset.get('ccy')
                if currency in ['USDT', 'USD', 'USDC']:
                    available_balance = float(asset.get('availBal', 0))
                    trading_asset = currency
                    break
            
            if not trading_asset or available_balance <= 0:
                return {'success': False, 'reason': 'No available balance to compound'}
                
            # Calculate profits since last compounding
            profit = total_equity - self.current_balance
            
            if profit <= 0:
                self.last_compound_time = datetime.now()
                return {'success': False, 'reason': 'No profit to compound'}
                
            # Calculate amount to compound based on target rate
            compound_amount = profit * (self.target_compound_rate / 100)
            
            # Record compounding action
            compound_data = {
                'timestamp': datetime.now().isoformat(),
                'profit': profit,
                'compound_amount': compound_amount,
                'balance_before': self.current_balance,
                'balance_after': self.current_balance + compound_amount
            }
            
            self.compound_history.append(compound_data)
            
            # Update current balance and last compound time
            self.current_balance += compound_amount
            self.last_compound_time = datetime.now()
            
            self.logger.info(f"Executed compounding: {compound_amount:.2f} {trading_asset} "
                             f"({self.target_compound_rate:.1f}% of {profit:.2f} profit)")
                             
            return {
                'success': True,
                'amount': compound_amount,
                'asset': trading_asset,
                'timestamp': datetime.now().isoformat(),
                'new_balance': self.current_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error executing compounding: {str(e)}")
            return {'success': False, 'reason': str(e)}
    
    def get_compounding_history(self) -> List[Dict]:
        """
        Get the compounding history.
        
        Returns:
            List of compounding history entries
        """
        return self.compound_history
    
    def get_next_compound_time(self) -> datetime:
        """
        Get the next scheduled compounding time.
        
        Returns:
            Datetime of next compounding
        """
        if not self.last_compound_time:
            return datetime.now()
            
        if self.compound_frequency == "Daily":
            return self.last_compound_time + timedelta(days=1)
        elif self.compound_frequency == "Weekly":
            return self.last_compound_time + timedelta(days=7)
        elif self.compound_frequency == "Monthly":
            # Approximate a month as 30 days
            return self.last_compound_time + timedelta(days=30)
        else:
            return datetime.now()
    
    def _is_time_to_compound(self) -> bool:
        """
        Check if it's time to execute compounding.
        
        Returns:
            True if it's time to compound, False otherwise
        """
        now = datetime.now()
        
        if not self.last_compound_time:
            return True
            
        time_diff = now - self.last_compound_time
        
        if self.compound_frequency == "Daily":
            return time_diff.total_seconds() >= 86400  # 24 hours
        elif self.compound_frequency == "Weekly":
            return time_diff.total_seconds() >= 604800  # 7 days
        elif self.compound_frequency == "Monthly":
            return time_diff.total_seconds() >= 2592000  # 30 days
        else:
            return False
    
    def _get_last_period_start(self) -> datetime:
        """
        Get the start of the last compounding period.
        
        Returns:
            Datetime of the last period start
        """
        now = datetime.now()
        
        if self.compound_frequency == "Daily":
            # Start of previous day
            return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.compound_frequency == "Weekly":
            # Start of previous week
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.compound_frequency == "Monthly":
            # Start of previous month
            first_day = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month = first_day - timedelta(days=1)
            return last_month.replace(day=1)
        else:
            return now - timedelta(days=1)
