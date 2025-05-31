from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging

class Strategy(ABC):
    """
    Strategy interface that defines methods all concrete trading strategies must implement.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            description: Description of what the strategy does
        """
        self.name = name
        self.description = description
        self.parameters = []
        self.logger = logging.getLogger(__name__)
        self._performance = {
            'trades_count': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'pnl': 0.0,
            'win_rate': 0.0
        }
    
    @abstractmethod
    def analyze(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market data and generate a trading signal.
        
        Args:
            market_data: Dictionary containing market data including price history
            
        Returns:
            Optional signal dictionary with action (buy/sell), price, etc. or None
        """
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        pass
    
    def get_performance(self) -> Dict:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._performance
    
    def update_performance(self, trade_result: Dict) -> None:
        """
        Update performance metrics based on trade result.
        
        Args:
            trade_result: Dictionary containing trade result data
        """
        self._performance['trades_count'] += 1
        
        if trade_result.get('success', False):
            self._performance['successful_trades'] += 1
            self._performance['pnl'] += trade_result.get('pnl', 0.0)
        else:
            self._performance['failed_trades'] += 1
            self._performance['pnl'] += trade_result.get('pnl', 0.0)
        
        # Calculate win rate
        if self._performance['trades_count'] > 0:
            self._performance['win_rate'] = (self._performance['successful_trades'] / 
                                             self._performance['trades_count']) * 100
