from typing import Dict, List, Optional, Any, Union
import numpy as np
from components.strategy import Strategy

class GridStrategy(Strategy):
    """
    Grid Trading Strategy that places buy and sell orders at regular price intervals.
    
    This strategy works by:
    1. Defining a price range (upper and lower limits)
    2. Dividing the range into a grid of equally spaced price levels
    3. Placing buy orders at grid levels below the current price
    4. Placing sell orders at grid levels above the current price
    5. When orders are filled, placing new orders on the opposite side
    """
    
    def __init__(self, upper_limit: float = 0, lower_limit: float = 0, grid_levels: int = 5):
        """
        Initialize the Grid Trading Strategy.
        
        Args:
            upper_limit: Upper price limit of the grid
            lower_limit: Lower price limit of the grid
            grid_levels: Number of grid levels to create
        """
        super().__init__("GridTrading", "Grid Trading Strategy that places orders at regular price intervals")
        
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.grid_levels = grid_levels
        self.current_positions = []
        self.grid_prices = []
        
        # Strategy parameters
        self.parameters = [
            {'name': 'upper_limit', 'value': upper_limit, 'type': 'float', 'description': 'Upper price limit'},
            {'name': 'lower_limit', 'value': lower_limit, 'type': 'float', 'description': 'Lower price limit'},
            {'name': 'grid_levels', 'value': grid_levels, 'type': 'int', 'description': 'Number of grid levels'}
        ]
        
    def analyze(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market data and generate trading signals for grid strategy.
        
        Args:
            market_data: Dictionary containing market data and price history
            
        Returns:
            Trading signal dictionary or None if no action needed
        """
        if not market_data or 'last' not in market_data:
            self.logger.warning("Market data is missing required information")
            return None
            
        current_price = float(market_data['last'])
        
        # If upper and lower limits are not set, calculate them based on the current price
        if self.upper_limit <= 0 or self.lower_limit <= 0:
            self._auto_set_limits(current_price)
            
        # Calculate grid levels if not already calculated
        if not self.grid_prices:
            self.calculate_grid_levels()
            
        # Find the closest grid levels to the current price
        buy_level, sell_level = self._find_nearest_levels(current_price)
        
        # Logic for generating buy/sell signals
        if buy_level is not None and current_price <= buy_level['price']:
            # Generate buy signal
            return {
                'action': 'buy',
                'price': buy_level['price'],
                'level': buy_level['level'],
                'reason': f"Price hit grid buy level {buy_level['level']}"
            }
        elif sell_level is not None and current_price >= sell_level['price']:
            # Generate sell signal
            return {
                'action': 'sell',
                'price': sell_level['price'],
                'level': sell_level['level'],
                'reason': f"Price hit grid sell level {sell_level['level']}"
            }
            
        return None
        
    def update_parameters(self, parameters: Dict) -> None:
        """
        Update grid strategy parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if 'upper_limit' in parameters:
            self.upper_limit = float(parameters['upper_limit'])
        if 'lower_limit' in parameters:
            self.lower_limit = float(parameters['lower_limit'])
        if 'grid_levels' in parameters:
            self.grid_levels = int(parameters['grid_levels'])
            
        # Update the parameters list
        self.parameters = [
            {'name': 'upper_limit', 'value': self.upper_limit, 'type': 'float', 'description': 'Upper price limit'},
            {'name': 'lower_limit', 'value': self.lower_limit, 'type': 'float', 'description': 'Lower price limit'},
            {'name': 'grid_levels', 'value': self.grid_levels, 'type': 'int', 'description': 'Number of grid levels'}
        ]
        
        # Recalculate grid levels with new parameters
        self.calculate_grid_levels()
        
    def calculate_grid_levels(self) -> List[Dict]:
        """
        Calculate the grid price levels based on upper and lower limits.
        
        Returns:
            List of dictionaries containing grid level information
        """
        if self.upper_limit <= self.lower_limit:
            self.logger.error("Upper limit must be greater than lower limit")
            return []
            
        # Calculate price step
        price_step = (self.upper_limit - self.lower_limit) / self.grid_levels
        
        # Generate grid levels
        self.grid_prices = []
        for i in range(self.grid_levels + 1):
            price = self.lower_limit + (i * price_step)
            self.grid_prices.append({
                'level': i,
                'price': price,
                'type': 'buy' if i < self.grid_levels / 2 else 'sell'
            })
            
        self.logger.info(f"Grid levels calculated: {len(self.grid_prices)} levels")
        return self.grid_prices
        
    def adjust_grid(self, market_condition: Dict) -> None:
        """
        Adjust the grid based on market conditions.
        
        Args:
            market_condition: Dictionary containing market condition data
        """
        current_price = market_condition.get('price', 0)
        volatility = market_condition.get('volatility', 0)
        
        if current_price <= 0:
            return
            
        # Adjust grid based on volatility
        if volatility > 0:
            new_upper_limit = current_price * (1 + volatility * 2)
            new_lower_limit = current_price * (1 - volatility * 2)
            
            self.upper_limit = new_upper_limit
            self.lower_limit = new_lower_limit
            
            # Recalculate grid levels
            self.calculate_grid_levels()
            
    def _auto_set_limits(self, current_price: float) -> None:
        """
        Automatically set upper and lower limits based on the current price.
        
        Args:
            current_price: Current market price
        """
        # Default to 10% above and below current price if limits are not set
        self.upper_limit = current_price * 1.1
        self.lower_limit = current_price * 0.9
        
        self.logger.info(f"Auto-set grid limits: Lower={self.lower_limit:.2f}, Upper={self.upper_limit:.2f}")
        
    def _find_nearest_levels(self, current_price: float) -> tuple:
        """
        Find the nearest buy and sell grid levels to the current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (nearest buy level, nearest sell level)
        """
        if not self.grid_prices:
            return None, None
            
        # Find closest buy level (below current price)
        buy_levels = [level for level in self.grid_prices if level['price'] <= current_price and level['type'] == 'buy']
        nearest_buy = max(buy_levels, key=lambda x: x['price']) if buy_levels else None
        
        # Find closest sell level (above current price)
        sell_levels = [level for level in self.grid_prices if level['price'] >= current_price and level['type'] == 'sell']
        nearest_sell = min(sell_levels, key=lambda x: x['price']) if sell_levels else None
        
        return nearest_buy, nearest_sell
