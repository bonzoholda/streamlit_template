from typing import Dict, List, Optional, Any, Union
import numpy as np
from components.strategy import Strategy

class TrendFollowingStrategy(Strategy):
    """
    Trend Following Strategy that uses moving averages to identify market trends
    and generates trading signals based on trend direction.
    
    This strategy works by:
    1. Calculating short-term and long-term exponential moving averages (EMAs)
    2. Generating buy signals when the short-term EMA crosses above the long-term EMA
    3. Generating sell signals when the short-term EMA crosses below the long-term EMA
    4. Implementing stop-loss and take-profit mechanisms to manage risk
    """
    
    def __init__(
        self, 
        short_ema: int = 9, 
        long_ema: int = 21, 
        stop_loss_percentage: float = 2.0, 
        take_profit_percentage: float = 4.0
    ):
        """
        Initialize the Trend Following Strategy.
        
        Args:
            short_ema: Period for short-term exponential moving average
            long_ema: Period for long-term exponential moving average
            stop_loss_percentage: Stop loss percentage
            take_profit_percentage: Take profit percentage
        """
        super().__init__("TrendFollowing", "Trend Following Strategy using EMA crossovers")
        
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        
        self.short_ema_values = []
        self.long_ema_values = []
        self.last_signal = None
        self.last_signal_price = 0.0
        
        # Strategy parameters
        self.parameters = [
            {'name': 'short_ema', 'value': short_ema, 'type': 'int', 'description': 'Short EMA period'},
            {'name': 'long_ema', 'value': long_ema, 'type': 'int', 'description': 'Long EMA period'},
            {'name': 'stop_loss_percentage', 'value': stop_loss_percentage, 'type': 'float', 'description': 'Stop loss percentage'},
            {'name': 'take_profit_percentage', 'value': take_profit_percentage, 'type': 'float', 'description': 'Take profit percentage'}
        ]
        
    def analyze(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market data and generate trading signals based on EMA crossovers.
        
        Args:
            market_data: Dictionary containing market data and price history
            
        Returns:
            Trading signal dictionary or None if no action needed
        """
        if not market_data:
            self.logger.warning("Market data is empty")
            return None
            
        # Extract price data
        price_data = self._extract_price_data(market_data)
        if not price_data or len(price_data) < self.long_ema:
            self.logger.warning(f"Not enough price data. Need at least {self.long_ema} data points")
            return None
            
        # Calculate EMAs
        self.short_ema_values = self.calculate_ema(price_data, self.short_ema)
        self.long_ema_values = self.calculate_ema(price_data, self.long_ema)
        
        if len(self.short_ema_values) < 2 or len(self.long_ema_values) < 2:
            self.logger.warning("Not enough data to calculate EMA crossover")
            return None
            
        # Get current values
        current_short_ema = self.short_ema_values[-1]
        current_long_ema = self.long_ema_values[-1]
        previous_short_ema = self.short_ema_values[-2]
        previous_long_ema = self.long_ema_values[-2]
        
        current_price = float(market_data.get('last', price_data[-1]))
        
        # Check for crossover
        signal = None
        
        # Buy signal: Short EMA crosses above Long EMA
        if (previous_short_ema <= previous_long_ema and current_short_ema > current_long_ema):
            signal = {
                'action': 'buy',
                'price': current_price,
                'reason': 'EMA Crossover (bullish)',
                'short_ema': current_short_ema,
                'long_ema': current_long_ema
            }
            self.last_signal = 'buy'
            self.last_signal_price = current_price
            
        # Sell signal: Short EMA crosses below Long EMA
        elif (previous_short_ema >= previous_long_ema and current_short_ema < current_long_ema):
            signal = {
                'action': 'sell',
                'price': current_price,
                'reason': 'EMA Crossover (bearish)',
                'short_ema': current_short_ema,
                'long_ema': current_long_ema
            }
            self.last_signal = 'sell'
            self.last_signal_price = current_price
            
        # Check stop loss and take profit if we have an active position
        elif self.last_signal and self.last_signal_price > 0:
            signal = self._check_exit_conditions(current_price)
            
        return signal
        
    def update_parameters(self, parameters: Dict) -> None:
        """
        Update trend following strategy parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if 'short_ema' in parameters:
            self.short_ema = int(parameters['short_ema'])
        if 'long_ema' in parameters:
            self.long_ema = int(parameters['long_ema'])
        if 'stop_loss_percentage' in parameters:
            self.stop_loss_percentage = float(parameters['stop_loss_percentage'])
        if 'take_profit_percentage' in parameters:
            self.take_profit_percentage = float(parameters['take_profit_percentage'])
            
        # Update the parameters list
        self.parameters = [
            {'name': 'short_ema', 'value': self.short_ema, 'type': 'int', 'description': 'Short EMA period'},
            {'name': 'long_ema', 'value': self.long_ema, 'type': 'int', 'description': 'Long EMA period'},
            {'name': 'stop_loss_percentage', 'value': self.stop_loss_percentage, 'type': 'float', 'description': 'Stop loss percentage'},
            {'name': 'take_profit_percentage', 'value': self.take_profit_percentage, 'type': 'float', 'description': 'Take profit percentage'}
        ]
        
    def calculate_ema(self, data: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: List of price data
            period: EMA period
            
        Returns:
            List of EMA values
        """
        if len(data) < period:
            return []
            
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate initial SMA (first EMA value)
        ema_values = [sum(data[:period]) / period]
        
        # Calculate EMA for remaining data
        for i in range(period, len(data)):
            ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
            
        return ema_values
        
    def identify_trend(self, short_ema: List[float], long_ema: List[float]) -> str:
        """
        Identify the current market trend based on EMA values.
        
        Args:
            short_ema: List of short-term EMA values
            long_ema: List of long-term EMA values
            
        Returns:
            Trend direction ("bullish", "bearish", or "sideways")
        """
        if len(short_ema) == 0 or len(long_ema) == 0:
            return "unknown"
            
        # Get the latest values
        current_short = short_ema[-1]
        current_long = long_ema[-1]
        
        # Calculate the difference as a percentage
        difference_pct = (current_short - current_long) / current_long * 100
        
        if difference_pct > 1.0:
            return "bullish"
        elif difference_pct < -1.0:
            return "bearish"
        else:
            return "sideways"
            
    def _extract_price_data(self, market_data: Dict) -> List[float]:
        """
        Extract price data from market data.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            List of close prices
        """
        # If market data contains historical candles, use them
        if 'candles' in market_data and market_data['candles']:
            return [float(candle['close']) for candle in market_data['candles']]
        
        # If the data is a single price point, use it
        if 'last' in market_data:
            return [float(market_data['last'])]
            
        return []
        
    def _check_exit_conditions(self, current_price: float) -> Optional[Dict]:
        """
        Check if stop loss or take profit conditions are met.
        
        Args:
            current_price: Current market price
            
        Returns:
            Exit signal if conditions are met, None otherwise
        """
        if not self.last_signal or self.last_signal_price <= 0:
            return None
            
        # Calculate price changes
        price_change_pct = ((current_price - self.last_signal_price) / self.last_signal_price) * 100
        
        # For long positions
        if self.last_signal == 'buy':
            # Check stop loss
            if price_change_pct <= -self.stop_loss_percentage:
                return {
                    'action': 'sell',
                    'price': current_price,
                    'reason': 'Stop Loss',
                    'entry_price': self.last_signal_price,
                    'price_change_pct': price_change_pct
                }
            # Check take profit
            elif price_change_pct >= self.take_profit_percentage:
                return {
                    'action': 'sell',
                    'price': current_price,
                    'reason': 'Take Profit',
                    'entry_price': self.last_signal_price,
                    'price_change_pct': price_change_pct
                }
                
        # For short positions
        elif self.last_signal == 'sell':
            # Check stop loss
            if price_change_pct >= self.stop_loss_percentage:
                return {
                    'action': 'buy',
                    'price': current_price,
                    'reason': 'Stop Loss',
                    'entry_price': self.last_signal_price,
                    'price_change_pct': -price_change_pct
                }
            # Check take profit
            elif price_change_pct <= -self.take_profit_percentage:
                return {
                    'action': 'buy',
                    'price': current_price,
                    'reason': 'Take Profit',
                    'entry_price': self.last_signal_price,
                    'price_change_pct': -price_change_pct
                }
                
        return None
