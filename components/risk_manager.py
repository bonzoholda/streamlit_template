from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging

class RiskManager:
    """
    Risk Manager for evaluating trading risks and enforcing risk limits.
    
    Responsibilities:
    1. Evaluating risk for each trading signal
    2. Calculating appropriate position sizes based on risk parameters
    3. Enforcing risk limits to prevent excessive exposure
    4. Tracking risk exposure and drawdown
    """
    
    def __init__(
        self,
        max_drawdown: float = 15.0,  # Maximum drawdown percentage
        max_position_size: float = 10.0,  # Maximum position size as percentage of balance
        daily_risk_limit: float = 5.0  # Maximum daily risk as percentage of balance
    ):
        """
        Initialize the Risk Manager.
        
        Args:
            max_drawdown: Maximum allowable drawdown percentage
            max_position_size: Maximum position size as percentage of balance
            daily_risk_limit: Maximum daily risk as percentage of balance
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.daily_risk_limit = daily_risk_limit
        self.current_risk_exposure = 0.0
        self.daily_pnl = 0.0
        self.last_reset_day = None
        self.logger = logging.getLogger(__name__)
        
    def evaluate_risk(self, signal: Dict, balance: Dict) -> Dict:
        """
        Evaluate risk for a trading signal.
        
        Args:
            signal: Trading signal dictionary
            balance: Account balance information
            
        Returns:
            Risk assessment dictionary with decision and details
        """
        if not signal or not balance:
            return {'approved': False, 'reason': 'Invalid signal or balance data'}
        
        # Extract the relevant balance information
        total_equity = float(balance.get('totalEq', 0))
        if total_equity <= 0:
            return {'approved': False, 'reason': 'Insufficient balance'}
        
        # Check signal validity
        if 'action' not in signal or signal['action'] not in ['buy', 'sell']:
            return {'approved': False, 'reason': 'Invalid signal action'}
        
        # If this is a stop loss or take profit, approve it to manage risk
        if signal.get('reason') in ['Stop Loss', 'Take Profit']:
            return {
                'approved': True,
                'position_size': signal.get('position_size', 0),
                'risk_percentage': 0  # No new risk as this is closing a position
            }
        
        # Reset daily risk if it's a new day
        from datetime import datetime
        today = datetime.now().date()
        if self.last_reset_day is None or self.last_reset_day != today:
            self.daily_pnl = 0.0
            self.last_reset_day = today
        
        # Check if we've exceeded the daily risk limit
        if abs(self.daily_pnl) > (total_equity * self.daily_risk_limit / 100):
            return {'approved': False, 'reason': 'Daily risk limit exceeded'}
        
        # Calculate the position size based on risk parameters
        price = float(signal.get('price', 0))
        if price <= 0:
            return {'approved': False, 'reason': 'Invalid price in signal'}
        
        # Get stop loss percentage from signal or use default
        stop_loss_pct = signal.get('stop_loss_percentage', 2.0)
        
        # Calculate position size based on risk percentage
        risk_per_trade = total_equity * 0.01  # Risk 1% of equity per trade
        
        # Calculate position size using stop loss
        position_size = self.calculate_position_size(price, stop_loss_pct, balance)
        
        # Enforce position size limit
        max_allowed_size = total_equity * self.max_position_size / 100
        if position_size > max_allowed_size:
            position_size = max_allowed_size
            
        # Enforce minimum position size
        min_position_size = 0.001  # Minimum position size
        if position_size < min_position_size:
            return {'approved': False, 'reason': 'Position size too small'}
        
        # Calculate potential drawdown
        potential_loss = position_size * stop_loss_pct / 100
        potential_drawdown_pct = (potential_loss / total_equity) * 100
        
        # Check if potential drawdown exceeds limit
        if potential_drawdown_pct > self.max_drawdown:
            return {
                'approved': False, 
                'reason': f'Potential drawdown ({potential_drawdown_pct:.2f}%) exceeds limit ({self.max_drawdown:.2f}%)'
            }
        
        # Approve the trade
        return {
            'approved': True,
            'position_size': position_size,
            'risk_percentage': potential_drawdown_pct,
            'stop_loss_price': self._calculate_stop_loss_price(price, stop_loss_pct, signal['action'])
        }
    
    def calculate_position_size(self, price: float, risk_percentage: float, balance: Dict) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            price: Current price
            risk_percentage: Risk percentage for stop loss
            balance: Account balance information
            
        Returns:
            Position size to trade
        """
        if price <= 0 or risk_percentage <= 0:
            return 0
            
        total_equity = float(balance.get('totalEq', 0))
        if total_equity <= 0:
            return 0
        
        # Risk-based position sizing: Risk 1% of account per trade
        risk_amount = total_equity * 0.01  # 1% of equity
        
        # Position size = Risk amount / (Price * (Risk percentage / 100))
        position_size = risk_amount / (price * (risk_percentage / 100))
        
        # Round to appropriate precision based on price
        if price < 10:
            position_size = round(position_size, 3)
        elif price < 100:
            position_size = round(position_size, 2)
        elif price < 1000:
            position_size = round(position_size, 1)
        else:
            position_size = round(position_size)
            
        return position_size
    
    def enforce_risk_limits(self, order: Dict) -> Dict:
        """
        Enforce risk limits on an order.
        
        Args:
            order: Order details dictionary
            
        Returns:
            Modified order with risk limits applied
        """
        # Make a copy of the order to avoid modifying the original
        adjusted_order = order.copy()
        
        # Apply position size limits
        if 'sz' in adjusted_order:
            sz = float(adjusted_order['sz'])
            
            # Ensure size is not too large (implementation-specific)
            if sz > self.max_position_size:
                adjusted_order['sz'] = str(self.max_position_size)
                self.logger.warning(f"Position size reduced from {sz} to {self.max_position_size} due to risk limits")
        
        return adjusted_order
    
    def update_risk_parameters(self, params: Dict) -> None:
        """
        Update risk management parameters.
        
        Args:
            params: Dictionary of risk parameters to update
        """
        if 'max_drawdown' in params:
            self.max_drawdown = float(params['max_drawdown'])
        if 'max_position_size' in params:
            self.max_position_size = float(params['max_position_size'])
        if 'daily_risk_limit' in params:
            self.daily_risk_limit = float(params['daily_risk_limit'])
            
        self.logger.info(f"Risk parameters updated: max_drawdown={self.max_drawdown}%, "
                         f"max_position_size={self.max_position_size}%, "
                         f"daily_risk_limit={self.daily_risk_limit}%")
    
    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update the daily PnL tracker.
        
        Args:
            pnl: Profit/loss amount to add
        """
        self.daily_pnl += pnl
        
    def _calculate_stop_loss_price(self, price: float, stop_loss_pct: float, action: str) -> float:
        """
        Calculate stop loss price based on entry price, percentage, and direction.
        
        Args:
            price: Entry price
            stop_loss_pct: Stop loss percentage
            action: Trade direction ('buy' or 'sell')
            
        Returns:
            Stop loss price
        """
        if action == 'buy':
            return price * (1 - stop_loss_pct / 100)
        elif action == 'sell':
            return price * (1 + stop_loss_pct / 100)
        return 0
