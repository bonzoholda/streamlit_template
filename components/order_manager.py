from typing import Dict, List, Optional, Any, Union, Mapping
import logging
import time
from datetime import datetime

class OrderManager:
    """
    Order Manager for handling order placement, cancellation, and tracking.
    
    Responsibilities:
    1. Placing orders through the exchange connector
    2. Cancelling or modifying existing orders
    3. Tracking open orders and order history
    4. Monitoring order status and execution
    """
    
    def __init__(self, exchange):
        """
        Initialize the Order Manager.
        
        Args:
            exchange: Exchange connector for API communication
        """
        self.exchange = exchange
        self.open_orders = {}  # Map of order_id to order details
        self.order_history = []  # List of past orders
        self.logger = logging.getLogger(__name__)
        
    def place_order(self, order_details: Dict) -> Dict:
        """
        Place a new order on the exchange.
        
        Args:
            order_details: Order parameters dictionary
            
        Returns:
            Order response or empty dict if failed
        """
        try:
            # Validate order details
            if not self._validate_order_details(order_details):
                self.logger.error(f"Invalid order details: {order_details}")
                return {}
            
            # Place the order
            order_result = self.exchange.place_order(order_details)
            
            if not order_result:
                self.logger.error("Order placement failed")
                return {}
            
            # Store the order in open orders
            order_id = order_result.get('ordId')
            if order_id:
                self.open_orders[order_id] = {
                    **order_result,
                    **order_details,
                    'created_time': datetime.now().isoformat()
                }
                
                self.logger.info(f"Order placed: {order_id}, {order_details.get('instId')}, "
                                f"{order_details.get('side')}, {order_details.get('sz')} units")
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            if order_id not in self.open_orders:
                self.logger.warning(f"Order ID not found in open orders: {order_id}")
                return False
                
            # Get the symbol for the order
            symbol = self.open_orders[order_id].get('instId')
            if not symbol:
                self.logger.error(f"Symbol not found for order ID: {order_id}")
                return False
                
            # Cancel the order
            result = self.exchange.cancel_order(order_id, symbol)
            
            if result:
                # Move order from open_orders to order_history
                order = self.open_orders.pop(order_id)
                order['status'] = 'cancelled'
                order['cancel_time'] = datetime.now().isoformat()
                self.order_history.append(order)
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def modify_order(self, order_id: str, new_details: Dict) -> Dict:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            new_details: New order parameters
            
        Returns:
            Modified order details or empty dict if failed
        """
        try:
            # OKX doesn't support direct order modification - we need to cancel and replace
            if order_id not in self.open_orders:
                self.logger.warning(f"Order ID not found in open orders: {order_id}")
                return {}
                
            # Get the current order details
            current_order = self.open_orders.get(order_id, {})
            
            # Cancel the existing order
            cancel_success = self.cancel_order(order_id)
            if not cancel_success:
                self.logger.error(f"Failed to cancel order for modification: {order_id}")
                return {}
                
            # Merge current and new details
            merged_details = {**current_order, **new_details}
            
            # Remove fields that shouldn't be included in the new order
            for field in ['ordId', 'clOrdId', 'created_time', 'status']:
                if field in merged_details:
                    del merged_details[field]
            
            # Place a new order with the merged details
            new_order = self.place_order(merged_details)
            
            if new_order:
                self.logger.info(f"Order modified: {order_id} -> {new_order.get('ordId')}")
                return new_order
            else:
                self.logger.error(f"Failed to place new order during modification")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            return {}
    
    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders.
        
        Returns:
            List of open order details
        """
        try:
            # Refresh open orders from exchange
            self._refresh_open_orders()
            
            # Return list of open orders
            return list(self.open_orders.values())
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")
            return []
    
    def get_order_history(self, filters: Dict = None) -> List[Dict]:
        """
        Get order history with optional filtering.
        
        Args:
            filters: Filter parameters (e.g., symbol, start_time, end_time)
            
        Returns:
            List of order history matching filters
        """
        try:
            if not filters:
                return self.order_history
                
            filtered_history = self.order_history
            
            # Apply filters
            if 'symbol' in filters:
                filtered_history = [order for order in filtered_history 
                                   if order.get('instId') == filters['symbol']]
                
            if 'start_time' in filters:
                start_time = filters['start_time']
                filtered_history = [order for order in filtered_history 
                                   if order.get('created_time', '') >= start_time]
                
            if 'end_time' in filters:
                end_time = filters['end_time']
                filtered_history = [order for order in filtered_history 
                                   if order.get('created_time', '') <= end_time]
                
            if 'status' in filters:
                filtered_history = [order for order in filtered_history 
                                   if order.get('status') == filters['status']]
                
            return filtered_history
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {str(e)}")
            return []
    
    def _refresh_open_orders(self) -> None:
        """
        Refresh the local open orders cache from the exchange.
        """
        try:
            # Get open orders from exchange
            exchange_open_orders = self.exchange.get_open_orders()
            
            if not exchange_open_orders:
                return
                
            # Convert to dictionary for easier lookup
            order_dict = {order['ordId']: order for order in exchange_open_orders if 'ordId' in order}
            
            # Update local cache
            for order_id, order_details in order_dict.items():
                if order_id in self.open_orders:
                    # Update existing order
                    self.open_orders[order_id].update(order_details)
                else:
                    # Add new order
                    self.open_orders[order_id] = order_details
                    
            # Remove orders that are no longer open
            closed_order_ids = []
            for order_id in self.open_orders:
                if order_id not in order_dict:
                    # Order is no longer open, move to history
                    order = self.open_orders[order_id]
                    order['status'] = 'filled'  # Assume it was filled
                    order['fill_time'] = datetime.now().isoformat()
                    self.order_history.append(order)
                    closed_order_ids.append(order_id)
                    
            for order_id in closed_order_ids:
                self.open_orders.pop(order_id)
                
        except Exception as e:
            self.logger.error(f"Error refreshing open orders: {str(e)}")
    
    def _validate_order_details(self, order_details: Dict) -> bool:
        """
        Validate order details before submission.
        
        Args:
            order_details: Order parameters dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['instId', 'tdMode', 'side', 'ordType', 'sz']
        for field in required_fields:
            if field not in order_details:
                self.logger.error(f"Missing required field in order details: {field}")
                return False
                
        # Check trade mode
        if order_details['tdMode'] not in ['cash', 'cross', 'isolated']:
            self.logger.error(f"Invalid trade mode: {order_details['tdMode']}")
            return False
            
        # Check side
        if order_details['side'] not in ['buy', 'sell']:
            self.logger.error(f"Invalid side: {order_details['side']}")
            return False
            
        # Check order type
        if order_details['ordType'] not in ['market', 'limit', 'post_only', 'fok', 'ioc']:
            self.logger.error(f"Invalid order type: {order_details['ordType']}")
            return False
            
        # Check if price is included for limit orders
        if order_details['ordType'] in ['limit', 'post_only'] and 'px' not in order_details:
            self.logger.error(f"Price required for {order_details['ordType']} orders")
            return False
            
        # Check order size
        try:
            sz = float(order_details['sz'])
            if sz <= 0:
                self.logger.error(f"Invalid order size: {sz}")
                return False
        except (ValueError, TypeError):
            self.logger.error(f"Invalid order size format: {order_details['sz']}")
            return False
            
        return True
