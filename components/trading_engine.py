import time
import threading
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

from components.strategy import Strategy
from components.risk_manager import RiskManager
from components.order_manager import OrderManager
from components.profit_compounder import ProfitCompounder
from components.market_data_manager import MarketDataManager
from components.database_service import DatabaseService
from components.config import Config

class TradingEngine:
    """
    Trading Engine responsible for orchestrating the trading process.
    
    This component coordinates the trading workflow by:
    1. Getting market data from the MarketDataManager
    2. Analyzing the data using trading strategies
    3. Evaluating risk using the RiskManager
    4. Executing orders through the OrderManager
    5. Managing profit compounding through the ProfitCompounder
    6. Storing trade data using the DatabaseService
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        risk_manager: RiskManager,
        order_manager: OrderManager,
        profit_compounder: ProfitCompounder,
        market_data_manager: MarketDataManager,
        database_service: DatabaseService,
        config: Config
    ):
        """
        Initialize the TradingEngine with its dependencies.
        
        Args:
            strategies: List of trading strategies to use
            risk_manager: Risk manager for evaluating trading risks
            order_manager: Order manager for executing trades
            profit_compounder: Profit compounder for reinvesting profits
            market_data_manager: Market data manager for retrieving market data
            database_service: Database service for storing trade data
            config: Configuration parameters
        """
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.profit_compounder = profit_compounder
        self.market_data_manager = market_data_manager
        self.database_service = database_service
        self.config = config
        
        self.logger = logging.getLogger(__name__)
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.status = "Initialized"
        self.last_update_time = datetime.now()
        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def initialize(self, config: Dict = None) -> bool:
        """
        Initialize the trading engine with configuration.
        
        Args:
            config: Configuration dictionary (optional)
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if config:
                self.config.update_config(config)
            
            # Initialize all components
            self.market_data_manager.initialize()
            for strategy in self.strategies:
                strategy.update_parameters(self.config.get_strategy_parameters())
            
            self.risk_manager.update_risk_parameters(self.config.get_risk_parameters())
            self.profit_compounder.initialize(self.config.get_compound_config())
            
            self.status = "Ready"
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trading engine: {str(e)}")
            self.status = "Error"
            return False
    
    def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return True
        
        if self.status == "Error":
            self.logger.error("Cannot start trading engine in error state")
            return False
        
        try:
            # Connect to the exchange
            if not self.market_data_manager.exchange.connect():
                self.logger.error("Failed to connect to exchange")
                self.status = "Error"
                return False
            
            self.is_running = True
            self.status = "Running"
            
            # Start trading in a separate thread
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.logger.info("Trading engine started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading engine: {str(e)}")
            self.status = "Error"
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the trading engine.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Trading engine is already stopped")
            return True
        
        try:
            self.is_running = False
            
            # Wait for trading thread to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Disconnect from the exchange
            self.market_data_manager.exchange.disconnect()
            
            self.status = "Stopped"
            self.logger.info("Trading engine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {str(e)}")
            self.status = "Error"
            return False
    
    def get_status(self) -> str:
        """
        Get current status of the trading engine.
        
        Returns:
            Status string
        """
        return self.status
    
    def update_config(self, config: Dict) -> bool:
        """
        Update configuration and reinitialize components.
        
        Args:
            config: New configuration dictionary
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            was_running = self.is_running
            
            # Stop trading if it's running
            if was_running:
                self.stop()
            
            # Update configuration
            self.config.update_config(config)
            
            # Re-initialize components
            success = self.initialize()
            
            # Restart trading if it was running before
            if was_running and success:
                self.start()
                
            return success
        
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            self.status = "Error"
            return False
    
    def _trading_loop(self) -> None:
        """
        Main trading loop that runs in a separate thread.
        """
        self.logger.info("Trading loop started")
        
        trading_symbol = self.config.get_trading_symbol()
        trading_interval = self.config.get_trading_interval()
        
        last_check_time = datetime.now()
        compounding_check_time = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if it's time to analyze market data (based on interval)
                if (current_time - last_check_time) >= self._get_interval_timedelta(trading_interval):
                    last_check_time = current_time
                    
                    # Get market data
                    market_data = self.market_data_manager.get_market_data(trading_symbol)
                    
                    if not market_data:
                        self.logger.warning(f"No market data available for {trading_symbol}")
                        continue
                    
                    # Get current account balance
                    account_balance = self.market_data_manager.exchange.get_account_balance()
                    
                    # Analyze market data with all strategies
                    for strategy in self.strategies:
                        signal = strategy.analyze(market_data)
                        
                        if signal:
                            # Evaluate risk
                            risk_assessment = self.risk_manager.evaluate_risk(signal, account_balance)
                            
                            if risk_assessment.get('approved', False):
                                # Place order
                                order_details = {
                                    'instId': trading_symbol,
                                    'tdMode': 'cross',  # Could be configurable
                                    'side': signal['action'],
                                    'ordType': 'market',  # Could be configurable
                                    'sz': risk_assessment['position_size'],
                                }
                                
                                # Add price if it's a limit order
                                if signal.get('price'):
                                    order_details['px'] = signal['price']
                                    order_details['ordType'] = 'limit'
                                
                                # Execute order
                                order_result = self.order_manager.place_order(order_details)
                                
                                if order_result:
                                    self.trade_count += 1
                                    self.successful_trades += 1
                                    
                                    # Save trade data
                                    trade_data = {
                                        'symbol': trading_symbol,
                                        'type': signal['action'],
                                        'entry_time': datetime.now().isoformat(),
                                        'entry_price': float(order_result.get('avgPx', signal.get('price', 0))),
                                        'position_size': float(order_result.get('sz', order_details['sz'])),
                                        'order_id': order_result.get('ordId', ''),
                                        'strategy': strategy.name,
                                        'status': 'Open'
                                    }
                                    
                                    self.database_service.save_trade_data(trade_data)
                                    self.logger.info(f"Trade executed: {trading_symbol} {signal['action']} at {trade_data['entry_price']}")
                                else:
                                    self.failed_trades += 1
                                    self.logger.error(f"Failed to execute trade: {trading_symbol} {signal['action']}")
                            else:
                                self.logger.info(f"Trade rejected by risk manager: {risk_assessment.get('reason', 'Unknown reason')}")
                    
                    # Check for profit compounding (daily)
                    if (current_time - compounding_check_time) >= timedelta(hours=24):
                        compounding_check_time = current_time
                        
                        # Execute profit compounding
                        if self.config.is_compounding_enabled():
                            compound_result = self.profit_compounder.execute_compounding(account_balance)
                            
                            if compound_result.get('success', False):
                                self.logger.info(f"Profit compounding executed: {compound_result.get('amount', 0)} USDT")
                
                # Save state periodically
                if (current_time - self.last_update_time) >= timedelta(minutes=5):
                    self.last_update_time = current_time
                    self._save_state()
                
                # Sleep to prevent high CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)  # Sleep longer if there's an error
        
        self.logger.info("Trading loop ended")
    
    def _get_interval_timedelta(self, interval: str) -> timedelta:
        """
        Convert interval string to timedelta object.
        
        Args:
            interval: Interval string (e.g., "1m", "5m", "1h")
            
        Returns:
            Timedelta object representing the interval
        """
        value = int(interval[:-1])
        unit = interval[-1]
        
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            return timedelta(minutes=15)  # Default to 15 minutes
    
    def _save_state(self) -> None:
        """
        Save current state and metrics.
        """
        try:
            # Calculate performance metrics
            win_rate = self.successful_trades / self.trade_count if self.trade_count > 0 else 0
            
            metrics = {
                'trade_count': self.trade_count,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'win_rate': win_rate,
                'last_update': datetime.now().isoformat()
            }
            
            # Save to database
            self.database_service.save_performance_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = self.database_service.get_performance_metrics()
            
            # Add additional metrics
            metrics.update({
                'trade_count': self.trade_count,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'win_rate': self.successful_trades / self.trade_count if self.trade_count > 0 else 0,
                'status': self.status,
                'is_running': self.is_running
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                'error': str(e),
                'status': self.status,
                'is_running': self.is_running
            }
    
    def get_active_strategies(self) -> List[Dict]:
        """
        Get information about active trading strategies.
        
        Returns:
            List of strategy information dictionaries
        """
        active_strategies = []
        
        for strategy in self.strategies:
            active_strategies.append({
                'name': strategy.name,
                'description': strategy.description,
                'performance': strategy.get_performance()
            })
        
        return active_strategies
