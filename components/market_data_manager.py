from typing import Dict, List, Optional, Any, Union
import logging
import time
import numpy as np
from datetime import datetime, timedelta

class MarketDataManager:
    """
    Market Data Manager for retrieving and processing market data.
    
    Responsibilities:
    1. Fetching market data from the exchange
    2. Processing and formatting data for strategy analysis
    3. Calculating technical indicators
    4. Providing historical and real-time data
    """
    
    def __init__(self, exchange):
        """
        Initialize the Market Data Manager.
        
        Args:
            exchange: Exchange connector for API communication
        """
        self.exchange = exchange
        self.cached_data = {}  # Cache for market data
        self.cache_expiry = {}  # Expiry timestamps for cached data
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """
        Initialize the Market Data Manager.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Test connection to exchange
            connected = self.exchange.connect()
            if not connected:
                self.logger.error("Failed to connect to exchange")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Market Data Manager: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str = "15m", limit: int = 100) -> Dict:
        """
        Get market data for a symbol, including historical candles.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            timeframe: Candlestick interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of historical candles to retrieve
            
        Returns:
            Dictionary containing market data
        """
        try:
            # Check if we have cached data that's still valid
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.cached_data and time.time() < self.cache_expiry.get(cache_key, 0):
                return self.cached_data[cache_key]
                
            # Get current ticker data
            ticker = self.exchange.get_market_data(symbol)
            if not ticker:
                self.logger.warning(f"Failed to get ticker data for {symbol}")
                return {}
                
            # Get historical candles
            candles = self.exchange.get_historical_candles(symbol, timeframe, limit)
            
            # Combine data
            market_data = {
                **ticker,
                'candles': candles,
                'timestamp': datetime.now().timestamp(),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Calculate technical indicators
            market_data['indicators'] = self._calculate_indicators(candles)
            
            # Calculate volatility
            if candles:
                close_prices = [candle['close'] for candle in candles]
                market_data['volatility'] = self._calculate_volatility(close_prices)
            
            # Cache the data
            self.cached_data[cache_key] = market_data
            
            # Set cache expiry based on timeframe
            expiry_seconds = self._get_timeframe_seconds(timeframe) / 2
            self.cache_expiry[cache_key] = time.time() + expiry_seconds
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            Latest price as a float
        """
        try:
            return self.exchange.get_latest_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return 0.0
    
    def subscribe_to_market_updates(self, symbols: List[str], callback) -> None:
        """
        Subscribe to real-time market updates via WebSocket.
        
        Args:
            symbols: List of trading pairs to subscribe to
            callback: Function to call with received updates
        """
        try:
            # Start an async task to handle WebSocket subscription
            import asyncio
            import threading
            
            async def websocket_task():
                await self.exchange.subscribe_to_market_data(symbols, "tickers")
                await self.exchange.listen_for_market_data(callback)
            
            # Run the async task in a separate thread
            def run_async_task():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(websocket_task())
                loop.close()
            
            # Start the thread
            websocket_thread = threading.Thread(target=run_async_task)
            websocket_thread.daemon = True
            websocket_thread.start()
            
            self.logger.info(f"Subscribed to market updates for {symbols}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to market updates: {str(e)}")
    
    def _calculate_indicators(self, candles: List[Dict]) -> Dict:
        """
        Calculate technical indicators from candlestick data.
        
        Args:
            candles: List of candlestick data
            
        Returns:
            Dictionary of technical indicators
        """
        if not candles or len(candles) < 14:
            return {}
            
        # Extract price data
        close_prices = np.array([candle['close'] for candle in candles])
        high_prices = np.array([candle['high'] for candle in candles])
        low_prices = np.array([candle['low'] for candle in candles])
        
        # Calculate 7-day and 14-day simple moving averages
        sma_7 = np.mean(close_prices[-7:])
        sma_14 = np.mean(close_prices[-14:])
        
        # Calculate 9-day and 21-day exponential moving averages
        ema_9 = self._calculate_ema(close_prices, 9)
        ema_21 = self._calculate_ema(close_prices, 21)
        
        # Calculate RSI (14-period)
        rsi = self._calculate_rsi(close_prices, 14)
        
        # Calculate MACD (12, 26, 9)
        ema_12 = self._calculate_ema(close_prices, 12)
        ema_26 = self._calculate_ema(close_prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = np.mean(macd_line[-9:])
        macd_histogram = macd_line - signal_line
        
        # Calculate Bollinger Bands (20, 2)
        bb_period = 20
        if len(close_prices) >= bb_period:
            bb_sma = np.mean(close_prices[-bb_period:])
            bb_std = np.std(close_prices[-bb_period:])
            bb_upper = bb_sma + (2 * bb_std)
            bb_lower = bb_sma - (2 * bb_std)
        else:
            bb_sma = np.mean(close_prices)
            bb_std = np.std(close_prices)
            bb_upper = bb_sma + (2 * bb_std)
            bb_lower = bb_sma - (2 * bb_std)
        
        return {
            'sma': {
                '7': float(sma_7),
                '14': float(sma_14)
            },
            'ema': {
                '9': float(ema_9),
                '21': float(ema_21)
            },
            'rsi': float(rsi),
            'macd': {
                'line': float(macd_line[-1]) if len(macd_line) > 0 else 0,
                'signal': float(signal_line),
                'histogram': float(macd_histogram[-1]) if len(macd_histogram) > 0 else 0
            },
            'bollinger_bands': {
                'upper': float(bb_upper),
                'middle': float(bb_sma),
                'lower': float(bb_lower)
            }
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Array of price data
            period: EMA period
            
        Returns:
            EMA value
        """
        if len(prices) < period:
            return float(np.mean(prices))
            
        k = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema = (prices[i] * k) + (ema * (1 - k))
            
        return float(ema)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Array of price data
            period: RSI period
            
        Returns:
            RSI value
        """
        if len(prices) < period + 1:
            return 50.0  # Default to neutral if not enough data
            
        # Calculate price changes
        changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.maximum(0, changes)
        losses = np.abs(np.minimum(0, changes))
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_volatility(self, prices: List[float], window: int = 14) -> float:
        """
        Calculate price volatility (standard deviation of returns).
        
        Args:
            prices: List of price data
            window: Period for volatility calculation
            
        Returns:
            Volatility as a percentage
        """
        if len(prices) < window:
            return 0.0
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate standard deviation of returns
        volatility = np.std(returns[-window:])
        
        # Annualize volatility (assuming daily bars)
        annualized_volatility = volatility * np.sqrt(365)
        
        return float(annualized_volatility)
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., "1m", "1h", "1d")
            
        Returns:
            Number of seconds in the timeframe
        """
        value = int(timeframe[:-1])
        unit = timeframe[-1]
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        else:
            return 900  # Default to 15 minutes
