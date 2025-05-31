import requests
import json
import time
import hmac
import base64
import hashlib
from datetime import datetime
import urllib.parse
from typing import Dict, List, Optional, Any, Union
import websockets
import asyncio
import logging

class ExchangeConnector:
    """
    OKX Exchange Connector for handling API interactions.
    Provides methods to connect to the OKX exchange API,
    retrieve market data, place orders, and manage account balances.
    """

    BASE_URL = "https://www.okx.com"
    TEST_URL = "https://www.okx-sandbox.com"
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "", test_mode: bool = True):
        """
        Initialize the OKX exchange connector.
        
        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
            test_mode: If True, use the sandbox environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.test_mode = test_mode
        self.base_url = self.TEST_URL if test_mode else self.BASE_URL
        self.ws_url = "wss://wspap.okx.com:8443/ws/v5/public" if not test_mode else "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
        self.ws_private_url = "wss://wspap.okx.com:8443/ws/v5/private" if not test_mode else "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"
        self.logger = logging.getLogger(__name__)
        self.ws_connection = None
        self.ws_private_connection = None
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Generate the signature required for API authentication.
        
        Args:
            timestamp: Current timestamp
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body for POST requests
        
        Returns:
            Base64 encoded signature
        """
        if not self.api_secret:
            return ""
            
        message = timestamp + method + request_path + (body if body else "")
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        d = mac.digest()
        return base64.b64encode(d).decode()
    
    def _get_header(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """
        Create the request headers with authentication.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body for POST requests
            
        Returns:
            Dictionary of request headers
        """
        timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        header = {
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
        }
        
        if self.test_mode:
            header['x-simulated-trading'] = '1'
            
        return header
    
    def _request(self, method: str, request_path: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make an API request to OKX.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            params: URL parameters for GET requests
            data: Body data for POST requests
            
        Returns:
            JSON response from the API
        """
        url = self.base_url + request_path
        
        # Add URL parameters for GET requests
        if method == 'GET' and params:
            query_string = '&'.join([f"{key}={urllib.parse.quote(str(value))}" for key, value in params.items()])
            request_path += '?' + query_string
            url = self.base_url + request_path
        
        body = json.dumps(data) if data else ""
        headers = self._get_header(method, request_path, body)
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, data=body)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            return {"code": "error", "msg": str(e), "data": []}
    
    def connect(self) -> bool:
        """
        Test the API connection.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            response = self._request('GET', '/api/v5/public/time')
            if response and response.get('code') == '0':
                self.logger.info("Successfully connected to OKX API")
                return True
            else:
                self.logger.error(f"Failed to connect to OKX API: {response}")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """
        Close any open connections.
        """
        # Close WebSocket connections if they exist
        if self.ws_connection:
            asyncio.create_task(self.ws_connection.close())
            self.ws_connection = None
        
        if self.ws_private_connection:
            asyncio.create_task(self.ws_private_connection.close())
            self.ws_private_connection = None
            
        self.logger.info("Disconnected from OKX API")
    
    async def _connect_websocket(self, private: bool = False) -> websockets.WebSocketClientProtocol:
        """
        Connect to the WebSocket API.
        
        Args:
            private: If True, connect to the private API
            
        Returns:
            WebSocket connection
        """
        url = self.ws_private_url if private else self.ws_url
        connection = await websockets.connect(url)
        self.logger.info(f"Connected to WebSocket API: {url}")
        return connection
    
    async def _authenticate_ws(self, ws: websockets.WebSocketClientProtocol) -> bool:
        """
        Authenticate with the private WebSocket API.
        
        Args:
            ws: WebSocket connection
            
        Returns:
            True if authenticated successfully, False otherwise
        """
        timestamp = str(int(time.time()))
        signature = self._generate_signature(timestamp, "GET", "/users/self/verify", "")
        
        auth_message = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        }
        
        await ws.send(json.dumps(auth_message))
        response = json.loads(await ws.recv())
        
        if response.get('code') == '0':
            self.logger.info("WebSocket authentication successful")
            return True
        else:
            self.logger.error(f"WebSocket authentication failed: {response}")
            return False
    
    async def subscribe_to_market_data(self, symbols: List[str], channel: str = "tickers") -> None:
        """
        Subscribe to market data via WebSocket.
        
        Args:
            symbols: List of trading pairs (e.g., ["BTC-USDT"])
            channel: Data channel to subscribe to (tickers, candles, trades, etc.)
        """
        if not self.ws_connection:
            self.ws_connection = await self._connect_websocket()
        
        args = [{"channel": channel, "instId": symbol} for symbol in symbols]
        subscribe_message = {
            "op": "subscribe",
            "args": args
        }
        
        await self.ws_connection.send(json.dumps(subscribe_message))
        self.logger.info(f"Subscribed to {channel} for {symbols}")
    
    async def subscribe_to_account_updates(self) -> None:
        """
        Subscribe to account balance and position updates.
        """
        if not self.ws_private_connection:
            self.ws_private_connection = await self._connect_websocket(private=True)
            authenticated = await self._authenticate_ws(self.ws_private_connection)
            if not authenticated:
                self.logger.error("Failed to subscribe to account updates: Authentication failed")
                return
        
        subscribe_message = {
            "op": "subscribe",
            "args": [
                {"channel": "account"},
                {"channel": "positions"}
            ]
        }
        
        await self.ws_private_connection.send(json.dumps(subscribe_message))
        self.logger.info("Subscribed to account and position updates")
    
    async def listen_for_market_data(self, callback) -> None:
        """
        Listen for market data updates from WebSocket.
        
        Args:
            callback: Function to call with received data
        """
        if not self.ws_connection:
            self.logger.error("WebSocket not connected. Call subscribe_to_market_data first.")
            return
        
        try:
            while True:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Market data WebSocket connection closed")
            self.ws_connection = None
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            Market data dictionary
        """
        endpoint = f"/api/v5/market/ticker?instId={symbol}"
        response = self._request('GET', endpoint)
        
        if response and response.get('code') == '0' and response.get('data'):
            return response['data'][0]
        else:
            self.logger.error(f"Failed to get market data for {symbol}: {response}")
            return {}
    
    def get_historical_candles(self, symbol: str, timeframe: str = "15m", limit: int = 100) -> List[Dict]:
        """
        Get historical candlestick data.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            timeframe: Candlestick interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of candles to retrieve (max 100)
            
        Returns:
            List of candle data
        """
        # Convert timeframe to OKX format
        bar_mapping = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D"
        }
        
        if timeframe not in bar_mapping:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return []
        
        bar = bar_mapping[timeframe]
        endpoint = f"/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
        response = self._request('GET', endpoint)
        
        if response and response.get('code') == '0' and response.get('data'):
            # OKX returns data in reverse chronological order, so reverse it
            candles = response['data']
            
            # Convert to more friendly format
            formatted_candles = []
            for candle in candles:
                formatted_candles.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return formatted_candles
        else:
            self.logger.error(f"Failed to get historical candles for {symbol}: {response}")
            return []
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            Latest price as a float
        """
        market_data = self.get_market_data(symbol)
        if market_data and 'last' in market_data:
            return float(market_data['last'])
        return 0.0
    
    def get_account_balance(self) -> Dict:
        """
        Get account balance information.
        
        Returns:
            Account balance data
        """
        endpoint = "/api/v5/account/balance"
        response = self._request('GET', endpoint)
        
        if response and response.get('code') == '0' and response.get('data'):
            return response['data'][0]
        else:
            self.logger.error(f"Failed to get account balance: {response}")
            return {}
    
    def place_order(self, order: Dict) -> Dict:
        """
        Place a new order.
        
        Args:
            order: Order details dictionary with the following keys:
                - instId: Instrument ID (e.g., "BTC-USDT")
                - tdMode: Trade mode (cash, cross, isolated)
                - side: Buy or sell
                - ordType: Order type (market, limit, post_only, etc.)
                - sz: Size (amount)
                - px: Price (for limit orders)
                
        Returns:
            Order response data
        """
        endpoint = "/api/v5/trade/order"
        response = self._request('POST', endpoint, data=order)
        
        if response and response.get('code') == '0' and response.get('data'):
            return response['data'][0]
        else:
            self.logger.error(f"Failed to place order: {response}")
            return {}
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        endpoint = "/api/v5/trade/cancel-order"
        data = {
            "instId": symbol,
            "ordId": order_id
        }
        
        response = self._request('POST', endpoint, data=data)
        
        if response and response.get('code') == '0':
            return True
        else:
            self.logger.error(f"Failed to cancel order: {response}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get list of open orders.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT") or None for all
            
        Returns:
            List of open order data
        """
        endpoint = "/api/v5/trade/orders-pending"
        params = {}
        if symbol:
            params["instId"] = symbol
        
        response = self._request('GET', endpoint, params=params)
        
        if response and response.get('code') == '0':
            return response.get('data', [])
        else:
            self.logger.error(f"Failed to get open orders: {response}")
            return []
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """
        Get status of a specific order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            Order status data
        """
        endpoint = "/api/v5/trade/order"
        params = {
            "instId": symbol,
            "ordId": order_id
        }
        
        response = self._request('GET', endpoint, params=params)
        
        if response and response.get('code') == '0' and response.get('data'):
            return response['data'][0]
        else:
            self.logger.error(f"Failed to get order status: {response}")
            return {}
