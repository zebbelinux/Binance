"""
BTCTURK API Entegrasyonu
REST API ve WebSocket bağlantıları
"""

import requests
import websocket
import json
import hmac
import hashlib
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

from config.config import config
from api.websocket_client import WebSocketClient, WebSocketMessage

class BTCTurkAPI:
    """BTCTURK API sınıfı"""
    
    def __init__(self):
        self.api_key = config.api.api_key
        self.secret_key = config.api.secret_key
        self.base_url = config.api.base_url
        self.websocket_url = config.api.websocket_url
        self.test_mode = config.api.test_mode
        
        self.session = requests.Session()
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        
        # Yeni WebSocket istemcisi
        self.websocket_client = WebSocketClient({
            'url': self.websocket_url,
            'symbols': ['BTCTRY', 'ETHTRY', 'ADATRY', 'AVAXTRY'],
            'reconnect_interval': 5,
            'max_reconnect_attempts': 10
        })
        
        # WebSocket callback'leri
        self.price_callbacks = []
        self.orderbook_callbacks = []
        self.trade_callbacks = []
        
        # WebSocket mesaj handler'ları
        self.websocket_client.add_message_callback(self._handle_websocket_message)
        self.websocket_client.add_error_callback(self._handle_websocket_error)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms
        
        self.logger = logging.getLogger(__name__)
    
    def _get_signature(self, message: str) -> str:
        """API imzası oluştur"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: dict = None, data: dict = None) -> dict:
        """API isteği gönder"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        
        url = f"{self.base_url}{endpoint}"
        # Public endpoint'ler icin auth header gonderme
        public_endpoints = {"/ticker", "/orderbook", "/trades", "/klines"}
        headers = { 'Content-Type': 'application/json' }
        
        if endpoint not in public_endpoints:
            stamp = int(time.time() * 1000)
            headers.update({
                'X-PCK': self.api_key,
                'X-Stamp': str(stamp)
            })
            message = f"{self.api_key}{stamp}"
            if params:
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                message += query_string
            elif data:
                message += json.dumps(data)
            # Private endpointlerde her durumda imza ekle
            headers['X-Signature'] = self._get_signature(message)
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params)

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as he:
                # /klines endpoint'i bazı ortamlarda yok -> 404 spam loglama
                if endpoint == "/klines" and response is not None and response.status_code == 404:
                    return {"error": f"404 Not Found: {url}"}
                # diger hatalar normal loglansın
                raise

            return response.json()

        except requests.exceptions.RequestException as e:
            # Yukarıda 404 /klines filtrelendi; kalanları logla
            self.logger.error(f"API isteği başarısız: {e}")
            return {"error": str(e)}
    
    # Market Data Methods
    def get_ticker(self, symbol: str = None) -> dict:
        """Ticker bilgilerini al"""
        endpoint = "/ticker"
        params = {"pairSymbol": symbol} if symbol else {}
        return self._make_request("GET", endpoint, params=params)
    
    def get_orderbook(self, symbol: str) -> dict:
        """Emir defterini al"""
        endpoint = "/orderbook"
        params = {"pairSymbol": symbol}
        return self._make_request("GET", endpoint, params=params)
    
    def get_trades(self, symbol: str, limit: int = 50) -> dict:
        """Son işlemleri al"""
        endpoint = "/trades"
        params = {"pairSymbol": symbol, "last": limit}
        return self._make_request("GET", endpoint, params=params)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> dict:
        """Mum verilerini al"""
        endpoint = "/klines"
        params = {
            "pairSymbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._make_request("GET", endpoint, params=params)
    
    def get_try_pairs_excluding_fans(self) -> List[str]:
        """TRY bazlı işlem çiftlerini döndür (fan token'lar hariç)."""
        try:
            resp = self.get_ticker()
            items = resp.get('data', []) if isinstance(resp, dict) else []
            # Bilinen fan token kısaltmaları (gerektikçe genişletilebilir)
            fan_tokens = {
                "GAL", "CITY", "JUV", "PSG", "ATM", "BAR", "NAP",
                "ACM", "INTER", "AFC", "PORTO", "ASR", "TRA"
            }
            pairs = set()
            for it in items:
                sym = it.get('pairSymbol') or it.get('pair') or ""
                if not isinstance(sym, str):
                    continue
                if not sym.endswith("TRY"):
                    continue
                base = sym[:-3]
                if base in fan_tokens:
                    continue
                pairs.add(sym)
            # En az bir fallback kuralım
            result = sorted(pairs)
            return result if result else ["BTCTRY", "ETHTRY"]
        except Exception as e:
            logging.getLogger(__name__).warning(f"TRY pariteleri alınamadı, fallback kullanılacak: {e}")
            return ["BTCTRY", "ETHTRY"]
    
    # Account Methods
    def get_balance(self) -> dict:
        """Hesap bakiyesini al"""
        endpoint = "/users/balances"
        return self._make_request("GET", endpoint)
    
    def get_open_orders(self, symbol: str = None) -> dict:
        """Açık emirleri al"""
        endpoint = "/openOrders"
        params = {"pairSymbol": symbol} if symbol else {}
        return self._make_request("GET", endpoint, params=params)
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> dict:
        """Emir geçmişini al"""
        endpoint = "/allOrders"
        params = {
            "pairSymbol": symbol,
            "limit": limit
        }
        return self._make_request("GET", endpoint, params=params)
    
    # Trading Methods
    def place_order(self, symbol: str, side: str, order_type: str, 
                   quantity: float, price: float = None, stop_price: float = None) -> dict:
        """Emir ver"""
        endpoint = "/order"
        data = {
            "pairSymbol": symbol,
            "orderType": order_type,
            "orderMethod": side,
            "quantity": str(quantity)
        }
        
        if price:
            data["price"] = str(price)
        if stop_price:
            data["stopPrice"] = str(stop_price)
        
        return self._make_request("POST", endpoint, data=data)
    
    def cancel_order(self, order_id: str) -> dict:
        """Emri iptal et"""
        endpoint = "/order"
        params = {"id": order_id}
        return self._make_request("DELETE", endpoint, params=params)
    
    def cancel_all_orders(self, symbol: str = None) -> dict:
        """Tüm emirleri iptal et"""
        endpoint = "/openOrders"
        params = {"pairSymbol": symbol} if symbol else {}
        return self._make_request("DELETE", endpoint, params=params)
    
    # WebSocket Methods
    def connect_websocket(self):
        """WebSocket bağlantısını başlat"""
        if self.is_connected:
            return
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_websocket_message(data)
            except Exception as e:
                self.logger.error(f"WebSocket mesaj işleme hatası: {e}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket hatası: {error}")
            self.is_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket bağlantısı kapandı")
            self.is_connected = False
        
        def on_open(ws):
            self.logger.info("WebSocket bağlantısı açıldı")
            self.is_connected = True
            # Ticker ve orderbook verilerini subscribe et
            self.subscribe_ticker()
            self.subscribe_orderbook()
        
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def disconnect_websocket(self):
        """WebSocket bağlantısını kapat"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
    
    def subscribe_ticker(self, symbols: List[str] = None):
        """Ticker verilerini subscribe et"""
        if not self.is_connected:
            return
        
        message = {
            "type": "ticker",
            "symbols": symbols or ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        }
        
        self.ws.send(json.dumps(message))
    
    def subscribe_orderbook(self, symbols: List[str] = None):
        """Orderbook verilerini subscribe et"""
        if not self.is_connected:
            return
        
        message = {
            "type": "orderbook",
            "symbols": symbols or ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        }
        
        self.ws.send(json.dumps(message))
    
    def subscribe_trades(self, symbols: List[str] = None):
        """Trade verilerini subscribe et"""
        if not self.is_connected:
            return
        
        message = {
            "type": "trades",
            "symbols": symbols or ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        }
        
        self.ws.send(json.dumps(message))
    
    def _handle_websocket_message(self, data: dict):
        """WebSocket mesajlarını işle"""
        message_type = data.get("type")
        
        if message_type == "ticker":
            for callback in self.price_callbacks:
                callback(data)
        elif message_type == "orderbook":
            for callback in self.orderbook_callbacks:
                callback(data)
        elif message_type == "trades":
            for callback in self.trade_callbacks:
                callback(data)
    
    def add_price_callback(self, callback: Callable):
        """Fiyat güncelleme callback'i ekle"""
        self.price_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable):
        """Orderbook güncelleme callback'i ekle"""
        self.orderbook_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Trade güncelleme callback'i ekle"""
        self.trade_callbacks.append(callback)
    
    def remove_callbacks(self):
        """Tüm callback'leri temizle"""
        self.price_callbacks.clear()
        self.orderbook_callbacks.clear()
        self.trade_callbacks.clear()
    
    def _handle_websocket_message(self, message: WebSocketMessage):
        """WebSocket mesajını işle"""
        try:
            if message.type == 'ticker':
                self._handle_ticker_message(message)
            elif message.type == 'orderbook':
                self._handle_orderbook_message(message)
            elif message.type == 'trade':
                self._handle_trade_message(message)
                
        except Exception as e:
            self.logger.error(f"WebSocket mesaj işleme hatası: {e}")
    
    def _handle_ticker_message(self, message: WebSocketMessage):
        """Ticker mesajını işle"""
        try:
            data = message.data
            ticker_data = {
                'symbol': message.symbol,
                'price': data.get('price', 0),
                'volume': data.get('volume', 0),
                'change': data.get('change', 0),
                'timestamp': message.timestamp
            }
            
            # Callback'leri çağır
            for callback in self.price_callbacks:
                try:
                    callback(ticker_data)
                except Exception as e:
                    self.logger.error(f"Ticker callback hatası: {e}")
                    
        except Exception as e:
            self.logger.error(f"Ticker mesaj işleme hatası: {e}")
    
    def _handle_orderbook_message(self, message: WebSocketMessage):
        """Orderbook mesajını işle"""
        try:
            data = message.data
            orderbook_data = {
                'symbol': message.symbol,
                'bids': data.get('bids', []),
                'asks': data.get('asks', []),
                'timestamp': message.timestamp
            }
            
            # Callback'leri çağır
            for callback in self.orderbook_callbacks:
                try:
                    callback(orderbook_data)
                except Exception as e:
                    self.logger.error(f"Orderbook callback hatası: {e}")
                    
        except Exception as e:
            self.logger.error(f"Orderbook mesaj işleme hatası: {e}")
    
    def _handle_trade_message(self, message: WebSocketMessage):
        """Trade mesajını işle"""
        try:
            data = message.data
            trade_data = {
                'symbol': message.symbol,
                'price': data.get('price', 0),
                'quantity': data.get('quantity', 0),
                'side': data.get('side', 'unknown'),
                'timestamp': message.timestamp
            }
            
            # Callback'leri çağır
            for callback in self.trade_callbacks:
                try:
                    callback(trade_data)
                except Exception as e:
                    self.logger.error(f"Trade callback hatası: {e}")
                    
        except Exception as e:
            self.logger.error(f"Trade mesaj işleme hatası: {e}")
    
    def _handle_websocket_error(self, error: Exception):
        """WebSocket hata işle"""
        self.logger.error(f"WebSocket hatası: {error}")
        self.is_connected = False
    
    def start_websocket(self):
        """WebSocket bağlantısını başlat"""
        try:
            self.websocket_client.start()
            self.is_connected = True
            self.logger.info("WebSocket bağlantısı başlatıldı")
        except Exception as e:
            self.logger.error(f"WebSocket başlatma hatası: {e}")
            raise
    
    def stop_websocket(self):
        """WebSocket bağlantısını durdur"""
        try:
            self.websocket_client.stop()
            self.is_connected = False
            self.logger.info("WebSocket bağlantısı durduruldu")
        except Exception as e:
            self.logger.error(f"WebSocket durdurma hatası: {e}")
    
    def get_websocket_status(self):
        """WebSocket durumunu al"""
        return self.websocket_client.get_status()

# Global API instance
api = BTCTurkAPI()
