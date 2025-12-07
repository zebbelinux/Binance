"""
WebSocket İstemcisi
Gerçek zamanlı veri akışı için WebSocket bağlantısı
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import queue
import ssl
import signal
import sys

@dataclass
class WebSocketMessage:
    """WebSocket mesajı"""
    type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime

class WebSocketStatus(Enum):
    """WebSocket durumu"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class WebSocketClient:
    """WebSocket istemcisi"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # WebSocket ayarları
        self.url = self.config.get('url', 'wss://ws-feed.btcturk.com')
        self.symbols = self.config.get('symbols', ['BTCTRY', 'ETHTRY'])
        self.reconnect_interval = self.config.get('reconnect_interval', 5)
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 10)
        self.ping_interval = self.config.get('ping_interval', 30)
        
        # Durum
        self.status = WebSocketStatus.DISCONNECTED
        self.websocket = None
        self.loop = None
        self.thread = None
        self.is_running = False
        self.reconnect_count = 0
        
        # Mesaj kuyruğu
        self.message_queue = queue.Queue()
        
        # Callback'ler
        self.message_callbacks = []
        self.error_callbacks = []
        self.status_callbacks = []
        
        # SSL context
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        self.logger.info("WebSocket istemcisi oluşturuldu")
    
    def add_message_callback(self, callback: Callable[[WebSocketMessage], None]):
        """Mesaj callback'i ekle"""
        self.message_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Hata callback'i ekle"""
        self.error_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[WebSocketStatus], None]):
        """Durum callback'i ekle"""
        self.status_callbacks.append(callback)
    
    def start(self):
        """WebSocket bağlantısını başlat"""
        if self.is_running:
            self.logger.warning("WebSocket zaten çalışıyor")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.thread.start()
        
        self.logger.info("WebSocket bağlantısı başlatıldı")
    
    def stop(self):
        """WebSocket bağlantısını durdur"""
        self.is_running = False
        
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self.logger.info("WebSocket bağlantısı durduruldu")
    
    def _run_websocket(self):
        """WebSocket döngüsünü çalıştır"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._websocket_loop())
        except Exception as e:
            self.logger.error(f"WebSocket döngü hatası: {e}")
            self._notify_error_callbacks(e)
        finally:
            if self.loop:
                self.loop.close()
    
    async def _websocket_loop(self):
        """Ana WebSocket döngüsü"""
        while self.is_running:
            try:
                self._update_status(WebSocketStatus.CONNECTING)
                
                async with websockets.connect(
                    self.url,
                    ssl=self.ssl_context,
                    ping_interval=self.ping_interval,
                    ping_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self._update_status(WebSocketStatus.CONNECTED)
                    self.reconnect_count = 0
                    
                    # Subscribe mesajları gönder
                    await self._subscribe_to_symbols()
                    
                    # Mesaj dinleme döngüsü
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket bağlantısı kapandı")
                self._update_status(WebSocketStatus.DISCONNECTED)
                
            except Exception as e:
                self.logger.error(f"WebSocket hatası: {e}")
                self._notify_error_callbacks(e)
                self._update_status(WebSocketStatus.ERROR)
            
            # Yeniden bağlanma
            if self.is_running and self.reconnect_count < self.max_reconnect_attempts:
                self._update_status(WebSocketStatus.RECONNECTING)
                self.reconnect_count += 1
                self.logger.info(f"Yeniden bağlanma denemesi {self.reconnect_count}/{self.max_reconnect_attempts}")
                await asyncio.sleep(self.reconnect_interval)
            else:
                self.logger.error("Maksimum yeniden bağlanma denemesi aşıldı")
                break
    
    async def _subscribe_to_symbols(self):
        """Sembollere abone ol"""
        try:
            for symbol in self.symbols:
                subscribe_message = {
                    "type": "subscribe",
                    "channel": "ticker",
                    "symbol": symbol
                }
                await self.websocket.send(json.dumps(subscribe_message))
                
                # Order book için de abone ol
                orderbook_message = {
                    "type": "subscribe",
                    "channel": "orderbook",
                    "symbol": symbol
                }
                await self.websocket.send(json.dumps(orderbook_message))
                
                self.logger.info(f"Abone olundu: {symbol}")
                
        except Exception as e:
            self.logger.error(f"Abone olma hatası: {e}")
            raise
    
    async def _handle_message(self, message: str):
        """Gelen mesajı işle"""
        try:
            data = json.loads(message)
            
            # Mesaj tipini belirle
            message_type = data.get('type', 'unknown')
            symbol = data.get('symbol', 'unknown')
            
            # WebSocket mesajı oluştur
            ws_message = WebSocketMessage(
                type=message_type,
                symbol=symbol,
                data=data,
                timestamp=datetime.now()
            )
            
            # Callback'leri çağır
            self._notify_message_callbacks(ws_message)
            
            # Mesaj kuyruğuna ekle
            try:
                self.message_queue.put_nowait(ws_message)
            except queue.Full:
                self.logger.warning("Mesaj kuyruğu dolu, eski mesaj atılıyor")
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait(ws_message)
                except queue.Empty:
                    pass
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse hatası: {e}")
        except Exception as e:
            self.logger.error(f"Mesaj işleme hatası: {e}")
    
    def _update_status(self, status: WebSocketStatus):
        """Durumu güncelle"""
        self.status = status
        self._notify_status_callbacks(status)
        
        if status == WebSocketStatus.CONNECTED:
            self.logger.info("WebSocket bağlantısı kuruldu")
        elif status == WebSocketStatus.DISCONNECTED:
            self.logger.info("WebSocket bağlantısı kesildi")
        elif status == WebSocketStatus.ERROR:
            self.logger.error("WebSocket hatası")
    
    def _notify_message_callbacks(self, message: WebSocketMessage):
        """Mesaj callback'lerini çağır"""
        for callback in self.message_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Mesaj callback hatası: {e}")
    
    def _notify_error_callbacks(self, error: Exception):
        """Hata callback'lerini çağır"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Hata callback hatası: {e}")
    
    def _notify_status_callbacks(self, status: WebSocketStatus):
        """Durum callback'lerini çağır"""
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                self.logger.error(f"Durum callback hatası: {e}")
    
    def get_message(self, timeout: float = 1.0) -> Optional[WebSocketMessage]:
        """Mesaj kuyruğundan mesaj al"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status(self) -> WebSocketStatus:
        """Mevcut durumu al"""
        return self.status
    
    def is_connected(self) -> bool:
        """Bağlantı durumunu kontrol et"""
        return self.status == WebSocketStatus.CONNECTED
    
    def send_message(self, message: Dict[str, Any]):
        """Mesaj gönder"""
        if self.websocket and self.is_connected():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            except Exception as e:
                self.logger.error(f"Mesaj gönderme hatası: {e}")
        else:
            self.logger.warning("WebSocket bağlantısı yok, mesaj gönderilemedi")
    
    def get_queue_size(self) -> int:
        """Mesaj kuyruğu boyutunu al"""
        return self.message_queue.qsize()
    
    def clear_queue(self):
        """Mesaj kuyruğunu temizle"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

# Global WebSocket istemcisi
websocket_client = WebSocketClient()

# Signal handler'ları
def signal_handler(signum, frame):
    """Signal handler"""
    print("\nWebSocket kapatılıyor...")
    websocket_client.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
