"""
Tick-by-Tick Veri Kaydı Modülü
Gerçek zamanlı tick verilerini kaydetme ve analiz etme sistemi
"""

import sqlite3
import threading
import time
import logging
import json
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import pandas as pd
import numpy as np
from pathlib import Path
import queue
import pickle
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TickData:
    """Tick verisi"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str = None
    order_id: str = None
    maker_order_id: str = None
    taker_order_id: str = None

@dataclass
class OrderBookSnapshot:
    """Order book anlık görüntüsü"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    last_price: float = 0.0
    last_volume: float = 0.0

@dataclass
class MarketDataSnapshot:
    """Piyasa verisi anlık görüntüsü"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    trade_count: int = 0

@dataclass
class TickRecordingConfig:
    """Tick kayıt konfigürasyonu"""
    enabled: bool = True
    symbols: List[str] = None
    data_types: List[str] = None  # ['tick', 'orderbook', 'ohlcv']
    compression_enabled: bool = True
    batch_size: int = 1000
    flush_interval: int = 60  # saniye
    max_file_size_mb: int = 100
    retention_days: int = 30
    storage_path: str = "tick_data"
    database_path: str = "tick_data.db"
    enable_real_time_analysis: bool = True
    analysis_window_minutes: int = 5

class DataType(Enum):
    """Veri tipleri"""
    TICK = "tick"
    ORDERBOOK = "orderbook"
    OHLCV = "ohlcv"
    TRADE = "trade"

class TickDataRecorder:
    """Tick veri kaydedici"""
    
    def __init__(self, config: TickRecordingConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or TickRecordingConfig()
        
        # Veri kuyrukları
        self.tick_queue = queue.Queue(maxsize=10000)
        self.orderbook_queue = queue.Queue(maxsize=1000)
        self.ohlcv_queue = queue.Queue(maxsize=1000)
        
        # Veri işleme thread'leri
        self.processing_threads = []
        self.is_recording = False
        
        # Veri analizi
        self.real_time_analysis = {}
        self.analysis_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Veritabanı bağlantısı
        self.db_connection = None
        self._initialize_database()
        
        # Dosya yönetimi
        self.current_files = {}
        self.file_handles = {}
        
        # Varsayılan semboller
        if not self.config.symbols:
            self.config.symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        
        # Varsayılan veri tipleri
        if not self.config.data_types:
            self.config.data_types = ["tick", "orderbook", "ohlcv"]
        
        # Storage dizinini oluştur
        Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Tick veri kaydedici başlatıldı")
    
    def _initialize_database(self):
        """Veritabanını başlat"""
        try:
            self.db_connection = sqlite3.connect(
                self.config.database_path,
                check_same_thread=False
            )
            cursor = self.db_connection.cursor()
            
            # Tick verileri tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    side TEXT NOT NULL,
                    trade_id TEXT,
                    order_id TEXT,
                    maker_order_id TEXT,
                    taker_order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Order book verileri tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orderbook_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    last_price REAL,
                    last_volume REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OHLCV verileri tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    vwap REAL,
                    trade_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # İndeksler oluştur
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tick_symbol_timestamp ON tick_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp ON orderbook_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data(symbol, timestamp)')
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Veritabanı başlatma hatası: {e}")
    
    def start_recording(self):
        """Kayıt işlemini başlat"""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        # Veri işleme thread'lerini başlat
        if "tick" in self.config.data_types:
            tick_thread = threading.Thread(target=self._process_tick_data, daemon=True)
            tick_thread.start()
            self.processing_threads.append(tick_thread)
        
        if "orderbook" in self.config.data_types:
            orderbook_thread = threading.Thread(target=self._process_orderbook_data, daemon=True)
            orderbook_thread.start()
            self.processing_threads.append(orderbook_thread)
        
        if "ohlcv" in self.config.data_types:
            ohlcv_thread = threading.Thread(target=self._process_ohlcv_data, daemon=True)
            ohlcv_thread.start()
            self.processing_threads.append(ohlcv_thread)
        
        # Veri temizleme thread'i
        cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        cleanup_thread.start()
        self.processing_threads.append(cleanup_thread)
        
        self.logger.info("Tick veri kaydı başlatıldı")
    
    def stop_recording(self):
        """Kayıt işlemini durdur"""
        self.is_recording = False
        
        # Tüm kuyrukları temizle
        while not self.tick_queue.empty():
            try:
                self.tick_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.orderbook_queue.empty():
            try:
                self.orderbook_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.ohlcv_queue.empty():
            try:
                self.ohlcv_queue.get_nowait()
            except queue.Empty:
                break
        
        # Dosya handle'larını kapat
        for handle in self.file_handles.values():
            try:
                handle.close()
            except:
                pass
        
        self.file_handles.clear()
        
        # Veritabanı bağlantısını kapat
        if self.db_connection:
            self.db_connection.close()
        
        self.logger.info("Tick veri kaydı durduruldu")
    
    def record_tick(self, tick_data: TickData):
        """Tick verisi kaydet"""
        try:
            if not self.is_recording or tick_data.symbol not in self.config.symbols:
                return
            
            # Tick verisini kuyruğa ekle
            self.tick_queue.put(tick_data, timeout=1)
            
            # Gerçek zamanlı analiz
            if self.config.enable_real_time_analysis:
                self._update_real_time_analysis(tick_data)
            
        except queue.Full:
            self.logger.warning("Tick kuyruğu dolu, veri kaybedildi")
        except Exception as e:
            self.logger.error(f"Tick kaydetme hatası: {e}")
    
    def record_orderbook(self, orderbook_data: OrderBookSnapshot):
        """Order book verisi kaydet"""
        try:
            if not self.is_recording or orderbook_data.symbol not in self.config.symbols:
                return
            
            # Order book verisini kuyruğa ekle
            self.orderbook_queue.put(orderbook_data, timeout=1)
            
        except queue.Full:
            self.logger.warning("Order book kuyruğu dolu, veri kaybedildi")
        except Exception as e:
            self.logger.error(f"Order book kaydetme hatası: {e}")
    
    def record_ohlcv(self, ohlcv_data: MarketDataSnapshot):
        """OHLCV verisi kaydet"""
        try:
            if not self.is_recording or ohlcv_data.symbol not in self.config.symbols:
                return
            
            # OHLCV verisini kuyruğa ekle
            self.ohlcv_queue.put(ohlcv_data, timeout=1)
            
        except queue.Full:
            self.logger.warning("OHLCV kuyruğu dolu, veri kaybedildi")
        except Exception as e:
            self.logger.error(f"OHLCV kaydetme hatası: {e}")
    
    def _process_tick_data(self):
        """Tick verilerini işle"""
        batch = []
        
        while self.is_recording:
            try:
                # Tick verisini kuyruktan al
                tick_data = self.tick_queue.get(timeout=1)
                batch.append(tick_data)
                
                # Batch boyutuna ulaştıysa veya timeout olduysa kaydet
                if len(batch) >= self.config.batch_size:
                    self._save_tick_batch(batch)
                    batch = []
                
            except queue.Empty:
                # Timeout oldu, mevcut batch'i kaydet
                if batch:
                    self._save_tick_batch(batch)
                    batch = []
            except Exception as e:
                self.logger.error(f"Tick veri işleme hatası: {e}")
    
    def _process_orderbook_data(self):
        """Order book verilerini işle"""
        batch = []
        
        while self.is_recording:
            try:
                # Order book verisini kuyruktan al
                orderbook_data = self.orderbook_queue.get(timeout=1)
                batch.append(orderbook_data)
                
                # Batch boyutuna ulaştıysa veya timeout olduysa kaydet
                if len(batch) >= self.config.batch_size:
                    self._save_orderbook_batch(batch)
                    batch = []
                
            except queue.Empty:
                # Timeout oldu, mevcut batch'i kaydet
                if batch:
                    self._save_orderbook_batch(batch)
                    batch = []
            except Exception as e:
                self.logger.error(f"Order book veri işleme hatası: {e}")
    
    def _process_ohlcv_data(self):
        """OHLCV verilerini işle"""
        batch = []
        
        while self.is_recording:
            try:
                # OHLCV verisini kuyruktan al
                ohlcv_data = self.ohlcv_queue.get(timeout=1)
                batch.append(ohlcv_data)
                
                # Batch boyutuna ulaştıysa veya timeout olduysa kaydet
                if len(batch) >= self.config.batch_size:
                    self._save_ohlcv_batch(batch)
                    batch = []
                
            except queue.Empty:
                # Timeout oldu, mevcut batch'i kaydet
                if batch:
                    self._save_ohlcv_batch(batch)
                    batch = []
            except Exception as e:
                self.logger.error(f"OHLCV veri işleme hatası: {e}")
    
    def _save_tick_batch(self, batch: List[TickData]):
        """Tick batch'ini kaydet"""
        try:
            if not batch:
                return
            
            # Veritabanına kaydet
            cursor = self.db_connection.cursor()
            
            for tick in batch:
                cursor.execute('''
                    INSERT INTO tick_data 
                    (symbol, timestamp, price, volume, side, trade_id, order_id, maker_order_id, taker_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tick.symbol,
                    tick.timestamp.isoformat(),
                    tick.price,
                    tick.volume,
                    tick.side,
                    tick.trade_id,
                    tick.order_id,
                    tick.maker_order_id,
                    tick.taker_order_id
                ))
            
            self.db_connection.commit()
            
            # Dosyaya kaydet
            self._save_tick_batch_to_file(batch)
            
            self.logger.debug(f"{len(batch)} tick verisi kaydedildi")
            
        except Exception as e:
            self.logger.error(f"Tick batch kaydetme hatası: {e}")
    
    def _save_orderbook_batch(self, batch: List[OrderBookSnapshot]):
        """Order book batch'ini kaydet"""
        try:
            if not batch:
                return
            
            # Veritabanına kaydet
            cursor = self.db_connection.cursor()
            
            for orderbook in batch:
                cursor.execute('''
                    INSERT INTO orderbook_data 
                    (symbol, timestamp, bids, asks, last_price, last_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    orderbook.symbol,
                    orderbook.timestamp.isoformat(),
                    json.dumps(orderbook.bids),
                    json.dumps(orderbook.asks),
                    orderbook.last_price,
                    orderbook.last_volume
                ))
            
            self.db_connection.commit()
            
            # Dosyaya kaydet
            self._save_orderbook_batch_to_file(batch)
            
            self.logger.debug(f"{len(batch)} order book verisi kaydedildi")
            
        except Exception as e:
            self.logger.error(f"Order book batch kaydetme hatası: {e}")
    
    def _save_ohlcv_batch(self, batch: List[MarketDataSnapshot]):
        """OHLCV batch'ini kaydet"""
        try:
            if not batch:
                return
            
            # Veritabanına kaydet
            cursor = self.db_connection.cursor()
            
            for ohlcv in batch:
                cursor.execute('''
                    INSERT INTO ohlcv_data 
                    (symbol, timestamp, open, high, low, close, volume, vwap, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ohlcv.symbol,
                    ohlcv.timestamp.isoformat(),
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume,
                    ohlcv.vwap,
                    ohlcv.trade_count
                ))
            
            self.db_connection.commit()
            
            # Dosyaya kaydet
            self._save_ohlcv_batch_to_file(batch)
            
            self.logger.debug(f"{len(batch)} OHLCV verisi kaydedildi")
            
        except Exception as e:
            self.logger.error(f"OHLCV batch kaydetme hatası: {e}")
    
    def _save_tick_batch_to_file(self, batch: List[TickData]):
        """Tick batch'ini dosyaya kaydet"""
        try:
            if not batch:
                return
            
            # Sembol bazında dosya oluştur
            symbol = batch[0].symbol
            date_str = batch[0].timestamp.strftime("%Y%m%d")
            
            filename = f"{symbol}_tick_{date_str}.pkl"
            filepath = Path(self.config.storage_path) / filename
            
            # Dosya boyutunu kontrol et
            if filepath.exists() and filepath.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                # Yeni dosya oluştur
                timestamp = int(time.time())
                filename = f"{symbol}_tick_{date_str}_{timestamp}.pkl"
                filepath = Path(self.config.storage_path) / filename
            
            # Veriyi dosyaya kaydet
            if self.config.compression_enabled:
                with gzip.open(f"{filepath}.gz", 'ab') as f:
                    pickle.dump(batch, f)
            else:
                with open(filepath, 'ab') as f:
                    pickle.dump(batch, f)
            
        except Exception as e:
            self.logger.error(f"Tick dosya kaydetme hatası: {e}")
    
    def _save_orderbook_batch_to_file(self, batch: List[OrderBookSnapshot]):
        """Order book batch'ini dosyaya kaydet"""
        try:
            if not batch:
                return
            
            # Sembol bazında dosya oluştur
            symbol = batch[0].symbol
            date_str = batch[0].timestamp.strftime("%Y%m%d")
            
            filename = f"{symbol}_orderbook_{date_str}.pkl"
            filepath = Path(self.config.storage_path) / filename
            
            # Dosya boyutunu kontrol et
            if filepath.exists() and filepath.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                # Yeni dosya oluştur
                timestamp = int(time.time())
                filename = f"{symbol}_orderbook_{date_str}_{timestamp}.pkl"
                filepath = Path(self.config.storage_path) / filename
            
            # Veriyi dosyaya kaydet
            if self.config.compression_enabled:
                with gzip.open(f"{filepath}.gz", 'ab') as f:
                    pickle.dump(batch, f)
            else:
                with open(filepath, 'ab') as f:
                    pickle.dump(batch, f)
            
        except Exception as e:
            self.logger.error(f"Order book dosya kaydetme hatası: {e}")
    
    def _save_ohlcv_batch_to_file(self, batch: List[MarketDataSnapshot]):
        """OHLCV batch'ini dosyaya kaydet"""
        try:
            if not batch:
                return
            
            # Sembol bazında dosya oluştur
            symbol = batch[0].symbol
            date_str = batch[0].timestamp.strftime("%Y%m%d")
            
            filename = f"{symbol}_ohlcv_{date_str}.pkl"
            filepath = Path(self.config.storage_path) / filename
            
            # Dosya boyutunu kontrol et
            if filepath.exists() and filepath.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                # Yeni dosya oluştur
                timestamp = int(time.time())
                filename = f"{symbol}_ohlcv_{date_str}_{timestamp}.pkl"
                filepath = Path(self.config.storage_path) / filename
            
            # Veriyi dosyaya kaydet
            if self.config.compression_enabled:
                with gzip.open(f"{filepath}.gz", 'ab') as f:
                    pickle.dump(batch, f)
            else:
                with open(filepath, 'ab') as f:
                    pickle.dump(batch, f)
            
        except Exception as e:
            self.logger.error(f"OHLCV dosya kaydetme hatası: {e}")
    
    def _update_real_time_analysis(self, tick_data: TickData):
        """Gerçek zamanlı analizi güncelle"""
        try:
            symbol = tick_data.symbol
            
            if symbol not in self.real_time_analysis:
                self.real_time_analysis[symbol] = {
                    'tick_count': 0,
                    'total_volume': 0.0,
                    'total_value': 0.0,
                    'last_price': 0.0,
                    'price_changes': [],
                    'volume_by_side': {'buy': 0.0, 'sell': 0.0},
                    'last_update': datetime.now()
                }
            
            analysis = self.real_time_analysis[symbol]
            
            # Temel istatistikleri güncelle
            analysis['tick_count'] += 1
            analysis['total_volume'] += tick_data.volume
            analysis['total_value'] += tick_data.price * tick_data.volume
            analysis['last_price'] = tick_data.price
            analysis['volume_by_side'][tick_data.side] += tick_data.volume
            analysis['last_update'] = datetime.now()
            
            # Fiyat değişimlerini kaydet
            if analysis['last_price'] > 0:
                price_change = (tick_data.price - analysis['last_price']) / analysis['last_price']
                analysis['price_changes'].append(price_change)
                
                # Son 100 değişimi tut
                if len(analysis['price_changes']) > 100:
                    analysis['price_changes'] = analysis['price_changes'][-100:]
            
            # Analiz callback'lerini çağır
            self._notify_analysis_callbacks(symbol, analysis)
            
        except Exception as e:
            self.logger.error(f"Gerçek zamanlı analiz güncelleme hatası: {e}")
    
    def _notify_analysis_callbacks(self, symbol: str, analysis: Dict[str, Any]):
        """Analiz callback'lerini çağır"""
        for callback in self.analysis_callbacks:
            try:
                callback(symbol, analysis)
            except Exception as e:
                self.logger.error(f"Analiz callback hatası: {e}")
    
    def _cleanup_old_data(self):
        """Eski verileri temizle"""
        while self.is_recording:
            try:
                # Bekle
                time.sleep(3600)  # 1 saat
                
                if not self.is_recording:
                    break
                
                # Eski dosyaları sil
                cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
                
                for file_path in Path(self.config.storage_path).glob("*"):
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_date.timestamp():
                        try:
                            file_path.unlink()
                            self.logger.info(f"Eski dosya silindi: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Dosya silme hatası: {e}")
                
                # Veritabanından eski verileri sil
                self._cleanup_old_database_data(cutoff_date)
                
            except Exception as e:
                self.logger.error(f"Veri temizleme hatası: {e}")
    
    def _cleanup_old_database_data(self, cutoff_date: datetime):
        """Veritabanından eski verileri temizle"""
        try:
            cursor = self.db_connection.cursor()
            
            # Eski tick verilerini sil
            cursor.execute('DELETE FROM tick_data WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            # Eski order book verilerini sil
            cursor.execute('DELETE FROM orderbook_data WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            # Eski OHLCV verilerini sil
            cursor.execute('DELETE FROM ohlcv_data WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            self.db_connection.commit()
            
            self.logger.info("Eski veritabanı verileri temizlendi")
            
        except Exception as e:
            self.logger.error(f"Veritabanı temizleme hatası: {e}")
    
    def get_tick_data(self, 
                     symbol: str, 
                     start_time: datetime, 
                     end_time: datetime,
                     limit: int = 1000) -> List[TickData]:
        """Tick verilerini al"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                SELECT symbol, timestamp, price, volume, side, trade_id, order_id, maker_order_id, taker_order_id
                FROM tick_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, start_time.isoformat(), end_time.isoformat(), limit))
            
            rows = cursor.fetchall()
            
            tick_data = []
            for row in rows:
                tick = TickData(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    price=row[2],
                    volume=row[3],
                    side=row[4],
                    trade_id=row[5],
                    order_id=row[6],
                    maker_order_id=row[7],
                    taker_order_id=row[8]
                )
                tick_data.append(tick)
            
            return tick_data
            
        except Exception as e:
            self.logger.error(f"Tick veri alma hatası: {e}")
            return []
    
    def get_orderbook_data(self, 
                          symbol: str, 
                          start_time: datetime, 
                          end_time: datetime,
                          limit: int = 1000) -> List[OrderBookSnapshot]:
        """Order book verilerini al"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                SELECT symbol, timestamp, bids, asks, last_price, last_volume
                FROM orderbook_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, start_time.isoformat(), end_time.isoformat(), limit))
            
            rows = cursor.fetchall()
            
            orderbook_data = []
            for row in rows:
                orderbook = OrderBookSnapshot(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    bids=json.loads(row[2]),
                    asks=json.loads(row[3]),
                    last_price=row[4],
                    last_volume=row[5]
                )
                orderbook_data.append(orderbook)
            
            return orderbook_data
            
        except Exception as e:
            self.logger.error(f"Order book veri alma hatası: {e}")
            return []
    
    def get_ohlcv_data(self, 
                      symbol: str, 
                      start_time: datetime, 
                      end_time: datetime,
                      limit: int = 1000) -> List[MarketDataSnapshot]:
        """OHLCV verilerini al"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                SELECT symbol, timestamp, open, high, low, close, volume, vwap, trade_count
                FROM ohlcv_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, start_time.isoformat(), end_time.isoformat(), limit))
            
            rows = cursor.fetchall()
            
            ohlcv_data = []
            for row in rows:
                ohlcv = MarketDataSnapshot(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    volume=row[6],
                    vwap=row[7],
                    trade_count=row[8]
                )
                ohlcv_data.append(ohlcv)
            
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"OHLCV veri alma hatası: {e}")
            return []
    
    def get_real_time_analysis(self, symbol: str) -> Dict[str, Any]:
        """Gerçek zamanlı analizi al"""
        try:
            if symbol not in self.real_time_analysis:
                return {}
            
            analysis = self.real_time_analysis[symbol].copy()
            
            # Ek hesaplamalar
            if analysis['total_volume'] > 0:
                analysis['vwap'] = analysis['total_value'] / analysis['total_volume']
            else:
                analysis['vwap'] = 0.0
            
            if analysis['price_changes']:
                analysis['volatility'] = np.std(analysis['price_changes'])
                analysis['avg_price_change'] = np.mean(analysis['price_changes'])
            else:
                analysis['volatility'] = 0.0
                analysis['avg_price_change'] = 0.0
            
            # Volume ratio
            total_volume = analysis['volume_by_side']['buy'] + analysis['volume_by_side']['sell']
            if total_volume > 0:
                analysis['buy_volume_ratio'] = analysis['volume_by_side']['buy'] / total_volume
                analysis['sell_volume_ratio'] = analysis['volume_by_side']['sell'] / total_volume
            else:
                analysis['buy_volume_ratio'] = 0.0
                analysis['sell_volume_ratio'] = 0.0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Gerçek zamanlı analiz alma hatası: {e}")
            return {}
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Veri istatistiklerini al"""
        try:
            cursor = self.db_connection.cursor()
            
            # Tick veri sayısı
            cursor.execute('SELECT COUNT(*) FROM tick_data')
            tick_count = cursor.fetchone()[0]
            
            # Order book veri sayısı
            cursor.execute('SELECT COUNT(*) FROM orderbook_data')
            orderbook_count = cursor.fetchone()[0]
            
            # OHLCV veri sayısı
            cursor.execute('SELECT COUNT(*) FROM ohlcv_data')
            ohlcv_count = cursor.fetchone()[0]
            
            # Sembol bazında istatistikler
            cursor.execute('SELECT symbol, COUNT(*) FROM tick_data GROUP BY symbol')
            symbol_stats = dict(cursor.fetchall())
            
            # Son güncelleme zamanı
            cursor.execute('SELECT MAX(timestamp) FROM tick_data')
            last_update = cursor.fetchone()[0]
            
            return {
                'tick_count': tick_count,
                'orderbook_count': orderbook_count,
                'ohlcv_count': ohlcv_count,
                'symbol_stats': symbol_stats,
                'last_update': last_update,
                'recording_active': self.is_recording,
                'queue_sizes': {
                    'tick_queue': self.tick_queue.qsize(),
                    'orderbook_queue': self.orderbook_queue.qsize(),
                    'ohlcv_queue': self.ohlcv_queue.qsize()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Veri istatistikleri alma hatası: {e}")
            return {}
    
    def add_analysis_callback(self, callback: Callable):
        """Analiz callback'i ekle"""
        self.analysis_callbacks.append(callback)
    
    def update_config(self, new_config: TickRecordingConfig):
        """Konfigürasyonu güncelle"""
        try:
            self.config = new_config
            self.logger.info("Tick kayıt konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global tick veri kaydedici
tick_data_recorder = TickDataRecorder()
