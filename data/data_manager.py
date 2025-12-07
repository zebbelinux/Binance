"""
Veri Yöneticisi
Veri toplama, saklama ve analiz sistemi
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sqlite3
import threading
import time
import json
import os
from dataclasses import dataclass
from enum import Enum

@dataclass
class DataConfig:
    """Veri konfigürasyonu"""
    database_path: str = "data/trading_bot.db"
    backup_interval: int = 3600  # 1 saat
    max_data_age: int = 30  # 30 gün
    compression_enabled: bool = True
    real_time_enabled: bool = True

class DataManager:
    """Veri yöneticisi sınıfı"""
    
    def __init__(self, config: DataConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or DataConfig()
        
        # Veritabanı bağlantısı
        self.db_connection = None
        self.db_lock = threading.Lock()
        
        # Veri cache
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
        # Thread yönetimi
        self.data_thread = None
        self.is_running = False
        
        # Callback'ler
        self.data_callbacks = []
        
        # Veritabanını başlat
        self._initialize_database()
        
        self.logger.info("Veri yöneticisi başlatıldı")
    
    def _initialize_database(self):
        """Veritabanını başlat"""
        try:
            # Veri klasörünü oluştur
            os.makedirs(os.path.dirname(self.config.database_path), exist_ok=True)
            
            # Veritabanı bağlantısı
            self.db_connection = sqlite3.connect(
                self.config.database_path,
                check_same_thread=False
            )
            
            # Tabloları oluştur
            self._create_tables()
            
            self.logger.info("Veritabanı başlatıldı")
            
        except Exception as e:
            self.logger.error(f"Veritabanı başlatma hatası: {e}")
    
    def _create_tables(self):
        """Veritabanı tablolarını oluştur"""
        try:
            cursor = self.db_connection.cursor()
            
            # Market data tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Trades tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    realized_pnl REAL,
                    commission REAL,
                    slippage REAL,
                    net_pnl REAL,
                    close_reason TEXT,
                    strategy_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signals tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    strength REAL NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    reason TEXT,
                    strategy_name TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_return REAL,
                    daily_pnl REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    volatility REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # AI analysis tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    sentiment TEXT,
                    confidence REAL,
                    recommendation TEXT,
                    risk_level TEXT,
                    market_regime TEXT,
                    analysis_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # İndeksler oluştur
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, entry_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_analysis_symbol_time ON ai_analysis(symbol, timestamp)')
            
            self.db_connection.commit()
            self.logger.info("Veritabanı tabloları oluşturuldu")
            
        except Exception as e:
            self.logger.error(f"Tablo oluşturma hatası: {e}")
    
    def start_data_collection(self):
        """Veri toplamayı başlat"""
        if self.is_running:
            return
        
        self.is_running = True
        self.data_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.data_thread.start()
        
        self.logger.info("Veri toplama başlatıldı")
    
    def stop_data_collection(self):
        """Veri toplamayı durdur"""
        self.is_running = False
        if self.data_thread:
            self.data_thread.join(timeout=5)
        
        self.logger.info("Veri toplama durduruldu")
    
    def _data_collection_loop(self):
        """Veri toplama döngüsü"""
        while self.is_running:
            try:
                # Veri temizleme
                self._clean_old_data()
                
                # Cache'i temizle
                self._clean_cache()
                
                # 60 saniye bekle
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Veri toplama döngüsü hatası: {e}")
                time.sleep(60)
    
    def save_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Market verisini kaydet"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    data.get('timestamp', datetime.now()),
                    data.get('open', 0),
                    data.get('high', 0),
                    data.get('low', 0),
                    data.get('close', 0),
                    data.get('volume', 0)
                ))
                
                self.db_connection.commit()
                
                # Cache'e ekle
                self._add_to_cache('market_data', symbol, data)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Market veri kaydetme hatası: {e}")
            return False
    
    def save_trade(self, trade: Dict[str, Any]) -> bool:
        """Trade kaydet"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (trade_id, symbol, side, size, entry_price, exit_price, 
                     entry_time, exit_time, realized_pnl, commission, slippage, 
                     net_pnl, close_reason, strategy_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get('id', ''),
                    trade.get('symbol', ''),
                    trade.get('side', ''),
                    trade.get('size', 0),
                    trade.get('entry_price', 0),
                    trade.get('exit_price', 0),
                    trade.get('entry_time', datetime.now()),
                    trade.get('exit_time'),
                    trade.get('realized_pnl', 0),
                    trade.get('commission', 0),
                    trade.get('slippage', 0),
                    trade.get('net_pnl', 0),
                    trade.get('close_reason', ''),
                    trade.get('strategy_name', '')
                ))
                
                self.db_connection.commit()
                
                # Cache'e ekle
                self._add_to_cache('trades', trade.get('symbol', ''), trade)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Trade kaydetme hatası: {e}")
            return False
    
    def save_signal(self, signal: Dict[str, Any]) -> bool:
        """Sinyal kaydet"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO signals 
                    (signal_id, symbol, side, strength, entry_price, stop_loss, 
                     take_profit, reason, strategy_name, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.get('id', ''),
                    signal.get('symbol', ''),
                    signal.get('side', ''),
                    signal.get('strength', 0),
                    signal.get('entry_price', 0),
                    signal.get('stop_loss', 0),
                    signal.get('take_profit', 0),
                    signal.get('reason', ''),
                    signal.get('strategy_name', ''),
                    signal.get('timestamp', datetime.now())
                ))
                
                self.db_connection.commit()
                
                # Cache'e ekle
                self._add_to_cache('signals', signal.get('symbol', ''), signal)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Sinyal kaydetme hatası: {e}")
            return False
    
    def save_performance(self, performance: Dict[str, Any]) -> bool:
        """Performans verisini kaydet"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO performance 
                    (date, total_return, daily_pnl, total_trades, winning_trades, 
                     losing_trades, win_rate, profit_factor, sharpe_ratio, 
                     max_drawdown, volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.get('date', datetime.now().date()),
                    performance.get('total_return', 0),
                    performance.get('daily_pnl', 0),
                    performance.get('total_trades', 0),
                    performance.get('winning_trades', 0),
                    performance.get('losing_trades', 0),
                    performance.get('win_rate', 0),
                    performance.get('profit_factor', 0),
                    performance.get('sharpe_ratio', 0),
                    performance.get('max_drawdown', 0),
                    performance.get('volatility', 0)
                ))
                
                self.db_connection.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Performans kaydetme hatası: {e}")
            return False
    
    def save_ai_analysis(self, symbol: str, analysis: Dict[str, Any]) -> bool:
        """AI analiz verisini kaydet"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                cursor.execute('''
                    INSERT INTO ai_analysis 
                    (symbol, timestamp, sentiment, confidence, recommendation, 
                     risk_level, market_regime, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    analysis.get('timestamp', datetime.now()),
                    analysis.get('sentiment', ''),
                    analysis.get('confidence', 0),
                    analysis.get('recommendation', ''),
                    analysis.get('risk_level', ''),
                    analysis.get('market_regime', ''),
                    json.dumps(analysis.get('analysis_data', {}))
                ))
                
                self.db_connection.commit()
                
                # Cache'e ekle
                self._add_to_cache('ai_analysis', symbol, analysis)
                
                return True
                
        except Exception as e:
            self.logger.error(f"AI analiz kaydetme hatası: {e}")
            return False
    
    def get_market_data(self, symbol: str, start_date: datetime = None, 
                       end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Market verisini al"""
        try:
            with self.db_lock:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ?
                '''
                params = [symbol]
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                df = pd.read_sql_query(query, self.db_connection, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Market veri alma hatası: {e}")
            return pd.DataFrame()
    
    def get_trades(self, symbol: str = None, start_date: datetime = None, 
                  end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Trade verisini al"""
        try:
            with self.db_lock:
                query = '''
                    SELECT trade_id, symbol, side, size, entry_price, exit_price,
                           entry_time, exit_time, realized_pnl, commission, slippage,
                           net_pnl, close_reason, strategy_name
                    FROM trades
                    WHERE 1=1
                '''
                params = []
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if start_date:
                    query += ' AND entry_time >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND entry_time <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY entry_time DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                df = pd.read_sql_query(query, self.db_connection, params=params)
                
                if not df.empty:
                    df['entry_time'] = pd.to_datetime(df['entry_time'])
                    df['exit_time'] = pd.to_datetime(df['exit_time'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Trade veri alma hatası: {e}")
            return pd.DataFrame()
    
    def get_signals(self, symbol: str = None, start_date: datetime = None, 
                   end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Sinyal verisini al"""
        try:
            with self.db_lock:
                query = '''
                    SELECT signal_id, symbol, side, strength, entry_price, stop_loss,
                           take_profit, reason, strategy_name, timestamp
                    FROM signals
                    WHERE 1=1
                '''
                params = []
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                df = pd.read_sql_query(query, self.db_connection, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Sinyal veri alma hatası: {e}")
            return pd.DataFrame()
    
    def get_performance(self, start_date: datetime = None, 
                       end_date: datetime = None) -> pd.DataFrame:
        """Performans verisini al"""
        try:
            with self.db_lock:
                query = '''
                    SELECT date, total_return, daily_pnl, total_trades, winning_trades,
                           losing_trades, win_rate, profit_factor, sharpe_ratio,
                           max_drawdown, volatility
                    FROM performance
                    WHERE 1=1
                '''
                params = []
                
                if start_date:
                    query += ' AND date >= ?'
                    params.append(start_date.date())
                
                if end_date:
                    query += ' AND date <= ?'
                    params.append(end_date.date())
                
                query += ' ORDER BY date DESC'
                
                df = pd.read_sql_query(query, self.db_connection, params=params)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Performans veri alma hatası: {e}")
            return pd.DataFrame()
    
    def get_ai_analysis(self, symbol: str = None, start_date: datetime = None, 
                       end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """AI analiz verisini al"""
        try:
            with self.db_lock:
                query = '''
                    SELECT symbol, timestamp, sentiment, confidence, recommendation,
                           risk_level, market_regime, analysis_data
                    FROM ai_analysis
                    WHERE 1=1
                '''
                params = []
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                df = pd.read_sql_query(query, self.db_connection, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # JSON verilerini parse et
                    df['analysis_data'] = df['analysis_data'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                return df
                
        except Exception as e:
            self.logger.error(f"AI analiz veri alma hatası: {e}")
            return pd.DataFrame()
    
    def _add_to_cache(self, data_type: str, key: str, data: Dict[str, Any]):
        """Cache'e veri ekle"""
        try:
            with self.cache_lock:
                if data_type not in self.data_cache:
                    self.data_cache[data_type] = {}
                
                if key not in self.data_cache[data_type]:
                    self.data_cache[data_type][key] = []
                
                self.data_cache[data_type][key].append({
                    'data': data,
                    'timestamp': datetime.now()
                })
                
                # Cache boyutunu sınırla
                if len(self.data_cache[data_type][key]) > 1000:
                    self.data_cache[data_type][key] = self.data_cache[data_type][key][-1000:]
                    
        except Exception as e:
            self.logger.error(f"Cache ekleme hatası: {e}")
    
    def _clean_cache(self):
        """Cache'i temizle"""
        try:
            with self.cache_lock:
                current_time = datetime.now()
                
                for data_type in self.data_cache:
                    for key in self.data_cache[data_type]:
                        # 1 saatten eski verileri kaldır
                        self.data_cache[data_type][key] = [
                            item for item in self.data_cache[data_type][key]
                            if (current_time - item['timestamp']).seconds < 3600
                        ]
                        
        except Exception as e:
            self.logger.error(f"Cache temizleme hatası: {e}")
    
    def _clean_old_data(self):
        """Eski verileri temizle"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                # Eski market data'yı temizle
                cutoff_date = datetime.now() - timedelta(days=self.config.max_data_age)
                
                cursor.execute('''
                    DELETE FROM market_data 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Eski AI analiz verilerini temizle
                cursor.execute('''
                    DELETE FROM ai_analysis 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Eski veri temizleme hatası: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Veritabanı istatistiklerini al"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                stats = {}
                
                # Tablo boyutları
                tables = ['market_data', 'trades', 'signals', 'performance', 'ai_analysis']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Veritabanı boyutu
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                stats['database_size_bytes'] = cursor.fetchone()[0]
                
                # Son güncelleme
                cursor.execute('SELECT MAX(created_at) FROM market_data')
                stats['last_market_data'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT MAX(created_at) FROM trades')
                stats['last_trade'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Veritabanı istatistikleri alma hatası: {e}")
            return {}
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Veritabanını yedekle"""
        try:
            if not backup_path:
                backup_path = f"data/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            # Veritabanını kopyala
            import shutil
            shutil.copy2(self.config.database_path, backup_path)
            
            self.logger.info(f"Veritabanı yedeklendi: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Veritabanı yedekleme hatası: {e}")
            return False
    
    def add_data_callback(self, callback):
        """Veri callback'i ekle"""
        self.data_callbacks.append(callback)
    
    def _notify_data_callbacks(self, data_type: str, data: Dict[str, Any]):
        """Veri callback'lerini çağır"""
        for callback in self.data_callbacks:
            try:
                callback(data_type, data)
            except Exception as e:
                self.logger.error(f"Veri callback hatası: {e}")
    
    def close(self):
        """Veri yöneticisini kapat"""
        try:
            self.stop_data_collection()
            
            if self.db_connection:
                self.db_connection.close()
            
            self.logger.info("Veri yöneticisi kapatıldı")
            
        except Exception as e:
            self.logger.error(f"Veri yöneticisi kapatma hatası: {e}")

# Global veri yöneticisi
data_manager = DataManager()




