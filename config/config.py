"""
BTCTURK Trading Bot Configuration
Merkezi konfigürasyon dosyası
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class APIConfig:
    """API konfigürasyonu"""
    base_url: str = "https://api.btcturk.com/api/v2"
    websocket_url: str = "wss://ws-feed.btcturk.com"
    api_key: str = ""
    secret_key: str = ""
    test_mode: bool = True
    app_mode: str = "paper"
    binance_api_key: str = ""
    binance_secret_key: str = ""

@dataclass
class TradingConfig:
    """Trading konfigürasyonu"""
    # Risk yönetimi
    max_position_size: float = 0.1  # Portföyün %10'u
    max_daily_loss: float = 0.05    # Günlük maksimum kayıp %5
    max_weekly_loss: float = 0.15   # Haftalık maksimum kayıp %15
    stop_loss_pct: float = 0.02     # Stop loss %2
    take_profit_pct: float = 0.04   # Take profit %4
    
    # Slippage ve komisyon
    slippage_tolerance: float = 0.001  # %0.1
    commission_rate: float = 0.001     # %0.1
    commission_rate_taker: float = 0.0010
    commission_rate_maker: float = 0.0009
    
    # Trading parametreleri
    min_trade_amount: float = 100.0    # Minimum işlem tutarı (TL)
    max_trade_amount: float = 10000.0  # Maksimum işlem tutarı (TL)

@dataclass
class StrategyConfig:
    """Strateji konfigürasyonu"""
    # Scalping
    scalping_enabled: bool = True
    scalping_profit_target: float = 0.005  # %0.5
    scalping_stop_loss: float = 0.003      # %0.3
    scalping_timeframe: str = "1m"
    
    # Grid Trading
    grid_enabled: bool = True
    grid_levels: int = 10
    grid_spacing: float = 0.01  # %1
    grid_profit_target: float = 0.02  # %2
    
    # Trend Following
    trend_enabled: bool = True
    trend_ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    trend_momentum_threshold: float = 0.02
    
    # Hedge Trading
    hedge_enabled: bool = False
    correlation_threshold: float = 0.7
    hedge_ratio: float = 0.5

@dataclass
class DatabaseConfig:
    """Veritabanı konfigürasyonu"""
    type: str = "sqlite"  # sqlite, postgresql
    sqlite_path: str = "data/trading_bot.db"
    postgresql_url: str = ""
    
    # Tablo isimleri
    trades_table: str = "trades"
    market_data_table: str = "market_data"
    signals_table: str = "signals"
    performance_table: str = "performance"

@dataclass
class LoggingConfig:
    """Logging konfigürasyonu"""
    level: str = "INFO"
    file_path: str = "logs/trading_bot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True

@dataclass
class GUIConfig:
    """GUI konfigürasyonu"""
    theme: str = "dark"
    window_size: tuple = (1200, 800)
    refresh_interval: int = 1000  # ms
    chart_candles: int = 1000

class Config:
    """Ana konfigürasyon sınıfı"""
    
    def __init__(self, config_file: str = "config/settings.json"):
        self.config_file = config_file
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.strategy = StrategyConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.gui = GUIConfig()
        
        self.load_config()
    
    def load_config(self):
        """Konfigürasyon dosyasını yükle"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Her bölümü güncelle
                if 'api' in config_data:
                    self._update_dataclass(self.api, config_data['api'])
                if 'trading' in config_data:
                    self._update_dataclass(self.trading, config_data['trading'])
                if 'strategy' in config_data:
                    self._update_dataclass(self.strategy, config_data['strategy'])
                if 'database' in config_data:
                    self._update_dataclass(self.database, config_data['database'])
                if 'logging' in config_data:
                    self._update_dataclass(self.logging, config_data['logging'])
                if 'gui' in config_data:
                    self._update_dataclass(self.gui, config_data['gui'])
                
                # ENV overrides
                self._apply_env_overrides()
                
            except Exception as e:
                print(f"Konfigürasyon yüklenirken hata: {e}")
        else:
            # Dosya yoksa varsayılanları dosyaya yaz
            try:
                self.save_config()
            except Exception as e:
                print(f"Varsayılan konfigürasyon oluşturulamadı: {e}")
    
    def save_config(self):
        """Konfigürasyonu dosyaya kaydet"""
        config_data = {
            'api': self._dataclass_to_dict(self.api),
            'trading': self._dataclass_to_dict(self.trading),
            'strategy': self._dataclass_to_dict(self.strategy),
            'database': self._dataclass_to_dict(self.database),
            'logging': self._dataclass_to_dict(self.logging),
            'gui': self._dataclass_to_dict(self.gui)
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
    
    def _update_dataclass(self, obj, data: dict):
        """Dataclass objesini güncelle"""
        # alias map
        alias_map = {
            'stop_loss_percentage': 'stop_loss_pct',
            'take_profit_percentage': 'take_profit_pct',
            'max_positions': 'max_open_positions',
            'max_concurrent_positions': 'max_open_positions',
            'slippage': 'slippage_tolerance',
        }
        for key, value in data.items():
            k = alias_map.get(key, key)
            if hasattr(obj, k):
                setattr(obj, k, value)
    
    def _dataclass_to_dict(self, obj) -> dict:
        """Dataclass objesini dictionary'ye çevir"""
        if hasattr(obj, '__dataclass_fields__'):
            return {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        return obj.__dict__

    def _apply_env_overrides(self):
        try:
            ak = os.getenv('BINANCE_API_KEY')
            sk = os.getenv('BINANCE_SECRET_KEY')
            am = os.getenv('APP_MODE')
            if ak:
                self.api.binance_api_key = ak
            if sk:
                self.api.binance_secret_key = sk
            if am:
                self.api.app_mode = am
        except Exception:
            pass

# Global konfigürasyon instance'ı
config = Config()
