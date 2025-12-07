"""
Örnek Strateji Eklentisi
Eklenti sisteminin nasıl kullanılacağını gösteren örnek strateji
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from plugins.plugin_manager import BasePlugin, PluginType
from utils.logger import get_logger, LogCategory

# Eklenti metadata'sı
__name__ = "example_strategy_plugin"
__version__ = "1.0.0"
__description__ = "Örnek strateji eklentisi - RSI tabanlı alım satım stratejisi"
__author__ = "Trading Bot Team"
__plugin_type__ = "strategy"
__dependencies__ = ["numpy", "pandas"]
__config_schema__ = {
    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50},
    "rsi_oversold": {"type": "float", "default": 30.0, "min": 10.0, "max": 40.0},
    "rsi_overbought": {"type": "float", "default": 70.0, "min": 60.0, "max": 90.0},
    "position_size": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
    "stop_loss_pct": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.1},
    "take_profit_pct": {"type": "float", "default": 0.04, "min": 0.01, "max": 0.2}
}

class ExampleStrategyPlugin(BasePlugin):
    """Örnek strateji eklentisi sınıfı"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Strateji parametreleri
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30.0)
        self.rsi_overbought = self.config.get('rsi_overbought', 70.0)
        self.position_size = self.config.get('position_size', 0.1)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)
        
        # Strateji durumu
        self.last_rsi = None
        self.last_signal = None
        self.signal_strength = 0.0
        
        self.logger.info(LogCategory.STRATEGY, f"Örnek strateji eklentisi başlatıldı: {name}")
    
    def _on_initialize(self) -> bool:
        """Başlatma işlemi"""
        try:
            self.logger.info(LogCategory.STRATEGY, "Strateji parametreleri yüklendi")
            self.logger.info(LogCategory.STRATEGY, f"RSI Periyodu: {self.rsi_period}")
            self.logger.info(LogCategory.STRATEGY, f"RSI Aşırı Satım: {self.rsi_oversold}")
            self.logger.info(LogCategory.STRATEGY, f"RSI Aşırı Alım: {self.rsi_overbought}")
            self.logger.info(LogCategory.STRATEGY, f"Pozisyon Büyüklüğü: {self.position_size}")
            
            return True
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Strateji başlatma hatası: {e}")
            return False
    
    def _on_activate(self) -> bool:
        """Etkinleştirme işlemi"""
        try:
            self.logger.info(LogCategory.STRATEGY, "Strateji etkinleştirildi")
            return True
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Strateji etkinleştirme hatası: {e}")
            return False
    
    def _on_deactivate(self) -> bool:
        """Devre dışı bırakma işlemi"""
        try:
            self.logger.info(LogCategory.STRATEGY, "Strateji devre dışı bırakıldı")
            return True
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Strateji devre dışı bırakma hatası: {e}")
            return False
    
    def _on_config_update(self, new_config: Dict[str, Any]) -> bool:
        """Konfigürasyon güncelleme işlemi"""
        try:
            # Parametreleri güncelle
            if 'rsi_period' in new_config:
                self.rsi_period = new_config['rsi_period']
            
            if 'rsi_oversold' in new_config:
                self.rsi_oversold = new_config['rsi_oversold']
            
            if 'rsi_overbought' in new_config:
                self.rsi_overbought = new_config['rsi_overbought']
            
            if 'position_size' in new_config:
                self.position_size = new_config['position_size']
            
            if 'stop_loss_pct' in new_config:
                self.stop_loss_pct = new_config['stop_loss_pct']
            
            if 'take_profit_pct' in new_config:
                self.take_profit_pct = new_config['take_profit_pct']
            
            self.logger.info(LogCategory.STRATEGY, "Strateji konfigürasyonu güncellendi")
            return True
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Konfigürasyon güncelleme hatası: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sinyal üret"""
        try:
            if not self.is_active:
                return []
            
            # Market verilerini al
            price = market_data.get('price', 0)
            technical_analysis = market_data.get('technical_analysis', {})
            
            if not price or not technical_analysis:
                return []
            
            # RSI değerini al
            rsi = technical_analysis.get('rsi', None)
            if rsi is None:
                return []
            
            self.last_rsi = rsi
            
            # Sinyal üret
            signals = []
            
            # Aşırı satım sinyali (Alım)
            if rsi < self.rsi_oversold:
                signal = self._create_buy_signal(price, rsi)
                if signal:
                    signals.append(signal)
                    self.last_signal = 'buy'
            
            # Aşırı alım sinyali (Satım)
            elif rsi > self.rsi_overbought:
                signal = self._create_sell_signal(price, rsi)
                if signal:
                    signals.append(signal)
                    self.last_signal = 'sell'
            
            # Sinyal gücünü hesapla
            self.signal_strength = self._calculate_signal_strength(rsi)
            
            return signals
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Sinyal üretme hatası: {e}")
            return []
    
    def _create_buy_signal(self, price: float, rsi: float) -> Optional[Dict[str, Any]]:
        """Alım sinyali oluştur"""
        try:
            # Sinyal gücünü hesapla
            strength = self._calculate_signal_strength(rsi)
            
            # Stop loss ve take profit hesapla
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            
            signal = {
                'id': f"buy_{int(datetime.now().timestamp() * 1000)}",
                'symbol': 'BTCUSDT',  # Varsayılan sembol
                'side': 'buy',
                'strength': strength,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': self.position_size,
                'reason': f'RSI aşırı satım: {rsi:.2f} < {self.rsi_oversold}',
                'strategy_name': self.name,
                'timestamp': datetime.now(),
                'metadata': {
                    'rsi': rsi,
                    'rsi_period': self.rsi_period,
                    'rsi_oversold': self.rsi_oversold,
                    'plugin_version': __version__
                }
            }
            
            self.logger.info(LogCategory.STRATEGY, f"Alım sinyali üretildi: RSI={rsi:.2f}, Güç={strength:.2f}")
            return signal
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Alım sinyali oluşturma hatası: {e}")
            return None
    
    def _create_sell_signal(self, price: float, rsi: float) -> Optional[Dict[str, Any]]:
        """Satım sinyali oluştur"""
        try:
            # Sinyal gücünü hesapla
            strength = self._calculate_signal_strength(rsi)
            
            # Stop loss ve take profit hesapla
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)
            
            signal = {
                'id': f"sell_{int(datetime.now().timestamp() * 1000)}",
                'symbol': 'BTCUSDT',  # Varsayılan sembol
                'side': 'sell',
                'strength': strength,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': self.position_size,
                'reason': f'RSI aşırı alım: {rsi:.2f} > {self.rsi_overbought}',
                'strategy_name': self.name,
                'timestamp': datetime.now(),
                'metadata': {
                    'rsi': rsi,
                    'rsi_period': self.rsi_period,
                    'rsi_overbought': self.rsi_overbought,
                    'plugin_version': __version__
                }
            }
            
            self.logger.info(LogCategory.STRATEGY, f"Satım sinyali üretildi: RSI={rsi:.2f}, Güç={strength:.2f}")
            return signal
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Satım sinyali oluşturma hatası: {e}")
            return None
    
    def _calculate_signal_strength(self, rsi: float) -> float:
        """Sinyal gücünü hesapla"""
        try:
            if rsi < self.rsi_oversold:
                # Aşırı satım bölgesinde - RSI ne kadar düşükse sinyal o kadar güçlü
                strength = (self.rsi_oversold - rsi) / self.rsi_oversold
                return min(strength, 1.0)
            
            elif rsi > self.rsi_overbought:
                # Aşırı alım bölgesinde - RSI ne kadar yüksekse sinyal o kadar güçlü
                strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                return min(strength, 1.0)
            
            else:
                # Nötr bölgede
                return 0.0
                
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Sinyal gücü hesaplama hatası: {e}")
            return 0.0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Strateji bilgilerini al"""
        return {
            'name': self.name,
            'version': __version__,
            'description': __description__,
            'author': __author__,
            'is_active': self.is_active,
            'parameters': {
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'current_state': {
                'last_rsi': self.last_rsi,
                'last_signal': self.last_signal,
                'signal_strength': self.signal_strength
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Konfigürasyonu doğrula"""
        try:
            # RSI periyodu kontrolü
            if 'rsi_period' in config:
                rsi_period = config['rsi_period']
                if not isinstance(rsi_period, int) or rsi_period < 5 or rsi_period > 50:
                    return False
            
            # RSI aşırı satım kontrolü
            if 'rsi_oversold' in config:
                rsi_oversold = config['rsi_oversold']
                if not isinstance(rsi_oversold, (int, float)) or rsi_oversold < 10 or rsi_oversold > 40:
                    return False
            
            # RSI aşırı alım kontrolü
            if 'rsi_overbought' in config:
                rsi_overbought = config['rsi_overbought']
                if not isinstance(rsi_overbought, (int, float)) or rsi_overbought < 60 or rsi_overbought > 90:
                    return False
            
            # Pozisyon büyüklüğü kontrolü
            if 'position_size' in config:
                position_size = config['position_size']
                if not isinstance(position_size, (int, float)) or position_size < 0.01 or position_size > 1.0:
                    return False
            
            # Stop loss kontrolü
            if 'stop_loss_pct' in config:
                stop_loss_pct = config['stop_loss_pct']
                if not isinstance(stop_loss_pct, (int, float)) or stop_loss_pct < 0.01 or stop_loss_pct > 0.1:
                    return False
            
            # Take profit kontrolü
            if 'take_profit_pct' in config:
                take_profit_pct = config['take_profit_pct']
                if not isinstance(take_profit_pct, (int, float)) or take_profit_pct < 0.01 or take_profit_pct > 0.2:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(LogCategory.STRATEGY, f"Konfigürasyon doğrulama hatası: {e}")
            return False




