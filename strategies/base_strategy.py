"""
Temel Strateji Sınıfı
Tüm trading stratejileri için base class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """Temel strateji sınıfı"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{name}")
        
        # Strateji durumu
        self.is_active = False
        self.positions = {}
        self.orders = {}
        
        # Performans metrikleri
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Sinyal geçmişi
        self.signal_history = []
        
        self.logger.info(f"Strateji '{name}' oluşturuldu")
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sinyal üret - tüm stratejiler bu metodu implement etmeli"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla"""
        pass
    
    def start(self):
        """Stratejiyi başlat"""
        self.is_active = True
        self.logger.info(f"Strateji '{self.name}' başlatıldı")
    
    def stop(self):
        """Stratejiyi durdur"""
        self.is_active = False
        self.logger.info(f"Strateji '{self.name}' durduruldu")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        self.config.update(new_config)
        self.logger.info(f"Strateji '{self.name}' konfigürasyonu güncellendi")
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    stop_loss: float = None, take_profit: float = None):
        """Pozisyon ekle"""
        position_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'created_at': datetime.now(),
            'status': 'open'
        }
        
        self.positions[position_id] = position
        self.logger.info(f"Pozisyon eklendi: {position_id}")
    
    def update_position(self, position_id: str, current_price: float):
        """Pozisyonu güncelle"""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        position['current_price'] = current_price
        
        # Unrealized P&L hesapla
        if position['side'] == 'buy':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
        
        # Drawdown hesapla
        if position['unrealized_pnl'] < 0:
            self.current_drawdown = abs(position['unrealized_pnl'])
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def close_position(self, position_id: str, exit_price: float, reason: str = "manual"):
        """Pozisyonu kapat"""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        position['exit_price'] = exit_price
        position['status'] = 'closed'
        position['closed_at'] = datetime.now()
        position['close_reason'] = reason
        
        # Realized P&L hesapla
        if position['side'] == 'buy':
            position['realized_pnl'] = (exit_price - position['entry_price']) * position['size']
        else:
            position['realized_pnl'] = (position['entry_price'] - exit_price) * position['size']
        
        # İstatistikleri güncelle
        self.total_trades += 1
        self.total_profit += position['realized_pnl']
        
        if position['realized_pnl'] > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.logger.info(f"Pozisyon kapatıldı: {position_id}, P&L: {position['realized_pnl']:.2f}")
    
    def check_stop_loss_take_profit(self, position_id: str, current_price: float) -> bool:
        """Stop loss ve take profit kontrolü"""
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        # Stop loss kontrolü
        if position['stop_loss']:
            if position['side'] == 'buy' and current_price <= position['stop_loss']:
                self.close_position(position_id, current_price, "stop_loss")
                return True
            elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                self.close_position(position_id, current_price, "stop_loss")
                return True
        
        # Take profit kontrolü
        if position['take_profit']:
            if position['side'] == 'buy' and current_price >= position['take_profit']:
                self.close_position(position_id, current_price, "take_profit")
                return True
            elif position['side'] == 'sell' and current_price <= position['take_profit']:
                self.close_position(position_id, current_price, "take_profit")
                return True
        
        return False
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Açık pozisyonları al"""
        return {pid: pos for pid, pos in self.positions.items() if pos['status'] == 'open'}
    
    def get_closed_positions(self) -> Dict[str, Dict[str, Any]]:
        """Kapatılmış pozisyonları al"""
        return {pid: pos for pid, pos in self.positions.items() if pos['status'] == 'closed'}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performans metriklerini al"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0
            }
        
        win_rate = (self.winning_trades / self.total_trades) * 100
        
        # Profit factor hesapla
        winning_profit = sum(pos['realized_pnl'] for pos in self.get_closed_positions().values() 
                           if pos['realized_pnl'] > 0)
        losing_loss = abs(sum(pos['realized_pnl'] for pos in self.get_closed_positions().values() 
                            if pos['realized_pnl'] < 0))
        
        profit_factor = winning_profit / losing_loss if losing_loss > 0 else float('inf')
        
        # Ortalama kazanç/kayıp
        wins = [pos['realized_pnl'] for pos in self.get_closed_positions().values() 
                if pos['realized_pnl'] > 0]
        losses = [pos['realized_pnl'] for pos in self.get_closed_positions().values() 
                 if pos['realized_pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'max_drawdown': self.max_drawdown,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
    
    def get_signal_strength(self, signal: Dict[str, Any]) -> float:
        """Sinyal gücünü hesapla (0-1 arası)"""
        # Bu metod alt sınıflarda override edilebilir
        return signal.get('strength', 0.5)
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Sinyal geçerliliğini kontrol et"""
        required_fields = ['symbol', 'side', 'strength']
        return all(field in signal for field in required_fields)
    
    def log_signal(self, signal: Dict[str, Any]):
        """Sinyali logla"""
        self.signal_history.append({
            'timestamp': datetime.now(),
            'signal': signal.copy()
        })
        
        # Son 1000 sinyali tut
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Son sinyalleri al"""
        return self.signal_history[-count:] if self.signal_history else []
    
    def reset_performance(self):
        """Performans verilerini sıfırla"""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.positions.clear()
        self.signal_history.clear()
        
        self.logger.info(f"Strateji '{self.name}' performans verileri sıfırlandı")
    
    def __str__(self):
        return f"Strategy({self.name}, active={self.is_active}, trades={self.total_trades})"
    
    def __repr__(self):
        return self.__str__()

