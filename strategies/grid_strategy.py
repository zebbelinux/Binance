"""
Grid Trading Stratejisi
Yatay piyasalarda otomatik kademeli emirlerle al/sat
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from strategies.base_strategy import BaseStrategy

class GridStrategy(BaseStrategy):
    """Grid trading stratejisi sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Grid", config)
        
        # Grid parametreleri
        self.grid_levels = config.get('grid_levels', 10)
        self.grid_spacing = config.get('grid_spacing', 0.01)  # %1
        # Spacing clip: 0.25%..0.80%
        try:
            self.grid_spacing = max(0.0025, min(0.0080, float(self.grid_spacing or 0.01)))
        except Exception:
            self.grid_spacing = 0.01
        self.grid_profit_target = config.get('grid_profit_target', 0.02)  # %2
        self.max_position_size = config.get('max_position_size', 0.2)  # Portföyün %20'si
        
        # Grid yönetimi
        self.grid_orders = {}  # {symbol: {buy_orders: [], sell_orders: []}}
        self.grid_positions = {}  # {symbol: {levels: [], total_size: 0}}
        self.grid_center_price = {}  # {symbol: center_price}
        
        # Risk yönetimi
        self.max_grid_loss = config.get('max_grid_loss', 0.05)  # %5
        self.grid_stop_loss = config.get('grid_stop_loss', 0.1)  # %10
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # %5
        
        # Piyasa koşulları
        self.min_volatility = config.get('min_volatility', 0.001)  # %0.1
        self.max_volatility = config.get('max_volatility', 0.05)   # %5
        self.sideways_threshold = config.get('sideways_threshold', 0.02)  # %2
        
        # Teknik analiz
        self.ma_period = config.get('ma_period', 20)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        
        self.logger.info(f"Grid stratejisi oluşturuldu - Seviyeler: {self.grid_levels}, Aralık: {self.grid_spacing*100:.1f}%")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Grid sinyalleri üret"""
        signals = []
        
        try:
            symbol = market_data.get('symbol')
            if not symbol:
                return signals
            
            price = market_data.get('price', 0)
            if price <= 0:
                return signals
            
            # Piyasa koşullarını kontrol et
            if not self._is_suitable_for_grid(market_data):
                return signals
            
            # Grid'i başlat veya güncelle
            if symbol not in self.grid_orders:
                self._initialize_grid(symbol, price)
            
            # Grid seviyelerini kontrol et
            grid_signals = self._check_grid_levels(symbol, price)
            signals.extend(grid_signals)
            
            # Grid rebalancing
            rebalance_signals = self._check_rebalancing(symbol, price)
            signals.extend(rebalance_signals)
            
            # Grid kapatma sinyalleri
            close_signals = self._check_grid_closing(symbol, price)
            signals.extend(close_signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Grid sinyal üretme hatası: {e}")
            return signals
    
    def _is_suitable_for_grid(self, market_data: Dict[str, Any]) -> bool:
        """Grid trading için uygun piyasa koşulları mı?"""
        try:
            # Volatilite kontrolü
            volatility = market_data.get('technical_analysis', {}).get('volatility', 0)
            if volatility < self.min_volatility or volatility > self.max_volatility:
                return False
            
            # Trend kontrolü - yatay piyasa olmalı
            technical = market_data.get('technical_analysis', {})
            ma_20 = technical.get('ma_20', 0)
            ma_50 = technical.get('ma_50', 0)
            
            if ma_20 > 0 and ma_50 > 0:
                trend_strength = abs(ma_20 - ma_50) / ma_50
                if trend_strength > self.sideways_threshold:
                    return False
            
            # Bollinger Bands genişliği kontrolü
            bb_upper = technical.get('bb_upper', 0)
            bb_lower = technical.get('bb_lower', 0)
            price = market_data.get('price', 0)
            
            if bb_upper > 0 and bb_lower > 0 and price > 0:
                bb_width = (bb_upper - bb_lower) / price
                if bb_width < self.grid_spacing * 2:  # Çok dar bant
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Grid uygunluk kontrolü hatası: {e}")
            return False
    
    def _initialize_grid(self, symbol: str, price: float):
        """Grid'i başlat"""
        try:
            self.grid_center_price[symbol] = price
            self.grid_orders[symbol] = {
                'buy_orders': [],
                'sell_orders': []
            }
            self.grid_positions[symbol] = {
                'levels': [],
                'total_size': 0
            }
            
            # Grid seviyelerini oluştur
            self._create_grid_levels(symbol, price)
            
            self.logger.info(f"Grid başlatıldı: {symbol} @ {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Grid başlatma hatası: {e}")
    
    def _create_grid_levels(self, symbol: str, center_price: float):
        """Grid seviyelerini oluştur"""
        try:
            # Spacing'i her çağrıda sınırla (dinamik ayarlarla tutarlılık)
            try:
                self.grid_spacing = max(0.0025, min(0.0080, float(self.grid_spacing or 0.01)))
            except Exception:
                pass
            buy_orders = []
            sell_orders = []
            
            # Alış seviyeleri (aşağı)
            for i in range(1, self.grid_levels + 1):
                level_price = center_price * (1 - self.grid_spacing * i)
                order = {
                    'level': i,
                    'price': level_price,
                    'size': 0,  # Pozisyon büyüklüğü hesaplanacak
                    'status': 'pending',
                    'created_at': datetime.now()
                }
                buy_orders.append(order)
            
            # Satış seviyeleri (yukarı)
            for i in range(1, self.grid_levels + 1):
                level_price = center_price * (1 + self.grid_spacing * i)
                order = {
                    'level': i,
                    'price': level_price,
                    'size': 0,  # Pozisyon büyüklüğü hesaplanacak
                    'status': 'pending',
                    'created_at': datetime.now()
                }
                sell_orders.append(order)
            
            self.grid_orders[symbol] = {
                'buy_orders': buy_orders,
                'sell_orders': sell_orders
            }
            
        except Exception as e:
            self.logger.error(f"Grid seviyeleri oluşturma hatası: {e}")
    
    def _check_grid_levels(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Grid seviyelerini kontrol et"""
        signals = []
        
        try:
            if symbol not in self.grid_orders:
                return signals
            
            grid = self.grid_orders[symbol]
            
            # Alış seviyelerini kontrol et
            for order in grid['buy_orders']:
                if (order['status'] == 'pending' and 
                    current_price <= order['price'] and 
                    order['size'] > 0):
                    
                    # Profit target tabanı
                    tp_target = self._ensure_profit_target_base(order['price'], side='buy', market_data=market_data if 'market_data' in locals() else None)
                    signal = {
                        'symbol': symbol,
                        'side': 'buy',
                        'strength': 0.8,
                        'entry_price': order['price'],
                        'stop_loss': order['price'] * (1 - self.grid_stop_loss),
                        'take_profit': order['price'] * (1 + max(self.grid_profit_target, tp_target)),
                        'reason': f'Grid buy level {order["level"]}',
                        'strategy': 'grid_buy',
                        'grid_level': order['level']
                    }
                    signals.append(signal)
                    order['status'] = 'triggered'
            
            # Satış seviyelerini kontrol et
            for order in grid['sell_orders']:
                if (order['status'] == 'pending' and 
                    current_price >= order['price'] and 
                    order['size'] > 0):
                    
                    # Profit target tabanı
                    tp_target = self._ensure_profit_target_base(order['price'], side='sell', market_data=market_data if 'market_data' in locals() else None)
                    signal = {
                        'symbol': symbol,
                        'side': 'sell',
                        'strength': 0.8,
                        'entry_price': order['price'],
                        'stop_loss': order['price'] * (1 + self.grid_stop_loss),
                        'take_profit': order['price'] * (1 - max(self.grid_profit_target, tp_target)),
                        'reason': f'Grid sell level {order["level"]}',
                        'strategy': 'grid_sell',
                        'grid_level': order['level']
                    }
                    signals.append(signal)
                    order['status'] = 'triggered'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Grid seviye kontrolü hatası: {e}")
            return signals

    def _ensure_profit_target_base(self, entry_price: float, side: str = 'buy', market_data: Dict[str, Any] | None = None) -> float:
        """Edge-cost tabanlı minimum kar hedefi yüzdesi döndürür (yüzdelik, örn 0.004)."""
        try:
            spread = 0.0002
            slip = 0.0006
            fee = 0.0010
            try:
                if isinstance(market_data, dict):
                    ms = (market_data.get('microstructure') or {}) if isinstance(market_data.get('microstructure'), dict) else {}
                    spread = float(ms.get('spread_pct', spread) or spread)
                    slip = float(ms.get('slippage_est_pct', slip) or slip)
            except Exception:
                pass
            edge_cost = 2.0 * fee + spread + slip
            base = 2.2 * edge_cost + 0.0005
            return max(0.0, float(base))
        except Exception:
            return 0.003
    
    def _check_rebalancing(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Grid rebalancing kontrolü"""
        signals = []
        
        try:
            if symbol not in self.grid_center_price:
                return signals
            
            center_price = self.grid_center_price[symbol]
            price_deviation = abs(current_price - center_price) / center_price
            
            # Fiyat merkezden çok uzaklaştıysa rebalance et
            if price_deviation > self.rebalance_threshold:
                # Yeni merkez fiyatı belirle
                new_center = current_price
                self.grid_center_price[symbol] = new_center
                
                # Grid'i yeniden oluştur
                self._create_grid_levels(symbol, new_center)
                
                signal = {
                    'symbol': symbol,
                    'side': 'rebalance',
                    'strength': 1.0,
                    'entry_price': new_center,
                    'reason': f'Grid rebalancing: {price_deviation*100:.2f}% deviation',
                    'strategy': 'grid_rebalance'
                }
                signals.append(signal)
                
                self.logger.info(f"Grid rebalancing: {symbol} @ {new_center:.2f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Grid rebalancing hatası: {e}")
            return signals
    
    def _check_grid_closing(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Grid kapatma sinyalleri"""
        signals = []
        
        try:
            if symbol not in self.grid_positions:
                return signals
            
            position = self.grid_positions[symbol]
            total_size = position['total_size']
            
            # Maksimum kayıp kontrolü
            if total_size > 0:
                unrealized_pnl = self._calculate_grid_pnl(symbol, current_price)
                loss_ratio = abs(unrealized_pnl) / total_size if total_size > 0 else 0
                
                if loss_ratio > self.max_grid_loss:
                    signal = {
                        'symbol': symbol,
                        'side': 'close_all',
                        'strength': 1.0,
                        'entry_price': current_price,
                        'reason': f'Grid stop loss: {loss_ratio*100:.2f}%',
                        'strategy': 'grid_stop'
                    }
                    signals.append(signal)
                    
                    # Grid'i kapat
                    self._close_grid(symbol)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Grid kapatma kontrolü hatası: {e}")
            return signals
    
    def _calculate_grid_pnl(self, symbol: str, current_price: float) -> float:
        """Grid P&L hesapla"""
        try:
            if symbol not in self.grid_positions:
                return 0
            
            total_pnl = 0
            for level in self.grid_positions[symbol]['levels']:
                entry_price = level['entry_price']
                size = level['size']
                side = level['side']
                
                if side == 'buy':
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                total_pnl += pnl
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Grid P&L hesaplama hatası: {e}")
            return 0
    
    def _close_grid(self, symbol: str):
        """Grid'i kapat"""
        try:
            if symbol in self.grid_orders:
                del self.grid_orders[symbol]
            if symbol in self.grid_positions:
                del self.grid_positions[symbol]
            if symbol in self.grid_center_price:
                del self.grid_center_price[symbol]
            
            self.logger.info(f"Grid kapatıldı: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Grid kapatma hatası: {e}")
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla"""
        try:
            symbol = signal.get('symbol')
            if not symbol:
                return 0
            
            # Grid için sabit pozisyon büyüklüğü
            base_size = account_balance * (self.max_position_size / self.grid_levels)
            
            # Maksimum pozisyon kontrolü
            if symbol in self.grid_positions:
                current_size = self.grid_positions[symbol]['total_size']
                remaining_size = account_balance * self.max_position_size - current_size
                base_size = min(base_size, remaining_size)
            
            # Minimum pozisyon kontrolü
            min_size = 50  # Minimum 50 TL
            if base_size < min_size:
                return 0
            
            return base_size
            
        except Exception as e:
            self.logger.error(f"Grid pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0
    
    def add_grid_position(self, symbol: str, side: str, size: float, entry_price: float, level: int):
        """Grid pozisyonu ekle"""
        try:
            if symbol not in self.grid_positions:
                self.grid_positions[symbol] = {
                    'levels': [],
                    'total_size': 0
                }
            
            position = {
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'level': level,
                'created_at': datetime.now()
            }
            
            self.grid_positions[symbol]['levels'].append(position)
            self.grid_positions[symbol]['total_size'] += size
            
            self.logger.info(f"Grid pozisyon eklendi: {symbol} {side} {size:.2f} @ {entry_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Grid pozisyon ekleme hatası: {e}")
    
    def get_grid_status(self, symbol: str) -> Dict[str, Any]:
        """Grid durumunu al"""
        try:
            if symbol not in self.grid_orders:
                return {'status': 'not_initialized'}
            
            grid = self.grid_orders[symbol]
            center_price = self.grid_center_price.get(symbol, 0)
            
            active_buy_orders = len([o for o in grid['buy_orders'] if o['status'] == 'pending'])
            active_sell_orders = len([o for o in grid['sell_orders'] if o['status'] == 'pending'])
            
            total_positions = 0
            if symbol in self.grid_positions:
                total_positions = len(self.grid_positions[symbol]['levels'])
            
            return {
                'status': 'active',
                'center_price': center_price,
                'active_buy_orders': active_buy_orders,
                'active_sell_orders': active_sell_orders,
                'total_positions': total_positions,
                'grid_levels': self.grid_levels,
                'grid_spacing': self.grid_spacing
            }
            
        except Exception as e:
            self.logger.error(f"Grid durumu alma hatası: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla"""
        try:
            # Grid için sabit pozisyon büyüklüğü
            base_size = account_balance * self.max_position_size
            
            # Grid seviyesine göre ayarla
            grid_level = signal.get('grid_level', 1)
            level_size = base_size / self.grid_levels
            
            # Sinyal gücüne göre ayarla
            signal_strength = signal.get('strength', 0.5)
            adjusted_size = level_size * signal_strength
            
            # Minimum ve maksimum sınırlar
            min_size = account_balance * 0.005  # %0.5 minimum
            max_size = level_size
            
            final_size = max(min_size, min(adjusted_size, max_size))
            
            self.logger.debug(f"Grid pozisyon büyüklüğü hesaplandı: {final_size:.2f} (Level: {grid_level})")
            return final_size
            
        except Exception as e:
            self.logger.error(f"Grid pozisyon büyüklüğü hesaplama hatası: {e}")
            return account_balance * 0.005  # Güvenli minimum
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Strateji bilgilerini al"""
        return {
            'name': self.name,
            'type': 'grid',
            'grid_levels': self.grid_levels,
            'grid_spacing': self.grid_spacing,
            'grid_profit_target': self.grid_profit_target,
            'max_position_size': self.max_position_size,
            'max_grid_loss': self.max_grid_loss,
            'rebalance_threshold': self.rebalance_threshold,
            'is_active': self.is_active,
            'performance': self.get_performance_metrics(),
            'active_grids': len(self.grid_orders)
        }

