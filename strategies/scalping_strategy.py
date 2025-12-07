"""
Scalping Stratejisi
Hızlı fiyat dalgalanmalarında küçük kazanç hedefli işlemler
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from strategies.base_strategy import BaseStrategy
from risk_management.position_sizer import position_sizer, PositionSizingMethod

class ScalpingStrategy(BaseStrategy):
    """Scalping stratejisi sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Scalping", config)
        
        # Scalping parametreleri
        self.profit_target = config.get('profit_target', 0.005)  # %0.5
        self.stop_loss = config.get('stop_loss', 0.003)          # %0.3
        self.timeframe = config.get('timeframe', '1m')
        self.min_volume = config.get('min_volume', 1000000)      # Minimum hacim
        self.max_spread = config.get('max_spread', 0.001)        # Maksimum spread %0.1
        
        # Teknik analiz parametreleri
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.ma_period = config.get('ma_period', 20)
        self.volume_ma_period = config.get('volume_ma_period', 20)
        
        # Sinyal filtreleri
        self.min_signal_strength = config.get('min_signal_strength', 0.6)
        self.max_position_size = config.get('max_position_size', 0.1)  # Portföyün %10'u
        # Confluence & Noise filtreleri
        self.confluence_threshold = config.get('confluence_threshold', 0.65)
        self.noise_min_atr_pct = config.get('noise_min_atr_pct', 0.001)  # ATR/price altı ise trade yok
        self.noise_min_bb_width = config.get('noise_min_bb_width', 0.005)  # (BBU-BBL)/price altı ise trade yok
        
        # Zaman filtreleri
        self.trading_hours = config.get('trading_hours', {
            'start': '09:00',
            'end': '18:00'
        })
        
        self.logger.info(f"Scalping stratejisi oluşturuldu - Profit: {self.profit_target*100:.2f}%, Stop: {self.stop_loss*100:.2f}%")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scalping sinyalleri üret"""
        signals = []
        
        try:
            symbol = market_data.get('symbol')
            if not symbol:
                return signals
            
            # Piyasa verilerini al
            price = market_data.get('price', 0)
            volume = market_data.get('volume_24h', 0)
            change_24h = market_data.get('change_24h', 0)
            
            # Teknik analiz verilerini al
            technical = market_data.get('technical_analysis', {})
            if not technical:
                return signals
            
            # AI analiz verilerini al
            ai_analysis = market_data.get('ai_analysis', {})
            
            # Temel filtreler
            if not self._is_valid_market_conditions(symbol, price, volume, change_24h):
                return signals
            
            # RSI sinyalleri
            rsi_signals = self._generate_rsi_signals(technical, price)
            signals.extend(rsi_signals)
            
            # Momentum sinyalleri
            momentum_signals = self._generate_momentum_signals(technical, price)
            signals.extend(momentum_signals)
            
            # Volume sinyalleri
            volume_signals = self._generate_volume_signals(technical, volume)
            signals.extend(volume_signals)
            
            # AI sinyalleri
            ai_signals = self._generate_ai_signals(ai_analysis, price)
            signals.extend(ai_signals)
            
            # Sinyalleri filtrele ve birleştir
            # Confluence uygulaması: BUY sinyalleri eşiğin altındaysa eler
            if conf_score is not None and conf_score < self.confluence_threshold:
                signals = [s for s in signals if (s.get('side') or '').lower() != 'buy']
            filtered_signals = self._filter_and_combine_signals(signals, price)
            
            # Sinyalleri logla
            for signal in filtered_signals:
                self.log_signal(signal)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Scalping sinyal üretme hatası: {e}")
            return signals

    def _confluence_score(self, technical: Dict[str, Any]) -> float:
        """score = (rsi_strength*0.3) + (macd_momentum*0.4) + (adx_trend*0.3)"""
        try:
            rsi = float(technical.get('rsi', 50) or 50)
            macd = float(technical.get('macd', 0) or 0)
            macd_sig = float(technical.get('macd_signal', 0) or 0)
            macd_hist = float(technical.get('macd_histogram', macd - macd_sig) or (macd - macd_sig))
            adx = float(technical.get('adx', 15) or 15)
            # Bileşenler (0..1)
            rsi_strength = max(0.0, min(1.0, (rsi - 50) / 50.0))
            macd_momentum = max(0.0, min(1.0, abs(macd_hist) * 10.0))
            adx_trend = max(0.0, min(1.0, adx / 50.0))
            score = rsi_strength * 0.3 + macd_momentum * 0.4 + adx_trend * 0.3
            return float(score)
        except Exception:
            return None

    def _noise_ok(self, technical: Dict[str, Any], price: float) -> bool:
        """ATR/BB genişliği ile düşük volatil ortamı filtrele."""
        try:
            if price <= 0:
                return False
            atr = technical.get('atr')
            if atr is not None:
                try:
                    atr_pct = float(atr) / float(price)
                    if atr_pct < float(self.noise_min_atr_pct):
                        return False
                except Exception:
                    pass
            # BB width
            bu = technical.get('bb_upper'); bl = technical.get('bb_lower')
            if bu is not None and bl is not None:
                try:
                    width = (float(bu) - float(bl)) / float(price)
                    if width < float(self.noise_min_bb_width):
                        return False
                except Exception:
                    pass
            return True
        except Exception:
            return True
    
    def _is_valid_market_conditions(self, symbol: str, price: float, volume: float, change_24h: float) -> bool:
        """Piyasa koşullarının geçerliliğini kontrol et"""
        # Fiyat kontrolü
        if price <= 0:
            return False
        
        # Hacim kontrolü
        if volume < self.min_volume:
            return False
        
        # Spread kontrolü (bid-ask farkı)
        # Bu veri market_data'da olmalı, şimdilik geçiyoruz
        
        # Aşırı volatilite kontrolü
        if abs(change_24h) > 0.1:  # %10'dan fazla değişim
            return False
        
        # Trading saatleri kontrolü
        current_hour = datetime.now().hour
        start_hour = int(self.trading_hours['start'].split(':')[0])
        end_hour = int(self.trading_hours['end'].split(':')[0])
        
        if not (start_hour <= current_hour <= end_hour):
            return False
        
        return True
    
    def _generate_rsi_signals(self, technical: Dict[str, Any], price: float) -> List[Dict[str, Any]]:
        """RSI tabanlı sinyaller üret"""
        signals = []
        
        rsi = technical.get('rsi')
        if rsi is None:
            return signals
        
        # RSI oversold - BUY sinyali
        if rsi < self.rsi_oversold:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'buy',
                'strength': min(0.9, (self.rsi_oversold - rsi) / self.rsi_oversold),
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss),
                'take_profit': price * (1 + self.profit_target),
                'reason': f'RSI oversold: {rsi:.2f}',
                'strategy': 'scalping_rsi'
            }
            signals.append(signal)
        
        # RSI overbought - SELL sinyali
        elif rsi > self.rsi_overbought:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'sell',
                'strength': min(0.9, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)),
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss),
                'take_profit': price * (1 - self.profit_target),
                'reason': f'RSI overbought: {rsi:.2f}',
                'strategy': 'scalping_rsi'
            }
            signals.append(signal)
        
        return signals
    
    def _generate_momentum_signals(self, technical: Dict[str, Any], price: float) -> List[Dict[str, Any]]:
        """Momentum tabanlı sinyaller üret"""
        signals = []
        
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        macd_hist = technical.get('macd_histogram', 0)
        
        # MACD bullish crossover
        if macd > macd_signal and macd_hist > 0:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'buy',
                'strength': min(0.8, abs(macd_hist) * 100),
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss),
                'take_profit': price * (1 + self.profit_target),
                'reason': f'MACD bullish: {macd:.4f} > {macd_signal:.4f}',
                'strategy': 'scalping_momentum'
            }
            signals.append(signal)
        
        # MACD bearish crossover
        elif macd < macd_signal and macd_hist < 0:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'sell',
                'strength': min(0.8, abs(macd_hist) * 100),
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss),
                'take_profit': price * (1 - self.profit_target),
                'reason': f'MACD bearish: {macd:.4f} < {macd_signal:.4f}',
                'strategy': 'scalping_momentum'
            }
            signals.append(signal)
        
        return signals
    
    def _generate_volume_signals(self, technical: Dict[str, Any], volume: float) -> List[Dict[str, Any]]:
        """Volume tabanlı sinyaller üret"""
        signals = []
        
        volume_ratio = technical.get('volume_ratio', 1.0)
        volume_sma = technical.get('volume_sma', volume)
        
        # Yüksek hacim + fiyat artışı
        if volume_ratio > 1.5 and technical.get('price_change', 0) > 0:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'buy',
                'strength': min(0.7, volume_ratio / 3),
                'entry_price': technical.get('price', 0),
                'stop_loss': technical.get('price', 0) * (1 - self.stop_loss),
                'take_profit': technical.get('price', 0) * (1 + self.profit_target),
                'reason': f'High volume breakout: {volume_ratio:.2f}x',
                'strategy': 'scalping_volume'
            }
            signals.append(signal)
        
        # Yüksek hacim + fiyat düşüşü
        elif volume_ratio > 1.5 and technical.get('price_change', 0) < 0:
            signal = {
                'symbol': technical.get('symbol', ''),
                'side': 'sell',
                'strength': min(0.7, volume_ratio / 3),
                'entry_price': technical.get('price', 0),
                'stop_loss': technical.get('price', 0) * (1 + self.stop_loss),
                'take_profit': technical.get('price', 0) * (1 - self.profit_target),
                'reason': f'High volume breakdown: {volume_ratio:.2f}x',
                'strategy': 'scalping_volume'
            }
            signals.append(signal)
        
        return signals
    
    def _generate_ai_signals(self, ai_analysis: Dict[str, Any], price: float) -> List[Dict[str, Any]]:
        """AI tabanlı sinyaller üret"""
        signals = []
        
        # AI sinyalleri
        ai_signals = ai_analysis.get('signals', {})
        if not ai_signals:
            return signals
        
        ai_signal = ai_signals.get('signal', 'HOLD')
        ai_strength = ai_signals.get('strength', 0.5)
        
        if ai_signal == 'BUY' and ai_strength >= self.min_signal_strength:
            signal = {
                'symbol': ai_analysis.get('symbol', ''),
                'side': 'buy',
                'strength': ai_strength,
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss),
                'take_profit': price * (1 + self.profit_target),
                'reason': f'AI BUY signal: {ai_strength:.2f}',
                'strategy': 'scalping_ai'
            }
            signals.append(signal)
        
        elif ai_signal == 'SELL' and ai_strength >= self.min_signal_strength:
            signal = {
                'symbol': ai_analysis.get('symbol', ''),
                'side': 'sell',
                'strength': ai_strength,
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss),
                'take_profit': price * (1 - self.profit_target),
                'reason': f'AI SELL signal: {ai_strength:.2f}',
                'strategy': 'scalping_ai'
            }
            signals.append(signal)
        
        return signals
    
    def _filter_and_combine_signals(self, signals: List[Dict[str, Any]], price: float) -> List[Dict[str, Any]]:
        """Sinyalleri filtrele ve birleştir"""
        if not signals:
            return []
        
        # Sinyal gücüne göre sırala
        signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # En güçlü sinyali al
        best_signal = signals[0]
        
        # Minimum sinyal gücü kontrolü
        if best_signal.get('strength', 0) < self.min_signal_strength:
            return []
        
        # Aynı sembol için açık pozisyon kontrolü
        symbol = best_signal.get('symbol')
        open_positions = self.get_open_positions()
        for pos in open_positions.values():
            if pos['symbol'] == symbol:
                return []  # Zaten açık pozisyon var
        
        return [best_signal]
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla - Position Sizer kullanılarak"""
        try:
            # Position_sizer kullanarak hesapla
            position_size, details = position_sizer.calculate_position_size(
                method=PositionSizingMethod.SIGNAL_BASED_FULL_CAPITAL,
                account_balance=account_balance,
                signal=signal,
                market_data={},  # Scalping için gerekli değil
                historical_performance=None
            )
            
            self.logger.info(f"Pozisyon büyüklüğü hesaplandı: {position_size:.2f} USD (Signal strength: {signal.get('strength', 0.5):.2f}, Method: {details.get('method', 'N/A')})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            # Fallback: minimum tutar
            return min(250.0, account_balance)
    
    def get_signal_strength(self, signal: Dict[str, Any]) -> float:
        """Sinyal gücünü hesapla"""
        base_strength = signal.get('strength', 0.5)
        
        # RSI sinyalleri için ekstra güçlendirme
        if signal.get('strategy') == 'scalping_rsi':
            rsi_value = signal.get('rsi_value', 50)
            if rsi_value < 25 or rsi_value > 75:
                base_strength *= 1.2
        
        # Volume sinyalleri için ekstra güçlendirme
        elif signal.get('strategy') == 'scalping_volume':
            volume_ratio = signal.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                base_strength *= 1.3
        
        return min(1.0, base_strength)
    
    def should_exit_position(self, position: Dict[str, Any], current_price: float) -> bool:
        """Pozisyon çıkış koşullarını kontrol et"""
        # Stop loss / Take profit kontrolü
        if self.check_stop_loss_take_profit(position['id'], current_price):
            return True
        
        # Zaman bazlı çıkış (5 dakika)
        position_age = datetime.now() - position['created_at']
        if position_age > timedelta(minutes=5):
            self.close_position(position['id'], current_price, "time_exit")
            return True
        
        # Hızlı kar alma (küçük kazanç)
        if position['unrealized_pnl'] > 0 and position['unrealized_pnl'] > position['size'] * 0.002:  # %0.2
            self.close_position(position['id'], current_price, "quick_profit")
            return True
        
        return False
    
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Strateji bilgilerini al"""
        return {
            'name': self.name,
            'type': 'scalping',
            'profit_target': self.profit_target,
            'stop_loss': self.stop_loss,
            'timeframe': self.timeframe,
            'min_signal_strength': self.min_signal_strength,
            'max_position_size': self.max_position_size,
            'trading_hours': self.trading_hours,
            'is_active': self.is_active,
            'performance': self.get_performance_metrics()
        }

