"""
Pozisyon Büyüklüğü Hesaplayıcı
Kelly Kriteri ve diğer pozisyon büyüklüğü yöntemleri
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from enum import Enum

class PositionSizingMethod(Enum):
    """Pozisyon büyüklüğü yöntemleri"""
    FIXED = "fixed"                    # Sabit yüzde
    KELLY = "kelly"                    # Kelly Kriteri
    FIXED_FRACTIONAL = "fixed_fractional"  # Sabit kesir
    VOLATILITY_BASED = "volatility_based"  # Volatilite bazlı
    RISK_PARITY = "risk_parity"        # Risk paritesi
    MARTINGALE = "martingale"          # Martingale
    ANTI_MARTINGALE = "anti_martingale"  # Anti-Martingale
    SIGNAL_BASED_FULL_CAPITAL = "signal_based_full_capital"  # Sinyal bazlı tam sermaye
    ATR_RISK_BASED = "atr_risk_based"  # ATR bazlı risk-per-trade

class PositionSizer:
    """Pozisyon büyüklüğü hesaplayıcı sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Varsayılan parametreler
        self.default_method = PositionSizingMethod.SIGNAL_BASED_FULL_CAPITAL
        self.fixed_percentage = 0.02  # %2
        self.max_position_size = 1.0  # %100 (tam sermaye)
        self.min_position_size = 0.001  # %0.1
        self.min_trade_amount = 250.0  # Minimum işlem tutarı (USD)
        
        # Kelly Kriteri parametreleri
        self.kelly_fraction = 0.25  # Kelly'nin %25'i
        self.min_win_rate = 0.3     # Minimum kazanma oranı
        self.min_profit_factor = 1.1  # Minimum kar faktörü
        
        # Volatilite bazlı parametreler
        self.volatility_target = 0.02  # %2 volatilite hedefi
        self.volatility_lookback = 20  # 20 periyot
        
        # Risk paritesi parametreleri
        self.risk_budget = 0.02  # %2 risk bütçesi
        
        self.logger.info("Pozisyon büyüklüğü hesaplayıcı başlatıldı")
    
    def calculate_position_size(self, 
                              method: PositionSizingMethod,
                              account_balance: float,
                              signal: Dict[str, Any],
                              market_data: Dict[str, Any],
                              historical_performance: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """Pozisyon büyüklüğü hesapla"""
        try:
            if method == PositionSizingMethod.FIXED:
                return self._calculate_fixed_size(account_balance, signal)
            
            elif method == PositionSizingMethod.KELLY:
                return self._calculate_kelly_size(account_balance, signal, historical_performance)
            
            elif method == PositionSizingMethod.FIXED_FRACTIONAL:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            elif method == PositionSizingMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based_size(account_balance, signal, market_data)
            
            elif method == PositionSizingMethod.RISK_PARITY:
                return self._calculate_risk_parity_size(account_balance, signal, market_data)
            
            elif method == PositionSizingMethod.MARTINGALE:
                return self._calculate_martingale_size(account_balance, signal, historical_performance)
            
            elif method == PositionSizingMethod.ANTI_MARTINGALE:
                return self._calculate_anti_martingale_size(account_balance, signal, historical_performance)
            
            elif method == PositionSizingMethod.SIGNAL_BASED_FULL_CAPITAL:
                return self._calculate_signal_based_full_capital(account_balance, signal)

            elif method == PositionSizingMethod.ATR_RISK_BASED:
                return self._calculate_atr_risk_based(account_balance, signal, market_data)
            
            else:
                return self._calculate_fixed_fractional_size(account_balance, signal)
                
        except Exception as e:
            self.logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_fixed_size(self, account_balance: float, signal: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Sabit yüzde yöntemi"""
        try:
            # Sinyal gücüne göre ayarla
            signal_strength = signal.get('strength', 0.5)
            adjusted_percentage = self.fixed_percentage * signal_strength
            
            # Limitler içinde tut
            adjusted_percentage = max(self.min_position_size, 
                                   min(adjusted_percentage, self.max_position_size))
            
            position_size = account_balance * adjusted_percentage
            
            return position_size, {
                'method': 'fixed',
                'percentage': adjusted_percentage,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            self.logger.error(f"Sabit pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_kelly_size(self, account_balance: float, signal: Dict[str, Any], 
                            historical_performance: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Kelly Kriteri yöntemi"""
        try:
            if not historical_performance:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # Performans metriklerini al
            win_rate = historical_performance.get('win_rate', 0.5)
            avg_win = historical_performance.get('average_win', 0.01)
            avg_loss = abs(historical_performance.get('average_loss', -0.01))
            
            # Kelly formülü: f = (bp - q) / b
            # b = ortalama kazanç / ortalama kayıp
            # p = kazanma olasılığı
            # q = kaybetme olasılığı (1-p)
            
            if avg_loss == 0:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            b = avg_win / avg_loss
            p = win_rate / 100  # Yüzdeyi ondalığa çevir
            q = 1 - p
            
            # Kelly fraksiyonu
            kelly_fraction = (b * p - q) / b
            
            # Kelly'nin sadece bir kısmını kullan (risk yönetimi için)
            kelly_fraction *= self.kelly_fraction
            
            # Minimum performans kontrolü
            if win_rate < self.min_win_rate or (avg_win / avg_loss) < self.min_profit_factor:
                kelly_fraction = 0.01  # Çok düşük pozisyon
            
            # Limitler içinde tut
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            position_size = account_balance * kelly_fraction
            
            return position_size, {
                'method': 'kelly',
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'b_ratio': b
            }
            
        except Exception as e:
            self.logger.error(f"Kelly pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_fixed_fractional_size(self, account_balance: float, signal: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Sabit kesir yöntemi"""
        try:
            # Sinyal gücüne göre ayarla
            signal_strength = signal.get('strength', 0.5)
            base_percentage = self.fixed_percentage
            
            # Sinyal gücüne göre ayarla
            if signal_strength > 0.8:
                adjusted_percentage = base_percentage * 1.5
            elif signal_strength > 0.6:
                adjusted_percentage = base_percentage * 1.2
            elif signal_strength < 0.4:
                adjusted_percentage = base_percentage * 0.5
            else:
                adjusted_percentage = base_percentage
            
            # Limitler içinde tut
            adjusted_percentage = max(self.min_position_size, 
                                   min(adjusted_percentage, self.max_position_size))
            
            position_size = account_balance * adjusted_percentage
            
            return position_size, {
                'method': 'fixed_fractional',
                'percentage': adjusted_percentage,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            self.logger.error(f"Sabit kesir pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0, {'error': str(e)}

    def _calculate_atr_risk_based(self, account_balance: float, signal: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """ATR bazlı risk-per-trade yöntemi.

        Önerilen formül:
        risk_per_trade = equity * 0.005
        qty = risk_per_trade / ATR

        Binance USDT bot senaryosunda executor genelde notional (USDT) ile çalıştığı için,
        burada risk_per_trade'i doğrudan kullanılacak notional tavanı olarak ele alıyoruz.
        ATR okunamazsa veya anlamsızsa sabit kesir yöntemine düşer.
        """
        try:
            # ATR ve fiyatı mümkünse sinyalden, yoksa market_data'dan oku
            atr = signal.get('atr')
            price = signal.get('price', signal.get('entry_price'))

            if atr is None or float(atr or 0.0) <= 0.0 or price is None or float(price or 0.0) <= 0.0:
                try:
                    ta = (market_data or {}).get('technical_analysis', {}) if isinstance(market_data, dict) else {}
                    if atr is None:
                        atr = ta.get('atr')
                    if price is None:
                        price = (market_data or {}).get('price')
                except Exception:
                    pass

            try:
                atr_val = float(atr or 0.0)
                price_val = float(price or 0.0)
            except Exception:
                atr_val, price_val = 0.0, 0.0

            # ATR veya fiyat hâlâ geçerli değilse sabit kesire dön
            if atr_val <= 0.0 or price_val <= 0.0:
                return self._calculate_fixed_fractional_size(account_balance, signal)

            # Trade başına risk bütçesi (ör: bakiyenin %0.5'i)
            risk_pct = 0.005
            risk_per_trade = max(0.0, float(account_balance) * risk_pct)

            # Kullanıcının önerdiği formül:
            # qty = risk_per_trade / ATR (ATR fiyat cinsinden)
            # notional = qty * price = risk_per_trade * price / atr
            notional = risk_per_trade * (price_val / atr_val)

            # Güvenlik: notional bakiyeyi aşmasın
            if notional > float(account_balance):
                notional = float(account_balance)

            # Minimum işlem tutarı kontrolü (min_trade_amount)
            if notional < self.min_trade_amount:
                if float(account_balance) < self.min_trade_amount:
                    notional = float(account_balance)
                else:
                    notional = self.min_trade_amount

            return notional, {
                'method': 'atr_risk_based',
                'risk_pct': risk_pct,
                'risk_per_trade': risk_per_trade,
                'notional': notional,
                'atr': atr_val,
                'price': price_val
            }
        except Exception as e:
            self.logger.error(f"ATR risk bazlı pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_volatility_based_size(self, account_balance: float, signal: Dict[str, Any], 
                                       market_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Volatilite bazlı yöntem"""
        try:
            # Volatilite verilerini al
            technical_analysis = market_data.get('technical_analysis', {})
            atr = technical_analysis.get('atr', 0)
            current_price = market_data.get('price', 0)
            
            if atr == 0 or current_price == 0:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # ATR yüzdesi
            atr_percentage = atr / current_price
            
            # Volatilite hedefi ile karşılaştır
            if atr_percentage > self.volatility_target * 2:  # Çok yüksek volatilite
                volatility_multiplier = 0.5
            elif atr_percentage > self.volatility_target * 1.5:  # Yüksek volatilite
                volatility_multiplier = 0.7
            elif atr_percentage < self.volatility_target * 0.5:  # Düşük volatilite
                volatility_multiplier = 1.5
            else:  # Normal volatilite
                volatility_multiplier = 1.0
            
            # Sinyal gücü ile birleştir
            signal_strength = signal.get('strength', 0.5)
            base_percentage = self.fixed_percentage * volatility_multiplier * signal_strength
            
            # Limitler içinde tut
            base_percentage = max(self.min_position_size, 
                               min(base_percentage, self.max_position_size))
            
            position_size = account_balance * base_percentage
            
            return position_size, {
                'method': 'volatility_based',
                'percentage': base_percentage,
                'atr_percentage': atr_percentage,
                'volatility_multiplier': volatility_multiplier,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            self.logger.error(f"Volatilite bazlı pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_risk_parity_size(self, account_balance: float, signal: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Risk paritesi yöntemi"""
        try:
            # Risk bütçesini al
            risk_budget = self.risk_budget * account_balance
            
            # Stop loss mesafesini hesapla
            entry_price = signal.get('entry_price', market_data.get('price', 0))
            stop_loss = signal.get('stop_loss', 0)
            
            if stop_loss == 0 or entry_price == 0:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # Risk miktarı (stop loss mesafesi)
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = risk_amount / entry_price
            
            # Pozisyon büyüklüğü = Risk bütçesi / Risk miktarı
            position_size = risk_budget / risk_amount
            
            # Maksimum pozisyon büyüklüğü kontrolü
            max_size = account_balance * self.max_position_size
            position_size = min(position_size, max_size)
            
            return position_size, {
                'method': 'risk_parity',
                'position_size': position_size,
                'risk_budget': risk_budget,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage
            }
            
        except Exception as e:
            self.logger.error(f"Risk paritesi pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_martingale_size(self, account_balance: float, signal: Dict[str, Any], 
                                 historical_performance: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Martingale yöntemi"""
        try:
            # Son kayıpları al
            recent_losses = historical_performance.get('recent_losses', [])
            
            if not recent_losses:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # Ardışık kayıp sayısı
            consecutive_losses = 0
            for loss in reversed(recent_losses):
                if loss < 0:
                    consecutive_losses += 1
                else:
                    break
            
            # Martingale çarpanı (2^n)
            martingale_multiplier = 2 ** consecutive_losses
            
            # Maksimum çarpan sınırı
            max_multiplier = 8  # Maksimum 8x
            martingale_multiplier = min(martingale_multiplier, max_multiplier)
            
            # Temel pozisyon büyüklüğü
            base_size = account_balance * self.fixed_percentage
            position_size = base_size * martingale_multiplier
            
            # Maksimum pozisyon büyüklüğü kontrolü
            max_size = account_balance * self.max_position_size
            position_size = min(position_size, max_size)
            
            return position_size, {
                'method': 'martingale',
                'position_size': position_size,
                'consecutive_losses': consecutive_losses,
                'martingale_multiplier': martingale_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Martingale pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_anti_martingale_size(self, account_balance: float, signal: Dict[str, Any], 
                                      historical_performance: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Anti-Martingale yöntemi"""
        try:
            # Son kazançları al
            recent_trades = historical_performance.get('recent_trades', [])
            
            if not recent_trades:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # Ardışık kazanç sayısı
            consecutive_wins = 0
            for trade in reversed(recent_trades):
                if trade > 0:
                    consecutive_wins += 1
                else:
                    break
            
            # Anti-Martingale çarpanı (1.5^n)
            anti_martingale_multiplier = 1.5 ** consecutive_wins
            
            # Maksimum çarpan sınırı
            max_multiplier = 4  # Maksimum 4x
            anti_martingale_multiplier = min(anti_martingale_multiplier, max_multiplier)
            
            # Temel pozisyon büyüklüğü
            base_size = account_balance * self.fixed_percentage
            position_size = base_size * anti_martingale_multiplier
            
            # Maksimum pozisyon büyüklüğü kontrolü
            max_size = account_balance * self.max_position_size
            position_size = min(position_size, max_size)
            
            return position_size, {
                'method': 'anti_martingale',
                'position_size': position_size,
                'consecutive_wins': consecutive_wins,
                'anti_martingale_multiplier': anti_martingale_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Anti-Martingale pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def _calculate_signal_based_full_capital(self, account_balance: float, signal: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Sinyal bazlı tam sermaye kullanımı
        
        Mantık:
        1. Sinyal strength'e göre bakiyenin yüzdesini kullan (0.5 = %50, 1.0 = %100)
        2. Minimum işlem tutarı: 250 USD
        3. Eğer kalan bakiye < 250 USD ise, kalan bakiyenin tamamını kullan
        4. Sinyal ne kadar güçlüyse o kadar fazla yatırım yap
        """
        try:
            # Sinyal gücünü al (0.0 - 1.0 arası)
            signal_strength = signal.get('strength', 0.5)
            
            # Sinyal confidence varsa onu da kullan
            confidence = signal.get('confidence', 1.0)
            
            # Kombine güç: strength * confidence
            combined_strength = signal_strength * confidence
            
            # Sinyal gücüne göre kullanılacak sermaye yüzdesi
            # Minimum %25, maksimum %100
            if combined_strength >= 0.9:  # Çok güçlü sinyal
                capital_percentage = 1.0  # %100
            elif combined_strength >= 0.8:  # Güçlü sinyal
                capital_percentage = 0.8  # %80
            elif combined_strength >= 0.7:  # İyi sinyal
                capital_percentage = 0.6  # %60
            elif combined_strength >= 0.6:  # Orta sinyal
                capital_percentage = 0.5  # %50
            elif combined_strength >= 0.5:  # Zayıf sinyal
                capital_percentage = 0.35  # %35
            else:  # Çok zayıf sinyal
                capital_percentage = 0.25  # %25
            
            # Pozisyon büyüklüğünü hesapla
            position_size = account_balance * capital_percentage
            
            # KRİTİK: Minimum işlem tutarı kontrolü (250 USD)
            if position_size < self.min_trade_amount:
                # Eğer bakiye 250 USD'den azsa, tamamını kullan
                if account_balance < self.min_trade_amount:
                    position_size = account_balance
                    self.logger.info(f"Bakiye < {self.min_trade_amount} USD, tüm bakiye kullanılıyor: {position_size:.2f} USD")
                else:
                    # Bakiye yeterliyse minimum tutarı kullan
                    position_size = self.min_trade_amount
                    self.logger.info(f"Minimum işlem tutarı uygulandı: {self.min_trade_amount} USD")
            
            # Maksimum kontrolü (bakiyeden fazla olmasın)
            if position_size > account_balance:
                position_size = account_balance
            
            return position_size, {
                'method': 'signal_based_full_capital',
                'position_size': position_size,
                'account_balance': account_balance,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'combined_strength': combined_strength,
                'capital_percentage': capital_percentage,
                'percentage_used': (position_size / account_balance) * 100 if account_balance > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Sinyal bazlı tam sermaye hesaplama hatası: {e}")
            # Hata durumunda minimum tutarı döndür
            fallback_size = min(self.min_trade_amount, account_balance)
            return fallback_size, {'error': str(e), 'fallback': True}
    
    def calculate_optimal_position_size(self, 
                                      account_balance: float,
                                      signal: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      historical_performance: Dict[str, Any] = None,
                                      method_preference: List[PositionSizingMethod] = None) -> Tuple[float, Dict[str, Any]]:
        """Optimal pozisyon büyüklüğü hesapla (birden fazla yöntem karşılaştırması)"""
        try:
            if method_preference is None:
                method_preference = [
                    PositionSizingMethod.FIXED_FRACTIONAL,
                    PositionSizingMethod.VOLATILITY_BASED,
                    PositionSizingMethod.RISK_PARITY
                ]
            
            results = {}
            
            # Her yöntemi dene
            for method in method_preference:
                try:
                    size, details = self.calculate_position_size(
                        method, account_balance, signal, market_data, historical_performance
                    )
                    results[method.value] = {
                        'size': size,
                        'details': details
                    }
                except Exception as e:
                    self.logger.warning(f"Pozisyon büyüklüğü hesaplama hatası ({method.value}): {e}")
                    continue
            
            if not results:
                return self._calculate_fixed_fractional_size(account_balance, signal)
            
            # En konservatif (en küçük) pozisyon büyüklüğünü seç
            min_size = min(result['size'] for result in results.values())
            
            # Hangi yöntemden geldiğini bul
            selected_method = None
            for method, result in results.items():
                if result['size'] == min_size:
                    selected_method = method
                    break
            
            return min_size, {
                'method': 'optimal',
                'selected_method': selected_method,
                'all_results': results,
                'final_size': min_size
            }
            
        except Exception as e:
            self.logger.error(f"Optimal pozisyon büyüklüğü hesaplama hatası: {e}")
            return self._calculate_fixed_fractional_size(account_balance, signal)
    
    def update_parameters(self, **kwargs):
        """Parametreleri güncelle"""
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    self.logger.info(f"Parametre güncellendi: {key} = {value}")
                else:
                    self.logger.warning(f"Bilinmeyen parametre: {key}")
        except Exception as e:
            self.logger.error(f"Parametre güncelleme hatası: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Mevcut parametreleri al"""
        return {
            'default_method': self.default_method.value,
            'fixed_percentage': self.fixed_percentage,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'kelly_fraction': self.kelly_fraction,
            'min_win_rate': self.min_win_rate,
            'min_profit_factor': self.min_profit_factor,
            'volatility_target': self.volatility_target,
            'volatility_lookback': self.volatility_lookback,
            'risk_budget': self.risk_budget
        }

# Global pozisyon büyüklüğü hesaplayıcı
position_sizer = PositionSizer()

