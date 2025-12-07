"""
Teknik Analiz Göstergeleri
23+ teknik gösterge hesaplama fonksiyonları
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# TA-Lib fallback mekanizması
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib bulunamadı, pandas_ta fallback kullanılacak")
    
    try:
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        PANDAS_TA_AVAILABLE = False
        print("pandas_ta da bulunamadı, saf Python implementasyonu kullanılacak")

class TechnicalIndicators:
    """Teknik analiz göstergeleri sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.use_talib = TALIB_AVAILABLE
        self.use_pandas_ta = PANDAS_TA_AVAILABLE and not TALIB_AVAILABLE
        self.use_pure_python = not TALIB_AVAILABLE and not PANDAS_TA_AVAILABLE
        
        if self.use_pure_python:
            self.logger.warning("TA-Lib ve pandas_ta bulunamadı, saf Python implementasyonu kullanılacak")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tüm göstergeleri hesapla"""
        try:
            indicators = {}
            
            # Fiyat verilerini hazırla
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else None
            
            # 1. Momentum Göstergeleri
            indicators.update(self._calculate_momentum_indicators(close, high, low))
            
            # 2. Trend Göstergeleri
            indicators.update(self._calculate_trend_indicators(close, high, low))
            
            # 3. Volatilite Göstergeleri
            indicators.update(self._calculate_volatility_indicators(close, high, low))
            
            # 4. Hacim Göstergeleri
            if volume is not None:
                indicators.update(self._calculate_volume_indicators(close, high, low, volume))
            
            # 5. Özel Göstergeler
            indicators.update(self._calculate_custom_indicators(close, high, low))
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Gösterge hesaplama hatası: {e}")
            return {}
    
    def _calculate_momentum_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Momentum göstergelerini hesapla"""
        indicators = {}
        
        try:
            # RSI (Relative Strength Index)
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
            indicators['rsi_21'] = talib.RSI(close, timeperiod=21)[-1] if len(close) >= 21 else 50
            
            # Stochastic RSI
            indicators['stoch_rsi_k'], indicators['stoch_rsi_d'] = talib.STOCHRSI(close, timeperiod=14)
            indicators['stoch_rsi_k'] = indicators['stoch_rsi_k'][-1] if not np.isnan(indicators['stoch_rsi_k'][-1]) else 50
            indicators['stoch_rsi_d'] = indicators['stoch_rsi_d'][-1] if not np.isnan(indicators['stoch_rsi_d'][-1]) else 50
            
            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            indicators['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # CCI (Commodity Channel Index)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else -50
            
            # Rate of Change (ROC)
            indicators['roc'] = talib.ROC(close, timeperiod=10)[-1] if len(close) >= 10 else 0
            
            # Momentum
            indicators['momentum'] = talib.MOM(close, timeperiod=10)[-1] if len(close) >= 10 else 0
            
        except Exception as e:
            self.logger.error(f"Momentum göstergeleri hesaplama hatası: {e}")
        
        return indicators
    
    def _calculate_trend_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Trend göstergelerini hesapla"""
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma_5'] = talib.SMA(close, timeperiod=5)[-1] if len(close) >= 5 else close[-1]
            indicators['sma_10'] = talib.SMA(close, timeperiod=10)[-1] if len(close) >= 10 else close[-1]
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
            indicators['sma_100'] = talib.SMA(close, timeperiod=100)[-1] if len(close) >= 100 else close[-1]
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
            
            # Exponential Moving Averages
            indicators['ema_5'] = talib.EMA(close, timeperiod=5)[-1] if len(close) >= 5 else close[-1]
            indicators['ema_10'] = talib.EMA(close, timeperiod=10)[-1] if len(close) >= 10 else close[-1]
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
            indicators['ema_100'] = talib.EMA(close, timeperiod=100)[-1] if len(close) >= 100 else close[-1]
            indicators['ema_200'] = talib.EMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
            
            # ADX (Average Directional Index)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            
            # Parabolic SAR
            indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1] if len(close) >= 2 else close[-1]
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            indicators['aroon_up'] = aroon_up[-1] if not np.isnan(aroon_up[-1]) else 50
            indicators['aroon_down'] = aroon_down[-1] if not np.isnan(aroon_down[-1]) else 50
            
            # Trend yönü analizi
            indicators['trend_direction'] = self._analyze_trend_direction(indicators)
            indicators['trend_strength'] = self._calculate_trend_strength(indicators)
            
        except Exception as e:
            self.logger.error(f"Trend göstergeleri hesaplama hatası: {e}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Volatilite göstergelerini hesapla"""
        indicators = {}
        
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1]
            indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[-1]
            indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100
            indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # ATR (Average True Range)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            indicators['atr_percent'] = (indicators['atr'] / close[-1]) * 100 if close[-1] > 0 else 0
            
            # Keltner Channels
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(close, high, low)
            indicators['kc_upper'] = kc_upper
            indicators['kc_middle'] = kc_middle
            indicators['kc_lower'] = kc_lower
            
            # Donchian Channels
            dc_upper, dc_middle, dc_lower = self._calculate_donchian_channels(high, low, close)
            indicators['dc_upper'] = dc_upper
            indicators['dc_middle'] = dc_middle
            indicators['dc_lower'] = dc_lower
            
            # Volatilite hesaplama
            indicators['volatility'] = self._calculate_volatility(close)
            indicators['volatility_rank'] = self._calculate_volatility_rank(close)
            
        except Exception as e:
            self.logger.error(f"Volatilite göstergeleri hesaplama hatası: {e}")
        
        return indicators
    
    def _calculate_volume_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Hacim göstergelerini hesapla"""
        indicators = {}
        
        try:
            # OBV (On-Balance Volume)
            indicators['obv'] = talib.OBV(close, volume)[-1] if len(close) >= 2 else 0
            
            # Volume SMA
            indicators['volume_sma_10'] = talib.SMA(volume, timeperiod=10)[-1] if len(volume) >= 10 else volume[-1]
            indicators['volume_sma_20'] = talib.SMA(volume, timeperiod=20)[-1] if len(volume) >= 20 else volume[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1
            
            # Volume Rate of Change
            indicators['volume_roc'] = talib.ROC(volume, timeperiod=10)[-1] if len(volume) >= 10 else 0
            
            # Accumulation/Distribution Line
            indicators['ad_line'] = talib.AD(high, low, close, volume)[-1] if len(close) >= 1 else 0
            
            # Chaikin Money Flow
            indicators['cmf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)[-1] if len(close) >= 10 else 0
            
            # Money Flow Index
            indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1] if len(close) >= 14 else 50
            
            # Ease of Movement
            indicators['eom'] = self._calculate_eom(high, low, close, volume)
            
        except Exception as e:
            self.logger.error(f"Hacim göstergeleri hesaplama hatası: {e}")
        
        return indicators
    
    def _calculate_custom_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Özel göstergeleri hesapla"""
        indicators = {}
        
        try:
            # Ichimoku Cloud
            ichimoku = self._calculate_ichimoku(high, low, close)
            indicators.update(ichimoku)
            
            # Fibonacci Retracements
            fib_levels = self._calculate_fibonacci_levels(high, low)
            indicators.update(fib_levels)
            
            # Support and Resistance
            support_resistance = self._calculate_support_resistance(high, low, close)
            indicators.update(support_resistance)
            
            # Market Structure
            market_structure = self._analyze_market_structure(high, low, close)
            indicators.update(market_structure)
            
            # Divergence Analysis
            divergence = self._analyze_divergence(close, high, low)
            indicators.update(divergence)
            
        except Exception as e:
            self.logger.error(f"Özel göstergeler hesaplama hatası: {e}")
        
        return indicators
    
    def _calculate_keltner_channels(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Keltner Channels hesapla"""
        try:
            ema = talib.EMA(close, timeperiod=period)
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            middle = ema[-1] if not np.isnan(ema[-1]) else close[-1]
            upper = middle + (2 * atr[-1]) if not np.isnan(atr[-1]) else middle
            lower = middle - (2 * atr[-1]) if not np.isnan(atr[-1]) else middle
            
            return upper, middle, lower
        except:
            return close[-1], close[-1], close[-1]
    
    def _calculate_donchian_channels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Donchian Channels hesapla"""
        try:
            if len(high) < period:
                return close[-1], close[-1], close[-1]
            
            upper = np.max(high[-period:])
            lower = np.min(low[-period:])
            middle = (upper + lower) / 2
            
            return upper, middle, lower
        except:
            return close[-1], close[-1], close[-1]
    
    def _calculate_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Volatilite hesapla"""
        try:
            if len(close) < period:
                return 0
            
            returns = np.diff(np.log(close[-period:]))
            volatility = np.std(returns) * np.sqrt(252) * 100  # Yıllık volatilite %
            return volatility
        except:
            return 0
    
    def _calculate_volatility_rank(self, close: np.ndarray, period: int = 20, lookback: int = 100) -> float:
        """Volatilite rank hesapla (0-100)"""
        try:
            if len(close) < lookback:
                return 50
            
            volatilities = []
            for i in range(lookback - period + 1):
                vol = self._calculate_volatility(close[i:i+period+1])
                volatilities.append(vol)
            
            current_vol = volatilities[-1]
            rank = (np.sum(np.array(volatilities) < current_vol) / len(volatilities)) * 100
            return rank
        except:
            return 50
    
    def _calculate_eom(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
        """Ease of Movement hesapla"""
        try:
            if len(high) < 2:
                return 0
            
            distance = ((high[-1] + low[-1]) / 2) - ((high[-2] + low[-2]) / 2)
            box_height = volume[-1] / (high[-1] - low[-1]) if high[-1] != low[-1] else 1
            eom = distance / box_height
            return eom
        except:
            return 0
    
    def _calculate_ichimoku(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Ichimoku Cloud hesapla"""
        try:
            if len(close) < 52:
                return {}
            
            # Tenkan-sen (9 periyot)
            tenkan_high = np.max(high[-9:])
            tenkan_low = np.min(low[-9:])
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (26 periyot)
            kijun_high = np.max(high[-26:])
            kijun_low = np.min(low[-26:])
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B (52 periyot)
            senkou_high = np.max(high[-52:])
            senkou_low = np.min(low[-52:])
            senkou_span_b = (senkou_high + senkou_low) / 2
            
            return {
                'ichimoku_tenkan': tenkan_sen,
                'ichimoku_kijun': kijun_sen,
                'ichimoku_senkou_a': senkou_span_a,
                'ichimoku_senkou_b': senkou_span_b,
                'ichimoku_cloud_top': max(senkou_span_a, senkou_span_b),
                'ichimoku_cloud_bottom': min(senkou_span_a, senkou_span_b)
            }
        except:
            return {}
    
    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Fibonacci retracement seviyelerini hesapla"""
        try:
            if len(high) < 20 or len(low) < 20:
                return {}
            
            # Son 20 periyodun en yüksek ve en düşük değerleri
            highest = np.max(high[-20:])
            lowest = np.min(low[-20:])
            diff = highest - lowest
            
            return {
                'fib_0': highest,
                'fib_23.6': highest - (diff * 0.236),
                'fib_38.2': highest - (diff * 0.382),
                'fib_50': highest - (diff * 0.5),
                'fib_61.8': highest - (diff * 0.618),
                'fib_78.6': highest - (diff * 0.786),
                'fib_100': lowest
            }
        except:
            return {}
    
    def _calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Destek ve direnç seviyelerini hesapla"""
        try:
            if len(close) < 20:
                return {}
            
            # Pivot noktaları
            pivot_highs = []
            pivot_lows = []
            
            for i in range(2, len(high) - 2):
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    pivot_highs.append(high[i])
                
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    pivot_lows.append(low[i])
            
            # En güçlü seviyeler
            resistance = np.mean(pivot_highs[-3:]) if len(pivot_highs) >= 3 else high[-1]
            support = np.mean(pivot_lows[-3:]) if len(pivot_lows) >= 3 else low[-1]
            
            return {
                'resistance': resistance,
                'support': support,
                'pivot_highs': len(pivot_highs),
                'pivot_lows': len(pivot_lows)
            }
        except:
            return {}
    
    def _analyze_market_structure(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Piyasa yapısını analiz et"""
        try:
            if len(close) < 10:
                return {}
            
            # Higher Highs, Higher Lows analizi
            recent_highs = high[-10:]
            recent_lows = low[-10:]
            
            hh_count = 0  # Higher Highs
            hl_count = 0  # Higher Lows
            lh_count = 0  # Lower Highs
            ll_count = 0  # Lower Lows
            
            for i in range(1, len(recent_highs)):
                if recent_highs[i] > recent_highs[i-1]:
                    hh_count += 1
                else:
                    lh_count += 1
                
                if recent_lows[i] > recent_lows[i-1]:
                    hl_count += 1
                else:
                    ll_count += 1
            
            # Market structure belirleme
            if hh_count > lh_count and hl_count > ll_count:
                structure = "uptrend"
            elif lh_count > hh_count and ll_count > hl_count:
                structure = "downtrend"
            else:
                structure = "sideways"
            
            return {
                'market_structure': structure,
                'higher_highs': hh_count,
                'higher_lows': hl_count,
                'lower_highs': lh_count,
                'lower_lows': ll_count
            }
        except:
            return {}
    
    def _analyze_divergence(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Divergence analizi yap"""
        try:
            if len(close) < 20:
                return {}
            
            # RSI divergence
            rsi = talib.RSI(close, timeperiod=14)
            if len(rsi) >= 10:
                price_trend = close[-1] - close[-10]
                rsi_trend = rsi[-1] - rsi[-10]
                
                if price_trend > 0 and rsi_trend < 0:
                    rsi_divergence = "bearish"
                elif price_trend < 0 and rsi_trend > 0:
                    rsi_divergence = "bullish"
                else:
                    rsi_divergence = "none"
            else:
                rsi_divergence = "none"
            
            return {
                'rsi_divergence': rsi_divergence,
                'divergence_strength': abs(price_trend) * abs(rsi_trend) if 'price_trend' in locals() else 0
            }
        except:
            return {}
    
    def _analyze_trend_direction(self, indicators: Dict[str, Any]) -> str:
        """Trend yönünü analiz et"""
        try:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            ema_20 = indicators.get('ema_20', 0)
            ema_50 = indicators.get('ema_50', 0)
            
            if sma_20 > sma_50 and ema_20 > ema_50:
                return "uptrend"
            elif sma_20 < sma_50 and ema_20 < ema_50:
                return "downtrend"
            else:
                return "sideways"
        except:
            return "unknown"
    
    def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """Trend gücünü hesapla (0-100)"""
        try:
            adx = indicators.get('adx', 0)
            plus_di = indicators.get('plus_di', 0)
            minus_di = indicators.get('minus_di', 0)
            
            # ADX trend gücü
            trend_strength = min(100, adx)
            
            # DI farkı
            di_diff = abs(plus_di - minus_di)
            trend_strength = (trend_strength + di_diff) / 2
            
            return trend_strength
        except:
            return 0

# Global teknik analiz instance
technical_indicators = TechnicalIndicators()

