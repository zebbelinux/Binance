from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from strategies.base_strategy import BaseStrategy
from risk_management.position_sizer import position_sizer, PositionSizingMethod

class TrendFollowingStrategy(BaseStrategy):
    """
    EMA crossover + ADX filter trend-following strategy.
    Expects 'technical_analysis' dict in market_data with keys:
    'ema_20', 'ema_50', 'adx', 'atr' (optional), 'atr_percent' (optional)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TrendFollowing", config)
        self.fast_ma = config.get('fast_ma', 20)
        self.slow_ma = config.get('slow_ma', 50)
        self.adx_th = config.get('adx_threshold', 25.0)
        self.atr_mult = config.get('atr_stop_multiplier', 2.5)
        self.tp_mult = config.get('atr_tp_multiplier', 2.5)
        self.min_signal_strength = config.get('min_signal_strength', 0.55)
        self.max_position_size = config.get('max_position_size', 0.10)  # fraction of portfolio
        # Confluence & Noise filtreleri
        self.confluence_threshold = config.get('confluence_threshold', 0.65)
        self.noise_min_atr_pct = config.get('noise_min_atr_pct', 0.001)
        self.noise_min_bb_width = config.get('noise_min_bb_width', 0.005)
        # Volatilite rejim eşiği: ATR yüzdesi bu eşikleri aşınca pozisyon küçülür
        self.volatility_high_pct = config.get('volatility_high_pct', 0.02)
        self.volatility_very_high_pct = config.get('volatility_very_high_pct', 0.03)

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        try:
            symbol = market_data.get('symbol', 'BTCTRY')
            price = float(market_data.get('price', 0.0) or 0.0)
            if price <= 0:
                return signals

            ta = market_data.get('technical_analysis', {}) or {}
            ema_fast = ta.get(f'ema_{self.fast_ma}', ta.get('ema_20'))
            ema_slow = ta.get(f'ema_{self.slow_ma}', ta.get('ema_50'))
            adx = float(ta.get('adx', 0.0) or 0.0)
            atr = ta.get('atr', None)
            atrp = ta.get('atr_percent', None)
            bb_upper = ta.get('bb_upper')
            bb_lower = ta.get('bb_lower')

            # if EMAs not prepared, skip
            if ema_fast is None or ema_slow is None:
                return signals

            # Noise / regime filter: ATR ve/veya BB genişliği yeterli değilse sinyal üretme
            # Özellikle BB width < threshold ise range kabul edilir ve trade edilmez.
            if not self._noise_ok(ta, price):
                return signals

            # Rejim filtresi: trend-following sadece trend varsa aktif
            # 1) ADX düşükse (ör: <20) trend yok say ve sinyal üretme
            low_trend_adx = 20.0
            if adx < low_trend_adx:
                return signals

            # 2) EMA fast > EMA slow ve ADX konfigüre eşik üzerinde olmalı
            if not (ema_fast > ema_slow and adx >= self.adx_th):
                return signals

            # ATR yoksa bu trend stratejisinde trade açma (sabit %SL fallback'ini kaldırıyoruz)
            if atr is None or float(atr or 0.0) <= 0:
                return signals

            # Confluence skoru hesapla
            conf_score = self._confluence_score(ta, self.fast_ma, self.slow_ma)

            # Volatilite rejimi: ATR yüzdesine göre confidence/pozisyon çarpanı belirle
            vol_conf = 1.0
            atr_pct_val = None
            try:
                if atrp is not None:
                    atr_pct_val = float(atrp)
                elif atr is not None and price > 0:
                    atr_pct_val = float(atr) / float(price)
            except Exception:
                atr_pct_val = None

            if atr_pct_val is not None:
                if atr_pct_val >= self.volatility_very_high_pct:
                    vol_conf = 0.5  # Çok yüksek volatilite: pozisyonu ciddi küçült
                elif atr_pct_val >= self.volatility_high_pct:
                    vol_conf = 0.7  # Yüksek volatilite: pozisyonu azalt

            # Long entry condition + confluence kapısı (EMA/ADX şartları yukarıda sağlandı)
            if conf_score is None or conf_score >= self.confluence_threshold:
                base_strength = (0.6 if adx < (self.adx_th + 5) else 0.8) * (
                    1.0 if conf_score is None else (0.9 + 0.2 * min(1.0, conf_score))
                )
                # Volatiliteye göre sinyal gücünü aşağı çek
                strength = max(0.0, min(0.95, base_strength * vol_conf))

                # Minimum strength altında kalan sinyalleri üretme
                if strength < float(self.min_signal_strength):
                    return signals

                stop = price - self.atr_mult * float(atr)
                take = price + self.tp_mult * float(atr)
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'strength': strength,
                    'atr': float(atr),
                    'price': price,
                    'confidence': vol_conf,
                    'entry_price': price,
                    'stop_loss': stop,
                    'take_profit': take,
                    'reason': f'EMA{self.fast_ma}>{self.slow_ma} & ADX {adx:.1f}',
                    'strategy': 'trend_following'
                })

            # Exit / shorting is optional; we keep it long-only for simplicity and safety.
            return signals
        except Exception as e:
            logging.getLogger(__name__).error(f"TrendFollowing generate_signals error: {e}")
            return signals

    def _confluence_score(self, ta: Dict[str, Any], fast: int, slow: int) -> float:
        """score = (rsi_strength*0.3) + (macd_momentum*0.4) + (adx_trend*0.3)"""
        try:
            rsi = float(ta.get('rsi', 50) or 50)
            macd = float(ta.get('macd', 0) or 0)
            macd_sig = float(ta.get('macd_signal', 0) or 0)
            macd_hist = float(ta.get('macd_histogram', macd - macd_sig) or (macd - macd_sig))
            adx = float(ta.get('adx', 15) or 15)
            rsi_strength = max(0.0, min(1.0, (rsi - 50) / 50.0))
            macd_momentum = max(0.0, min(1.0, abs(macd_hist) * 10.0))
            adx_trend = max(0.0, min(1.0, adx / 50.0))
            return float(rsi_strength*0.3 + macd_momentum*0.4 + adx_trend*0.3)
        except Exception:
            return None

    def _noise_ok(self, ta: Dict[str, Any], price: float) -> bool:
        """ATR veya BB genişliği eşiği sağlanmıyorsa trade üretme."""
        try:
            if price <= 0:
                return False
            atr = ta.get('atr')
            if atr is not None:
                try:
                    atr_pct = float(atr) / float(price)
                    if atr_pct < float(self.noise_min_atr_pct):
                        return False
                except Exception:
                    pass
            bu = ta.get('bb_upper'); bl = ta.get('bb_lower')
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

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla - Position Sizer kullanılarak"""
        try:
            position_size, details = position_sizer.calculate_position_size(
                method=PositionSizingMethod.ATR_RISK_BASED,
                account_balance=account_balance,
                signal=signal,
                market_data={}
            )
            return position_size
        except Exception as e:
            logging.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return min(250.0, account_balance)
