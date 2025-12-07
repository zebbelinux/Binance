"""
Mean Reversion Strategy (Bollinger/RSI Combo)
Bollinger bandının dışına taşma + RSI aşırı bölge kombinasyonu ile ters yönlü sinyal.
"""
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('mean_reversion', config or {})
        # Varsayılan parametreler
        self.config.setdefault('bb_period', 20)
        self.config.setdefault('bb_std', 2.0)
        self.config.setdefault('rsi_period', 14)
        self.config.setdefault('rsi_overbought', 70)
        self.config.setdefault('rsi_oversold', 30)
        self.config.setdefault('min_strength', 0.55)
        self.config.setdefault('symbols', ['BTCTRY', 'ETHTRY'])

    def _extract_closes(self, market_data: Dict[str, Any], symbol: str) -> List[float]:
        try:
            symd = market_data.get(symbol) if isinstance(market_data.get(symbol), dict) else None
            if symd:
                for k in ('closes', 'prices', 'close'):
                    v = symd.get(k)
                    if isinstance(v, list) and len(v) >= 30:
                        return [float(x) for x in v if x is not None]
            d = market_data.get('data')
            if isinstance(d, dict):
                symd = d.get(symbol)
                if isinstance(symd, dict):
                    for k in ('closes', 'prices', 'close'):
                        v = symd.get(k)
                        if isinstance(v, list) and len(v) >= 30:
                            return [float(x) for x in v if x is not None]
            for v in market_data.values():
                if isinstance(v, dict):
                    for k in ('closes', 'prices', 'close'):
                        arr = v.get(k)
                        if isinstance(arr, list) and len(arr) >= 30:
                            return [float(x) for x in arr if x is not None]
        except Exception:
            pass
        return []

    def _rsi(self, prices: List[float], period: int) -> float:
        if len(prices) <= period:
            return 50.0
        arr = np.array(prices[-(period+1):], dtype=float)
        diff = np.diff(arr)
        up = np.where(diff > 0, diff, 0.0)
        down = np.where(diff < 0, -diff, 0.0)
        avg_gain = np.mean(up[-period:])
        avg_loss = np.mean(down[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        try:
            symbols = self.config.get('symbols') or []
            period = int(self.config.get('bb_period', 20))
            bb_std = float(self.config.get('bb_std', 2.0))
            rsi_period = int(self.config.get('rsi_period', 14))
            rsi_overbought = float(self.config.get('rsi_overbought', 70))
            rsi_oversold = float(self.config.get('rsi_oversold', 30))
            min_strength = float(self.config.get('min_strength', 0.55))

            for sym in symbols:
                closes = self._extract_closes(market_data, sym)
                if len(closes) < period + 2:
                    continue
                arr = np.array(closes[-(period+2):], dtype=float)
                ma = np.mean(arr[-period:])
                std = np.std(arr[-period:])
                upper = ma + bb_std * std
                lower = ma - bb_std * std
                c_prev = float(arr[-2])
                c_now = float(arr[-1])
                rsi_now = float(self._rsi(closes, rsi_period))

                # Aşağı bandın dışına taşma + RSI aşırı satım => yukarı yönlü mean reversion
                if c_now < lower and rsi_now <= rsi_oversold:
                    strength = min(1.0, (lower - c_now) / (std + 1e-9) * 0.1 + 0.6)
                    if strength >= min_strength:
                        signals.append({
                            'symbol': sym,
                            'side': 'buy',
                            'strength': float(strength),
                            'expected_pnl': 0.0,
                            'note': 'BB lower + RSI oversold'
                        })
                # Üst bandın dışına taşma + RSI aşırı alım => aşağı yönlü mean reversion
                elif c_now > upper and rsi_now >= rsi_overbought:
                    strength = min(1.0, (c_now - upper) / (std + 1e-9) * 0.1 + 0.6)
                    if strength >= min_strength:
                        signals.append({
                            'symbol': sym,
                            'side': 'sell',
                            'strength': float(strength),
                            'expected_pnl': 0.0,
                            'note': 'BB upper + RSI overbought'
                        })
        except Exception:
            pass
        return signals

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        # Basit risk: %2 sermaye
        try:
            risk_fraction = float(self.config.get('risk_fraction', 0.02))
        except Exception:
            risk_fraction = 0.02
        return max(0.0, account_balance * risk_fraction)
