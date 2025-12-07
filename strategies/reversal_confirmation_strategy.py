"""
Reversal Confirmation Strategy
RSI divergence + MA crossover kombinasyonu ile dönüş sinyali.
"""
from typing import Dict, List, Any
import numpy as np
from strategies.base_strategy import BaseStrategy

class ReversalConfirmationStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('reversal_confirmation', config or {})
        self.config.setdefault('ma_fast', 9)
        self.config.setdefault('ma_slow', 21)
        self.config.setdefault('rsi_period', 14)
        self.config.setdefault('min_strength', 0.55)
        self.config.setdefault('symbols', ['BTCUSDT'])

    def _extract_closes(self, md: Dict[str, Any], symbol: str) -> List[float]:
        try:
            d = md.get(symbol)
            if isinstance(d, dict):
                closes = d.get('closes') or d.get('prices') or []
                if isinstance(closes, list):
                    return [float(x) for x in closes if x is not None][-200:]
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

    def _ma(self, prices: List[float], n: int) -> float:
        if len(prices) < n:
            return float(np.mean(prices)) if prices else 0.0
        return float(np.mean(prices[-n:]))

    def _has_bullish_divergence(self, closes: List[float], rsi_period: int) -> bool:
        if len(closes) < rsi_period + 20:
            return False
        p = closes[-20:]
        rsi_vals = []
        for i in range(20 - rsi_period, 20):
            rsi_vals.append(self._rsi(closes[:len(closes)-20+i+1], rsi_period))
        try:
            price_low_now = min(p[-5:])
            price_low_prev = min(p[:5])
            rsi_now = float(rsi_vals[-1]) if rsi_vals else 50.0
            rsi_prev = float(rsi_vals[0]) if rsi_vals else 50.0
            return (price_low_now < price_low_prev) and (rsi_now > rsi_prev)
        except Exception:
            return False

    def _has_bearish_divergence(self, closes: List[float], rsi_period: int) -> bool:
        if len(closes) < rsi_period + 20:
            return False
        p = closes[-20:]
        rsi_vals = []
        for i in range(20 - rsi_period, 20):
            rsi_vals.append(self._rsi(closes[:len(closes)-20+i+1], rsi_period))
        try:
            price_high_now = max(p[-5:])
            price_high_prev = max(p[:5])
            rsi_now = float(rsi_vals[-1]) if rsi_vals else 50.0
            rsi_prev = float(rsi_vals[0]) if rsi_vals else 50.0
            return (price_high_now > price_high_prev) and (rsi_now < rsi_prev)
        except Exception:
            return False

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        syms = self.config.get('symbols') or []
        fast = int(self.config.get('ma_fast', 9))
        slow = int(self.config.get('ma_slow', 21))
        rsi_p = int(self.config.get('rsi_period', 14))
        min_strength = float(self.config.get('min_strength', 0.55))
        for sym in syms:
            closes = self._extract_closes(market_data, sym)
            if len(closes) < max(slow+5, rsi_p+20):
                continue
            ma_fast_prev = self._ma(closes[:-1], fast)
            ma_slow_prev = self._ma(closes[:-1], slow)
            ma_fast_now = self._ma(closes, fast)
            ma_slow_now = self._ma(closes, slow)
            bull_cross = ma_fast_prev <= ma_slow_prev and ma_fast_now > ma_slow_now
            bear_cross = ma_fast_prev >= ma_slow_prev and ma_fast_now < ma_slow_now
            if bull_cross and self._has_bullish_divergence(closes, rsi_p):
                strength = 0.6
                if strength >= min_strength:
                    sigs.append({'symbol': sym, 'side': 'buy', 'strength': strength, 'expected_pnl': 0.0, 'note': 'RSI div + MA bull cross'})
            if bear_cross and self._has_bearish_divergence(closes, rsi_p):
                strength = 0.6
                if strength >= min_strength:
                    sigs.append({'symbol': sym, 'side': 'sell', 'strength': strength, 'expected_pnl': 0.0, 'note': 'RSI div + MA bear cross'})
        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        frac = float(self.config.get('risk_fraction', 0.02))
        return max(0.0, account_balance * frac)
