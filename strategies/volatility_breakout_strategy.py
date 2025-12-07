"""
Volatility Breakout Strategy
Basit Bollinger/volatilite genişlemesi ile kırılım sinyali üretir.
"""
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from strategies.base_strategy import BaseStrategy

class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('volatility_breakout', config or {})
        # Varsayılan parametreler
        self.config.setdefault('bb_period', 20)
        self.config.setdefault('bb_std', 2.0)
        self.config.setdefault('min_strength', 0.55)
        self.config.setdefault('symbols', ['BTCUSDT', 'ETHUSDT'])

    def _extract_closes(self, market_data: Dict[str, Any], symbol: str) -> List[float]:
        # Beklenen şemayı bilmiyoruz; güvenli çıkarım yapıyoruz
        # Öncelik: market_data[symbol]['closes'] veya ['prices']
        try:
            symd = market_data.get(symbol) if isinstance(market_data.get(symbol), dict) else None
            if symd:
                for k in ('closes', 'prices', 'close'):
                    v = symd.get(k)
                    if isinstance(v, list) and len(v) >= 25:
                        return [float(x) for x in v if x is not None]
            # Alternatif: market_data['data'][symbol]
            d = market_data.get('data')
            if isinstance(d, dict):
                symd = d.get(symbol)
                if isinstance(symd, dict):
                    for k in ('closes', 'prices', 'close'):
                        v = symd.get(k)
                        if isinstance(v, list) and len(v) >= 25:
                            return [float(x) for x in v if x is not None]
            # Son çare: market_data içinde herhangi bir dict'te closes
            for v in market_data.values():
                if isinstance(v, dict):
                    for k in ('closes', 'prices', 'close'):
                        arr = v.get(k)
                        if isinstance(arr, list) and len(arr) >= 25:
                            return [float(x) for x in arr if x is not None]
        except Exception:
            pass
        return []

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        try:
            symbols = self.config.get('symbols') or []
            period = int(self.config.get('bb_period', 20))
            bb_std = float(self.config.get('bb_std', 2.0))
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

                # Kırılım doğrulaması: bar kapanışı bant dışına çıkmalı ve ivme olmalı
                if c_now > upper and c_prev <= upper:
                    strength = min(1.0, (c_now - upper) / (std + 1e-9) * 0.1 + 0.6)
                    if strength >= min_strength:
                        signals.append({
                            'symbol': sym,
                            'side': 'buy',
                            'strength': float(strength),
                            'expected_pnl': 0.0,
                            'note': 'BB breakout up'
                        })
                elif c_now < lower and c_prev >= lower:
                    strength = min(1.0, (lower - c_now) / (std + 1e-9) * 0.1 + 0.6)
                    if strength >= min_strength:
                        signals.append({
                            'symbol': sym,
                            'side': 'sell',
                            'strength': float(strength),
                            'expected_pnl': 0.0,
                            'note': 'BB breakout down'
                        })
        except Exception:
            pass
        return signals

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        # Basit risk: %3 sermaye
        try:
            risk_fraction = float(self.config.get('risk_fraction', 0.03))
        except Exception:
            risk_fraction = 0.03
        return max(0.0, account_balance * risk_fraction)
