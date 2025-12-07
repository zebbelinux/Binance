"""
Correlation & Altseason Rotation Strategy
BTC dominansı ve korelasyon dinamiklerine göre altcoin rotasyonunu hedefler.
Not: Dış API'ler olmadan, market_data içindeki basit metrikleri kullanır (varsa).
"""
from typing import Dict, List, Any
import numpy as np
from strategies.base_strategy import BaseStrategy

class CorrelationRotationStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('correlation_rotation', config or {})
        self.config.setdefault('symbols', ['ETHUSDT','SOLUSDT','AVAXUSDT'])
        self.config.setdefault('min_strength', 0.55)
        self.config.setdefault('dominance_drop', 0.5)  # puan bazlı düşüş (ör: 52.3 -> 51.8 = 0.5)

    def _get_global_metrics(self, md: Dict[str, Any]):
        gm = md.get('global_metrics') if isinstance(md.get('global_metrics'), dict) else {}
        dom = gm.get('btc_dominance')
        dom_prev = gm.get('btc_dominance_prev')
        return (float(dom) if dom is not None else None, float(dom_prev) if dom_prev is not None else None)

    def _extract_closes(self, md: Dict[str, Any], symbol: str) -> List[float]:
        try:
            d = md.get(symbol)
            if isinstance(d, dict):
                closes = d.get('closes') or d.get('prices') or []
                if isinstance(closes, list):
                    return [float(x) for x in closes if x is not None][-120:]
        except Exception:
            pass
        return []

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        dom, dom_prev = self._get_global_metrics(market_data)
        if dom is None or dom_prev is None:
            return sigs
        drop_th = float(self.config.get('dominance_drop', 0.5))
        min_strength = float(self.config.get('min_strength', 0.55))
        if (dom_prev - dom) < drop_th:
            return sigs
        # BTC dominansı düşüyorsa, güçlü ivme gösteren altlarda al sinyali
        for sym in (self.config.get('symbols') or []):
            closes = self._extract_closes(market_data, sym)
            if len(closes) < 10:
                continue
            try:
                p_now, p_prev = closes[-1], closes[-6]
                mom = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0
                if mom > 0.01:  # son ~5 bar %1+ yükseliş
                    strength = min(1.0, 0.55 + mom * 10)
                    if strength >= min_strength:
                        sigs.append({'symbol': sym, 'side': 'buy', 'strength': float(strength), 'expected_pnl': 0.0, 'note': f'dom↓ {dom_prev:.2f}->{dom:.2f}, mom {mom:.4f}'})
            except Exception:
                continue
        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        frac = float(self.config.get('risk_fraction', 0.03))
        return max(0.0, account_balance * frac)
