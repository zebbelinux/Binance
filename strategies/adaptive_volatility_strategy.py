"""
Adaptive Volatility Strategy (Meta)
Volatilite rejimine göre basit yönlü/ters yönlü sinyal üretir.
Not: Meta-strateji kurgusu; gerçek geçiş mekanizması StrategyManager/Selector ile entegre edilebilir.
"""
from typing import Dict, List, Any
import numpy as np
from strategies.base_strategy import BaseStrategy

class AdaptiveVolatilityStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('adaptive_volatility', config or {})
        self.config.setdefault('vol_window', 30)
        self.config.setdefault('high_vol', 0.03)   # %3
        self.config.setdefault('low_vol', 0.01)    # %1
        self.config.setdefault('symbols', ['BTCUSDT'])
        self.config.setdefault('min_strength', 0.55)

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

    def _volatility(self, closes: List[float], window: int) -> float:
        if len(closes) < window+1:
            return 0.0
        arr = np.array(closes[-(window+1):], dtype=float)
        rets = np.diff(arr) / arr[:-1]
        return float(np.std(rets))

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        w = int(self.config.get('vol_window', 30))
        high_v = float(self.config.get('high_vol', 0.03))
        low_v = float(self.config.get('low_vol', 0.01))
        min_strength = float(self.config.get('min_strength', 0.55))
        for sym in (self.config.get('symbols') or []):
            closes = self._extract_closes(market_data, sym)
            if len(closes) < w+2:
                continue
            vol = self._volatility(closes, w)
            # Basit kural: yüksek vol -> kısa süreli yön (scalping benzeri) => son momentum yönünde
            # düşük vol -> mean reversion eğilimli => son barın ters yönü
            p_now, p_prev = closes[-1], closes[-2]
            mom = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0
            if vol >= high_v:
                side = 'buy' if mom > 0 else 'sell'
                strength = min(1.0, 0.6 + min(0.4, abs(mom)*50))
            elif vol <= low_v:
                side = 'sell' if mom > 0 else 'buy'
                strength = 0.6
            else:
                continue
            if strength >= min_strength:
                sigs.append({'symbol': sym, 'side': side, 'strength': float(strength), 'expected_pnl': 0.0, 'note': f'vol {vol:.4f}, mom {mom:.4f}'})
        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        frac = float(self.config.get('risk_fraction', 0.02))
        return max(0.0, account_balance * frac)
