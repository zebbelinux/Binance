"""
Momentum Ignition / Acceleration Strategy
Kısa periyotta hacim ve fiyat ivmesi sıçramalarını tespit eder.
"""
from typing import Dict, List, Any
import numpy as np
from strategies.base_strategy import BaseStrategy

class MomentumIgnitionStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('momentum_ignition', config or {})
        self.config.setdefault('lookback_secs', 60)   # kısa pencere
        self.config.setdefault('volume_spike', 2.0)   # ortalamanın katı
        self.config.setdefault('price_momentum', 0.002)  # %0.2
        self.config.setdefault('symbols', ['BTCUSDT'])
        self.config.setdefault('min_strength', 0.55)

    def _extract_series(self, market_data: Dict[str, Any], symbol: str):
        # Beklenen anahtarlar: prices, volumes (kısa pencere)
        try:
            d = market_data.get(symbol)
            if isinstance(d, dict):
                prices = d.get('prices') or d.get('closes') or []
                vols = d.get('volumes') or []
                return prices[-120:], vols[-120:]
        except Exception:
            pass
        return [], []

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        symbols = self.config.get('symbols') or []
        vol_spike = float(self.config.get('volume_spike', 2.0))
        mom_th = float(self.config.get('price_momentum', 0.002))
        min_strength = float(self.config.get('min_strength', 0.55))
        for sym in symbols:
            prices, vols = self._extract_series(market_data, sym)
            if len(prices) < 10 or len(vols) < 10:
                continue
            try:
                arrp = np.array(prices[-30:], dtype=float)
                arrv = np.array(vols[-30:], dtype=float)
                if len(arrp) < 5:
                    continue
                p_now, p_prev = float(arrp[-1]), float(arrp[-5])
                mom = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0
                v_now = float(arrv[-1])
                v_avg = float(np.mean(arrv[:-1])) if len(arrv) > 1 else 0.0
                spike = (v_now / (v_avg + 1e-9)) if v_avg > 0 else 0.0
                if spike >= vol_spike and abs(mom) >= mom_th:
                    side = 'buy' if mom > 0 else 'sell'
                    strength = min(1.0, 0.5 + abs(mom) * 50.0)
                    if strength >= min_strength:
                        sigs.append({
                            'symbol': sym,
                            'side': side,
                            'strength': float(strength),
                            'expected_pnl': 0.0,
                            'note': f'vol_spike x{spike:.2f}, mom {mom:.4f}'
                        })
            except Exception:
                continue
        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        frac = float(self.config.get('risk_fraction', 0.015))
        return max(0.0, account_balance * frac)
