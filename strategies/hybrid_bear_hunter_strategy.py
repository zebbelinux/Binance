from typing import Dict, List, Any
from datetime import datetime

import numpy as np

from strategies.base_strategy import BaseStrategy


class HybridBearHunterStrategy(BaseStrategy):
    """USDT pariteleri için ayı piyasası odaklı karma strateji.

    Mantık (özet):
    - Global ayı filtresi: Fiyat < MA200 ve MA50 < MA200.
    - Üç alt modül aynı anda çalışır, uygun olan SELL sinyalini üretir:
      1) Trend pullback: Güçlü düşüş trendinde fiyatın MA20'ye doğru tepki verip tekrar zayıflaması.
      2) RSI re-entry: Aşırı satımdan gelen zayıf toparlanmaların tekrar aşağı dönmesi.
      3) BB breakdown: Dar bant sonrası aşağı yönlü kırılım ve momentum.
    - ATR benzeri volatilite tahmini ile SL/TP yüzdeleri ve pozisyon gücü kalibre edilir.

    Not: Bu strateji SELL sinyali üretir. Spot kağıt hesapta, fiilen sadece
    mevcut long pozisyonları azaltmak/kapatmak için kullanılacaktır.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__("hybrid_bear_hunter", config or {})
        cfg = self.config
        # Hedef semboller: yalnızca USDT pariteleri
        cfg.setdefault("symbols", ["BTCUSDT", "ETHUSDT"])
        # MA parametreleri
        cfg.setdefault("fast_ma", 20)
        cfg.setdefault("mid_ma", 50)
        cfg.setdefault("slow_ma", 200)
        # RSI parametreleri
        cfg.setdefault("rsi_period", 14)
        cfg.setdefault("rsi_oversold", 30.0)
        cfg.setdefault("rsi_reentry_low", 40.0)
        cfg.setdefault("rsi_reentry_high", 55.0)
        # Bollinger / volatilite
        cfg.setdefault("bb_period", 20)
        cfg.setdefault("bb_std", 2.0)
        # Volatilite penceresi (ATR benzeri)
        cfg.setdefault("vol_window", 20)
        # Minimum sinyal gücü
        cfg.setdefault("min_strength", 0.55)

    # --- Yardımcılar ---
    def _extract_closes(self, market_data: Dict[str, Any], symbol: str) -> List[float]:
        """Bu projedeki diğer stratejilerle uyumlu şekilde kapanış serisini bul."""
        try:
            d = market_data.get(symbol)
            if isinstance(d, dict):
                for k in ("closes", "prices", "close"):
                    v = d.get(k)
                    if isinstance(v, list) and len(v) >= 30:
                        return [float(x) for x in v if x is not None]
            droot = market_data.get("data")
            if isinstance(droot, dict):
                sd = droot.get(symbol)
                if isinstance(sd, dict):
                    for k in ("closes", "prices", "close"):
                        v = sd.get(k)
                        if isinstance(v, list) and len(v) >= 30:
                            return [float(x) for x in v if x is not None]
        except Exception:
            pass
        return []

    def _sma(self, arr: np.ndarray, win: int) -> float:
        if len(arr) < win:
            return float(arr[-1]) if len(arr) else 0.0
        return float(np.mean(arr[-win:]))

    def _rsi(self, prices: List[float], period: int) -> float:
        if len(prices) <= period:
            return 50.0
        arr = np.array(prices[-(period + 1):], dtype=float)
        diff = np.diff(arr)
        up = np.where(diff > 0, diff, 0.0)
        down = np.where(diff < 0, -diff, 0.0)
        avg_gain = np.mean(up[-period:])
        avg_loss = np.mean(down[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _volatility_pct(self, closes: List[float], window: int) -> float:
        if len(closes) < window + 1:
            return 0.0
        arr = np.array(closes[-(window + 1):], dtype=float)
        rets = np.diff(arr) / arr[:-1]
        return float(np.std(rets))

    def _bollinger(self, closes: List[float], period: int, std_mult: float):
        if len(closes) < period:
            c = float(closes[-1]) if closes else 0.0
            return c, c, c
        arr = np.array(closes[-period:], dtype=float)
        ma = float(np.mean(arr))
        std = float(np.std(arr))
        return ma + std_mult * std, ma, ma - std_mult * std

    # --- Ana sinyal üretimi ---
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        cfg = self.config
        symbols = cfg.get("symbols") or []
        fast = int(cfg.get("fast_ma", 20))
        mid = int(cfg.get("mid_ma", 50))
        slow = int(cfg.get("slow_ma", 200))
        rsi_p = int(cfg.get("rsi_period", 14))
        rsi_os = float(cfg.get("rsi_oversold", 30.0))
        rsi_re_lo = float(cfg.get("rsi_reentry_low", 40.0))
        rsi_re_hi = float(cfg.get("rsi_reentry_high", 55.0))
        bb_p = int(cfg.get("bb_period", 20))
        bb_std = float(cfg.get("bb_std", 2.0))
        vol_w = int(cfg.get("vol_window", 20))
        min_strength = float(cfg.get("min_strength", 0.55))

        for sym in symbols:
            if not str(sym).upper().endswith("USDT"):
                continue
            closes = self._extract_closes(market_data, sym)
            if len(closes) < max(slow, bb_p) + 5:
                continue
            arr = np.array(closes, dtype=float)
            price = float(arr[-1])
            sma_fast = self._sma(arr, fast)
            sma_mid = self._sma(arr, mid)
            sma_slow = self._sma(arr, slow)

            # Global ayı filtresi: downtrend
            if not (price < sma_slow and sma_mid < sma_slow):
                continue

            rsi_now = self._rsi(closes, rsi_p)
            bb_up, bb_mid, bb_low = self._bollinger(closes, bb_p, bb_std)
            vol_pct = self._volatility_pct(closes, vol_w)

            # 1) Trend pullback modülü
            pullback_sig = None
            try:
                prev = float(arr[-4])
                # Son günlerde kısa bir tepki (yukarı) ve şimdi zayıflama
                mom_short = (price - prev) / prev if prev > 0 else 0.0
            except Exception:
                mom_short = 0.0
            if price > sma_fast and price <= sma_mid and mom_short <= 0:
                # Güçlü ayı trendinde MA20/50 bandına tepki sonrası zayıflama → SELL
                strength = min(1.0, 0.6 + min(0.4, abs(mom_short) * 40.0 + max(0.0, (sma_mid - price) / max(price, 1e-9) * 10)))
                if strength >= min_strength:
                    pullback_sig = ("trend_pullback", strength)

            # 2) RSI re-entry modülü (oversold sonrası zayıf toparlanma)
            rsi_sig = None
            if rsi_now >= rsi_re_lo and rsi_now <= rsi_re_hi and price < bb_mid:
                strength = min(1.0, 0.55 + (rsi_now - rsi_re_lo) / max(1.0, (rsi_re_hi - rsi_re_lo)) * 0.25)
                if strength >= min_strength:
                    rsi_sig = ("mean_reversion_reentry", strength)

            # 3) BB breakdown modülü (dar bant sonrası aşağı kırılım + momentum)
            bb_sig = None
            width_rel = (bb_up - bb_low) / price if price > 0 else 0.0
            mom_last = (price - float(arr[-2])) / float(arr[-2]) if arr[-2] > 0 else 0.0
            if width_rel <= 0.04 and price < bb_low and mom_last <= -0.005 and vol_pct > 0:
                strength = min(1.0, 0.6 + min(0.4, abs(mom_last) * 80.0 + width_rel * 5.0))
                if strength >= min_strength:
                    bb_sig = ("bb_breakdown", strength)

            # En güçlü modülü seç
            candidates = [x for x in (pullback_sig, rsi_sig, bb_sig) if x is not None]
            if not candidates:
                continue
            module_name, strength = max(candidates, key=lambda t: t[1])

            # Volatiliteye göre ATR benzeri yüzdelik (pozisyon ve SL/TP ipucu)
            atr_pct = max(0.002, min(0.03, vol_pct * np.sqrt(vol_w))) if vol_pct > 0 else 0.01
            sl_pct = min(0.06, max(0.01, 1.8 * atr_pct))
            tp_pct = min(0.12, max(0.03, 3.0 * atr_pct))

            sigs.append({
                "symbol": sym,
                "side": "sell",
                "strength": float(strength),
                "entry_price": price,
                "stop_loss": price * (1.0 + sl_pct),
                "take_profit": price * (1.0 - tp_pct),
                "module": module_name,
                "atr_pct": float(atr_pct),
                "reason": f"HBH-{module_name}",
                "strategy": "hybrid_bear_hunter"
            })

        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Volatiliteye göre ölçeklenen basit pozisyon boyutu."""
        try:
            base_frac = float(self.config.get("base_fraction", 0.05))
        except Exception:
            base_frac = 0.05
        atr_pct = float(signal.get("atr_pct", 0.01) or 0.01)
        # Volatilite yüksekse fraksiyonu azalt (inverse volatility)
        vol_scale = 0.5 + max(0.2, min(1.5, 0.02 / max(atr_pct, 1e-4)))
        frac = max(0.01, min(0.25, base_frac / vol_scale))
        # Mean reversion re-entry modülünde yarım boy
        if signal.get("module") == "mean_reversion_reentry":
            frac *= 0.5
        return max(0.0, account_balance * frac)
