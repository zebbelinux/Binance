"""Inventory TWAP / PIEZO-TWAP helper
Sembol bağımsız, envanter boşaltma odaklı slice üretici.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


@dataclass
class TwapState:
    symbol: str
    initial_qty: float
    remaining_qty: float
    started_at: datetime
    last_slice_at: Optional[datetime] = None
    slice_no: int = 0
    cooldown_until: Optional[datetime] = None
    finished: bool = False


class InventoryTwapLiquidator:
    """Generic envanter boşaltma (PIEZO-TWAP) mantığı.

    Bu sınıf doğrudan emir göndermez; sadece her çağrıda
    PaperTradeExecutor.execute() ile uyumlu sinyal dict üretir.
    """

    def __init__(self, slice_interval_sec: float = 5.0):
        self.slice_interval_sec = float(slice_interval_sec)
        # symbol -> TwapState
        self._states: Dict[str, TwapState] = {}

    # --- Public API ---
    def start(self, symbol: str, total_qty: float) -> None:
        """Yeni bir envanter boşaltma oturumu başlat.

        Sembol bağımsız, yalnızca qty takip eder.
        """
        symbol = str(symbol).upper()
        now = datetime.now()
        self._states[symbol] = TwapState(
            symbol=symbol,
            initial_qty=float(total_qty or 0.0),
            remaining_qty=float(total_qty or 0.0),
            started_at=now,
        )

    def stop(self, symbol: str) -> None:
        """Sembol için TWAP oturumunu sonlandır."""
        symbol = str(symbol).upper()
        st = self._states.get(symbol)
        if st:
            st.finished = True
            st.cooldown_until = datetime.now() + timedelta(minutes=30)

    def get_state(self, symbol: str) -> Optional[TwapState]:
        return self._states.get(str(symbol).upper())

    def generate_slice_signal(
        self,
        symbol: str,
        inventory_qty: float,
        last_1m_volume: float,
        best_bid: float,
        best_ask: float,
        mid_price: Optional[float] = None,
        atr_1m_pct: Optional[float] = None,
        vwap_1m: Optional[float] = None,
        slippage_bp_recent: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Bir sonraki slice için SELL sinyali üret.

        Parametreler, exchange/market data katmanından beslenmeli.
        """
        symbol = str(symbol).upper()
        if inventory_qty <= 0:
            return None

        st = self._states.get(symbol)
        now = datetime.now()

        # İlk çağrıda state yoksa başlat
        if not st:
            self.start(symbol, inventory_qty)
            st = self._states[symbol]

        # Cooldown / finished kontrolü
        if st.finished:
            if st.cooldown_until and now < st.cooldown_until:
                return None
            # Cooldown bitti, state resetlenebilir
            return None

        # Envanter güncelle (dış dünya ile senkron)
        st.remaining_qty = float(max(0.0, inventory_qty))

        # Zero-inventory kesici (< %5)
        if st.initial_qty > 0 and st.remaining_qty / st.initial_qty < 0.05:
            if st.remaining_qty <= 0:
                st.finished = True
                st.cooldown_until = now + timedelta(minutes=30)
                return None
            # Kalanı tek seferde kapat
            slice_qty = st.remaining_qty
            st.remaining_qty = 0.0
            st.finished = True
            st.cooldown_until = now + timedelta(minutes=30)
            limit_px = self._compute_limit_price(best_bid, best_ask, mid_price)
            st.slice_no += 1
            st.last_slice_at = now
            return self._build_signal(
                symbol=symbol,
                qty=slice_qty,
                limit_px=limit_px,
                slice_no=st.slice_no,
                vwap_1m=vwap_1m,
                slippage_bp_recent=slippage_bp_recent,
                reason="zero_inventory_closer",
            )

        # Slice interval kontrolü
        if st.last_slice_at is not None:
            dt = (now - st.last_slice_at).total_seconds()
            # Slippage ve ATR'e göre adaptif interval
            interval = self._compute_interval(self.slice_interval_sec, atr_1m_pct, slippage_bp_recent)
            if dt < interval:
                return None

        # Slice qty hesapla
        slice_qty = self._compute_slice_qty(
            remaining_qty=st.remaining_qty,
            last_1m_volume=last_1m_volume,
            atr_1m_pct=atr_1m_pct,
        )
        if slice_qty <= 0:
            return None

        # Limit fiyat
        limit_px = self._compute_limit_price(best_bid, best_ask, mid_price)
        if limit_px <= 0:
            return None

        # State güncelle
        st.slice_no += 1
        st.last_slice_at = now
        st.remaining_qty = max(0.0, st.remaining_qty - slice_qty)

        return self._build_signal(
            symbol=symbol,
            qty=slice_qty,
            limit_px=limit_px,
            slice_no=st.slice_no,
            vwap_1m=vwap_1m,
            slippage_bp_recent=slippage_bp_recent,
            reason="inventory_twap_slice",
        )

    # --- İç yardımcılar ---
    @staticmethod
    def _compute_limit_price(best_bid: float, best_ask: float, mid_price: Optional[float]) -> float:
        try:
            best_bid = float(best_bid or 0.0)
            best_ask = float(best_ask or 0.0)
            if mid_price is None or mid_price <= 0:
                if best_bid > 0 and best_ask > 0:
                    mid_price = (best_bid + best_ask) / 2.0
                else:
                    mid_price = best_bid or best_ask
            if not mid_price or mid_price <= 0:
                return 0.0
            # limit_px = max(bestBid, midPrice - 0.01%)
            limit_px = max(best_bid, mid_price * (1.0 - 0.0001))
            return float(limit_px or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _compute_slice_qty(
        remaining_qty: float,
        last_1m_volume: float,
        atr_1m_pct: Optional[float] = None,
    ) -> float:
        try:
            remaining_qty = float(remaining_qty or 0.0)
            last_1m_volume = float(last_1m_volume or 0.0)
            if remaining_qty <= 0:
                return 0.0

            # Base: 1-dk hacmin %20'si
            base = 0.2 * last_1m_volume if last_1m_volume > 0 else remaining_qty

            # ATR(1 dk) < %0,4 ise slice ×1,5; ATR > %0,8 ise slice ×0,5
            if atr_1m_pct is not None:
                try:
                    atr_1m_pct = float(atr_1m_pct or 0.0)
                    if atr_1m_pct < 0.004:
                        base *= 1.5
                    elif atr_1m_pct > 0.008:
                        base *= 0.5
                except Exception:
                    pass

            # En fazla kalan envanter kadar
            return max(0.0, min(remaining_qty, base))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_interval(
        base_interval: float,
        atr_1m_pct: Optional[float],
        slippage_bp_recent: Optional[float],
    ) -> float:
        """Slippage > 3bp ise interval'i 5s -> 10s'e esnet."""
        interval = float(base_interval or 5.0)
        try:
            if slippage_bp_recent is not None and float(slippage_bp_recent) > 3.0:
                interval = max(interval, 10.0)
        except Exception:
            pass
        return interval

    @staticmethod
    def _build_signal(
        symbol: str,
        qty: float,
        limit_px: float,
        slice_no: int,
        vwap_1m: Optional[float],
        slippage_bp_recent: Optional[float],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "side": "sell",
            "size": float(qty or 0.0),
            "entry_price": float(limit_px or 0.0),
            "is_maker": True,  # post-only mantığı için ipucu
            "liquidity": "maker",
            "strategy_name": "inventory_twap",
            "reason_code": reason,
            "slice_no": int(slice_no),
            "vwap_1m": float(vwap_1m) if vwap_1m is not None else None,
            "slippage_bp_recent": float(slippage_bp_recent) if slippage_bp_recent is not None else None,
            "ts": datetime.now().isoformat(),
        }
