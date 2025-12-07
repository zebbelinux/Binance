"""
Pump/Dump Strategy
- Momentum Breakout (Pump yakalama)
- Dump Reversal (Kapitulasyon tepmesi)
- Pump Trap Filter (Aşırı ısınmayı eleme)
Beklenen market_data alanları:
- symbol, price
- technical_analysis: { rsi, macd, macd_signal, macd_histogram, atr, ma_20, bb_upper, bb_lower, volume_ratio, volume_sma, price_change_5m }
"""
from typing import Dict, List, Any
from datetime import datetime
from strategies.base_strategy import BaseStrategy

class PumpDumpStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('pump_dump', config or {})
        c = self.config
        # Momentum Breakout
        c.setdefault('volume_spike_ratio', 2.0)
        c.setdefault('atr_mult_breakout', 1.5)
        c.setdefault('risk_R', 0.25)  # 0.25R mini scalping
        # Dump Reversal
        c.setdefault('rsi_dump_threshold', 25)
        c.setdefault('dump_volume_ratio', 1.5)
        # Pump Trap Filter
        c.setdefault('trap_rsi', 80)
        c.setdefault('trap_change_5m', 0.08)
        c.setdefault('trap_volume_ratio', 5.0)
        # Genel
        c.setdefault('symbols', ['BTCUSDT'])

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            symbol = market_data.get('symbol')
            price = float(market_data.get('price', 0) or 0)
            ta = market_data.get('technical_analysis', {}) or {}
            if not symbol or price <= 0 or not ta:
                return out

            # Trap filter: aşırı ısınmış ortamda BUY engelle
            if self._is_pump_trap(ta):
                return out

            # A) Momentum Breakout
            sig = self._momentum_breakout(symbol, price, ta)
            if sig:
                out.append(sig)

            # B) Dump Reversal
            sig = self._dump_reversal(symbol, price, ta)
            if sig:
                out.append(sig)

            return out
        except Exception:
            return out

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Risk bazlı pozisyon boyutu: risk_amount / stop_distance.
        risk_amount: account_balance * risk_per_trade (varsayılan 1%).
        stop_distance: entry_price - stop_loss (BUY için). Fallback: ATR.
        """
        try:
            price = float(signal.get('entry_price', 0) or 0)
            stop = float(signal.get('stop_loss', 0) or 0)
            if price <= 0:
                return 0.0
            stop_distance = abs(price - stop) if stop and stop > 0 else 0.0
            # ATR fallback
            if stop_distance <= 0:
                atr = float(self.config.get('atr_fallback', 0.005) or 0.005) * price
                stop_distance = max(atr, price * 0.003)
            risk_per_trade = float(self.config.get('risk_per_trade', 0.01) or 0.01)
            risk_amount = max(0.0, account_balance * risk_per_trade)
            if stop_distance <= 0:
                return 0.0
            size = risk_amount / stop_distance
            # Maks pozisyon sınırı (portföy yüzdesi)
            max_frac = float(self.config.get('max_position_size', 0.10) or 0.10)
            max_notional = account_balance * max_frac
            notional = size * price
            if notional > max_notional and price > 0:
                size = max_notional / price
            return max(0.0, float(size))
        except Exception:
            return 0.0

    def _momentum_breakout(self, symbol: str, price: float, ta: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            volr = float(ta.get('volume_ratio', 1.0) or 1.0)
            ma20 = ta.get('ma_20')
            atr = ta.get('atr')
            macd_hist = float(ta.get('macd_histogram', 0.0) or 0.0)
            if ma20 is None or atr is None:
                return None
            # Hacim + Fiyat kırılımı + momentum
            if (volr >= float(self.config['volume_spike_ratio']) and
                price > float(ma20) + float(self.config['atr_mult_breakout']) * float(atr) and
                macd_hist > 0):
                stop = self._swing_low_stop(price, ta)
                take = price + 2.5 * float(atr)
                return {
                    'symbol': symbol,
                    'side': 'buy',
                    'strength': 0.8,
                    'entry_price': price,
                    'stop_loss': stop,
                    'take_profit': take,
                    'reason': 'pump breakout',
                    'strategy': 'pump_breakout'
                }
            return None
        except Exception:
            return None

    def _dump_reversal(self, symbol: str, price: float, ta: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            rsi = float(ta.get('rsi', 50) or 50)
            bb_lower = ta.get('bb_lower')
            volr = float(ta.get('volume_ratio', 1.0) or 1.0)
            ma20 = ta.get('ma_20')
            pc5 = ta.get('price_change_5m', None)
            if bb_lower is None or ma20 is None:
                return None
            # Kapitulasyon: RSI düşük + BB altından dönüş + yüksek hacim + (opsiyonel) 5dk sert düşüş
            cond_core = (rsi < float(self.config['rsi_dump_threshold']) and price > float(bb_lower) and volr >= float(self.config['dump_volume_ratio']))
            cond_pc = True
            if pc5 is not None:
                try:
                    cond_pc = float(pc5) <= -0.08
                except Exception:
                    pass
            if cond_core and cond_pc:
                stop = self._swing_low_stop(price, ta)
                take = float(ma20)
                return {
                    'symbol': symbol,
                    'side': 'buy',
                    'strength': 0.7,
                    'entry_price': price,
                    'stop_loss': stop,
                    'take_profit': take,
                    'reason': 'dump reversal',
                    'strategy': 'dump_reversal'
                }
            return None
        except Exception:
            return None

    def _is_pump_trap(self, ta: Dict[str, Any]) -> bool:
        try:
            rsi = float(ta.get('rsi', 50) or 50)
            pc5 = float(ta.get('price_change_5m', 0) or 0)
            volr = float(ta.get('volume_ratio', 1.0) or 1.0)
            if (rsi >= float(self.config['trap_rsi']) and pc5 >= float(self.config['trap_change_5m']) and volr >= float(self.config['trap_volume_ratio'])):
                return True
            return False
        except Exception:
            return False

    def _swing_low_stop(self, price: float, ta: Dict[str, Any]) -> float:
        """Basit swing low stop: fiyat - 1*ATR fallback."""
        try:
            atr = float(ta.get('atr', 0) or 0)
            return price - max(0.0, atr)
        except Exception:
            return price * 0.99
