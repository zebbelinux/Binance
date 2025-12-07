"""
AI Sentiment Fusion Strategy
Teknik sinyaller ile (varsa) sentiment/on-chain skorlarını birleştirir.
"""
from typing import Dict, List, Any
import numpy as np
from strategies.base_strategy import BaseStrategy

class AISentimentFusionStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__('ai_sentiment_fusion', config or {})
        # Varsayılanlar
        self.config.setdefault('symbols', ['BTCUSDT','ETHUSDT'])
        self.config.setdefault('rsi_period', 14)
        self.config.setdefault('bb_period', 20)
        self.config.setdefault('bb_std', 2.0)
        self.config.setdefault('sentiment_weight', 0.4)  # 0..1
        self.config.setdefault('min_strength', 0.55)
        self.config.setdefault('min_confidence', 0.68)

    def _extract_closes(self, md: Dict[str, Any], symbol: str):
        try:
            d = md.get(symbol)
            if isinstance(d, dict):
                closes = d.get('closes') or d.get('prices') or []
                if isinstance(closes, list):
                    return [float(x) for x in closes if x is not None][-100:]
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

    def _bb_bands(self, prices: List[float], period: int, std_mult: float):
        if len(prices) < period:
            m = float(np.mean(prices)) if prices else 0.0
            return m, m, m
        arr = np.array(prices[-period:], dtype=float)
        ma = float(np.mean(arr))
        sd = float(np.std(arr))
        return (ma + std_mult*sd, ma, ma - std_mult*sd)

    def _get_sentiment(self, md: Dict[str, Any], symbol: str) -> float:
        # Beklenen: md['sentiment'][symbol] 0..1 veya md['global_sentiment'] 0..1
        try:
            sdict = md.get('sentiment')
            if isinstance(sdict, dict):
                val = sdict.get(symbol)
                if val is not None:
                    return float(val)
            gs = md.get('global_sentiment')
            if gs is not None:
                return float(gs)
        except Exception:
            pass
        return 0.5  # nötr

    def _get_ml_confidence(self, md: Dict[str, Any], symbol: str) -> float:
        try:
            # Öncelik: md['ml_confidence'][symbol]
            m = md.get('ml_confidence')
            if isinstance(m, dict) and symbol in m:
                return float(m.get(symbol) or 0.0)
            # Alternatif tekil alan: md['ml_success_prob']
            v = md.get('ml_success_prob')
            if v is not None:
                return float(v)
        except Exception:
            pass
        return 0.0

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        syms = self.config.get('symbols') or []
        rsi_p = int(self.config.get('rsi_period', 14))
        bb_p = int(self.config.get('bb_period', 20))
        bb_std = float(self.config.get('bb_std', 2.0))
        w_sent = float(self.config.get('sentiment_weight', 0.4))
        min_strength = float(self.config.get('min_strength', 0.55))
        min_conf = float(self.config.get('min_confidence', 0.68))
        for sym in syms:
            closes = self._extract_closes(market_data, sym)
            if len(closes) < max(bb_p+1, rsi_p+1):
                continue
            rsi = self._rsi(closes, rsi_p)
            upper, ma, lower = self._bb_bands(closes, bb_p, bb_std)
            c_now = closes[-1]
            # Basit teknik sinyal: MA üzerinde + RSI>55 -> buy; MA altında + RSI<45 -> sell
            tech_score = 0.5
            if c_now > ma and rsi > 55:
                tech_score = min(1.0, 0.6 + (rsi-55)/100.0)
                side = 'buy'
            elif c_now < ma and rsi < 45:
                tech_score = min(1.0, 0.6 + (45-rsi)/100.0)
                side = 'sell'
            else:
                continue
            sent = self._get_sentiment(market_data, sym)  # 0..1
            fused = (1-w_sent) * tech_score + w_sent * sent
            mlc = self._get_ml_confidence(market_data, sym)
            # ML güven eşiği kontrolü
            if mlc and mlc < min_conf:
                continue
            if fused >= min_strength:
                sigs.append({'symbol': sym, 'side': side, 'strength': float(fused), 'expected_pnl': 0.0, 'ml_confidence': float(mlc or 0.0), 'note': f'sent {sent:.2f}, rsi {rsi:.1f}'})
        return sigs

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        frac = float(self.config.get('risk_fraction', 0.02))
        return max(0.0, account_balance * frac)
