
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from utils.logger import get_logger, LogCategory

REGIMES = ["sideways", "volatile", "trend", "crash"]

@dataclass
class RegimeConfig:
    lookback: int = 200            # feature window
    min_bars: int = 220            # warmup
    refresh_secs: int = 60         # prediction cadence
    prob_floor: float = 0.10       # minimum class prob
    crash_return_th: float = -0.03 # 1-bar pct change threshold
    vol_high_th: float = 0.02      # high volatility threshold (ATR% / BB width)
    trend_adx_th: float = 20.0     # ADX threshold
    sideways_spread_th: float = 0.008  # BB width / price

class MarketRegimeDetector:
    """
    Lightweight hybrid classifier for market regime detection.
    Works with OHLCV DataFrame that already includes selected technical features:
    rsi_14, adx, bb_upper, bb_lower, macd, macd_signal, macd_histogram, atr_percent.
    """
    def __init__(self, config: RegimeConfig = None):
        self.cfg = config or RegimeConfig()
        self.model = LogisticRegression(max_iter=200, multi_class='auto')
        self._is_fitted = False
        self.logger = get_logger("market_regime")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close']
        f['ret_1'] = close.pct_change()
        f['ret_5'] = close.pct_change(5)
        f['ret_20'] = close.pct_change(20)

        for col in ['rsi_14','adx','bb_upper','bb_lower','atr_percent','macd','macd_signal','macd_histogram']:
            if col in df.columns:
                f[col] = df[col]

        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            width = (df['bb_upper'] - df['bb_lower']).abs()
            with np.errstate(divide='ignore', invalid='ignore'):
                f['bb_width'] = (width / close.replace(0, np.nan))
                f['bb_pos'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

        f = f.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()
        return f

    def _rule_based_label(self, row: pd.Series) -> str:
        ret_1 = row.get('ret_1', 0.0) or 0.0
        adx = row.get('adx', 0.0) or 0.0
        bb_w = row.get('bb_width', 0.0) or 0.0
        atrp = row.get('atr_percent', 0.0) or 0.0

        if ret_1 <= self.cfg.crash_return_th:
            return "crash"
        if adx >= self.cfg.trend_adx_th:
            return "trend"
        if (atrp >= self.cfg.vol_high_th) or (bb_w >= self.cfg.vol_high_th):
            return "volatile"
        if bb_w <= self.cfg.sideways_spread_th:
            return "sideways"
        return "sideways"

    def fit_if_needed(self, feat: pd.DataFrame):
        if self._is_fitted or len(feat) < 300:
            return
        y = []
        X = []
        for i in range(len(feat)):
            row = feat.iloc[i]
            label = self._rule_based_label(row)
            y.append(REGIMES.index(label))
            X.append(row.values)
        try:
            self.model.fit(np.array(X), np.array(y))
            self._is_fitted = True
        except Exception as e:
            # If sklearn isn't available or fitting fails, fallback to rule-based only
            self._is_fitted = False
            self.logger.warning(LogCategory.AI, f"Regime modeli fit edilemedi, rule-based'e düşüldü: {e}")

    def predict(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        if len(df) < self.cfg.min_bars:
            probs = {k: (1.0 if k == "sideways" else 0.0) for k in REGIMES}
            return "sideways", probs, {}

        feat = self._build_features(df.tail(self.cfg.lookback + 50))
        if len(feat) == 0:
            probs = {k: (1.0 if k == "sideways" else 0.0) for k in REGIMES}
            return "sideways", probs, {}

        self.fit_if_needed(feat)

        x = feat.iloc[-1].values.reshape(1, -1)
        if self._is_fitted:
            try:
                proba = self.model.predict_proba(x)[0]
                probs = {REGIMES[i]: float(max(self.cfg.prob_floor, p)) for i, p in enumerate(proba)}
            except Exception as e:
                rb = self._rule_based_label(feat.iloc[-1])
                probs = {k: self.cfg.prob_floor for k in REGIMES}
                probs[rb] = 1.0
                self.logger.warning(LogCategory.AI, f"Regime proba hesaplanamadı, rule-based'e düşüldü: {e}")
        else:
            rb = self._rule_based_label(feat.iloc[-1])
            probs = {k: self.cfg.prob_floor for k in REGIMES}
            probs[rb] = 1.0

        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        label = max(probs.items(), key=lambda kv: kv[1])[0]

        diagnostics = {
            'bb_width': float(feat.iloc[-1].get('bb_width', np.nan)) if 'bb_width' in feat.columns else np.nan,
            'adx': float(feat.iloc[-1].get('adx', np.nan)) if 'adx' in feat.columns else np.nan,
            'atr_percent': float(feat.iloc[-1].get('atr_percent', np.nan)) if 'atr_percent' in feat.columns else np.nan,
            'ret_1': float(feat.iloc[-1].get('ret_1', np.nan))
        }
        return label, probs, diagnostics
