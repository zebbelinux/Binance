"""
Signal Classifier Wrapper
- Eğer eğitilmiş bir RandomForest (veya sklearn modeli) varsa joblib ile yükler.
- Yoksa teknik göstergelere dayalı basit heuristik skor üretir.
Girdi özellikleri (beklenen): rsi, macd, adx, atr, volume_ratio
Çıktı: başarı olasılığı (0..1)
"""
from typing import Dict, Any
import os
import joblib

class SignalClassifier:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'models', 'rf_signal.pkl')
        try:
            if os.path.isfile(self.model_path):
                self.model = joblib.load(self.model_path)
        except Exception:
            self.model = None

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Başarı olasılığı tahmini (0..1). Model yoksa heuristik skor döner."""
        try:
            rsi = float(features.get('rsi', 50))
            macd = float(features.get('macd', 0))
            adx = float(features.get('adx', 15))
            atr = float(features.get('atr', 0))
            volr = float(features.get('volume_ratio', 1))
        except Exception:
            rsi, macd, adx, atr, volr = 50.0, 0.0, 15.0, 0.0, 1.0

        if self.model is not None:
            try:
                import numpy as np
                X = [[rsi, macd, adx, atr, volr]]
                proba = self.model.predict_proba(X)[0]
                # İkili sınıf varsayımı: [fail, success]
                return float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                pass
        # Heuristik fallback
        score = 0.5
        # RSI 45-65 arası daha sağlıklı
        score += max(0, 1 - abs(rsi - 55) / 55) * 0.15
        # MACD pozitif ise bir miktar ek
        score += (0.1 if macd > 0 else -0.05)
        # ADX orta-yüksek ise trend teyidi
        if adx >= 20:
            score += 0.1
        # Volatilite/ATR çok yüksekse hata payı
        score -= min(0.15, atr * 2)
        # Hacim oranı >1 iyidir
        if volr > 1:
            score += min(0.1, (volr - 1) * 0.05)
        return max(0.0, min(1.0, score))
