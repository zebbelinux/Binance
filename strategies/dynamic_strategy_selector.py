"""
Dinamik Strateji Seçimi
AI tabanlı otomatik strateji değişimi ve piyasa rejimi analizi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
from collections import deque
import json
from strategies.base_strategy import BaseStrategy
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os
from pathlib import Path
import pickle

@dataclass
class MarketRegime:
    """Piyasa rejimi"""
    regime_type: str  # 'trending', 'sideways', 'volatile', 'crash'
    confidence: float
    duration_estimate: str
    volatility_level: float
    trend_strength: float
    volume_profile: str
    timestamp: datetime

@dataclass
class StrategyPerformance:
    """Strateji performansı"""
    strategy_name: str
    regime_type: str
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    last_updated: datetime

@dataclass
class StrategyRecommendation:
    """Strateji önerisi"""
    strategy_name: str
    confidence: float
    expected_return: float
    risk_level: str
    reasoning: str
    parameters: Dict[str, Any]
    timestamp: datetime

class DynamicStrategySelector:
    """Dinamik strateji seçici"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Strateji performans geçmişi
        self.strategy_performance = {}  # {strategy_name: StrategyPerformance}
        self.regime_history = deque(maxlen=1000)  # Son 1000 rejim
        self.market_data_history = deque(maxlen=5000)  # Son 5000 market data
        
        # AI model parametreleri
        self.ai_config = {
            'regime_detection_window': 50,  # 50 bar'lık pencere
            'strategy_evaluation_period': 100,  # 100 bar'lık değerlendirme
            'min_trades_for_evaluation': 10,  # Minimum trade sayısı
            'confidence_threshold': 0.6,  # Minimum güven eşiği
            'regime_change_threshold': 0.3,  # Rejim değişim eşiği
            'performance_weight': 0.4,  # Performans ağırlığı
            'regime_fit_weight': 0.3,  # Rejim uyumu ağırlığı
            'risk_adjustment_weight': 0.3  # Risk ayarlama ağırlığı
        }
        
        # Strateji-rejim eşleştirmeleri
        self.strategy_regime_mapping = {
            'scalping': ['volatile', 'sideways'],
            'grid_trading': ['sideways', 'low_volatility'],
            'trend_following': ['trending', 'strong_trend'],
            'hedge_pairs_trading': ['sideways', 'correlated'],
            'hedge_correlation': ['correlated', 'sideways'],
            'hedge_lead_lag': ['trending', 'volatile'],
            'hedge_statistical_arbitrage': ['sideways', 'mean_reverting'],
            'hedge_market_neutral': ['volatile', 'uncertain'],
            'dca': ['crash', 'declining']
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Analiz thread'i
        self.analysis_thread = None
        self.is_analyzing = False
        
        # Callback'ler
        self.strategy_change_callbacks = []
        self.regime_change_callbacks = []
        
        self.logger.info("Dinamik strateji seçici başlatıldı")
    
    def start_analysis(self):
        """Analizi başlat"""
        if self.is_analyzing:
            self.logger.warning("⚠️ Analiz zaten çalışıyor!")
            return
        
        callback_count = len(self.regime_change_callbacks)
        self.logger.info(f"🚀 start_analysis çağrıldı")
        self.logger.info(f"  Kayıtlı regime callback sayısı: {callback_count}")
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("✅ Dinamik strateji analizi başlatıldı")
    
    def stop_analysis(self):
        """Analizi durdur"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.logger.info("Dinamik strateji analizi durduruldu")
    
    def _analysis_loop(self):
        """Ana analiz döngüsü"""
        self.logger.info("🔄 _analysis_loop thread başladı")
        iteration = 0
        
        while self.is_analyzing:
            try:
                iteration += 1
                self.logger.info(f"🔁 Analysis loop iteration #{iteration}")
                
                # Piyasa rejimini analiz et
                current_regime = self._analyze_market_regime()
                self.logger.info(f"  Current regime: {current_regime}")
                
                if current_regime:
                    self.regime_history.append(current_regime)
                    
                    # Rejim değişimi kontrolü
                    is_changed = self._is_regime_changed(current_regime)
                    self.logger.info(f"🔍 Dynamic Selector: Regime changed? {is_changed}")
                    
                    if is_changed:
                        self.logger.info(f"⚡ Regime değişiyor: {current_regime.regime_type} (güven: {current_regime.confidence:.2f})")
                        self._notify_regime_change_callbacks(current_regime)
                    else:
                        self.logger.debug(f"⏭️ Callback atlanıyor (regime değişimi yok)")
                    
                    # Strateji performansını değerlendir
                    self._evaluate_strategy_performance(current_regime)
                    
                    # En iyi stratejiyi seç
                    recommendation = self._select_optimal_strategy(current_regime)
                    
                    if recommendation and recommendation.confidence > self.ai_config['confidence_threshold']:
                        self._notify_strategy_change_callbacks(recommendation)
                
                # 60 saniye bekle
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Analiz döngüsü hatası: {e}")
                time.sleep(60)
    
    def add_market_data(self, market_data: Dict[str, Any]):
        """Piyasa verisi ekle"""
        try:
            with self.lock:
                self.market_data_history.append({
                    'timestamp': datetime.now(),
                    'data': market_data
                })
                
        except Exception as e:
            self.logger.error(f"Piyasa verisi ekleme hatası: {e}")
    
    def _analyze_market_regime(self) -> Optional[MarketRegime]:
        """Piyasa rejimini analiz et"""
        try:
            if len(self.market_data_history) < self.ai_config['regime_detection_window']:
                return None
            
            # Son N bar'ı al
            recent_data = list(self.market_data_history)[-self.ai_config['regime_detection_window']:]
            
            # Teknik göstergeleri hesapla
            indicators = self._calculate_regime_indicators(recent_data)
            
            # AI ile rejim tahmini
            regime_prediction = self._predict_regime_with_ai(indicators)
            
            return regime_prediction
            
        except Exception as e:
            self.logger.error(f"Piyasa rejimi analiz hatası: {e}")
            return None
    
    def _calculate_regime_indicators(self, data: List[Dict]) -> Dict[str, Any]:
        """Rejim göstergelerini hesapla"""
        try:
            if not data:
                return {}
            
            # Fiyat serilerini hazırla
            prices = [d['data'].get('price', 0) for d in data]
            volumes = [d['data'].get('volume', 0) for d in data]
            
            if len(prices) < 10:
                return {}
            
            # Volatilite hesapla
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Yıllık volatilite
            
            # Trend gücü hesapla
            price_change = (prices[-1] - prices[0]) / prices[0]
            trend_strength = abs(price_change)
            
            # Trend yönü
            trend_direction = 1 if price_change > 0 else -1
            
            # Hacim analizi
            avg_volume = np.mean(volumes) if volumes else 0
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0
            
            # RSI benzeri momentum
            if len(prices) >= 14:
                gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
                losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
                
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    momentum = 100 - (100 / (1 + rs))
                else:
                    momentum = 100
            else:
                momentum = 50
            
            # Bollinger Band genişliği
            if len(prices) >= 20:
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                bb_width = (2 * std) / sma if sma > 0 else 0
            else:
                bb_width = 0
            
            # ADX benzeri trend gücü
            if len(prices) >= 14:
                high_low = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                high_close = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                low_close = [abs(prices[i-1] - prices[i]) for i in range(1, len(prices))]
                
                true_range = [max(h, c, l) for h, c, l in zip(high_low, high_close, low_close)]
                atr = np.mean(true_range[-14:]) if len(true_range) >= 14 else 0
                
                trend_strength_adx = min(100, (atr / prices[-1]) * 100) if prices[-1] > 0 else 0
            else:
                trend_strength_adx = 0
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'momentum': momentum,
                'bb_width': bb_width,
                'trend_strength_adx': trend_strength_adx,
                'volume_trend': volume_trend,
                'avg_volume': avg_volume,
                'price_change': price_change,
                'data_points': len(prices)
            }
            
        except Exception as e:
            self.logger.error(f"Rejim göstergeleri hesaplama hatası: {e}")
            return {}
    
    def _predict_regime_with_ai(self, indicators: Dict[str, Any]) -> Optional[MarketRegime]:
        """AI ile rejim tahmini"""
        try:
            if not indicators:
                return None
            
            # Basit kural tabanlı rejim belirleme (AI modeli yerine)
            volatility = indicators.get('volatility', 0)
            trend_strength = indicators.get('trend_strength', 0)
            momentum = indicators.get('momentum', 50)
            bb_width = indicators.get('bb_width', 0)
            trend_direction = indicators.get('trend_direction', 0)
            
            # Rejim belirleme kuralları
            if volatility > 0.3:  # Yüksek volatilite
                if trend_strength > 0.1:  # Güçlü trend
                    regime_type = 'volatile_trending'
                    confidence = min(0.9, volatility + trend_strength)
                else:
                    regime_type = 'volatile'
                    confidence = min(0.8, volatility)
            elif trend_strength > 0.05:  # Orta-yüksek trend
                if trend_direction > 0:
                    regime_type = 'uptrend'
                else:
                    regime_type = 'downtrend'
                confidence = min(0.8, trend_strength * 2)
            elif bb_width < 0.02:  # Dar Bollinger Band
                regime_type = 'sideways'
                confidence = min(0.7, 1 - bb_width * 10)
            elif momentum > 70:  # Aşırı alım
                regime_type = 'overbought'
                confidence = 0.6
            elif momentum < 30:  # Aşırı satım
                regime_type = 'oversold'
                confidence = 0.6
            else:
                regime_type = 'sideways'
                confidence = 0.5
            
            # Crash detection
            if volatility > 0.5 and trend_direction < 0:
                regime_type = 'crash'
                confidence = min(0.95, volatility)
            
            # Süre tahmini
            duration_estimate = self._estimate_regime_duration(regime_type, indicators)
            
            # Hacim profili
            volume_profile = self._analyze_volume_profile(indicators)
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                duration_estimate=duration_estimate,
                volatility_level=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"AI rejim tahmini hatası: {e}")
            return None
    
    def _estimate_regime_duration(self, regime_type: str, indicators: Dict[str, Any]) -> str:
        """Rejim süresini tahmin et"""
        try:
            volatility = indicators.get('volatility', 0)
            trend_strength = indicators.get('trend_strength', 0)
            
            if regime_type in ['crash', 'volatile']:
                if volatility > 0.4:
                    return "1-3 saat"
                else:
                    return "3-6 saat"
            elif regime_type in ['uptrend', 'downtrend']:
                if trend_strength > 0.1:
                    return "6-12 saat"
                else:
                    return "12-24 saat"
            elif regime_type == 'sideways':
                return "24-48 saat"
            else:
                return "3-6 saat"
                
        except Exception as e:
            self.logger.error(f"Süre tahmini hatası: {e}")
            return "3-6 saat"
    
    def _analyze_volume_profile(self, indicators: Dict[str, Any]) -> str:
        """Hacim profili analizi"""
        try:
            volume_trend = indicators.get('volume_trend', 0)
            avg_volume = indicators.get('avg_volume', 0)
            
            if volume_trend > avg_volume * 0.1:
                return "artan"
            elif volume_trend < -avg_volume * 0.1:
                return "azalan"
            else:
                return "stabil"
                
        except Exception as e:
            self.logger.error(f"Hacim profili analiz hatası: {e}")
            return "stabil"
    
    def _is_regime_changed(self, current_regime: MarketRegime) -> bool:
        """Rejim değişimi kontrolü"""
        try:
            # İlk regime tespiti: callback'leri tetikle (başlangıç durumu)
            if len(self.regime_history) < 2:
                self.logger.info(f"🎯 İlk regime tespit edildi: {current_regime.regime_type}")
                return True  # ✅ İlk regime'i de callback'lere bildir
            
            previous_regime = self.regime_history[-2]
            
            # Rejim türü değişimi
            if previous_regime.regime_type != current_regime.regime_type:
                return True
            
            # Güven seviyesi değişimi
            confidence_change = abs(current_regime.confidence - previous_regime.confidence)
            if confidence_change > self.ai_config['regime_change_threshold']:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rejim değişimi kontrol hatası: {e}")
            return False
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla"""
        try:
            # Dinamik strateji seçici için temel pozisyon büyüklüğü
            base_size = account_balance * 0.05  # %5 temel
            
            # Sinyal gücüne göre ayarla
            signal_strength = signal.get('strength', 0.5)
            adjusted_size = base_size * signal_strength
            
            # Rejim güvenine göre ayarla
            regime_confidence = signal.get('regime_confidence', 0.5)
            confidence_adjusted_size = adjusted_size * regime_confidence
            
            # Minimum ve maksimum sınırlar
            min_size = account_balance * 0.01  # %1 minimum
            max_size = account_balance * 0.1   # %10 maksimum
            
            final_size = max(min_size, min(confidence_adjusted_size, max_size))
            
            self.logger.debug(f"Dinamik pozisyon büyüklüğü hesaplandı: {final_size:.2f}")
            return final_size
            
        except Exception as e:
            self.logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return account_balance * 0.01  # Güvenli minimum
    
    def _evaluate_strategy_performance(self, current_regime: MarketRegime):
        """Strateji performansını değerlendir"""
        try:
            # Her strateji için performans hesapla
            for strategy_name in self.strategy_regime_mapping.keys():
                performance = self._calculate_strategy_performance(strategy_name, current_regime)
                
                if performance:
                    self.strategy_performance[strategy_name] = performance
                    
        except Exception as e:
            self.logger.error(f"Strateji performans değerlendirme hatası: {e}")
    
    def _calculate_strategy_performance(self, strategy_name: str, regime: MarketRegime) -> Optional[StrategyPerformance]:
        """Strateji performansını hesapla"""
        try:
            # Bu strateji için geçmiş trade'leri al (simüle edilmiş)
            # Gerçek implementasyonda trade geçmişinden alınacak
            trades = self._get_strategy_trades(strategy_name, regime.regime_type)
            
            if len(trades) < self.ai_config['min_trades_for_evaluation']:
                return None
            
            # Performans metrikleri hesapla
            returns = [trade['return'] for trade in trades]
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            win_rate = len(winning_trades) / len(returns) if returns else 0
            avg_return = np.mean(returns) if returns else 0
            sharpe_ratio = avg_return / np.std(returns) if returns and np.std(returns) > 0 else 0
            
            # Max drawdown hesapla
            cumulative_returns = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            return StrategyPerformance(
                strategy_name=strategy_name,
                regime_type=regime.regime_type,
                win_rate=win_rate,
                avg_return=avg_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_trades=len(trades),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Strateji performans hesaplama hatası ({strategy_name}): {e}")
            return None
    
    def _get_strategy_trades(self, strategy_name: str, regime_type: str) -> List[Dict[str, Any]]:
        """Strateji trade'lerini al (simüle edilmiş)"""
        try:
            # Bu fonksiyon gerçek implementasyonda veritabanından trade geçmişini alacak
            # Şimdilik simüle edilmiş veri döndürüyoruz
            
            # Rejim türüne göre simüle edilmiş performans
            regime_performance = {
                'scalping': {'win_rate': 0.65, 'avg_return': 0.002, 'volatility': 0.001},
                'grid_trading': {'win_rate': 0.70, 'avg_return': 0.0015, 'volatility': 0.0008},
                'trend_following': {'win_rate': 0.60, 'avg_return': 0.003, 'volatility': 0.002},
                'hedge_pairs_trading': {'win_rate': 0.68, 'avg_return': 0.0018, 'volatility': 0.0012},
                'hedge_correlation': {'win_rate': 0.72, 'avg_return': 0.0012, 'volatility': 0.0009},
                'hedge_lead_lag': {'win_rate': 0.58, 'avg_return': 0.0025, 'volatility': 0.0015},
                'hedge_statistical_arbitrage': {'win_rate': 0.75, 'avg_return': 0.001, 'volatility': 0.0007},
                'hedge_market_neutral': {'win_rate': 0.62, 'avg_return': 0.0015, 'volatility': 0.001},
                'dca': {'win_rate': 0.55, 'avg_return': 0.002, 'volatility': 0.0015}
            }
            
            # Rejim uyumu faktörü
            regime_fit_factors = {
                'volatile': {'scalping': 1.2, 'hedge_lead_lag': 1.1, 'hedge_market_neutral': 1.0},
                'sideways': {'grid_trading': 1.3, 'hedge_pairs_trading': 1.2, 'hedge_statistical_arbitrage': 1.1},
                'trending': {'trend_following': 1.3, 'hedge_lead_lag': 1.1, 'scalping': 0.8},
                'uptrend': {'trend_following': 1.4, 'hedge_lead_lag': 1.2, 'grid_trading': 0.7},
                'downtrend': {'dca': 1.3, 'hedge_market_neutral': 1.1, 'trend_following': 0.6},
                'crash': {'dca': 1.5, 'hedge_market_neutral': 1.2, 'scalping': 0.5}
            }
            
            base_perf = regime_performance.get(strategy_name, {'win_rate': 0.5, 'avg_return': 0.001, 'volatility': 0.001})
            fit_factor = regime_fit_factors.get(regime_type, {}).get(strategy_name, 1.0)
            
            # Simüle edilmiş trade'ler oluştur
            num_trades = np.random.randint(15, 50)
            trades = []
            
            for _ in range(num_trades):
                # Win rate'e göre trade sonucu belirle
                is_winning = np.random.random() < base_perf['win_rate'] * fit_factor
                
                if is_winning:
                    return_val = np.random.normal(base_perf['avg_return'] * fit_factor, base_perf['volatility'])
                else:
                    return_val = np.random.normal(-base_perf['avg_return'] * 0.5, base_perf['volatility'])
                
                trades.append({
                    'return': return_val,
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 168))
                })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Strateji trade'leri alma hatası ({strategy_name}): {e}")
            return []
    
    def _select_optimal_strategy(self, current_regime: MarketRegime) -> Optional[StrategyRecommendation]:
        """En iyi stratejiyi seç"""
        try:
            if not self.strategy_performance:
                return None
            
            best_strategy = None
            best_score = -float('inf')
            
            # Her strateji için skor hesapla
            for strategy_name, performance in self.strategy_performance.items():
                score = self._calculate_strategy_score(strategy_name, performance, current_regime)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
            
            if best_strategy and best_score > 0:
                # Önerilen parametreleri belirle
                parameters = self._get_optimal_parameters(best_strategy, current_regime)
                
                return StrategyRecommendation(
                    strategy_name=best_strategy,
                    confidence=min(best_score, 1.0),
                    expected_return=performance.avg_return,
                    risk_level=self._assess_risk_level(performance),
                    reasoning=self._generate_reasoning(best_strategy, current_regime, performance),
                    parameters=parameters,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Optimal strateji seçimi hatası: {e}")
            return None
    
    def _calculate_strategy_score(self, strategy_name: str, performance: StrategyPerformance, regime: MarketRegime) -> float:
        """Strateji skorunu hesapla"""
        try:
            # Performans skoru
            performance_score = (
                performance.win_rate * 0.3 +
                performance.avg_return * 100 * 0.3 +
                performance.sharpe_ratio * 0.2 +
                (1 + performance.max_drawdown) * 0.2  # Drawdown negatif, bu yüzden 1+ ekliyoruz
            )
            
            # Rejim uyumu skoru
            regime_fit_score = self._calculate_regime_fit_score(strategy_name, regime)
            
            # Risk ayarlama skoru
            risk_score = self._calculate_risk_score(performance, regime)
            
            # Ağırlıklı toplam
            total_score = (
                performance_score * self.ai_config['performance_weight'] +
                regime_fit_score * self.ai_config['regime_fit_weight'] +
                risk_score * self.ai_config['risk_adjustment_weight']
            )
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Strateji skor hesaplama hatası ({strategy_name}): {e}")
            return 0.0
    
    def _calculate_regime_fit_score(self, strategy_name: str, regime: MarketRegime) -> float:
        """Rejim uyumu skorunu hesapla"""
        try:
            # Strateji-rejim eşleştirmesi
            suitable_regimes = self.strategy_regime_mapping.get(strategy_name, [])
            
            if regime.regime_type in suitable_regimes:
                base_score = 0.8
            else:
                base_score = 0.3
            
            # Güven seviyesi ile çarp
            confidence_multiplier = regime.confidence
            
            return base_score * confidence_multiplier
            
        except Exception as e:
            self.logger.error(f"Rejim uyumu skor hesaplama hatası: {e}")
            return 0.5
    
    def _calculate_risk_score(self, performance: StrategyPerformance, regime: MarketRegime) -> float:
        """Risk skorunu hesapla"""
        try:
            # Drawdown'a göre risk skoru
            drawdown_score = 1 + performance.max_drawdown  # Drawdown negatif
            
            # Volatiliteye göre risk ayarlama
            if regime.volatility_level > 0.3:  # Yüksek volatilite
                volatility_factor = 0.8
            elif regime.volatility_level < 0.1:  # Düşük volatilite
                volatility_factor = 1.2
            else:
                volatility_factor = 1.0
            
            return drawdown_score * volatility_factor
            
        except Exception as e:
            self.logger.error(f"Risk skor hesaplama hatası: {e}")
            return 1.0
    
    def _get_optimal_parameters(self, strategy_name: str, regime: MarketRegime) -> Dict[str, Any]:
        """Optimal parametreleri belirle"""
        try:
            base_parameters = {
                'scalping': {
                    'profit_target': 0.005,
                    'stop_loss': 0.003,
                    'timeframe': '1m'
                },
                'grid_trading': {
                    'grid_levels': 10,
                    'grid_spacing': 0.01,
                    'max_position_size': 0.2
                },
                'trend_following': {
                    'ma_period': 20,
                    'trend_threshold': 0.02,
                    'stop_loss': 0.05
                },
                'hedge_pairs_trading': {
                    'correlation_threshold': 0.7,
                    'z_score_threshold': 2.0,
                    'max_hedge_ratio': 0.5
                },
                'hedge_correlation': {
                    'correlation_threshold': 0.8,
                    'max_hedge_ratio': 0.3
                },
                'hedge_lead_lag': {
                    'cross_correlation_threshold': 0.6,
                    'lead_lag_window': 5
                },
                'hedge_statistical_arbitrage': {
                    'correlation_threshold': 0.8,
                    'max_hedge_ratio': 0.4
                },
                'hedge_market_neutral': {
                    'max_hedge_ratio': 0.6
                },
                'dca': {
                    'dca_levels': 5,
                    'dca_spacing': 0.05,
                    'max_position_size': 0.3
                }
            }
            
            parameters = base_parameters.get(strategy_name, {}).copy()
            
            # Rejime göre parametreleri ayarla
            if regime.volatility_level > 0.3:  # Yüksek volatilite
                if 'stop_loss' in parameters:
                    parameters['stop_loss'] *= 1.5
                if 'profit_target' in parameters:
                    parameters['profit_target'] *= 1.2
            elif regime.volatility_level < 0.1:  # Düşük volatilite
                if 'stop_loss' in parameters:
                    parameters['stop_loss'] *= 0.7
                if 'profit_target' in parameters:
                    parameters['profit_target'] *= 0.8
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Optimal parametreler belirleme hatası: {e}")
            return {}
    
    def _assess_risk_level(self, performance: StrategyPerformance) -> str:
        """Risk seviyesini değerlendir"""
        try:
            if performance.max_drawdown < -0.05 and performance.sharpe_ratio < 1.0:
                return 'high'
            elif performance.max_drawdown < -0.02 and performance.sharpe_ratio < 1.5:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Risk seviyesi değerlendirme hatası: {e}")
            return 'medium'
    
    def _generate_reasoning(self, strategy_name: str, regime: MarketRegime, performance: StrategyPerformance) -> str:
        """Gerekçe metni oluştur"""
        try:
            reasoning_parts = []
            
            # Performans gerekçesi
            if performance.win_rate > 0.7:
                reasoning_parts.append(f"Yüksek kazanma oranı ({performance.win_rate:.1%})")
            elif performance.win_rate > 0.6:
                reasoning_parts.append(f"İyi kazanma oranı ({performance.win_rate:.1%})")
            
            if performance.sharpe_ratio > 1.5:
                reasoning_parts.append(f"Mükemmel risk-ayarlı getiri (Sharpe: {performance.sharpe_ratio:.2f})")
            elif performance.sharpe_ratio > 1.0:
                reasoning_parts.append(f"İyi risk-ayarlı getiri (Sharpe: {performance.sharpe_ratio:.2f})")
            
            # Rejim uyumu gerekçesi
            suitable_regimes = self.strategy_regime_mapping.get(strategy_name, [])
            if regime.regime_type in suitable_regimes:
                reasoning_parts.append(f"Mevcut piyasa rejimi ({regime.regime_type}) için uygun")
            
            # Risk gerekçesi
            if performance.max_drawdown > -0.02:
                reasoning_parts.append("Düşük maksimum düşüş")
            elif performance.max_drawdown > -0.05:
                reasoning_parts.append("Orta seviye maksimum düşüş")
            
            return "; ".join(reasoning_parts) if reasoning_parts else "Genel performans değerlendirmesi"
            
        except Exception as e:
            self.logger.error(f"Gerekçe oluşturma hatası: {e}")
            return "AI analizi sonucu"
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Mevcut rejimi al"""
        try:
            with self.lock:
                if self.regime_history:
                    return self.regime_history[-1]
                return None
        except Exception as e:
            self.logger.error(f"Mevcut rejim alma hatası: {e}")
            return None
    
    def get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """Strateji performanslarını al"""
        try:
            with self.lock:
                return self.strategy_performance.copy()
        except Exception as e:
            self.logger.error(f"Strateji performansları alma hatası: {e}")
            return {}
    
    def get_regime_history(self) -> List[MarketRegime]:
        """Rejim geçmişini al"""
        try:
            with self.lock:
                return list(self.regime_history)
        except Exception as e:
            self.logger.error(f"Rejim geçmişi alma hatası: {e}")
            return []
    
    def add_strategy_change_callback(self, callback):
        """Strateji değişim callback'i ekle"""
        self.strategy_change_callbacks.append(callback)
    
    def add_regime_change_callback(self, callback):
        """Rejim değişim callback'i ekle"""
        self.regime_change_callbacks.append(callback)
    
    def _notify_strategy_change_callbacks(self, recommendation: StrategyRecommendation):
        """Strateji değişim callback'lerini çağır"""
        for callback in self.strategy_change_callbacks:
            try:
                callback(recommendation)
            except Exception as e:
                self.logger.error(f"Strateji değişim callback hatası: {e}")
    
    def _notify_regime_change_callbacks(self, regime: MarketRegime):
        """Rejim değişim callback'lerini çağır"""
        try:
            callback_count = len(self.regime_change_callbacks)
            self.logger.info(f"📢 Regime change callback çağrılıyor: {callback_count} listener")
            
            if callback_count == 0:
                self.logger.warning(f"⚠️ Callback listesi BOŞ!")
                return
            
            for idx, callback in enumerate(self.regime_change_callbacks, 1):
                try:
                    self.logger.info(f"→ Callback {idx}/{callback_count} çağrılıyor...")
                    self.logger.info(f"  Callback: {callback}")
                    self.logger.info(f"  Regime: {regime}")
                    callback(regime)
                    self.logger.info(f"✅ Callback {idx}/{callback_count} başarılı!")
                except Exception as e:
                    self.logger.error(f"❌ Callback {idx}/{callback_count} hatası: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
        except Exception as e:
            self.logger.error(f"❌ _notify_regime_change_callbacks FATAL hatası: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            if 'ai_config' in new_config:
                self.ai_config.update(new_config['ai_config'])
            
            if 'strategy_regime_mapping' in new_config:
                self.strategy_regime_mapping.update(new_config['strategy_regime_mapping'])
            
            self.logger.info("Dinamik strateji seçici konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global dinamik strateji seçici
dynamic_strategy_selector = DynamicStrategySelector()
