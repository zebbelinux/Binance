"""
Makine Öğrenmesi - Sinyal Kalibrasyonu ve Portföy Yönetimi Modülü
ML tabanlı sinyal kalibrasyonu ve portföy optimizasyonu
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
from datetime import datetime, timedelta
import threading
from collections import deque
import json
import sqlite3
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SignalCalibrationConfig:
    """Sinyal kalibrasyon konfigürasyonu"""
    model_type: str = "random_forest"  # random_forest, gradient_boosting, logistic_regression, svm
    retrain_interval: int = 3600  # 1 saat
    min_training_samples: int = 1000
    feature_window: int = 100  # Son N veri noktası
    prediction_threshold: float = 0.6
    enable_online_learning: bool = True
    model_save_path: str = "ml_models"

@dataclass
class PortfolioMLConfig:
    """Portföy ML konfigürasyonu"""
    rebalance_interval: int = 1800  # 30 dakika
    risk_tolerance: float = 0.05  # %5 risk toleransı
    max_position_size: float = 0.3  # %30 maksimum pozisyon
    min_position_size: float = 0.01  # %1 minimum pozisyon
    enable_dynamic_rebalancing: bool = True
    model_confidence_threshold: float = 0.7

class SignalType(Enum):
    """Sinyal tipleri"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class MLSignalCalibrator:
    """ML tabanlı sinyal kalibratörü"""
    
    def __init__(self, config: SignalCalibrationConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or SignalCalibrationConfig()
        
        # ML modelleri
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Veri geçmişi
        self.training_data = deque(maxlen=10000)
        self.prediction_history = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Model eğitimi
        self.is_training = False
        self.last_training_time = None
        
        # Callback'ler
        self.prediction_callbacks = []
        
        self.logger.info("ML sinyal kalibratörü başlatıldı")
    
    def _create_model(self, model_type: str):
        """ML modeli oluştur"""
        try:
            if model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_type == "logistic_regression":
                return LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            elif model_type == "svm":
                return SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            else:
                raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Model oluşturma hatası: {e}")
            return None
    
    def add_training_data(self, features: Dict[str, float], signal: str, outcome: float):
        """Eğitim verisi ekle"""
        try:
            training_sample = {
                'features': features,
                'signal': signal,
                'outcome': outcome,  # 1: başarılı, 0: başarısız
                'timestamp': datetime.now()
            }
            
            with self.lock:
                self.training_data.append(training_sample)
            
            # Online learning
            if self.config.enable_online_learning and len(self.training_data) >= self.config.min_training_samples:
                self._retrain_model_async()
            
        except Exception as e:
            self.logger.error(f"Eğitim verisi ekleme hatası: {e}")
    
    def _retrain_model_async(self):
        """Modeli asenkron olarak yeniden eğit"""
        if self.is_training:
            return
        
        def train_worker():
            try:
                self.is_training = True
                self._train_model()
                self.is_training = False
            except Exception as e:
                self.logger.error(f"Model eğitimi hatası: {e}")
                self.is_training = False
        
        training_thread = threading.Thread(target=train_worker, daemon=True)
        training_thread.start()
    
    def _train_model(self):
        """Modeli eğit"""
        try:
            if len(self.training_data) < self.config.min_training_samples:
                self.logger.warning("Eğitim için yeterli veri yok")
                return
            
            # Veriyi hazırla
            X, y = self._prepare_training_data()
            
            if X is None or y is None:
                return
            
            # Model oluştur
            model = self._create_model(self.config.model_type)
            if model is None:
                return
            
            # Veriyi ölçekle
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modeli eğit
            model.fit(X_scaled, y)
            
            # Model performansını değerlendir
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            
            # Modeli kaydet
            with self.lock:
                self.models[self.config.model_type] = model
                self.scalers[self.config.model_type] = scaler
                self.feature_importance[self.config.model_type] = self._get_feature_importance(model, X.columns)
            
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Model eğitimi tamamlandı. Ortalama skor: {avg_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Model eğitimi hatası: {e}")
    
    def _prepare_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Eğitim verisini hazırla"""
        try:
            if not self.training_data:
                return None, None
            
            # Veriyi DataFrame'e çevir
            data_list = []
            for sample in self.training_data:
                row = sample['features'].copy()
                row['signal'] = sample['signal']
                row['outcome'] = sample['outcome']
                data_list.append(row)
            
            df = pd.DataFrame(data_list)
            
            # Özellikler ve hedef değişken
            feature_columns = [col for col in df.columns if col not in ['signal', 'outcome']]
            X = df[feature_columns]
            y = df['outcome']
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Eğitim verisi hazırlama hatası: {e}")
            return None, None
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Özellik önemini al"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Özellik önemi alma hatası: {e}")
            return {}
    
    def predict_signal(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Sinyal tahmini yap"""
        try:
            if not self.models or self.config.model_type not in self.models:
                return {
                    'prediction': 'hold',
                    'confidence': 0.0,
                    'model_available': False
                }
            
            model = self.models[self.config.model_type]
            scaler = self.scalers[self.config.model_type]
            
            # Özellikleri hazırla
            feature_df = pd.DataFrame([features])
            X_scaled = scaler.transform(feature_df)
            
            # Tahmin yap
            prediction_proba = model.predict_proba(X_scaled)[0]
            prediction_class = model.predict(X_scaled)[0]
            
            # Güven skoru
            confidence = max(prediction_proba)
            
            # Sinyal tipini belirle
            if prediction_class == 1 and confidence >= self.config.prediction_threshold:
                signal = 'buy'
            elif prediction_class == 0 and confidence >= self.config.prediction_threshold:
                signal = 'sell'
            else:
                signal = 'hold'
            
            result = {
                'prediction': signal,
                'confidence': confidence,
                'model_available': True,
                'feature_importance': self.feature_importance.get(self.config.model_type, {})
            }
            
            # Tahmin geçmişine ekle
            with self.lock:
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'features': features,
                    'prediction': result
                })
            
            # Callback'leri çağır
            self._notify_prediction_callbacks(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sinyal tahmini hatası: {e}")
            return {
                'prediction': 'hold',
                'confidence': 0.0,
                'model_available': False,
                'error': str(e)
            }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Model performansını al"""
        try:
            if not self.models or self.config.model_type not in self.models:
                return {'model_available': False}
            
            model = self.models[self.config.model_type]
            
            # Son tahminlerin performansını değerlendir
            if len(self.prediction_history) < 100:
                return {'model_available': True, 'insufficient_data': True}
            
            # Basit performans metrikleri
            recent_predictions = list(self.prediction_history)[-100:]
            avg_confidence = np.mean([p['prediction']['confidence'] for p in recent_predictions])
            
            return {
                'model_available': True,
                'model_type': self.config.model_type,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_samples': len(self.training_data),
                'avg_confidence': avg_confidence,
                'feature_importance': self.feature_importance.get(self.config.model_type, {})
            }
            
        except Exception as e:
            self.logger.error(f"Model performansı alma hatası: {e}")
            return {'model_available': False, 'error': str(e)}
    
    def save_model(self, model_path: str = None):
        """Modeli kaydet"""
        try:
            if not self.models or self.config.model_type not in self.models:
                self.logger.warning("Kaydedilecek model bulunamadı")
                return False
            
            if model_path is None:
                model_path = f"{self.config.model_save_path}/signal_calibrator_{self.config.model_type}.pkl"
            
            # Model ve scaler'ı kaydet
            model_data = {
                'model': self.models[self.config.model_type],
                'scaler': self.scalers[self.config.model_type],
                'feature_importance': self.feature_importance[self.config.model_type],
                'config': self.config,
                'last_training_time': self.last_training_time
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model kaydedildi: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def load_model(self, model_path: str):
        """Modeli yükle"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            with self.lock:
                self.models[self.config.model_type] = model_data['model']
                self.scalers[self.config.model_type] = model_data['scaler']
                self.feature_importance[self.config.model_type] = model_data['feature_importance']
                self.last_training_time = model_data['last_training_time']
            
            self.logger.info(f"Model yüklendi: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return False
    
    def add_prediction_callback(self, callback: Callable):
        """Tahmin callback'i ekle"""
        self.prediction_callbacks.append(callback)
    
    def _notify_prediction_callbacks(self, prediction: Dict[str, Any]):
        """Tahmin callback'lerini çağır"""
        for callback in self.prediction_callbacks:
            try:
                callback(prediction)
            except Exception as e:
                self.logger.error(f"Tahmin callback hatası: {e}")

class MLPortfolioManager:
    """ML tabanlı portföy yöneticisi"""
    
    def __init__(self, config: PortfolioMLConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or PortfolioMLConfig()
        
        # Portföy durumu
        self.current_portfolio = {}
        self.target_portfolio = {}
        self.portfolio_history = deque(maxlen=1000)
        
        # ML modelleri
        self.allocation_model = None
        self.risk_model = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.rebalance_callbacks = []
        
        self.logger.info("ML portföy yöneticisi başlatıldı")
    
    def update_portfolio_allocation(self, market_data: Dict[str, Any], risk_metrics: Dict[str, Any]):
        """Portföy tahsisini güncelle"""
        try:
            # ML modeli ile tahsis hesapla
            allocation = self._calculate_ml_allocation(market_data, risk_metrics)
            
            # Risk kontrolü
            allocation = self._apply_risk_controls(allocation)
            
            # Hedef portföyü güncelle
            with self.lock:
                self.target_portfolio = allocation
            
            # Rebalancing gerekli mi kontrol et
            if self._should_rebalance():
                self._execute_rebalancing()
            
        except Exception as e:
            self.logger.error(f"Portföy tahsisi güncelleme hatası: {e}")
    
    def _calculate_ml_allocation(self, market_data: Dict[str, Any], risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        """ML ile tahsis hesapla"""
        try:
            # Basit ML tabanlı tahsis (gerçek implementasyonda daha karmaşık olacak)
            symbols = list(market_data.keys())
            
            if not symbols:
                return {}
            
            # Risk-adjusted returns hesapla
            risk_adjusted_returns = {}
            for symbol, data in market_data.items():
                if 'return' in data and 'volatility' in data:
                    risk_adjusted_returns[symbol] = data['return'] / data['volatility']
                else:
                    risk_adjusted_returns[symbol] = 0.0
            
            # Softmax ile tahsis hesapla
            if risk_adjusted_returns:
                values = list(risk_adjusted_returns.values())
                exp_values = np.exp(np.array(values) - np.max(values))
                probabilities = exp_values / np.sum(exp_values)
                
                allocation = dict(zip(symbols, probabilities))
            else:
                # Eşit dağılım
                allocation = {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"ML tahsis hesaplama hatası: {e}")
            return {}
    
    def _apply_risk_controls(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Risk kontrollerini uygula"""
        try:
            # Pozisyon boyutu sınırları
            controlled_allocation = {}
            for symbol, weight in allocation.items():
                # Minimum pozisyon boyutu
                if weight < self.config.min_position_size:
                    weight = 0.0
                # Maksimum pozisyon boyutu
                elif weight > self.config.max_position_size:
                    weight = self.config.max_position_size
                
                controlled_allocation[symbol] = weight
            
            # Toplam ağırlığı normalize et
            total_weight = sum(controlled_allocation.values())
            if total_weight > 0:
                controlled_allocation = {
                    symbol: weight / total_weight 
                    for symbol, weight in controlled_allocation.items()
                }
            
            return controlled_allocation
            
        except Exception as e:
            self.logger.error(f"Risk kontrolü uygulama hatası: {e}")
            return allocation
    
    def _should_rebalance(self) -> bool:
        """Rebalancing gerekli mi kontrol et"""
        try:
            if not self.current_portfolio or not self.target_portfolio:
                return True
            
            # Ağırlık farklarını hesapla
            max_diff = 0.0
            for symbol in set(self.current_portfolio.keys()) | set(self.target_portfolio.keys()):
                current_weight = self.current_portfolio.get(symbol, 0.0)
                target_weight = self.target_portfolio.get(symbol, 0.0)
                diff = abs(current_weight - target_weight)
                max_diff = max(max_diff, diff)
            
            # %5'ten büyük fark varsa rebalancing gerekli
            return max_diff > 0.05
            
        except Exception as e:
            self.logger.error(f"Rebalancing kontrol hatası: {e}")
            return False
    
    def _execute_rebalancing(self):
        """Rebalancing işlemini gerçekleştir"""
        try:
            rebalance_orders = self._calculate_rebalance_orders()
            
            if rebalance_orders:
                # Rebalancing emirlerini gönder
                self._send_rebalance_orders(rebalance_orders)
                
                # Portföy geçmişine ekle
                with self.lock:
                    self.portfolio_history.append({
                        'timestamp': datetime.now(),
                        'current_portfolio': self.current_portfolio.copy(),
                        'target_portfolio': self.target_portfolio.copy(),
                        'rebalance_orders': rebalance_orders
                    })
                
                # Callback'leri çağır
                self._notify_rebalance_callbacks(rebalance_orders)
                
                self.logger.info(f"Rebalancing gerçekleştirildi: {len(rebalance_orders)} emir")
            
        except Exception as e:
            self.logger.error(f"Rebalancing işlemi hatası: {e}")
    
    def _calculate_rebalance_orders(self) -> List[Dict[str, Any]]:
        """Rebalancing emirlerini hesapla"""
        try:
            orders = []
            
            for symbol in set(self.current_portfolio.keys()) | set(self.target_portfolio.keys()):
                current_weight = self.current_portfolio.get(symbol, 0.0)
                target_weight = self.target_portfolio.get(symbol, 0.0)
                
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # %1'den büyük fark
                    order = {
                        'symbol': symbol,
                        'action': 'buy' if weight_diff > 0 else 'sell',
                        'weight_change': weight_diff,
                        'current_weight': current_weight,
                        'target_weight': target_weight
                    }
                    orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Rebalancing emir hesaplama hatası: {e}")
            return []
    
    def _send_rebalance_orders(self, orders: List[Dict[str, Any]]):
        """Rebalancing emirlerini gönder"""
        try:
            # Bu kısım gerçek trading API'si ile entegre edilecek
            for order in orders:
                self.logger.info(f"Rebalancing emri: {order}")
                # Gerçek implementasyonda burada trading API'si çağrılacak
            
        except Exception as e:
            self.logger.error(f"Rebalancing emir gönderme hatası: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Portföy durumunu al"""
        try:
            with self.lock:
                return {
                    'current_portfolio': self.current_portfolio.copy(),
                    'target_portfolio': self.target_portfolio.copy(),
                    'rebalance_needed': self._should_rebalance(),
                    'portfolio_history_count': len(self.portfolio_history)
                }
            
        except Exception as e:
            self.logger.error(f"Portföy durumu alma hatası: {e}")
            return {}
    
    def add_rebalance_callback(self, callback: Callable):
        """Rebalancing callback'i ekle"""
        self.rebalance_callbacks.append(callback)
    
    def _notify_rebalance_callbacks(self, orders: List[Dict[str, Any]]):
        """Rebalancing callback'lerini çağır"""
        for callback in self.rebalance_callbacks:
            try:
                callback(orders)
            except Exception as e:
                self.logger.error(f"Rebalancing callback hatası: {e}")

# Global ML modülleri
ml_signal_calibrator = MLSignalCalibrator()
ml_portfolio_manager = MLPortfolioManager()

# Ana sınıf alias'ı
MLSignalCalibration = MLSignalCalibrator
