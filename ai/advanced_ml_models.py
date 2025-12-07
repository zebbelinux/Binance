"""
Gelişmiş ML Modelleri
LSTM/Transformer tabanlı piyasa rejimi tanıma ve sinyal üretimi
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
import pickle
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os
from pathlib import Path
import joblib

# ML kütüphaneleri (opsiyonel)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

@dataclass
class MLModelConfig:
    """ML model konfigürasyonu"""
    model_type: str  # 'lstm', 'transformer', 'hybrid'
    sequence_length: int = 60
    features_count: int = 10
    hidden_units: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10

@dataclass
class MLPrediction:
    """ML tahmini"""
    regime: str
    confidence: float
    probabilities: Dict[str, float]
    features_importance: Dict[str, float]
    timestamp: datetime
    model_version: str

class LSTMModel:
    """LSTM tabanlı piyasa rejimi tahmin modeli"""
    
    def __init__(self, config: MLModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_names = []
        
    def build_model(self):
        """LSTM modelini oluştur"""
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("TensorFlow bulunamadı, LSTM modeli oluşturulamadı")
                return False
            
            model = Sequential()
            
            # İlk LSTM katmanı
            model.add(LSTM(
                self.config.hidden_units,
                return_sequences=True,
                input_shape=(self.config.sequence_length, self.config.features_count)
            ))
            model.add(Dropout(self.config.dropout_rate))
            
            # Ek LSTM katmanları
            for _ in range(self.config.num_layers - 1):
                model.add(LSTM(
                    self.config.hidden_units,
                    return_sequences=True
                ))
                model.add(Dropout(self.config.dropout_rate))
            
            # Son LSTM katmanı
            model.add(LSTM(self.config.hidden_units))
            model.add(Dropout(self.config.dropout_rate))
            
            # Çıkış katmanları
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(4, activation='softmax'))  # 4 rejim türü
            
            # Modeli derle
            model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.logger.info("LSTM modeli oluşturuldu")
            return True
            
        except Exception as e:
            self.logger.error(f"LSTM model oluşturma hatası: {e}")
            return False
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi hazırla"""
        try:
            # Özellikleri seç
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr'
            ]
            
            # Mevcut sütunları kontrol et
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < 5:
                self.logger.warning(f"Yeterli özellik bulunamadı: {available_features}")
                return None, None
            
            # Veriyi normalize et
            features = data[available_features].values
            features_scaled = self.scaler.fit_transform(features)
            
            # Rejim etiketlerini oluştur
            labels = self._create_regime_labels(data)
            
            # Sequence'ları oluştur
            X, y = self._create_sequences(features_scaled, labels)
            
            self.feature_names = available_features
            return X, y
            
        except Exception as e:
            self.logger.error(f"Veri hazırlama hatası: {e}")
            return None, None
    
    def _create_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Rejim etiketlerini oluştur"""
        try:
            labels = []
            
            for i in range(len(data)):
                # Basit kural tabanlı etiketleme
                volatility = data.iloc[i].get('atr', 0) / data.iloc[i].get('close', 1)
                trend_strength = abs(data.iloc[i].get('macd', 0))
                rsi = data.iloc[i].get('rsi', 50)
                
                if volatility > 0.02:  # Yüksek volatilite
                    if trend_strength > 0.01:
                        label = 0  # 'volatile_trending'
                    else:
                        label = 1  # 'volatile'
                elif trend_strength > 0.005:  # Trend
                    label = 2  # 'trending'
                else:
                    label = 3  # 'sideways'
                
                labels.append(label)
            
            return np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Etiket oluşturma hatası: {e}")
            return np.zeros(len(data))
    
    def _create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sequence'ları oluştur"""
        try:
            X, y = [], []
            
            for i in range(self.config.sequence_length, len(features)):
                X.append(features[i-self.config.sequence_length:i])
                y.append(labels[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # One-hot encoding
            y_categorical = tf.keras.utils.to_categorical(y, num_classes=4)
            
            return X, y_categorical
            
        except Exception as e:
            self.logger.error(f"Sequence oluşturma hatası: {e}")
            return None, None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Modeli eğit"""
        try:
            if not self.model:
                if not self.build_model():
                    return False
            
            # Veriyi train/validation olarak böl
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Callback'ler
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Modeli eğit
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            self.logger.info(f"LSTM modeli eğitildi - Son accuracy: {history.history['accuracy'][-1]:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"LSTM model eğitimi hatası: {e}")
            return False
    
    def predict(self, data: np.ndarray) -> MLPrediction:
        """Tahmin yap"""
        try:
            if not self.is_trained or not self.model:
                return self._create_default_prediction()
            
            # Veriyi hazırla
            if len(data.shape) == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])
            
            # Tahmin yap
            probabilities = self.model.predict(data, verbose=0)[0]
            
            # En yüksek olasılıklı rejimi bul
            regime_index = np.argmax(probabilities)
            regime_names = ['volatile_trending', 'volatile', 'trending', 'sideways']
            predicted_regime = regime_names[regime_index]
            confidence = probabilities[regime_index]
            
            # Özellik önemini hesapla (basit yaklaşım)
            features_importance = self._calculate_feature_importance(data)
            
            return MLPrediction(
                regime=predicted_regime,
                confidence=confidence,
                probabilities={name: prob for name, prob in zip(regime_names, probabilities)},
                features_importance=features_importance,
                timestamp=datetime.now(),
                model_version='lstm_v1'
            )
            
        except Exception as e:
            self.logger.error(f"LSTM tahmin hatası: {e}")
            return self._create_default_prediction()
    
    def _calculate_feature_importance(self, data: np.ndarray) -> Dict[str, float]:
        """Özellik önemini hesapla"""
        try:
            if not self.feature_names:
                return {}
            
            # Basit permutation importance
            base_prediction = self.model.predict(data, verbose=0)[0]
            base_score = np.max(base_prediction)
            
            importance_scores = {}
            
            for i, feature_name in enumerate(self.feature_names):
                # Özelliği shuffle et
                data_permuted = data.copy()
                data_permuted[0, :, i] = np.random.permutation(data_permuted[0, :, i])
                
                # Tahmin yap
                permuted_prediction = self.model.predict(data_permuted, verbose=0)[0]
                permuted_score = np.max(permuted_prediction)
                
                # Önem skoru
                importance_scores[feature_name] = base_score - permuted_score
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Özellik önemi hesaplama hatası: {e}")
            return {}
    
    def _create_default_prediction(self) -> MLPrediction:
        """Varsayılan tahmin"""
        return MLPrediction(
            regime='sideways',
            confidence=0.5,
            probabilities={'sideways': 0.5, 'trending': 0.2, 'volatile': 0.2, 'volatile_trending': 0.1},
            features_importance={},
            timestamp=datetime.now(),
            model_version='default'
        )
    
    def save_model(self, filepath: str) -> bool:
        """Modeli kaydet"""
        try:
            if not self.model:
                return False
            
            # Modeli kaydet
            self.model.save(filepath)
            
            # Scaler'ı kaydet
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Konfigürasyonu kaydet
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            self.logger.info(f"LSTM modeli kaydedildi: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Modeli yükle"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return False
            
            # Modeli yükle
            self.model = tf.keras.models.load_model(filepath)
            
            # Scaler'ı yükle
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Konfigürasyonu yükle
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = MLModelConfig(**config_dict)
            
            self.is_trained = True
            self.logger.info(f"LSTM modeli yüklendi: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return False

class TransformerModel:
    """Transformer tabanlı piyasa rejimi tahmin modeli"""
    
    def __init__(self, config: MLModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def build_model(self):
        """Transformer modelini oluştur"""
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("TensorFlow bulunamadı, Transformer modeli oluşturulamadı")
                return False
            
            # Input layer
            inputs = Input(shape=(self.config.sequence_length, self.config.features_count))
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=8,
                key_dim=self.config.hidden_units // 8
            )(inputs, inputs)
            
            # Layer normalization
            attention_output = LayerNormalization()(attention_output)
            
            # Feed forward network
            ffn_output = Dense(self.config.hidden_units, activation='relu')(attention_output)
            ffn_output = Dropout(self.config.dropout_rate)(ffn_output)
            ffn_output = Dense(self.config.features_count)(ffn_output)
            
            # Residual connection
            ffn_output = LayerNormalization()(ffn_output + attention_output)
            
            # Global average pooling
            pooled_output = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
            
            # Classification head
            dense1 = Dense(64, activation='relu')(pooled_output)
            dense1 = Dropout(self.config.dropout_rate)(dense1)
            dense2 = Dense(32, activation='relu')(dense1)
            outputs = Dense(4, activation='softmax')(dense2)
            
            # Modeli oluştur
            self.model = Model(inputs=inputs, outputs=outputs)
            
            # Modeli derle
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info("Transformer modeli oluşturuldu")
            return True
            
        except Exception as e:
            self.logger.error(f"Transformer model oluşturma hatası: {e}")
            return False
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi hazırla"""
        try:
            # Özellikleri seç
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr'
            ]
            
            # Mevcut sütunları kontrol et
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < 5:
                self.logger.warning(f"Yeterli özellik bulunamadı: {available_features}")
                return None, None
            
            # Veriyi normalize et
            features = data[available_features].values
            features_scaled = self.scaler.fit_transform(features)
            
            # Rejim etiketlerini oluştur
            labels = self._create_regime_labels(data)
            
            # Sequence'ları oluştur
            X, y = self._create_sequences(features_scaled, labels)
            
            self.feature_names = available_features
            return X, y
            
        except Exception as e:
            self.logger.error(f"Veri hazırlama hatası: {e}")
            return None, None
    
    def _create_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Rejim etiketlerini oluştur"""
        try:
            labels = []
            
            for i in range(len(data)):
                # Gelişmiş etiketleme
                volatility = data.iloc[i].get('atr', 0) / data.iloc[i].get('close', 1)
                trend_strength = abs(data.iloc[i].get('macd', 0))
                rsi = data.iloc[i].get('rsi', 50)
                bb_position = (data.iloc[i].get('close', 0) - data.iloc[i].get('bb_lower', 0)) / \
                             (data.iloc[i].get('bb_upper', 1) - data.iloc[i].get('bb_lower', 0))
                
                # Çoklu kriter etiketleme
                if volatility > 0.025 and trend_strength > 0.01:
                    label = 0  # 'volatile_trending'
                elif volatility > 0.02:
                    label = 1  # 'volatile'
                elif trend_strength > 0.008 and (bb_position > 0.7 or bb_position < 0.3):
                    label = 2  # 'trending'
                else:
                    label = 3  # 'sideways'
                
                labels.append(label)
            
            return np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Etiket oluşturma hatası: {e}")
            return np.zeros(len(data))
    
    def _create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sequence'ları oluştur"""
        try:
            X, y = [], []
            
            for i in range(self.config.sequence_length, len(features)):
                X.append(features[i-self.config.sequence_length:i])
                y.append(labels[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # One-hot encoding
            y_categorical = tf.keras.utils.to_categorical(y, num_classes=4)
            
            return X, y_categorical
            
        except Exception as e:
            self.logger.error(f"Sequence oluşturma hatası: {e}")
            return None, None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Modeli eğit"""
        try:
            if not self.model:
                if not self.build_model():
                    return False
            
            # Veriyi train/validation olarak böl
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Callback'ler
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Modeli eğit
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            self.logger.info(f"Transformer modeli eğitildi - Son accuracy: {history.history['accuracy'][-1]:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Transformer model eğitimi hatası: {e}")
            return False
    
    def predict(self, data: np.ndarray) -> MLPrediction:
        """Tahmin yap"""
        try:
            if not self.is_trained or not self.model:
                return self._create_default_prediction()
            
            # Veriyi hazırla
            if len(data.shape) == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])
            
            # Tahmin yap
            probabilities = self.model.predict(data, verbose=0)[0]
            
            # En yüksek olasılıklı rejimi bul
            regime_index = np.argmax(probabilities)
            regime_names = ['volatile_trending', 'volatile', 'trending', 'sideways']
            predicted_regime = regime_names[regime_index]
            confidence = probabilities[regime_index]
            
            # Özellik önemini hesapla
            features_importance = self._calculate_feature_importance(data)
            
            return MLPrediction(
                regime=predicted_regime,
                confidence=confidence,
                probabilities={name: prob for name, prob in zip(regime_names, probabilities)},
                features_importance=features_importance,
                timestamp=datetime.now(),
                model_version='transformer_v1'
            )
            
        except Exception as e:
            self.logger.error(f"Transformer tahmin hatası: {e}")
            return self._create_default_prediction()
    
    def _calculate_feature_importance(self, data: np.ndarray) -> Dict[str, float]:
        """Özellik önemini hesapla"""
        try:
            if not self.feature_names:
                return {}
            
            # Attention weights'ten özellik önemini çıkar
            attention_layer = self.model.layers[1]  # MultiHeadAttention layer
            
            # Basit permutation importance
            base_prediction = self.model.predict(data, verbose=0)[0]
            base_score = np.max(base_prediction)
            
            importance_scores = {}
            
            for i, feature_name in enumerate(self.feature_names):
                # Özelliği shuffle et
                data_permuted = data.copy()
                data_permuted[0, :, i] = np.random.permutation(data_permuted[0, :, i])
                
                # Tahmin yap
                permuted_prediction = self.model.predict(data_permuted, verbose=0)[0]
                permuted_score = np.max(permuted_prediction)
                
                # Önem skoru
                importance_scores[feature_name] = base_score - permuted_score
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Özellik önemi hesaplama hatası: {e}")
            return {}
    
    def _create_default_prediction(self) -> MLPrediction:
        """Varsayılan tahmin"""
        return MLPrediction(
            regime='sideways',
            confidence=0.5,
            probabilities={'sideways': 0.5, 'trending': 0.2, 'volatile': 0.2, 'volatile_trending': 0.1},
            features_importance={},
            timestamp=datetime.now(),
            model_version='default'
        )
    
    def save_model(self, filepath: str) -> bool:
        """Modeli kaydet"""
        try:
            if not self.model:
                return False
            
            # Modeli kaydet
            self.model.save(filepath)
            
            # Scaler'ı kaydet
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Konfigürasyonu kaydet
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            self.logger.info(f"Transformer modeli kaydedildi: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Modeli yükle"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return False
            
            # Modeli yükle
            self.model = tf.keras.models.load_model(filepath)
            
            # Scaler'ı yükle
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Konfigürasyonu yükle
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = MLModelConfig(**config_dict)
            
            self.is_trained = True
            self.logger.info(f"Transformer modeli yüklendi: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return False

class HybridMLModel:
    """Hibrit ML modeli (LSTM + Transformer)"""
    
    def __init__(self, config: MLModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.lstm_model = LSTMModel(config)
        self.transformer_model = TransformerModel(config)
        self.ensemble_weights = {'lstm': 0.6, 'transformer': 0.4}
        self.is_trained = False
        
    def build_model(self):
        """Hibrit modeli oluştur"""
        try:
            lstm_success = self.lstm_model.build_model()
            transformer_success = self.transformer_model.build_model()
            
            return lstm_success and transformer_success
            
        except Exception as e:
            self.logger.error(f"Hibrit model oluşturma hatası: {e}")
            return False
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi hazırla"""
        try:
            # Her iki model için veriyi hazırla
            lstm_X, lstm_y = self.lstm_model.prepare_data(data)
            transformer_X, transformer_y = self.transformer_model.prepare_data(data)
            
            # Verilerin uyumlu olduğunu kontrol et
            if lstm_X is not None and transformer_X is not None:
                return lstm_X, lstm_y
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Hibrit veri hazırlama hatası: {e}")
            return None, None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Hibrit modeli eğit"""
        try:
            # Her iki modeli eğit
            lstm_success = self.lstm_model.train(X, y)
            transformer_success = self.transformer_model.train(X, y)
            
            self.is_trained = lstm_success and transformer_success
            
            if self.is_trained:
                self.logger.info("Hibrit model eğitildi")
            
            return self.is_trained
            
        except Exception as e:
            self.logger.error(f"Hibrit model eğitimi hatası: {e}")
            return False
    
    def predict(self, data: np.ndarray) -> MLPrediction:
        """Hibrit tahmin yap"""
        try:
            if not self.is_trained:
                return self._create_default_prediction()
            
            # Her iki modelden tahmin al
            lstm_prediction = self.lstm_model.predict(data)
            transformer_prediction = self.transformer_model.predict(data)
            
            # Ensemble tahmin
            ensemble_probabilities = {}
            regime_names = ['volatile_trending', 'volatile', 'trending', 'sideways']
            
            for regime in regime_names:
                ensemble_prob = (
                    lstm_prediction.probabilities.get(regime, 0) * self.ensemble_weights['lstm'] +
                    transformer_prediction.probabilities.get(regime, 0) * self.ensemble_weights['transformer']
                )
                ensemble_probabilities[regime] = ensemble_prob
            
            # En yüksek olasılıklı rejimi bul
            predicted_regime = max(ensemble_probabilities.items(), key=lambda x: x[1])[0]
            confidence = ensemble_probabilities[predicted_regime]
            
            # Özellik önemini birleştir
            combined_importance = {}
            for feature in lstm_prediction.features_importance.keys():
                combined_importance[feature] = (
                    lstm_prediction.features_importance.get(feature, 0) * self.ensemble_weights['lstm'] +
                    transformer_prediction.features_importance.get(feature, 0) * self.ensemble_weights['transformer']
                )
            
            return MLPrediction(
                regime=predicted_regime,
                confidence=confidence,
                probabilities=ensemble_probabilities,
                features_importance=combined_importance,
                timestamp=datetime.now(),
                model_version='hybrid_v1'
            )
            
        except Exception as e:
            self.logger.error(f"Hibrit tahmin hatası: {e}")
            return self._create_default_prediction()
    
    def _create_default_prediction(self) -> MLPrediction:
        """Varsayılan tahmin"""
        return MLPrediction(
            regime='sideways',
            confidence=0.5,
            probabilities={'sideways': 0.5, 'trending': 0.2, 'volatile': 0.2, 'volatile_trending': 0.1},
            features_importance={},
            timestamp=datetime.now(),
            model_version='default'
        )
    
    def save_model(self, filepath: str) -> bool:
        """Hibrit modeli kaydet"""
        try:
            lstm_path = filepath.replace('.h5', '_lstm.h5')
            transformer_path = filepath.replace('.h5', '_transformer.h5')
            
            lstm_success = self.lstm_model.save_model(lstm_path)
            transformer_success = self.transformer_model.save_model(transformer_path)
            
            # Ensemble weights'ı kaydet
            weights_path = filepath.replace('.h5', '_weights.json')
            with open(weights_path, 'w') as f:
                json.dump(self.ensemble_weights, f, indent=2)
            
            return lstm_success and transformer_success
            
        except Exception as e:
            self.logger.error(f"Hibrit model kaydetme hatası: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Hibrit modeli yükle"""
        try:
            lstm_path = filepath.replace('.h5', '_lstm.h5')
            transformer_path = filepath.replace('.h5', '_transformer.h5')
            
            lstm_success = self.lstm_model.load_model(lstm_path)
            transformer_success = self.transformer_model.load_model(transformer_path)
            
            # Ensemble weights'ı yükle
            weights_path = filepath.replace('.h5', '_weights.json')
            try:
                with open(weights_path, 'r') as f:
                    self.ensemble_weights = json.load(f)
            except:
                pass  # Varsayılan weights kullan
            
            self.is_trained = lstm_success and transformer_success
            return self.is_trained
            
        except Exception as e:
            self.logger.error(f"Hibrit model yükleme hatası: {e}")
            return False

class AdvancedMLManager:
    """Gelişmiş ML model yöneticisi"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Model konfigürasyonları
        self.model_configs = {
            'lstm': MLModelConfig(
                model_type='lstm',
                sequence_length=60,
                features_count=10,
                hidden_units=128,
                num_layers=2,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=100
            ),
            'transformer': MLModelConfig(
                model_type='transformer',
                sequence_length=60,
                features_count=10,
                hidden_units=128,
                num_layers=2,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=100
            ),
            'hybrid': MLModelConfig(
                model_type='hybrid',
                sequence_length=60,
                features_count=10,
                hidden_units=128,
                num_layers=2,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=100
            )
        }
        
        # Aktif modeller
        self.active_models = {}
        self.model_predictions = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.prediction_callbacks = []
        
        self.logger.info("Gelişmiş ML yöneticisi başlatıldı")
    
    def create_model(self, model_type: str) -> bool:
        """Model oluştur"""
        try:
            if model_type not in self.model_configs:
                self.logger.error(f"Desteklenmeyen model türü: {model_type}")
                return False
            
            config = self.model_configs[model_type]
            
            if model_type == 'lstm':
                model = LSTMModel(config)
            elif model_type == 'transformer':
                model = TransformerModel(config)
            elif model_type == 'hybrid':
                model = HybridMLModel(config)
            else:
                return False
            
            if model.build_model():
                self.active_models[model_type] = model
                self.logger.info(f"{model_type} modeli oluşturuldu")
                return True
            else:
                self.logger.error(f"{model_type} modeli oluşturulamadı")
                return False
                
        except Exception as e:
            self.logger.error(f"Model oluşturma hatası ({model_type}): {e}")
            return False
    
    def train_model(self, model_type: str, data: pd.DataFrame) -> bool:
        """Modeli eğit"""
        try:
            if model_type not in self.active_models:
                if not self.create_model(model_type):
                    return False
            
            model = self.active_models[model_type]
            
            # Veriyi hazırla
            X, y = model.prepare_data(data)
            
            if X is None or y is None:
                self.logger.error(f"Veri hazırlama başarısız ({model_type})")
                return False
            
            # Modeli eğit
            success = model.train(X, y)
            
            if success:
                self.logger.info(f"{model_type} modeli eğitildi")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model eğitimi hatası ({model_type}): {e}")
            return False
    
    def predict(self, model_type: str, data: np.ndarray) -> Optional[MLPrediction]:
        """Tahmin yap"""
        try:
            if model_type not in self.active_models:
                return None
            
            model = self.active_models[model_type]
            
            if not model.is_trained:
                return None
            
            prediction = model.predict(data)
            
            # Tahmini kaydet
            with self.lock:
                self.model_predictions.append(prediction)
            
            # Callback'leri çağır
            self._notify_prediction_callbacks(prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Tahmin hatası ({model_type}): {e}")
            return None
    
    def ensemble_predict(self, data: np.ndarray) -> Optional[MLPrediction]:
        """Ensemble tahmin yap"""
        try:
            predictions = []
            
            # Tüm aktif modellerden tahmin al
            for model_type, model in self.active_models.items():
                if model.is_trained:
                    prediction = model.predict(data)
                    if prediction:
                        predictions.append(prediction)
            
            if not predictions:
                return None
            
            # Ensemble tahmin
            ensemble_probabilities = {}
            regime_names = ['volatile_trending', 'volatile', 'trending', 'sideways']
            
            for regime in regime_names:
                total_prob = 0
                for prediction in predictions:
                    total_prob += prediction.probabilities.get(regime, 0)
                ensemble_probabilities[regime] = total_prob / len(predictions)
            
            # En yüksek olasılıklı rejimi bul
            predicted_regime = max(ensemble_probabilities.items(), key=lambda x: x[1])[0]
            confidence = ensemble_probabilities[predicted_regime]
            
            # Özellik önemini birleştir
            combined_importance = {}
            for prediction in predictions:
                for feature, importance in prediction.features_importance.items():
                    if feature not in combined_importance:
                        combined_importance[feature] = 0
                    combined_importance[feature] += importance
            
            # Ortalama al
            for feature in combined_importance:
                combined_importance[feature] /= len(predictions)
            
            ensemble_prediction = MLPrediction(
                regime=predicted_regime,
                confidence=confidence,
                probabilities=ensemble_probabilities,
                features_importance=combined_importance,
                timestamp=datetime.now(),
                model_version='ensemble'
            )
            
            # Tahmini kaydet
            with self.lock:
                self.model_predictions.append(ensemble_prediction)
            
            # Callback'leri çağır
            self._notify_prediction_callbacks(ensemble_prediction)
            
            return ensemble_prediction
            
        except Exception as e:
            self.logger.error(f"Ensemble tahmin hatası: {e}")
            return None
    
    def save_models(self, base_path: str) -> bool:
        """Tüm modelleri kaydet"""
        try:
            success = True
            
            for model_type, model in self.active_models.items():
                filepath = f"{base_path}_{model_type}.h5"
                if not model.save_model(filepath):
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def load_models(self, base_path: str) -> bool:
        """Tüm modelleri yükle"""
        try:
            success = True
            
            for model_type in self.model_configs.keys():
                filepath = f"{base_path}_{model_type}.h5"
                model = self.active_models.get(model_type)
                
                if model and model.load_model(filepath):
                    continue
                else:
                    # Model yoksa oluştur ve yükle
                    if self.create_model(model_type):
                        model = self.active_models[model_type]
                        if not model.load_model(filepath):
                            success = False
                    else:
                        success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Model performansını al"""
        try:
            performance = {}
            
            for model_type, model in self.active_models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    performance[model_type] = {
                        'is_trained': True,
                        'model_type': model_type,
                        'config': model.config.__dict__ if hasattr(model, 'config') else {}
                    }
                else:
                    performance[model_type] = {
                        'is_trained': False,
                        'model_type': model_type
                    }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Model performansı alma hatası: {e}")
            return {}
    
    def get_recent_predictions(self, limit: int = 100) -> List[MLPrediction]:
        """Son tahminleri al"""
        try:
            with self.lock:
                return list(self.model_predictions)[-limit:]
        except Exception as e:
            self.logger.error(f"Son tahminler alma hatası: {e}")
            return []
    
    def add_prediction_callback(self, callback):
        """Tahmin callback'i ekle"""
        self.prediction_callbacks.append(callback)
    
    def _notify_prediction_callbacks(self, prediction: MLPrediction):
        """Tahmin callback'lerini çağır"""
        for callback in self.prediction_callbacks:
            try:
                callback(prediction)
            except Exception as e:
                self.logger.error(f"Tahmin callback hatası: {e}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            if 'model_configs' in new_config:
                for model_type, config_dict in new_config['model_configs'].items():
                    if model_type in self.model_configs:
                        self.model_configs[model_type] = MLModelConfig(**config_dict)
            
            self.logger.info("ML yöneticisi konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global ML yöneticisi
advanced_ml_manager = AdvancedMLManager()

# Ana sınıf alias'ı
AdvancedMLModels = AdvancedMLManager
