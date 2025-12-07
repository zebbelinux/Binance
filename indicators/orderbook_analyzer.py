"""
Order Book Analizi
Depth imbalance, spoofing detection ve order book analizi
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
import sqlite3
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pickle

@dataclass
class OrderBookLevel:
    """Order book seviyesi"""
    price: float
    quantity: float
    timestamp: datetime

@dataclass
class OrderBookSnapshot:
    """Order book anlık görüntüsü"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: float
    best_ask: float
    spread: float
    mid_price: float

class SpoofingPattern(Enum):
    """Spoofing pattern türleri"""
    LARGE_ORDER_CANCELLATION = "large_order_cancellation"
    RAPID_ORDER_MODIFICATION = "rapid_order_modification"
    LAYER_MANIPULATION = "layer_manipulation"
    MOMENTUM_IGNITION = "momentum_ignition"

class OrderBookAnalyzer:
    """Order book analiz sınıfı"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Order book geçmişi
        self.orderbook_history = deque(maxlen=1000)  # Son 1000 snapshot
        self.order_history = deque(maxlen=10000)  # Son 10000 order
        
        # Spoofing detection parametreleri
        self.spoofing_config = {
            'large_order_threshold': 0.1,  # %10'dan büyük emirler
            'cancellation_time_threshold': 5,  # 5 saniye içinde iptal
            'modification_frequency_threshold': 10,  # 10 saniyede 10+ değişiklik
            'layer_thickness_threshold': 0.05,  # %5'ten kalın katmanlar
            'momentum_threshold': 0.02  # %2'den büyük momentum
        }
        
        # Depth imbalance parametreleri
        self.depth_config = {
            'levels_to_analyze': 10,  # İlk 10 seviye
            'imbalance_threshold': 0.3,  # %30'dan büyük dengesizlik
            'volume_weighted': True,  # Hacim ağırlıklı analiz
            'time_decay_factor': 0.95  # Zaman aşımı faktörü
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.spoofing_callbacks = []
        self.imbalance_callbacks = []
        self.whale_callbacks = []
        
        self.logger.info("Order book analizörü başlatıldı")
    
    def add_orderbook_snapshot(self, snapshot: OrderBookSnapshot):
        """Order book snapshot'ı ekle"""
        try:
            with self.lock:
                self.orderbook_history.append(snapshot)
                
                # Depth imbalance analizi
                imbalance_analysis = self._analyze_depth_imbalance(snapshot)
                
                # Spoofing detection
                spoofing_analysis = self._detect_spoofing_patterns(snapshot)
                
                # Whale activity detection
                whale_activity = self._detect_whale_activity(snapshot)
                whale_analysis = {
                    'is_whale_activity': whale_activity,
                    'timestamp': snapshot.timestamp,
                    'details': 'Büyük emir tespit edildi' if whale_activity else 'Normal aktivite'
                }
                
                # Callback'leri çağır
                if imbalance_analysis['significant']:
                    self._notify_imbalance_callbacks(imbalance_analysis)
                
                if spoofing_analysis['detected']:
                    self._notify_spoofing_callbacks(spoofing_analysis)
                
                if whale_activity:
                    self._notify_whale_callbacks(whale_analysis)
                
        except Exception as e:
            self.logger.error(f"Order book snapshot ekleme hatası: {e}")
    
    def add_order_event(self, order_event: Dict[str, Any]):
        """Order event'i ekle (create, modify, cancel)"""
        try:
            with self.lock:
                self.order_history.append({
                    'timestamp': datetime.now(),
                    'event': order_event
                })
                
        except Exception as e:
            self.logger.error(f"Order event ekleme hatası: {e}")
    
    def _analyze_depth_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Depth imbalance analizi"""
        try:
            analysis = {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'significant': False,
                'imbalance_ratio': 0.0,
                'bid_strength': 0.0,
                'ask_strength': 0.0,
                'levels_analyzed': 0,
                'volume_imbalance': 0.0,
                'price_impact': 0.0
            }
            
            if not snapshot.bids or not snapshot.asks:
                return analysis
            
            # İlk N seviyeyi analiz et
            levels_to_analyze = min(
                self.depth_config['levels_to_analyze'],
                len(snapshot.bids),
                len(snapshot.asks)
            )
            
            bid_volume = 0.0
            ask_volume = 0.0
            bid_weighted_price = 0.0
            ask_weighted_price = 0.0
            
            for i in range(levels_to_analyze):
                bid_level = snapshot.bids[i]
                ask_level = snapshot.asks[i]
                
                # Hacim ağırlıklı analiz
                if self.depth_config['volume_weighted']:
                    weight = (levels_to_analyze - i) / levels_to_analyze  # Yakın seviyeler daha ağırlıklı
                    bid_volume += bid_level.quantity * weight
                    ask_volume += ask_level.quantity * weight
                    bid_weighted_price += bid_level.price * bid_level.quantity * weight
                    ask_weighted_price += ask_level.price * ask_level.quantity * weight
                else:
                    bid_volume += bid_level.quantity
                    ask_volume += ask_level.quantity
                    bid_weighted_price += bid_level.price * bid_level.quantity
                    ask_weighted_price += ask_level.price * ask_level.quantity
            
            # Imbalance ratio hesapla
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                analysis['imbalance_ratio'] = abs(bid_volume - ask_volume) / total_volume
                analysis['bid_strength'] = bid_volume / total_volume
                analysis['ask_strength'] = ask_volume / total_volume
                analysis['volume_imbalance'] = bid_volume - ask_volume
                
                # Ağırlıklı fiyat hesapla
                if bid_volume > 0:
                    analysis['bid_weighted_price'] = bid_weighted_price / bid_volume
                if ask_volume > 0:
                    analysis['ask_weighted_price'] = ask_weighted_price / ask_volume
                
                # Price impact hesapla
                if analysis['bid_weighted_price'] > 0 and analysis['ask_weighted_price'] > 0:
                    analysis['price_impact'] = (analysis['ask_weighted_price'] - analysis['bid_weighted_price']) / snapshot.mid_price
            
            analysis['levels_analyzed'] = levels_to_analyze
            
            # Önemli dengesizlik kontrolü
            if analysis['imbalance_ratio'] > self.depth_config['imbalance_threshold']:
                analysis['significant'] = True
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Depth imbalance analiz hatası: {e}")
            return {'timestamp': snapshot.timestamp, 'symbol': snapshot.symbol, 'significant': False}
    
    def _detect_spoofing_patterns(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Spoofing pattern detection"""
        try:
            analysis = {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'detected': False,
                'patterns': [],
                'confidence': 0.0,
                'risk_level': 'low'
            }
            
            # Son 100 order'ı analiz et
            recent_orders = list(self.order_history)[-100:]
            
            # 1. Large Order Cancellation Detection
            large_cancellations = self._detect_large_order_cancellations(recent_orders)
            if large_cancellations['detected']:
                analysis['patterns'].append({
                    'type': SpoofingPattern.LARGE_ORDER_CANCELLATION,
                    'confidence': large_cancellations['confidence'],
                    'details': large_cancellations['details']
                })
            
            # 2. Rapid Order Modification Detection
            rapid_modifications = self._detect_rapid_modifications(recent_orders)
            if rapid_modifications['detected']:
                analysis['patterns'].append({
                    'type': SpoofingPattern.RAPID_ORDER_MODIFICATION,
                    'confidence': rapid_modifications['confidence'],
                    'details': rapid_modifications['details']
                })
            
            # 3. Layer Manipulation Detection
            layer_manipulation = self._detect_layer_manipulation(snapshot)
            if layer_manipulation['detected']:
                analysis['patterns'].append({
                    'type': SpoofingPattern.LAYER_MANIPULATION,
                    'confidence': layer_manipulation['confidence'],
                    'details': layer_manipulation['details']
                })
            
            # 4. Momentum Ignition Detection
            momentum_ignition = self._detect_momentum_ignition(snapshot, recent_orders)
            if momentum_ignition['detected']:
                analysis['patterns'].append({
                    'type': SpoofingPattern.MOMENTUM_IGNITION,
                    'confidence': momentum_ignition['confidence'],
                    'details': momentum_ignition['details']
                })
            
            # Genel analiz
            if analysis['patterns']:
                analysis['detected'] = True
                analysis['confidence'] = max([p['confidence'] for p in analysis['patterns']])
                
                # Risk seviyesi belirle
                if analysis['confidence'] > 0.8:
                    analysis['risk_level'] = 'high'
                elif analysis['confidence'] > 0.6:
                    analysis['risk_level'] = 'medium'
                else:
                    analysis['risk_level'] = 'low'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Spoofing detection hatası: {e}")
            return {'timestamp': snapshot.timestamp, 'symbol': snapshot.symbol, 'detected': False}
    
    def _detect_large_order_cancellations(self, orders: List[Dict]) -> Dict[str, Any]:
        """Büyük emir iptalleri tespit et"""
        try:
            result = {'detected': False, 'confidence': 0.0, 'details': []}
            
            # Son 5 dakikadaki emirleri analiz et
            cutoff_time = datetime.now() - timedelta(minutes=5)
            recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
            
            # Create -> Cancel pattern'leri bul
            creates = [o for o in recent_orders if o['event']['action'] == 'create']
            cancels = [o for o in recent_orders if o['event']['action'] == 'cancel']
            
            for create_order in creates:
                create_time = create_order['timestamp']
                order_id = create_order['event']['order_id']
                order_size = create_order['event']['quantity']
                
                # Aynı order_id'li cancel'ları bul
                matching_cancels = [
                    c for c in cancels 
                    if c['event']['order_id'] == order_id and 
                    c['timestamp'] > create_time
                ]
                
                for cancel_order in matching_cancels:
                    time_diff = (cancel_order['timestamp'] - create_time).total_seconds()
                    
                    # Büyük emir ve hızlı iptal kontrolü
                    if (order_size > self.spoofing_config['large_order_threshold'] and 
                        time_diff < self.spoofing_config['cancellation_time_threshold']):
                        
                        result['detected'] = True
                        result['confidence'] = min(1.0, result['confidence'] + 0.3)
                        result['details'].append({
                            'order_id': order_id,
                            'size': order_size,
                            'time_to_cancel': time_diff,
                            'create_time': create_time,
                            'cancel_time': cancel_order['timestamp']
                        })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Large order cancellation detection hatası: {e}")
            return {'detected': False, 'confidence': 0.0, 'details': []}
    
    def _detect_rapid_modifications(self, orders: List[Dict]) -> Dict[str, Any]:
        """Hızlı emir değişiklikleri tespit et"""
        try:
            result = {'detected': False, 'confidence': 0.0, 'details': []}
            
            # Son 10 dakikadaki modify'leri analiz et
            cutoff_time = datetime.now() - timedelta(minutes=10)
            recent_modifies = [
                o for o in orders 
                if o['event']['action'] == 'modify' and o['timestamp'] > cutoff_time
            ]
            
            # Order ID'ye göre grupla
            order_modifications = {}
            for modify in recent_modifies:
                order_id = modify['event']['order_id']
                if order_id not in order_modifications:
                    order_modifications[order_id] = []
                order_modifications[order_id].append(modify)
            
            # Her order için modification frequency kontrol et
            for order_id, modifications in order_modifications.items():
                if len(modifications) >= self.spoofing_config['modification_frequency_threshold']:
                    result['detected'] = True
                    result['confidence'] = min(1.0, result['confidence'] + 0.2)
                    result['details'].append({
                        'order_id': order_id,
                        'modification_count': len(modifications),
                        'time_span': (modifications[-1]['timestamp'] - modifications[0]['timestamp']).total_seconds(),
                        'modifications': modifications
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rapid modification detection hatası: {e}")
            return {'detected': False, 'confidence': 0.0, 'details': []}
    
    def _detect_layer_manipulation(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Katman manipülasyonu tespit et"""
        try:
            result = {'detected': False, 'confidence': 0.0, 'details': []}
            
            if not snapshot.bids or not snapshot.asks:
                return result
            
            # Bid ve ask katmanlarını analiz et
            bid_layers = snapshot.bids[:5]  # İlk 5 seviye
            ask_layers = snapshot.asks[:5]
            
            # Katman kalınlığı kontrolü
            for i, (bid, ask) in enumerate(zip(bid_layers, ask_layers)):
                bid_thickness = bid.quantity / snapshot.mid_price
                ask_thickness = ask.quantity / snapshot.mid_price
                
                if (bid_thickness > self.spoofing_config['layer_thickness_threshold'] or 
                    ask_thickness > self.spoofing_config['layer_thickness_threshold']):
                    
                    result['detected'] = True
                    result['confidence'] = min(1.0, result['confidence'] + 0.15)
                    result['details'].append({
                        'level': i + 1,
                        'side': 'bid' if bid_thickness > ask_thickness else 'ask',
                        'thickness': max(bid_thickness, ask_thickness),
                        'bid_thickness': bid_thickness,
                        'ask_thickness': ask_thickness
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Layer manipulation detection hatası: {e}")
            return {'detected': False, 'confidence': 0.0, 'details': []}
    
    def _detect_momentum_ignition(self, snapshot: OrderBookSnapshot, orders: List[Dict]) -> Dict[str, Any]:
        """Momentum ateşleme tespit et"""
        try:
            result = {'detected': False, 'confidence': 0.0, 'details': []}
            
            # Son 2 dakikadaki fiyat hareketini analiz et
            if len(self.orderbook_history) < 10:
                return result
            
            recent_snapshots = list(self.orderbook_history)[-10:]
            price_changes = []
            
            for i in range(1, len(recent_snapshots)):
                prev_price = recent_snapshots[i-1].mid_price
                curr_price = recent_snapshots[i].mid_price
                price_change = (curr_price - prev_price) / prev_price
                price_changes.append(price_change)
            
            # Momentum hesapla
            momentum = np.mean(price_changes) if price_changes else 0
            
            # Büyük momentum ve order pattern kontrolü
            if abs(momentum) > self.spoofing_config['momentum_threshold']:
                # Son dakikadaki order pattern'ini kontrol et
                cutoff_time = datetime.now() - timedelta(minutes=1)
                recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
                
                # Tek yönlü order pattern kontrolü
                buy_orders = [o for o in recent_orders if o['event']['side'] == 'buy']
                sell_orders = [o for o in recent_orders if o['event']['side'] == 'sell']
                
                if len(buy_orders) > len(sell_orders) * 3 or len(sell_orders) > len(buy_orders) * 3:
                    result['detected'] = True
                    result['confidence'] = min(1.0, abs(momentum) * 10)  # Momentum'a göre confidence
                    result['details'].append({
                        'momentum': momentum,
                        'buy_orders': len(buy_orders),
                        'sell_orders': len(sell_orders),
                        'order_imbalance': len(buy_orders) - len(sell_orders),
                        'price_changes': price_changes
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Momentum ignition detection hatası: {e}")
            return {'detected': False, 'confidence': 0.0, 'details': []}
    
    def get_orderbook_metrics(self, symbol: str = None) -> Dict[str, Any]:
        """Order book metriklerini al"""
        try:
            with self.lock:
                if not self.orderbook_history:
                    return {}
                
                # Son snapshot'ları al
                recent_snapshots = list(self.orderbook_history)[-100:]
                
                if symbol:
                    recent_snapshots = [s for s in recent_snapshots if s.symbol == symbol]
                
                if not recent_snapshots:
                    return {}
                
                # Metrikleri hesapla
                spreads = [s.spread for s in recent_snapshots]
                mid_prices = [s.mid_price for s in recent_snapshots]
                
                metrics = {
                    'avg_spread': np.mean(spreads),
                    'max_spread': np.max(spreads),
                    'min_spread': np.min(spreads),
                    'spread_volatility': np.std(spreads),
                    'price_volatility': np.std(mid_prices),
                    'snapshot_count': len(recent_snapshots),
                    'last_update': recent_snapshots[-1].timestamp if recent_snapshots else None
                }
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Order book metrikleri alma hatası: {e}")
            return {}
    
    def get_spoofing_alerts(self, symbol: str = None, hours: int = 1) -> List[Dict[str, Any]]:
        """Spoofing uyarılarını al"""
        try:
            with self.lock:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                alerts = []
                for snapshot in self.orderbook_history:
                    if snapshot.timestamp > cutoff_time:
                        if symbol and snapshot.symbol != symbol:
                            continue
                        
                        analysis = self._detect_spoofing_patterns(snapshot)
                        if analysis['detected']:
                            alerts.append(analysis)
                
                return alerts
                
        except Exception as e:
            self.logger.error(f"Spoofing alerts alma hatası: {e}")
            return []
    
    def add_spoofing_callback(self, callback):
        """Spoofing callback'i ekle"""
        self.spoofing_callbacks.append(callback)
    
    def add_imbalance_callback(self, callback):
        """Imbalance callback'i ekle"""
        self.imbalance_callbacks.append(callback)
    
    def add_whale_callback(self, callback):
        """Whale activity callback'i ekle"""
        self.whale_callbacks.append(callback)
    
    def _notify_spoofing_callbacks(self, analysis: Dict[str, Any]):
        """Spoofing callback'lerini çağır"""
        for callback in self.spoofing_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Spoofing callback hatası: {e}")
    
    def _notify_imbalance_callbacks(self, analysis: Dict[str, Any]):
        """Imbalance callback'lerini çağır"""
        for callback in self.imbalance_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Imbalance callback hatası: {e}")
    
    def _notify_whale_callbacks(self, analysis: Dict[str, Any]):
        """Whale activity callback'lerini çağır"""
        for callback in self.whale_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Whale callback hatası: {e}")
    
    def _calculate_order_flow_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """Order flow imbalance hesapla"""
        try:
            if not snapshot.bids or not snapshot.asks:
                return 0.0
            
            # Bid ve ask hacimlerini topla
            bid_volume = sum(level.quantity for level in snapshot.bids[:5])
            ask_volume = sum(level.quantity for level in snapshot.asks[:5])
            
            if bid_volume + ask_volume == 0:
                return 0.0
            
            # Imbalance oranı (-1 ile 1 arası)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return imbalance
            
        except Exception as e:
            self.logger.error(f"Order flow imbalance hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_market_impact(self, snapshot: OrderBookSnapshot) -> float:
        """Market impact hesapla"""
        try:
            if not snapshot.bids or not snapshot.asks:
                return 0.0
            
            # Spread'in mid price'a oranı
            spread_ratio = snapshot.spread / snapshot.mid_price if snapshot.mid_price > 0 else 0.0
            
            # Likidite skoruna göre ayarla
            liquidity_score = self._calculate_liquidity_score(snapshot)
            impact = spread_ratio * (1.0 - liquidity_score)
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Market impact hesaplama hatası: {e}")
            return 0.0
    
    def _identify_support_resistance(self, snapshot: OrderBookSnapshot) -> List[Dict[str, Any]]:
        """Support/Resistance seviyelerini tespit et"""
        try:
            levels = []
            
            # Büyük hacimli seviyeleri tespit et
            for level in snapshot.bids[:10]:
                if level.quantity > np.mean([l.quantity for l in snapshot.bids[:10]]) * 1.5:
                    levels.append({
                        'price': level.price,
                        'type': 'support',
                        'strength': level.quantity,
                        'timestamp': level.timestamp
                    })
            
            for level in snapshot.asks[:10]:
                if level.quantity > np.mean([l.quantity for l in snapshot.asks[:10]]) * 1.5:
                    levels.append({
                        'price': level.price,
                        'type': 'resistance',
                        'strength': level.quantity,
                        'timestamp': level.timestamp
                    })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Support/Resistance tespit hatası: {e}")
            return []
    
    def _detect_whale_activity(self, snapshot: OrderBookSnapshot) -> bool:
        """Whale aktivitesi tespit et"""
        try:
            if not snapshot.bids or not snapshot.asks:
                return False
            
            # Büyük emirleri tespit et
            large_order_threshold = np.mean([l.quantity for l in snapshot.bids + snapshot.asks]) * 3
            
            for level in snapshot.bids + snapshot.asks:
                if level.quantity > large_order_threshold:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Whale aktivitesi tespit hatası: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            if 'spoofing' in new_config:
                self.spoofing_config.update(new_config['spoofing'])
            
            if 'depth' in new_config:
                self.depth_config.update(new_config['depth'])
            
            self.logger.info("Order book analizör konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global order book analizör
orderbook_analyzer = OrderBookAnalyzer()
