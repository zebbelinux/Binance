"""
Hedge/Korelasyon/Lead-Lag Trading Stratejileri
Pariteler arası korelasyon ve hedge stratejileri
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
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from strategies.base_strategy import BaseStrategy
import sqlite3
import os
from pathlib import Path
import pickle
import json

@dataclass
class CorrelationPair:
    """Korelasyon çifti"""
    symbol1: str
    symbol2: str
    correlation: float
    p_value: float
    confidence: float
    last_updated: datetime

@dataclass
class HedgePosition:
    """Hedge pozisyonu"""
    primary_symbol: str
    hedge_symbol: str
    primary_side: str  # 'long' or 'short'
    hedge_side: str   # 'long' or 'short'
    primary_size: float
    hedge_size: float
    hedge_ratio: float
    correlation: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

class HedgeStrategyType(Enum):
    """Hedge strateji türleri"""
    PAIRS_TRADING = "pairs_trading"
    CORRELATION_HEDGE = "correlation_hedge"
    LEAD_LAG_TRADING = "lead_lag_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_NEUTRAL = "market_neutral"

class HedgeStrategy(BaseStrategy):
    """Hedge/Korelasyon trading stratejisi"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Hedge", config)
        
        # Strateji parametreleri
        self.strategy_type = HedgeStrategyType(config.get('strategy_type', 'pairs_trading'))
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.min_correlation_period = config.get('min_correlation_period', 20)  # 20 bar
        self.max_correlation_period = config.get('max_correlation_period', 100)  # 100 bar
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 0.5)
        
        # Risk parametreleri
        self.max_hedge_ratio = config.get('max_hedge_ratio', 0.5)  # %50 hedge
        self.stop_loss_threshold = config.get('stop_loss_threshold', 0.05)  # %5 stop loss
        self.take_profit_threshold = config.get('take_profit_threshold', 0.03)  # %3 take profit
        
        # Lead-lag parametreleri
        self.lead_lag_window = config.get('lead_lag_window', 5)  # 5 bar
        self.cross_correlation_threshold = config.get('cross_correlation_threshold', 0.6)
        
        # Veri saklama
        self.price_data = {}  # {symbol: [prices]}
        self.correlation_pairs = {}  # {pair_key: CorrelationPair}
        self.hedge_positions = {}  # {position_id: HedgePosition}
        self.spread_history = {}  # {pair_key: [spreads]}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Analiz thread'i
        self.analysis_thread = None
        self.is_analyzing = False
        
        self.logger.info(f"Hedge stratejisi oluşturuldu - Tip: {self.strategy_type.value}")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sinyal üret"""
        try:
            signals = []
            
            # Piyasa verilerini güncelle
            self._update_price_data(market_data)
            
            if self.strategy_type == HedgeStrategyType.PAIRS_TRADING:
                signals.extend(self._generate_pairs_trading_signals())
            elif self.strategy_type == HedgeStrategyType.CORRELATION_HEDGE:
                signals.extend(self._generate_correlation_hedge_signals())
            elif self.strategy_type == HedgeStrategyType.LEAD_LAG_TRADING:
                signals.extend(self._generate_lead_lag_signals())
            elif self.strategy_type == HedgeStrategyType.STATISTICAL_ARBITRAGE:
                signals.extend(self._generate_statistical_arbitrage_signals())
            elif self.strategy_type == HedgeStrategyType.MARKET_NEUTRAL:
                signals.extend(self._generate_market_neutral_signals())
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Hedge sinyal üretme hatası: {e}")
            return []
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """Pozisyon büyüklüğü hesapla"""
        try:
            # Kelly Criterion ile pozisyon büyüklüğü hesapla
            win_rate = signal.get('win_rate', 0.5)
            avg_win = signal.get('avg_win', 0.02)
            avg_loss = signal.get('avg_loss', 0.01)
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max %25
            else:
                kelly_fraction = 0.1  # Varsayılan %10
            
            # Hedge stratejisi için daha konservatif
            hedge_multiplier = 0.5 if signal.get('strategy_type') == 'hedge' else 1.0
            
            position_size = account_balance * kelly_fraction * hedge_multiplier
            
            return min(position_size, account_balance * self.max_hedge_ratio)
            
        except Exception as e:
            self.logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return account_balance * 0.05  # Varsayılan %5
    
    def _update_price_data(self, market_data: Dict[str, Any]):
        """Fiyat verilerini güncelle"""
        try:
            symbol = market_data.get('symbol')
            price = market_data.get('price')
            
            if not symbol or not price:
                return
            
            with self.lock:
                if symbol not in self.price_data:
                    self.price_data[symbol] = []
                
                self.price_data[symbol].append({
                    'timestamp': datetime.now(),
                    'price': price,
                    'volume': market_data.get('volume', 0)
                })
                
                # Son 1000 veriyi sakla
                if len(self.price_data[symbol]) > 1000:
                    self.price_data[symbol] = self.price_data[symbol][-1000:]
                
        except Exception as e:
            self.logger.error(f"Fiyat verisi güncelleme hatası: {e}")
    
    def _generate_pairs_trading_signals(self) -> List[Dict[str, Any]]:
        """Pairs trading sinyalleri üret"""
        try:
            signals = []
            
            # Mevcut sembolleri al
            symbols = list(self.price_data.keys())
            if len(symbols) < 2:
                return signals
            
            # Tüm sembol çiftlerini analiz et
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    # Korelasyon analizi
                    correlation_analysis = self._analyze_correlation(symbol1, symbol2)
                    
                    if correlation_analysis['correlation'] > self.correlation_threshold:
                        # Spread analizi
                        spread_analysis = self._analyze_spread(symbol1, symbol2)
                        
                        if spread_analysis['z_score'] > self.z_score_threshold:
                            # Mean reversion sinyali
                            signal = self._create_pairs_trading_signal(
                                symbol1, symbol2, correlation_analysis, spread_analysis
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Pairs trading sinyal üretme hatası: {e}")
            return []
    
    def _generate_correlation_hedge_signals(self) -> List[Dict[str, Any]]:
        """Korelasyon hedge sinyalleri üret"""
        try:
            signals = []
            
            # Mevcut pozisyonları kontrol et
            for position_id, position in self.hedge_positions.items():
                # Korelasyon değişimini kontrol et
                current_correlation = self._get_current_correlation(
                    position.primary_symbol, position.hedge_symbol
                )
                
                # Korelasyon düştüyse hedge'i kapat
                if current_correlation < self.correlation_threshold * 0.8:
                    signal = {
                        'id': f"hedge_close_{position_id}",
                        'symbol': position.primary_symbol,
                        'side': 'close',
                        'strength': 0.8,
                        'reason': f"Korelasyon düştü: {current_correlation:.3f}",
                        'strategy_name': 'correlation_hedge',
                        'timestamp': datetime.now(),
                        'position_id': position_id,
                        'hedge_symbol': position.hedge_symbol
                    }
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Korelasyon hedge sinyal üretme hatası: {e}")
            return []
    
    def _generate_lead_lag_signals(self) -> List[Dict[str, Any]]:
        """Lead-lag trading sinyalleri üret"""
        try:
            signals = []
            
            symbols = list(self.price_data.keys())
            if len(symbols) < 2:
                return signals
            
            # Cross-correlation analizi
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    # Lead-lag analizi
                    lead_lag_analysis = self._analyze_lead_lag(symbol1, symbol2)
                    
                    if lead_lag_analysis['max_correlation'] > self.cross_correlation_threshold:
                        # Lead-lag sinyali üret
                        signal = self._create_lead_lag_signal(
                            symbol1, symbol2, lead_lag_analysis
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Lead-lag sinyal üretme hatası: {e}")
            return []
    
    def _generate_statistical_arbitrage_signals(self) -> List[Dict[str, Any]]:
        """İstatistiksel arbitraj sinyalleri üret"""
        try:
            signals = []
            
            # Çoklu sembol analizi
            symbols = list(self.price_data.keys())
            if len(symbols) < 3:
                return signals
            
            # Cointegration test
            cointegration_results = self._test_cointegration(symbols)
            
            for result in cointegration_results:
                if result['is_cointegrated']:
                    # Arbitraj fırsatı
                    signal = self._create_arbitrage_signal(result)
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"İstatistiksel arbitraj sinyal üretme hatası: {e}")
            return []
    
    def _generate_market_neutral_signals(self) -> List[Dict[str, Any]]:
        """Market neutral sinyalleri üret"""
        try:
            signals = []
            
            # Market beta hesapla
            market_beta = self._calculate_market_beta()
            
            # Beta'ya göre pozisyon ayarla
            for symbol in self.price_data.keys():
                symbol_beta = market_beta.get(symbol, 1.0)
                
                if abs(symbol_beta) > 0.8:  # Yüksek beta
                    # Market'e karşı pozisyon
                    signal = {
                        'id': f"market_neutral_{symbol}",
                        'symbol': symbol,
                        'side': 'short' if symbol_beta > 0 else 'long',
                        'strength': min(abs(symbol_beta), 1.0),
                        'reason': f"Market neutral - Beta: {symbol_beta:.3f}",
                        'strategy_name': 'market_neutral',
                        'timestamp': datetime.now(),
                        'beta': symbol_beta
                    }
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Market neutral sinyal üretme hatası: {e}")
            return []
    
    def _analyze_correlation(self, symbol1: str, symbol2: str) -> Dict[str, Any]:
        """Korelasyon analizi"""
        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return {'correlation': 0, 'p_value': 1, 'confidence': 0}
            
            # Son N bar'ı al
            data1 = self.price_data[symbol1][-self.max_correlation_period:]
            data2 = self.price_data[symbol2][-self.max_correlation_period:]
            
            if len(data1) < self.min_correlation_period or len(data2) < self.min_correlation_period:
                return {'correlation': 0, 'p_value': 1, 'confidence': 0}
            
            # Fiyat serilerini hazırla
            prices1 = [d['price'] for d in data1]
            prices2 = [d['price'] for d in data2]
            
            # Korelasyon hesapla
            correlation, p_value = stats.pearsonr(prices1, prices2)
            
            # Confidence hesapla
            confidence = 1 - p_value if not np.isnan(p_value) else 0
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                'confidence': confidence,
                'period': len(data1)
            }
            
        except Exception as e:
            self.logger.error(f"Korelasyon analiz hatası ({symbol1}-{symbol2}): {e}")
            return {'correlation': 0, 'p_value': 1, 'confidence': 0}
    
    def _analyze_spread(self, symbol1: str, symbol2: str) -> Dict[str, Any]:
        """Spread analizi"""
        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return {'z_score': 0, 'mean': 0, 'std': 1}
            
            # Son fiyatları al
            price1 = self.price_data[symbol1][-1]['price']
            price2 = self.price_data[symbol2][-1]['price']
            
            # Spread hesapla (normalize edilmiş)
            spread = (price1 - price2) / price2
            
            # Geçmiş spread'leri al
            pair_key = f"{symbol1}_{symbol2}"
            if pair_key not in self.spread_history:
                self.spread_history[pair_key] = []
            
            self.spread_history[pair_key].append(spread)
            
            # Son 100 spread'i sakla
            if len(self.spread_history[pair_key]) > 100:
                self.spread_history[pair_key] = self.spread_history[pair_key][-100:]
            
            # Z-score hesapla
            if len(self.spread_history[pair_key]) >= 20:
                spreads = self.spread_history[pair_key]
                mean_spread = np.mean(spreads)
                std_spread = np.std(spreads)
                
                if std_spread > 0:
                    z_score = (spread - mean_spread) / std_spread
                else:
                    z_score = 0
                
                return {
                    'z_score': z_score,
                    'mean': mean_spread,
                    'std': std_spread,
                    'current_spread': spread
                }
            
            return {'z_score': 0, 'mean': 0, 'std': 1, 'current_spread': spread}
            
        except Exception as e:
            self.logger.error(f"Spread analiz hatası ({symbol1}-{symbol2}): {e}")
            return {'z_score': 0, 'mean': 0, 'std': 1}
    
    def _analyze_lead_lag(self, symbol1: str, symbol2: str) -> Dict[str, Any]:
        """Lead-lag analizi"""
        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return {'max_correlation': 0, 'lag': 0, 'lead_symbol': None}
            
            # Son verileri al
            data1 = self.price_data[symbol1][-50:]  # Son 50 bar
            data2 = self.price_data[symbol2][-50:]
            
            if len(data1) < 20 or len(data2) < 20:
                return {'max_correlation': 0, 'lag': 0, 'lead_symbol': None}
            
            # Fiyat serilerini hazırla
            prices1 = [d['price'] for d in data1]
            prices2 = [d['price'] for d in data2]
            
            # Cross-correlation hesapla
            max_correlation = 0
            best_lag = 0
            lead_symbol = None
            
            for lag in range(-self.lead_lag_window, self.lead_lag_window + 1):
                if lag == 0:
                    continue
                
                if lag > 0:
                    # symbol1 leads symbol2
                    if len(prices1) > lag and len(prices2) > lag:
                        series1 = prices1[:-lag] if lag > 0 else prices1
                        series2 = prices2[lag:] if lag > 0 else prices2
                        
                        if len(series1) == len(series2) and len(series1) > 10:
                            correlation = np.corrcoef(series1, series2)[0, 1]
                            if not np.isnan(correlation) and abs(correlation) > abs(max_correlation):
                                max_correlation = correlation
                                best_lag = lag
                                lead_symbol = symbol1
                else:
                    # symbol2 leads symbol1
                    lag_abs = abs(lag)
                    if len(prices2) > lag_abs and len(prices1) > lag_abs:
                        series2 = prices2[:-lag_abs] if lag_abs > 0 else prices2
                        series1 = prices1[lag_abs:] if lag_abs > 0 else prices1
                        
                        if len(series1) == len(series2) and len(series1) > 10:
                            correlation = np.corrcoef(series1, series2)[0, 1]
                            if not np.isnan(correlation) and abs(correlation) > abs(max_correlation):
                                max_correlation = correlation
                                best_lag = lag
                                lead_symbol = symbol2
            
            return {
                'max_correlation': max_correlation,
                'lag': best_lag,
                'lead_symbol': lead_symbol
            }
            
        except Exception as e:
            self.logger.error(f"Lead-lag analiz hatası ({symbol1}-{symbol2}): {e}")
            return {'max_correlation': 0, 'lag': 0, 'lead_symbol': None}
    
    def _test_cointegration(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Cointegration test"""
        try:
            results = []
            
            # İki sembol için cointegration test
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    if symbol1 not in self.price_data or symbol2 not in self.price_data:
                        continue
                    
                    # Son verileri al
                    data1 = self.price_data[symbol1][-100:]
                    data2 = self.price_data[symbol2][-100:]
                    
                    if len(data1) < 50 or len(data2) < 50:
                        continue
                    
                    # Fiyat serilerini hazırla
                    prices1 = [d['price'] for d in data1]
                    prices2 = [d['price'] for d in data2]
                    
                    # Engle-Granger cointegration test
                    try:
                        from statsmodels.tsa.stattools import coint
                        score, p_value, _ = coint(prices1, prices2)
                        
                        is_cointegrated = p_value < 0.05
                        
                        results.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'is_cointegrated': is_cointegrated,
                            'p_value': p_value,
                            'score': score,
                            'confidence': 1 - p_value if not np.isnan(p_value) else 0
                        })
                        
                    except ImportError:
                        # Basit korelasyon testi
                        correlation = np.corrcoef(prices1, prices2)[0, 1]
                        is_cointegrated = abs(correlation) > 0.8
                        
                        results.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'is_cointegrated': is_cointegrated,
                            'p_value': 1 - abs(correlation),
                            'score': correlation,
                            'confidence': abs(correlation)
                        })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cointegration test hatası: {e}")
            return []
    
    def _calculate_market_beta(self) -> Dict[str, float]:
        """Market beta hesapla"""
        try:
            betas = {}
            
            # Market proxy olarak BTC kullan
            market_symbol = 'BTCUSDT'
            if market_symbol not in self.price_data:
                return betas
            
            market_data = self.price_data[market_symbol]
            if len(market_data) < 50:
                return betas
            
            market_prices = [d['price'] for d in market_data[-50:]]
            market_returns = np.diff(market_prices) / market_prices[:-1]
            
            for symbol in self.price_data.keys():
                if symbol == market_symbol:
                    betas[symbol] = 1.0
                    continue
                
                symbol_data = self.price_data[symbol]
                if len(symbol_data) < 50:
                    continue
                
                symbol_prices = [d['price'] for d in symbol_data[-50:]]
                symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
                
                # Beta hesapla
                if len(symbol_returns) == len(market_returns) and len(symbol_returns) > 10:
                    covariance = np.cov(symbol_returns, market_returns)[0, 1]
                    market_variance = np.var(market_returns)
                    
                    if market_variance > 0:
                        beta = covariance / market_variance
                        betas[symbol] = beta
            
            return betas
            
        except Exception as e:
            self.logger.error(f"Market beta hesaplama hatası: {e}")
            return {}
    
    def _create_pairs_trading_signal(self, symbol1: str, symbol2: str, 
                                   correlation_analysis: Dict, spread_analysis: Dict) -> Dict[str, Any]:
        """Pairs trading sinyali oluştur"""
        try:
            z_score = spread_analysis['z_score']
            
            # Z-score'a göre pozisyon belirle
            if z_score > self.z_score_threshold:
                # Spread yüksek, short spread (long symbol2, short symbol1)
                primary_side = 'short'
                hedge_side = 'long'
                primary_symbol = symbol1
                hedge_symbol = symbol2
            elif z_score < -self.z_score_threshold:
                # Spread düşük, long spread (long symbol1, short symbol2)
                primary_side = 'long'
                hedge_side = 'short'
                primary_symbol = symbol1
                hedge_symbol = symbol2
            else:
                return None
            
            signal = {
                'id': f"pairs_{primary_symbol}_{hedge_symbol}_{datetime.now().timestamp()}",
                'symbol': primary_symbol,
                'side': primary_side,
                'strength': min(abs(z_score) / self.z_score_threshold, 1.0),
                'reason': f"Pairs trading - Z-score: {z_score:.3f}, Korelasyon: {correlation_analysis['correlation']:.3f}",
                'strategy_name': 'pairs_trading',
                'timestamp': datetime.now(),
                'hedge_symbol': hedge_symbol,
                'hedge_side': hedge_side,
                'z_score': z_score,
                'correlation': correlation_analysis['correlation'],
                'spread_mean': spread_analysis['mean'],
                'spread_std': spread_analysis['std']
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Pairs trading sinyal oluşturma hatası: {e}")
            return None
    
    def _create_lead_lag_signal(self, symbol1: str, symbol2: str, 
                              lead_lag_analysis: Dict) -> Dict[str, Any]:
        """Lead-lag sinyali oluştur"""
        try:
            lead_symbol = lead_lag_analysis['lead_symbol']
            lag = lead_lag_analysis['lag']
            correlation = lead_lag_analysis['max_correlation']
            
            if not lead_symbol:
                return None
            
            # Lead symbol'ün fiyat hareketini takip et
            if lead_symbol == symbol1:
                primary_symbol = symbol1
                hedge_symbol = symbol2
            else:
                primary_symbol = symbol2
                hedge_symbol = symbol1
            
            # Momentum'a göre pozisyon belirle
            primary_data = self.price_data[primary_symbol]
            if len(primary_data) < 5:
                return None
            
            recent_prices = [d['price'] for d in primary_data[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if momentum > 0.01:  # %1'den büyük momentum
                primary_side = 'long'
                hedge_side = 'short'
            elif momentum < -0.01:  # %1'den küçük momentum
                primary_side = 'short'
                hedge_side = 'long'
            else:
                return None
            
            signal = {
                'id': f"lead_lag_{primary_symbol}_{hedge_symbol}_{datetime.now().timestamp()}",
                'symbol': primary_symbol,
                'side': primary_side,
                'strength': min(abs(correlation), 1.0),
                'reason': f"Lead-lag trading - Lead: {lead_symbol}, Lag: {lag}, Korelasyon: {correlation:.3f}",
                'strategy_name': 'lead_lag_trading',
                'timestamp': datetime.now(),
                'hedge_symbol': hedge_symbol,
                'hedge_side': hedge_side,
                'lead_symbol': lead_symbol,
                'lag': lag,
                'correlation': correlation,
                'momentum': momentum
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Lead-lag sinyal oluşturma hatası: {e}")
            return None
    
    def _create_arbitrage_signal(self, cointegration_result: Dict) -> Dict[str, Any]:
        """Arbitraj sinyali oluştur"""
        try:
            symbol1 = cointegration_result['symbol1']
            symbol2 = cointegration_result['symbol2']
            
            # Mevcut fiyatları al
            price1 = self.price_data[symbol1][-1]['price']
            price2 = self.price_data[symbol2][-1]['price']
            
            # Fiyat oranına göre pozisyon belirle
            ratio = price1 / price2
            historical_ratio = self._get_historical_ratio(symbol1, symbol2)
            
            if ratio > historical_ratio * 1.02:  # %2'den büyük sapma
                # symbol1 pahalı, symbol2 ucuz
                primary_side = 'short'
                hedge_side = 'long'
                primary_symbol = symbol1
                hedge_symbol = symbol2
            elif ratio < historical_ratio * 0.98:  # %2'den küçük sapma
                # symbol1 ucuz, symbol2 pahalı
                primary_side = 'long'
                hedge_side = 'short'
                primary_symbol = symbol1
                hedge_symbol = symbol2
            else:
                return None
            
            signal = {
                'id': f"arbitrage_{primary_symbol}_{hedge_symbol}_{datetime.now().timestamp()}",
                'symbol': primary_symbol,
                'side': primary_side,
                'strength': min(abs(ratio - historical_ratio) / historical_ratio * 10, 1.0),
                'reason': f"İstatistiksel arbitraj - Oran: {ratio:.4f}, Tarihsel: {historical_ratio:.4f}",
                'strategy_name': 'statistical_arbitrage',
                'timestamp': datetime.now(),
                'hedge_symbol': hedge_symbol,
                'hedge_side': hedge_side,
                'current_ratio': ratio,
                'historical_ratio': historical_ratio,
                'deviation': abs(ratio - historical_ratio) / historical_ratio
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Arbitraj sinyal oluşturma hatası: {e}")
            return None
    
    def _get_historical_ratio(self, symbol1: str, symbol2: str) -> float:
        """Tarihsel fiyat oranını al"""
        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                return 1.0
            
            data1 = self.price_data[symbol1][-50:]  # Son 50 bar
            data2 = self.price_data[symbol2][-50:]
            
            if len(data1) < 20 or len(data2) < 20:
                return 1.0
            
            ratios = []
            for i in range(min(len(data1), len(data2))):
                ratio = data1[i]['price'] / data2[i]['price']
                ratios.append(ratio)
            
            return np.mean(ratios) if ratios else 1.0
            
        except Exception as e:
            self.logger.error(f"Tarihsel oran hesaplama hatası: {e}")
            return 1.0
    
    def _get_current_correlation(self, symbol1: str, symbol2: str) -> float:
        """Mevcut korelasyonu al"""
        try:
            analysis = self._analyze_correlation(symbol1, symbol2)
            return analysis['correlation']
        except Exception as e:
            self.logger.error(f"Mevcut korelasyon alma hatası: {e}")
            return 0.0
    
    def add_hedge_position(self, position: HedgePosition):
        """Hedge pozisyonu ekle"""
        try:
            with self.lock:
                position_id = f"hedge_{position.primary_symbol}_{position.hedge_symbol}_{datetime.now().timestamp()}"
                self.hedge_positions[position_id] = position
                
        except Exception as e:
            self.logger.error(f"Hedge pozisyon ekleme hatası: {e}")
    
    def remove_hedge_position(self, position_id: str):
        """Hedge pozisyonu kaldır"""
        try:
            with self.lock:
                if position_id in self.hedge_positions:
                    del self.hedge_positions[position_id]
                    
        except Exception as e:
            self.logger.error(f"Hedge pozisyon kaldırma hatası: {e}")
    
    def get_hedge_positions(self) -> Dict[str, HedgePosition]:
        """Hedge pozisyonlarını al"""
        try:
            with self.lock:
                return self.hedge_positions.copy()
        except Exception as e:
            self.logger.error(f"Hedge pozisyonları alma hatası: {e}")
            return {}
    
    def get_correlation_pairs(self) -> Dict[str, CorrelationPair]:
        """Korelasyon çiftlerini al"""
        try:
            with self.lock:
                return self.correlation_pairs.copy()
        except Exception as e:
            self.logger.error(f"Korelasyon çiftleri alma hatası: {e}")
            return {}
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            # Parametreleri güncelle
            if 'correlation_threshold' in new_config:
                self.correlation_threshold = new_config['correlation_threshold']
            if 'z_score_threshold' in new_config:
                self.z_score_threshold = new_config['z_score_threshold']
            if 'max_hedge_ratio' in new_config:
                self.max_hedge_ratio = new_config['max_hedge_ratio']
            
            self.logger.info("Hedge stratejisi konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Hedge stratejisi instance'ları
hedge_strategies = {
    'pairs_trading': HedgeStrategy({
        'strategy_type': 'pairs_trading',
        'correlation_threshold': 0.7,
        'z_score_threshold': 2.0,
        'max_hedge_ratio': 0.5
    }),
    'correlation_hedge': HedgeStrategy({
        'strategy_type': 'correlation_hedge',
        'correlation_threshold': 0.8,
        'max_hedge_ratio': 0.3
    }),
    'lead_lag_trading': HedgeStrategy({
        'strategy_type': 'lead_lag_trading',
        'cross_correlation_threshold': 0.6,
        'lead_lag_window': 5
    }),
    'statistical_arbitrage': HedgeStrategy({
        'strategy_type': 'statistical_arbitrage',
        'correlation_threshold': 0.8,
        'max_hedge_ratio': 0.4
    }),
    'market_neutral': HedgeStrategy({
        'strategy_type': 'market_neutral',
        'max_hedge_ratio': 0.6
    })
}
