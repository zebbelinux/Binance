"""
Portföy Optimizasyonu Modülü
Markowitz Modern Portfolio Theory ve Kelly Criterion ile risk/kar optimizasyonu
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
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
from pathlib import Path
import sqlite3
import os

# Optimizasyon kütüphaneleri
try:
    from scipy.optimize import minimize
    from scipy.linalg import cholesky
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

@dataclass
class Asset:
    """Varlık bilgisi"""
    symbol: str
    expected_return: float
    volatility: float
    weight: float = 0.0
    beta: float = 1.0
    sharpe_ratio: float = 0.0

@dataclass
class Portfolio:
    """Portföy"""
    assets: List[Asset]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    max_drawdown: float
    diversification_ratio: float
    timestamp: datetime

@dataclass
class OptimizationResult:
    """Optimizasyon sonucu"""
    optimal_weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    max_drawdown: float
    optimization_method: str
    timestamp: datetime

class OptimizationMethod(Enum):
    """Optimizasyon yöntemleri"""
    MARKOWITZ = "markowitz"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MINIMUM_VARIANCE = "minimum_variance"
    EQUAL_WEIGHT = "equal_weight"

class PortfolioOptimizer:
    """Portföy optimizasyon sınıfı"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Optimizasyon parametreleri
        self.optimization_config = {
            'risk_free_rate': 0.02,  # %2 risk-free rate
            'max_weight': 0.4,  # Maksimum %40 ağırlık
            'min_weight': 0.01,  # Minimum %1 ağırlık
            'target_return': None,  # Hedef getiri
            'target_volatility': None,  # Hedef volatilite
            'rebalance_frequency': 'daily',  # Yeniden dengeleme sıklığı
            'transaction_costs': 0.001,  # %0.1 işlem maliyeti
            'max_turnover': 0.2,  # Maksimum %20 turnover
            'constraints': {
                'long_only': True,  # Sadece long pozisyonlar
                'max_concentration': 0.3,  # Maksimum %30 konsantrasyon
                'min_diversification': 0.5  # Minimum %50 çeşitlendirme
            }
        }
        
        # Portföy geçmişi
        self.portfolio_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.optimization_callbacks = []
        self.rebalance_callbacks = []
        
        self.logger.info("Portföy optimizasyon modülü başlatıldı")
    
    def optimize_portfolio(self, 
                          assets: List[Asset],
                          method: OptimizationMethod = OptimizationMethod.MAXIMUM_SHARPE,
                          returns_data: pd.DataFrame = None,
                          covariance_matrix: np.ndarray = None) -> OptimizationResult:
        """Portföy optimizasyonu yap"""
        try:
            self.logger.info(f"Portföy optimizasyonu başlatılıyor - Yöntem: {method.value}")
            
            # Varlık sayısını kontrol et
            if len(assets) < 2:
                self.logger.error("En az 2 varlık gerekli")
                return None
            
            # Getiri ve kovaryans matrisini hesapla
            if returns_data is not None:
                expected_returns, cov_matrix = self._calculate_statistics(returns_data)
            elif covariance_matrix is not None:
                expected_returns = np.array([asset.expected_return for asset in assets])
                cov_matrix = covariance_matrix
            else:
                # Varsayılan istatistikler
                expected_returns, cov_matrix = self._estimate_statistics(assets)
            
            # Optimizasyon yöntemine göre ağırlıkları hesapla
            if method == OptimizationMethod.MARKOWITZ:
                weights = self._markowitz_optimization(expected_returns, cov_matrix)
            elif method == OptimizationMethod.KELLY_CRITERION:
                weights = self._kelly_criterion_optimization(assets, expected_returns, cov_matrix)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(cov_matrix)
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                weights = self._maximum_sharpe_optimization(expected_returns, cov_matrix)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = self._minimum_variance_optimization(cov_matrix)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                weights = self._equal_weight_optimization(len(assets))
            else:
                raise ValueError(f"Desteklenmeyen optimizasyon yöntemi: {method}")
            
            # Ağırlıkları normalize et
            weights = self._normalize_weights(weights)
            
            # Portföy metriklerini hesapla
            portfolio_metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            # Sonucu oluştur
            result = OptimizationResult(
                optimal_weights={assets[i].symbol: weights[i] for i in range(len(assets))},
                expected_return=portfolio_metrics['expected_return'],
                volatility=portfolio_metrics['volatility'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                var_95=portfolio_metrics['var_95'],
                var_99=portfolio_metrics['var_99'],
                max_drawdown=portfolio_metrics['max_drawdown'],
                optimization_method=method.value,
                timestamp=datetime.now()
            )
            
            # Sonucu kaydet
            with self.lock:
                self.optimization_history.append(result)
            
            # Callback'leri çağır
            self._notify_optimization_callbacks(result)
            
            self.logger.info(f"Portföy optimizasyonu tamamlandı - Sharpe: {result.sharpe_ratio:.3f}")
            
            return result
                
        except Exception as e:
            self.logger.error(f"Portföy optimizasyon hatası: {e}")
            return None
    
    def _calculate_statistics(self, returns_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Getiri verilerinden istatistikleri hesapla"""
        try:
            # Beklenen getiriler
            expected_returns = returns_data.mean().values * 252  # Yıllık
            
            # Kovaryans matrisi
            cov_matrix = returns_data.cov().values * 252  # Yıllık
            
            return expected_returns, cov_matrix
            
        except Exception as e:
            self.logger.error(f"İstatistik hesaplama hatası: {e}")
            return None, None
    
    def _estimate_statistics(self, assets: List[Asset]) -> Tuple[np.ndarray, np.ndarray]:
        """Varlık bilgilerinden istatistikleri tahmin et"""
        try:
            n = len(assets)
            expected_returns = np.array([asset.expected_return for asset in assets])
            
            # Basit kovaryans matrisi tahmini
            cov_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        cov_matrix[i, j] = assets[i].volatility ** 2
                    else:
                        # Korelasyon tahmini (basit)
                        correlation = 0.3  # Varsayılan korelasyon
                        cov_matrix[i, j] = correlation * assets[i].volatility * assets[j].volatility
            
            return expected_returns, cov_matrix
            
        except Exception as e:
            self.logger.error(f"İstatistik tahmin hatası: {e}")
            return None, None
    
    def _markowitz_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Markowitz Modern Portfolio Theory optimizasyonu"""
        try:
            n = len(expected_returns)
            
            # Risk toleransı parametresi
            risk_aversion = 3.0
            
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
            
            # Kısıtlamalar
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Ağırlıklar toplamı 1
            ]
            
            # Sınırlar
            bounds = [(self.optimization_config['min_weight'], self.optimization_config['max_weight']) for _ in range(n)]
            
            # Başlangıç noktası
            x0 = np.ones(n) / n
            
            # Optimizasyon
            if SCIPY_AVAILABLE:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x
            else:
                # Basit optimizasyon
                return self._simple_optimization(objective, bounds, constraints, x0)
            
        except Exception as e:
            self.logger.error(f"Markowitz optimizasyon hatası: {e}")
            return np.ones(len(expected_returns)) / len(expected_returns)
    
    def _kelly_criterion_optimization(self, assets: List[Asset], expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Kelly Criterion optimizasyonu"""
        try:
            n = len(assets)
            weights = np.zeros(n)
            
            # Her varlık için Kelly fraksiyonu hesapla
            for i in range(n):
                asset = assets[i]
                
                # Kelly fraksiyonu: f = (bp - q) / b
                # b = odds (1 + expected_return)
                # p = kazanma olasılığı
                # q = kaybetme olasılığı (1 - p)
                
                # Basit Kelly hesaplama
                if asset.expected_return > 0:
                    # Kazanma olasılığını tahmin et
                    win_prob = 0.5 + asset.expected_return * 0.1  # Basit tahmin
                    win_prob = min(0.9, max(0.1, win_prob))  # Sınırla
                    
                    odds = 1 + asset.expected_return
                    kelly_fraction = (odds * win_prob - (1 - win_prob)) / odds
                    
                    # Kelly fraksiyonunu sınırla
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max %25
                    
                    weights[i] = kelly_fraction
                else:
                    weights[i] = 0
            
            # Ağırlıkları normalize et
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n) / n
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Kelly Criterion optimizasyon hatası: {e}")
            return np.ones(len(assets)) / len(assets)
    
    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk Parity optimizasyonu"""
        try:
            n = cov_matrix.shape[0]
            
            def objective(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_variance
                return np.sum((risk_contributions - 1/n) ** 2)
            
            # Kısıtlamalar
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Sınırlar
            bounds = [(self.optimization_config['min_weight'], self.optimization_config['max_weight']) for _ in range(n)]
            
            # Başlangıç noktası
            x0 = np.ones(n) / n
            
            # Optimizasyon
            if SCIPY_AVAILABLE:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            self.logger.error(f"Risk Parity optimizasyon hatası: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    def _maximum_sharpe_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Maksimum Sharpe Ratio optimizasyonu"""
        try:
            n = len(expected_returns)
            risk_free_rate = self.optimization_config['risk_free_rate']
            
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return -float('inf')
                
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Minimize etmek için negatif
            
            # Kısıtlamalar
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Sınırlar
            bounds = [(self.optimization_config['min_weight'], self.optimization_config['max_weight']) for _ in range(n)]
            
            # Başlangıç noktası
            x0 = np.ones(n) / n
            
            # Optimizasyon
            if SCIPY_AVAILABLE:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            self.logger.error(f"Maksimum Sharpe optimizasyon hatası: {e}")
            return np.ones(len(expected_returns)) / len(expected_returns)
    
    def _minimum_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum Variance optimizasyonu"""
        try:
            n = cov_matrix.shape[0]
            
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Kısıtlamalar
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Sınırlar
            bounds = [(self.optimization_config['min_weight'], self.optimization_config['max_weight']) for _ in range(n)]
            
            # Başlangıç noktası
            x0 = np.ones(n) / n
            
            # Optimizasyon
            if SCIPY_AVAILABLE:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            self.logger.error(f"Minimum Variance optimizasyon hatası: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    def _equal_weight_optimization(self, n_assets: int) -> np.ndarray:
        """Equal Weight optimizasyonu"""
        try:
            return np.ones(n_assets) / n_assets
            
        except Exception as e:
            self.logger.error(f"Equal Weight optimizasyon hatası: {e}")
            return np.ones(n_assets) / n_assets
    
    def _simple_optimization(self, objective, bounds, constraints, x0):
        """Basit optimizasyon (SciPy yoksa)"""
        try:
            # Gradient descent benzeri basit optimizasyon
            x = x0.copy()
            learning_rate = 0.01
            max_iterations = 1000
            
            for _ in range(max_iterations):
                # Gradient hesapla (finite difference)
                grad = np.zeros_like(x)
                fx = objective(x)
                
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += 1e-6
                    grad[i] = (objective(x_plus) - fx) / 1e-6
                
                # Gradient descent step
                x_new = x - learning_rate * grad
                
                # Sınırları kontrol et
                for i in range(len(x_new)):
                    x_new[i] = max(bounds[i][0], min(bounds[i][1], x_new[i]))
                
                # Kısıtlamaları kontrol et
                if abs(np.sum(x_new) - 1.0) < 1e-6:
                    x = x_new
                else:
                    # Normalize et
                    x = x_new / np.sum(x_new)
                
                # Yakınsama kontrolü
                if np.linalg.norm(x_new - x) < 1e-6:
                    break
            
            return x
            
        except Exception as e:
            self.logger.error(f"Basit optimizasyon hatası: {e}")
            return x0
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Ağırlıkları normalize et ve kısıtlamaları uygula"""
        try:
            # Negatif ağırlıkları sıfırla (long-only)
            if self.optimization_config['constraints']['long_only']:
                weights = np.maximum(weights, 0)
            
            # Maksimum ağırlık kısıtlaması
            max_weight = self.optimization_config['max_weight']
            weights = np.minimum(weights, max_weight)
            
            # Minimum ağırlık kısıtlaması
            min_weight = self.optimization_config['min_weight']
            weights = np.maximum(weights, min_weight)
            
            # Normalize et
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Ağırlık normalize etme hatası: {e}")
            return weights
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, float]:
        """Portföy metriklerini hesapla"""
        try:
            # Beklenen getiri
            expected_return = np.dot(weights, expected_returns)
            
            # Volatilite
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            risk_free_rate = self.optimization_config['risk_free_rate']
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # VaR hesaplama (basit normal dağılım varsayımı)
            var_95 = expected_return - 1.645 * volatility
            var_99 = expected_return - 2.326 * volatility
            
            # Max drawdown tahmini
            max_drawdown = volatility * 2.5  # Basit tahmin
            
            return {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Portföy metrikleri hesaplama hatası: {e}")
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'var_95': 0,
                'var_99': 0,
                'max_drawdown': 0
            }
    
    def efficient_frontier(self, assets: List[Asset], n_points: int = 50) -> Dict[str, List[float]]:
        """Efficient Frontier hesapla"""
        try:
            # İstatistikleri hesapla
            expected_returns, cov_matrix = self._estimate_statistics(assets)
            
            if expected_returns is None or cov_matrix is None:
                return {}
            
            # Hedef getiriler
            min_return = np.min(expected_returns)
            max_return = np.max(expected_returns)
            target_returns = np.linspace(min_return, max_return, n_points)
            
            returns = []
            volatilities = []
            sharpe_ratios = []
            
            for target_return in target_returns:
                # Hedef getiri için minimum varyans portföyü bul
                weights = self._target_return_optimization(target_return, expected_returns, cov_matrix)
                
                if weights is not None:
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                    portfolio_volatility = np.sqrt(portfolio_variance)
                    
                    risk_free_rate = self.optimization_config['risk_free_rate']
                    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                    
                    returns.append(portfolio_return)
                    volatilities.append(portfolio_volatility)
                    sharpe_ratios.append(sharpe_ratio)
            
            return {
                'returns': returns,
                'volatilities': volatilities,
                'sharpe_ratios': sharpe_ratios
            }
            
        except Exception as e:
            self.logger.error(f"Efficient Frontier hesaplama hatası: {e}")
            return {}
    
    def _target_return_optimization(self, target_return: float, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Hedef getiri için minimum varyans optimizasyonu"""
        try:
            n = len(expected_returns)
            
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Kısıtlamalar
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            # Sınırlar
            bounds = [(self.optimization_config['min_weight'], self.optimization_config['max_weight']) for _ in range(n)]
            
            # Başlangıç noktası
            x0 = np.ones(n) / n
            
            # Optimizasyon
            if SCIPY_AVAILABLE:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success:
                    return result.x
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Hedef getiri optimizasyon hatası: {e}")
            return None
    
    def monte_carlo_simulation(self, assets: List[Asset], n_simulations: int = 10000) -> Dict[str, Any]:
        """Monte Carlo portföy simülasyonu"""
        try:
            expected_returns, cov_matrix = self._estimate_statistics(assets)
            
            if expected_returns is None or cov_matrix is None:
                return {}
            
            # Rastgele portföy ağırlıkları üret
            random_weights = np.random.dirichlet(np.ones(len(assets)), n_simulations)
            
            # Her portföy için metrikleri hesapla
            returns = []
            volatilities = []
            sharpe_ratios = []
            
            for weights in random_weights:
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                risk_free_rate = self.optimization_config['risk_free_rate']
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                returns.append(portfolio_return)
                volatilities.append(portfolio_volatility)
                sharpe_ratios.append(sharpe_ratio)
            
            return {
                'returns': returns,
                'volatilities': volatilities,
                'sharpe_ratios': sharpe_ratios,
                'n_simulations': n_simulations
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simülasyon hatası: {e}")
            return {}
    
    def backtest_portfolio(self, 
                          assets: List[Asset],
                          returns_data: pd.DataFrame,
                          rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
        """Portföy backtest"""
        try:
            if returns_data.empty:
                return {}
            
            # Optimizasyon yap
            result = self.optimize_portfolio(assets, OptimizationMethod.MAXIMUM_SHARPE, returns_data)
            
            if not result:
                return {}
            
            # Portföy getirilerini hesapla
            weights = np.array([result.optimal_weights[asset.symbol] for asset in assets])
            portfolio_returns = returns_data.dot(weights)
            
            # Performans metrikleri
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown hesaplama
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # VaR hesaplama
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'portfolio_returns': portfolio_returns,
                'optimal_weights': result.optimal_weights
            }
            
        except Exception as e:
            self.logger.error(f"Portföy backtest hatası: {e}")
            return {}
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Optimizasyon geçmişini al"""
        try:
            with self.lock:
                return list(self.optimization_history)
        except Exception as e:
            self.logger.error(f"Optimizasyon geçmişi alma hatası: {e}")
            return []
    
    def compare_optimization_methods(self, assets: List[Asset], returns_data: pd.DataFrame = None) -> Dict[str, OptimizationResult]:
        """Optimizasyon yöntemlerini karşılaştır"""
        try:
            methods = [
                OptimizationMethod.MAXIMUM_SHARPE,
                OptimizationMethod.MINIMUM_VARIANCE,
                OptimizationMethod.RISK_PARITY,
                OptimizationMethod.KELLY_CRITERION,
                OptimizationMethod.EQUAL_WEIGHT
            ]
            
            results = {}
            
            for method in methods:
                result = self.optimize_portfolio(assets, method, returns_data)
                if result:
                    results[method.value] = result
            
            # Sonuçları karşılaştır
            self._log_comparison_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimizasyon yöntemleri karşılaştırma hatası: {e}")
            return {}
    
    def _log_comparison_results(self, results: Dict[str, OptimizationResult]):
        """Karşılaştırma sonuçlarını logla"""
        try:
            self.logger.info("=== Portföy Optimizasyon Yöntemleri Karşılaştırması ===")
            
            for method, result in results.items():
                self.logger.info(f"{method}:")
                self.logger.info(f"  Beklenen Getiri: {result.expected_return:.4f}")
                self.logger.info(f"  Volatilite: {result.volatility:.4f}")
                self.logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
                self.logger.info(f"  VaR (95%): {result.var_95:.4f}")
                self.logger.info(f"  Max Drawdown: {result.max_drawdown:.4f}")
            
            # En iyi yöntemi bul
            best_method = max(results.items(), key=lambda x: x[1].sharpe_ratio)
            self.logger.info(f"En iyi yöntem: {best_method[0]} (Sharpe: {best_method[1].sharpe_ratio:.4f})")
            
        except Exception as e:
            self.logger.error(f"Karşılaştırma sonuçları loglama hatası: {e}")
    
    def add_optimization_callback(self, callback):
        """Optimizasyon callback'i ekle"""
        self.optimization_callbacks.append(callback)
    
    def add_rebalance_callback(self, callback):
        """Yeniden dengeleme callback'i ekle"""
        self.rebalance_callbacks.append(callback)
    
    def _notify_optimization_callbacks(self, result: OptimizationResult):
        """Optimizasyon callback'lerini çağır"""
        for callback in self.optimization_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Optimizasyon callback hatası: {e}")
    
    def _notify_rebalance_callbacks(self, old_weights: Dict[str, float], new_weights: Dict[str, float]):
        """Yeniden dengeleme callback'lerini çağır"""
        for callback in self.rebalance_callbacks:
            try:
                callback(old_weights, new_weights)
            except Exception as e:
                self.logger.error(f"Yeniden dengeleme callback hatası: {e}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            if 'optimization_config' in new_config:
                self.optimization_config.update(new_config['optimization_config'])
            
            self.logger.info("Portföy optimizasyon konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global portföy optimizasyon modülü
portfolio_optimizer = PortfolioOptimizer()