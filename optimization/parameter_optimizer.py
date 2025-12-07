"""
Parametre Optimizasyonu Modülü
Genetik algoritma ve Grid search ile strateji parametrelerini optimize etme
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
import random
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os

# Optimizasyon kütüphaneleri
try:
    from scipy.optimize import differential_evolution, minimize
    from sklearn.model_selection import ParameterGrid
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class ParameterRange:
    """Parametre aralığı"""
    name: str
    min_value: float
    max_value: float
    step: float = None
    param_type: str = 'float'  # 'float', 'int', 'bool', 'choice'
    choices: List[Any] = None

@dataclass
class OptimizationResult:
    """Optimizasyon sonucu"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_method: str
    iterations: int
    execution_time: float
    convergence_history: List[float]
    timestamp: datetime
    strategy_name: str

@dataclass
class Individual:
    """Genetik algoritma bireyi"""
    parameters: Dict[str, Any]
    fitness: float = -float('inf')
    generation: int = 0

class OptimizationMethod(Enum):
    """Optimizasyon yöntemleri"""
    GRID_SEARCH = "grid_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class ParameterOptimizer:
    """Parametre optimizasyon sınıfı"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Optimizasyon parametreleri
        self.optimization_config = {
            'max_iterations': 1000,
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 5,
            'convergence_threshold': 1e-6,
            'patience': 50,
            'n_jobs': -1,  # Tüm CPU çekirdekleri
            'timeout': 3600  # 1 saat timeout
        }
        
        # Optimizasyon geçmişi
        self.optimization_history = deque(maxlen=100)
        self.best_results = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.progress_callbacks = []
        self.completion_callbacks = []
        
        self.logger.info("Parametre optimizasyon modülü başlatıldı")
    
    def optimize_strategy(self, 
                         strategy_name: str,
                         parameter_ranges: List[ParameterRange],
                         objective_function: Callable,
                         method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM,
                         max_iterations: int = None) -> OptimizationResult:
        """Strateji parametrelerini optimize et"""
        try:
            start_time = time.time()
            
            if max_iterations:
                self.optimization_config['max_iterations'] = max_iterations
            
            self.logger.info(f"{strategy_name} stratejisi için {method.value} optimizasyonu başlatılıyor")
            
            if method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(
                    strategy_name, parameter_ranges, objective_function
                )
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(
                    strategy_name, parameter_ranges, objective_function
                )
            elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
                result = self._differential_evolution_optimization(
                    strategy_name, parameter_ranges, objective_function
                )
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(
                    strategy_name, parameter_ranges, objective_function
                )
            elif method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(
                    strategy_name, parameter_ranges, objective_function
                )
            else:
                raise ValueError(f"Desteklenmeyen optimizasyon yöntemi: {method}")
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Sonucu kaydet
            with self.lock:
                self.optimization_history.append(result)
                self.best_results[strategy_name] = result
            
            # Callback'leri çağır
            self._notify_completion_callbacks(result)
            
            self.logger.info(f"{strategy_name} optimizasyonu tamamlandı - En iyi skor: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimizasyon hatası ({strategy_name}): {e}")
            return None
    
    def _grid_search_optimization(self, 
                                 strategy_name: str,
                                 parameter_ranges: List[ParameterRange],
                                 objective_function: Callable) -> OptimizationResult:
        """Grid search optimizasyonu"""
        try:
            # Parametre grid'i oluştur
            param_grid = {}
            for param_range in parameter_ranges:
                if param_range.param_type == 'float':
                    values = np.arange(
                        param_range.min_value, 
                        param_range.max_value + param_range.step, 
                        param_range.step
                    )
                elif param_range.param_type == 'int':
                    values = list(range(
                        int(param_range.min_value), 
                        int(param_range.max_value) + 1, 
                        int(param_range.step) if param_range.step else 1
                    ))
                elif param_range.param_type == 'bool':
                    values = [True, False]
                elif param_range.param_type == 'choice':
                    values = param_range.choices
                else:
                    continue
                
                param_grid[param_range.name] = values
            
            # Grid search
            if SCIPY_AVAILABLE:
                grid = ParameterGrid(param_grid)
                total_combinations = len(list(grid))
            else:
                # Manuel grid oluştur
                keys = list(param_grid.keys())
                values = list(param_grid.values())
                combinations = list(itertools.product(*values))
                total_combinations = len(combinations)
            
            self.logger.info(f"Grid search: {total_combinations} kombinasyon test edilecek")
            
            best_score = -float('inf')
            best_parameters = {}
            convergence_history = []
            
            # Paralel hesaplama
            if self.optimization_config['n_jobs'] != 1:
                with ThreadPoolExecutor(max_workers=self.optimization_config['n_jobs']) as executor:
                    if SCIPY_AVAILABLE:
                        futures = []
                        for params in grid:
                            future = executor.submit(self._evaluate_parameters, params, objective_function)
                            futures.append((future, params))
                        
                        for i, (future, params) in enumerate(futures):
                            try:
                                score = future.result(timeout=self.optimization_config['timeout'])
                                convergence_history.append(score)
                                
                                if score > best_score:
                                    best_score = score
                                    best_parameters = params.copy()
                                
                                # İlerleme callback'i
                                progress = (i + 1) / total_combinations
                                self._notify_progress_callbacks(progress, best_score)
                                
                            except Exception as e:
                                self.logger.error(f"Grid search değerlendirme hatası: {e}")
                    else:
                        # Manuel grid için
                        futures = []
                        for params_dict in combinations:
                            params = dict(zip(keys, params_dict))
                            future = executor.submit(self._evaluate_parameters, params, objective_function)
                            futures.append((future, params))
                        
                        for i, (future, params) in enumerate(futures):
                            try:
                                score = future.result(timeout=self.optimization_config['timeout'])
                                convergence_history.append(score)
                                
                                if score > best_score:
                                    best_score = score
                                    best_parameters = params.copy()
                                
                                progress = (i + 1) / total_combinations
                                self._notify_progress_callbacks(progress, best_score)
                                
                            except Exception as e:
                                self.logger.error(f"Grid search değerlendirme hatası: {e}")
            else:
                # Sıralı hesaplama
                if SCIPY_AVAILABLE:
                    for i, params in enumerate(grid):
                        try:
                            score = self._evaluate_parameters(params, objective_function)
                            convergence_history.append(score)
                            
                            if score > best_score:
                                best_score = score
                                best_parameters = params.copy()
                            
                            progress = (i + 1) / total_combinations
                            self._notify_progress_callbacks(progress, best_score)
                            
                        except Exception as e:
                            self.logger.error(f"Grid search değerlendirme hatası: {e}")
                else:
                    for i, params_dict in enumerate(combinations):
                        try:
                            params = dict(zip(keys, params_dict))
                            score = self._evaluate_parameters(params, objective_function)
                            convergence_history.append(score)
                            
                            if score > best_score:
                                best_score = score
                                best_parameters = params.copy()
                            
                            progress = (i + 1) / total_combinations
                            self._notify_progress_callbacks(progress, best_score)
                            
                        except Exception as e:
                            self.logger.error(f"Grid search değerlendirme hatası: {e}")
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method='grid_search',
                iterations=total_combinations,
                execution_time=0,  # Sonradan set edilecek
                convergence_history=convergence_history,
                timestamp=datetime.now(),
                strategy_name=strategy_name
            )
            
        except Exception as e:
            self.logger.error(f"Grid search optimizasyon hatası: {e}")
            return None
    
    def _genetic_algorithm_optimization(self, 
                                      strategy_name: str,
                                      parameter_ranges: List[ParameterRange],
                                      objective_function: Callable) -> OptimizationResult:
        """Genetik algoritma optimizasyonu"""
        try:
            population_size = self.optimization_config['population_size']
            max_iterations = self.optimization_config['max_iterations']
            mutation_rate = self.optimization_config['mutation_rate']
            crossover_rate = self.optimization_config['crossover_rate']
            elite_size = self.optimization_config['elite_size']
            
            # İlk popülasyonu oluştur
            population = self._create_initial_population(parameter_ranges, population_size)
            
            best_score = -float('inf')
            best_parameters = {}
            convergence_history = []
            no_improvement_count = 0
            
            for generation in range(max_iterations):
                # Fitness değerlendirmesi
                for individual in population:
                    if individual.fitness == -float('inf'):
                        individual.fitness = self._evaluate_parameters(individual.parameters, objective_function)
                
                # En iyi bireyi bul
                generation_best = max(population, key=lambda x: x.fitness)
                
                if generation_best.fitness > best_score:
                    best_score = generation_best.fitness
                    best_parameters = generation_best.parameters.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                convergence_history.append(best_score)
                
                # İlerleme callback'i
                progress = generation / max_iterations
                self._notify_progress_callbacks(progress, best_score)
                
                # Yakınsama kontrolü
                if no_improvement_count >= self.optimization_config['patience']:
                    self.logger.info(f"Yakınsama sağlandı - Nesil: {generation}")
                    break
                
                # Yeni nesil oluştur
                new_population = []
                
                # Elite bireyleri koru
                elite = sorted(population, key=lambda x: x.fitness, reverse=True)[:elite_size]
                for individual in elite:
                    new_population.append(Individual(
                        parameters=individual.parameters.copy(),
                        fitness=individual.fitness,
                        generation=generation + 1
                    ))
                
                # Kalan bireyleri üret
                while len(new_population) < population_size:
                    # Seçim
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Çaprazlama
                    if random.random() < crossover_rate:
                        child_params = self._crossover(parent1.parameters, parent2.parameters, parameter_ranges)
                    else:
                        child_params = parent1.parameters.copy()
                    
                    # Mutasyon
                    if random.random() < mutation_rate:
                        child_params = self._mutate(child_params, parameter_ranges)
                    
                    new_population.append(Individual(
                        parameters=child_params,
                        generation=generation + 1
                    ))
                
                population = new_population
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method='genetic_algorithm',
                iterations=generation + 1,
                execution_time=0,  # Sonradan set edilecek
                convergence_history=convergence_history,
                timestamp=datetime.now(),
                strategy_name=strategy_name
            )
            
        except Exception as e:
            self.logger.error(f"Genetik algoritma optimizasyon hatası: {e}")
            return None
    
    def _differential_evolution_optimization(self, 
                                           strategy_name: str,
                                           parameter_ranges: List[ParameterRange],
                                           objective_function: Callable) -> OptimizationResult:
        """Differential evolution optimizasyonu"""
        try:
            if not SCIPY_AVAILABLE:
                self.logger.warning("SciPy bulunamadı, differential evolution kullanılamıyor")
                return None
            
            # Parametre sınırları
            bounds = []
            param_names = []
            
            for param_range in parameter_ranges:
                if param_range.param_type in ['float', 'int']:
                    bounds.append((param_range.min_value, param_range.max_value))
                    param_names.append(param_range.name)
            
            def objective_wrapper(params):
                param_dict = dict(zip(param_names, params))
                return -self._evaluate_parameters(param_dict, objective_function)  # Minimize için negatif
            
            # Differential evolution
            result = differential_evolution(
                objective_wrapper,
                bounds,
                maxiter=self.optimization_config['max_iterations'],
                popsize=self.optimization_config['population_size'],
                seed=42
            )
            
            # Sonuçları hazırla
            best_parameters = dict(zip(param_names, result.x))
            best_score = -result.fun
            convergence_history = []
            
            # Yakınsama geçmişi (simüle edilmiş)
            for i in range(result.nit):
                convergence_history.append(best_score * (1 - i / result.nit))
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method='differential_evolution',
                iterations=result.nit,
                execution_time=0,  # Sonradan set edilecek
                convergence_history=convergence_history,
                timestamp=datetime.now(),
                strategy_name=strategy_name
            )
            
        except Exception as e:
            self.logger.error(f"Differential evolution optimizasyon hatası: {e}")
            return None
    
    def _random_search_optimization(self, 
                                  strategy_name: str,
                                  parameter_ranges: List[ParameterRange],
                                  objective_function: Callable) -> OptimizationResult:
        """Random search optimizasyonu"""
        try:
            max_iterations = self.optimization_config['max_iterations']
            
            best_score = -float('inf')
            best_parameters = {}
            convergence_history = []
            
            for iteration in range(max_iterations):
                # Rastgele parametreler oluştur
                params = self._generate_random_parameters(parameter_ranges)
                
                # Değerlendir
                score = self._evaluate_parameters(params, objective_function)
                convergence_history.append(score)
                
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                # İlerleme callback'i
                progress = iteration / max_iterations
                self._notify_progress_callbacks(progress, best_score)
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method='random_search',
                iterations=max_iterations,
                execution_time=0,  # Sonradan set edilecek
                convergence_history=convergence_history,
                timestamp=datetime.now(),
                strategy_name=strategy_name
            )
            
        except Exception as e:
            self.logger.error(f"Random search optimizasyon hatası: {e}")
            return None
    
    def _bayesian_optimization(self, 
                              strategy_name: str,
                              parameter_ranges: List[ParameterRange],
                              objective_function: Callable) -> OptimizationResult:
        """Bayesian optimizasyonu (Optuna ile)"""
        try:
            if not OPTUNA_AVAILABLE:
                self.logger.warning("Optuna bulunamadı, Bayesian optimizasyon kullanılamıyor")
                return None
            
            def objective_wrapper(trial):
                params = {}
                for param_range in parameter_ranges:
                    if param_range.param_type == 'float':
                        params[param_range.name] = trial.suggest_float(
                            param_range.name, param_range.min_value, param_range.max_value
                        )
                    elif param_range.param_type == 'int':
                        params[param_range.name] = trial.suggest_int(
                            param_range.name, int(param_range.min_value), int(param_range.max_value)
                        )
                    elif param_range.param_type == 'bool':
                        params[param_range.name] = trial.suggest_categorical(
                            param_range.name, [True, False]
                        )
                    elif param_range.param_type == 'choice':
                        params[param_range.name] = trial.suggest_categorical(
                            param_range.name, param_range.choices
                        )
                
                return self._evaluate_parameters(params, objective_function)
            
            # Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective_wrapper, 
                n_trials=self.optimization_config['max_iterations']
            )
            
            # Sonuçları hazırla
            best_parameters = study.best_params
            best_score = study.best_value
            convergence_history = [trial.value for trial in study.trials]
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method='bayesian_optimization',
                iterations=len(study.trials),
                execution_time=0,  # Sonradan set edilecek
                convergence_history=convergence_history,
                timestamp=datetime.now(),
                strategy_name=strategy_name
            )
            
        except Exception as e:
            self.logger.error(f"Bayesian optimizasyon hatası: {e}")
            return None
    
    def _create_initial_population(self, parameter_ranges: List[ParameterRange], population_size: int) -> List[Individual]:
        """İlk popülasyonu oluştur"""
        try:
            population = []
            
            for _ in range(population_size):
                params = self._generate_random_parameters(parameter_ranges)
                individual = Individual(parameters=params)
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.error(f"İlk popülasyon oluşturma hatası: {e}")
            return []
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Rastgele parametreler oluştur"""
        try:
            params = {}
            
            for param_range in parameter_ranges:
                if param_range.param_type == 'float':
                    params[param_range.name] = random.uniform(
                        param_range.min_value, param_range.max_value
                    )
                elif param_range.param_type == 'int':
                    params[param_range.name] = random.randint(
                        int(param_range.min_value), int(param_range.max_value)
                    )
                elif param_range.param_type == 'bool':
                    params[param_range.name] = random.choice([True, False])
                elif param_range.param_type == 'choice':
                    params[param_range.name] = random.choice(param_range.choices)
            
            return params
            
        except Exception as e:
            self.logger.error(f"Rastgele parametre oluşturma hatası: {e}")
            return {}
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Turnuva seçimi"""
        try:
            tournament = random.sample(population, min(tournament_size, len(population)))
            return max(tournament, key=lambda x: x.fitness)
            
        except Exception as e:
            self.logger.error(f"Turnuva seçimi hatası: {e}")
            return population[0]
    
    def _crossover(self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any], 
                  parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Çaprazlama"""
        try:
            child_params = {}
            
            for param_range in parameter_ranges:
                name = param_range.name
                
                if param_range.param_type in ['float', 'int']:
                    # Aritmetik çaprazlama
                    alpha = random.random()
                    val1 = parent1_params[name]
                    val2 = parent2_params[name]
                    
                    if param_range.param_type == 'float':
                        child_val = alpha * val1 + (1 - alpha) * val2
                    else:
                        child_val = int(alpha * val1 + (1 - alpha) * val2)
                    
                    # Sınırları kontrol et
                    child_val = max(param_range.min_value, min(param_range.max_value, child_val))
                    child_params[name] = child_val
                    
                elif param_range.param_type == 'bool':
                    # Rastgele seçim
                    child_params[name] = random.choice([parent1_params[name], parent2_params[name]])
                    
                elif param_range.param_type == 'choice':
                    # Rastgele seçim
                    child_params[name] = random.choice([parent1_params[name], parent2_params[name]])
            
            return child_params
            
        except Exception as e:
            self.logger.error(f"Çaprazlama hatası: {e}")
            return parent1_params
    
    def _mutate(self, params: Dict[str, Any], parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Mutasyon"""
        try:
            mutated_params = params.copy()
            
            for param_range in parameter_ranges:
                name = param_range.name
                
                if random.random() < 0.1:  # %10 mutasyon olasılığı
                    if param_range.param_type == 'float':
                        # Gaussian mutasyon
                        noise = random.gauss(0, (param_range.max_value - param_range.min_value) * 0.1)
                        mutated_val = mutated_params[name] + noise
                        mutated_val = max(param_range.min_value, min(param_range.max_value, mutated_val))
                        mutated_params[name] = mutated_val
                        
                    elif param_range.param_type == 'int':
                        # Uniform mutasyon
                        mutated_val = random.randint(int(param_range.min_value), int(param_range.max_value))
                        mutated_params[name] = mutated_val
                        
                    elif param_range.param_type == 'bool':
                        # Toggle mutasyon
                        mutated_params[name] = not mutated_params[name]
                        
                    elif param_range.param_type == 'choice':
                        # Rastgele seçim
                        mutated_params[name] = random.choice(param_range.choices)
            
            return mutated_params
            
        except Exception as e:
            self.logger.error(f"Mutasyon hatası: {e}")
            return params
    
    def _evaluate_parameters(self, parameters: Dict[str, Any], objective_function: Callable) -> float:
        """Parametreleri değerlendir"""
        try:
            return objective_function(parameters)
            
        except Exception as e:
            self.logger.error(f"Parametre değerlendirme hatası: {e}")
            return -float('inf')
    
    def optimize_multiple_strategies(self, 
                                   strategies_config: Dict[str, Dict[str, Any]],
                                   objective_function: Callable,
                                   method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM) -> Dict[str, OptimizationResult]:
        """Birden fazla stratejiyi optimize et"""
        try:
            results = {}
            
            for strategy_name, config in strategies_config.items():
                parameter_ranges = config.get('parameter_ranges', [])
                max_iterations = config.get('max_iterations', None)
                
                result = self.optimize_strategy(
                    strategy_name, parameter_ranges, objective_function, method, max_iterations
                )
                
                if result:
                    results[strategy_name] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Çoklu strateji optimizasyon hatası: {e}")
            return {}
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Optimizasyon geçmişini al"""
        try:
            with self.lock:
                return list(self.optimization_history)
        except Exception as e:
            self.logger.error(f"Optimizasyon geçmişi alma hatası: {e}")
            return []
    
    def get_best_result(self, strategy_name: str) -> Optional[OptimizationResult]:
        """En iyi sonucu al"""
        try:
            with self.lock:
                return self.best_results.get(strategy_name)
        except Exception as e:
            self.logger.error(f"En iyi sonuç alma hatası: {e}")
            return None
    
    def compare_optimization_methods(self, 
                                   strategy_name: str,
                                   parameter_ranges: List[ParameterRange],
                                   objective_function: Callable,
                                   methods: List[OptimizationMethod] = None) -> Dict[str, OptimizationResult]:
        """Optimizasyon yöntemlerini karşılaştır"""
        try:
            if methods is None:
                methods = [
                    OptimizationMethod.GRID_SEARCH,
                    OptimizationMethod.GENETIC_ALGORITHM,
                    OptimizationMethod.RANDOM_SEARCH
                ]
            
            results = {}
            
            for method in methods:
                self.logger.info(f"{strategy_name} için {method.value} yöntemi test ediliyor")
                
                result = self.optimize_strategy(
                    f"{strategy_name}_{method.value}",
                    parameter_ranges,
                    objective_function,
                    method
                )
                
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
            self.logger.info("=== Optimizasyon Yöntemleri Karşılaştırması ===")
            
            for method, result in results.items():
                self.logger.info(f"{method}:")
                self.logger.info(f"  En iyi skor: {result.best_score:.4f}")
                self.logger.info(f"  İterasyon sayısı: {result.iterations}")
                self.logger.info(f"  Çalışma süresi: {result.execution_time:.2f} saniye")
                self.logger.info(f"  En iyi parametreler: {result.best_parameters}")
            
            # En iyi yöntemi bul
            best_method = max(results.items(), key=lambda x: x[1].best_score)
            self.logger.info(f"En iyi yöntem: {best_method[0]} (Skor: {best_method[1].best_score:.4f})")
            
        except Exception as e:
            self.logger.error(f"Karşılaştırma sonuçları loglama hatası: {e}")
    
    def save_optimization_results(self, filepath: str) -> bool:
        """Optimizasyon sonuçlarını kaydet"""
        try:
            with self.lock:
                data = {
                    'optimization_history': [
                        {
                            'best_parameters': result.best_parameters,
                            'best_score': result.best_score,
                            'optimization_method': result.optimization_method,
                            'iterations': result.iterations,
                            'execution_time': result.execution_time,
                            'convergence_history': result.convergence_history,
                            'timestamp': result.timestamp.isoformat(),
                            'strategy_name': result.strategy_name
                        }
                        for result in self.optimization_history
                    ],
                    'best_results': {
                        strategy_name: {
                            'best_parameters': result.best_parameters,
                            'best_score': result.best_score,
                            'optimization_method': result.optimization_method,
                            'iterations': result.iterations,
                            'execution_time': result.execution_time,
                            'convergence_history': result.convergence_history,
                            'timestamp': result.timestamp.isoformat(),
                            'strategy_name': result.strategy_name
                        }
                        for strategy_name, result in self.best_results.items()
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.logger.info(f"Optimizasyon sonuçları kaydedildi: {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Optimizasyon sonuçları kaydetme hatası: {e}")
            return False
    
    def load_optimization_results(self, filepath: str) -> bool:
        """Optimizasyon sonuçlarını yükle"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self.lock:
                # Geçmişi yükle
                self.optimization_history.clear()
                for result_data in data.get('optimization_history', []):
                    result = OptimizationResult(
                        best_parameters=result_data['best_parameters'],
                        best_score=result_data['best_score'],
                        optimization_method=result_data['optimization_method'],
                        iterations=result_data['iterations'],
                        execution_time=result_data['execution_time'],
                        convergence_history=result_data['convergence_history'],
                        timestamp=datetime.fromisoformat(result_data['timestamp']),
                        strategy_name=result_data['strategy_name']
                    )
                    self.optimization_history.append(result)
                
                # En iyi sonuçları yükle
                self.best_results.clear()
                for strategy_name, result_data in data.get('best_results', {}).items():
                    result = OptimizationResult(
                        best_parameters=result_data['best_parameters'],
                        best_score=result_data['best_score'],
                        optimization_method=result_data['optimization_method'],
                        iterations=result_data['iterations'],
                        execution_time=result_data['execution_time'],
                        convergence_history=result_data['convergence_history'],
                        timestamp=datetime.fromisoformat(result_data['timestamp']),
                        strategy_name=result_data['strategy_name']
                    )
                    self.best_results[strategy_name] = result
            
            self.logger.info(f"Optimizasyon sonuçları yüklendi: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Optimizasyon sonuçları yükleme hatası: {e}")
            return False
    
    def add_progress_callback(self, callback):
        """İlerleme callback'i ekle"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback):
        """Tamamlanma callback'i ekle"""
        self.completion_callbacks.append(callback)
    
    def _notify_progress_callbacks(self, progress: float, best_score: float):
        """İlerleme callback'lerini çağır"""
        for callback in self.progress_callbacks:
            try:
                callback(progress, best_score)
            except Exception as e:
                self.logger.error(f"İlerleme callback hatası: {e}")
    
    def _notify_completion_callbacks(self, result: OptimizationResult):
        """Tamamlanma callback'lerini çağır"""
        for callback in self.completion_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Tamamlanma callback hatası: {e}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            if 'optimization_config' in new_config:
                self.optimization_config.update(new_config['optimization_config'])
            
            self.logger.info("Parametre optimizasyon konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global parametre optimizasyon modülü
parameter_optimizer = ParameterOptimizer()
