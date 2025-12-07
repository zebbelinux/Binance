"""
Monte Carlo Simülasyonu
Strateji performansının istatistiksel analizi
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class MonteCarloSimulator:
    """Monte Carlo simülatörü sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Simülasyon parametreleri
        self.num_simulations = 1000
        self.confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Risk metrikleri
        self.var_confidence = 0.05
        self.cvar_confidence = 0.05
        
        self.logger.info("Monte Carlo simülatörü başlatıldı")
    
    def run_monte_carlo(self, 
                       backtest_results: Dict[str, Any],
                       num_simulations: int = None) -> Dict[str, Any]:
        """Monte Carlo simülasyonu çalıştır"""
        try:
            if num_simulations:
                self.num_simulations = num_simulations
            
            # Backtest sonuçlarından veri çıkar
            trades = backtest_results.get('results', {}).get('trades', [])
            if not trades:
                return {'error': 'Backtest sonuçları bulunamadı'}
            
            # Trade returns hesapla
            trade_returns = [trade['net_pnl'] for trade in trades]
            if not trade_returns:
                return {'error': 'Trade returns hesaplanamadı'}
            
            # Simülasyonları çalıştır
            simulation_results = self._run_simulations(trade_returns, backtest_results)
            
            # İstatistiksel analiz
            statistical_analysis = self._analyze_simulations(simulation_results)
            
            # Risk analizi
            risk_analysis = self._analyze_risk(simulation_results)
            
            # Senaryo analizi
            scenario_analysis = self._analyze_scenarios(simulation_results)
            
            return {
                'simulation_results': simulation_results,
                'statistical_analysis': statistical_analysis,
                'risk_analysis': risk_analysis,
                'scenario_analysis': scenario_analysis,
                'num_simulations': self.num_simulations,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simülasyon hatası: {e}")
            return {'error': str(e)}
    
    def _run_simulations(self, trade_returns: List[float], backtest_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simülasyonları çalıştır"""
        try:
            simulation_results = []
            
            # Trade returns istatistikleri
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            
            # Autocorrelation analizi
            if len(trade_returns) > 1:
                autocorr = np.corrcoef(trade_returns[:-1], trade_returns[1:])[0, 1]
            else:
                autocorr = 0
            
            # Her simülasyon için
            for i in range(self.num_simulations):
                # Random walk simülasyonu
                simulated_returns = self._generate_random_walk(
                    trade_returns, mean_return, std_return, autocorr
                )
                
                # Simülasyon sonuçları
                simulation_result = self._calculate_simulation_metrics(
                    simulated_returns, backtest_results
                )
                
                simulation_result['simulation_id'] = i
                simulation_results.append(simulation_result)
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Simülasyon çalıştırma hatası: {e}")
            return []
    
    def _generate_random_walk(self, 
                             original_returns: List[float],
                             mean_return: float,
                             std_return: float,
                             autocorr: float) -> List[float]:
        """Random walk oluştur"""
        try:
            n_trades = len(original_returns)
            
            if autocorr == 0 or abs(autocorr) < 0.1:
                # Bağımsız simülasyon
                simulated_returns = np.random.normal(mean_return, std_return, n_trades)
            else:
                # Autocorrelation ile simülasyon
                simulated_returns = np.zeros(n_trades)
                simulated_returns[0] = np.random.normal(mean_return, std_return)
                
                for i in range(1, n_trades):
                    # AR(1) model: r_t = mean + autocorr * r_{t-1} + error
                    error = np.random.normal(0, std_return * np.sqrt(1 - autocorr**2))
                    simulated_returns[i] = mean_return + autocorr * (simulated_returns[i-1] - mean_return) + error
            
            return simulated_returns.tolist()
            
        except Exception as e:
            self.logger.error(f"Random walk oluşturma hatası: {e}")
            return original_returns
    
    def _calculate_simulation_metrics(self, 
                                    simulated_returns: List[float],
                                    backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simülasyon metriklerini hesapla"""
        try:
            # Temel metrikler
            total_return = sum(simulated_returns)
            mean_return = np.mean(simulated_returns)
            std_return = np.std(simulated_returns)
            
            # Win rate
            winning_trades = len([r for r in simulated_returns if r > 0])
            win_rate = winning_trades / len(simulated_returns) * 100 if simulated_returns else 0
            
            # Profit factor
            gross_profit = sum([r for r in simulated_returns if r > 0])
            gross_loss = abs(sum([r for r in simulated_returns if r < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Drawdown analizi
            cumulative_returns = np.cumsum(simulated_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / (peak + 1e-10)
            max_drawdown = np.min(drawdown)
            
            # Sharpe ratio (basitleştirilmiş)
            sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            # VaR ve CVaR
            var_95 = np.percentile(simulated_returns, 5)
            cvar_95 = np.mean([r for r in simulated_returns if r <= var_95])
            
            return {
                'total_return': total_return,
                'mean_return': mean_return,
                'std_return': std_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'cumulative_returns': cumulative_returns.tolist(),
                'drawdown': drawdown.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Simülasyon metrikleri hesaplama hatası: {e}")
            return {}
    
    def _analyze_simulations(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simülasyonları istatistiksel olarak analiz et"""
        try:
            if not simulation_results:
                return {'error': 'Simülasyon sonuçları yok'}
            
            # Metrikleri topla
            metrics = ['total_return', 'mean_return', 'std_return', 'win_rate', 
                      'profit_factor', 'max_drawdown', 'sharpe_ratio', 'var_95', 'cvar_95']
            
            analysis = {}
            
            for metric in metrics:
                values = [sim[metric] for sim in simulation_results if metric in sim]
                
                if values:
                    analysis[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'percentiles': {
                            f'p{int(p*100)}': np.percentile(values, p*100) 
                            for p in self.confidence_levels
                        }
                    }
            
            # Korelasyon analizi
            correlation_matrix = self._calculate_correlation_matrix(simulation_results, metrics)
            analysis['correlations'] = correlation_matrix
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"İstatistiksel analiz hatası: {e}")
            return {'error': str(e)}
    
    def _analyze_risk(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Risk analizi yap"""
        try:
            if not simulation_results:
                return {'error': 'Simülasyon sonuçları yok'}
            
            # VaR analizi
            total_returns = [sim['total_return'] for sim in simulation_results]
            var_analysis = {
                'var_95': np.percentile(total_returns, 5),
                'var_99': np.percentile(total_returns, 1),
                'expected_shortfall_95': np.mean([r for r in total_returns if r <= np.percentile(total_returns, 5)]),
                'expected_shortfall_99': np.mean([r for r in total_returns if r <= np.percentile(total_returns, 1)])
            }
            
            # Drawdown analizi
            max_drawdowns = [sim['max_drawdown'] for sim in simulation_results]
            drawdown_analysis = {
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_drawdown': np.min(max_drawdowns),
                'drawdown_95_percentile': np.percentile(max_drawdowns, 95),
                'drawdown_probability': len([dd for dd in max_drawdowns if dd < -0.1]) / len(max_drawdowns)  # %10'dan fazla düşüş olasılığı
            }
            
            # Tail risk analizi
            tail_analysis = self._analyze_tail_risk(simulation_results)
            
            return {
                'var_analysis': var_analysis,
                'drawdown_analysis': drawdown_analysis,
                'tail_analysis': tail_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Risk analizi hatası: {e}")
            return {'error': str(e)}
    
    def _analyze_tail_risk(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tail risk analizi"""
        try:
            # Extreme returns analizi
            all_returns = []
            for sim in simulation_results:
                all_returns.extend(sim.get('cumulative_returns', []))
            
            if not all_returns:
                return {'error': 'Cumulative returns yok'}
            
            # Extreme value theory
            returns_array = np.array(all_returns)
            
            # Tail index (Hill estimator)
            sorted_returns = np.sort(returns_array)
            n = len(sorted_returns)
            k = max(1, int(n * 0.05))  # En kötü %5
            
            if k > 1:
                tail_index = 1 / np.mean(np.log(sorted_returns[-k:] / sorted_returns[-k-1]))
            else:
                tail_index = 1
            
            # Extreme quantiles
            extreme_quantiles = {
                'p99.9': np.percentile(returns_array, 0.1),
                'p99': np.percentile(returns_array, 1),
                'p95': np.percentile(returns_array, 5),
                'p5': np.percentile(returns_array, 95),
                'p1': np.percentile(returns_array, 99),
                'p0.1': np.percentile(returns_array, 99.9)
            }
            
            return {
                'tail_index': tail_index,
                'extreme_quantiles': extreme_quantiles,
                'skewness': stats.skew(returns_array),
                'kurtosis': stats.kurtosis(returns_array)
            }
            
        except Exception as e:
            self.logger.error(f"Tail risk analizi hatası: {e}")
            return {'error': str(e)}
    
    def _analyze_scenarios(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Senaryo analizi"""
        try:
            if not simulation_results:
                return {'error': 'Simülasyon sonuçları yok'}
            
            # Senaryo tanımları
            scenarios = {
                'best_case': {
                    'description': 'En iyi %5 senaryo',
                    'threshold': 95,
                    'metric': 'total_return'
                },
                'worst_case': {
                    'description': 'En kötü %5 senaryo',
                    'threshold': 5,
                    'metric': 'total_return'
                },
                'high_volatility': {
                    'description': 'Yüksek volatilite senaryosu',
                    'threshold': 95,
                    'metric': 'std_return'
                },
                'low_volatility': {
                    'description': 'Düşük volatilite senaryosu',
                    'threshold': 5,
                    'metric': 'std_return'
                },
                'high_drawdown': {
                    'description': 'Yüksek drawdown senaryosu',
                    'threshold': 95,
                    'metric': 'max_drawdown'
                }
            }
            
            scenario_results = {}
            
            for scenario_name, scenario_config in scenarios.items():
                metric = scenario_config['metric']
                threshold = scenario_config['threshold']
                
                values = [sim[metric] for sim in simulation_results if metric in sim]
                
                if values:
                    threshold_value = np.percentile(values, threshold)
                    
                    # Bu threshold'u karşılayan simülasyonları bul
                    matching_sims = [sim for sim in simulation_results 
                                   if sim.get(metric, 0) >= threshold_value]
                    
                    if matching_sims:
                        # Bu simülasyonların ortalama performansı
                        avg_metrics = {}
                        for metric_name in ['total_return', 'win_rate', 'profit_factor', 'max_drawdown', 'sharpe_ratio']:
                            metric_values = [sim[metric_name] for sim in matching_sims if metric_name in sim]
                            if metric_values:
                                avg_metrics[metric_name] = np.mean(metric_values)
                        
                        scenario_results[scenario_name] = {
                            'description': scenario_config['description'],
                            'threshold_value': threshold_value,
                            'num_simulations': len(matching_sims),
                            'avg_metrics': avg_metrics
                        }
            
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Senaryo analizi hatası: {e}")
            return {'error': str(e)}
    
    def _calculate_correlation_matrix(self, simulation_results: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
        """Korelasyon matrisi hesapla"""
        try:
            # Metrik değerlerini topla
            data = {}
            for metric in metrics:
                values = [sim[metric] for sim in simulation_results if metric in sim]
                if values:
                    data[metric] = values
            
            if not data:
                return {}
            
            # DataFrame oluştur
            df = pd.DataFrame(data)
            
            # Korelasyon matrisi
            correlation_matrix = df.corr()
            
            return correlation_matrix.to_dict()
            
        except Exception as e:
            self.logger.error(f"Korelasyon matrisi hesaplama hatası: {e}")
            return {}
    
    def generate_plots(self, simulation_results: List[Dict[str, Any]], 
                      output_dir: str = "plots") -> Dict[str, str]:
        """Grafikler oluştur"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            plot_files = {}
            
            # 1. Total Return Dağılımı
            total_returns = [sim['total_return'] for sim in simulation_results]
            plt.figure(figsize=(10, 6))
            plt.hist(total_returns, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(total_returns), color='red', linestyle='--', label=f'Mean: {np.mean(total_returns):.2f}')
            plt.axvline(np.percentile(total_returns, 5), color='orange', linestyle='--', label=f'5th percentile: {np.percentile(total_returns, 5):.2f}')
            plt.xlabel('Total Return')
            plt.ylabel('Frequency')
            plt.title('Monte Carlo Simulation - Total Return Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_file = os.path.join(output_dir, 'total_return_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['total_return_distribution'] = plot_file
            
            # 2. Drawdown Dağılımı
            max_drawdowns = [sim['max_drawdown'] for sim in simulation_results]
            plt.figure(figsize=(10, 6))
            plt.hist(max_drawdowns, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(max_drawdowns), color='red', linestyle='--', label=f'Mean: {np.mean(max_drawdowns):.2f}')
            plt.axvline(np.percentile(max_drawdowns, 95), color='orange', linestyle='--', label=f'95th percentile: {np.percentile(max_drawdowns, 95):.2f}')
            plt.xlabel('Maximum Drawdown')
            plt.ylabel('Frequency')
            plt.title('Monte Carlo Simulation - Maximum Drawdown Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_file = os.path.join(output_dir, 'drawdown_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['drawdown_distribution'] = plot_file
            
            # 3. Sharpe Ratio vs Total Return
            sharpe_ratios = [sim['sharpe_ratio'] for sim in simulation_results]
            plt.figure(figsize=(10, 6))
            plt.scatter(sharpe_ratios, total_returns, alpha=0.6)
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Total Return')
            plt.title('Monte Carlo Simulation - Sharpe Ratio vs Total Return')
            plt.grid(True, alpha=0.3)
            plot_file = os.path.join(output_dir, 'sharpe_vs_return.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['sharpe_vs_return'] = plot_file
            
            # 4. Equity Curve Örnekleri
            plt.figure(figsize=(12, 8))
            for i in range(min(20, len(simulation_results))):  # İlk 20 simülasyon
                sim = simulation_results[i]
                if 'cumulative_returns' in sim:
                    plt.plot(sim['cumulative_returns'], alpha=0.3, linewidth=0.5)
            
            # Ortalama equity curve
            if simulation_results:
                avg_cumulative = np.mean([sim.get('cumulative_returns', []) for sim in simulation_results], axis=0)
                if len(avg_cumulative) > 0:
                    plt.plot(avg_cumulative, color='red', linewidth=2, label='Average')
            
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative Return')
            plt.title('Monte Carlo Simulation - Equity Curve Examples')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_file = os.path.join(output_dir, 'equity_curves.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['equity_curves'] = plot_file
            
            return plot_files
            
        except Exception as e:
            self.logger.error(f"Grafik oluşturma hatası: {e}")
            return {'error': str(e)}

# Global Monte Carlo simülatörü
monte_carlo_simulator = MonteCarloSimulator()

