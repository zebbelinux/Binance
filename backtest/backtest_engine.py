"""
Backtest Motoru
Geçmiş verilerle strateji testi ve performans analizi
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

@dataclass
class BacktestConfig:
    """Backtest konfigürasyonu"""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 100000.0
    commission_rate: float = 0.001  # %0.1
    slippage_rate: float = 0.0005  # %0.05
    data_frequency: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    warmup_periods: int = 200  # Isınma periyodu
    max_positions: int = 5  # Maksimum pozisyon sayısı

class BacktestEngine:
    """Backtest motoru sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Backtest durumu
        self.is_running = False
        self.current_step = 0
        self.total_steps = 0
        
        # Performans metrikleri
        self.performance_metrics = {}
        
        # Callback'ler
        self.progress_callbacks = []
        self.completion_callbacks = []
        
        self.logger.info("Backtest motoru başlatıldı")
    
    def run_backtest(self, 
                    strategy,
                    market_data: pd.DataFrame,
                    config: BacktestConfig) -> Dict[str, Any]:
        """Backtest çalıştır"""
        try:
            self.logger.info(f"Backtest başlatılıyor: {config.start_date} - {config.end_date}")
            
            # Veri hazırlığı
            prepared_data = self._prepare_data(market_data, config)
            if prepared_data.empty:
                return {'error': 'Veri hazırlama hatası'}
            
            # Backtest durumunu başlat
            self.is_running = True
            self.total_steps = len(prepared_data)
            self.current_step = 0
            
            # Backtest simülasyonu
            results = self._run_simulation(strategy, prepared_data, config)
            
            # Performans analizi
            performance = self._calculate_performance_metrics(results, config)
            
            # Sonuçları birleştir
            backtest_results = {
                'config': config.__dict__,
                'results': results,
                'performance': performance,
                'success': True,
                'completed_at': datetime.now()
            }
            
            self.logger.info("Backtest tamamlandı")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Backtest hatası: {e}")
            return {'error': str(e), 'success': False}
        finally:
            self.is_running = False
    
    def _prepare_data(self, market_data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        """Veri hazırlığı"""
        try:
            # Tarih aralığını filtrele
            mask = (market_data.index >= config.start_date) & (market_data.index <= config.end_date)
            filtered_data = market_data[mask].copy()
            
            if filtered_data.empty:
                self.logger.warning("Filtrelenmiş veri boş")
                return pd.DataFrame()
            
            # Gerekli sütunları kontrol et
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in filtered_data.columns]
            
            if missing_columns:
                self.logger.error(f"Eksik sütunlar: {missing_columns}")
                return pd.DataFrame()
            
            # Veri kalitesini kontrol et
            filtered_data = filtered_data.dropna()
            
            # Isınma periyodu ekle
            if len(filtered_data) < config.warmup_periods:
                self.logger.warning(f"Veri yetersiz: {len(filtered_data)} < {config.warmup_periods}")
                return pd.DataFrame()
            
            self.logger.info(f"Veri hazırlandı: {len(filtered_data)} bar")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Veri hazırlama hatası: {e}")
            return pd.DataFrame()
    
    def _run_simulation(self, strategy, data: pd.DataFrame, config: BacktestConfig) -> Dict[str, Any]:
        """Simülasyon çalıştır"""
        try:
            # Simülasyon durumu
            balance = config.initial_balance
            available_balance = balance
            used_balance = 0.0
            
            positions = {}
            trades = []
            equity_curve = []
            
            # Teknik analiz verilerini hazırla
            technical_data = self._calculate_technical_indicators(data)
            
            # Simülasyon döngüsü
            for i in range(config.warmup_periods, len(data)):
                current_time = data.index[i]
                current_data = data.iloc[i]
                
                # Mevcut pozisyonları güncelle
                self._update_positions(positions, current_data, config)
                
                # Kapatılan pozisyonları işle
                closed_positions = self._process_closed_positions(positions, current_data, config)
                trades.extend(closed_positions)
                
                # Bakiyeyi güncelle
                balance, available_balance, used_balance = self._update_balance(
                    balance, available_balance, used_balance, closed_positions
                )
                
                # Strateji sinyalleri üret
                market_data_dict = {
                    'symbol': 'BTCUSDT',
                    'price': current_data['close'],
                    'high': current_data['high'],
                    'low': current_data['low'],
                    'volume': current_data['volume'],
                    'timestamp': current_time,
                    'technical_analysis': technical_data.iloc[i].to_dict() if i < len(technical_data) else {}
                }
                
                signals = strategy.generate_signals(market_data_dict)
                
                # Sinyalleri işle
                for signal in signals:
                    if self._validate_signal(signal, available_balance, config):
                        position = self._create_position(signal, current_data, config)
                        if position:
                            positions[position['id']] = position
                            used_balance += position['size'] * position['entry_price']
                            available_balance -= position['size'] * position['entry_price']
                
                # Equity curve güncelle
                current_equity = balance + sum(pos['unrealized_pnl'] for pos in positions.values())
                equity_curve.append({
                    'timestamp': current_time,
                    'equity': current_equity,
                    'balance': balance,
                    'unrealized_pnl': sum(pos['unrealized_pnl'] for pos in positions.values()),
                    'open_positions': len(positions)
                })
                
                # İlerleme güncelle
                self.current_step = i - config.warmup_periods + 1
                self._notify_progress_callbacks(self.current_step, self.total_steps)
            
            # Kalan pozisyonları kapat
            final_trades = self._close_all_positions(positions, data.iloc[-1], config)
            trades.extend(final_trades)
            
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'final_balance': balance,
                'final_equity': balance + sum(pos['unrealized_pnl'] for pos in positions.values()),
                'total_trades': len(trades),
                'open_positions': len(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Simülasyon hatası: {e}")
            return {'error': str(e)}
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Teknik göstergeleri hesapla"""
        try:
            from indicators.technical_indicators import technical_indicators
            
            indicators_data = []
            
            for i in range(len(data)):
                if i < 200:  # Yeterli veri yok
                    indicators_data.append({})
                    continue
                
                # Son 200 bar'ı al
                window_data = data.iloc[i-199:i+1]
                
                # Teknik göstergeleri hesapla
                indicators = technical_indicators.calculate_all_indicators(window_data)
                indicators_data.append(indicators)
            
            return pd.DataFrame(indicators_data, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Teknik göstergeler hesaplama hatası: {e}")
            return pd.DataFrame()
    
    def _update_positions(self, positions: Dict[str, Any], current_data: pd.Series, config: BacktestConfig):
        """Pozisyonları güncelle"""
        try:
            current_price = current_data['close']
            
            for position_id, position in positions.items():
                if position['status'] != 'open':
                    continue
                
                # Unrealized P&L hesapla
                if position['side'] == 'buy':
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
                else:
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
                
                position['current_price'] = current_price
                
                # Stop loss kontrolü
                if position['side'] == 'buy' and current_price <= position['stop_loss']:
                    position['status'] = 'closed'
                    position['exit_price'] = current_price
                    position['close_reason'] = 'stop_loss'
                elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                    position['status'] = 'closed'
                    position['exit_price'] = current_price
                    position['close_reason'] = 'stop_loss'
                
                # Take profit kontrolü
                elif position['side'] == 'buy' and current_price >= position['take_profit']:
                    position['status'] = 'closed'
                    position['exit_price'] = current_price
                    position['close_reason'] = 'take_profit'
                elif position['side'] == 'sell' and current_price <= position['take_profit']:
                    position['status'] = 'closed'
                    position['exit_price'] = current_price
                    position['close_reason'] = 'take_profit'
                
        except Exception as e:
            self.logger.error(f"Pozisyon güncelleme hatası: {e}")
    
    def _process_closed_positions(self, positions: Dict[str, Any], current_data: pd.Series, config: BacktestConfig) -> List[Dict[str, Any]]:
        """Kapatılan pozisyonları işle"""
        closed_trades = []
        
        try:
            for position_id, position in list(positions.items()):
                if position['status'] == 'closed':
                    # Komisyon hesapla
                    trade_value = position['size'] * position['exit_price']
                    commission = trade_value * config.commission_rate
                    
                    # Slippage hesapla
                    slippage = trade_value * config.slippage_rate
                    
                    # Realized P&L hesapla
                    if position['side'] == 'buy':
                        realized_pnl = (position['exit_price'] - position['entry_price']) * position['size']
                    else:
                        realized_pnl = (position['entry_price'] - position['exit_price']) * position['size']
                    
                    # Net P&L
                    net_pnl = realized_pnl - commission - slippage
                    
                    # Trade kaydı oluştur
                    trade = {
                        'id': position_id,
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': position['size'],
                        'entry_price': position['entry_price'],
                        'exit_price': position['exit_price'],
                        'entry_time': position['created_at'],
                        'exit_time': current_data.name,
                        'realized_pnl': realized_pnl,
                        'commission': commission,
                        'slippage': slippage,
                        'net_pnl': net_pnl,
                        'close_reason': position['close_reason']
                    }
                    
                    closed_trades.append(trade)
                    
                    # Pozisyonu kaldır
                    del positions[position_id]
            
            return closed_trades
            
        except Exception as e:
            self.logger.error(f"Kapatılan pozisyon işleme hatası: {e}")
            return closed_trades
    
    def _update_balance(self, balance: float, available_balance: float, used_balance: float, 
                       closed_trades: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """Bakiyeyi güncelle"""
        try:
            for trade in closed_trades:
                # Kullanılan bakiyeyi geri ver
                trade_value = trade['size'] * trade['entry_price']
                used_balance -= trade_value
                available_balance += trade_value
                
                # Net P&L'yi ekle
                balance += trade['net_pnl']
                available_balance += trade['net_pnl']
            
            return balance, available_balance, used_balance
            
        except Exception as e:
            self.logger.error(f"Bakiye güncelleme hatası: {e}")
            return balance, available_balance, used_balance
    
    def _validate_signal(self, signal: Dict[str, Any], available_balance: float, config: BacktestConfig) -> bool:
        """Sinyal geçerliliğini kontrol et"""
        try:
            # Temel kontroller
            if not signal.get('symbol') or not signal.get('side') or not signal.get('strength'):
                return False
            
            # Pozisyon büyüklüğü kontrolü
            position_size = signal.get('size', 0)
            if position_size <= 0:
                return False
            
            # Bakiye kontrolü
            if position_size > available_balance:
                return False
            
            # Maksimum pozisyon kontrolü
            # Bu basitleştirilmiş bir kontrol, gerçek uygulamada daha detaylı olmalı
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sinyal doğrulama hatası: {e}")
            return False
    
    def _create_position(self, signal: Dict[str, Any], current_data: pd.Series, config: BacktestConfig) -> Optional[Dict[str, Any]]:
        """Pozisyon oluştur"""
        try:
            position_id = f"{signal['symbol']}_{signal['side']}_{current_data.name.strftime('%Y%m%d_%H%M%S')}"
            
            # Slippage uygula
            entry_price = current_data['close']
            if signal['side'] == 'buy':
                entry_price *= (1 + config.slippage_rate)
            else:
                entry_price *= (1 - config.slippage_rate)
            
            # Stop loss ve take profit hesapla
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if stop_loss == 0:
                if signal['side'] == 'buy':
                    stop_loss = entry_price * 0.98  # %2 stop loss
                else:
                    stop_loss = entry_price * 1.02  # %2 stop loss
            
            if take_profit == 0:
                if signal['side'] == 'buy':
                    take_profit = entry_price * 1.04  # %4 take profit
                else:
                    take_profit = entry_price * 0.96  # %4 take profit
            
            position = {
                'id': position_id,
                'symbol': signal['symbol'],
                'side': signal['side'],
                'size': signal['size'],
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'unrealized_pnl': 0.0,
                'status': 'open',
                'created_at': current_data.name
            }
            
            return position
            
        except Exception as e:
            self.logger.error(f"Pozisyon oluşturma hatası: {e}")
            return None
    
    def _close_all_positions(self, positions: Dict[str, Any], final_data: pd.Series, config: BacktestConfig) -> List[Dict[str, Any]]:
        """Tüm pozisyonları kapat"""
        try:
            # Tüm açık pozisyonları kapat
            for position in positions.values():
                if position['status'] == 'open':
                    position['status'] = 'closed'
                    position['exit_price'] = final_data['close']
                    position['close_reason'] = 'end_of_backtest'
            
            # Kapatılan pozisyonları işle
            return self._process_closed_positions(positions, final_data, config)
            
        except Exception as e:
            self.logger.error(f"Pozisyon kapatma hatası: {e}")
            return []
    
    def _calculate_performance_metrics(self, results: Dict[str, Any], config: BacktestConfig) -> Dict[str, Any]:
        """Performans metriklerini hesapla"""
        try:
            trades = results.get('trades', [])
            equity_curve = results.get('equity_curve', [])
            
            if not trades or not equity_curve:
                return {'error': 'Yeterli veri yok'}
            
            # Temel metrikler
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['net_pnl'] > 0])
            losing_trades = len([t for t in trades if t['net_pnl'] < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrikleri
            total_pnl = sum(t['net_pnl'] for t in trades)
            gross_profit = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
            gross_loss = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Ortalama kazanç/kayıp
            avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
            
            # Equity curve analizi
            equity_values = [e['equity'] for e in equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            # Getiri metrikleri
            total_return = (equity_values[-1] - config.initial_balance) / config.initial_balance
            annualized_return = (1 + total_return) ** (365 / len(equity_curve)) - 1
            
            # Volatilite
            volatility = np.std(returns) * np.sqrt(252)  # Yıllık volatilite
            
            # Sharpe ratio
            risk_free_rate = 0.05  # %5
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Drawdown analizi
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR ve CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'final_equity': equity_values[-1],
                'initial_balance': config.initial_balance
            }
            
        except Exception as e:
            self.logger.error(f"Performans metrikleri hesaplama hatası: {e}")
            return {'error': str(e)}
    
    def add_progress_callback(self, callback):
        """İlerleme callback'i ekle"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback):
        """Tamamlanma callback'i ekle"""
        self.completion_callbacks.append(callback)
    
    def _notify_progress_callbacks(self, current_step: int, total_steps: int):
        """İlerleme callback'lerini çağır"""
        progress = {
            'current_step': current_step,
            'total_steps': total_steps,
            'percentage': (current_step / total_steps * 100) if total_steps > 0 else 0
        }
        
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                self.logger.error(f"İlerleme callback hatası: {e}")

# Global backtest motoru
backtest_engine = BacktestEngine()

