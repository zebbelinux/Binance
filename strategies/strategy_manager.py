"""
Strateji Yöneticisi
Tüm trading stratejilerini yönetir ve koordine eder
"""

from typing import Dict, List, Any, Optional
import logging
import threading
import time
from datetime import datetime
from enum import Enum

from strategies.base_strategy import BaseStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.grid_strategy import GridStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.dca_strategy import DCAStrategy
from strategies.hedge_strategy import HedgeStrategy
from strategies.hedge.pairs import PairsTradingStrategy
from strategies.hedge.cointegration import CointegrationStrategy
from strategies.hedge.lead_lag import LeadLagStrategy
from strategies.dynamic_strategy_selector import DynamicStrategySelector
from strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_ignition_strategy import MomentumIgnitionStrategy
from strategies.reversal_confirmation_strategy import ReversalConfirmationStrategy
from strategies.adaptive_volatility_strategy import AdaptiveVolatilityStrategy
from strategies.correlation_rotation_strategy import CorrelationRotationStrategy
from strategies.ai_sentiment_fusion_strategy import AISentimentFusionStrategy
from strategies.hybrid_bear_hunter_strategy import HybridBearHunterStrategy
from ai.market_analyzer import market_analyzer
from ai.signal_classifier import SignalClassifier
from utils.trade_journal import journal
try:
    from data.external_data_manager import external_data_manager as external_dm
except Exception:
    external_dm = None

class StrategyStatus(Enum):
    """Strateji durumu"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"

class StrategyManager:
    """Strateji yöneticisi sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Stratejiler
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_status: Dict[str, StrategyStatus] = {}
        
        # Strateji performansı
        self.strategy_performance = {}
        
        # Thread yönetimi
        self.is_running = False
        self.strategy_thread = None
        self.lock = threading.RLock()
        
        # Smart Entry & Position Sizing konfigürasyonu
        self.smart_entry_cfg = {
            'enabled': True,
            'confirmation_candles': 1
        }
        # Bekleyen sinyaller (smart confirmation için kuyruk)
        # Her öğe: { 'signal': {...}, 'enqueued_at': datetime, 'symbol': str, 'base_close_len': int }
        self.pending_signals: List[Dict[str, Any]] = []
        # Adaptif pozisyon boyutu konfigürasyonu
        self.position_sizing_cfg = {
            'base_fraction': 0.02,   # bakiyenin %2'si temel
            'vol_window': 20         # volatilite penceresi
        }
        # ML doğrulama ve rejim farkındalığı
        self.ml_validation_cfg = {
            'enabled': True,
            'accept_threshold': 0.55,
            'regime_weighting': True,
            'purged_cv': True,
            'embargo_bars': 1
        }
        try:
            self.signal_classifier = SignalClassifier()
        except Exception:
            self.signal_classifier = None
        # Harici veri filtreleri (Glassnode/CoinGecko)
        self.external_filter_cfg = {
            'enabled': True,
            'exchange_inflow_threshold': 1.5,   # daha gevşek eşik
            'sentiment_min_threshold': 0.25     # daha gevşek eşik
        }
        self.external_data = external_dm
        # Saat×Rejim whitelist (opsiyonel)
        self.hour_regime_whitelist = self._load_hour_regime_whitelist()
        
        # Callback'ler
        self.signal_callbacks = []
        self.performance_callbacks = []
        
        # Strateji limitleri ve rejim yönetimi
        self.max_active_strategies = 3  # Sideways ve Volatile için 3 strateji
        self._regime_name = 'unknown'  # Mevcut piyasa rejimi
        self._regime_apply_cooldown = 30.0  # 30 saniye cooldown
        self._last_regime_change = 0.0  # Son rejim değişim zamanı
        
        # Varsayılan stratejileri yükle
        self._load_default_strategies()

        # AI tabanlı piyasa rejimi için MarketAnalyzer analiz callback'ine abone ol
        try:
            if hasattr(market_analyzer, 'add_analysis_callback'):
                market_analyzer.add_analysis_callback(self._on_market_analysis)
                self.logger.info("MarketAnalyzer analysis callback kaydedildi (BTCUSDT rejimi için)")
        except Exception as e:
            self.logger.warning(f"MarketAnalyzer analysis callback kaydedilemedi: {e}")
    
    def _load_default_strategies(self):
        """Varsayılan stratejileri yükle"""
        try:
            # Trend Following stratejisi
            trend_config = {
                'fast_ma': 12,
                'slow_ma': 36,
                'adx_threshold': 20.0,
                'atr_stop_multiplier': 2.5,
                'atr_tp_multiplier': 2.5,
                'min_signal_strength': 0.55,
                'max_position_size': 0.10,
                'min_volume': 500000
            }
            self.add_strategy('trend_following', TrendFollowingStrategy(trend_config))
            # Volatility Breakout
            vbo_cfg = {
                'bb_period': 20,
                'bb_std': 2.0,
                'min_strength': 0.45,
                'symbols': ['BTCUSDT','ETHUSDT']
            }
            self.add_strategy('volatility_breakout', VolatilityBreakoutStrategy(vbo_cfg))
            # Mean Reversion
            mr_cfg = {
                'bb_period': 20,
                'bb_std': 1.8,
                'rsi_period': 10,
                'rsi_overbought': 75,
                'rsi_oversold': 25,
                'min_strength': 0.65,
                'symbols': ['BTCTRY','ETHTRY']
            }
            self.add_strategy('mean_reversion', MeanReversionStrategy(mr_cfg))

            self.logger.info("Tüm stratejiler yüklendi")
            
        except Exception as e:
            self.logger.error(f"Varsayılan strateji yükleme hatası: {e}")
    
    def add_strategy(self, name: str, strategy: BaseStrategy) -> bool:
        """Strateji ekle"""
        try:
            with self.lock:
                if name in self.strategies:
                    self.logger.warning(f"Strateji '{name}' zaten mevcut")
                    return False
                
                self.strategies[name] = strategy
                self.strategy_status[name] = StrategyStatus.INACTIVE
                self.strategy_performance[name] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'failed_signals': 0,
                    'total_pnl': 0.0,
                    'last_update': datetime.now()
                }
                
                self.logger.info(f"Strateji '{name}' eklendi")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji ekleme hatası: {e}")
            return False
    
    def remove_strategy(self, name: str) -> bool:
        """Strateji kaldır"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.warning(f"Strateji '{name}' bulunamadı")
                    return False
                
                # Stratejiyi durdur
                if self.strategy_status[name] == StrategyStatus.ACTIVE:
                    self.strategies[name].stop()
                
                del self.strategies[name]
                del self.strategy_status[name]
                del self.strategy_performance[name]
                
                self.logger.info(f"Strateji '{name}' kaldırıldı")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji kaldırma hatası: {e}")
            return False
    
    def start_strategy(self, name: str) -> bool:
        """Stratejiyi başlat"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                if self.strategy_status[name] == StrategyStatus.ACTIVE:
                    self.logger.warning(f"Strateji '{name}' zaten aktif")
                    return True
                
                strategy = self.strategies[name]
                # Bazı stratejiler (dynamic_selector) start yerine start_analysis kullanır
                if hasattr(strategy, 'start') and callable(getattr(strategy, 'start')):
                    strategy.start()
                elif hasattr(strategy, 'start_analysis') and callable(getattr(strategy, 'start_analysis')):
                    strategy.start_analysis()
                else:
                    # Start metodu yoksa yine de ACTIVE olarak işaretleyip devam edelim
                    self.logger.warning(f"Strateji '{name}' start() metodunu desteklemiyor; ACTIVE işaretlendi")
                self.strategy_status[name] = StrategyStatus.ACTIVE
                
                self.logger.info(f"Strateji '{name}' başlatıldı")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji başlatma hatası: {e}")
            self.strategy_status[name] = StrategyStatus.ERROR
            return False
    
    def stop_strategy(self, name: str) -> bool:
        """Stratejiyi durdur"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                if self.strategy_status[name] != StrategyStatus.ACTIVE:
                    self.logger.warning(f"Strateji '{name}' zaten durmuş")
                    return True
                
                strategy = self.strategies[name]
                if hasattr(strategy, 'stop') and callable(getattr(strategy, 'stop')):
                    strategy.stop()
                elif hasattr(strategy, 'stop_analysis') and callable(getattr(strategy, 'stop_analysis')):
                    strategy.stop_analysis()
                else:
                    self.logger.warning(f"Strateji '{name}' stop() metodunu desteklemiyor")
                    self.strategy_status[name] = StrategyStatus.INACTIVE
                    return True
                self.strategy_status[name] = StrategyStatus.INACTIVE
                
                self.logger.info(f"Strateji '{name}' durduruldu")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji durdurma hatası: {e}")
            return False
    
    def pause_strategy(self, name: str) -> bool:
        """Stratejiyi duraklat"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                if self.strategy_status[name] != StrategyStatus.ACTIVE:
                    self.logger.warning(f"Strateji '{name}' zaten duraklatılmış")
                    return True
                
                self.strategy_status[name] = StrategyStatus.PAUSED
                
                self.logger.info(f"Strateji '{name}' duraklatıldı")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji duraklatma hatası: {e}")
            return False
    
    def resume_strategy(self, name: str) -> bool:
        """Stratejiyi devam ettir"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                if self.strategy_status[name] != StrategyStatus.PAUSED:
                    self.logger.warning(f"Strateji '{name}' duraklatılmamış")
                    return True
                
                self.strategy_status[name] = StrategyStatus.ACTIVE
                
                self.logger.info(f"Strateji '{name}' devam ettirildi")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji devam ettirme hatası: {e}")
            return False
    
    def start_all_strategies(self):
        """Tüm stratejileri başlat"""
        try:
            with self.lock:
                for name in self.strategies:
                    if self.strategy_status[name] == StrategyStatus.INACTIVE:
                        self.start_strategy(name)
                
                self.logger.info("Tüm stratejiler başlatıldı")
                
        except Exception as e:
            self.logger.error(f"Tüm stratejileri başlatma hatası: {e}")
    
    def stop_all_strategies(self):
        """Tüm stratejileri durdur"""
        try:
            with self.lock:
                for name in self.strategies:
                    if self.strategy_status[name] == StrategyStatus.ACTIVE:
                        self.stop_strategy(name)
                
                self.logger.info("Tüm stratejiler durduruldu")
                
        except Exception as e:
            self.logger.error(f"Tüm stratejileri durdurma hatası: {e}")
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Piyasa verilerini işle ve sinyaller üret"""
        all_signals = []
        
        try:
            with self.lock:
                for name, strategy in self.strategies.items():
                    if self.strategy_status[name] != StrategyStatus.ACTIVE:
                        continue
                    
                    try:
                        # Strateji sinyalleri üret (destekliyorsa)
                        if hasattr(strategy, 'generate_signals') and callable(getattr(strategy, 'generate_signals')):
                            signals = strategy.generate_signals(market_data)
                        else:
                            signals = []
                        
                        # Sinyalleri işle
                        for signal in signals:
                            signal['strategy_name'] = name
                            signal['timestamp'] = datetime.now()
                            try:
                                sym_loc = signal.get('symbol')
                                latest_loc = market_analyzer.get_latest_analysis(sym_loc)
                                self._micro_enrich(signal, latest_loc)
                                # Whitelist kontrolü (hour×regime)
                                try:
                                    if not self._is_whitelist_allowed(latest_loc):
                                        self._update_strategy_performance(name, signal, False)
                                        try:
                                            journal.log_signal_skipped(
                                                strategy_name=name,
                                                symbol=signal.get('symbol'),
                                                side=signal.get('side'),
                                                reason_code="hour_regime_whitelist",
                                                details={
                                                    'regime': (latest_loc or {}).get('ai_analysis', {}).get('regime') if isinstance(latest_loc, dict) else None
                                                },
                                            )
                                        except Exception:
                                            pass
                                        continue
                                except Exception:
                                    pass
                                is_chop = self._is_chop_regime(latest_loc)
                                if is_chop and 'trend_following' in name:
                                    self._update_strategy_performance(name, signal, False)
                                    try:
                                        journal.log_signal_skipped(
                                            strategy_name=name,
                                            symbol=signal.get('symbol'),
                                            side=signal.get('side'),
                                            reason_code="choppy_regime_trend_block",
                                            details={},
                                        )
                                    except Exception:
                                        pass
                                    continue
                                if is_chop and name == 'scalping':
                                    tp = float(signal.get('take_profit', 0.0) or 0.0)
                                    ep = float(signal.get('entry_price', 0.0) or 0.0)
                                    if tp > 0 and ep > 0:
                                        tp_pct = (tp - ep) / ep
                                    else:
                                        tp_pct = float(strategy.config.get('profit_target', 0.0) or 0.0)
                                    spread = float(signal.get('spread_pct', 0.0) or 0.0)
                                    fee = float(signal.get('fee_rate', 0.0010) or 0.0010)
                                    req = 3.0 * (spread + fee)
                                    if not (tp_pct >= req):
                                        self._update_strategy_performance(name, signal, False)
                                        try:
                                            journal.log_signal_skipped(
                                                strategy_name=name,
                                                symbol=signal.get('symbol'),
                                                side=signal.get('side'),
                                                reason_code="scalping_edge_not_enough",
                                                details={'tp_pct': tp_pct, 'required': req},
                                            )
                                        except Exception:
                                            pass
                                        continue
                                # Hedge/Pairs maliyet kapısı
                                if name in ('hedge', 'pairs_trading'):
                                    tp = float(signal.get('take_profit', 0.0) or 0.0)
                                    ep = float(signal.get('entry_price', 0.0) or 0.0)
                                    if tp > 0 and ep > 0:
                                        tp_pct = abs(tp - ep) / ep
                                    else:
                                        tp_pct = 0.0
                                    fee = float(signal.get('fee_rate', 0.0010) or 0.0010)
                                    slip = float(signal.get('slippage_est_pct', 0.0006) or 0.0006)
                                    if not (tp_pct >= 2.0 * (fee + slip)):
                                        self._update_strategy_performance(name, signal, False)
                                        try:
                                            journal.log_signal_skipped(
                                                strategy_name=name,
                                                symbol=signal.get('symbol'),
                                                side=signal.get('side'),
                                                reason_code="hedge_cost_edge_fail",
                                                details={'tp_pct': tp_pct, 'fee': fee, 'slip': slip},
                                            )
                                        except Exception:
                                            pass
                                        continue
                                # Grid spacing'i dinamik ayarla (ATR%)
                                if name == 'grid':
                                    try:
                                        tech = (latest_loc or {}).get('technical_analysis', {}) if isinstance(latest_loc, dict) else {}
                                        price = float((market_data or {}).get('price', 0.0) or 0.0)
                                        atr = float(tech.get('atr', 0.0) or 0.0)
                                        spread = float(signal.get('spread_pct', 0.0) or 0.0)
                                        if atr > 0 and price > 0 and hasattr(strategy, 'config'):
                                            atr_pct = atr / price
                                            base = 0.35
                                            if spread > 0.001:
                                                base = 0.40
                                            elif spread < 0.0004:
                                                base = 0.30
                                            spacing = atr_pct * base
                                            spacing = max(0.0025, min(0.0080, spacing))
                                            strategy.config['grid_spacing'] = spacing
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            
                            # Sinyal geçerliliğini kontrol et (destekliyorsa)
                            is_valid = True
                            if hasattr(strategy, 'validate_signal') and callable(getattr(strategy, 'validate_signal')):
                                is_valid = strategy.validate_signal(signal)
                            if not is_valid:
                                self._update_strategy_performance(name, signal, False)
                                try:
                                    journal.log_signal_skipped(
                                        strategy_name=name,
                                        symbol=signal.get('symbol'),
                                        side=signal.get('side'),
                                        reason_code="strategy_validate_signal",
                                        details={},
                                    )
                                except Exception:
                                    pass
                                continue

                            # Adaptif pozisyon boyutu ipucu ekle
                            try:
                                sym = signal.get('symbol')
                                closes = self._extract_closes(market_data, sym)
                                self._enrich_position_size_hint(signal, closes)
                            except Exception:
                                pass

                            # ML doğrulama ve rejim ağırlığı uygula (geçerli değilse filtrele)
                            try:
                                if self.ml_validation_cfg.get('enabled', True):
                                    ok = self._apply_ml_and_regime(signal, market_data)
                                    if not ok:
                                        self._update_strategy_performance(name, signal, False)
                                        try:
                                            journal.log_signal_skipped(
                                                strategy_name=name,
                                                symbol=signal.get('symbol'),
                                                side=signal.get('side'),
                                                reason_code="ml_regime_filter",
                                                details={'ml_success_prob': signal.get('ml_success_prob')},
                                            )
                                        except Exception:
                                            pass
                                        continue
                            except Exception:
                                pass

                            # Harici veri filtreleri (Glassnode/CoinGecko)
                            try:
                                if self.external_filter_cfg.get('enabled', True):
                                    ok_ext = self._apply_external_filters(signal, market_data)
                                    if not ok_ext:
                                        self._update_strategy_performance(name, signal, False)
                                        try:
                                            journal.log_signal_skipped(
                                                strategy_name=name,
                                                symbol=signal.get('symbol'),
                                                side=signal.get('side'),
                                                reason_code="external_filters",
                                                details={},
                                            )
                                        except Exception:
                                            pass
                                        continue
                                # Golf filtresi (mikroyapı) - GEÇİCİ OLARAK KAPALI
                                # try:
                                #     ms = (latest_loc or {}).get('microstructure', {}) if isinstance(latest_loc, dict) else {}
                                #     spread = float(signal.get('spread_pct', ms.get('spread_pct', 0.0)) or 0.0)
                                #     ob_imb = float(ms.get('orderbook_imbalance_5lvl', 0.0) or 0.0)
                                #     vol_z = float(ms.get('volume_zscore_30s', 0.0) or 0.0)
                                #     tur = float(ms.get('ticks_up_ratio_30s', 0.0) or 0.0)
                                #     if spread > 0.0012:
                                #         self._update_strategy_performance(name, signal, False)
                                #         continue
                                #     if ob_imb < 0.52:
                                #         self._update_strategy_performance(name, signal, False)
                                #         continue
                                #     if not (vol_z >= 1.2 or tur >= 0.58):
                                #         self._update_strategy_performance(name, signal, False)
                                #         continue
                                # except Exception:
                                #     pass
                                pass  # GEÇİCİ: Golf filtresi kapalı
                                # Adaptif maliyet eşiği (AL kapısı) - GEÇİCİ OLARAK KAPALI
                                # try:
                                #     side = (signal.get('side') or '').lower()
                                #     if side == 'buy':
                                #         fee = float(signal.get('fee_rate', 0.0010) or 0.0010)
                                #         ms = (latest_loc or {}).get('microstructure', {}) if isinstance(latest_loc, dict) else {}
                                #         spread = float(signal.get('spread_pct', ms.get('spread_pct', 0.0)) or 0.0)
                                #         slip = float(ms.get('slippage_est_pct', 0.0006) or 0.0006)
                                #         edge_cost = 2.0 * fee + spread + slip
                                #         p_ref = float(ms.get('p_ref_1h_90p', 0.0) or 0.0)
                                #         buy_thresh = max(3.0 * edge_cost, p_ref)
                                #         p_fast = float(ms.get('pct_change_10_30s', 0.0) or 0.0)
                                #         if p_fast != 0.0 and p_fast < buy_thresh:
                                #             self._update_strategy_performance(name, signal, False)
                                #             continue
                                #         signal['edge_cost'] = edge_cost
                                # except Exception:
                                #     pass
                                pass  # GEÇİCİ: AL kapısı kapalı
                            except Exception:
                                pass

                            # Smart Entry: GEÇİCİ OLARAK KAPALI - Sinyaller hemen geçiyor
                            # if self.smart_entry_cfg.get('enabled', True):
                            #     try:
                            #         sym = signal.get('symbol')
                            #         closes = self._extract_closes(market_data, sym)
                            #         base_len = len(closes) if isinstance(closes, list) else 0
                            #     except Exception:
                            #         base_len = 0
                            #     self.pending_signals.append({
                            #         'signal': signal,
                            #         'enqueued_at': datetime.now(),
                            #         'symbol': signal.get('symbol'),
                            #         'base_close_len': base_len
                            #     })
                            # else:
                            all_signals.append(signal)
                            try:
                                journal.log_signal_decision(
                                    strategy_name=name,
                                    symbol=signal.get('symbol'),
                                    side=signal.get('side'),
                                    accepted=True,
                                    reason_code="accepted",
                                    details={'strength': signal.get('strength'), 'ml_success_prob': signal.get('ml_success_prob')},
                                )
                            except Exception:
                                pass
                            # Performansı güncelle
                            self._update_strategy_performance(name, signal, True)
                        
                        # Callback'leri çağır
                        self._notify_signal_callbacks(signals)
                        
                    except Exception as e:
                        self.logger.error(f"Strateji '{name}' işleme hatası: {e}")
                        self.strategy_status[name] = StrategyStatus.ERROR
                        self._update_strategy_performance(name, None, False)
                
                # Smart Entry kuyruğunu kontrol et ve teyit olanları geçir
                try:
                    if self.smart_entry_cfg.get('enabled', True) and self.pending_signals:
                        needed = int(self.smart_entry_cfg.get('confirmation_candles', 2))
                        confirmed = []
                        remaining = []
                        for item in self.pending_signals:
                            sig = item.get('signal', {})
                            sym = item.get('symbol')
                            base_len = int(item.get('base_close_len', 0))
                            closes = self._extract_closes(market_data, sym)
                            ok = False
                            try:
                                if isinstance(closes, list) and len(closes) >= base_len + needed:
                                    ok = self._smart_entry_confirmed(sig, closes, needed)
                            except Exception:
                                ok = False
                            if ok:
                                confirmed.append(sig)
                            else:
                                remaining.append(item)
                        for sig in confirmed:
                            try:
                                all_signals.append(sig)
                                self._update_strategy_performance(sig.get('strategy_name', ''), sig, True)
                            except Exception:
                                pass
                        self.pending_signals = remaining
                except Exception as e:
                    self.logger.warning(f"Smart entry teyit kontrolü hatası: {e}")
                
                return all_signals
            
        except Exception as e:
            self.logger.error(f"Piyasa verisi işleme hatası: {e}")
            return all_signals

    def _load_hour_regime_whitelist(self) -> dict:
        try:
            import json, os
            root = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(root, 'data', 'hour_regime_whitelist.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception as e:
            self.logger.warning(f"Whitelist yükleme hatası: {e}")
        return {}

    def _is_whitelist_allowed(self, latest: Dict[str, Any] | None) -> bool:
        try:
            if not self.hour_regime_whitelist:
                return True
            from datetime import datetime as _dt
            hour = str(_dt.now().hour)
            ai = (latest or {}).get('ai_analysis', {}) if isinstance(latest, dict) else {}
            regime = str(ai.get('regime') or '').lower()
            row = self.hour_regime_whitelist.get(hour)
            if isinstance(row, dict):
                cell = row.get(regime)
                if isinstance(cell, str) and cell.upper() == 'OFF':
                    return False
                if isinstance(cell, bool) and cell is False:
                    return False
            return True
        except Exception:
            return True

    def _is_chop_regime(self, latest: Dict[str, Any] | None) -> bool:
        try:
            tech = (latest or {}).get('technical_analysis', {}) if isinstance(latest, dict) else {}
            adx = float(tech.get('adx', 0.0) or 0.0)
            ma20 = float(tech.get('ma_20', tech.get('ma20', 0.0)) or 0.0)
            ma50 = float(tech.get('ma_50', tech.get('ma50', 0.0)) or 0.0)
            atr = float(tech.get('atr', 0.0) or 0.0)
            if adx < 18.0 and atr > 0.0 and abs(ma20 - ma50) < 0.1 * atr:
                return True
            return False
        except Exception:
            return False

    def _micro_enrich(self, signal: Dict[str, Any], latest: Dict[str, Any] | None):
        try:
            ms = (latest or {}).get('microstructure', {}) if isinstance(latest, dict) else {}
            tech = (latest or {}).get('technical_analysis', {}) if isinstance(latest, dict) else {}
            bid = float(ms.get('bid', 0.0) or 0.0)
            ask = float(ms.get('ask', 0.0) or 0.0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(signal.get('entry_price', 0.0) or 0.0)
            spread_pct = ((ask - bid) / mid) if bid > 0 and ask > 0 and mid > 0 else float(ms.get('spread_pct', 0.0) or 0.0)
            regime = (latest or {}).get('ai_analysis', {}).get('regime') if isinstance(latest, dict) else None
            fee_rate = 0.0010
            slippage_est = float(ms.get('slippage_est_pct', 0.0006) or 0.0006)
            edge_cost = 2.0 * fee_rate + float(spread_pct or 0.0) + slippage_est
            signal['bid'] = bid
            signal['ask'] = ask
            signal['mid'] = mid
            signal['spread_pct'] = spread_pct
            signal['regime'] = regime
            signal['fee_rate'] = fee_rate
            signal['edge_cost'] = edge_cost
            try:
                if 'ml_success_prob' in signal:
                    signal['confluence'] = float(signal.get('ml_success_prob') or 0.0)
            except Exception:
                pass
        except Exception:
            pass

    def _extract_closes(self, market_data: Dict[str, Any], symbol: str) -> List[float]:
        """market_data içinden kapanış serisini güvenle çıkar"""
        try:
            if not symbol:
                return []
            sd = market_data.get(symbol)
            if isinstance(sd, dict):
                for k in ('closes', 'prices', 'close'):
                    v = sd.get(k)
                    if isinstance(v, list) and len(v) >= 5:
                        return [float(x) for x in v if x is not None]
            d = market_data.get('data')
            if isinstance(d, dict):
                sd = d.get(symbol)
                if isinstance(sd, dict):
                    for k in ('closes', 'prices', 'close'):
                        v = sd.get(k)
                        if isinstance(v, list) and len(v) >= 5:
                            return [float(x) for x in v if x is not None]
        except Exception:
            pass
        return []

    def _smart_entry_confirmed(self, signal: Dict[str, Any], closes: List[float], needed: int) -> bool:
        """Basit teyit: sinyal yönünde ardışık 'needed' kapanış momentum onayı"""
        try:
            side = (signal.get('side') or '').lower()
            if side not in ('buy', 'sell'):
                return False
            seq = closes[-(needed+1):]
            if len(seq) < needed + 1:
                return False
            for i in range(1, len(seq)):
                if side == 'buy' and not (seq[i] > seq[i-1]):
                    return False
                if side == 'sell' and not (seq[i] < seq[i-1]):
                    return False
            return True
        except Exception:
            return False

    def _volatility_factor(self, closes: List[float], window: int) -> float:
        """Getiri std'ünü basitçe 0..0.8 aralığına ölçekle."""
        try:
            if not closes or len(closes) < window + 1:
                return 0.2
            arr = [float(x) for x in closes[-(window+1):]]
            rets = []
            for i in range(1, len(arr)):
                prev = arr[i-1]
                rets.append((arr[i] - prev) / prev if prev != 0 else 0.0)
            # std hesapla
            m = sum(rets) / len(rets)
            var = sum((x - m)**2 for x in rets) / max(1, (len(rets)-1))
            sd = var ** 0.5
            vf = min(0.8, max(0.0, sd * 10.0))
            return float(vf)
        except Exception:
            return 0.2

    def _enrich_position_size_hint(self, signal: Dict[str, Any], closes: List[float]):
        """Sinyale position_size_hint: base * strength * (1 - vol_factor)"""
        try:
            strength = float(signal.get('strength', 0.5) or 0.5)
            base = float(self.position_sizing_cfg.get('base_fraction', 0.02))
            vw = int(self.position_sizing_cfg.get('vol_window', 20))
            vf = self._volatility_factor(closes, vw)
            # Kelly + confidence entegrasyonu
            strat_name = (signal.get('strategy_name') or '').lower()
            kelly = self._kelly_fraction(strat_name)
            conf = float(signal.get('ml_success_prob', strength))
            kelly_conf = max(0.0, min(1.0, kelly * conf))
            size_hint = max(0.0, base * strength * (1.0 - vf) * (1.0 + kelly_conf))
            signal['position_size_hint'] = size_hint
            signal['sizing_info'] = {
                'base_fraction': base,
                'strength': strength,
                'volatility_factor': vf,
                'kelly': kelly,
                'confidence': conf,
            }
        except Exception:
            pass

    def _kelly_fraction(self, strategy_name: str) -> float:
        """Basit Kelly oranı tahmini: W - (1-W)/R; R varsayılan 1.5, 0..0.1 arası sınırla."""
        try:
            perf = self.strategy_performance.get(strategy_name, {})
            total = float(perf.get('total_signals', 0) or 0)
            wins = float(perf.get('successful_signals', 0) or 0)
            if total <= 10:
                return 0.02
            W = wins / total if total > 0 else 0.5
            R = 1.5
            f = W - (1 - W) / max(0.1, R)
            return max(0.0, min(0.10, f))
        except Exception:
            return 0.02

    def _apply_ml_and_regime(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """ML tabanlı doğrulama ve rejim-ağırlıklı strength ayarı.
        True dönerse sinyal kabul, False ise filtrelenir.
        """
        try:
            # ML doğrulama devre dışıysa geç
            if not self.ml_validation_cfg.get('enabled', True):
                return True

            symbol = signal.get('symbol')
            strength = float(signal.get('strength', 0.5) or 0.5)
            accept_th = float(self.ml_validation_cfg.get('accept_threshold', 0.55))

            # Son analizden teknik özellikleri çek
            latest = None
            try:
                latest = market_analyzer.get_latest_analysis(symbol)
            except Exception:
                latest = None
            tech = (latest or {}).get('technical_analysis', {}) if isinstance(latest, dict) else {}
            features = {
                'rsi': tech.get('rsi', 50),
                'macd': tech.get('macd', 0),
                'adx': tech.get('adx', 15),
                'atr': tech.get('atr', 0),
                'volume_ratio': tech.get('volume_ratio', 1)
            }

            # ML başarı olasılığı
            p_succ = 0.6
            if hasattr(self, 'signal_classifier') and self.signal_classifier is not None:
                try:
                    p_succ = float(self.signal_classifier.predict_proba(features))
                except Exception:
                    p_succ = 0.6
            signal['ml_success_prob'] = p_succ
            if p_succ < accept_th:
                return False

            # Rejim ağırlığı
            if self.ml_validation_cfg.get('regime_weighting', True):
                regime = None
                ai = (latest or {}).get('ai_analysis', {}) if isinstance(latest, dict) else {}
                try:
                    regime = ai.get('regime')
                except Exception:
                    regime = None
                regime_str = str(regime or '').lower()

                strategy_name = (signal.get('strategy_name') or '').lower()
                weight = 1.0
                if 'trend' in regime_str:
                    # Trend rejimi: trend stratejilerine +, mean reversion'a -
                    if any(k in strategy_name for k in ['trend', 'reversal', 'volatility_breakout']):
                        weight = 1.1
                    elif 'mean_reversion' in strategy_name or 'grid' in strategy_name:
                        weight = 0.9
                elif 'sideways' in regime_str:
                    # Yatay rejim: mean reversion/grid +
                    if 'mean_reversion' in strategy_name or 'grid' in strategy_name:
                        weight = 1.1
                    elif any(k in strategy_name for k in ['trend', 'volatility_breakout']):
                        weight = 0.95
                elif 'volatile' in regime_str:
                    # Volatil rejim: momentum/volatility +
                    if any(k in strategy_name for k in ['momentum', 'volatility_breakout']):
                        weight = 1.1
                    else:
                        weight = 1.0

                # ML olasılığını da yumuşak çarpan olarak uygula
                w_ml = 0.8 + 0.4 * max(0.0, min(1.0, p_succ))  # 0.8..1.2
                new_strength = max(0.0, min(1.0, strength * weight * w_ml))
                signal['strength'] = new_strength
                signal['regime_info'] = {
                    'regime': regime_str,
                    'weight': weight,
                    'ml_weight': w_ml
                }
            return True
        except Exception:
            return True

    def _apply_external_filters(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Glassnode Exchange Inflow ve CoinGecko Sentiment Index ile ek filtreleme.
        - Inflow yüksekse satış baskısı: AL sinyalini reddet veya güç düşür.
        - Sentiment düşükse AL sinyalini filtrele (ayı piyasası haberi etkisi).
        True: kabul, False: filtrele.
        """
        try:
            side = (signal.get('side') or '').lower()
            if side not in ('buy', 'sell'):
                return True

            inflow_th = float(self.external_filter_cfg.get('exchange_inflow_threshold', 1.2))
            sent_th = float(self.external_filter_cfg.get('sentiment_min_threshold', 0.35))

            # Veriyi alma: öncelik external_data_manager, yoksa market_data['external']
            inflow = None
            sentiment = None
            # external_data_manager arayüzü belirsiz olduğu için güvenli erişim
            try:
                if self.external_data is not None:
                    # Beklenen alanlar (örnek): get_exchange_inflow(symbol) ve get_market_sentiment()
                    sym = signal.get('symbol')
                    if hasattr(self.external_data, 'get_exchange_inflow'):
                        inflow = self.external_data.get_exchange_inflow(sym)
                    if hasattr(self.external_data, 'get_market_sentiment'):
                        sentiment = self.external_data.get_market_sentiment()
            except Exception:
                pass

            ext = market_data.get('external', {}) if isinstance(market_data, dict) else {}
            if inflow is None:
                inflow = ext.get('exchange_inflow')
            if sentiment is None:
                # CoinGecko Fear&Greed benzeri 0..1 normalize
                sentiment = ext.get('market_sentiment_index')

            # AL sinyali filtreleme koşulları
            if side == 'buy':
                # Exchange inflow yüksekse satış baskısı
                if inflow is not None:
                    try:
                        if float(inflow) >= inflow_th:
                            return False
                    except Exception:
                        pass
                # Sentiment çok düşükse AL filtrele
                if sentiment is not None:
                    try:
                        if float(sentiment) < sent_th:
                            return False
                    except Exception:
                        pass

            # SELL için yumuşak güçlendirme: inflow yüksekse strength'i biraz artır
            if side == 'sell' and inflow is not None:
                try:
                    if float(inflow) >= inflow_th:
                        s = float(signal.get('strength', 0.5))
                        signal['strength'] = min(1.0, s * 1.05)
                except Exception:
                    pass
            return True
        except Exception:
            return True
    
    def _update_strategy_performance(self, strategy_name: str, signal: Dict[str, Any], success: bool):
        """Strateji performansını güncelle"""
        try:
            if strategy_name not in self.strategy_performance:
                return
            
            perf = self.strategy_performance[strategy_name]
            perf['total_signals'] += 1
            perf['last_update'] = datetime.now()
            
            if success:
                perf['successful_signals'] += 1
            else:
                perf['failed_signals'] += 1
            
            # P&L güncellemesi (eğer sinyal başarılıysa)
            if success and signal:
                pnl = signal.get('expected_pnl', 0)
                perf['total_pnl'] += pnl
            
        except Exception as e:
            self.logger.error(f"Performans güncelleme hatası: {e}")
    
    def get_strategy_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Strateji durumunu al"""
        try:
            with self.lock:
                if name not in self.strategies:
                    return None
                
                strategy = self.strategies[name]
                status = self.strategy_status[name]
                performance = self.strategy_performance.get(name, {})
                
                try:
                    perf_metrics = strategy.get_performance_metrics()
                except Exception:
                    perf_metrics = {}
                try:
                    strat_info = strategy.get_strategy_info() if hasattr(strategy, 'get_strategy_info') else {}
                except Exception:
                    strat_info = {}
                return {
                    'name': name,
                    'status': status.value,
                    'is_active': getattr(strategy, 'is_active', False),
                    'total_trades': getattr(strategy, 'total_trades', 0),
                    'winning_trades': getattr(strategy, 'winning_trades', 0),
                    'losing_trades': getattr(strategy, 'losing_trades', 0),
                    'total_profit': getattr(strategy, 'total_profit', 0.0),
                    'max_drawdown': getattr(strategy, 'max_drawdown', 0.0),
                    'performance_metrics': perf_metrics,
                    'strategy_info': strat_info,
                    'performance_stats': performance
                }
                
        except Exception as e:
            self.logger.error(f"Strateji durumu alma hatası: {e}")
            return None
    
    def get_all_strategies_status(self) -> Dict[str, Dict[str, Any]]:
        """Tüm strateji durumlarını al"""
        try:
            with self.lock:
                status = {}
                for name in self.strategies:
                    status[name] = self.get_strategy_status(name)
                return status
                
        except Exception as e:
            self.logger.error(f"Tüm strateji durumları alma hatası: {e}")
            return {}
    
    def get_active_strategies(self) -> List[str]:
        """Aktif stratejileri al"""
        try:
            with self.lock:
                return [name for name, status in self.strategy_status.items() 
                       if status == StrategyStatus.ACTIVE]
                
        except Exception as e:
            self.logger.error(f"Aktif stratejiler alma hatası: {e}")
            return []
    
    def get_available_strategies(self) -> List[str]:
        """Mevcut stratejilerin listesini döndür"""
        try:
            with self.lock:
                return list(self.strategies.keys())
        except Exception as e:
            self.logger.error(f"Mevcut stratejiler alma hatası: {e}")
            return []

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Strateji hakkında bilgi döndür (temel alanlar + config/performance mümkünse)"""
        try:
            with self.lock:
                if name not in self.strategies:
                    return None
                strategy = self.strategies[name]
                status = self.strategy_status.get(name, StrategyStatus.INACTIVE)
                info: Dict[str, Any] = {
                    'name': name,
                    'is_active': getattr(strategy, 'is_active', False),
                    'type': strategy.__class__.__name__,
                    'performance': {},
                    'config': getattr(strategy, 'config', {}),
                }
                try:
                    if hasattr(strategy, 'get_performance_metrics'):
                        info['performance'] = strategy.get_performance_metrics()
                except Exception:
                    pass
                info['status'] = status.value if isinstance(status, StrategyStatus) else str(status)
                return info
        except Exception as e:
            self.logger.error(f"Strateji bilgisi alma hatası: {e}")
            return None

    def activate_strategy(self, name: str) -> bool:
        """Uyumlu arayüz: stratejiyi aktif et"""
        return self.start_strategy(name)

    def deactivate_strategy(self, name: str) -> bool:
        """Uyumlu arayüz: stratejiyi durdur"""
        return self.stop_strategy(name)
    
    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, Any]:
        """Strateji performansını döndür"""
        try:
            with self.lock:
                if strategy_name:
                    return self.strategy_performance.get(strategy_name, {})
                else:
                    return self.strategy_performance.copy()
        except Exception as e:
            self.logger.error(f"Strateji performansı alma hatası: {e}")
            return {}
    
    def update_strategy_config(self, name: str, new_config: Dict[str, Any]) -> bool:
        """Strateji konfigürasyonunu güncelle"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                strategy = self.strategies[name]
                strategy.update_config(new_config)
                
                self.logger.info(f"Strateji '{name}' konfigürasyonu güncellendi")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji konfigürasyon güncelleme hatası: {e}")
            return False
    
    def reset_strategy_performance(self, name: str) -> bool:
        """Strateji performansını sıfırla"""
        try:
            with self.lock:
                if name not in self.strategies:
                    self.logger.error(f"Strateji '{name}' bulunamadı")
                    return False
                
                strategy = self.strategies[name]
                strategy.reset_performance()
                
                # Performans istatistiklerini sıfırla
                self.strategy_performance[name] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'failed_signals': 0,
                    'total_pnl': 0.0,
                    'last_update': datetime.now()
                }
                
                self.logger.info(f"Strateji '{name}' performansı sıfırlandı")
                return True
                
        except Exception as e:
            self.logger.error(f"Strateji performans sıfırlama hatası: {e}")
            return False
    
    def add_signal_callback(self, callback):
        """Sinyal callback'i ekle"""
        self.signal_callbacks.append(callback)
    
    def add_performance_callback(self, callback):
        """Performans callback'i ekle"""
        self.performance_callbacks.append(callback)
    
    def _notify_signal_callbacks(self, signals: List[Dict[str, Any]]):
        """Sinyal callback'lerini çağır"""
        for callback in self.signal_callbacks:
            try:
                callback(signals)
            except Exception as e:
                self.logger.error(f"Sinyal callback hatası: {e}")
    
    def _notify_performance_callbacks(self, performance: Dict[str, Any]):
        """Performans callback'lerini çağır"""
        for callback in self.performance_callbacks:
            try:
                callback(performance)
            except Exception as e:
                self.logger.error(f"Performans callback hatası: {e}")
    
    def _on_regime_change(self, regime):
        """Piyasa rejimi değiştiğinde çağrılır"""
        try:
            import time
            
            # Cooldown kontrolü
            now = time.time()
            if now - self._last_regime_change < self._regime_apply_cooldown:
                self.logger.debug(f"Rejim değişimi cooldown içinde, atlanıyor")
                return
            
            self._last_regime_change = now
            
            if regime is None:
                self.logger.error(f"Regime None!")
                return
            
            regime_type = getattr(regime, 'regime_type', 'UNKNOWN')
            confidence = getattr(regime, 'confidence', 0.0)
            
            self.logger.info(f"🔄 Piyasa rejimi değişti: {regime_type} (güven: {confidence:.2f})")
            
            # Regime adını kaydet
            regime_map = {
                'trending': 'trend',
                'sideways': 'yatay',
                'volatile': 'volatil',
                'crash': 'çöküş'
            }
            self._regime_name = regime_map.get(str(regime_type).lower(), str(regime_type).lower())
            
            # Rejime göre strateji değiştir
            regime_strategies = {
                'trending': ['trend_following', 'volatility_breakout'],
                'sideways': ['grid', 'mean_reversion'],
                'volatile': ['scalping', 'momentum_ignition', 'volatility_breakout'],
                'crash': ['dca', 'hedge']
            }
            
            target_strategies = regime_strategies.get(str(regime_type).lower(), [])
            if target_strategies:
                # Mevcut aktif stratejileri al
                active_strategies = [name for name, status in self.strategy_status.items() 
                                   if status == StrategyStatus.ACTIVE and name != 'dynamic_selector']
                
                # Hedefte olmayan stratejileri durdur
                for strategy_name in active_strategies:
                    if strategy_name not in target_strategies:
                        self.stop_strategy(strategy_name)
                        self.logger.info(f"Strateji durduruldu: {strategy_name}")
                
                # Hedef stratejileri başlat (limit dahilinde)
                started_count = 0
                for strategy_name in target_strategies:
                    if started_count >= self.max_active_strategies:
                        break
                    if strategy_name in self.strategies:
                        if self.strategy_status.get(strategy_name) != StrategyStatus.ACTIVE:
                            success = self.start_strategy(strategy_name)
                            if success:
                                self.logger.info(f"Strateji başlatıldı: {strategy_name}")
                                started_count += 1
                
                self.logger.info(f"Rejim değişimi tamamlandı: {regime_type} -> {started_count} strateji aktif")
                
        except Exception as e:
            self.logger.error(f"_on_regime_change callback hatası: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _on_market_analysis(self, analysis: dict):
        """MarketAnalyzer analiz callback'i: BTCUSDT için DeepSeek rejimini StrategyManager'a yansıt."""
        try:
            if not isinstance(analysis, dict):
                return
            sym = str(analysis.get('symbol') or '').upper()
            if sym != 'BTCUSDT':
                return
            ai = analysis.get('ai_analysis') or {}
            regime_info = ai.get('regime')
            if not regime_info:
                return
            # regime_info dict veya string olabilir; her iki durumda da sarmala
            if isinstance(regime_info, dict):
                regime_type = regime_info.get('regime_type') or regime_info.get('regime') or regime_info.get('label')
                try:
                    confidence = float(regime_info.get('confidence', 0.0) or 0.0)
                except Exception:
                    confidence = 0.0
            else:
                regime_type = str(regime_info)
                confidence = 0.0
            if not regime_type:
                return
            # _on_regime_change, attribute tabanlı bir nesne bekliyor; hafif bir wrapper oluştur
            regime_obj = type('Regime', (), {'regime_type': regime_type, 'confidence': confidence})()
            self._on_regime_change(regime_obj)
        except Exception as e:
            try:
                self.logger.error(f"_on_market_analysis callback hatası: {e}")
            except Exception:
                pass
    
    def _on_strategy_change(self, recommendation):
        """Strateji önerisi değiştiğinde çağrılır"""
        try:
            if recommendation is None:
                return
            
            strategy_name = getattr(recommendation, 'strategy_name', None)
            confidence = getattr(recommendation, 'confidence', 0.0)
            
            if strategy_name:
                self.logger.info(f"🎯 Strateji önerisi: {strategy_name} (güven: {confidence:.2f})")
                
        except Exception as e:
            self.logger.error(f"_on_strategy_change callback hatası: {e}")
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """Genel performans metriklerini al"""
        try:
            with self.lock:
                total_trades = 0
                total_profit = 0.0
                total_signals = 0
                successful_signals = 0
                
                for name, strategy in self.strategies.items():
                    try:
                        total_trades += getattr(strategy, 'total_trades', 0)
                    except Exception:
                        pass
                    try:
                        total_profit += float(getattr(strategy, 'total_profit', 0.0) or 0.0)
                    except Exception:
                        pass
                    
                    perf = self.strategy_performance.get(name, {})
                    total_signals += perf.get('total_signals', 0)
                    successful_signals += perf.get('successful_signals', 0)
                
                success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0
                try:
                    active_count = len([name for name, status in self.strategy_status.items() if status == StrategyStatus.ACTIVE])
                except Exception:
                    active_count = 0
                return {
                    'total_trades': total_trades,
                    'total_profit': total_profit,
                    'total_signals': total_signals,
                    'successful_signals': successful_signals,
                    'success_rate': success_rate,
                    'active_strategies': active_count,
                    'total_strategies': len(self.strategies)
                }
                
        except Exception as e:
            self.logger.error(f"Genel performans alma hatası: {e}")
            return {}
    
    def auto_switch_strategy(self, market_regime: str) -> bool:
        """Piyasa rejimine göre otomatik strateji değiştir"""
        try:
            # Rejim-strateji eşleştirmesi
            regime_strategy_map = {
                'yatay': 'grid',
                'volatil': 'scalping', 
                'trend': 'trend_following',
                'cokus': 'dca',
                'korelasyon': 'hedge'
            }
            
            target_strategy = regime_strategy_map.get(market_regime.lower())
            if not target_strategy:
                self.logger.warning(f"Bilinmeyen piyasa rejimi: {market_regime}")
                return False
            
            # Mevcut aktif stratejiyi durdur
            active_strategies = self.get_active_strategies()
            for strategy_name in active_strategies.keys():
                if strategy_name != target_strategy:
                    self.stop_strategy(strategy_name)
                    self.logger.info(f"Eski strateji durduruldu: {strategy_name}")
            
            # Hedef stratejiyi başlat
            if target_strategy in self.strategies:
                success = self.start_strategy(target_strategy)
                if success:
                    self.logger.info(f"Otomatik strateji değişimi: {market_regime} -> {target_strategy}")
                    return True
                else:
                    self.logger.error(f"Hedef strateji başlatılamadı: {target_strategy}")
                    return False
            else:
                self.logger.error(f"Hedef strateji bulunamadı: {target_strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Otomatik strateji değişim hatası: {e}")
            return False

# Global strateji yöneticisi
strategy_manager = StrategyManager()

