"""
Risk Yönetimi Modülü
Pozisyon, kayıp ve risk kontrolü sistemi
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from config.config import config
import math
import numpy as np

@dataclass
class RiskLimits:
    """Risk limitleri"""
    max_position_size: float = 1.0  # Portföyün %100'u (tam sermaye kullanımı)
    max_daily_loss: float = 0.05    # Günlük maksimum kayıp %5
    max_weekly_loss: float = 0.15   # Haftalık maksimum kayıp %15
    max_monthly_loss: float = 0.30  # Aylık maksimum kayıp %30
    max_drawdown: float = 0.20      # Maksimum düşüş %20
    max_open_positions: int = 5     # Maksimum açık pozisyon
    max_correlation: float = 0.7    # Maksimum korelasyon
    stop_loss_pct: float = 0.02     # Stop loss %2
    take_profit_pct: float = 0.04   # Take profit %4
    trailing_stop_pct: float = 0.01 # Trailing stop %1
    slippage_tolerance: float = 0.001 # Slippage toleransı %0.1
    min_trade_amount: float = 250.0 # Minimum işlem tutarı (USD)
    max_position_hours: float = 24.0
    time_stop_hours: float = 3.0
    max_daily_volatility: float = 0.01   # %1 günlük hedef vol limiti
    volatility_lookback_hours: int = 24  # son 24 saati baz al

class RiskLevel(Enum):
    """Risk seviyeleri"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskManager:
    """Risk yöneticisi sınıfı"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.logger = logging.getLogger(__name__)
        
        # Risk limitleri
        self.limits = RiskLimits()
        
        # Hesap bilgileri
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.available_balance = initial_balance
        self.used_balance = 0.0
        
        # Pozisyon yönetimi
        self.positions = {}  # {position_id: position_data}
        self.pending_orders = {}  # {order_id: order_data}
        
        # Risk takibi
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.total_pnl = 0.0
        self.max_balance = initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Zaman takibi
        self.last_reset_date = datetime.now().date()
        self.last_week_reset = datetime.now().date()
        self.last_month_reset = datetime.now().date()
        
        # Risk durumu
        self.risk_level = RiskLevel.LOW
        self.trading_enabled = True
        self.emergency_stop = False
        
        # Thread yönetimi
        self.lock = threading.Lock()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Callback'ler
        self.risk_alert_callbacks = []
        self.position_callbacks = []

        # Günlük equity oynaklığı için state
        self.equity_history: list[tuple[datetime, float]] = []
        self.current_volatility: float = 0.0

        # Piyasa maliyetleri (Binance): yüzdeler
        try:
            self.spread_pct: float = 0.0
            self.fee_pct: float = float(getattr(config.trading, 'commission_rate_taker', 0.001) or 0.001)
            self.slip_pct: float = float(getattr(config.trading, 'slippage_tolerance', 0.001) or 0.001)
        except Exception:
            self.spread_pct = 0.0
            self.fee_pct = 0.001
            self.slip_pct = 0.001
        
        self.logger.info(f"Risk yöneticisi başlatıldı - Başlangıç bakiyesi: {initial_balance:,.2f}")
    
    def apply_settings(self, settings: Optional[Dict[str, Any]] = None):
        """GUI'den çağrılan ayar uygulama metodu (opsiyonel).
        Sağlanan ayarlarda limit alanları varsa günceller, aksi halde no-op.
        """
        try:
            if settings and isinstance(settings, dict):
                # RiskLimits alanları ile eşleşenleri güncelle
                for k, v in settings.items():
                    if hasattr(self.limits, k):
                        try:
                            setattr(self.limits, k, v)
                        except Exception:
                            pass
                self.logger.info("Risk ayarları uygulandı (GUI)")
            else:
                # No-op: sadece logla
                self.logger.info("apply_settings çağrıldı (parametresiz)")
        except Exception as e:
            self.logger.warning(f"apply_settings hata: {e}")

    def start_monitoring(self):
        """Risk izlemeyi başlat"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Risk izleme başlatıldı")
    
    def stop_monitoring(self):
        """Risk izlemeyi durdur"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Risk izleme durduruldu")
    
    def _monitoring_loop(self):
        """Risk izleme döngüsü"""
        while self.is_monitoring:
            try:
                # Equity snapshot'ını güncelle (günlük vol hesabı için)
                try:
                    self._update_equity_snapshot()
                except Exception:
                    pass
                
                # Günlük sıfırlama kontrolü
                self._check_daily_reset()
                
                # Risk seviyesini güncelle
                self._update_risk_level()
                
                # Pozisyonları kontrol et
                self._monitor_positions()
                
                # Risk limitlerini kontrol et
                self._check_risk_limits()

                # Günlük realized volatilite limitini kontrol et
                try:
                    self._check_volatility_limit()
                except Exception:
                    pass
                
                # 30 saniye bekle
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Risk izleme döngüsü hatası: {e}")
                time.sleep(60)

    def _check_volatility_limit(self):
        """Günlük realized vol limitini kontrol et."""
        try:
            if not self.equity_history or len(self.equity_history) < 5:
                return  # yeterli veri yok

            times, equities = zip(*self.equity_history)
            # Zaman penceresi
            window_hours = (times[-1] - times[0]).total_seconds() / 3600.0
            if window_hours <= 0:
                return

            # Log-return serisi
            rets = []
            prev_eq = equities[0]
            for eq in equities[1:]:
                if prev_eq <= 0 or eq <= 0:
                    prev_eq = eq
                    continue
                r = math.log(eq / prev_eq)
                rets.append(r)
                prev_eq = eq

            if len(rets) < 3:
                return

            sigma_window = float(np.std(rets, ddof=1))

            # Pencereyi günlük hale ölçekle
            if window_hours < 24:
                sigma_daily = sigma_window * math.sqrt(24.0 / window_hours)
            else:
                sigma_daily = sigma_window  # pencere ≈ 1 gün kabul

            self.current_volatility = sigma_daily

            if sigma_daily > self.limits.max_daily_volatility:
                self.trading_enabled = False
                msg = (
                    f"Günlük realized volatilite limiti aşıldı: "
                    f"{sigma_daily:.2%} > {self.limits.max_daily_volatility:.2%}"
                )
                self._notify_risk_alert("DAILY_VOL_LIMIT", msg)
                self.logger.warning(msg)

        except Exception as e:
            self.logger.error(f"Volatilite limiti kontrol hatası: {e}")

    def _update_equity_snapshot(self):
        try:
            now = datetime.now()
            # Basit equity hesabı: current_balance + tüm unrealized PnL
            total_unrealized = 0.0
            for pos in self.positions.values():
                try:
                    upnl = float(pos.get('unrealized_pnl', 0.0) or 0.0)
                    total_unrealized += upnl
                except Exception:
                    continue
            equity = self.current_balance + total_unrealized
            if equity <= 0:
                return

            # Listeye ekle
            self.equity_history.append((now, equity))

            # Lookback dışındaki verileri temizle
            lookback = timedelta(hours=self.limits.volatility_lookback_hours)
            cutoff = now - lookback
            self.equity_history = [(t, e) for (t, e) in self.equity_history if t >= cutoff]

        except Exception as e:
            self.logger.warning(f"Equity snapshot güncelleme hatası: {e}")
    
    def _check_daily_reset(self):
        """Günlük sıfırlama kontrolü"""
        current_date = datetime.now().date()
        
        # Günlük sıfırlama
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            # Günlük equity serisini ve vol ölçümünü de sıfırla
            self.equity_history = []
            self.current_volatility = 0.0
            self.last_reset_date = current_date
            self.logger.info("Günlük P&L sıfırlandı")
        
        # Haftalık sıfırlama
        if current_date > self.last_week_reset + timedelta(days=7):
            self.weekly_pnl = 0.0
            self.last_week_reset = current_date
            self.logger.info("Haftalık P&L sıfırlandı")
        
        # Aylık sıfırlama
        if current_date > self.last_month_reset + timedelta(days=30):
            self.monthly_pnl = 0.0
            self.last_month_reset = current_date
            self.logger.info("Aylık P&L sıfırlandı")
    
    def _update_risk_level(self):
        """Risk seviyesini güncelle"""
        try:
            # Drawdown hesapla
            self.current_drawdown = (self.max_balance - self.current_balance) / self.max_balance
            
            # Risk seviyesini belirle
            if self.current_drawdown > 0.15 or self.daily_pnl < -self.limits.max_daily_loss * self.current_balance:
                self.risk_level = RiskLevel.CRITICAL
            elif self.current_drawdown > 0.10 or self.weekly_pnl < -self.limits.max_weekly_loss * self.current_balance:
                self.risk_level = RiskLevel.HIGH
            elif self.current_drawdown > 0.05 or self.monthly_pnl < -self.limits.max_monthly_loss * self.current_balance:
                self.risk_level = RiskLevel.MEDIUM
            else:
                self.risk_level = RiskLevel.LOW
            
            # Trading durumunu güncelle
            if self.risk_level == RiskLevel.CRITICAL:
                self.trading_enabled = False
                self.emergency_stop = True
                self._notify_risk_alert("CRITICAL", "Trading durduruldu - Kritik risk seviyesi")
            elif self.risk_level == RiskLevel.HIGH:
                self.trading_enabled = False
                self._notify_risk_alert("HIGH", "Trading durduruldu - Yüksek risk seviyesi")
            else:
                self.trading_enabled = True
                self.emergency_stop = False
                
        except Exception as e:
            self.logger.error(f"Risk seviyesi güncelleme hatası: {e}")
    
    def _monitor_positions(self):
        """Pozisyonları izle"""
        try:
            for position_id, position in self.positions.items():
                # Stop loss kontrolü
                if self._check_stop_loss(position):
                    self._close_position(position_id, "stop_loss")
                
                # Take profit kontrolü
                elif self._check_take_profit(position):
                    self._close_position(position_id, "take_profit")
                
                # Trailing stop kontrolü
                elif self._check_trailing_stop(position):
                    self._close_position(position_id, "trailing_stop")
                
                # Zaman bazlı çıkış kontrolü
                elif self._check_time_based_exit(position):
                    self._close_position(position_id, "time_exit")
                
        except Exception as e:
            self.logger.error(f"Pozisyon izleme hatası: {e}")
    
    def _check_risk_limits(self):
        """Risk limitlerini kontrol et"""
        try:
            # Günlük kayıp kontrolü
            if self.daily_pnl < -self.limits.max_daily_loss * self.current_balance:
                self._notify_risk_alert("DAILY_LOSS", f"Günlük kayıp limiti aşıldı: {self.daily_pnl:,.2f}")
                self.trading_enabled = False
            
            # Haftalık kayıp kontrolü
            if self.weekly_pnl < -self.limits.max_weekly_loss * self.current_balance:
                self._notify_risk_alert("WEEKLY_LOSS", f"Haftalık kayıp limiti aşıldı: {self.weekly_pnl:,.2f}")
                self.trading_enabled = False
            
            # Aylık kayıp kontrolü
            if self.monthly_pnl < -self.limits.max_monthly_loss * self.current_balance:
                self._notify_risk_alert("MONTHLY_LOSS", f"Aylık kayıp limiti aşıldı: {self.monthly_pnl:,.2f}")
                self.trading_enabled = False
            
            # Maksimum düşüş kontrolü
            if self.current_drawdown > self.limits.max_drawdown:
                self._notify_risk_alert("MAX_DRAWDOWN", f"Maksimum düşüş limiti aşıldı: {self.current_drawdown:.2%}")
                self.trading_enabled = False
            
            # Açık pozisyon sayısı kontrolü
            open_positions = len([p for p in self.positions.values() if p['status'] == 'open'])
            if open_positions >= self.limits.max_open_positions:
                self._notify_risk_alert("MAX_POSITIONS", f"Maksimum pozisyon sayısı aşıldı: {open_positions}")
                self.trading_enabled = False
                
        except Exception as e:
            self.logger.error(f"Risk limit kontrolü hatası: {e}")
    
    def validate_position(self, symbol: str, side: str, size: float, price: float) -> Tuple[bool, str]:
        """Pozisyon geçerliliğini kontrol et"""
        try:
            with self.lock:
                # Trading durumu kontrolü
                if not self.trading_enabled:
                    return False, "Trading devre dışı"
                
                if self.emergency_stop:
                    return False, "Acil durdurma aktif"
                
                # Pozisyon büyüklüğü kontrolü
                position_value = size * price
                if position_value > self.available_balance * self.limits.max_position_size:
                    return False, f"Pozisyon büyüklüğü limiti aşıldı: {position_value:,.2f}"
                
                # Minimum pozisyon büyüklüğü
                min_amount = getattr(self.limits, 'min_trade_amount', 250.0)
                if position_value < min_amount:
                    # Eğer bakiye minimum tutarın altındaysa, tüm bakiyeyi kullanmaya izin ver
                    if self.available_balance < min_amount:
                        if position_value < self.available_balance * 0.95:
                            return False, f"Bakiye < {min_amount} USD olduğunda tüm bakiye kullanılmalı"
                    else:
                        return False, f"Pozisyon büyüklüğü minimum tutarın ({min_amount} USD) altında"

                # Cost-floor koruması: TP eşiği ≥ 3×(spread + 2×fee + slip)
                # Mevcut limitlerdeki take_profit_pct değeri ile kıyasla
                try:
                    effective_cost = float(self.spread_pct or 0.0) + 2.0*float(self.fee_pct or 0.0) + float(self.slip_pct or 0.0)
                    required_tp = 3.0 * effective_cost
                    if float(self.limits.take_profit_pct or 0.0) < required_tp:
                        return False, f"Maliyet tabanı eşiği sağlanmadı (gereken TP ≥ {required_tp:.4f})"
                except Exception:
                    pass
                
                # Aynı sembol için açık pozisyon kontrolü
                open_positions = [p for p in self.positions.values() 
                                if p['symbol'] == symbol and p['status'] == 'open']
                if len(open_positions) > 0:
                    return False, f"{symbol} için zaten açık pozisyon var"
                
                # Korelasyon kontrolü
                if not self._check_correlation(symbol, side):
                    return False, "Yüksek korelasyon riski"
                
                # Volatilite kontrolü
                if not self._check_volatility(symbol, price):
                    return False, "Yüksek volatilite riski"
                
                return True, "Pozisyon onaylandı"
                
        except Exception as e:
            self.logger.error(f"Pozisyon doğrulama hatası: {e}")
            return False, f"Doğrulama hatası: {e}"

    def update_market_costs(self, spread_pct: Optional[float] = None, fee_taker_pct: Optional[float] = None, slip_pct: Optional[float] = None):
        """Piyasa maliyetlerini dinamik güncelle (yüzde değerler)."""
        try:
            if spread_pct is not None:
                self.spread_pct = max(0.0, float(spread_pct))
            if fee_taker_pct is not None:
                self.fee_pct = max(0.0, float(fee_taker_pct))
            if slip_pct is not None:
                self.slip_pct = max(0.0, float(slip_pct))
            self.logger.debug(f"Market costs updated spread={self.spread_pct:.6f}, fee={self.fee_pct:.6f}, slip={self.slip_pct:.6f}")
        except Exception as e:
            self.logger.warning(f"update_market_costs hata: {e}")
    
    def add_position(self, symbol: str, side: str, size: float, price: float, 
                    stop_loss: float = None, take_profit: float = None) -> str:
        """Pozisyon ekle"""
        try:
            with self.lock:
                # Pozisyon doğrulama
                is_valid, message = self.validate_position(symbol, side, size, price)
                if not is_valid:
                    self.logger.warning(f"Pozisyon reddedildi: {message}")
                    return None
                
                # Pozisyon ID oluştur
                position_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # Stop loss ve take profit hesapla
                if stop_loss is None:
                    if side == 'buy':
                        stop_loss = price * (1 - self.limits.stop_loss_pct)
                    else:
                        stop_loss = price * (1 + self.limits.stop_loss_pct)
                
                if take_profit is None:
                    if side == 'buy':
                        take_profit = price * (1 + self.limits.take_profit_pct)
                    else:
                        take_profit = price * (1 - self.limits.take_profit_pct)
                
                # Pozisyon oluştur
                position = {
                    'id': position_id,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': price,
                    'current_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,  # Trailing stop başlangıçta stop_loss seviyesinde
                    'max_price': price,  # Long için en yüksek fiyat takibi
                    'min_price': price,  # Short için en düşük fiyat takibi
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'status': 'open',
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }
                
                # Pozisyonu ekle
                self.positions[position_id] = position
                
                # Bakiyeyi güncelle
                position_value = size * price
                self.used_balance += position_value
                self.available_balance -= position_value
                
                self.logger.info(f"Pozisyon eklendi: {position_id} - {symbol} {side} {size:.4f} @ {price:.2f}")
                
                # Callback'leri çağır
                self._notify_position_callbacks('position_added', position)
                
                return position_id
                
        except Exception as e:
            self.logger.error(f"Pozisyon ekleme hatası: {e}")
            return None
    
    def update_position(self, position_id: str, current_price: float):
        """Pozisyonu güncelle"""
        try:
            with self.lock:
                if position_id not in self.positions:
                    return
                
                position = self.positions[position_id]
                position['current_price'] = current_price
                position['updated_at'] = datetime.now()
                
                # Unrealized P&L hesapla
                if position['side'] == 'buy':
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
                else:
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
                
                # Trailing stop güncelle - %2 KAR SONRASI AKTİF
                # Pozisyon minimum %2 kar yapana kadar trailing stop devre dışı
                # Bu sayede küçük dalgalanmalarda erken çıkış engellenmiş olur
                
                profit_pct = position['unrealized_pnl'] / (position['size'] * position['entry_price']) if position['size'] * position['entry_price'] > 0 else 0
                trailing_activation_threshold = 0.02  # %2 kar eşiği
                
                if position['side'] == 'buy':
                    # Long pozisyon: Fiyat yükseldikçe trailing stop yukarı kayar
                    
                    # En yüksek fiyatı takip et
                    if 'max_price' not in position:
                        position['max_price'] = position['entry_price']
                    if current_price > position['max_price']:
                        position['max_price'] = current_price
                    
                    # TRAILING STOP SADECE %2 KAR SONRASI AKTİF
                    if profit_pct >= trailing_activation_threshold:
                        # Trailing stop hesapla (en yüksek fiyattan trailing_stop_pct düşük)
                        calculated_trailing = position['max_price'] * (1 - self.limits.trailing_stop_pct)
                        
                        # Trailing stop asla aşağı inmez, sadece yukarı çıkar
                        if calculated_trailing > position['trailing_stop']:
                            position['trailing_stop'] = calculated_trailing
                        
                        # Trailing stop asla entry_price'ın altına düşmemeli (kar koruma)
                        min_trailing = position['entry_price'] * 1.01  # En az %1 kar garantile
                        if position['trailing_stop'] < min_trailing:
                            position['trailing_stop'] = min_trailing
                    else:
                        # %2 kar yapılmadı, trailing stop başlangıç stop loss seviyesinde kalır
                        # Böylece sadece normal stop loss aktif olur
                        position['trailing_stop'] = position['stop_loss']
                
                elif position['side'] == 'sell':
                    # Short pozisyon: Fiyat düştükçe trailing stop aşağı kayar
                    
                    # En düşük fiyatı takip et
                    if 'min_price' not in position:
                        position['min_price'] = position['entry_price']
                    if current_price < position['min_price']:
                        position['min_price'] = current_price
                    
                    # TRAILING STOP SADECE %2 KAR SONRASI AKTİF
                    if profit_pct >= trailing_activation_threshold:
                        # Trailing stop hesapla (en düşük fiyattan trailing_stop_pct yüksek)
                        calculated_trailing = position['min_price'] * (1 + self.limits.trailing_stop_pct)
                        
                        # Trailing stop asla yukarı çıkmaz, sadece aşağı iner
                        if calculated_trailing < position['trailing_stop']:
                            position['trailing_stop'] = calculated_trailing
                        
                        # Trailing stop asla entry_price'ın üstüne çıkmamalı (kar koruma)
                        max_trailing = position['entry_price'] * 0.99  # En az %1 kar garantile
                        if position['trailing_stop'] > max_trailing:
                            position['trailing_stop'] = max_trailing
                    else:
                        # %2 kar yapılmadı, trailing stop başlangıç stop loss seviyesinde kalır
                        # Böylece sadece normal stop loss aktif olur
                        position['trailing_stop'] = position['stop_loss']
                
                # Maksimum bakiyeyi güncelle
                current_balance = self.initial_balance + self.total_pnl + position['unrealized_pnl']
                if current_balance > self.max_balance:
                    self.max_balance = current_balance
                
                self.current_balance = current_balance
                
                # Equity snapshot'ını güncelle (equity = balance + unrealized)
                try:
                    self._update_equity_snapshot()
                except Exception:
                    pass
                
        except Exception as e:
            self.logger.error(f"Pozisyon güncelleme hatası: {e}")
    
    def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """Pozisyonu kapat"""
        try:
            with self.lock:
                if position_id not in self.positions:
                    return False
                
                position = self.positions[position_id]
                if position['status'] != 'open':
                    return False
                
                # Realized P&L hesapla
                current_price = position['current_price']
                if position['side'] == 'buy':
                    realized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    realized_pnl = (position['entry_price'] - current_price) * position['size']
                
                # Pozisyonu kapat
                position['status'] = 'closed'
                position['exit_price'] = current_price
                position['realized_pnl'] = realized_pnl
                position['close_reason'] = reason
                position['closed_at'] = datetime.now()
                
                # Bakiyeyi güncelle
                position_value = position['size'] * position['entry_price']
                self.used_balance -= position_value
                self.available_balance += position_value + realized_pnl
                
                # P&L'yi güncelle
                self.total_pnl += realized_pnl
                self.daily_pnl += realized_pnl
                self.weekly_pnl += realized_pnl
                self.monthly_pnl += realized_pnl
                
                self.logger.info(f"Pozisyon kapatıldı: {position_id} - P&L: {realized_pnl:,.2f} - Sebep: {reason}")
                
                # Callback'leri çağır
                self._notify_position_callbacks('position_closed', position)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Pozisyon kapatma hatası: {e}")
            return False
    
    def _check_stop_loss(self, position: Dict[str, Any]) -> bool:
        """Stop loss kontrolü"""
        try:
            current_price = position['current_price']
            stop_loss = position['stop_loss']
            
            if position['side'] == 'buy':
                return current_price <= stop_loss
            else:
                return current_price >= stop_loss
        except:
            return False
    
    def _check_take_profit(self, position: Dict[str, Any]) -> bool:
        """Take profit kontrolü"""
        try:
            current_price = position['current_price']
            take_profit = position['take_profit']
            
            if position['side'] == 'buy':
                return current_price >= take_profit
            else:
                return current_price <= take_profit
        except:
            return False
    
    def _check_trailing_stop(self, position: Dict[str, Any]) -> bool:
        """Trailing stop kontrolü"""
        try:
            current_price = position['current_price']
            trailing_stop = position['trailing_stop']
            
            if position['side'] == 'buy':
                return current_price <= trailing_stop
            else:
                return current_price >= trailing_stop
        except:
            return False
    
    def _check_time_based_exit(self, position: Dict[str, Any]) -> bool:
        """Zaman bazlı çıkış kontrolü"""
        try:
            max_hours = getattr(self.limits, 'max_position_hours', 24.0)
            time_stop_hours = getattr(self.limits, 'time_stop_hours', 0.0)

            position_age = datetime.now() - position.get('created_at', datetime.now())

            max_duration = timedelta(hours=max_hours)
            if position_age > max_duration:
                return True

            if time_stop_hours and time_stop_hours > 0:
                idle_duration = timedelta(hours=time_stop_hours)
                if position_age > idle_duration:
                    entry_price = float(position.get('entry_price', 0.0) or 0.0)
                    current_price = float(position.get('current_price', 0.0) or 0.0)
                    side = position.get('side', 'buy')

                    if entry_price > 0 and current_price > 0:
                        if side == 'buy':
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price

                        if pnl_pct < 0.0 or abs(pnl_pct) < 0.005:
                            return True

            return False
        except:
            return False
    
    def _check_correlation(self, symbol: str, side: str) -> bool:
        """Korelasyon kontrolü"""
        try:
            # Basit korelasyon kontrolü
            # Gerçek uygulamada daha gelişmiş korelasyon analizi yapılabilir
            open_positions = [p for p in self.positions.values() if p['status'] == 'open']
            
            # Aynı sembol için zaten pozisyon var mı?
            for pos in open_positions:
                if pos['symbol'] == symbol:
                    return False
            
            return True
        except:
            return True
    
    def _check_volatility(self, symbol: str, price: float) -> bool:
        """Volatilite kontrolü"""
        try:
            # Basit volatilite kontrolü
            # Gerçek uygulamada ATR veya diğer volatilite göstergeleri kullanılabilir
            return True
        except:
            return True
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Risk metriklerini al"""
        try:
            with self.lock:
                open_positions = len([p for p in self.positions.values() if p['status'] == 'open'])
                total_exposure = sum(p['size'] * p['current_price'] for p in self.positions.values() if p['status'] == 'open')
                
                return {
                    'current_balance': self.current_balance,
                    'available_balance': self.available_balance,
                    'used_balance': self.used_balance,
                    'total_pnl': self.total_pnl,
                    'daily_pnl': self.daily_pnl,
                    'weekly_pnl': self.weekly_pnl,
                    'monthly_pnl': self.monthly_pnl,
                    'current_drawdown': self.current_drawdown,
                    'max_drawdown': self.max_drawdown,
                    'risk_level': self.risk_level.value,
                    'trading_enabled': self.trading_enabled,
                    'emergency_stop': self.emergency_stop,
                    'open_positions': open_positions,
                    'total_exposure': total_exposure,
                    'exposure_ratio': total_exposure / self.current_balance if self.current_balance > 0 else 0
                }
        except Exception as e:
            self.logger.error(f"Risk metrikleri alma hatası: {e}")
            return {}
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Pozisyon özetini al"""
        try:
            with self.lock:
                open_positions = [p for p in self.positions.values() if p['status'] == 'open']
                closed_positions = [p for p in self.positions.values() if p['status'] == 'closed']
                
                total_unrealized_pnl = sum(p['unrealized_pnl'] for p in open_positions)
                total_realized_pnl = sum(p['realized_pnl'] for p in closed_positions)
                
                return {
                    'open_positions': len(open_positions),
                    'closed_positions': len(closed_positions),
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'total_realized_pnl': total_realized_pnl,
                    'positions': {
                        'open': open_positions,
                        'closed': closed_positions
                    }
                }
        except Exception as e:
            self.logger.error(f"Pozisyon özeti alma hatası: {e}")
            return {}
    
    def add_risk_alert_callback(self, callback):
        """Risk uyarı callback'i ekle"""
        self.risk_alert_callbacks.append(callback)
    
    def add_position_callback(self, callback):
        """Pozisyon callback'i ekle"""
        self.position_callbacks.append(callback)
    
    def _notify_risk_alert(self, alert_type: str, message: str):
        """Risk uyarısı gönder"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'risk_level': self.risk_level.value
        }
        
        for callback in self.risk_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Risk uyarı callback hatası: {e}")
    
    def _notify_position_callbacks(self, event_type: str, position: Dict[str, Any]):
        """Pozisyon callback'lerini çağır"""
        event = {
            'type': event_type,
            'position': position,
            'timestamp': datetime.now()
        }
        
        for callback in self.position_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Pozisyon callback hatası: {e}")
    
    def update_limits(self, new_limits: RiskLimits):
        """Risk limitlerini güncelle"""
        try:
            with self.lock:
                self.limits = new_limits
                self.logger.info("Risk limitleri güncellendi")
        except Exception as e:
            self.logger.error(f"Risk limitleri güncelleme hatası: {e}")
    
    def reset_emergency_stop(self):
        """Acil durdurmayı sıfırla"""
        try:
            with self.lock:
                self.emergency_stop = False
                self.trading_enabled = True
                self.logger.info("Acil durdurma sıfırlandı")
        except Exception as e:
            self.logger.error(f"Acil durdurma sıfırlama hatası: {e}")

# Global risk yöneticisi
risk_manager = RiskManager()

