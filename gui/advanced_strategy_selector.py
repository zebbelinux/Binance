"""
Gelişmiş Strateji Seçici GUI Modülü
Gelişmiş strateji değiştirme, performans analizi ve gerçek zamanlı durum takibi
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import threading
import time
import json

class AdvancedStrategySelector:
    """Gelişmiş strateji seçici sınıfı"""
    
    def __init__(self, parent_frame: ttk.Frame, strategy_manager, on_strategy_changed: Callable = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parent_frame = parent_frame
        self.strategy_manager = strategy_manager
        self.on_strategy_changed = on_strategy_changed
        
        # GUI bileşenleri
        self.strategy_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Hazır")
        self.performance_data = {}
        self.update_thread = None
        self.is_running = False
        
        self._create_widgets()
        self._start_performance_monitoring()
        
        self.logger.info("AdvancedStrategySelector başlatıldı")
    
    def _create_widgets(self):
        """GUI bileşenlerini oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.parent_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = ttk.Label(main_frame, text="Gelişmiş Strateji Yönetimi", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Üst panel - Strateji seçimi ve durum
        top_frame = ttk.LabelFrame(main_frame, text="Strateji Seçimi", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Strateji seçimi
        strategy_select_frame = ttk.Frame(top_frame)
        strategy_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(strategy_select_frame, text="Aktif Strateji:").pack(side=tk.LEFT, padx=(0, 10))
        self.strategy_combo = ttk.Combobox(
            strategy_select_frame,
            textvariable=self.strategy_var,
            state="readonly",
            width=25
        )
        self.strategy_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_selected)
        
        # Durum göstergesi
        self.status_label = ttk.Label(strategy_select_frame, text="Durum: Hazır")
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Kontrol butonları
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(fill=tk.X)
        
        # Aktif et butonu
        self.activate_btn = ttk.Button(control_frame, text="Aktif Et", command=self._activate_strategy)
        self.activate_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Durdur butonu
        self.stop_btn = ttk.Button(control_frame, text="Durdur", command=self._stop_strategy)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Yenile butonu
        self.refresh_btn = ttk.Button(control_frame, text="Yenile", command=self._refresh_strategies)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Ayarlar butonu
        self.settings_btn = ttk.Button(control_frame, text="Ayarlar", command=self._open_settings)
        self.settings_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Orta panel - Performans analizi
        performance_frame = ttk.LabelFrame(main_frame, text="Performans Analizi", padding=10)
        performance_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Performans metrikleri
        metrics_frame = ttk.Frame(performance_frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sol kolon - Temel metrikler
        left_metrics = ttk.Frame(metrics_frame)
        left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(left_metrics, text="Temel Metrikler", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.total_trades_label = ttk.Label(left_metrics, text="Toplam İşlem: 0")
        self.total_trades_label.pack(anchor=tk.W, pady=2)
        
        self.win_rate_label = ttk.Label(left_metrics, text="Kazanma Oranı: 0%")
        self.win_rate_label.pack(anchor=tk.W, pady=2)
        
        self.total_pnl_label = ttk.Label(left_metrics, text="Toplam P&L: 0.00 TL")
        self.total_pnl_label.pack(anchor=tk.W, pady=2)
        
        self.avg_trade_label = ttk.Label(left_metrics, text="Ortalama İşlem: 0.00 TL")
        self.avg_trade_label.pack(anchor=tk.W, pady=2)
        
        # Sağ kolon - Risk metrikleri
        right_metrics = ttk.Frame(metrics_frame)
        right_metrics.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_metrics, text="Risk Metrikleri", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.max_drawdown_label = ttk.Label(right_metrics, text="Max Drawdown: 0%")
        self.max_drawdown_label.pack(anchor=tk.W, pady=2)
        
        self.sharpe_ratio_label = ttk.Label(right_metrics, text="Sharpe Ratio: 0.00")
        self.sharpe_ratio_label.pack(anchor=tk.W, pady=2)
        
        self.volatility_label = ttk.Label(right_metrics, text="Volatilite: 0%")
        self.volatility_label.pack(anchor=tk.W, pady=2)
        
        self.var_95_label = ttk.Label(right_metrics, text="VaR (95%): 0.00 TL")
        self.var_95_label.pack(anchor=tk.W, pady=2)
        
        # Alt panel - Strateji detayları
        details_frame = ttk.LabelFrame(main_frame, text="Strateji Detayları", padding=10)
        details_frame.pack(fill=tk.X)
        
        # Strateji bilgileri
        self.strategy_info_text = tk.Text(details_frame, height=6, width=80)
        self.strategy_info_text.pack(fill=tk.X)
        
        # Durum çubuğu
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # İlk yükleme
        self._load_strategies()
    
    def _load_strategies(self):
        """Stratejileri yükle"""
        try:
            strategy_names = list(self.strategy_manager.get_available_strategies() or [])
            
            self.strategy_combo['values'] = strategy_names
            
            if strategy_names:
                # Aktif stratejiyi bul
                active_strategy = None
                for name in strategy_names:
                    try:
                        info = self.strategy_manager.get_strategy_info(name) or {}
                        if info.get('is_active', False):
                            active_strategy = name
                            break
                    except Exception:
                        continue
                
                if active_strategy:
                    self.strategy_var.set(active_strategy)
                else:
                    self.strategy_var.set(strategy_names[0])
                
                self._on_strategy_selected()
            
            self._update_status(f"Stratejiler yüklendi: {len(strategy_names)} adet")
            
        except Exception as e:
            self.logger.error(f"Strateji yükleme hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _on_strategy_selected(self, event=None):
        """Strateji seçildiğinde"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                return
            
            # Strateji bilgilerini al
            strategy_info = self.strategy_manager.get_strategy_info(strategy_name)
            if not strategy_info:
                self._update_status(f"Strateji bulunamadı: {strategy_name}")
                return
            
            # Strateji detaylarını göster
            self._display_strategy_details(strategy_name, strategy_info)
            
            # Performans verilerini güncelle
            self._update_performance_display(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Strateji seçim hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _display_strategy_details(self, strategy_name: str, strategy_info: Dict[str, Any]):
        """Strateji detaylarını göster"""
        try:
            details = f"Strateji: {strategy_name}\n"
            details += f"Tip: {strategy_info.get('type', 'Bilinmiyor')}\n"
            details += f"Durum: {'Aktif' if strategy_info.get('is_active', False) else 'Pasif'}\n"
            details += f"Oluşturulma: {strategy_info.get('created_at', 'Bilinmiyor')}\n"
            details += f"Son Güncelleme: {strategy_info.get('updated_at', 'Bilinmiyor')}\n"
            
            # Konfigürasyon
            config = strategy_info.get('config', {})
            if config:
                details += "\nKonfigürasyon:\n"
                for key, value in config.items():
                    details += f"  {key}: {value}\n"
            
            # Performans
            performance = strategy_info.get('performance', {})
            if performance:
                details += "\nPerformans:\n"
                for key, value in performance.items():
                    details += f"  {key}: {value}\n"
            
            self.strategy_info_text.delete(1.0, tk.END)
            self.strategy_info_text.insert(1.0, details)
            
        except Exception as e:
            self.logger.error(f"Strateji detay gösterme hatası: {e}")
    
    def _update_performance_display(self, strategy_name: str):
        """Performans görünümünü güncelle"""
        try:
            # Performans verilerini al (gerçek implementasyonda veritabanından)
            performance = self.performance_data.get(strategy_name, {})
            
            # Temel metrikler
            total_trades = performance.get('total_trades', 0)
            win_rate = performance.get('win_rate', 0.0)
            total_pnl = performance.get('total_pnl', 0.0)
            avg_trade = performance.get('avg_trade', 0.0)
            
            self.total_trades_label.config(text=f"Toplam İşlem: {total_trades}")
            self.win_rate_label.config(text=f"Kazanma Oranı: {win_rate:.1f}%")
            self.total_pnl_label.config(text=f"Toplam P&L: {total_pnl:.2f} TL")
            self.avg_trade_label.config(text=f"Ortalama İşlem: {avg_trade:.2f} TL")
            
            # Risk metrikleri
            max_drawdown = performance.get('max_drawdown', 0.0)
            sharpe_ratio = performance.get('sharpe_ratio', 0.0)
            volatility = performance.get('volatility', 0.0)
            var_95 = performance.get('var_95', 0.0)
            
            self.max_drawdown_label.config(text=f"Max Drawdown: {max_drawdown:.1f}%")
            self.sharpe_ratio_label.config(text=f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.volatility_label.config(text=f"Volatilite: {volatility:.1f}%")
            self.var_95_label.config(text=f"VaR (95%): {var_95:.2f} TL")
            
        except Exception as e:
            self.logger.error(f"Performans güncelleme hatası: {e}")
    
    def _activate_strategy(self):
        """Stratejiyi aktif et"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            # Mevcut aktif stratejiyi durdur
            active_strategies = list(self.strategy_manager.get_active_strategies() or [])
            for active_name in active_strategies:
                if active_name != strategy_name:
                    self.strategy_manager.deactivate_strategy(active_name)
            
            # Yeni stratejiyi aktif et
            success = self.strategy_manager.activate_strategy(strategy_name)
            
            if success:
                self._update_status(f"Strateji aktif edildi: {strategy_name}")
                messagebox.showinfo("Başarılı", f"Strateji '{strategy_name}' aktif edildi")
                
                # Callback çağır
                if self.on_strategy_changed:
                    self.on_strategy_changed(strategy_name, 'activated')
            else:
                messagebox.showerror("Hata", f"Strateji '{strategy_name}' aktif edilemedi")
                
        except Exception as e:
            self.logger.error(f"Strateji aktif etme hatası: {e}")
            messagebox.showerror("Hata", f"Strateji aktif edilemedi: {e}")
    
    def _stop_strategy(self):
        """Stratejiyi durdur"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            success = self.strategy_manager.deactivate_strategy(strategy_name)
            
            if success:
                self._update_status(f"Strateji durduruldu: {strategy_name}")
                messagebox.showinfo("Başarılı", f"Strateji '{strategy_name}' durduruldu")
                
                # Callback çağır
                if self.on_strategy_changed:
                    self.on_strategy_changed(strategy_name, 'deactivated')
            else:
                messagebox.showerror("Hata", f"Strateji '{strategy_name}' durdurulamadı")
                
        except Exception as e:
            self.logger.error(f"Strateji durdurma hatası: {e}")
            messagebox.showerror("Hata", f"Strateji durdurulamadı: {e}")
    
    def _refresh_strategies(self):
        """Stratejileri yenile"""
        try:
            self._load_strategies()
            self._update_status("Stratejiler yenilendi")
            
        except Exception as e:
            self.logger.error(f"Strateji yenileme hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _open_settings(self):
        """Strateji ayarlarını aç"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            # Strateji ayarları penceresini aç
            settings_window = tk.Toplevel(self.parent_frame.winfo_toplevel())
            settings_window.title(f"Strateji Ayarları - {strategy_name}")
            settings_window.geometry("800x600")
            settings_window.resizable(True, True)
            
            # Ana frame
            main_frame = ttk.Frame(settings_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Strateji ayarları GUI'sini oluştur
            from gui.strategy_settings_gui import StrategySettingsGUI
            strategy_settings = StrategySettingsGUI(main_frame, self.strategy_manager)
            
            # Seçili stratejiyi ayarla
            strategy_settings.strategy_var.set(strategy_name)
            strategy_settings._load_strategy_config()
            
            self.logger.info(f"Strateji ayarları açıldı: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Strateji ayarları açma hatası: {e}")
            messagebox.showerror("Hata", f"Strateji ayarları açılamadı: {e}")
    
    def _start_performance_monitoring(self):
        """Performans izleme thread'ini başlat"""
        try:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
            self.update_thread.start()
            
        except Exception as e:
            self.logger.error(f"Performans izleme başlatma hatası: {e}")
    
    def _performance_monitoring_loop(self):
        """Performans izleme döngüsü"""
        while self.is_running:
            try:
                # Performans verilerini güncelle
                self._update_performance_data()
                
                # 5 saniye bekle
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Performans izleme hatası: {e}")
                time.sleep(5)
    
    def _update_performance_data(self):
        """Performans verilerini güncelle"""
        try:
            # Gerçek implementasyonda veritabanından performans verilerini al
            # Şimdilik örnek veriler
            strategy_names = list(self.strategy_manager.get_available_strategies() or [])
            for strategy_name in strategy_names:
                if strategy_name not in self.performance_data:
                    self.performance_data[strategy_name] = {
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'avg_trade': 0.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0,
                        'volatility': 0.0,
                        'var_95': 0.0
                    }
            
            # Seçili stratejiyi güncelle
            current_strategy = self.strategy_var.get()
            if current_strategy:
                self._update_performance_display(current_strategy)
                
        except Exception as e:
            self.logger.error(f"Performans veri güncelleme hatası: {e}")
    
    def _update_status(self, message: str):
        """Durum çubuğunu güncelle"""
        self.status_var.set(message)
        self.logger.info(message)
    
    def stop(self):
        """Performans izlemeyi durdur"""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
    
    def get_current_strategy(self) -> Optional[str]:
        """Mevcut stratejiyi al"""
        return self.strategy_var.get()
    
    def set_strategy(self, strategy_name: str):
        """Stratejiyi ayarla"""
        self.strategy_var.set(strategy_name)
        self._on_strategy_selected()
