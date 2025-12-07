"""
Strateji Ayarları GUI Modülü
Strateji parametrelerini düzenleme ve yapılandırma ekranı
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
import json
import threading
import time

class StrategySettingsGUI:
    """Strateji ayarları GUI sınıfı"""
    
    def __init__(self, parent_frame: ttk.Frame, strategy_manager, on_settings_changed: Callable = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parent_frame = parent_frame
        self.strategy_manager = strategy_manager
        self.on_settings_changed = on_settings_changed
        
        # Mevcut strateji
        self.current_strategy = None
        self.strategy_configs = {}
        
        # GUI bileşenleri
        self.strategy_var = tk.StringVar()
        self.parameter_frames = {}
        self.parameter_vars = {}
        
        self._create_widgets()
        self._load_strategies()
        
        self.logger.info("StrategySettingsGUI başlatıldı")
    
    def _create_widgets(self):
        """GUI bileşenlerini oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.parent_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = ttk.Label(main_frame, text="Strateji Ayarları", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Strateji seçimi frame
        strategy_frame = ttk.LabelFrame(main_frame, text="Strateji Seçimi", padding=10)
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Strateji dropdown
        ttk.Label(strategy_frame, text="Aktif Strateji:").pack(side=tk.LEFT, padx=(0, 10))
        self.strategy_combo = ttk.Combobox(
            strategy_frame, 
            textvariable=self.strategy_var,
            state="readonly",
            width=20
        )
        self.strategy_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_selected)
        
        # Strateji yükle butonu
        load_btn = ttk.Button(strategy_frame, text="Yükle", command=self._load_strategy_config)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Kaydet butonu
        save_btn = ttk.Button(strategy_frame, text="Kaydet", command=self._save_strategy_config)
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Sıfırla butonu
        reset_btn = ttk.Button(strategy_frame, text="Sıfırla", command=self._reset_strategy_config)
        reset_btn.pack(side=tk.LEFT)
        
        # Parametreler frame
        self.parameters_frame = ttk.LabelFrame(main_frame, text="Strateji Parametreleri", padding=10)
        self.parameters_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame için canvas
        canvas = tk.Canvas(self.parameters_frame)
        scrollbar = ttk.Scrollbar(self.parameters_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Durum çubuğu
        self.status_var = tk.StringVar(value="Hazır")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _load_strategies(self):
        """Mevcut stratejileri yükle"""
        try:
            strategy_names = list(self.strategy_manager.get_available_strategies() or [])
            
            self.strategy_combo['values'] = strategy_names
            
            if strategy_names:
                self.strategy_var.set(strategy_names[0])
                self._load_strategy_config()
            
            self.logger.info(f"Stratejiler yüklendi: {strategy_names}")
            
        except Exception as e:
            self.logger.error(f"Strateji yükleme hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _on_strategy_selected(self, event=None):
        """Strateji seçildiğinde"""
        self._load_strategy_config()
    
    def _load_strategy_config(self):
        """Seçili stratejinin konfigürasyonunu yükle"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                return
            
            # Strateji bilgilerini al
            strategy_info = self.strategy_manager.get_strategy_info(strategy_name)
            if not strategy_info:
                self._update_status(f"Strateji bulunamadı: {strategy_name}")
                return
            
            self.current_strategy = strategy_name
            
            # Mevcut parametreleri temizle
            self._clear_parameter_widgets()
            
            # Strateji tipine göre parametreleri oluştur
            self._create_parameter_widgets(strategy_name, strategy_info)
            
            self._update_status(f"Strateji yüklendi: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Strateji konfigürasyon yükleme hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _create_parameter_widgets(self, strategy_name: str, strategy_info: Dict[str, Any]):
        """Strateji parametreleri için widget'ları oluştur"""
        try:
            # Strateji tipine göre parametreler
            strategy_type = strategy_info.get('type', 'unknown')
            
            if strategy_type == 'scalping':
                self._create_scalping_parameters()
            elif strategy_type == 'grid':
                self._create_grid_parameters()
            elif strategy_type == 'trend_following':
                self._create_trend_parameters()
            elif strategy_type == 'dca':
                self._create_dca_parameters()
            elif strategy_type == 'hedge':
                self._create_hedge_parameters()
            else:
                self._create_generic_parameters(strategy_info)
                
        except Exception as e:
            self.logger.error(f"Parametre widget oluşturma hatası: {e}")
            self._update_status(f"Hata: {e}")
    
    def _create_scalping_parameters(self):
        """Scalping stratejisi parametreleri"""
        params = {
            'profit_target': {'label': 'Kar Hedefi (%)', 'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 5.0},
            'stop_loss': {'label': 'Stop Loss (%)', 'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 2.0},
            'min_volume': {'label': 'Min Hacim', 'type': 'int', 'default': 1000000, 'min': 100000, 'max': 10000000},
            'max_spread': {'label': 'Max Spread (%)', 'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'rsi_period': {'label': 'RSI Periyodu', 'type': 'int', 'default': 14, 'min': 5, 'max': 50},
            'rsi_oversold': {'label': 'RSI Aşırı Satım', 'type': 'int', 'default': 30, 'min': 10, 'max': 40},
            'rsi_overbought': {'label': 'RSI Aşırı Alım', 'type': 'int', 'default': 70, 'min': 60, 'max': 90},
            'ma_period': {'label': 'MA Periyodu', 'type': 'int', 'default': 20, 'min': 5, 'max': 100},
            'min_signal_strength': {'label': 'Min Sinyal Gücü', 'type': 'float', 'default': 0.6, 'min': 0.1, 'max': 1.0},
            'max_position_size': {'label': 'Max Pozisyon (%)', 'type': 'float', 'default': 10.0, 'min': 1.0, 'max': 50.0}
        }
        
        self._create_parameter_inputs(params)
    
    def _create_grid_parameters(self):
        """Grid stratejisi parametreleri"""
        params = {
            'grid_levels': {'label': 'Grid Seviyeleri', 'type': 'int', 'default': 10, 'min': 5, 'max': 50},
            'grid_spacing': {'label': 'Grid Aralığı (%)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 5.0},
            'grid_profit_target': {'label': 'Grid Kar Hedefi (%)', 'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 10.0},
            'max_position_size': {'label': 'Max Pozisyon (%)', 'type': 'float', 'default': 20.0, 'min': 5.0, 'max': 50.0},
            'max_grid_loss': {'label': 'Max Grid Kaybı (%)', 'type': 'float', 'default': 5.0, 'min': 1.0, 'max': 20.0},
            'grid_stop_loss': {'label': 'Grid Stop Loss (%)', 'type': 'float', 'default': 10.0, 'min': 2.0, 'max': 30.0},
            'rebalance_threshold': {'label': 'Rebalance Eşiği (%)', 'type': 'float', 'default': 5.0, 'min': 1.0, 'max': 15.0},
            'min_volatility': {'label': 'Min Volatilite (%)', 'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'max_volatility': {'label': 'Max Volatilite (%)', 'type': 'float', 'default': 5.0, 'min': 1.0, 'max': 20.0},
            'sideways_threshold': {'label': 'Yatay Piyasa Eşiği (%)', 'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 10.0}
        }
        
        self._create_parameter_inputs(params)
    
    def _create_trend_parameters(self):
        """Trend Following stratejisi parametreleri"""
        params = {
            'fast_ma': {'label': 'Hızlı MA', 'type': 'int', 'default': 20, 'min': 5, 'max': 50},
            'slow_ma': {'label': 'Yavaş MA', 'type': 'int', 'default': 50, 'min': 20, 'max': 200},
            'adx_threshold': {'label': 'ADX Eşiği', 'type': 'float', 'default': 25.0, 'min': 15.0, 'max': 50.0},
            'atr_stop_multiplier': {'label': 'ATR Stop Çarpanı', 'type': 'float', 'default': 2.5, 'min': 1.0, 'max': 5.0},
            'atr_tp_multiplier': {'label': 'ATR TP Çarpanı', 'type': 'float', 'default': 2.5, 'min': 1.0, 'max': 5.0},
            'min_signal_strength': {'label': 'Min Sinyal Gücü', 'type': 'float', 'default': 0.55, 'min': 0.1, 'max': 1.0},
            'max_position_size': {'label': 'Max Pozisyon (%)', 'type': 'float', 'default': 10.0, 'min': 1.0, 'max': 30.0}
        }
        
        self._create_parameter_inputs(params)
    
    def _create_dca_parameters(self):
        """DCA stratejisi parametreleri"""
        params = {
            'dca_step_pct': {'label': 'DCA Adım (%)', 'type': 'float', 'default': 5.0, 'min': 1.0, 'max': 20.0},
            'dca_max_steps': {'label': 'Max DCA Adımları', 'type': 'int', 'default': 6, 'min': 3, 'max': 20},
            'dca_budget_fraction': {'label': 'DCA Bütçe (%)', 'type': 'float', 'default': 10.0, 'min': 5.0, 'max': 50.0},
            'per_step_fraction': {'label': 'Adım Başına (%)', 'type': 'float', 'default': 1.5, 'min': 0.5, 'max': 10.0}
        }
        
        self._create_parameter_inputs(params)
    
    def _create_hedge_parameters(self):
        """Hedge stratejisi parametreleri"""
        params = {
            'correlation_threshold': {'label': 'Korelasyon Eşiği', 'type': 'float', 'default': 0.7, 'min': 0.3, 'max': 0.95},
            'min_correlation_period': {'label': 'Min Korelasyon Periyodu', 'type': 'int', 'default': 20, 'min': 10, 'max': 100},
            'max_correlation_period': {'label': 'Max Korelasyon Periyodu', 'type': 'int', 'default': 100, 'min': 50, 'max': 500},
            'z_score_threshold': {'label': 'Z-Score Eşiği', 'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 4.0},
            'mean_reversion_threshold': {'label': 'Mean Reversion Eşiği', 'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0},
            'max_hedge_ratio': {'label': 'Max Hedge Oranı', 'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 1.0},
            'stop_loss_threshold': {'label': 'Stop Loss Eşiği', 'type': 'float', 'default': 0.05, 'min': 0.01, 'max': 0.2},
            'take_profit_threshold': {'label': 'Take Profit Eşiği', 'type': 'float', 'default': 0.03, 'min': 0.01, 'max': 0.15}
        }
        
        self._create_parameter_inputs(params)
    
    def _create_generic_parameters(self, strategy_info: Dict[str, Any]):
        """Genel parametreler"""
        # Strateji bilgilerinden parametreleri çıkar
        config = strategy_info.get('config', {})
        
        for param_name, param_value in config.items():
            if isinstance(param_value, (int, float)):
                param_type = 'float' if isinstance(param_value, float) else 'int'
                self._create_single_parameter(param_name, param_name.replace('_', ' ').title(), param_type, param_value)
    
    def _create_parameter_inputs(self, params: Dict[str, Dict[str, Any]]):
        """Parametre giriş alanlarını oluştur"""
        for param_name, param_config in params.items():
            self._create_single_parameter(
                param_name,
                param_config['label'],
                param_config['type'],
                param_config['default'],
                param_config.get('min'),
                param_config.get('max')
            )
    
    def _create_single_parameter(self, param_name: str, label: str, param_type: str, default_value: Any, min_val: Any = None, max_val: Any = None):
        """Tek parametre girişi oluştur"""
        # Parametre frame
        param_frame = ttk.Frame(self.scrollable_frame)
        param_frame.pack(fill=tk.X, pady=2)
        
        # Label
        label_widget = ttk.Label(param_frame, text=label, width=25, anchor=tk.W)
        label_widget.pack(side=tk.LEFT, padx=(0, 10))
        
        # Input widget
        if param_type == 'int':
            var = tk.IntVar(value=default_value)
            widget = ttk.Spinbox(
                param_frame,
                textvariable=var,
                from_=min_val or 0,
                to=max_val or 1000,
                width=15
            )
        elif param_type == 'float':
            var = tk.DoubleVar(value=default_value)
            widget = ttk.Spinbox(
                param_frame,
                textvariable=var,
                from_=min_val or 0.0,
                to=max_val or 100.0,
                increment=0.1,
                width=15
            )
        else:
            var = tk.StringVar(value=str(default_value))
            widget = ttk.Entry(param_frame, textvariable=var, width=15)
        
        widget.pack(side=tk.LEFT, padx=(0, 10))
        
        # Değer gösterici
        value_label = ttk.Label(param_frame, text=str(default_value), width=10)
        value_label.pack(side=tk.LEFT)
        
        # Değişiklik callback'i
        def update_value(*args):
            try:
                if param_type == 'int':
                    value = var.get()
                elif param_type == 'float':
                    value = var.get()
                else:
                    value = var.get()
                
                value_label.config(text=str(value))
            except:
                pass
        
        var.trace('w', update_value)
        
        # Kaydet
        self.parameter_vars[param_name] = var
        self.parameter_frames[param_name] = param_frame
    
    def _clear_parameter_widgets(self):
        """Parametre widget'larını temizle"""
        for frame in self.parameter_frames.values():
            frame.destroy()
        
        self.parameter_frames.clear()
        self.parameter_vars.clear()
    
    def _save_strategy_config(self):
        """Strateji konfigürasyonunu kaydet"""
        try:
            if not self.current_strategy:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            # Parametreleri topla
            new_config = {}
            for param_name, var in self.parameter_vars.items():
                try:
                    if isinstance(var, tk.IntVar):
                        new_config[param_name] = var.get()
                    elif isinstance(var, tk.DoubleVar):
                        new_config[param_name] = var.get()
                    else:
                        new_config[param_name] = var.get()
                except:
                    self.logger.warning(f"Parametre dönüştürme hatası: {param_name}")
                    continue
            
            # Strateji konfigürasyonunu güncelle
            success = self.strategy_manager.update_strategy_config(self.current_strategy, new_config)
            
            if success:
                self._update_status(f"Strateji kaydedildi: {self.current_strategy}")
                messagebox.showinfo("Başarılı", f"Strateji '{self.current_strategy}' kaydedildi")
                
                # Callback çağır
                if self.on_settings_changed:
                    self.on_settings_changed(self.current_strategy, new_config)
            else:
                messagebox.showerror("Hata", "Strateji kaydedilemedi")
                
        except Exception as e:
            self.logger.error(f"Strateji kaydetme hatası: {e}")
            messagebox.showerror("Hata", f"Kaydetme hatası: {e}")
    
    def _reset_strategy_config(self):
        """Strateji konfigürasyonunu sıfırla"""
        try:
            if not self.current_strategy:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            # Varsayılan konfigürasyonu yükle
            self._load_strategy_config()
            self._update_status(f"Strateji sıfırlandı: {self.current_strategy}")
            
        except Exception as e:
            self.logger.error(f"Strateji sıfırlama hatası: {e}")
            messagebox.showerror("Hata", f"Sıfırlama hatası: {e}")
    
    def _update_status(self, message: str):
        """Durum çubuğunu güncelle"""
        self.status_var.set(message)
        self.logger.info(message)
    
    def refresh_strategies(self):
        """Stratejileri yenile"""
        self._load_strategies()
    
    def get_current_strategy(self) -> Optional[str]:
        """Mevcut stratejiyi al"""
        return self.current_strategy
    
    def get_current_config(self) -> Dict[str, Any]:
        """Mevcut konfigürasyonu al"""
        config = {}
        for param_name, var in self.parameter_vars.items():
            try:
                if isinstance(var, tk.IntVar):
                    config[param_name] = var.get()
                elif isinstance(var, tk.DoubleVar):
                    config[param_name] = var.get()
                else:
                    config[param_name] = var.get()
            except:
                continue
        return config
