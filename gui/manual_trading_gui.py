"""
Manuel Trading GUI Modülü
GUI üzerinden manuel işlem yapma ve pozisyon yönetimi
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os
from pathlib import Path
import pickle

@dataclass
class ManualOrder:
    """Manuel emir"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: float = None
    stop_price: float = None
    status: str = 'pending'  # 'pending', 'filled', 'cancelled', 'rejected'
    timestamp: datetime = None
    filled_price: float = None
    filled_quantity: float = 0.0
    commission: float = 0.0

@dataclass
class Position:
    """Pozisyon"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: float = None
    take_profit: float = None

@dataclass
class AccountInfo:
    """Hesap bilgisi"""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    leverage: float
    free_margin: float
    margin_level: float

class OrderType(Enum):
    """Emir türleri"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class ManualTradingGUI:
    """Manuel Trading GUI sınıfı"""
    
    def __init__(self, parent=None, api_client=None):
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.api_client = api_client
        self.root = None
        
        # Trading verileri
        self.current_symbol = "BTCUSDT"
        self.current_price = 50000.0
        self.account_info = AccountInfo(
            balance=100000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=100000.0,
            leverage=1.0,
            free_margin=100000.0,
            margin_level=0.0
        )
        
        # Emirler ve pozisyonlar
        self.orders = {}  # {order_id: ManualOrder}
        self.positions = {}  # {symbol: Position}
        self.order_history = deque(maxlen=1000)
        
        # GUI elemanları
        self.main_frame = None
        self.trading_frame = None
        self.orders_frame = None
        self.positions_frame = None
        self.account_frame = None
        
        # Trading form elemanları
        self.symbol_var = None
        self.side_var = None
        self.order_type_var = None
        self.quantity_var = None
        self.price_var = None
        self.stop_price_var = None
        
        # Tablolar
        self.orders_tree = None
        self.positions_tree = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback'ler
        self.order_callbacks = []
        self.position_callbacks = []
        
        self.logger.info("Manuel Trading GUI başlatıldı")
    
    def create_gui(self):
        """GUI oluştur"""
        try:
            if self.parent:
                self.root = self.parent
            else:
                self.root = tk.Tk()
                self.root.title("Manuel Trading")
                self.root.geometry("1400x900")
            
            # Ana frame
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Notebook (tabbed interface)
            notebook = ttk.Notebook(self.main_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # Trading tab
            self.trading_frame = ttk.Frame(notebook)
            notebook.add(self.trading_frame, text="Trading")
            self.create_trading_tab()
            
            # Orders tab
            self.orders_frame = ttk.Frame(notebook)
            notebook.add(self.orders_frame, text="Emirler")
            self.create_orders_tab()
            
            # Positions tab
            self.positions_frame = ttk.Frame(notebook)
            notebook.add(self.positions_frame, text="Pozisyonlar")
            self.create_positions_tab()
            
            # Account tab
            self.account_frame = ttk.Frame(notebook)
            notebook.add(self.account_frame, text="Hesap")
            self.create_account_tab()
            
            # Başlangıç verilerini yükle
            self.load_sample_data()
            
            # Güncelleme döngüsünü başlat
            self.start_update_loop()
            
            self.logger.info("Manuel Trading GUI oluşturuldu")
            
        except Exception as e:
            self.logger.error(f"GUI oluşturma hatası: {e}")
    
    def create_trading_tab(self):
        """Trading tab'ını oluştur"""
        try:
            # Sol panel - Emir formu
            left_panel = ttk.LabelFrame(self.trading_frame, text="Emir Formu", padding=10)
            left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            # Sembol seçimi
            symbol_frame = ttk.Frame(left_panel)
            symbol_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(symbol_frame, text="Sembol:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.symbol_var = tk.StringVar(value=self.current_symbol)
            symbol_combo = ttk.Combobox(
                symbol_frame,
                textvariable=self.symbol_var,
                values=["BTCUSDT", "ETHUSDT", "AVAXUSDT", "ADAUSDT", "SOLUSDT"],
                state="readonly",
                width=15
            )
            symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
            symbol_combo.bind("<<ComboboxSelected>>", self.on_symbol_changed)
            
            # Mevcut fiyat
            price_frame = ttk.Frame(left_panel)
            price_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(price_frame, text="Mevcut Fiyat:").pack(side=tk.LEFT, padx=(0, 5))
            self.current_price_label = ttk.Label(price_frame, text=f"{self.current_price:.2f}", font=("Arial", 12, "bold"))
            self.current_price_label.pack(side=tk.LEFT)
            
            # Alış/Satış seçimi
            side_frame = ttk.Frame(left_panel)
            side_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(side_frame, text="Yön:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.side_var = tk.StringVar(value="buy")
            ttk.Radiobutton(side_frame, text="Alış", variable=self.side_var, value="buy").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(side_frame, text="Satış", variable=self.side_var, value="sell").pack(side=tk.LEFT)
            
            # Emir türü
            order_type_frame = ttk.Frame(left_panel)
            order_type_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(order_type_frame, text="Emir Türü:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.order_type_var = tk.StringVar(value="market")
            order_type_combo = ttk.Combobox(
                order_type_frame,
                textvariable=self.order_type_var,
                values=["market", "limit", "stop"],
                state="readonly",
                width=15
            )
            order_type_combo.pack(side=tk.LEFT, padx=(0, 10))
            order_type_combo.bind("<<ComboboxSelected>>", self.on_order_type_changed)
            
            # Miktar
            quantity_frame = ttk.Frame(left_panel)
            quantity_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(quantity_frame, text="Miktar:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.quantity_var = tk.StringVar(value="0.001")
            quantity_entry = ttk.Entry(quantity_frame, textvariable=self.quantity_var, width=15)
            quantity_entry.pack(side=tk.LEFT, padx=(0, 10))
            
            # Hızlı miktar butonları
            quick_quantity_frame = ttk.Frame(left_panel)
            quick_quantity_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(quick_quantity_frame, text="%25", command=lambda: self.set_quantity_percentage(0.25)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_quantity_frame, text="%50", command=lambda: self.set_quantity_percentage(0.50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_quantity_frame, text="%75", command=lambda: self.set_quantity_percentage(0.75)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_quantity_frame, text="%100", command=lambda: self.set_quantity_percentage(1.00)).pack(side=tk.LEFT, padx=(0, 5))
            
            # Fiyat (limit ve stop emirler için)
            price_frame = ttk.Frame(left_panel)
            price_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(price_frame, text="Fiyat:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.price_var = tk.StringVar()
            price_entry = ttk.Entry(price_frame, textvariable=self.price_var, width=15)
            price_entry.pack(side=tk.LEFT, padx=(0, 10))
            
            # Stop fiyat (stop emirler için)
            stop_price_frame = ttk.Frame(left_panel)
            stop_price_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(stop_price_frame, text="Stop Fiyat:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.stop_price_var = tk.StringVar()
            stop_price_entry = ttk.Entry(stop_price_frame, textvariable=self.stop_price_var, width=15)
            stop_price_entry.pack(side=tk.LEFT, padx=(0, 10))
            
            # Emir butonları
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(button_frame, text="Emir Gönder", command=self.submit_order, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Temizle", command=self.clear_form).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Fiyat Güncelle", command=self.update_price).pack(side=tk.LEFT)
            
            # Sağ panel - Grafik ve bilgiler
            right_panel = ttk.LabelFrame(self.trading_frame, text="Piyasa Bilgileri", padding=10)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Grafik
            chart_frame = ttk.Frame(right_panel)
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            
            self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Piyasa bilgileri
            info_frame = ttk.LabelFrame(right_panel, text="Piyasa Bilgileri", padding=5)
            info_frame.pack(fill=tk.X)
            
            self.market_info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
            self.market_info_text.pack(fill=tk.BOTH, expand=True)
            
            # Başlangıç grafiği
            self.update_chart()
            
        except Exception as e:
            self.logger.error(f"Trading tab oluşturma hatası: {e}")
    
    def create_orders_tab(self):
        """Orders tab'ını oluştur"""
        try:
            # Kontrol paneli
            control_frame = ttk.Frame(self.orders_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(control_frame, text="Yenile", command=self.refresh_orders).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Tümünü İptal", command=self.cancel_all_orders).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Export", command=self.export_orders).pack(side=tk.LEFT)
            
            # Orders tablosu
            orders_frame = ttk.LabelFrame(self.orders_frame, text="Aktif Emirler", padding=10)
            orders_frame.pack(fill=tk.BOTH, expand=True)
            
            columns = ("ID", "Sembol", "Yön", "Tür", "Miktar", "Fiyat", "Stop Fiyat", "Durum", "Tarih")
            self.orders_tree = ttk.Treeview(orders_frame, columns=columns, show="headings", height=15)
            
            for col in columns:
                self.orders_tree.heading(col, text=col)
                self.orders_tree.column(col, width=100, anchor="center")
            
            # Scrollbar
            orders_scrollbar = ttk.Scrollbar(orders_frame, orient=tk.VERTICAL, command=self.orders_tree.yview)
            self.orders_tree.configure(yscrollcommand=orders_scrollbar.set)
            
            self.orders_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            orders_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Context menu
            self.orders_tree.bind("<Button-3>", self.show_orders_context_menu)
            
        except Exception as e:
            self.logger.error(f"Orders tab oluşturma hatası: {e}")
    
    def create_positions_tab(self):
        """Positions tab'ını oluştur"""
        try:
            # Kontrol paneli
            control_frame = ttk.Frame(self.positions_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(control_frame, text="Yenile", command=self.refresh_positions).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Tümünü Kapat", command=self.close_all_positions).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Export", command=self.export_positions).pack(side=tk.LEFT)
            
            # Positions tablosu
            positions_frame = ttk.LabelFrame(self.positions_frame, text="Aktif Pozisyonlar", padding=10)
            positions_frame.pack(fill=tk.BOTH, expand=True)
            
            columns = ("Sembol", "Yön", "Miktar", "Giriş Fiyatı", "Mevcut Fiyat", "PnL", "Stop Loss", "Take Profit", "Tarih")
            self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings", height=15)
            
            for col in columns:
                self.positions_tree.heading(col, text=col)
                self.positions_tree.column(col, width=100, anchor="center")
            
            # Scrollbar
            positions_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
            self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
            
            self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Context menu
            self.positions_tree.bind("<Button-3>", self.show_positions_context_menu)
            
        except Exception as e:
            self.logger.error(f"Positions tab oluşturma hatası: {e}")
    
    def create_account_tab(self):
        """Account tab'ını oluştur"""
        try:
            # Hesap bilgileri
            account_info_frame = ttk.LabelFrame(self.account_frame, text="Hesap Bilgileri", padding=10)
            account_info_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Sol kolon
            left_col = ttk.Frame(account_info_frame)
            left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Label(left_col, text="Bakiye:").pack(anchor=tk.W)
            self.balance_label = ttk.Label(left_col, text=f"{self.account_info.balance:.2f}", font=("Arial", 12, "bold"))
            self.balance_label.pack(anchor=tk.W)
            
            ttk.Label(left_col, text="Özkaynak:").pack(anchor=tk.W)
            self.equity_label = ttk.Label(left_col, text=f"{self.account_info.equity:.2f}", font=("Arial", 12, "bold"))
            self.equity_label.pack(anchor=tk.W)
            
            ttk.Label(left_col, text="Kullanılan Margin:").pack(anchor=tk.W)
            self.margin_used_label = ttk.Label(left_col, text=f"{self.account_info.margin_used:.2f}", font=("Arial", 12, "bold"))
            self.margin_used_label.pack(anchor=tk.W)
            
            # Orta kolon
            middle_col = ttk.Frame(account_info_frame)
            middle_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Label(middle_col, text="Kullanılabilir Margin:").pack(anchor=tk.W)
            self.margin_available_label = ttk.Label(middle_col, text=f"{self.account_info.margin_available:.2f}", font=("Arial", 12, "bold"))
            self.margin_available_label.pack(anchor=tk.W)
            
            ttk.Label(middle_col, text="Kaldıraç:").pack(anchor=tk.W)
            self.leverage_label = ttk.Label(middle_col, text=f"{self.account_info.leverage:.1f}x", font=("Arial", 12, "bold"))
            self.leverage_label.pack(anchor=tk.W)
            
            ttk.Label(middle_col, text="Serbest Margin:").pack(anchor=tk.W)
            self.free_margin_label = ttk.Label(middle_col, text=f"{self.account_info.free_margin:.2f}", font=("Arial", 12, "bold"))
            self.free_margin_label.pack(anchor=tk.W)
            
            # Sağ kolon
            right_col = ttk.Frame(account_info_frame)
            right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            ttk.Label(right_col, text="Margin Seviyesi:").pack(anchor=tk.W)
            self.margin_level_label = ttk.Label(right_col, text=f"{self.account_info.margin_level:.2f}%", font=("Arial", 12, "bold"))
            self.margin_level_label.pack(anchor=tk.W)
            
            # Performans grafiği
            performance_frame = ttk.LabelFrame(self.account_frame, text="Performans Grafiği", padding=10)
            performance_frame.pack(fill=tk.BOTH, expand=True)
            
            self.perf_fig = Figure(figsize=(10, 6), dpi=100)
            self.perf_ax = self.perf_fig.add_subplot(111)
            
            self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, performance_frame)
            self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Başlangıç performans grafiği
            self.update_performance_chart()
            
        except Exception as e:
            self.logger.error(f"Account tab oluşturma hatası: {e}")
    
    def load_sample_data(self):
        """Örnek veri yükle"""
        try:
            # Örnek emirler
            sample_orders = [
                ManualOrder(
                    order_id="ORD001",
                    symbol="BTCUSDT",
                    side="buy",
                    order_type="limit",
                    quantity=0.001,
                    price=49500.0,
                    status="pending",
                    timestamp=datetime.now() - timedelta(minutes=30)
                ),
                ManualOrder(
                    order_id="ORD002",
                    symbol="ETHUSDT",
                    side="sell",
                    order_type="market",
                    quantity=0.1,
                    status="filled",
                    timestamp=datetime.now() - timedelta(minutes=15),
                    filled_price=3200.0,
                    filled_quantity=0.1
                )
            ]
            
            for order in sample_orders:
                self.orders[order.order_id] = order
            
            # Örnek pozisyonlar
            sample_positions = [
                Position(
                    symbol="BTCUSDT",
                    side="long",
                    quantity=0.001,
                    entry_price=50000.0,
                    current_price=50100.0,
                    unrealized_pnl=0.1,
                    realized_pnl=0.0,
                    timestamp=datetime.now() - timedelta(hours=2),
                    stop_loss=49000.0,
                    take_profit=52000.0
                )
            ]
            
            for position in sample_positions:
                self.positions[position.symbol] = position
            
            # Tabloları güncelle
            self.update_orders_table()
            self.update_positions_table()
            self.update_account_info()
            
        except Exception as e:
            self.logger.error(f"Örnek veri yükleme hatası: {e}")
    
    def on_symbol_changed(self, event=None):
        """Sembol değiştiğinde"""
        try:
            self.current_symbol = self.symbol_var.get()
            self.update_price()
            self.logger.info(f"Sembol değiştirildi: {self.current_symbol}")
            
        except Exception as e:
            self.logger.error(f"Sembol değiştirme hatası: {e}")
    
    def on_order_type_changed(self, event=None):
        """Emir türü değiştiğinde"""
        try:
            order_type = self.order_type_var.get()
            
            # Fiyat alanlarını aktif/pasif yap
            if order_type == "market":
                self.price_var.set("")
                self.stop_price_var.set("")
            elif order_type == "limit":
                self.stop_price_var.set("")
            elif order_type == "stop":
                pass  # Her iki alan da aktif
            
        except Exception as e:
            self.logger.error(f"Emir türü değiştirme hatası: {e}")
    
    def set_quantity_percentage(self, percentage: float):
        """Miktarı yüzdeye göre ayarla"""
        try:
            # Hesap bakiyesinin yüzdesi
            max_quantity = self.account_info.balance * percentage / self.current_price
            self.quantity_var.set(f"{max_quantity:.6f}")
            
        except Exception as e:
            self.logger.error(f"Miktar ayarlama hatası: {e}")
    
    def submit_order(self):
        """Emir gönder"""
        try:
            # Form verilerini al
            symbol = self.symbol_var.get()
            side = self.side_var.get()
            order_type = self.order_type_var.get()
            
            try:
                quantity = float(self.quantity_var.get())
            except ValueError:
                messagebox.showerror("Hata", "Geçersiz miktar")
                return
            
            price = None
            stop_price = None
            
            if order_type in ["limit", "stop"]:
                try:
                    price = float(self.price_var.get()) if self.price_var.get() else None
                except ValueError:
                    messagebox.showerror("Hata", "Geçersiz fiyat")
                    return
            
            if order_type == "stop":
                try:
                    stop_price = float(self.stop_price_var.get()) if self.stop_price_var.get() else None
                except ValueError:
                    messagebox.showerror("Hata", "Geçersiz stop fiyat")
                    return
            
            # Validasyon
            if quantity <= 0:
                messagebox.showerror("Hata", "Miktar sıfırdan büyük olmalı")
                return
            
            if order_type == "market":
                price = self.current_price
            elif order_type in ["limit", "stop"] and price is None:
                messagebox.showerror("Hata", "Fiyat gerekli")
                return
            
            if order_type == "stop" and stop_price is None:
                messagebox.showerror("Hata", "Stop fiyat gerekli")
                return
            
            # Margin kontrolü
            required_margin = quantity * price * 0.1  # %10 margin
            if required_margin > self.account_info.margin_available:
                messagebox.showerror("Hata", "Yetersiz margin")
                return
            
            # Emir oluştur
            order_id = f"ORD{len(self.orders) + 1:03d}"
            order = ManualOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status="pending",
                timestamp=datetime.now()
            )
            
            # Emri ekle
            with self.lock:
                self.orders[order_id] = order
                self.order_history.append(order)
            
            # Dış entegrasyon: api_client bir callable olarak verilmişse çağır
            try:
                if callable(self.api_client):
                    self.api_client(symbol, side, order_type, price, quantity)
            except Exception as cb_e:
                self.logger.error(f"Harici emir callback hatası: {cb_e}")
            
            # Basit yürütme simülasyonu: market/limit/stop koşulsuz doldur (demo)
            try:
                with self.lock:
                    order.status = "filled"
                    order.filled_price = price or self.current_price
                    order.filled_quantity = quantity
            except Exception:
                pass
            
            # Tabloları güncelle
            self.update_orders_table()
            self.update_account_info()
            
            # Onay mesajı
            messagebox.showinfo("Başarılı", f"Emir gönderildi: {order_id}")
            
            # Callback'leri çağır
            self._notify_order_callbacks(order)
            
            self.logger.info(f"Emir gönderildi: {order_id}")
            
        except Exception as e:
            self.logger.error(f"Emir gönderme hatası: {e}")
            messagebox.showerror("Hata", f"Emir gönderme hatası: {e}")
    
    def clear_form(self):
        """Formu temizle"""
        try:
            self.quantity_var.set("0.001")
            self.price_var.set("")
            self.stop_price_var.set("")
            
        except Exception as e:
            self.logger.error(f"Form temizleme hatası: {e}")
    
    def update_price(self):
        """Fiyatı güncelle"""
        try:
            # Simüle edilmiş fiyat güncelleme
            price_change = np.random.normal(0, 0.001)  # %0.1 volatilite
            self.current_price *= (1 + price_change)
            
            self.current_price_label.config(text=f"{self.current_price:.2f}")
            
            # Pozisyonları güncelle
            self.update_positions_prices()
            
        except Exception as e:
            self.logger.error(f"Fiyat güncelleme hatası: {e}")
    
    def update_positions_prices(self):
        """Pozisyon fiyatlarını güncelle"""
        try:
            with self.lock:
                for symbol, position in self.positions.items():
                    if symbol == self.current_symbol:
                        position.current_price = self.current_price
                        
                        # PnL hesapla
                        if position.side == "long":
                            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                        else:
                            position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
            
            # Tabloları güncelle
            self.update_positions_table()
            self.update_account_info()
            
        except Exception as e:
            self.logger.error(f"Pozisyon fiyat güncelleme hatası: {e}")
    
    def update_orders_table(self):
        """Orders tablosunu güncelle"""
        try:
            # Tabloyu temizle
            for item in self.orders_tree.get_children():
                self.orders_tree.delete(item)
            
            # Emirleri ekle
            with self.lock:
                for order in self.orders.values():
                    self.orders_tree.insert("", "end", values=(
                        order.order_id,
                        order.symbol,
                        order.side.upper(),
                        order.order_type.upper(),
                        f"{order.quantity:.6f}",
                        f"{order.price:.2f}" if order.price else "-",
                        f"{order.stop_price:.2f}" if order.stop_price else "-",
                        order.status.upper(),
                        order.timestamp.strftime("%H:%M:%S")
                    ))
            
        except Exception as e:
            self.logger.error(f"Orders tablosu güncelleme hatası: {e}")
    
    def update_positions_table(self):
        """Positions tablosunu güncelle"""
        try:
            # Tabloyu temizle
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Pozisyonları ekle
            with self.lock:
                for position in self.positions.values():
                    pnl_color = "green" if position.unrealized_pnl >= 0 else "red"
                    
                    self.positions_tree.insert("", "end", values=(
                        position.symbol,
                        position.side.upper(),
                        f"{position.quantity:.6f}",
                        f"{position.entry_price:.2f}",
                        f"{position.current_price:.2f}",
                        f"{position.unrealized_pnl:.2f}",
                        f"{position.stop_loss:.2f}" if position.stop_loss else "-",
                        f"{position.take_profit:.2f}" if position.take_profit else "-",
                        position.timestamp.strftime("%H:%M:%S")
                    ))
            
        except Exception as e:
            self.logger.error(f"Positions tablosu güncelleme hatası: {e}")
    
    def update_account_info(self):
        """Hesap bilgilerini güncelle"""
        try:
            # Toplam unrealized PnL
            total_unrealized_pnl = sum([pos.unrealized_pnl for pos in self.positions.values()])
            
            # Hesap bilgilerini güncelle
            self.account_info.equity = self.account_info.balance + total_unrealized_pnl
            self.account_info.margin_used = sum([pos.quantity * pos.current_price * 0.1 for pos in self.positions.values()])
            self.account_info.margin_available = self.account_info.balance - self.account_info.margin_used
            self.account_info.free_margin = self.account_info.margin_available
            self.account_info.margin_level = (self.account_info.equity / self.account_info.margin_used * 100) if self.account_info.margin_used > 0 else 0
            
            # Etiketleri güncelle
            self.balance_label.config(text=f"{self.account_info.balance:.2f}")
            self.equity_label.config(text=f"{self.account_info.equity:.2f}")
            self.margin_used_label.config(text=f"{self.account_info.margin_used:.2f}")
            self.margin_available_label.config(text=f"{self.account_info.margin_available:.2f}")
            self.leverage_label.config(text=f"{self.account_info.leverage:.1f}x")
            self.free_margin_label.config(text=f"{self.account_info.free_margin:.2f}")
            self.margin_level_label.config(text=f"{self.account_info.margin_level:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Hesap bilgileri güncelleme hatası: {e}")
    
    def update_chart(self):
        """Grafiği güncelle"""
        try:
            self.ax.clear()
            
            # Simüle edilmiş fiyat verisi
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1H')
            prices = [self.current_price * (1 + np.random.normal(0, 0.01)) for _ in range(len(times))]
            
            # Grafik çiz
            self.ax.plot(times, prices, 'b-', linewidth=2, label='Fiyat')
            self.ax.axhline(y=self.current_price, color='r', linestyle='--', alpha=0.7, label='Mevcut Fiyat')
            
            self.ax.set_xlabel('Zaman')
            self.ax.set_ylabel('Fiyat')
            self.ax.set_title(f'{self.current_symbol} Fiyat Grafiği')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # X ekseni formatı
            self.fig.autofmt_xdate()
            
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Grafik güncelleme hatası: {e}")
    
    def update_performance_chart(self):
        """Performans grafiğini güncelle"""
        try:
            self.perf_ax.clear()
            
            # Simüle edilmiş performans verisi
            times = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1D')
            equity_values = [self.account_info.balance + np.random.normal(0, 1000) for _ in range(len(times))]
            
            # Grafik çiz
            self.perf_ax.plot(times, equity_values, 'g-', linewidth=2, label='Özkaynak')
            self.perf_ax.axhline(y=self.account_info.balance, color='b', linestyle='--', alpha=0.7, label='Başlangıç')
            
            self.perf_ax.set_xlabel('Tarih')
            self.perf_ax.set_ylabel('Özkaynak')
            self.perf_ax.set_title('Hesap Performansı')
            self.perf_ax.legend()
            self.perf_ax.grid(True, alpha=0.3)
            
            # X ekseni formatı
            self.perf_fig.autofmt_xdate()
            
            self.perf_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Performans grafiği güncelleme hatası: {e}")
    
    def show_orders_context_menu(self, event):
        """Orders context menüsünü göster"""
        try:
            # Seçili öğeyi al
            item = self.orders_tree.selection()[0] if self.orders_tree.selection() else None
            if not item:
                return
            
            # Context menu oluştur
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="İptal Et", command=lambda: self.cancel_order(item))
            context_menu.add_command(label="Detaylar", command=lambda: self.show_order_details(item))
            
            # Menüyü göster
            context_menu.tk_popup(event.x_root, event.y_root)
            
        except Exception as e:
            self.logger.error(f"Orders context menu hatası: {e}")
    
    def show_positions_context_menu(self, event):
        """Positions context menüsünü göster"""
        try:
            # Seçili öğeyi al
            item = self.positions_tree.selection()[0] if self.positions_tree.selection() else None
            if not item:
                return
            
            # Context menu oluştur
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="Kapat", command=lambda: self.close_position(item))
            context_menu.add_command(label="Stop Loss Ekle", command=lambda: self.add_stop_loss(item))
            context_menu.add_command(label="Take Profit Ekle", command=lambda: self.add_take_profit(item))
            context_menu.add_command(label="Detaylar", command=lambda: self.show_position_details(item))
            
            # Menüyü göster
            context_menu.tk_popup(event.x_root, event.y_root)
            
        except Exception as e:
            self.logger.error(f"Positions context menu hatası: {e}")
    
    def cancel_order(self, item):
        """Emri iptal et"""
        try:
            # Emir ID'sini al
            order_id = self.orders_tree.item(item, "values")[0]
            
            # Emri iptal et
            with self.lock:
                if order_id in self.orders:
                    self.orders[order_id].status = "cancelled"
            
            # Tabloyu güncelle
            self.update_orders_table()
            
            messagebox.showinfo("Başarılı", f"Emir iptal edildi: {order_id}")
            
        except Exception as e:
            self.logger.error(f"Emir iptal etme hatası: {e}")
    
    def close_position(self, item):
        """Pozisyonu kapat"""
        try:
            # Pozisyon sembolünü al
            symbol = self.positions_tree.item(item, "values")[0]
            
            # Pozisyonu kapat
            with self.lock:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Kapatma emri oluştur
                    close_side = "sell" if position.side == "long" else "buy"
                    order_id = f"CLOSE{len(self.orders) + 1:03d}"
                    
                    close_order = ManualOrder(
                        order_id=order_id,
                        symbol=symbol,
                        side=close_side,
                        order_type="market",
                        quantity=position.quantity,
                        price=self.current_price,
                        status="filled",
                        timestamp=datetime.now(),
                        filled_price=self.current_price,
                        filled_quantity=position.quantity
                    )
                    
                    # Emri ekle
                    self.orders[order_id] = close_order
                    self.order_history.append(close_order)
                    
                    # Pozisyonu kaldır
                    del self.positions[symbol]
            
            # Tabloları güncelle
            self.update_orders_table()
            self.update_positions_table()
            self.update_account_info()
            
            messagebox.showinfo("Başarılı", f"Pozisyon kapatıldı: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Pozisyon kapatma hatası: {e}")
    
    def add_stop_loss(self, item):
        """Stop loss ekle"""
        try:
            # Pozisyon sembolünü al
            symbol = self.positions_tree.item(item, "values")[0]
            
            # Stop loss fiyatını al
            stop_price = simpledialog.askfloat("Stop Loss", "Stop Loss fiyatını girin:")
            if stop_price is None:
                return
            
            # Stop loss ekle
            with self.lock:
                if symbol in self.positions:
                    self.positions[symbol].stop_loss = stop_price
            
            # Tabloyu güncelle
            self.update_positions_table()
            
            messagebox.showinfo("Başarılı", f"Stop Loss eklendi: {stop_price}")
            
        except Exception as e:
            self.logger.error(f"Stop loss ekleme hatası: {e}")
    
    def add_take_profit(self, item):
        """Take profit ekle"""
        try:
            # Pozisyon sembolünü al
            symbol = self.positions_tree.item(item, "values")[0]
            
            # Take profit fiyatını al
            take_profit = simpledialog.askfloat("Take Profit", "Take Profit fiyatını girin:")
            if take_profit is None:
                return
            
            # Take profit ekle
            with self.lock:
                if symbol in self.positions:
                    self.positions[symbol].take_profit = take_profit
            
            # Tabloyu güncelle
            self.update_positions_table()
            
            messagebox.showinfo("Başarılı", f"Take Profit eklendi: {take_profit}")
            
        except Exception as e:
            self.logger.error(f"Take profit ekleme hatası: {e}")
    
    def show_order_details(self, item):
        """Emir detaylarını göster"""
        try:
            # Emir ID'sini al
            order_id = self.orders_tree.item(item, "values")[0]
            
            # Emir bilgilerini al
            with self.lock:
                order = self.orders.get(order_id)
                if not order:
                    return
            
            # Detay penceresi
            details_window = tk.Toplevel(self.root)
            details_window.title(f"Emir Detayları - {order_id}")
            details_window.geometry("400x300")
            
            # Detay bilgileri
            details_text = tk.Text(details_window, wrap=tk.WORD, padx=10, pady=10)
            details_text.pack(fill=tk.BOTH, expand=True)
            
            details = f"""
EMİR DETAYLARI
==============
Emir ID: {order.order_id}
Sembol: {order.symbol}
Yön: {order.side.upper()}
Tür: {order.order_type.upper()}
Miktar: {order.quantity:.6f}
Fiyat: {order.price:.2f if order.price else 'Market'}
Stop Fiyat: {order.stop_price:.2f if order.stop_price else '-'}
Durum: {order.status.upper()}
Tarih: {order.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Doldurulan Fiyat: {order.filled_price:.2f if order.filled_price else '-'}
Doldurulan Miktar: {order.filled_quantity:.6f}
Komisyon: {order.commission:.6f}
"""
            
            details_text.insert(tk.END, details)
            details_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Emir detayları gösterme hatası: {e}")
    
    def show_position_details(self, item):
        """Pozisyon detaylarını göster"""
        try:
            # Pozisyon sembolünü al
            symbol = self.positions_tree.item(item, "values")[0]
            
            # Pozisyon bilgilerini al
            with self.lock:
                position = self.positions.get(symbol)
                if not position:
                    return
            
            # Detay penceresi
            details_window = tk.Toplevel(self.root)
            details_window.title(f"Pozisyon Detayları - {symbol}")
            details_window.geometry("400x300")
            
            # Detay bilgileri
            details_text = tk.Text(details_window, wrap=tk.WORD, padx=10, pady=10)
            details_text.pack(fill=tk.BOTH, expand=True)
            
            details = f"""
POZİSYON DETAYLARI
==================
Sembol: {position.symbol}
Yön: {position.side.upper()}
Miktar: {position.quantity:.6f}
Giriş Fiyatı: {position.entry_price:.2f}
Mevcut Fiyat: {position.current_price:.2f}
Gerçekleşmemiş PnL: {position.unrealized_pnl:.2f}
Gerçekleşmiş PnL: {position.realized_pnl:.2f}
Tarih: {position.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Stop Loss: {position.stop_loss:.2f if position.stop_loss else '-'}
Take Profit: {position.take_profit:.2f if position.take_profit else '-'}
"""
            
            details_text.insert(tk.END, details)
            details_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Pozisyon detayları gösterme hatası: {e}")
    
    def refresh_orders(self):
        """Emirleri yenile"""
        try:
            self.update_orders_table()
            messagebox.showinfo("Başarılı", "Emirler yenilendi")
            
        except Exception as e:
            self.logger.error(f"Emirler yenileme hatası: {e}")
    
    def refresh_positions(self):
        """Pozisyonları yenile"""
        try:
            self.update_positions_table()
            messagebox.showinfo("Başarılı", "Pozisyonlar yenilendi")
            
        except Exception as e:
            self.logger.error(f"Pozisyonlar yenileme hatası: {e}")
    
    def cancel_all_orders(self):
        """Tüm emirleri iptal et"""
        try:
            result = messagebox.askyesno("Onay", "Tüm emirleri iptal etmek istediğinizden emin misiniz?")
            if not result:
                return
            
            # Tüm emirleri iptal et
            with self.lock:
                for order in self.orders.values():
                    if order.status == "pending":
                        order.status = "cancelled"
            
            # Tabloyu güncelle
            self.update_orders_table()
            
            messagebox.showinfo("Başarılı", "Tüm emirler iptal edildi")
            
        except Exception as e:
            self.logger.error(f"Tüm emirleri iptal etme hatası: {e}")
    
    def close_all_positions(self):
        """Tüm pozisyonları kapat"""
        try:
            result = messagebox.askyesno("Onay", "Tüm pozisyonları kapatmak istediğinizden emin misiniz?")
            if not result:
                return
            
            # Tüm pozisyonları kapat
            with self.lock:
                for symbol, position in list(self.positions.items()):
                    close_side = "sell" if position.side == "long" else "buy"
                    order_id = f"CLOSE{len(self.orders) + 1:03d}"
                    
                    close_order = ManualOrder(
                        order_id=order_id,
                        symbol=symbol,
                        side=close_side,
                        order_type="market",
                        quantity=position.quantity,
                        price=self.current_price,
                        status="filled",
                        timestamp=datetime.now(),
                        filled_price=self.current_price,
                        filled_quantity=position.quantity
                    )
                    
                    self.orders[order_id] = close_order
                    self.order_history.append(close_order)
                
                self.positions.clear()
            
            # Tabloları güncelle
            self.update_orders_table()
            self.update_positions_table()
            self.update_account_info()
            
            messagebox.showinfo("Başarılı", "Tüm pozisyonlar kapatıldı")
            
        except Exception as e:
            self.logger.error(f"Tüm pozisyonları kapatma hatası: {e}")
    
    def export_orders(self):
        """Emirleri export et"""
        try:
            filename = f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            data = []
            for order in self.orders.values():
                data.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type,
                    'quantity': order.quantity,
                    'price': order.price,
                    'stop_price': order.stop_price,
                    'status': order.status,
                    'timestamp': order.timestamp,
                    'filled_price': order.filled_price,
                    'filled_quantity': order.filled_quantity,
                    'commission': order.commission
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Başarılı", f"Emirler {filename} dosyasına export edildi")
            
        except Exception as e:
            self.logger.error(f"Emirler export hatası: {e}")
            messagebox.showerror("Hata", f"Export hatası: {e}")
    
    def export_positions(self):
        """Pozisyonları export et"""
        try:
            filename = f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            data = []
            for position in self.positions.values():
                data.append({
                    'symbol': position.symbol,
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'timestamp': position.timestamp,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Başarılı", f"Pozisyonlar {filename} dosyasına export edildi")
            
        except Exception as e:
            self.logger.error(f"Pozisyonlar export hatası: {e}")
            messagebox.showerror("Hata", f"Export hatası: {e}")
    
    def start_update_loop(self):
        """Güncelleme döngüsünü başlat"""
        try:
            def update_loop():
                while True:
                    try:
                        # Fiyatları güncelle
                        if self.root and self.root.winfo_exists():
                            self.root.after(0, self.update_price)
                        
                        time.sleep(10)  # 10 saniyede bir güncelle
                        
                    except Exception as e:
                        self.logger.error(f"Güncelleme döngüsü hatası: {e}")
                        time.sleep(30)
            
            # Thread başlat
            update_thread = threading.Thread(target=update_loop, daemon=True)
            update_thread.start()
            
        except Exception as e:
            self.logger.error(f"Güncelleme döngüsü başlatma hatası: {e}")
    
    def add_order_callback(self, callback):
        """Emir callback'i ekle"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback):
        """Pozisyon callback'i ekle"""
        self.position_callbacks.append(callback)
    
    def _notify_order_callbacks(self, order: ManualOrder):
        """Emir callback'lerini çağır"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Emir callback hatası: {e}")
    
    def _notify_position_callbacks(self, position: Position):
        """Pozisyon callback'lerini çağır"""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Pozisyon callback hatası: {e}")
    
    def run(self):
        """GUI'yi çalıştır"""
        try:
            if not self.parent:
                self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI çalıştırma hatası: {e}")

# Global Manuel Trading GUI
manual_trading_gui = ManualTradingGUI()
