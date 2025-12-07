"""
GUI Order Book Görüntüleme Modülü
Gerçek zamanlı order book görselleştirmesi ve analizi
"""

import tkinter as tk
from tkinter import ttk, messagebox
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
class OrderBookLevel:
    """Order book seviyesi"""
    price: float
    quantity: float
    timestamp: datetime

@dataclass
class OrderBookSnapshot:
    """Order book anlık görüntüsü"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: float
    best_ask: float
    spread: float
    mid_price: float

class OrderBookDisplayMode(Enum):
    """Order book görüntüleme modları"""
    STANDARD = "standard"
    DEPTH_CHART = "depth_chart"
    HEATMAP = "heatmap"
    ANALYTICS = "analytics"

class OrderBookGUI:
    """Order Book GUI sınıfı"""
    
    def __init__(self, parent=None):
        self.logger = logging.getLogger("orderbook_gui")
        self.parent = parent
        self.root = None
        
        # Order book verileri
        self.orderbook_data = deque(maxlen=1000)  # Son 1000 snapshot
        self.current_symbol = "BTCUSDT"
        self.display_mode = OrderBookDisplayMode.STANDARD
        
        # GUI elemanları
        self.main_frame = None
        self.orderbook_frame = None
        self.analytics_frame = None
        self.control_frame = None
        
        # Matplotlib elemanları
        self.fig = None
        self.ax = None
        self.canvas = None
        
        # Order book tablosu
        self.bids_tree = None
        self.asks_tree = None
        self.combined_tree = None
        
        # Analiz verileri
        self.depth_imbalance_history = deque(maxlen=100)
        self.spread_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Callback'ler
        self.orderbook_callbacks = []
        
        self.logger.info("Order Book GUI başlatıldı")
    
    def create_gui(self):
        """GUI oluştur"""
        try:
            if self.parent:
                self.root = self.parent
            else:
                self.root = tk.Tk()
                self.root.title("Order Book Viewer")
                self.root.geometry("1200x800")
            
            # Treeview görünürlüğünü garanti et (tema/stil)
            try:
                style = ttk.Style(self.root)
                # Mevcut temayı koru, satır yüksekliği ve renkleri belirgin yap
                style.configure("Treeview", rowheight=22, foreground="black", background="white")
                style.configure("Treeview.Heading", foreground="black")
                style.map("Treeview", background=[("selected", "#cce5ff")])
            except Exception:
                pass

            # Ana frame
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Kontrol paneli
            self.create_control_panel()
            
            # Order book görüntüleme alanı
            self.create_orderbook_display()
            
            # Analiz paneli
            self.create_analytics_panel()
            
            # Başlangıç verilerini yükle
            for _ in range(5):
                self.load_sample_data()
            
            # Güncelleme döngüsünü başlat
            self.start_update_loop()
            
            self.logger.info("Order Book GUI oluşturuldu")
            
        except Exception as e:
            self.logger.error(f"GUI oluşturma hatası: {e}")
    
    def create_control_panel(self):
        """Kontrol panelini oluştur"""
        try:
            self.control_frame = ttk.LabelFrame(self.main_frame, text="Kontroller", padding=10)
            self.control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Sembol seçimi
            symbol_frame = ttk.Frame(self.control_frame)
            symbol_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(symbol_frame, text="Sembol:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.symbol_var = tk.StringVar(value=self.current_symbol)
            symbol_combo = ttk.Combobox(
                symbol_frame, 
                textvariable=self.symbol_var,
                values=["BTCUSDT", "ETHUSDT", "AVAXUSDT", "ADAUSDT", "SOLUSDT"],
                state="readonly",
                width=10
            )
            symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
            symbol_combo.bind("<<ComboboxSelected>>", self.on_symbol_changed)
            
            # Görüntüleme modu
            mode_frame = ttk.Frame(self.control_frame)
            mode_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(mode_frame, text="Mod:").pack(side=tk.LEFT, padx=(0, 5))
            
            self.mode_var = tk.StringVar(value=self.display_mode.value)
            mode_combo = ttk.Combobox(
                mode_frame,
                textvariable=self.mode_var,
                values=[mode.value for mode in OrderBookDisplayMode],
                state="readonly",
                width=15
            )
            mode_combo.pack(side=tk.LEFT, padx=(0, 10))
            mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)
            
            # Kontrol butonları
            button_frame = ttk.Frame(self.control_frame)
            button_frame.pack(fill=tk.X)
            
            ttk.Button(button_frame, text="Yenile", command=self.refresh_data).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Analiz", command=self.show_analytics).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Export", command=self.export_data).pack(side=tk.LEFT, padx=(0, 5))
            
            # Durum etiketi
            self.status_label = ttk.Label(self.control_frame, text="Hazır", foreground="green")
            self.status_label.pack(side=tk.RIGHT)
            
        except Exception as e:
            self.logger.error(f"Kontrol paneli oluşturma hatası: {e}")
    
    def create_orderbook_display(self):
        """Order book görüntüleme alanını oluştur"""
        try:
            self.orderbook_frame = ttk.LabelFrame(self.main_frame, text="Order Book", padding=10)
            self.orderbook_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Ana görüntüleme alanı
            display_frame = ttk.Frame(self.orderbook_frame)
            display_frame.pack(fill=tk.BOTH, expand=True)
            
            # Sol panel - Order book tablosu
            table_frame = ttk.Frame(display_frame)
            table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            # ASK tablosu
            ask_frame = ttk.LabelFrame(table_frame, text="ASK (Satış)", padding=5)
            ask_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            ask_columns = ("Fiyat", "Miktar", "Toplam")
            self.asks_tree = ttk.Treeview(ask_frame, columns=ask_columns, show="headings", height=10)
            
            for col in ask_columns:
                self.asks_tree.heading(col, text=col)
                self.asks_tree.column(col, width=80, anchor="e")
            
            ask_scrollbar = ttk.Scrollbar(ask_frame, orient=tk.VERTICAL, command=self.asks_tree.yview)
            self.asks_tree.configure(yscrollcommand=ask_scrollbar.set)
            
            self.asks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            ask_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            # Eğer henüz veri yoksa görünürlüğü test etmek için birkaç satır ekle
            if not self.orderbook_data:
                try:
                    for price, qty, cum in [(50050.0, 0.5000, 0.5000), (50060.0, 0.2500, 0.7500)]:
                        self.asks_tree.insert("", 0, values=(f"{price:.2f}", f"{qty:.4f}", f"{cum:.4f}"))
                except Exception:
                    pass
            
            # BID tablosu
            bid_frame = ttk.LabelFrame(table_frame, text="BID (Alış)", padding=5)
            bid_frame.pack(fill=tk.BOTH, expand=True)
            
            bid_columns = ("Fiyat", "Miktar", "Toplam")
            self.bids_tree = ttk.Treeview(bid_frame, columns=bid_columns, show="headings", height=10)
            
            for col in bid_columns:
                self.bids_tree.heading(col, text=col)
                self.bids_tree.column(col, width=80, anchor="e")
            
            bid_scrollbar = ttk.Scrollbar(bid_frame, orient=tk.VERTICAL, command=self.bids_tree.yview)
            self.bids_tree.configure(yscrollcommand=bid_scrollbar.set)
            
            self.bids_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            bid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            # Eğer henüz veri yoksa görünürlüğü test etmek için birkaç satır ekle
            if not self.orderbook_data:
                try:
                    for price, qty, cum in [(49950.0, 0.4000, 0.4000), (49940.0, 0.3000, 0.7000)]:
                        self.bids_tree.insert("", "end", values=(f"{price:.2f}", f"{qty:.4f}", f"{cum:.4f}"))
                except Exception:
                    pass

            # Birleşik tablo (Fiyat, Miktar, Toplam, Tip)
            combined_frame = ttk.LabelFrame(table_frame, text="Emir Defteri", padding=5)
            combined_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            combined_columns = ("Fiyat", "Miktar", "Toplam", "Tip")
            self.combined_tree = ttk.Treeview(combined_frame, columns=combined_columns, show="headings", height=14)
            for col in combined_columns:
                self.combined_tree.heading(col, text=col)
                anchor = "center" if col == "Tip" else "e"
                width = 70 if col == "Tip" else 100
                self.combined_tree.column(col, width=width, anchor=anchor)
            combined_scrollbar = ttk.Scrollbar(combined_frame, orient=tk.VERTICAL, command=self.combined_tree.yview)
            self.combined_tree.configure(yscrollcommand=combined_scrollbar.set)
            self.combined_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            combined_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            if not self.orderbook_data:
                try:
                    # Anında ekle
                    self.combined_tree.insert("", "end", values=("50050.00", "0.5000", "0.5000", "ASK"))
                    self.combined_tree.insert("", "end", values=("49950.00", "0.4000", "0.4000", "BID"))
                    # Zamanlamaya bağlı görünmezlik ihtimaline karşı bir de after ile tekrar ekle
                    self.root.after(50, lambda: self.combined_tree.insert("", "end", values=("50060.00", "0.2500", "0.7500", "ASK")))
                    self.root.after(50, lambda: self.combined_tree.insert("", "end", values=("49940.00", "0.3000", "0.7000", "BID")))
                except Exception:
                    pass
            
            # Sağ panel - Grafik
            chart_frame = ttk.LabelFrame(display_frame, text="Grafik", padding=5)
            chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Matplotlib figürü
            self.fig = Figure(figsize=(6, 8), dpi=100)
            self.ax = self.fig.add_subplot(111)
            
            self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Başlangıç grafiği
            self.update_chart()
            
        except Exception as e:
            self.logger.error(f"Order book görüntüleme alanı oluşturma hatası: {e}")
    
    def create_analytics_panel(self):
        """Analiz panelini oluştur"""
        try:
            self.analytics_frame = ttk.LabelFrame(self.main_frame, text="Analiz", padding=10)
            self.analytics_frame.pack(fill=tk.X)
            
            # Analiz metrikleri
            metrics_frame = ttk.Frame(self.analytics_frame)
            metrics_frame.pack(fill=tk.X)
            
            # Sol kolon
            left_col = ttk.Frame(metrics_frame)
            left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Label(left_col, text="Spread:").pack(anchor=tk.W)
            self.spread_label = ttk.Label(left_col, text="0.00", font=("Arial", 10, "bold"))
            self.spread_label.pack(anchor=tk.W)
            
            ttk.Label(left_col, text="Mid Price:").pack(anchor=tk.W)
            self.mid_price_label = ttk.Label(left_col, text="0.00", font=("Arial", 10, "bold"))
            self.mid_price_label.pack(anchor=tk.W)
            
            ttk.Label(left_col, text="Depth Imbalance:").pack(anchor=tk.W)
            self.imbalance_label = ttk.Label(left_col, text="0.00", font=("Arial", 10, "bold"))
            self.imbalance_label.pack(anchor=tk.W)
            
            # Orta kolon
            middle_col = ttk.Frame(metrics_frame)
            middle_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Label(middle_col, text="Total Volume:").pack(anchor=tk.W)
            self.volume_label = ttk.Label(middle_col, text="0.00", font=("Arial", 10, "bold"))
            self.volume_label.pack(anchor=tk.W)
            
            ttk.Label(middle_col, text="Bid Volume:").pack(anchor=tk.W)
            self.bid_volume_label = ttk.Label(middle_col, text="0.00", font=("Arial", 10, "bold"))
            self.bid_volume_label.pack(anchor=tk.W)
            
            ttk.Label(middle_col, text="Ask Volume:").pack(anchor=tk.W)
            self.ask_volume_label = ttk.Label(middle_col, text="0.00", font=("Arial", 10, "bold"))
            self.ask_volume_label.pack(anchor=tk.W)
            
            # Sağ kolon
            right_col = ttk.Frame(metrics_frame)
            right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            ttk.Label(right_col, text="Spoofing Alert:").pack(anchor=tk.W)
            self.spoofing_label = ttk.Label(right_col, text="Temiz", foreground="green", font=("Arial", 10, "bold"))
            self.spoofing_label.pack(anchor=tk.W)
            
            ttk.Label(right_col, text="Market Impact:").pack(anchor=tk.W)
            self.impact_label = ttk.Label(right_col, text="Düşük", font=("Arial", 10, "bold"))
            self.impact_label.pack(anchor=tk.W)
            
            ttk.Label(right_col, text="Last Update:").pack(anchor=tk.W)
            self.update_label = ttk.Label(right_col, text="--:--:--", font=("Arial", 10, "bold"))
            self.update_label.pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Analiz paneli oluşturma hatası: {e}")
    
    def load_sample_data(self):
        """Örnek veri yükle"""
        try:
            # Simüle edilmiş order book verisi
            base_price = 50000.0  # BTC fiyatı
            
            # ASK seviyeleri (satış)
            asks = []
            for i in range(20):
                price = base_price + (i + 1) * 10
                quantity = np.random.uniform(0.1, 5.0)
                asks.append(OrderBookLevel(price, quantity, datetime.now()))
            
            # BID seviyeleri (alış)
            bids = []
            for i in range(20):
                price = base_price - (i + 1) * 10
                quantity = np.random.uniform(0.1, 5.0)
                bids.append(OrderBookLevel(price, quantity, datetime.now()))
            
            # Order book snapshot
            snapshot = OrderBookSnapshot(
                symbol=self.current_symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                best_bid=bids[0].price if bids else base_price,
                best_ask=asks[0].price if asks else base_price,
                spread=(asks[0].price - bids[0].price) if asks and bids else 0,
                mid_price=(asks[0].price + bids[0].price) / 2 if asks and bids else base_price
            )
            
            self.orderbook_data.append(snapshot)
            self.logger.info(f"Demo snapshot üretildi: bids={len(bids)}, asks={len(asks)}, spread={snapshot.spread:.2f}")
            self.update_display()
            
        except Exception as e:
            self.logger.error(f"Örnek veri yükleme hatası: {e}")
    
    def update_display(self):
        """Görüntüyü güncelle"""
        try:
            if not self.orderbook_data:
                return
            
            latest_snapshot = self.orderbook_data[-1]
            
            # Tabloları güncelle
            self.update_orderbook_tables(latest_snapshot)
            
            # Grafiği güncelle
            self.update_chart()
            
            # Analiz metriklerini güncelle
            self.update_analytics(latest_snapshot)
            
            # Durum etiketini güncelle
            try:
                lbl = getattr(self, 'status_label', None)
                if lbl and str(lbl) and getattr(lbl, 'winfo_exists', lambda: True)():
                    self.status_label.config(text=f"Güncellendi: {datetime.now().strftime('%H:%M:%S')}")
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Görüntü güncelleme hatası: {e}")
    
    def update_orderbook_tables(self, snapshot: OrderBookSnapshot):
        """Order book tablolarını güncelle"""
        try:
            if not self.asks_tree or not self.bids_tree:
                return
            # ASK tablosunu temizle ve doldur
            for item in self.asks_tree.get_children():
                self.asks_tree.delete(item)
            
            cumulative_ask = 0
            ask_count = 0
            for i, ask in enumerate(snapshot.asks[:15]):  # İlk 15 seviye
                cumulative_ask += ask.quantity
                self.asks_tree.insert("", 0, values=(
                    f"{ask.price:.2f}",
                    f"{ask.quantity:.4f}",
                    f"{cumulative_ask:.4f}"
                ))
                ask_count += 1
            
            # BID tablosunu temizle ve doldur
            for item in self.bids_tree.get_children():
                self.bids_tree.delete(item)
            
            cumulative_bid = 0
            bid_count = 0
            for i, bid in enumerate(snapshot.bids[:15]):  # İlk 15 seviye
                cumulative_bid += bid.quantity
                self.bids_tree.insert("", "end", values=(
                    f"{bid.price:.2f}",
                    f"{bid.quantity:.4f}",
                    f"{cumulative_bid:.4f}"
                ))
                bid_count += 1
            if self.combined_tree:
                for item in self.combined_tree.get_children():
                    self.combined_tree.delete(item)
                cumulative = 0
                for ask in snapshot.asks[:15]:
                    cumulative += ask.quantity
                    self.combined_tree.insert("", "end", values=(
                        f"{ask.price:.2f}", f"{ask.quantity:.4f}", f"{cumulative:.4f}", "ASK"
                    ))
                cumulative = 0
                for bid in snapshot.bids[:15]:
                    cumulative += bid.quantity
                    self.combined_tree.insert("", "end", values=(
                        f"{bid.price:.2f}", f"{bid.quantity:.4f}", f"{cumulative:.4f}", "BID"
                    ))
            self.logger.info(f"Tablo dolduruldu: asks={ask_count}, bids={bid_count}")
            
        except Exception as e:
            self.logger.error(f"Order book tabloları güncelleme hatası: {e}")
    
    def update_chart(self):
        """Grafiği güncelle"""
        try:
            if not self.orderbook_data:
                return
            if not self.ax or not self.canvas:
                return
            
            latest_snapshot = self.orderbook_data[-1]
            
            self.ax.clear()
            
            if self.display_mode == OrderBookDisplayMode.STANDARD:
                self._draw_standard_chart(latest_snapshot)
            elif self.display_mode == OrderBookDisplayMode.DEPTH_CHART:
                self._draw_depth_chart(latest_snapshot)
            elif self.display_mode == OrderBookDisplayMode.HEATMAP:
                self._draw_heatmap_chart(latest_snapshot)
            elif self.display_mode == OrderBookDisplayMode.ANALYTICS:
                self._draw_analytics_chart()
            
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Grafik güncelleme hatası: {e}")
    
    def _draw_standard_chart(self, snapshot: OrderBookSnapshot):
        """Standart order book grafiği"""
        try:
            # ASK seviyeleri
            ask_prices = [ask.price for ask in snapshot.asks[:10]]
            ask_quantities = [ask.quantity for ask in snapshot.asks[:10]]
            
            # BID seviyeleri
            bid_prices = [bid.price for bid in snapshot.bids[:10]]
            bid_quantities = [bid.quantity for bid in snapshot.bids[:10]]
            
            # Grafik çiz
            self.ax.barh(ask_prices, ask_quantities, height=5, color='red', alpha=0.7, label='ASK')
            self.ax.barh(bid_prices, bid_quantities, height=5, color='green', alpha=0.7, label='BID')
            
            # Mid price çizgisi
            self.ax.axhline(y=snapshot.mid_price, color='blue', linestyle='--', alpha=0.8, label='Mid Price')
            
            self.ax.set_xlabel('Miktar')
            self.ax.set_ylabel('Fiyat')
            self.ax.set_title(f'{snapshot.symbol} Order Book')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Standart grafik çizme hatası: {e}")
    
    def _draw_depth_chart(self, snapshot: OrderBookSnapshot):
        """Depth chart grafiği"""
        try:
            # Tüm seviyeleri birleştir
            all_prices = []
            all_quantities = []
            colors = []
            
            # ASK seviyeleri (kırmızı)
            for ask in snapshot.asks[:15]:
                all_prices.append(ask.price)
                all_quantities.append(ask.quantity)
                colors.append('red')
            
            # BID seviyeleri (yeşil)
            for bid in snapshot.bids[:15]:
                all_prices.append(bid.price)
                all_quantities.append(bid.quantity)
                colors.append('green')
            
            # Grafik çiz
            self.ax.scatter(all_quantities, all_prices, c=colors, alpha=0.7, s=50)
            
            # Mid price çizgisi
            self.ax.axhline(y=snapshot.mid_price, color='blue', linestyle='--', alpha=0.8)
            
            self.ax.set_xlabel('Miktar')
            self.ax.set_ylabel('Fiyat')
            self.ax.set_title(f'{snapshot.symbol} Depth Chart')
            self.ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Depth chart çizme hatası: {e}")
    
    def _draw_heatmap_chart(self, snapshot: OrderBookSnapshot):
        """Heatmap grafiği"""
        try:
            # Fiyat aralığını belirle
            min_price = min([bid.price for bid in snapshot.bids[:10]] + [ask.price for ask in snapshot.asks[:10]])
            max_price = max([bid.price for bid in snapshot.bids[:10]] + [ask.price for ask in snapshot.asks[:10]])
            
            # Grid oluştur
            price_bins = np.linspace(min_price, max_price, 20)
            quantity_bins = np.linspace(0, max([ask.quantity for ask in snapshot.asks[:10]] + [bid.quantity for bid in snapshot.bids[:10]]), 10)
            
            # Heatmap verisi
            heatmap_data = np.zeros((len(price_bins)-1, len(quantity_bins)-1))
            
            # ASK seviyeleri
            for ask in snapshot.asks[:10]:
                price_idx = np.digitize(ask.price, price_bins) - 1
                quantity_idx = np.digitize(ask.quantity, quantity_bins) - 1
                if 0 <= price_idx < len(price_bins)-1 and 0 <= quantity_idx < len(quantity_bins)-1:
                    heatmap_data[price_idx, quantity_idx] += ask.quantity
            
            # BID seviyeleri
            for bid in snapshot.bids[:10]:
                price_idx = np.digitize(bid.price, price_bins) - 1
                quantity_idx = np.digitize(bid.quantity, quantity_bins) - 1
                if 0 <= price_idx < len(price_bins)-1 and 0 <= quantity_idx < len(quantity_bins)-1:
                    heatmap_data[price_idx, quantity_idx] += bid.quantity
            
            # Heatmap çiz
            im = self.ax.imshow(heatmap_data.T, cmap='RdYlGn', aspect='auto', origin='lower')
            
            # Renk çubuğu
            self.fig.colorbar(im, ax=self.ax, label='Miktar')
            
            self.ax.set_xlabel('Fiyat Bins')
            self.ax.set_ylabel('Miktar Bins')
            self.ax.set_title(f'{snapshot.symbol} Order Book Heatmap')
            
        except Exception as e:
            self.logger.error(f"Heatmap çizme hatası: {e}")
    
    def _draw_analytics_chart(self):
        """Analiz grafiği"""
        try:
            if len(self.depth_imbalance_history) < 2:
                return
            
            # Zaman serisi verileri
            times = list(range(len(self.depth_imbalance_history)))
            imbalance_data = list(self.depth_imbalance_history)
            spread_data = list(self.spread_history)
            
            # İki alt grafik
            ax1 = self.ax
            ax2 = ax1.twinx()
            
            # Depth imbalance
            ax1.plot(times, imbalance_data, 'b-', label='Depth Imbalance', linewidth=2)
            ax1.set_ylabel('Depth Imbalance', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Spread
            ax2.plot(times, spread_data, 'r-', label='Spread', linewidth=2)
            ax2.set_ylabel('Spread', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax1.set_xlabel('Zaman')
            ax1.set_title('Order Book Analiz Metrikleri')
            ax1.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Analiz grafiği çizme hatası: {e}")
    
    def update_analytics(self, snapshot: OrderBookSnapshot):
        """Analiz metriklerini güncelle"""
        try:
            spread_label = getattr(self, 'spread_label', None)
            mid_price_label = getattr(self, 'mid_price_label', None)
            imbalance_label = getattr(self, 'imbalance_label', None)
            volume_label = getattr(self, 'volume_label', None)
            bid_volume_label = getattr(self, 'bid_volume_label', None)
            ask_volume_label = getattr(self, 'ask_volume_label', None)
            spoofing_label = getattr(self, 'spoofing_label', None)
            impact_label = getattr(self, 'impact_label', None)
            update_label = getattr(self, 'update_label', None)
            if not all([spread_label, mid_price_label, imbalance_label, volume_label, bid_volume_label, ask_volume_label, spoofing_label, impact_label, update_label]):
                return
            # Spread
            spread = snapshot.spread
            if getattr(spread_label, 'winfo_exists', lambda: True)():
                spread_label.config(text=f"{spread:.2f}")
            self.spread_history.append(spread)
            
            # Mid price
            mid_price = snapshot.mid_price
            if getattr(mid_price_label, 'winfo_exists', lambda: True)():
                mid_price_label.config(text=f"{mid_price:.2f}")
            
            # Depth imbalance
            bid_volume = sum([bid.quantity for bid in snapshot.bids[:10]])
            ask_volume = sum([ask.quantity for ask in snapshot.asks[:10]])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0
            
            if getattr(imbalance_label, 'winfo_exists', lambda: True)():
                imbalance_label.config(text=f"{imbalance:.3f}")
            self.depth_imbalance_history.append(imbalance)
            
            # Volume bilgileri
            if getattr(volume_label, 'winfo_exists', lambda: True)():
                volume_label.config(text=f"{total_volume:.2f}")
            if getattr(bid_volume_label, 'winfo_exists', lambda: True)():
                bid_volume_label.config(text=f"{bid_volume:.2f}")
            if getattr(ask_volume_label, 'winfo_exists', lambda: True)():
                ask_volume_label.config(text=f"{ask_volume:.2f}")
            
            # Spoofing analizi (basit)
            spoofing_score = self._calculate_spoofing_score(snapshot)
            if spoofing_score > 0.7:
                if getattr(spoofing_label, 'winfo_exists', lambda: True)():
                    spoofing_label.config(text="Yüksek Risk", foreground="red")
            elif spoofing_score > 0.4:
                if getattr(spoofing_label, 'winfo_exists', lambda: True)():
                    spoofing_label.config(text="Orta Risk", foreground="orange")
            else:
                if getattr(spoofing_label, 'winfo_exists', lambda: True)():
                    spoofing_label.config(text="Temiz", foreground="green")
            
            # Market impact
            impact = self._calculate_market_impact(snapshot)
            if impact > 0.02:
                if getattr(impact_label, 'winfo_exists', lambda: True)():
                    impact_label.config(text="Yüksek", foreground="red")
            elif impact > 0.01:
                if getattr(impact_label, 'winfo_exists', lambda: True)():
                    impact_label.config(text="Orta", foreground="orange")
            else:
                if getattr(impact_label, 'winfo_exists', lambda: True)():
                    impact_label.config(text="Düşük", foreground="green")
            
            # Son güncelleme
            if getattr(update_label, 'winfo_exists', lambda: True)():
                update_label.config(text=snapshot.timestamp.strftime("%H:%M:%S"))
            
        except Exception as e:
            self.logger.error(f"Analiz güncelleme hatası: {e}")
    
    def _calculate_spoofing_score(self, snapshot: OrderBookSnapshot) -> float:
        """Spoofing skoru hesapla"""
        try:
            score = 0.0
            
            # Büyük emirler kontrolü
            large_orders = 0
            for bid in snapshot.bids[:5]:
                if bid.quantity > 10:  # Büyük emir
                    large_orders += 1
            
            for ask in snapshot.asks[:5]:
                if ask.quantity > 10:  # Büyük emir
                    large_orders += 1
            
            score += min(large_orders / 10, 0.5)
            
            # Spread anomalisi
            if snapshot.spread > snapshot.mid_price * 0.01:  # %1'den büyük spread
                score += 0.3
            
            # Volume dengesizliği
            bid_volume = sum([bid.quantity for bid in snapshot.bids[:5]])
            ask_volume = sum([ask.quantity for ask in snapshot.asks[:5]])
            
            if abs(bid_volume - ask_volume) / (bid_volume + ask_volume) > 0.5:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Spoofing skoru hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_market_impact(self, snapshot: OrderBookSnapshot) -> float:
        """Market impact hesapla"""
        try:
            # Basit market impact hesaplama
            total_volume = sum([bid.quantity for bid in snapshot.bids[:10]]) + sum([ask.quantity for ask in snapshot.asks[:10]])
            
            if total_volume > 0:
                # Büyük emirlerin oranı
                large_volume = sum([bid.quantity for bid in snapshot.bids[:3]]) + sum([ask.quantity for ask in snapshot.asks[:3]])
                impact = large_volume / total_volume
            else:
                impact = 0
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Market impact hesaplama hatası: {e}")
            return 0.0
    
    def on_symbol_changed(self, event=None):
        """Sembol değiştiğinde"""
        try:
            self.current_symbol = self.symbol_var.get()
            self.load_sample_data()
            self.logger.info(f"Sembol değiştirildi: {self.current_symbol}")
            
        except Exception as e:
            self.logger.error(f"Sembol değiştirme hatası: {e}")
    
    def on_mode_changed(self, event=None):
        """Görüntüleme modu değiştiğinde"""
        try:
            mode_value = self.mode_var.get()
            self.display_mode = OrderBookDisplayMode(mode_value)
            self.update_chart()
            self.logger.info(f"Görüntüleme modu değiştirildi: {mode_value}")
            
        except Exception as e:
            self.logger.error(f"Mod değiştirme hatası: {e}")
    
    def refresh_data(self):
        """Veriyi yenile"""
        try:
            # Simüle edilmiş veri güncelleme
            self.load_sample_data()
            self.status_label.config(text="Veri yenilendi", foreground="blue")
            
        except Exception as e:
            self.logger.error(f"Veri yenileme hatası: {e}")
    
    def show_analytics(self):
        """Analiz penceresi göster"""
        try:
            analytics_window = tk.Toplevel(self.root)
            analytics_window.title("Detaylı Analiz")
            analytics_window.geometry("800x600")
            
            # Analiz içeriği
            analysis_text = tk.Text(analytics_window, wrap=tk.WORD, padx=10, pady=10)
            analysis_text.pack(fill=tk.BOTH, expand=True)
            
            # Analiz raporu
            report = self._generate_analysis_report()
            analysis_text.insert(tk.END, report)
            analysis_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Analiz penceresi hatası: {e}")
    
    def _generate_analysis_report(self) -> str:
        """Analiz raporu oluştur"""
        try:
            if not self.orderbook_data:
                return "Veri bulunamadı."
            
            latest_snapshot = self.orderbook_data[-1]
            
            report = f"""
ORDER BOOK ANALİZ RAPORU
========================
Sembol: {latest_snapshot.symbol}
Tarih: {latest_snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

TEMEL METRİKLER
---------------
Best Bid: {latest_snapshot.best_bid:.2f}
Best Ask: {latest_snapshot.best_ask:.2f}
Spread: {latest_snapshot.spread:.2f}
Mid Price: {latest_snapshot.mid_price:.2f}

VOLUME ANALİZİ
--------------
Toplam Bid Volume: {sum([bid.quantity for bid in latest_snapshot.bids[:10]]):.2f}
Toplam Ask Volume: {sum([ask.quantity for ask in latest_snapshot.asks[:10]]):.2f}
Volume Dengesizliği: {self._calculate_volume_imbalance(latest_snapshot):.3f}

DERİNLİK ANALİZİ
----------------
Toplam Seviye Sayısı: {len(latest_snapshot.bids) + len(latest_snapshot.asks)}
Ortalama Seviye Derinliği: {self._calculate_average_depth(latest_snapshot):.2f}
Maksimum Seviye Derinliği: {self._calculate_max_depth(latest_snapshot):.2f}

RİSK DEĞERLENDİRMESİ
--------------------
Spoofing Skoru: {self._calculate_spoofing_score(latest_snapshot):.3f}
Market Impact: {self._calculate_market_impact(latest_snapshot):.3f}
Likidite Skoru: {self._calculate_liquidity_score(latest_snapshot):.3f}

ÖNERİLER
--------
{self._generate_recommendations(latest_snapshot)}
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Analiz raporu oluşturma hatası: {e}")
            return "Rapor oluşturulamadı."
    
    def _calculate_volume_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """Volume dengesizliği hesapla"""
        try:
            bid_volume = sum([bid.quantity for bid in snapshot.bids[:10]])
            ask_volume = sum([ask.quantity for ask in snapshot.asks[:10]])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                return (bid_volume - ask_volume) / total_volume
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Volume dengesizliği hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_average_depth(self, snapshot: OrderBookSnapshot) -> float:
        """Ortalama derinlik hesapla"""
        try:
            all_quantities = [bid.quantity for bid in snapshot.bids[:10]] + [ask.quantity for ask in snapshot.asks[:10]]
            return np.mean(all_quantities) if all_quantities else 0.0
            
        except Exception as e:
            self.logger.error(f"Ortalama derinlik hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_max_depth(self, snapshot: OrderBookSnapshot) -> float:
        """Maksimum derinlik hesapla"""
        try:
            all_quantities = [bid.quantity for bid in snapshot.bids[:10]] + [ask.quantity for ask in snapshot.asks[:10]]
            return np.max(all_quantities) if all_quantities else 0.0
            
        except Exception as e:
            self.logger.error(f"Maksimum derinlik hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, snapshot: OrderBookSnapshot) -> float:
        """Likidite skoru hesapla"""
        try:
            # Toplam volume
            total_volume = sum([bid.quantity for bid in snapshot.bids[:10]]) + sum([ask.quantity for ask in snapshot.asks[:10]])
            
            # Seviye sayısı
            level_count = len(snapshot.bids[:10]) + len(snapshot.asks[:10])
            
            # Likidite skoru (0-1 arası)
            volume_score = min(total_volume / 100, 1.0)  # 100'e normalize et
            level_score = min(level_count / 20, 1.0)  # 20'ye normalize et
            
            return (volume_score + level_score) / 2
            
        except Exception as e:
            self.logger.error(f"Likidite skoru hesaplama hatası: {e}")
            return 0.0
    
    def _generate_recommendations(self, snapshot: OrderBookSnapshot) -> str:
        """Öneriler oluştur"""
        try:
            recommendations = []
            
            # Spread analizi
            if snapshot.spread > snapshot.mid_price * 0.005:  # %0.5'ten büyük spread
                recommendations.append("- Spread yüksek, dikkatli işlem yapın")
            
            # Volume dengesizliği
            imbalance = self._calculate_volume_imbalance(snapshot)
            if abs(imbalance) > 0.3:
                if imbalance > 0:
                    recommendations.append("- Bid volume fazla, fiyat yükselişi beklenebilir")
                else:
                    recommendations.append("- Ask volume fazla, fiyat düşüşü beklenebilir")
            
            # Spoofing uyarısı
            spoofing_score = self._calculate_spoofing_score(snapshot)
            if spoofing_score > 0.5:
                recommendations.append("- Spoofing riski yüksek, büyük emirlerden kaçının")
            
            # Likidite uyarısı
            liquidity_score = self._calculate_liquidity_score(snapshot)
            if liquidity_score < 0.3:
                recommendations.append("- Likidite düşük, küçük pozisyonlar tercih edin")
            
            if not recommendations:
                recommendations.append("- Order book sağlıklı görünüyor")
            
            return "\n".join(recommendations)
            
        except Exception as e:
            self.logger.error(f"Öneri oluşturma hatası: {e}")
            return "- Analiz tamamlanamadı"
    
    def export_data(self):
        """Veriyi export et"""
        try:
            if not self.orderbook_data:
                messagebox.showwarning("Uyarı", "Export edilecek veri bulunamadı")
                return
            
            # Export penceresi
            export_window = tk.Toplevel(self.root)
            export_window.title("Veri Export")
            export_window.geometry("400x300")
            
            # Export seçenekleri
            format_frame = ttk.LabelFrame(export_window, text="Format", padding=10)
            format_frame.pack(fill=tk.X, padx=10, pady=10)
            
            format_var = tk.StringVar(value="csv")
            ttk.Radiobutton(format_frame, text="CSV", variable=format_var, value="csv").pack(anchor=tk.W)
            ttk.Radiobutton(format_frame, text="JSON", variable=format_var, value="json").pack(anchor=tk.W)
            ttk.Radiobutton(format_frame, text="Excel", variable=format_var, value="excel").pack(anchor=tk.W)
            
            # Export butonu
            ttk.Button(export_window, text="Export", command=lambda: self._do_export(format_var.get())).pack(pady=10)
            
        except Exception as e:
            self.logger.error(f"Export penceresi hatası: {e}")
    
    def _do_export(self, format_type: str):
        """Export işlemini gerçekleştir"""
        try:
            filename = f"orderbook_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
            
            if format_type == "csv":
                self._export_to_csv(filename)
            elif format_type == "json":
                self._export_to_json(filename)
            elif format_type == "excel":
                self._export_to_excel(filename)
            
            messagebox.showinfo("Başarılı", f"Veri {filename} dosyasına export edildi")
            
        except Exception as e:
            self.logger.error(f"Export hatası: {e}")
            messagebox.showerror("Hata", f"Export hatası: {e}")
    
    def _export_to_csv(self, filename: str):
        """CSV export"""
        try:
            data = []
            for snapshot in self.orderbook_data:
                for bid in snapshot.bids[:10]:
                    data.append({
                        'timestamp': snapshot.timestamp,
                        'symbol': snapshot.symbol,
                        'side': 'BID',
                        'price': bid.price,
                        'quantity': bid.quantity
                    })
                for ask in snapshot.asks[:10]:
                    data.append({
                        'timestamp': snapshot.timestamp,
                        'symbol': snapshot.symbol,
                        'side': 'ASK',
                        'price': ask.price,
                        'quantity': ask.quantity
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
        except Exception as e:
            self.logger.error(f"CSV export hatası: {e}")
            raise
    
    def _export_to_json(self, filename: str):
        """JSON export"""
        try:
            data = []
            for snapshot in self.orderbook_data:
                snapshot_data = {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'symbol': snapshot.symbol,
                    'best_bid': snapshot.best_bid,
                    'best_ask': snapshot.best_ask,
                    'spread': snapshot.spread,
                    'mid_price': snapshot.mid_price,
                    'bids': [{'price': bid.price, 'quantity': bid.quantity} for bid in snapshot.bids[:10]],
                    'asks': [{'price': ask.price, 'quantity': ask.quantity} for ask in snapshot.asks[:10]]
                }
                data.append(snapshot_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"JSON export hatası: {e}")
            raise
    
    def _export_to_excel(self, filename: str):
        """Excel export"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Order book verisi
                data = []
                for snapshot in self.orderbook_data:
                    for bid in snapshot.bids[:10]:
                        data.append({
                            'timestamp': snapshot.timestamp,
                            'symbol': snapshot.symbol,
                            'side': 'BID',
                            'price': bid.price,
                            'quantity': bid.quantity
                        })
                    for ask in snapshot.asks[:10]:
                        data.append({
                            'timestamp': snapshot.timestamp,
                            'symbol': snapshot.symbol,
                            'side': 'ASK',
                            'price': ask.price,
                            'quantity': ask.quantity
                        })
                
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='OrderBook', index=False)
                
                # Analiz metrikleri
                metrics_data = []
                for snapshot in self.orderbook_data:
                    metrics_data.append({
                        'timestamp': snapshot.timestamp,
                        'symbol': snapshot.symbol,
                        'spread': snapshot.spread,
                        'mid_price': snapshot.mid_price,
                        'bid_volume': sum([bid.quantity for bid in snapshot.bids[:10]]),
                        'ask_volume': sum([ask.quantity for ask in snapshot.asks[:10]])
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
        except Exception as e:
            self.logger.error(f"Excel export hatası: {e}")
            raise
    
    def start_update_loop(self):
        """Güncelleme döngüsünü başlat"""
        try:
            def update_loop():
                while not self._stop_event.is_set():
                    try:
                        # Simüle edilmiş veri güncelleme
                        self._simulate_data_update()
                        
                        # GUI güncelleme
                        if self.root and self.root.winfo_exists():
                            self.root.after(0, self.update_display)
                        
                        time.sleep(5)  # 5 saniyede bir güncelle
                        
                    except Exception as e:
                        self.logger.error(f"Güncelleme döngüsü hatası: {e}")
                        time.sleep(10)
            
            # Thread başlat
            update_thread = threading.Thread(target=update_loop, daemon=True)
            update_thread.start()
            
        except Exception as e:
            self.logger.error(f"Güncelleme döngüsü başlatma hatası: {e}")

    def stop_update_loop(self):
        """Güncelleme döngüsünü durdur"""
        try:
            self._stop_event.set()
        except Exception:
            pass
    
    def _simulate_data_update(self):
        """Veri güncellemesini simüle et"""
        try:
            if not self.orderbook_data:
                return
            
            latest_snapshot = self.orderbook_data[-1]
            
            # Fiyatları rastgele güncelle
            price_change = np.random.normal(0, 0.001)  # %0.1 volatilite
            
            # Yeni snapshot oluştur
            new_bids = []
            new_asks = []
            
            base_price = latest_snapshot.mid_price * (1 + price_change)
            
            # BID seviyeleri
            for i in range(20):
                price = base_price - (i + 1) * 10
                quantity = max(0.1, latest_snapshot.bids[i].quantity + np.random.normal(0, 0.1))
                new_bids.append(OrderBookLevel(price, quantity, datetime.now()))
            
            # ASK seviyeleri
            for i in range(20):
                price = base_price + (i + 1) * 10
                quantity = max(0.1, latest_snapshot.asks[i].quantity + np.random.normal(0, 0.1))
                new_asks.append(OrderBookLevel(price, quantity, datetime.now()))
            
            # Yeni snapshot
            new_snapshot = OrderBookSnapshot(
                symbol=latest_snapshot.symbol,
                timestamp=datetime.now(),
                bids=new_bids,
                asks=new_asks,
                best_bid=new_bids[0].price if new_bids else base_price,
                best_ask=new_asks[0].price if new_asks else base_price,
                spread=(new_asks[0].price - new_bids[0].price) if new_asks and new_bids else 0,
                mid_price=(new_asks[0].price + new_bids[0].price) / 2 if new_asks and new_bids else base_price
            )
            
            self.orderbook_data.append(new_snapshot)
            
        except Exception as e:
            self.logger.error(f"Veri güncelleme simülasyon hatası: {e}")
    
    def add_orderbook_callback(self, callback):
        """Order book callback'i ekle"""
        self.orderbook_callbacks.append(callback)
    
    def run(self):
        """GUI'yi çalıştır"""
        try:
            if not self.parent:
                self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI çalıştırma hatası: {e}")
    
    def setup_gui(self):
        """GUI'yi kur"""
        try:
            if not self.main_frame:
                self.main_frame = ttk.Frame(self.parent)
                self.main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Order book görüntüleme alanı
            self.create_orderbook_display()
            
            # Kontrol paneli
            self.create_control_panel()
            
            # Analiz paneli
            self.create_analytics_panel()
            
            self.logger.info("OrderBookGUI kuruldu")
            
        except Exception as e:
            self.logger.error(f"GUI kurulum hatası: {e}")
    
    def create_orderbook_display(self):
        """Order book görüntüleme alanını oluştur"""
        try:
            # Order book frame
            self.orderbook_frame = ttk.LabelFrame(self.main_frame, text="Emir Defteri", padding=10)
            self.orderbook_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Treeview oluştur
            columns = ("Fiyat", "Miktar", "Toplam", "Tip")
            self.orderbook_tree = ttk.Treeview(self.orderbook_frame, columns=columns, show="headings", height=15)
            
            for col in columns:
                self.orderbook_tree.heading(col, text=col)
                self.orderbook_tree.column(col, width=120, anchor=tk.CENTER)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(self.orderbook_frame, orient=tk.VERTICAL, command=self.orderbook_tree.yview)
            self.orderbook_tree.configure(yscrollcommand=scrollbar.set)
            
            self.orderbook_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            self.logger.error(f"Order book görüntüleme oluşturma hatası: {e}")
    
    def create_control_panel(self):
        """Kontrol panelini oluştur"""
        try:
            self.control_frame = ttk.LabelFrame(self.main_frame, text="Kontroller", padding=10)
            self.control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Butonlar
            ttk.Button(self.control_frame, text="Verileri Yenile", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
            ttk.Button(self.control_frame, text="Analizi Göster", command=self.show_analytics).pack(side=tk.LEFT, padx=5)
            ttk.Button(self.control_frame, text="Verileri Dışa Aktar", command=self.export_data).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Kontrol paneli oluşturma hatası: {e}")
    
    def create_analytics_panel(self):
        """Analiz panelini oluştur"""
        try:
            self.analytics_frame = ttk.LabelFrame(self.main_frame, text="Analiz Metrikleri", padding=10)
            self.analytics_frame.pack(fill=tk.X)
            
            # Metrikler için frame
            metrics_frame = ttk.Frame(self.analytics_frame)
            metrics_frame.pack(fill=tk.X)
            
            # Derinlik dengesizliği
            ttk.Label(metrics_frame, text="Derinlik Dengesizliği:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5)
            self.depth_imbalance_var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=self.depth_imbalance_var, font=("Arial", 12)).grid(row=0, column=1, sticky=tk.W, padx=5)
            
            # Emir akışı
            ttk.Label(metrics_frame, text="Emir Akışı:", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5)
            self.order_flow_var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=self.order_flow_var, font=("Arial", 12)).grid(row=0, column=3, sticky=tk.W, padx=5)
            
            # Likidite skoru
            ttk.Label(metrics_frame, text="Likidite Skoru:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5)
            self.liquidity_score_var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=self.liquidity_score_var, font=("Arial", 12)).grid(row=1, column=1, sticky=tk.W, padx=5)
            
            # Piyasa etkisi
            ttk.Label(metrics_frame, text="Piyasa Etkisi:", font=("Arial", 10, "bold")).grid(row=1, column=2, sticky=tk.W, padx=5)
            self.market_impact_var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=self.market_impact_var, font=("Arial", 12)).grid(row=1, column=3, sticky=tk.W, padx=5)
            
            # Sahte emir tespiti
            ttk.Label(metrics_frame, text="Sahte Emir:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5)
            self.spoofing_var = tk.StringVar(value="Hayır")
            self.spoofing_label = ttk.Label(metrics_frame, textvariable=self.spoofing_var, font=("Arial", 12), foreground="green")
            self.spoofing_label.grid(row=2, column=1, sticky=tk.W, padx=5)
            
            # Balina aktivitesi
            ttk.Label(metrics_frame, text="Balina Aktivitesi:", font=("Arial", 10, "bold")).grid(row=2, column=2, sticky=tk.W, padx=5)
            self.whale_var = tk.StringVar(value="Hayır")
            self.whale_label = ttk.Label(metrics_frame, textvariable=self.whale_var, font=("Arial", 12), foreground="green")
            self.whale_label.grid(row=2, column=3, sticky=tk.W, padx=5)
            
        except Exception as e:
            self.logger.error(f"Analiz paneli oluşturma hatası: {e}")
    
    def update_orderbook_data(self, snapshot):
        """Order book verilerini güncelle"""
        try:
            # Snapshot'ı kuyruğa ekle ve tüm UI'yi tazele
            self.orderbook_data.append(snapshot)
            self.update_display()
        except Exception as e:
            self.logger.error(f"Order book veri güncelleme hatası: {e}")
    
    def update_imbalance_metrics(self, imbalance_data):
        """Imbalance metriklerini güncelle"""
        try:
            val = float(imbalance_data.get('imbalance', 0) or 0)
            if hasattr(self, 'depth_imbalance_var'):
                self.depth_imbalance_var.set(f"{val:.4f}")
            if hasattr(self, 'imbalance_label'):
                self.imbalance_label.config(text=f"{val:.3f}")
            if hasattr(self, 'order_flow_var'):
                self.order_flow_var.set(f"{float(imbalance_data.get('order_flow', 0) or 0):.4f}")
        except Exception as e:
            self.logger.error(f"Imbalance metrik güncelleme hatası: {e}")
    
    def update_spoofing_alert(self, spoofing_data):
        """Spoofing uyarısını güncelle"""
        try:
            if hasattr(self, 'spoofing_var'):
                is_spoofing = spoofing_data.get('is_spoofing', False)
                self.spoofing_var.set("Evet" if is_spoofing else "Hayır")
                if hasattr(self, 'spoofing_label'):
                    self.spoofing_label.config(foreground="red" if is_spoofing else "green")
        except Exception as e:
            self.logger.error(f"Spoofing uyarı güncelleme hatası: {e}")
    
    def update_whale_alert(self, whale_data):
        """Whale uyarısını güncelle"""
        try:
            if hasattr(self, 'whale_var'):
                is_whale = whale_data.get('is_whale_activity', False)
                self.whale_var.set("Evet" if is_whale else "Hayır")
                if hasattr(self, 'whale_label'):
                    self.whale_label.config(foreground="red" if is_whale else "green")
        except Exception as e:
            self.logger.error(f"Whale uyarı güncelleme hatası: {e}")
    
    def refresh_data(self):
        """Verileri yenile"""
        try:
            self.logger.info("Order book verileri yenileniyor...")
            # Veri yenileme işlemi
        except Exception as e:
            self.logger.error(f"Veri yenileme hatası: {e}")
    
    def show_analytics(self):
        """Analizi göster"""
        try:
            self.logger.info("Order book analizi gösteriliyor...")
            # Analiz gösterme işlemi
        except Exception as e:
            self.logger.error(f"Analiz gösterme hatası: {e}")
    
    def export_data(self):
        """Verileri dışa aktar"""
        try:
            self.logger.info("Order book verileri dışa aktarılıyor...")
            # Veri dışa aktarma işlemi
        except Exception as e:
            self.logger.error(f"Veri dışa aktarma hatası: {e}")

# Global Order Book GUI
orderbook_gui = OrderBookGUI()
