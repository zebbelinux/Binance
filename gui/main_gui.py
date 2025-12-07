"""
Ana Trading GUI
Trading bot'un ana arayüzü
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time

class MainTradingGUI:
    """Ana trading GUI sınıfı"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.root = tk.Tk()
        self.root.title("BTCTURK Trading Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Veri saklama
        self.price_data = {}
        self.analysis_data = {}
        self.trade_history = []
        
        # GUI elemanları
        self.status_label = None
        self.price_labels = {}
        self.analysis_labels = {}
        
        # Grafik
        self.fig = None
        self.ax = None
        self.canvas = None
        
        self.create_gui()
        self.start_data_refresh()
    
    def create_gui(self):
        """Ana GUI'yi oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Üst panel - Kontroller
        self.create_control_panel(main_frame)
        
        # Orta panel - Grafik ve Fiyatlar
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Sol panel - Fiyat ve Analiz
        left_panel = ttk.LabelFrame(middle_frame, text="Piyasa Verileri", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.create_price_panel(left_panel)
        self.create_analysis_panel(left_panel)
        
        # Sağ panel - Grafik
        right_panel = ttk.LabelFrame(middle_frame, text="Fiyat Grafiği", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.create_chart_panel(right_panel)
        
        # Alt panel - Trade Geçmişi ve Loglar
        self.create_bottom_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Kontrol panelini oluştur"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sol taraf - Durum ve Butonlar
        left_control = ttk.Frame(control_frame)
        left_control.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Durum etiketi
        self.status_label = ttk.Label(left_control, text="Hazır", font=("Arial", 12, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Kontrol butonları
        button_frame = ttk.Frame(left_control)
        button_frame.pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Trading Başlat", command=self.start_trading).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Trading Durdur", command=self.stop_trading).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="API Yönetimi", command=self.open_api_management).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Ayarlar", command=self.open_settings).pack(side=tk.LEFT)
        
        # Sağ taraf - Sistem bilgileri
        right_control = ttk.Frame(control_frame)
        right_control.pack(side=tk.RIGHT)
        
        self.system_info_label = ttk.Label(right_control, text="Sistem: Hazır")
        self.system_info_label.pack(side=tk.RIGHT)
    
    def create_price_panel(self, parent):
        """Fiyat panelini oluştur"""
        price_frame = ttk.LabelFrame(parent, text="Fiyat Bilgileri", padding=5)
        price_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Fiyat etiketleri
        symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        
        for i, symbol in enumerate(symbols):
            frame = ttk.Frame(price_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{symbol}:", width=10).pack(side=tk.LEFT)
            self.price_labels[symbol] = ttk.Label(frame, text="Yükleniyor...", width=15)
            self.price_labels[symbol].pack(side=tk.LEFT, padx=(5, 0))
    
    def create_analysis_panel(self, parent):
        """Analiz panelini oluştur"""
        analysis_frame = ttk.LabelFrame(parent, text="AI Analiz", padding=5)
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        
        # Analiz etiketleri
        analysis_items = [
            ("sentiment", "Duyarlılık:"),
            ("recommendation", "Tavsiye:"),
            ("risk_level", "Risk Seviyesi:"),
            ("combined_score", "Kombine Skor:")
        ]
        
        for key, label in analysis_items:
            frame = ttk.Frame(analysis_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
            self.analysis_labels[key] = ttk.Label(frame, text="Bekleniyor...", width=15)
            self.analysis_labels[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def create_chart_panel(self, parent):
        """Grafik panelini oluştur"""
        # Matplotlib figure oluştur
        self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        # Canvas oluştur
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Başlangıç grafiği
        self.update_chart()
    
    def create_bottom_panel(self, parent):
        """Alt paneli oluştur"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Sol - Trade geçmişi
        trade_frame = ttk.LabelFrame(bottom_frame, text="Trade Geçmişi", padding=5)
        trade_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Trade listesi
        columns = ("Zaman", "Sembol", "Tip", "Miktar", "Fiyat", "Durum")
        self.trade_tree = ttk.Treeview(trade_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        trade_scrollbar = ttk.Scrollbar(trade_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=trade_scrollbar.set)
        
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trade_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sağ - Loglar
        log_frame = ttk.LabelFrame(bottom_frame, text="Sistem Logları", padding=5)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Log metni
        self.log_text = tk.Text(log_frame, height=8, width=50, bg='#2b2b2b', fg='white', 
                               font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def start_trading(self):
        """Trading'i başlat"""
        self.bot.start_trading()
    
    def stop_trading(self):
        """Trading'i durdur"""
        self.bot.stop_trading()
    
    def open_api_management(self):
        """API yönetimi aç"""
        self.bot.open_api_management()
    
    def open_settings(self):
        """Ayarlar penceresi aç"""
        messagebox.showinfo("Bilgi", "Ayarlar penceresi yakında eklenecek")
    
    def update_status(self, status: str):
        """Durum etiketini güncelle"""
        if self.status_label:
            self.status_label.config(text=status)
    
    def update_analysis(self, analysis):
        """Analiz verilerini güncelle"""
        try:
            symbol = analysis.get('symbol', '')
            
            # Fiyat verilerini güncelle
            market_data = analysis.get('market_data', {})
            if market_data and symbol in self.price_labels:
                price = market_data.get('price', 0)
                change = market_data.get('change_24h', 0)
                price_text = f"₺{price:,.2f} ({change:+.2f}%)"
                self.price_labels[symbol].config(text=price_text)
            
            # Analiz verilerini güncelle
            ai_analysis = analysis.get('ai_analysis', {})
            if ai_analysis:
                # Duyarlılık
                sentiment = ai_analysis.get('sentiment', {})
                if sentiment:
                    sentiment_text = sentiment.get('sentiment', 'Bilinmiyor')
                    self.analysis_labels['sentiment'].config(text=sentiment_text)
                
                # Tavsiye
                signals = ai_analysis.get('signals', {})
                if signals:
                    recommendation = signals.get('signal', 'HOLD')
                    self.analysis_labels['recommendation'].config(text=recommendation)
            
            # Risk seviyesi
            risk_level = analysis.get('risk_level', 'Orta')
            self.analysis_labels['risk_level'].config(text=risk_level)
            
            # Kombine skor
            combined_score = analysis.get('combined_score', 0.5)
            score_text = f"{combined_score:.2f}"
            self.analysis_labels['combined_score'].config(text=score_text)
            
            # Grafiği güncelle
            self.update_chart()
            
        except Exception as e:
            print(f"Analiz güncelleme hatası: {e}")
    
    def update_chart(self):
        """Grafiği güncelle"""
        try:
            # Örnek veri oluştur (gerçek veri yerine)
            if not self.price_data:
                # Son 100 gün için örnek veri
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                prices = 100000 + np.cumsum(np.random.randn(100) * 1000)
                self.price_data = {'dates': dates, 'prices': prices}
            
            # Grafiği temizle
            self.ax.clear()
            
            # Fiyat çizgisi
            self.ax.plot(self.price_data['dates'], self.price_data['prices'], 
                        color='#00ff00', linewidth=2, label='BTC Fiyatı')
            
            # Grafik ayarları
            self.ax.set_title('BTC Fiyat Grafiği', color='white', fontsize=14)
            self.ax.set_xlabel('Tarih', color='white')
            self.ax.set_ylabel('Fiyat (₺)', color='white')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # X ekseni formatı
            self.ax.tick_params(axis='x', rotation=45)
            
            # Canvas'ı güncelle
            self.canvas.draw()
            
        except Exception as e:
            print(f"Grafik güncelleme hatası: {e}")
    
    def add_trade(self, symbol: str, trade_type: str, quantity: float, price: float, status: str):
        """Trade geçmişine ekle"""
        try:
            trade_data = (
                datetime.now().strftime("%H:%M:%S"),
                symbol,
                trade_type,
                f"{quantity:.4f}",
                f"₺{price:,.2f}",
                status
            )
            
            self.trade_tree.insert("", 0, values=trade_data)
            
            # Maksimum 100 satır tut
            items = self.trade_tree.get_children()
            if len(items) > 100:
                self.trade_tree.delete(items[-1])
                
        except Exception as e:
            print(f"Trade ekleme hatası: {e}")
    
    def add_log(self, message: str):
        """Log ekle"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
            
            # Maksimum 1000 satır tut
            lines = self.log_text.get("1.0", tk.END).split('\n')
            if len(lines) > 1000:
                self.log_text.delete("1.0", f"{len(lines)-1000}.0")
                
        except Exception as e:
            print(f"Log ekleme hatası: {e}")
    
    def start_data_refresh(self):
        """Veri yenileme thread'ini başlat"""
        def refresh_loop():
            while True:
                try:
                    # Fiyat verilerini güncelle
                    self.update_price_data()
                    
                    # Sistem durumunu güncelle
                    self.update_system_status()
                    
                    time.sleep(5)  # 5 saniyede bir güncelle
                    
                except Exception as e:
                    print(f"Veri yenileme hatası: {e}")
                    time.sleep(10)
        
        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()
    
    def update_price_data(self):
        """Fiyat verilerini güncelle"""
        try:
            # Bu fonksiyon gerçek API çağrıları yapacak
            # Şimdilik örnek veri
            pass
            
        except Exception as e:
            print(f"Fiyat güncelleme hatası: {e}")
    
    def update_system_status(self):
        """Sistem durumunu güncelle"""
        try:
            status = self.bot.get_system_status()
            
            if 'api_status' in status:
                api_status = status['api_status']
                active_apis = api_status.get('active_apis', 0)
                success_rate = api_status.get('success_rate', 0)
                
                status_text = f"Aktif API: {active_apis} | Başarı: {success_rate:.1f}%"
                self.system_info_label.config(text=status_text)
                
        except Exception as e:
            print(f"Sistem durumu güncelleme hatası: {e}")
    
    def run(self):
        """GUI'yi çalıştır"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI çalıştırma hatası: {e}")

if __name__ == "__main__":
    # Test için
    class MockBot:
        def start_trading(self): pass
        def stop_trading(self): pass
        def open_api_management(self): pass
        def get_system_status(self): return {}
    
    app = MainTradingGUI(MockBot())
    app.run()
