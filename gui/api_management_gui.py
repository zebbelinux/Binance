"""
API Yönetimi GUI
Çoklu API key yönetimi için arayüz
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import threading
from typing import Dict, List, Any
from datetime import datetime
from utils.logger import get_logger, LogCategory

from api.multi_api_manager import multi_api_manager, APIStatus

class APIManagementGUI:
    """API Yönetimi GUI sınıfı"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.root = None
        self.tree = None
        self.status_frame = None
        self.control_frame = None
        self._stop_event = threading.Event()
        
        # GUI elemanları
        self.name_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.secret_key_var = tk.StringVar()
        self.daily_limit_var = tk.StringVar(value="1000")
        self.max_errors_var = tk.StringVar(value="5")
        
        self.load_balancing_var = tk.StringVar(value="round_robin")
        self.failover_var = tk.BooleanVar(value=True)
        self.rate_limit_var = tk.DoubleVar(value=0.8)
        
        self.logger = get_logger("api_management_gui")
        self.create_gui()
        self.load_api_keys()
        self.start_status_update()
    
    def create_gui(self):
        """GUI oluştur"""
        if self.parent:
            self.root = tk.Toplevel(self.parent)
        else:
            self.root = tk.Tk()
        
        self.root.title("BTCTURK API Yönetimi")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        # Pencere kapatma güvenli kapatma
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Ana frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - API Key Yönetimi
        left_frame = ttk.LabelFrame(main_frame, text="API Key Yönetimi", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # API Key Ekleme Formu
        form_frame = ttk.Frame(left_frame)
        form_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(form_frame, text="API Key Adı:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(form_frame, textvariable=self.name_var, width=20).grid(row=0, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(form_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(form_frame, textvariable=self.api_key_var, width=40, show="*").grid(row=1, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(form_frame, text="Secret Key:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(form_frame, textvariable=self.secret_key_var, width=40, show="*").grid(row=2, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(form_frame, text="Günlük Limit:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(form_frame, textvariable=self.daily_limit_var, width=10).grid(row=3, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(form_frame, text="Max Hata:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(form_frame, textvariable=self.max_errors_var, width=10).grid(row=4, column=1, padx=(5, 0), pady=2)
        
        # Butonlar
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="API Key Ekle", command=self.add_api_key).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Seçiliyi Güncelle", command=self.update_api_key).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Seçiliyi Sil", command=self.delete_api_key).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Temizle", command=self.clear_form).pack(side=tk.LEFT)
        
        # API Key Listesi
        list_frame = ttk.LabelFrame(left_frame, text="Mevcut API Key'ler", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        columns = ("name", "status", "requests", "limit", "usage", "errors", "last_used")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        # Sütun başlıkları
        self.tree.heading("name", text="Ad")
        self.tree.heading("status", text="Durum")
        self.tree.heading("requests", text="İstekler")
        self.tree.heading("limit", text="Limit")
        self.tree.heading("usage", text="Kullanım %")
        self.tree.heading("errors", text="Hatalar")
        self.tree.heading("last_used", text="Son Kullanım")
        
        # Sütun genişlikleri
        self.tree.column("name", width=100)
        self.tree.column("status", width=80)
        self.tree.column("requests", width=80)
        self.tree.column("limit", width=80)
        self.tree.column("usage", width=80)
        self.tree.column("errors", width=60)
        self.tree.column("last_used", width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Seçim olayı
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        
        # Sağ panel - Ayarlar ve Durum
        right_frame = ttk.LabelFrame(main_frame, text="Ayarlar ve Durum", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Load Balancing Ayarları
        lb_frame = ttk.LabelFrame(right_frame, text="Load Balancing", padding=5)
        lb_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(lb_frame, text="Strateji:").pack(anchor=tk.W)
        strategy_combo = ttk.Combobox(lb_frame, textvariable=self.load_balancing_var, 
                                    values=["round_robin", "random", "least_used"], state="readonly")
        strategy_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(lb_frame, text="Failover Etkin", variable=self.failover_var).pack(anchor=tk.W)
        
        ttk.Label(lb_frame, text="Rate Limit Buffer:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(lb_frame, from_=0.1, to=1.0, variable=self.rate_limit_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        ttk.Button(lb_frame, text="Ayarları Kaydet", command=self.save_settings).pack(fill=tk.X, pady=(10, 0))
        
        # Durum Bilgileri
        self.status_frame = ttk.LabelFrame(right_frame, text="Sistem Durumu", padding=5)
        self.status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_labels = {}
        status_items = [
            ("total_apis", "Toplam API:"),
            ("active_apis", "Aktif API:"),
            ("rate_limited_apis", "Rate Limited:"),
            ("error_apis", "Hatalı API:"),
            ("total_requests", "Toplam İstek:"),
            ("success_rate", "Başarı Oranı:"),
            ("api_switches", "API Geçişleri:")
        ]
        
        for key, label in status_items:
            frame = ttk.Frame(self.status_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            self.status_labels[key] = ttk.Label(frame, text="0")
            self.status_labels[key].pack(side=tk.RIGHT)
        
        # Kontrol Butonları
        self.control_frame = ttk.LabelFrame(right_frame, text="Kontrol", padding=5)
        self.control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(self.control_frame, text="Durumu Yenile", command=self.refresh_status).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Sayaçları Sıfırla", command=self.reset_counters).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Konfigürasyonu Kaydet", command=self.save_config).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Konfigürasyonu Yükle", command=self.load_config).pack(fill=tk.X, pady=2)
    
    def add_api_key(self):
        """API key ekle"""
        name = self.name_var.get().strip()
        api_key = self.api_key_var.get().strip()
        secret_key = self.secret_key_var.get().strip()
        
        if not all([name, api_key, secret_key]):
            messagebox.showerror("Hata", "Tüm alanları doldurun")
            return
        
        try:
            daily_limit = int(self.daily_limit_var.get())
            max_errors = int(self.max_errors_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Günlük limit ve max hata sayısal olmalı")
            return
        
        if multi_api_manager.add_api_key(name, api_key, secret_key, daily_limit, max_errors):
            messagebox.showinfo("Başarılı", f"API key '{name}' eklendi")
            # Otomatik kaydet
            try:
                multi_api_manager.save_config()
            except Exception as e:
                self.logger.warning(LogCategory.SYSTEM, f"API key kaydetme hatası: {e}")
            self.clear_form()
            self.refresh_api_list()
        else:
            messagebox.showerror("Hata", "API key eklenemedi")
    
    def update_api_key(self):
        """Seçili API key'i güncelle"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Uyarı", "Güncellenecek API key seçin")
            return
        # Seçili kaydın adı
        item = self.tree.item(selection[0])
        name = item['values'][0]
        # Form verileri
        api_key = (self.api_key_var.get() or '').strip()
        secret_key = (self.secret_key_var.get() or '').strip()
        try:
            daily_limit = int((self.daily_limit_var.get() or '0').strip())
        except ValueError:
            messagebox.showerror("Hata", "Günlük limit sayısal olmalı")
            return
        try:
            max_errors = int((self.max_errors_var.get() or '0').strip())
        except ValueError:
            messagebox.showerror("Hata", "Max hata sayısal olmalı")
            return

        ok = multi_api_manager.update_api_key(
            name=name,
            api_key=api_key if api_key and api_key != '***' else None,
            secret_key=secret_key if secret_key and secret_key != '***' else None,
            daily_limit=daily_limit,
            max_errors=max_errors
        )
        if ok:
            try:
                multi_api_manager.save_config()
            except Exception as e:
                self.logger.warning(LogCategory.SYSTEM, f"API key güncelleme kaydetme hatası: {e}")
            messagebox.showinfo("Başarılı", "API key güncellendi")
            self.refresh_api_list()
        else:
            messagebox.showerror("Hata", "API key güncellenemedi")
    
    def delete_api_key(self):
        """Seçili API key'i sil"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Uyarı", "Silinecek API key seçin")
            return
        
        item = self.tree.item(selection[0])
        name = item['values'][0]
        
        if messagebox.askyesno("Onay", f"API key '{name}' silinsin mi?"):
            if multi_api_manager.remove_api_key(name):
                messagebox.showinfo("Başarılı", f"API key '{name}' silindi")
                # Otomatik kaydet
                try:
                    multi_api_manager.save_config()
                except Exception as e:
                    self.logger.warning(LogCategory.SYSTEM, f"API key silme kaydetme hatası: {e}")
                self.clear_form()
                self.refresh_api_list()
            else:
                messagebox.showerror("Hata", "API key silinemedi")
    
    def clear_form(self):
        """Formu temizle"""
        self.name_var.set("")
        self.api_key_var.set("")
        self.secret_key_var.set("")
        self.daily_limit_var.set("1000")
        self.max_errors_var.set("5")
    
    def on_select(self, event):
        """API key seçildiğinde formu doldur"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        values = item['values']
        
        # API key bilgilerini formda göster (güvenlik için key'ler gizli)
        self.name_var.set(values[0])
        self.api_key_var.set("***")
        self.secret_key_var.set("***")
        self.daily_limit_var.set(values[3])
        self.max_errors_var.set(values[5])
    
    def refresh_api_list(self):
        """API listesini yenile"""
        # Mevcut öğeleri temizle
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # API durumunu al
        status = multi_api_manager.get_api_status()
        
        # Her API key için satır ekle
        for api_detail in status.get("api_details", []):
            values = (
                api_detail["name"],
                api_detail["status"],
                api_detail["daily_requests"],
                api_detail["daily_limit"],
                f"{api_detail['usage_percentage']:.1f}%",
                api_detail["error_count"],
                api_detail["last_used"][:19] if api_detail["last_used"] else "Hiç"
            )
            self.tree.insert("", tk.END, values=values)
    
    def refresh_status(self):
        """Durum bilgilerini yenile"""
        try:
            if not self.root or not self.root.winfo_exists():
                return
            status = multi_api_manager.get_api_status()
            
            # Etiketler mevcut mu kontrol et
            keys = [
                "total_apis","active_apis","rate_limited_apis","error_apis",
                "total_requests","success_rate","api_switches"
            ]
            for k in keys:
                lbl = self.status_labels.get(k)
                if not lbl or not lbl.winfo_exists():
                    return
            
            self.status_labels["total_apis"].config(text=str(status["total_apis"]))
            self.status_labels["active_apis"].config(text=str(status["active_apis"]))
            self.status_labels["rate_limited_apis"].config(text=str(status["rate_limited_apis"]))
            self.status_labels["error_apis"].config(text=str(status["error_apis"]))
            self.status_labels["total_requests"].config(text=str(status["total_requests"]))
            self.status_labels["success_rate"].config(text=f"{status['success_rate']:.1f}%")
            self.status_labels["api_switches"].config(text=str(status["api_switches"]))
            
            self.refresh_api_list()
        except tk.TclError:
            # Widgetler yok edilmis olabilir; sessizce cik
            return
    
    def reset_counters(self):
        """Sayaçları sıfırla"""
        if messagebox.askyesno("Onay", "Tüm sayaçlar sıfırlansın mı?"):
            multi_api_manager.reset_daily_counters()
            messagebox.showinfo("Başarılı", "Sayaçlar sıfırlandı")
            self.refresh_status()
    
    def save_settings(self):
        """Ayarları kaydet"""
        multi_api_manager.set_load_balancing_strategy(self.load_balancing_var.get())
        multi_api_manager.enable_failover(self.failover_var.get())
        multi_api_manager.rate_limit_buffer = self.rate_limit_var.get()
        
        messagebox.showinfo("Başarılı", "Ayarlar kaydedildi")
    
    def save_config(self):
        """Konfigürasyonu kaydet"""
        multi_api_manager.save_config()
        messagebox.showinfo("Başarılı", "Konfigürasyon kaydedildi")
    
    def load_config(self):
        """Konfigürasyonu yükle"""
        multi_api_manager.load_config()
        messagebox.showinfo("Başarılı", "Konfigürasyon yüklendi")
        self.refresh_status()
    
    def load_api_keys(self):
        """API key'leri yükle"""
        multi_api_manager.load_config()
        self.refresh_status()
    
    def start_status_update(self):
        """Durum güncelleme başlat"""
        def update_loop():
            while not self._stop_event.is_set():
                try:
                    if self.root and self.root.winfo_exists():
                        self.root.after(0, self.refresh_status)
                    else:
                        break
                    threading.Event().wait(5)  # 5 saniyede bir güncelle
                except Exception as e:
                    self.logger.warning(LogCategory.SYSTEM, f"Durum döngüsü hatası: {e}")
                    break
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def run(self):
        """GUI'yi çalıştır"""
        self.root.mainloop()

    def _on_close(self):
        """Pencere kapatılırken arka plan güncellemeyi durdur ve pencereyi yok et"""
        try:
            # Son durumu kaydet
            try:
                multi_api_manager.save_config()
            except Exception as e:
                self.logger.warning(LogCategory.SYSTEM, f"Kapanışta kaydetme hatası: {e}")
            self._stop_event.set()
        finally:
            if self.root and self.root.winfo_exists():
                self.root.destroy()

if __name__ == "__main__":
    app = APIManagementGUI()
    app.run()
