"""
BTCTURK Trading Bot - Ana Uygulama
Kapsamlı kripto para trading botu
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import ctypes
import logging
from datetime import datetime

# Proje modüllerini import et
from gui.trading_dashboard import TradingDashboard
from api.multi_api_manager import multi_api_manager
from ai.deepseek_api import DeepSeekAPI
from ai.market_analyzer import market_analyzer
from ai.fundamental_analysis import FundamentalAnalyzer
from ai.advanced_ml_models import AdvancedMLModels
from ai.ml_signal_calibration import MLSignalCalibration
from indicators.orderbook_analyzer import OrderBookAnalyzer
from optimization.parameter_optimizer import ParameterOptimizer
from risk_management.portfolio_optimizer import PortfolioOptimizer
from data.external_data_manager import ExternalDataManager
from data.tick_data_recorder import TickDataRecorder
from utils.notification_system import NotificationSystem
from utils.auto_restart_manager import AutoRestartManager
from utils.logger import get_logger, LogCategory, log_manager
from utils.error_handler import error_handler, handle_error, ErrorCategory, ErrorSeverity
from utils.config_validator import config_validator, ConfigType
from data.data_manager import data_manager
from risk_management.risk_manager import risk_manager
from plugins.plugin_manager import plugin_manager
from strategies.strategy_manager import strategy_manager

# Environment variables yükle (opsiyonel)
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # .env dosyasini main.py'nin bulundugu klasorden yukle
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH)
except Exception as e:
    get_logger("main").warning(LogCategory.SYSTEM, f".env yükleme hatası veya dotenv import edilemedi: {e}")
# Basit bir bilgi logu (maskeli)
try:
    _k = os.getenv("DEEPSEEK_API_KEY")
    if _k:
        _masked = _k[:6] + "***" + _k[-4:]
        print(f"[ENV] DEEPSEEK_API_KEY yüklendi: {_masked}")
    else:
        print("[ENV] DEEPSEEK_API_KEY bulunamadı. .env dosyasını kontrol edin.")
except Exception as e:
    get_logger("main").warning(LogCategory.SYSTEM, f"ENV anahtar kontrolünde hata: {e}")

class TradingBotApp:
    """Ana trading bot uygulaması"""
    
    def __init__(self):
        self.logger = get_logger("main_app")
        self.is_running = False
        self._init_done = None
        
        # Ağır başlatmaları GUI'den sonra arka planda yapacağız
        self.logger.info(LogCategory.SYSTEM, "Trading bot uygulaması başlatıldı")
    
    def initialize_components(self):
        """Bileşenleri başlat"""
        try:
            import threading
            if self._init_done is None:
                self._init_done = threading.Event()
            # Logging sistemini başlat
            self.setup_logging()
            
            # Hata yönetimini başlat
            self.setup_error_handling()
            
            # Konfigürasyon validasyonu
            self.validate_configurations()
            
            # Yeni modülleri başlat
            self.setup_new_modules()
            
            # Veri yöneticisini başlat
            data_manager.start_data_collection()
            
            # Risk yöneticisini başlat
            risk_manager.start_monitoring()
            
            # Eklenti sistemini başlat
            plugin_manager.load_plugins_from_directory()
            
            self.logger.info(LogCategory.SYSTEM, "Tüm bileşenler başlatıldı")
            try:
                # Başlatma tamamlandı sinyali
                self._init_done.set()
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Bileşen başlatma hatası: {e}")
            raise
    
    def validate_configurations(self):
        """Konfigürasyon dosyalarını doğrula"""
        try:
            self.logger.info(LogCategory.SYSTEM, "Konfigürasyon validasyonu başlatılıyor...")
            
            # Yazılabilir yerel konfigürasyon dizinini tercih et
            cfg_dir = os.path.join(os.path.dirname(__file__), 'config_local')
            try:
                os.makedirs(cfg_dir, exist_ok=True)
            except Exception as e:
                self.logger.warning(LogCategory.SYSTEM, f"config_local oluşturulamadı, varsayılan 'config' kullanılacak: {e}")
                cfg_dir = os.path.join(os.path.dirname(__file__), 'config')

            # Eksikse varsayılan settings ve risk dosyalarını oluştur (api_keys kullanıcıya bırakılır)
            try:
                settings_path = os.path.join(cfg_dir, 'settings.json')
                risk_path = os.path.join(cfg_dir, 'risk_settings.json')
                if not os.path.exists(settings_path):
                    config_validator.create_default_config(ConfigType.SETTINGS, settings_path)
                if not os.path.exists(risk_path):
                    config_validator.create_default_config(ConfigType.RISK, risk_path)
            except Exception as e:
                self.logger.warning(LogCategory.SYSTEM, f"Varsayılan konfigürasyonlar oluşturulamadı: {e}")

            # Tüm konfigürasyonları doğrula (yerel dizinden)
            validation_results = config_validator.validate_all_configs(cfg_dir)
            
            has_errors = False
            for filename, result in validation_results.items():
                if not result['valid']:
                    has_errors = True
                    self.logger.error(LogCategory.SYSTEM, f"Konfigürasyon hatası ({filename}): {result['errors']}")
                else:
                    self.logger.info(LogCategory.SYSTEM, f"Konfigürasyon doğrulandı: {filename}")
                
                # Uyarıları göster
                for warning in result.get('warnings', []):
                    self.logger.warning(LogCategory.SYSTEM, f"Konfigürasyon uyarısı ({filename}): {warning}")
            
            if has_errors:
                self.logger.error(LogCategory.SYSTEM, "Konfigürasyon hataları tespit edildi, varsayılan değerler kullanılacak")
            else:
                self.logger.info(LogCategory.SYSTEM, "Tüm konfigürasyonlar geçerli")
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Konfigürasyon validasyon hatası: {e}")
            # Kritik değil, devam et
    
    def setup_new_modules(self):
        """Yeni modülleri başlat"""
        try:
            # Temel analiz modülü
            self.fundamental_analyzer = FundamentalAnalyzer()
            
            # Gelişmiş ML modelleri
            self.advanced_ml = AdvancedMLModels()
            
            # ML sinyal kalibrasyonu
            self.ml_signal_calibration = MLSignalCalibration()
            
            # Order book analizi
            self.orderbook_analyzer = OrderBookAnalyzer()
            
            # Parametre optimizasyonu
            self.parameter_optimizer = ParameterOptimizer()
            
            # Portföy optimizasyonu
            self.portfolio_optimizer = PortfolioOptimizer()
            
            # Harici veri yöneticisi
            self.external_data_manager = ExternalDataManager()
            
            # Tick veri kaydedici
            self.tick_data_recorder = TickDataRecorder()
            
            # Bildirim sistemi
            self.notification_system = NotificationSystem()
            
            # Otomatik yeniden başlatma
            self.auto_restart_manager = AutoRestartManager()
            
            self.logger.info(LogCategory.SYSTEM, "Yeni modüller başlatıldı")
            
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Yeni modül başlatma hatası: {e}")
            raise
    
    def setup_logging(self):
        """Logging sistemini kur"""
        try:
            # Ana logger'ı al
            main_logger = get_logger("main")
            
            # Log seviyesini ayarla
            main_logger.logger.setLevel(logging.INFO)
            
            # Sistem başlatma logu
            main_logger.info(LogCategory.SYSTEM, "Logging sistemi başlatıldı")
            
        except Exception as e:
            print(f"Logging kurulum hatası: {e}")
    
    def setup_error_handling(self):
        """Hata yönetimini kur"""
        try:
            # Hata kuralları ekle
            error_handler.add_error_rule("critical_errors", {
                "min_severity": 4,  # CRITICAL
                "max_frequency": 3,
                "time_window": 300,  # 5 dakika
                "actions": [
                    {"type": "log", "message": "Kritik hata tespit edildi"},
                    {"type": "callback", "callback": self.handle_critical_error}
                ]
            })
            
            # Hata callback'i ekle
            error_handler.add_error_callback(self.on_error_occurred)
            
            self.logger.info(LogCategory.SYSTEM, "Hata yönetimi başlatıldı")
            
        except Exception as e:
            print(f"Hata yönetimi kurulum hatası: {e}")
    
    def handle_critical_error(self, error_info, rule):
        """Kritik hata işle"""
        try:
            self.logger.critical(LogCategory.SYSTEM, f"Kritik hata: {error_info.error_message}")
            
            # Bot'u güvenli şekilde durdur
            self.shutdown()
            
        except Exception as e:
            print(f"Kritik hata işleme hatası: {e}")
    
    def on_error_occurred(self, error_info):
        """Hata oluştuğunda çağrılır"""
        try:
            # Hata logla
            self.logger.error(LogCategory.SYSTEM, f"Hata: {error_info.error_message}")
            
        except Exception as e:
            print(f"Hata callback hatası: {e}")
    
    def run(self):
        """Uygulamayı çalıştır"""
        try:
            # Ana pencere oluştur
            root = tk.Tk()
            
            # Pencere ayarları
            root.title("Binance Spot (Paper) Trading Bot")
            root.geometry("1400x900")
            root.minsize(1200, 800)
            
            # Bileşenleri başlat
            deepseek_api = DeepSeekAPI()
            # Global market_analyzer örneğini kullan ve BTCUSDT için analiz döngüsünü başlat
            try:
                market_analyzer.start_analysis(["BTCUSDT"])
            except Exception as e:
                get_logger("main").error(LogCategory.SYSTEM, f"MarketAnalyzer start_analysis hatası: {e}")
            # Global singleton multi_api_manager kullan
            try:
                # Kayitli API anahtarlarini yukle
                multi_api_manager.load_config()
            except Exception as e:
                get_logger("main").error(LogCategory.SYSTEM, f"API anahtarları yüklenemedi: {e}")
            
            # Basic mod seçeneği (adım adım başlatma)
            basic_mode = False

            # Arka plan init işlevi
            import threading
            if self._init_done is None:
                self._init_done = threading.Event()
            def _init_then_start():
                try:
                    self.initialize_components()
                    # Başlatma tamamlandıktan sonra stratejileri arka planda başlat
                    def _start_strategies_bg():
                        try:
                            strategy_manager.start_all_strategies()
                        except Exception as e:
                            get_logger("main").error(LogCategory.SYSTEM, f"Strateji baslatma hatasi: {e}")
                    threading.Thread(target=_start_strategies_bg, daemon=True).start()
                except Exception as e:
                    get_logger("main").error(LogCategory.SYSTEM, f"Arka plan init hatası: {e}")

            def _step_setup():
                try:
                    self.setup_logging()
                    self.setup_error_handling()
                    self.validate_configurations()
                except Exception as e:
                    get_logger("main").error(LogCategory.SYSTEM, f"Init-Setup hatası: {e}")
            def _step_modules():
                try:
                    self.setup_new_modules()
                except Exception as e:
                    get_logger("main").error(LogCategory.SYSTEM, f"Init-Modules hatası: {e}")
            def _step_services():
                try:
                    data_manager.start_data_collection()
                    risk_manager.start_monitoring()
                except Exception as e:
                    get_logger("main").error(LogCategory.SYSTEM, f"Init-Services hatası: {e}")
            def _step_plugins():
                try:
                    plugin_manager.load_plugins_from_directory()
                except Exception as e:
                    get_logger("main").error(LogCategory.SYSTEM, f"Init-Plugins hatası: {e}")
            _steps = {
                'setup': lambda: threading.Thread(target=_step_setup, daemon=True).start(),
                'modules': lambda: threading.Thread(target=_step_modules, daemon=True).start(),
                'services': lambda: threading.Thread(target=_step_services, daemon=True).start(),
                'plugins': lambda: threading.Thread(target=_step_plugins, daemon=True).start(),
            }
            dashboard = TradingDashboard(root, multi_api_manager, market_analyzer, basic_mode=basic_mode, init_callback=lambda: threading.Thread(target=_init_then_start, daemon=True).start(), init_steps=_steps)
            # Pencereyi görünür kıl (bazı sistemlerde arka planda kalabilir)
            try:
                root.update_idletasks()
                root.deiconify()
                root.lift()
                root.attributes('-topmost', True)
                root.after(100, lambda: root.attributes('-topmost', False))
            except Exception as e:
                get_logger("main").warning(LogCategory.GUI, f"Pencere ön plana getirilemedi: {e}")
            
            # API anahtarlarını kontrol et (pencere görünür olduktan sonra göster)
            if not multi_api_manager.api_keys:
                try:
                    root.after(200, lambda: messagebox.showwarning(
                        "API Anahtarı (Opsiyonel)",
                        "Paper modda gerçek emir gönderilmez; API anahtarı zorunlu değildir.\n\nBinance gerçek/Testnet erişimi için Ayarlar > API Yönetimi bölümünden anahtar ekleyebilirsiniz."
                    ))
                except Exception as e:
                    get_logger("main").warning(LogCategory.GUI, f"API anahtarı uyarısı gösterilemedi: {e}")
            # Ağır başlatmaları arka planda otomatik başlat: basic_mode değilse
            if not basic_mode:
                threading.Thread(target=_init_then_start, daemon=True).start()
            
            # Uygulamayı çalıştır
            self.is_running = True
            self.logger.info(LogCategory.SYSTEM, "Trading bot başlatıldı")
            
            try:
                root.mainloop()
            except KeyboardInterrupt:
                self.logger.info(LogCategory.SYSTEM, "Kullanıcı tarafından durduruldu")
            finally:
                self.shutdown()
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Uygulama çalıştırma hatası: {e}")
            handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL)
    
    def shutdown(self):
        """Uygulamayı güvenli şekilde kapat"""
        try:
            if not self.is_running:
                return
            
            self.logger.info(LogCategory.SYSTEM, "Uygulama kapatılıyor...")
            
            # Bot'u durdur
            self.is_running = False
            
            # Bileşenleri kapat
            try:
                risk_manager.stop_monitoring()
                data_manager.stop_data_collection()
                plugin_manager.cleanup_all_plugins()
            except Exception as e:
                self.logger.error(LogCategory.SYSTEM, f"Bileşen kapatma hatası: {e}")
            
            # Veritabanı bağlantılarını kapat
            try:
                data_manager.close()
            except Exception as e:
                self.logger.error(LogCategory.SYSTEM, f"Veritabanı kapatma hatası: {e}")
            
            self.logger.info(LogCategory.SYSTEM, "Uygulama başarıyla kapatıldı")
            
        except Exception as e:
            print(f"Uygulama kapatma hatası: {e}")

def main():
    """Ana fonksiyon"""
    try:
        # Windows CMD başlığını ayarla
        try:
            if sys.platform.startswith('win'):
                try:
                    ctypes.windll.kernel32.SetConsoleTitleW("BINANCE")
                except Exception:
                    os.system("title BINANCE")
        except Exception:
            pass
        # Uygulama oluştur ve çalıştır
        app = TradingBotApp()
        app.run()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            # GUI üzerinden de göster
            messagebox.showerror("Başlatma Hatası", f"Uygulama başlatılamadı:\n{e}")
        except Exception as mb_e:
            get_logger("main").warning(LogCategory.GUI, f"Hata mesajı gösterilemedi: {mb_e}")
        try:
            # Konsol hemen kapanmasın
            input("\n[HATA] Kapatmadan önce Enter'a basın...")
        except Exception as in_e:
            get_logger("main").warning(LogCategory.SYSTEM, f"Hata bekleme girişi başarısız: {in_e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Çalıştırma Hatası", str(e))
        except Exception as mb_e:
            get_logger("main").warning(LogCategory.GUI, f"Çalıştırma hatası mesajı gösterilemedi: {mb_e}")
        try:
            input("\n[HATA] Kapatmadan önce Enter'a basın...")
        except Exception as in_e:
            get_logger("main").warning(LogCategory.SYSTEM, f"Çalıştırma hatası bekleme girişi başarısız: {in_e}")