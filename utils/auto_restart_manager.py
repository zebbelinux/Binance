"""
Otomatik Yeniden Başlatma Modülü
Sistem hatalarında otomatik yeniden başlatma ve sağlık kontrolü
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
import json
import sqlite3
import requests
from pathlib import Path
import platform
import zipfile
import glob
import warnings
warnings.filterwarnings('ignore')
import pickle

@dataclass
class HealthCheck:
    """Sağlık kontrolü"""
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    timestamp: datetime
    response_time: float = 0.0
    last_error: str = None

@dataclass
class RestartConfig:
    """Yeniden başlatma konfigürasyonu"""
    enabled: bool = True
    max_restarts_per_hour: int = 3
    max_restarts_per_day: int = 10
    restart_delay_seconds: int = 30
    health_check_interval: int = 60
    critical_components: List[str] = None
    restart_triggers: List[str] = None
    backup_before_restart: bool = True
    notify_on_restart: bool = True

@dataclass
class RestartEvent:
    """Yeniden başlatma olayı"""
    timestamp: datetime
    reason: str
    component: str
    restart_count: int
    success: bool
    error_message: str = None

class ComponentStatus(Enum):
    """Bileşen durumları"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    STOPPED = "stopped"

class RestartReason(Enum):
    """Yeniden başlatma nedenleri"""
    HEALTH_CHECK_FAILED = "health_check_failed"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    CPU_LIMIT_EXCEEDED = "cpu_limit_exceeded"
    CONNECTION_LOST = "connection_lost"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    MANUAL_RESTART = "manual_restart"
    SCHEDULED_RESTART = "scheduled_restart"
    CONFIG_CHANGE = "config_change"

class AutoRestartManager:
    """Otomatik yeniden başlatma yöneticisi"""
    
    def __init__(self, config: RestartConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or RestartConfig()
        
        # Sağlık kontrolü geçmişi
        self.health_history = deque(maxlen=1000)
        self.restart_history = deque(maxlen=100)
        
        # Yeniden başlatma sayaçları
        self.restart_counts = {
            'hourly': deque(maxlen=60),  # Son 60 dakika
            'daily': deque(maxlen=24)     # Son 24 saat
        }
        
        # Bileşen durumları
        self.component_status = {}
        self.component_callbacks = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring thread'i
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Veritabanı
        self.db_path = "restart_manager.db"
        self._initialize_database()
        
        # Varsayılan kritik bileşenler
        if not self.config.critical_components:
            self.config.critical_components = [
                'api_connection',
                'websocket_connection',
                'strategy_manager',
                'risk_manager',
                'data_manager'
            ]
        
        # Varsayılan yeniden başlatma tetikleyicileri
        if not self.config.restart_triggers:
            self.config.restart_triggers = [
                'health_check_failed',
                'memory_limit_exceeded',
                'connection_lost',
                'error_threshold_exceeded'
            ]
        
        # Callback'ler
        self.restart_callbacks = []
        self.health_callbacks = []
        
        self.logger.info("Otomatik yeniden başlatma yöneticisi başlatıldı")
    
    def _initialize_database(self):
        """Veritabanını başlat"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sağlık kontrolü geçmişi tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    response_time REAL,
                    last_error TEXT
                )
            ''')
            
            # Yeniden başlatma geçmişi tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS restart_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    component TEXT NOT NULL,
                    restart_count INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            ''')
            
            # Sistem durumu tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_status TEXT,
                    active_connections INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Veritabanı başlatma hatası: {e}")
    
    def start_monitoring(self):
        """Monitoring'i başlat"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Otomatik yeniden başlatma monitoring'i başlatıldı")
    
    def stop_monitoring(self):
        """Monitoring'i durdur"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Otomatik yeniden başlatma monitoring'i durduruldu")
    
    def _monitoring_loop(self):
        """Monitoring döngüsü"""
        while self.is_monitoring:
            try:
                # Sistem durumunu kontrol et
                self._check_system_health()
                
                # Bileşen durumlarını kontrol et
                self._check_component_health()
                
                # Yeniden başlatma gereksinimini kontrol et
                self._check_restart_requirements()
                
                # Veritabanını temizle
                self._cleanup_old_data()
                
                # Belirlenen aralıkta bekle
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring döngüsü hatası: {e}")
                time.sleep(60)
    
    def _check_system_health(self):
        """Sistem sağlığını kontrol et"""
        try:
            # CPU kullanımı
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Bellek kullanımı
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk kullanımı
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Ağ durumu
            network_status = "connected"
            try:
                requests.get("https://www.google.com", timeout=5)
            except:
                network_status = "disconnected"
            
            # Aktif bağlantılar
            active_connections = len(psutil.net_connections())
            
            # Sistem durumu kaydet
            system_status = {
                'timestamp': datetime.now(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_status': network_status,
                'active_connections': active_connections
            }
            
            self._save_system_status(system_status)
            
            # Kritik eşikleri kontrol et
            if cpu_usage > 90:
                self._trigger_restart(RestartReason.CPU_LIMIT_EXCEEDED, f"CPU kullanımı %{cpu_usage:.1f}")
            
            if memory_usage > 90:
                self._trigger_restart(RestartReason.MEMORY_LIMIT_EXCEEDED, f"Bellek kullanımı %{memory_usage:.1f}")
            
            if disk_usage > 95:
                self.logger.critical(f"Disk kullanımı kritik seviyede: %{disk_usage:.1f}")
            
            if network_status == "disconnected":
                self._trigger_restart(RestartReason.CONNECTION_LOST, "Ağ bağlantısı kesildi")
            
        except Exception as e:
            self.logger.error(f"Sistem sağlık kontrolü hatası: {e}")
    
    def _check_component_health(self):
        """Bileşen sağlığını kontrol et"""
        try:
            for component in self.config.critical_components:
                health_check = self._perform_component_health_check(component)
                
                if health_check:
                    self.health_history.append(health_check)
                    self._save_health_check(health_check)
                    
                    # Bileşen durumunu güncelle
                    with self.lock:
                        self.component_status[component] = health_check.status
                    
                    # Callback'leri çağır
                    self._notify_health_callbacks(health_check)
                    
                    # Kritik durum kontrolü
                    if health_check.status == ComponentStatus.CRITICAL.value:
                        self._trigger_restart(
                            RestartReason.HEALTH_CHECK_FAILED, 
                            f"Bileşen kritik durumda: {component}"
                        )
            
        except Exception as e:
            self.logger.error(f"Bileşen sağlık kontrolü hatası: {e}")
    
    def _perform_component_health_check(self, component: str) -> Optional[HealthCheck]:
        """Bileşen sağlık kontrolü yap"""
        try:
            start_time = time.time()
            
            if component == 'api_connection':
                return self._check_api_connection()
            elif component == 'websocket_connection':
                return self._check_websocket_connection()
            elif component == 'strategy_manager':
                return self._check_strategy_manager()
            elif component == 'risk_manager':
                return self._check_risk_manager()
            elif component == 'data_manager':
                return self._check_data_manager()
            else:
                # Genel bileşen kontrolü
                return self._check_generic_component(component)
            
        except Exception as e:
            self.logger.error(f"Bileşen sağlık kontrolü hatası ({component}): {e}")
            return HealthCheck(
                component=component,
                status=ComponentStatus.UNKNOWN.value,
                message=f"Kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_api_connection(self) -> HealthCheck:
        """API bağlantısını kontrol et"""
        try:
            # BTCTURK API'sine ping at
            response = requests.get("https://api.btcturk.com/api/v2/ticker", timeout=10)
            
            if response.status_code == 200:
                return HealthCheck(
                    component='api_connection',
                    status=ComponentStatus.HEALTHY.value,
                    message="API bağlantısı sağlıklı",
                    timestamp=datetime.now(),
                    response_time=response.elapsed.total_seconds()
                )
            else:
                return HealthCheck(
                    component='api_connection',
                    status=ComponentStatus.CRITICAL.value,
                    message=f"API yanıt hatası: {response.status_code}",
                    timestamp=datetime.now(),
                    response_time=response.elapsed.total_seconds()
                )
                
        except Exception as e:
            return HealthCheck(
                component='api_connection',
                status=ComponentStatus.CRITICAL.value,
                message=f"API bağlantı hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_websocket_connection(self) -> HealthCheck:
        """WebSocket bağlantısını kontrol et"""
        try:
            # WebSocket bağlantı durumunu kontrol et
            # Bu gerçek implementasyonda WebSocket client'ın durumunu kontrol edecek
            
            # Simüle edilmiş kontrol
            is_connected = True  # Gerçek implementasyonda WebSocket durumu kontrol edilecek
            
            if is_connected:
                return HealthCheck(
                    component='websocket_connection',
                    status=ComponentStatus.HEALTHY.value,
                    message="WebSocket bağlantısı sağlıklı",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    component='websocket_connection',
                    status=ComponentStatus.CRITICAL.value,
                    message="WebSocket bağlantısı kesildi",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                component='websocket_connection',
                status=ComponentStatus.CRITICAL.value,
                message=f"WebSocket kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_strategy_manager(self) -> HealthCheck:
        """Strateji yöneticisini kontrol et"""
        try:
            # Strateji yöneticisi durumunu kontrol et
            # Bu gerçek implementasyonda strategy manager'ın durumunu kontrol edecek
            
            # Simüle edilmiş kontrol
            is_running = True  # Gerçek implementasyonda strategy manager durumu kontrol edilecek
            
            if is_running:
                return HealthCheck(
                    component='strategy_manager',
                    status=ComponentStatus.HEALTHY.value,
                    message="Strateji yöneticisi çalışıyor",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    component='strategy_manager',
                    status=ComponentStatus.CRITICAL.value,
                    message="Strateji yöneticisi durdu",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                component='strategy_manager',
                status=ComponentStatus.CRITICAL.value,
                message=f"Strateji yöneticisi kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_risk_manager(self) -> HealthCheck:
        """Risk yöneticisini kontrol et"""
        try:
            # Risk yöneticisi durumunu kontrol et
            # Bu gerçek implementasyonda risk manager'ın durumunu kontrol edecek
            
            # Simüle edilmiş kontrol
            is_running = True  # Gerçek implementasyonda risk manager durumu kontrol edilecek
            
            if is_running:
                return HealthCheck(
                    component='risk_manager',
                    status=ComponentStatus.HEALTHY.value,
                    message="Risk yöneticisi çalışıyor",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    component='risk_manager',
                    status=ComponentStatus.CRITICAL.value,
                    message="Risk yöneticisi durdu",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                component='risk_manager',
                status=ComponentStatus.CRITICAL.value,
                message=f"Risk yöneticisi kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_data_manager(self) -> HealthCheck:
        """Veri yöneticisini kontrol et"""
        try:
            # Veri yöneticisi durumunu kontrol et
            # Bu gerçek implementasyonda data manager'ın durumunu kontrol edecek
            
            # Simüle edilmiş kontrol
            is_running = True  # Gerçek implementasyonda data manager durumu kontrol edilecek
            
            if is_running:
                return HealthCheck(
                    component='data_manager',
                    status=ComponentStatus.HEALTHY.value,
                    message="Veri yöneticisi çalışıyor",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    component='data_manager',
                    status=ComponentStatus.CRITICAL.value,
                    message="Veri yöneticisi durdu",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                component='data_manager',
                status=ComponentStatus.CRITICAL.value,
                message=f"Veri yöneticisi kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_generic_component(self, component: str) -> HealthCheck:
        """Genel bileşen kontrolü"""
        try:
            # Genel bileşen durumu kontrolü
            # Bu gerçek implementasyonda bileşenin durumunu kontrol edecek
            
            # Simüle edilmiş kontrol
            is_running = True  # Gerçek implementasyonda bileşen durumu kontrol edilecek
            
            if is_running:
                return HealthCheck(
                    component=component,
                    status=ComponentStatus.HEALTHY.value,
                    message=f"{component} çalışıyor",
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    component=component,
                    status=ComponentStatus.CRITICAL.value,
                    message=f"{component} durdu",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                component=component,
                status=ComponentStatus.CRITICAL.value,
                message=f"{component} kontrol hatası: {e}",
                timestamp=datetime.now(),
                last_error=str(e)
            )
    
    def _check_restart_requirements(self):
        """Yeniden başlatma gereksinimlerini kontrol et"""
        try:
            if not self.config.enabled:
                return
            
            # Yeniden başlatma limitlerini kontrol et
            if self._is_restart_limit_exceeded():
                self.logger.warning("Yeniden başlatma limiti aşıldı")
                return
            
            # Kritik bileşenlerin durumunu kontrol et
            critical_failed = False
            for component in self.config.critical_components:
                if self.component_status.get(component) == ComponentStatus.CRITICAL.value:
                    critical_failed = True
                    break
            
            if critical_failed:
                self._trigger_restart(RestartReason.HEALTH_CHECK_FAILED, "Kritik bileşenler başarısız")
            
        except Exception as e:
            self.logger.error(f"Yeniden başlatma gereksinimleri kontrol hatası: {e}")
    
    def _is_restart_limit_exceeded(self) -> bool:
        """Yeniden başlatma limiti aşıldı mı kontrol et"""
        try:
            now = datetime.now()
            
            # Saatlik limit kontrolü
            hourly_restarts = [
                r for r in self.restart_counts['hourly']
                if now - r < timedelta(hours=1)
            ]
            
            if len(hourly_restarts) >= self.config.max_restarts_per_hour:
                return True
            
            # Günlük limit kontrolü
            daily_restarts = [
                r for r in self.restart_counts['daily']
                if now - r < timedelta(days=1)
            ]
            
            if len(daily_restarts) >= self.config.max_restarts_per_day:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Yeniden başlatma limiti kontrol hatası: {e}")
            return False
    
    def _trigger_restart(self, reason: RestartReason, message: str):
        """Yeniden başlatmayı tetikle"""
        try:
            if not self.config.enabled:
                return
            
            if self._is_restart_limit_exceeded():
                self.logger.critical("Yeniden başlatma limiti aşıldı, yeniden başlatma yapılamıyor")
                return
            
            self.logger.warning(f"Yeniden başlatma tetiklendi: {reason.value} - {message}")
            
            # Yeniden başlatma olayını kaydet
            restart_event = RestartEvent(
                timestamp=datetime.now(),
                reason=reason.value,
                component="system",
                restart_count=len(self.restart_counts['hourly']),
                success=False
            )
            
            # Yeniden başlatma sayacını güncelle
            with self.lock:
                self.restart_counts['hourly'].append(restart_event.timestamp)
                self.restart_counts['daily'].append(restart_event.timestamp)
            
            # Yeniden başlatma işlemini başlat
            self._perform_restart(restart_event)
            
        except Exception as e:
            self.logger.error(f"Yeniden başlatma tetikleme hatası: {e}")
    
    def _perform_restart(self, restart_event: RestartEvent):
        """Yeniden başlatma işlemini gerçekleştir"""
        try:
            self.logger.info("Yeniden başlatma işlemi başlatılıyor...")
            
            # Backup oluştur
            if self.config.backup_before_restart:
                self._create_backup()
            
            # Bildirim gönder
            if self.config.notify_on_restart:
                self._notify_restart_callbacks(restart_event)
            
            # Yeniden başlatma gecikmesi
            time.sleep(self.config.restart_delay_seconds)
            
            # Yeniden başlatma işlemi
            if self._is_docker_environment():
                self._restart_docker_container()
            else:
                self._restart_application()
            
            restart_event.success = True
            self.logger.info("Yeniden başlatma işlemi tamamlandı")
            
        except Exception as e:
            restart_event.error_message = str(e)
            self.logger.error(f"Yeniden başlatma işlemi hatası: {e}")
        
        finally:
            # Yeniden başlatma olayını kaydet
            self.restart_history.append(restart_event)
            self._save_restart_event(restart_event)
    
    def _is_docker_environment(self) -> bool:
        """Docker ortamında mı kontrol et"""
        try:
            return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
        except:
            return False
    
    def _restart_docker_container(self):
        """Docker container'ı yeniden başlat"""
        try:
            import platform
            if platform.system() == "Windows":
                self.logger.warning("Windows ortamında Docker restart desteklenmiyor")
                return False
                
            container_name = os.environ.get('CONTAINER_NAME', 'btcturk-trading-bot')
            
            # Docker komutunun varlığını kontrol et
            try:
                subprocess.run(['docker', '--version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error("Docker bulunamadı")
                return False
            
            # Docker container'ı yeniden başlat
            subprocess.run([
                'docker', 'restart', container_name
            ], check=True, timeout=30)
            
            self.logger.info(f"Docker container yeniden başlatıldı: {container_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker container yeniden başlatma hatası: {e}")
            return False
    
    def _restart_application(self):
        """Uygulamayı yeniden başlat"""
        try:
            # Mevcut process'i sonlandır
            current_pid = os.getpid()
            
            # Yeni process başlat
            script_path = sys.argv[0]
            subprocess.Popen([sys.executable, script_path] + sys.argv[1:])
            
            # Mevcut process'i sonlandır (OS'e göre)
            try:
                if platform.system() == "Windows":
                    p = psutil.Process(current_pid)
                    p.terminate()
                else:
                    os.kill(current_pid, signal.SIGTERM)
            except Exception:
                # Son çare: sert çıkış
                os._exit(0)
            
            self.logger.info("Uygulama yeniden başlatıldı")
            
        except Exception as e:
            self.logger.error(f"Uygulama yeniden başlatma hatası: {e}")
            raise
    
    def _create_backup(self):
        """Backup oluştur"""
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.zip"
            
            # Önemli dosyaları backup'la
            important_files = [
                "config/",
                "data/",
                "logs/",
                "*.db",
                "*.json"
            ]
            
            # Zipfile ile backup oluştur (Windows uyumlu) ve glob desenlerini destekle
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for pattern in important_files:
                    matches = glob.glob(pattern, recursive=True)
                    for path in matches:
                        if os.path.isfile(path):
                            arcname = os.path.relpath(path, '.')
                            zipf.write(path, arcname)
                        elif os.path.isdir(path):
                            for root, dirs, files in os.walk(path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, '.')
                                    zipf.write(file_path, arcname)
            
            self.logger.info(f"Backup oluşturuldu: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Backup oluşturma hatası: {e}")
    
    def _cleanup_old_data(self):
        """Eski verileri temizle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 30 günden eski sağlık kontrolü verilerini sil
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('DELETE FROM health_checks WHERE timestamp < ?', (cutoff_date,))
            
            # 7 günden eski yeniden başlatma verilerini sil
            cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('DELETE FROM restart_events WHERE timestamp < ?', (cutoff_date,))
            
            # 24 saatten eski sistem durumu verilerini sil
            cutoff_date = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor.execute('DELETE FROM system_status WHERE timestamp < ?', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Veri temizleme hatası: {e}")
    
    def _save_health_check(self, health_check: HealthCheck):
        """Sağlık kontrolünü veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_checks 
                (component, status, message, timestamp, response_time, last_error)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                health_check.component,
                health_check.status,
                health_check.message,
                health_check.timestamp.isoformat(),
                health_check.response_time,
                health_check.last_error
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Sağlık kontrolü kaydetme hatası: {e}")
    
    def _save_restart_event(self, restart_event: RestartEvent):
        """Yeniden başlatma olayını veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO restart_events 
                (timestamp, reason, component, restart_count, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                restart_event.timestamp.isoformat(),
                restart_event.reason,
                restart_event.component,
                restart_event.restart_count,
                restart_event.success,
                restart_event.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Yeniden başlatma olayı kaydetme hatası: {e}")
    
    def _save_system_status(self, system_status: Dict[str, Any]):
        """Sistem durumunu veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_status 
                (timestamp, cpu_usage, memory_usage, disk_usage, network_status, active_connections)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                system_status['timestamp'].isoformat(),
                system_status['cpu_usage'],
                system_status['memory_usage'],
                system_status['disk_usage'],
                system_status['network_status'],
                system_status['active_connections']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Sistem durumu kaydetme hatası: {e}")
    
    def manual_restart(self, reason: str = "manual_restart"):
        """Manuel yeniden başlatma"""
        try:
            restart_event = RestartEvent(
                timestamp=datetime.now(),
                reason=reason,
                component="system",
                restart_count=len(self.restart_counts['hourly']),
                success=False
            )
            
            # Yeniden başlatma sayacını güncelle
            with self.lock:
                self.restart_counts['hourly'].append(restart_event.timestamp)
                self.restart_counts['daily'].append(restart_event.timestamp)
            
            # Yeniden başlatma işlemini başlat
            self._perform_restart(restart_event)
            
        except Exception as e:
            self.logger.error(f"Manuel yeniden başlatma hatası: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Sağlık durumunu al"""
        try:
            with self.lock:
                # Son sağlık kontrolü sonuçları
                recent_health = list(self.health_history)[-10:] if self.health_history else []
                
                # Bileşen durumları
                component_status = self.component_status.copy()
                
                # Genel durum
                overall_status = "healthy"
                if any(status == ComponentStatus.CRITICAL.value for status in component_status.values()):
                    overall_status = "critical"
                elif any(status == ComponentStatus.WARNING.value for status in component_status.values()):
                    overall_status = "warning"
                
                return {
                    'overall_status': overall_status,
                    'component_status': component_status,
                    'recent_health_checks': [
                        {
                            'component': h.component,
                            'status': h.status,
                            'message': h.message,
                            'timestamp': h.timestamp.isoformat(),
                            'response_time': h.response_time
                        }
                        for h in recent_health
                    ],
                    'monitoring_active': self.is_monitoring,
                    'restart_enabled': self.config.enabled
                }
                
        except Exception as e:
            self.logger.error(f"Sağlık durumu alma hatası: {e}")
            return {}
    
    def get_restart_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Yeniden başlatma geçmişini al"""
        try:
            with self.lock:
                return [
                    {
                        'timestamp': r.timestamp.isoformat(),
                        'reason': r.reason,
                        'component': r.component,
                        'restart_count': r.restart_count,
                        'success': r.success,
                        'error_message': r.error_message
                    }
                    for r in list(self.restart_history)[-limit:]
                ]
        except Exception as e:
            self.logger.error(f"Yeniden başlatma geçmişi alma hatası: {e}")
            return []
    
    def get_restart_stats(self) -> Dict[str, Any]:
        """Yeniden başlatma istatistiklerini al"""
        try:
            with self.lock:
                now = datetime.now()
                
                # Saatlik yeniden başlatma sayısı
                hourly_restarts = len([
                    r for r in self.restart_counts['hourly']
                    if now - r < timedelta(hours=1)
                ])
                
                # Günlük yeniden başlatma sayısı
                daily_restarts = len([
                    r for r in self.restart_counts['daily']
                    if now - r < timedelta(days=1)
                ])
                
                # Başarılı/başarısız yeniden başlatma sayıları
                successful_restarts = len([r for r in self.restart_history if r.success])
                failed_restarts = len([r for r in self.restart_history if not r.success])
                
                # Yeniden başlatma nedenleri
                restart_reasons = {}
                for restart in self.restart_history:
                    reason = restart.reason
                    restart_reasons[reason] = restart_reasons.get(reason, 0) + 1
                
                return {
                    'hourly_restarts': hourly_restarts,
                    'daily_restarts': daily_restarts,
                    'total_restarts': len(self.restart_history),
                    'successful_restarts': successful_restarts,
                    'failed_restarts': failed_restarts,
                    'success_rate': successful_restarts / len(self.restart_history) if self.restart_history else 0,
                    'restart_reasons': restart_reasons,
                    'max_hourly_limit': self.config.max_restarts_per_hour,
                    'max_daily_limit': self.config.max_restarts_per_day
                }
                
        except Exception as e:
            self.logger.error(f"Yeniden başlatma istatistikleri alma hatası: {e}")
            return {}
    
    def add_component_callback(self, component: str, callback: Callable):
        """Bileşen callback'i ekle"""
        try:
            if component not in self.component_callbacks:
                self.component_callbacks[component] = []
            
            self.component_callbacks[component].append(callback)
            
        except Exception as e:
            self.logger.error(f"Bileşen callback ekleme hatası: {e}")
    
    def add_restart_callback(self, callback: Callable):
        """Yeniden başlatma callback'i ekle"""
        self.restart_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable):
        """Sağlık callback'i ekle"""
        self.health_callbacks.append(callback)
    
    def _notify_restart_callbacks(self, restart_event: RestartEvent):
        """Yeniden başlatma callback'lerini çağır"""
        for callback in self.restart_callbacks:
            try:
                callback(restart_event)
            except Exception as e:
                self.logger.error(f"Yeniden başlatma callback hatası: {e}")
    
    def _notify_health_callbacks(self, health_check: HealthCheck):
        """Sağlık callback'lerini çağır"""
        for callback in self.health_callbacks:
            try:
                callback(health_check)
            except Exception as e:
                self.logger.error(f"Sağlık callback hatası: {e}")
    
    def update_config(self, new_config: RestartConfig):
        """Konfigürasyonu güncelle"""
        try:
            self.config = new_config
            self.logger.info("Otomatik yeniden başlatma konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global otomatik yeniden başlatma yöneticisi
auto_restart_manager = AutoRestartManager()
