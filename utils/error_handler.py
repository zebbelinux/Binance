"""
Hata Yönetimi Sistemi
Merkezi hata yakalama, raporlama ve iyileştirme sistemi
"""

import sys
import traceback
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os

from utils.logger import get_logger, LogLevel, LogCategory

class ErrorSeverity(Enum):
    """Hata şiddeti"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Hata kategorisi"""
    SYSTEM = "system"
    API = "api"
    TRADING = "trading"
    STRATEGY = "strategy"
    RISK = "risk"
    DATA = "data"
    AI = "ai"
    GUI = "gui"
    NETWORK = "network"
    DATABASE = "database"

@dataclass
class ErrorInfo:
    """Hata bilgisi"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    module: str
    function: str
    line_number: int
    traceback: str
    context: Dict[str, Any]
    thread_id: int
    process_id: int

class ErrorHandler:
    """Hata yöneticisi sınıfı"""
    
    def __init__(self):
        self.logger = get_logger("error_handler")
        
        # Hata istatistikleri
        self.error_stats = {
            'total_errors': 0,
            'by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'by_category': {category.value: 0 for category in ErrorCategory},
            'by_module': {},
            'recent_errors': []
        }
        
        # Hata işleme kuralları
        self.error_rules = {}
        
        # Callback'ler
        self.error_callbacks = []
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Sistem hata yakalayıcıları
        self._setup_system_handlers()
        
        self.logger.info(LogCategory.SYSTEM, "Hata yöneticisi başlatıldı")
    
    def _setup_system_handlers(self):
        """Sistem hata yakalayıcılarını kur"""
        try:
            # Python hata yakalayıcısı
            sys.excepthook = self._handle_system_exception
            
            # Thread hata yakalayıcısı
            threading.excepthook = self._handle_thread_exception
            
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Sistem hata yakalayıcı kurulum hatası: {e}")
    
    def _handle_system_exception(self, exc_type, exc_value, exc_traceback):
        """Sistem exception yakalayıcısı"""
        try:
            if exc_type == KeyboardInterrupt:
                return
            
            error_info = self._create_error_info(
                exc_type, exc_value, exc_traceback,
                ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM
            )
            
            self._process_error(error_info)
            
        except Exception as e:
            print(f"Hata yakalayıcı hatası: {e}")
    
    def _handle_thread_exception(self, args):
        """Thread exception yakalayıcısı"""
        try:
            exc_type, exc_value, exc_traceback, thread = args
            
            error_info = self._create_error_info(
                exc_type, exc_value, exc_traceback,
                ErrorSeverity.HIGH, ErrorCategory.SYSTEM
            )
            
            error_info.context['thread_name'] = thread.name
            
            self._process_error(error_info)
            
        except Exception as e:
            print(f"Thread hata yakalayıcı hatası: {e}")
    
    def _create_error_info(self, exc_type, exc_value, exc_traceback, 
                          severity: ErrorSeverity, category: ErrorCategory) -> ErrorInfo:
        """Hata bilgisi oluştur"""
        try:
            # Traceback'i al
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_str = ''.join(tb_lines)
            
            # Modül ve fonksiyon bilgisi
            if exc_traceback:
                frame = exc_traceback.tb_frame
                module = frame.f_globals.get('__name__', 'unknown')
                function = frame.f_code.co_name
                line_number = exc_traceback.tb_lineno
            else:
                module = 'unknown'
                function = 'unknown'
                line_number = 0
            
            # Context bilgisi
            context = {
                'python_version': sys.version,
                'platform': sys.platform,
                'executable': sys.executable
            }
            
            return ErrorInfo(
                error_id=f"ERR_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                error_type=exc_type.__name__,
                error_message=str(exc_value),
                severity=severity,
                category=category,
                module=module,
                function=function,
                line_number=line_number,
                traceback=tb_str,
                context=context,
                thread_id=threading.get_ident(),
                process_id=os.getpid()
            )
            
        except Exception as e:
            print(f"Hata bilgisi oluşturma hatası: {e}")
            return None
    
    def _process_error(self, error_info: ErrorInfo):
        """Hata işle"""
        try:
            if not error_info:
                return
            
            # İstatistikleri güncelle
            self._update_error_stats(error_info)
            
            # Log kaydet
            self._log_error(error_info)
            
            # Hata kurallarını kontrol et
            self._check_error_rules(error_info)
            
            # Callback'leri çağır
            self._notify_error_callbacks(error_info)
            
            # Kritik hatalar için özel işlem
            if error_info.severity == ErrorSeverity.CRITICAL:
                self._handle_critical_error(error_info)
                
        except Exception as e:
            print(f"Hata işleme hatası: {e}")
    
    def _update_error_stats(self, error_info: ErrorInfo):
        """Hata istatistiklerini güncelle"""
        try:
            with self.stats_lock:
                self.error_stats['total_errors'] += 1
                self.error_stats['by_severity'][error_info.severity.value] += 1
                self.error_stats['by_category'][error_info.category.value] += 1
                
                # Modül istatistikleri
                module = error_info.module
                if module not in self.error_stats['by_module']:
                    self.error_stats['by_module'][module] = 0
                self.error_stats['by_module'][module] += 1
                
                # Son hatalar listesi
                self.error_stats['recent_errors'].append({
                    'error_id': error_info.error_id,
                    'timestamp': error_info.timestamp.isoformat(),
                    'error_type': error_info.error_type,
                    'severity': error_info.severity.value,
                    'category': error_info.category.value,
                    'module': error_info.module
                })
                
                # Son 100 hatayı tut
                if len(self.error_stats['recent_errors']) > 100:
                    self.error_stats['recent_errors'] = self.error_stats['recent_errors'][-100:]
                    
        except Exception as e:
            print(f"İstatistik güncelleme hatası: {e}")
    
    def _log_error(self, error_info: ErrorInfo):
        """Hatayı logla"""
        try:
            log_level = LogLevel.ERROR
            if error_info.severity == ErrorSeverity.CRITICAL:
                log_level = LogLevel.CRITICAL
            elif error_info.severity == ErrorSeverity.HIGH:
                log_level = LogLevel.ERROR
            elif error_info.severity == ErrorSeverity.MEDIUM:
                log_level = LogLevel.WARNING
            else:
                log_level = LogLevel.INFO
            
            # Log kategorisi
            log_category = LogCategory.SYSTEM
            if error_info.category == ErrorCategory.API:
                log_category = LogCategory.API
            elif error_info.category == ErrorCategory.TRADING:
                log_category = LogCategory.TRADING
            elif error_info.category == ErrorCategory.STRATEGY:
                log_category = LogCategory.STRATEGY
            elif error_info.category == ErrorCategory.RISK:
                log_category = LogCategory.RISK
            elif error_info.category == ErrorCategory.DATA:
                log_category = LogCategory.DATA
            elif error_info.category == ErrorCategory.AI:
                log_category = LogCategory.AI
            elif error_info.category == ErrorCategory.GUI:
                log_category = LogCategory.GUI
            
            # Extra data
            extra_data = {
                'error_id': error_info.error_id,
                'error_type': error_info.error_type,
                'severity': error_info.severity.value,
                'category': error_info.category.value,
                'module': error_info.module,
                'function': error_info.function,
                'line_number': error_info.line_number,
                'traceback': error_info.traceback,
                'context': error_info.context
            }
            
            # Log kaydet
            self.logger.log(log_level, log_category, error_info.error_message, extra_data)
            
        except Exception as e:
            print(f"Hata loglama hatası: {e}")
    
    def _check_error_rules(self, error_info: ErrorInfo):
        """Hata kurallarını kontrol et"""
        try:
            for rule_name, rule in self.error_rules.items():
                if self._evaluate_error_rule(error_info, rule):
                    self._execute_error_rule(rule_name, rule, error_info)
                    
        except Exception as e:
            print(f"Hata kural kontrolü hatası: {e}")
    
    def _evaluate_error_rule(self, error_info: ErrorInfo, rule: Dict[str, Any]) -> bool:
        """Hata kuralını değerlendir"""
        try:
            # Severity kontrolü
            if 'min_severity' in rule:
                severity_levels = {
                    ErrorSeverity.LOW: 1,
                    ErrorSeverity.MEDIUM: 2,
                    ErrorSeverity.HIGH: 3,
                    ErrorSeverity.CRITICAL: 4
                }
                if severity_levels[error_info.severity] < rule['min_severity']:
                    return False
            
            # Kategori kontrolü
            if 'categories' in rule:
                if error_info.category not in rule['categories']:
                    return False
            
            # Modül kontrolü
            if 'modules' in rule:
                if error_info.module not in rule['modules']:
                    return False
            
            # Hata tipi kontrolü
            if 'error_types' in rule:
                if error_info.error_type not in rule['error_types']:
                    return False
            
            # Frekans kontrolü
            if 'max_frequency' in rule:
                recent_errors = self._get_recent_errors_by_rule(rule, error_info)
                if len(recent_errors) >= rule['max_frequency']:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Hata kural değerlendirme hatası: {e}")
            return False
    
    def _get_recent_errors_by_rule(self, rule: Dict[str, Any], error_info: ErrorInfo) -> List[Dict[str, Any]]:
        """Kurala göre son hataları al"""
        try:
            recent_errors = []
            time_window = rule.get('time_window', 300)  # 5 dakika varsayılan
            
            cutoff_time = datetime.now().timestamp() - time_window
            
            for error in self.error_stats['recent_errors']:
                error_time = datetime.fromisoformat(error['timestamp']).timestamp()
                if error_time >= cutoff_time:
                    # Kural kriterlerini kontrol et
                    if self._error_matches_rule_criteria(error, rule):
                        recent_errors.append(error)
            
            return recent_errors
            
        except Exception as e:
            print(f"Son hatalar alma hatası: {e}")
            return []
    
    def _error_matches_rule_criteria(self, error: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Hata kural kriterlerine uyuyor mu?"""
        try:
            if 'categories' in rule:
                if error['category'] not in [c.value for c in rule['categories']]:
                    return False
            
            if 'modules' in rule:
                if error['module'] not in rule['modules']:
                    return False
            
            if 'error_types' in rule:
                if error['error_type'] not in rule['error_types']:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Hata kriter kontrolü hatası: {e}")
            return False
    
    def _execute_error_rule(self, rule_name: str, rule: Dict[str, Any], error_info: ErrorInfo):
        """Hata kuralını çalıştır"""
        try:
            actions = rule.get('actions', [])
            
            for action in actions:
                if action['type'] == 'log':
                    self.logger.warning(
                        LogCategory.SYSTEM,
                        f"Hata kuralı tetiklendi: {rule_name}",
                        {'rule': rule_name, 'error_id': error_info.error_id}
                    )
                
                elif action['type'] == 'callback':
                    callback = action.get('callback')
                    if callback and callable(callback):
                        callback(error_info, rule)
                
                elif action['type'] == 'restart':
                    self.logger.critical(
                        LogCategory.SYSTEM,
                        f"Uygulama yeniden başlatılıyor - Kural: {rule_name}",
                        {'rule': rule_name, 'error_id': error_info.error_id}
                    )
                    # Restart işlemi burada yapılabilir
                
                elif action['type'] == 'shutdown':
                    self.logger.critical(
                        LogCategory.SYSTEM,
                        f"Uygulama kapatılıyor - Kural: {rule_name}",
                        {'rule': rule_name, 'error_id': error_info.error_id}
                    )
                    # Shutdown işlemi burada yapılabilir
                    
        except Exception as e:
            print(f"Hata kural çalıştırma hatası: {e}")
    
    def _handle_critical_error(self, error_info: ErrorInfo):
        """Kritik hata işle"""
        try:
            self.logger.critical(
                LogCategory.SYSTEM,
                f"Kritik hata: {error_info.error_message}",
                {
                    'error_id': error_info.error_id,
                    'error_type': error_info.error_type,
                    'module': error_info.module,
                    'function': error_info.function,
                    'line_number': error_info.line_number
                }
            )
            
            # Kritik hata için özel işlemler
            # - Veritabanı bağlantılarını kapat
            # - Açık pozisyonları kapat
            # - Sistem durumunu kaydet
            # - Uyarı gönder
            
        except Exception as e:
            print(f"Kritik hata işleme hatası: {e}")
    
    def _notify_error_callbacks(self, error_info: ErrorInfo):
        """Hata callback'lerini çağır"""
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                print(f"Hata callback hatası: {e}")
    
    def add_error_rule(self, rule_name: str, rule: Dict[str, Any]):
        """Hata kuralı ekle"""
        try:
            self.error_rules[rule_name] = rule
            self.logger.info(
                LogCategory.SYSTEM,
                f"Hata kuralı eklendi: {rule_name}",
                {'rule': rule_name}
            )
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Hata kuralı ekleme hatası: {e}")
    
    def remove_error_rule(self, rule_name: str):
        """Hata kuralını kaldır"""
        try:
            if rule_name in self.error_rules:
                del self.error_rules[rule_name]
                self.logger.info(
                    LogCategory.SYSTEM,
                    f"Hata kuralı kaldırıldı: {rule_name}",
                    {'rule': rule_name}
                )
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Hata kuralı kaldırma hatası: {e}")
    
    def add_error_callback(self, callback: Callable):
        """Hata callback'i ekle"""
        self.error_callbacks.append(callback)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Hata istatistiklerini al"""
        with self.stats_lock:
            return self.error_stats.copy()
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Son hataları al"""
        with self.stats_lock:
            return self.error_stats['recent_errors'][-limit:]
    
    def clear_error_stats(self):
        """Hata istatistiklerini temizle"""
        with self.stats_lock:
            self.error_stats = {
                'total_errors': 0,
                'by_severity': {severity.value: 0 for severity in ErrorSeverity},
                'by_category': {category.value: 0 for category in ErrorCategory},
                'by_module': {},
                'recent_errors': []
            }

# Global hata yöneticisi
error_handler = ErrorHandler()

# Kolay kullanım için fonksiyonlar
def handle_error(error: Exception, category: ErrorCategory = ErrorCategory.SYSTEM, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
    """Hata işle"""
    try:
        error_info = error_handler._create_error_info(
            type(error), error, error.__traceback__,
            severity, category
        )
        
        if context:
            error_info.context.update(context)
        
        error_handler._process_error(error_info)
        
    except Exception as e:
        print(f"Hata işleme fonksiyonu hatası: {e}")

def log_and_handle_error(error: Exception, category: ErrorCategory = ErrorCategory.SYSTEM, 
                        severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                        context: Dict[str, Any] = None):
    """Hata logla ve işle"""
    handle_error(error, category, severity, context)

# Decorator'lar
def error_handler_decorator(category: ErrorCategory = ErrorCategory.SYSTEM, 
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Hata yakalama decorator'ı"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, category, severity, {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                raise
        return wrapper
    return decorator

def safe_execute(func, *args, category: ErrorCategory = ErrorCategory.SYSTEM, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                default_return=None, **kwargs):
    """Güvenli fonksiyon çalıştırma"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, category, severity, {
            'function': func.__name__,
            'args': str(args),
            'kwargs': str(kwargs)
        })
        return default_return




