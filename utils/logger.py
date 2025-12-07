"""
Gelişmiş Logging Sistemi
Kategorize edilmiş, timestamp'li ve hata ayıklama destekli logging
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import threading
import time

class LogLevel(Enum):
    """Log seviyeleri"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log kategorileri"""
    SYSTEM = "SYSTEM"
    API = "API"
    TRADING = "TRADING"
    STRATEGY = "STRATEGY"
    RISK = "RISK"
    DATA = "DATA"
    AI = "AI"
    GUI = "GUI"
    BACKTEST = "BACKTEST"
    PERFORMANCE = "PERFORMANCE"

@dataclass
class LogConfig:
    """Log konfigürasyonu"""
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    include_traceback: bool = True
    log_level: LogLevel = LogLevel.INFO

class AdvancedLogger:
    """Gelişmiş logger sınıfı"""
    
    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        
        # Log klasörünü oluştur
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Logger oluştur
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Mevcut handler'ları temizle
        self.logger.handlers.clear()
        
        # Handler'ları ekle
        self._setup_handlers()
        
        # Log istatistikleri
        self.log_stats = {
            'total_logs': 0,
            'by_level': {level.value: 0 for level in LogLevel},
            'by_category': {category.value: 0 for category in LogCategory},
            'errors': 0,
            'warnings': 0
        }
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Kategori alanı gerektiginden, kendi wrapper'imizi kullan
        self.info(LogCategory.SYSTEM, f"Logger '{name}' başlatıldı")
    
    def _setup_handlers(self):
        """Handler'ları kur"""
        try:
            # Console handler
            if self.config.console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(getattr(logging, self.config.log_level.value))
                console_formatter = self._get_console_formatter()
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # File handler
            if self.config.file_output:
                # Ana log dosyası
                main_log_file = os.path.join(self.config.log_dir, f"{self.name}.log")
                main_handler = logging.handlers.RotatingFileHandler(
                    main_log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                main_handler.setLevel(logging.DEBUG)
                main_formatter = self._get_file_formatter()
                main_handler.setFormatter(main_formatter)
                self.logger.addHandler(main_handler)
                
                # Hata log dosyası
                error_log_file = os.path.join(self.config.log_dir, f"{self.name}_errors.log")
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(main_formatter)
                self.logger.addHandler(error_handler)
                
                # Kategori bazlı log dosyaları
                for category in LogCategory:
                    category_file = os.path.join(self.config.log_dir, f"{self.name}_{category.value.lower()}.log")
                    category_handler = logging.handlers.RotatingFileHandler(
                        category_file,
                        maxBytes=self.config.max_file_size,
                        backupCount=self.config.backup_count,
                        encoding='utf-8'
                    )
                    category_handler.setLevel(logging.DEBUG)
                    category_handler.setFormatter(main_formatter)
                    
                    # Kategori filtresi ekle
                    category_filter = CategoryFilter(category)
                    category_handler.addFilter(category_filter)
                    
                    self.logger.addHandler(category_handler)
            
        except Exception as e:
            print(f"Handler kurulum hatası: {e}")
    
    def _get_console_formatter(self):
        """Console formatter oluştur"""
        if self.config.json_format:
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(category)-10s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def _get_file_formatter(self):
        """File formatter oluştur"""
        if self.config.json_format:
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(category)-10s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            extra_data: Dict[str, Any] = None, exception: Exception = None):
        """Log kaydet"""
        try:
            # Extra data hazırla
            extra = {
                'category': category.value,
                'timestamp': datetime.now().isoformat(),
                'thread_id': threading.get_ident(),
                'process_id': os.getpid()
            }
            
            if extra_data:
                # LogRecord rezervli anahtarlarla cakismayi onle
                reserved = {
                    'name','msg','args','levelname','levelno','pathname','filename','module',
                    'exc_info','exc_text','stack_info','lineno','funcName','created','msecs',
                    'relativeCreated','thread','threadName','processName','process','getMessage',
                    'asctime'
                }
                safe_extra = {k: v for k, v in extra_data.items() if k not in reserved}
                extra.update(safe_extra)
            
            # Exception bilgisi ekle
            if exception and self.config.include_traceback:
                extra['exception'] = {
                    'type': type(exception).__name__,
                    'message': str(exception),
                    'traceback': traceback.format_exc()
                }
            
            # Log kaydet
            log_method = getattr(self.logger, level.value.lower())
            log_method(message, extra=extra)
            
            # İstatistikleri güncelle
            self._update_stats(level, category)
            
        except Exception as e:
            print(f"Log kaydetme hatası: {e}")
    
    def debug(self, category: LogCategory, message: str, extra_data: Dict[str, Any] = None):
        """Debug log"""
        self.log(LogLevel.DEBUG, category, message, extra_data)
    
    def info(self, category: LogCategory, message: str, extra_data: Dict[str, Any] = None):
        """Info log"""
        self.log(LogLevel.INFO, category, message, extra_data)
    
    def warning(self, category: LogCategory, message: str, extra_data: Dict[str, Any] = None):
        """Warning log"""
        self.log(LogLevel.WARNING, category, message, extra_data)
    
    def error(self, category: LogCategory, message: str, extra_data: Dict[str, Any] = None, 
             exception: Exception = None):
        """Error log"""
        self.log(LogLevel.ERROR, category, message, extra_data, exception)
    
    def critical(self, category: LogCategory, message: str, extra_data: Dict[str, Any] = None, 
                exception: Exception = None):
        """Critical log"""
        self.log(LogLevel.CRITICAL, category, message, extra_data, exception)
    
    def _update_stats(self, level: LogLevel, category: LogCategory):
        """İstatistikleri güncelle"""
        try:
            with self.stats_lock:
                self.log_stats['total_logs'] += 1
                self.log_stats['by_level'][level.value] += 1
                self.log_stats['by_category'][category.value] += 1
                
                if level == LogLevel.ERROR:
                    self.log_stats['errors'] += 1
                elif level == LogLevel.WARNING:
                    self.log_stats['warnings'] += 1
                    
        except Exception as e:
            print(f"İstatistik güncelleme hatası: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Log istatistiklerini al"""
        with self.stats_lock:
            return self.log_stats.copy()
    
    def reset_stats(self):
        """İstatistikleri sıfırla"""
        with self.stats_lock:
            self.log_stats = {
                'total_logs': 0,
                'by_level': {level.value: 0 for level in LogLevel},
                'by_category': {category.value: 0 for category in LogCategory},
                'errors': 0,
                'warnings': 0
            }

class CategoryFilter(logging.Filter):
    """Kategori filtresi"""
    
    def __init__(self, category: LogCategory):
        super().__init__()
        self.category = category
    
    def filter(self, record):
        return getattr(record, 'category', '') == self.category.value

class JsonFormatter(logging.Formatter):
    """JSON formatter"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'category': getattr(record, 'category', 'UNKNOWN'),
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': getattr(record, 'thread_id', ''),
            'process_id': getattr(record, 'process_id', '')
        }
        
        # Extra data ekle
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'category', 'thread_id', 'process_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

class LogManager:
    """Log yöneticisi"""
    
    def __init__(self):
        self.loggers = {}
        self.config = LogConfig()
        
    def get_logger(self, name: str, config: LogConfig = None) -> AdvancedLogger:
        """Logger al veya oluştur"""
        if name not in self.loggers:
            self.loggers[name] = AdvancedLogger(name, config or self.config)
        
        return self.loggers[name]
    
    def get_all_loggers(self) -> Dict[str, AdvancedLogger]:
        """Tüm logger'ları al"""
        return self.loggers.copy()
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Tüm logger'ların istatistiklerini birleştir"""
        combined_stats = {
            'total_logs': 0,
            'by_level': {level.value: 0 for level in LogLevel},
            'by_category': {category.value: 0 for category in LogCategory},
            'errors': 0,
            'warnings': 0,
            'loggers': {}
        }
        
        for name, logger in self.loggers.items():
            stats = logger.get_stats()
            combined_stats['loggers'][name] = stats
            
            # Toplam istatistikleri güncelle
            combined_stats['total_logs'] += stats['total_logs']
            combined_stats['errors'] += stats['errors']
            combined_stats['warnings'] += stats['warnings']
            
            for level, count in stats['by_level'].items():
                combined_stats['by_level'][level] += count
            
            for category, count in stats['by_category'].items():
                combined_stats['by_category'][category] += count
        
        return combined_stats
    
    def cleanup_old_logs(self, days: int = 30):
        """Eski log dosyalarını temizle"""
        try:
            import glob
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for logger in self.loggers.values():
                log_dir = logger.config.log_dir
                
                # Tüm log dosyalarını bul
                log_files = glob.glob(os.path.join(log_dir, "*.log*"))
                
                for log_file in log_files:
                    # Dosya tarihini kontrol et
                    file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    
                    if file_time < cutoff_date:
                        try:
                            os.remove(log_file)
                            print(f"Eski log dosyası silindi: {log_file}")
                        except Exception as e:
                            print(f"Log dosyası silinemedi {log_file}: {e}")
                            
        except Exception as e:
            print(f"Log temizleme hatası: {e}")

# Global log yöneticisi
log_manager = LogManager()

# Kolay kullanım için fonksiyonlar
def get_logger(name: str, config: LogConfig = None) -> AdvancedLogger:
    """Logger al"""
    return log_manager.get_logger(name, config)

def log_system(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Sistem logu"""
    logger = get_logger("system")
    getattr(logger, level.value.lower())(LogCategory.SYSTEM, message, extra_data)

def log_api(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """API logu"""
    logger = get_logger("api")
    getattr(logger, level.value.lower())(LogCategory.API, message, extra_data)

def log_trading(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Trading logu"""
    logger = get_logger("trading")
    getattr(logger, level.value.lower())(LogCategory.TRADING, message, extra_data)

def log_strategy(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Strateji logu"""
    logger = get_logger("strategy")
    getattr(logger, level.value.lower())(LogCategory.STRATEGY, message, extra_data)

def log_risk(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Risk logu"""
    logger = get_logger("risk")
    getattr(logger, level.value.lower())(LogCategory.RISK, message, extra_data)

def log_data(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Veri logu"""
    logger = get_logger("data")
    getattr(logger, level.value.lower())(LogCategory.DATA, message, extra_data)

def log_ai(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """AI logu"""
    logger = get_logger("ai")
    getattr(logger, level.value.lower())(LogCategory.AI, message, extra_data)

def log_gui(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """GUI logu"""
    logger = get_logger("gui")
    getattr(logger, level.value.lower())(LogCategory.GUI, message, extra_data)

def log_backtest(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Backtest logu"""
    logger = get_logger("backtest")
    getattr(logger, level.value.lower())(LogCategory.BACKTEST, message, extra_data)

def log_performance(message: str, level: LogLevel = LogLevel.INFO, extra_data: Dict[str, Any] = None):
    """Performans logu"""
    logger = get_logger("performance")
    getattr(logger, level.value.lower())(LogCategory.PERFORMANCE, message, extra_data)




