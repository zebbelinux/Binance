"""
Konfigürasyon Validasyon Modülü
JSON konfigürasyon dosyalarının şema doğrulaması
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

class ConfigType(Enum):
    """Konfigürasyon türleri"""
    API_KEYS = "api_keys"
    SETTINGS = "settings"
    STRATEGY = "strategy"
    RISK = "risk"

@dataclass
class ValidationRule:
    """Validasyon kuralı"""
    field: str
    required: bool = True
    field_type: type = str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None

class ConfigValidator:
    """Konfigürasyon validatörü"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[ConfigType, List[ValidationRule]]:
        """Validasyon kurallarını yükle"""
        return {
            ConfigType.API_KEYS: [
                ValidationRule("btcturk_api_key", required=True, field_type=str),
                ValidationRule("btcturk_secret_key", required=True, field_type=str),
                ValidationRule("deepseek_api_key", required=False, field_type=str),
                ValidationRule("test_mode", required=True, field_type=bool)
            ],
            ConfigType.SETTINGS: [
                ValidationRule("trading_enabled", required=True, field_type=bool),
                ValidationRule("max_position_size", required=True, field_type=float, min_value=0.01, max_value=1.0),
                ValidationRule("risk_per_trade", required=True, field_type=float, min_value=0.001, max_value=0.1),
                ValidationRule("max_daily_loss", required=True, field_type=float, min_value=0.01, max_value=0.5),
                ValidationRule("update_interval", required=True, field_type=int, min_value=1, max_value=3600)
            ],
            ConfigType.STRATEGY: [
                ValidationRule("strategy_name", required=True, field_type=str, 
                             allowed_values=["scalping", "grid", "trend_following", "dca", "hedge"]),
                ValidationRule("enabled", required=True, field_type=bool),
                ValidationRule("max_position_size", required=True, field_type=float, min_value=0.01, max_value=1.0)
            ],
            ConfigType.RISK: [
                ValidationRule("max_drawdown", required=True, field_type=float, min_value=0.01, max_value=0.5),
                ValidationRule("stop_loss_pct", required=True, field_type=float, min_value=0.001, max_value=0.2),
                ValidationRule("take_profit_pct", required=True, field_type=float, min_value=0.001, max_value=1.0),
                ValidationRule("max_trades_per_day", required=True, field_type=int, min_value=1, max_value=1000)
            ]
        }
    
    def validate_config_file(self, file_path: str, config_type: ConfigType) -> Dict[str, Any]:
        """Konfigürasyon dosyasını doğrula"""
        try:
            # Dosya varlığını kontrol et
            if not os.path.exists(file_path):
                return {
                    'valid': False,
                    'errors': [f"Dosya bulunamadı: {file_path}"],
                    'warnings': []
                }
            
            # JSON dosyasını yükle
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validasyon kurallarını uygula
            return self._validate_config_data(config_data, config_type)
            
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'errors': [f"JSON parse hatası: {e}"],
                'warnings': []
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validasyon hatası: {e}"],
                'warnings': []
            }
    
    def _validate_config_data(self, config_data: Dict[str, Any], config_type: ConfigType) -> Dict[str, Any]:
        """Konfigürasyon verisini doğrula"""
        errors = []
        warnings = []
        
        rules = self.validation_rules.get(config_type, [])
        
        for rule in rules:
            # Zorunlu alan kontrolü
            if rule.required and rule.field not in config_data:
                errors.append(f"Zorunlu alan eksik: {rule.field}")
                continue
            
            if rule.field not in config_data:
                continue
            
            value = config_data[rule.field]
            
            # Tip kontrolü
            if not isinstance(value, rule.field_type):
                errors.append(f"'{rule.field}' alanı {rule.field_type.__name__} tipinde olmalı, {type(value).__name__} bulundu")
                continue
            
            # Sayısal değer kontrolleri
            if isinstance(value, (int, float)):
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(f"'{rule.field}' değeri {rule.min_value}'dan küçük olamaz")
                
                if rule.max_value is not None and value > rule.max_value:
                    errors.append(f"'{rule.field}' değeri {rule.max_value}'dan büyük olamaz")
            
            # İzin verilen değerler kontrolü
            if rule.allowed_values is not None and value not in rule.allowed_values:
                errors.append(f"'{rule.field}' değeri {rule.allowed_values} arasında olmalı")
            
            # Pattern kontrolü (string için)
            if rule.pattern and isinstance(value, str):
                import re
                if not re.match(rule.pattern, value):
                    errors.append(f"'{rule.field}' değeri geçersiz format")
        
        # Ek validasyonlar
        if config_type == ConfigType.API_KEYS:
            self._validate_api_keys(config_data, errors, warnings)
        elif config_type == ConfigType.SETTINGS:
            self._validate_settings(config_data, errors, warnings)
        elif config_type == ConfigType.RISK:
            self._validate_risk_settings(config_data, errors, warnings)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_api_keys(self, config_data: Dict[str, Any], errors: List[str], warnings: List[str]):
        """API anahtarları özel validasyonu"""
        # API anahtarı format kontrolü
        api_key = config_data.get('btcturk_api_key', '')
        if api_key and not api_key.startswith('sk-'):
            warnings.append("BTCTURK API anahtarı beklenen formatta değil")
        
        # Test mode kontrolü
        if config_data.get('test_mode', False):
            warnings.append("Test modu aktif - gerçek işlem yapılmayacak")
    
    def _validate_settings(self, config_data: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Ayarlar özel validasyonu"""
        # Pozisyon büyüklüğü kontrolü
        max_pos = config_data.get('max_position_size', 0)
        risk_per_trade = config_data.get('risk_per_trade', 0)
        
        if max_pos > 0.5:
            warnings.append("Maksimum pozisyon büyüklüğü %50'den yüksek")
        
        if risk_per_trade > 0.05:
            warnings.append("İşlem başına risk %5'ten yüksek")
    
    def _validate_risk_settings(self, config_data: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Risk ayarları özel validasyonu"""
        stop_loss = config_data.get('stop_loss_pct', 0)
        take_profit = config_data.get('take_profit_pct', 0)
        
        if stop_loss > 0.1:
            warnings.append("Stop loss %10'dan yüksek")
        
        if take_profit < stop_loss * 2:
            warnings.append("Take profit stop loss'un en az 2 katı olmalı")
    
    def validate_all_configs(self, config_dir: str = "config") -> Dict[str, Any]:
        """Tüm konfigürasyon dosyalarını doğrula"""
        results = {}
        
        config_files = {
            "api_keys.json": ConfigType.API_KEYS,
            "settings.json": ConfigType.SETTINGS,
            "risk_settings.json": ConfigType.RISK
        }
        
        for filename, config_type in config_files.items():
            file_path = os.path.join(config_dir, filename)
            results[filename] = self.validate_config_file(file_path, config_type)
        
        return results
    
    def create_default_config(self, config_type: ConfigType, file_path: str) -> bool:
        """Varsayılan konfigürasyon dosyası oluştur"""
        try:
            default_configs = {
                ConfigType.API_KEYS: {
                    "btcturk_api_key": "",
                    "btcturk_secret_key": "",
                    "deepseek_api_key": "",
                    "test_mode": True
                },
                ConfigType.SETTINGS: {
                    "trading_enabled": False,
                    "max_position_size": 0.1,
                    "risk_per_trade": 0.02,
                    "max_daily_loss": 0.1,
                    "update_interval": 60
                },
                ConfigType.RISK: {
                    "max_drawdown": 0.2,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.04,
                    "max_trades_per_day": 100
                }
            }
            
            default_config = default_configs.get(config_type, {})
            
            # Dizin oluştur
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Dosyayı yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Varsayılan konfigürasyon oluşturuldu: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Varsayılan konfigürasyon oluşturma hatası: {e}")
            return False

# Global validatör
config_validator = ConfigValidator()
