"""
Eklenti Yöneticisi
Modüler eklenti sistemi ve dinamik yükleme
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import json
import threading
from dataclasses import dataclass
from enum import Enum
import logging

from utils.logger import get_logger, LogLevel, LogCategory

class PluginType(Enum):
    """Eklenti türleri"""
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    DATA_SOURCE = "data_source"
    RISK_MANAGER = "risk_manager"
    NOTIFICATION = "notification"
    EXPORT = "export"
    CUSTOM = "custom"

class PluginStatus(Enum):
    """Eklenti durumu"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class PluginInfo:
    """Eklenti bilgisi"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    config_schema: Dict[str, Any]
    status: PluginStatus
    file_path: str
    loaded_at: datetime
    last_updated: datetime

class BasePlugin:
    """Temel eklenti sınıfı"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"plugin.{name}")
        self.is_active = False
        
    def initialize(self) -> bool:
        """Eklentiyi başlat"""
        try:
            self.logger.info(LogCategory.SYSTEM, f"Eklenti başlatılıyor: {self.name}")
            return self._on_initialize()
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti başlatma hatası: {e}")
            return False
    
    def activate(self) -> bool:
        """Eklentiyi etkinleştir"""
        try:
            self.logger.info(LogCategory.SYSTEM, f"Eklenti etkinleştiriliyor: {self.name}")
            self.is_active = True
            return self._on_activate()
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti etkinleştirme hatası: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Eklentiyi devre dışı bırak"""
        try:
            self.logger.info(LogCategory.SYSTEM, f"Eklenti devre dışı bırakılıyor: {self.name}")
            self.is_active = False
            return self._on_deactivate()
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti devre dışı bırakma hatası: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Eklentiyi temizle"""
        try:
            self.logger.info(LogCategory.SYSTEM, f"Eklenti temizleniyor: {self.name}")
            return self._on_cleanup()
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti temizleme hatası: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            return self._on_config_update(new_config)
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Konfigürasyon güncelleme hatası: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Eklenti bilgilerini al"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'config': self.config
        }
    
    # Alt sınıflarda override edilecek metodlar
    def _on_initialize(self) -> bool:
        """Başlatma işlemi"""
        return True
    
    def _on_activate(self) -> bool:
        """Etkinleştirme işlemi"""
        return True
    
    def _on_deactivate(self) -> bool:
        """Devre dışı bırakma işlemi"""
        return True
    
    def _on_cleanup(self) -> bool:
        """Temizleme işlemi"""
        return True
    
    def _on_config_update(self, new_config: Dict[str, Any]) -> bool:
        """Konfigürasyon güncelleme işlemi"""
        return True

class PluginManager:
    """Eklenti yöneticisi sınıfı"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = plugins_dir
        self.logger = get_logger("plugin_manager")
        
        # Eklenti kayıtları
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        
        # Eklenti türleri
        self.plugin_types: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Eklenti dizinini oluştur
        os.makedirs(plugins_dir, exist_ok=True)
        
        self.logger.info(LogCategory.SYSTEM, "Eklenti yöneticisi başlatıldı")
    
    def load_plugin(self, plugin_file: str, config: Dict[str, Any] = None) -> bool:
        """Eklenti yükle"""
        try:
            with self.lock:
                plugin_name = os.path.splitext(os.path.basename(plugin_file))[0]
                
                if plugin_name in self.plugins:
                    self.logger.warning(LogCategory.SYSTEM, f"Eklenti zaten yüklü: {plugin_name}")
                    return True
                
                # Eklenti modülünü yükle
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if not spec or not spec.loader:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti modülü yüklenemedi: {plugin_file}")
                    return False
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Eklenti sınıfını bul
                plugin_class = self._find_plugin_class(module)
                if not plugin_class:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti sınıfı bulunamadı: {plugin_file}")
                    return False
                
                # Eklenti bilgilerini al
                plugin_info = self._extract_plugin_info(module, plugin_file)
                if not plugin_info:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti bilgileri alınamadı: {plugin_file}")
                    return False
                
                # Eklenti instance'ı oluştur
                plugin_instance = plugin_class(plugin_name, config or {})
                
                # Eklentiyi başlat
                if not plugin_instance.initialize():
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti başlatılamadı: {plugin_name}")
                    return False
                
                # Kaydet
                self.plugins[plugin_name] = plugin_instance
                self.plugin_info[plugin_name] = plugin_info
                self.plugin_types[plugin_info.plugin_type].append(plugin_name)
                
                self.logger.info(LogCategory.SYSTEM, f"Eklenti yüklendi: {plugin_name}")
                return True
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti yükleme hatası: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Eklenti kaldır"""
        try:
            with self.lock:
                if plugin_name not in self.plugins:
                    self.logger.warning(LogCategory.SYSTEM, f"Eklenti bulunamadı: {plugin_name}")
                    return False
                
                plugin = self.plugins[plugin_name]
                plugin_info = self.plugin_info[plugin_name]
                
                # Eklentiyi devre dışı bırak
                if plugin.is_active:
                    plugin.deactivate()
                
                # Eklentiyi temizle
                plugin.cleanup()
                
                # Kayıtlardan kaldır
                del self.plugins[plugin_name]
                del self.plugin_info[plugin_name]
                self.plugin_types[plugin_info.plugin_type].remove(plugin_name)
                
                self.logger.info(LogCategory.SYSTEM, f"Eklenti kaldırıldı: {plugin_name}")
                return True
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti kaldırma hatası: {e}")
            return False
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Eklentiyi etkinleştir"""
        try:
            with self.lock:
                if plugin_name not in self.plugins:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti bulunamadı: {plugin_name}")
                    return False
                
                plugin = self.plugins[plugin_name]
                
                if plugin.is_active:
                    self.logger.warning(LogCategory.SYSTEM, f"Eklenti zaten aktif: {plugin_name}")
                    return True
                
                if plugin.activate():
                    self.plugin_info[plugin_name].status = PluginStatus.ACTIVE
                    self.logger.info(LogCategory.SYSTEM, f"Eklenti etkinleştirildi: {plugin_name}")
                    return True
                else:
                    self.plugin_info[plugin_name].status = PluginStatus.ERROR
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti etkinleştirilemedi: {plugin_name}")
                    return False
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti etkinleştirme hatası: {e}")
            return False
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Eklentiyi devre dışı bırak"""
        try:
            with self.lock:
                if plugin_name not in self.plugins:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti bulunamadı: {plugin_name}")
                    return False
                
                plugin = self.plugins[plugin_name]
                
                if not plugin.is_active:
                    self.logger.warning(LogCategory.SYSTEM, f"Eklenti zaten devre dışı: {plugin_name}")
                    return True
                
                if plugin.deactivate():
                    self.plugin_info[plugin_name].status = PluginStatus.INACTIVE
                    self.logger.info(LogCategory.SYSTEM, f"Eklenti devre dışı bırakıldı: {plugin_name}")
                    return True
                else:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti devre dışı bırakılamadı: {plugin_name}")
                    return False
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti devre dışı bırakma hatası: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Eklenti al"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Türe göre eklentileri al"""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def get_active_plugins(self) -> List[BasePlugin]:
        """Aktif eklentileri al"""
        return [plugin for plugin in self.plugins.values() if plugin.is_active]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Eklenti bilgilerini al"""
        return self.plugin_info.get(plugin_name)
    
    def get_all_plugin_info(self) -> Dict[str, PluginInfo]:
        """Tüm eklenti bilgilerini al"""
        return self.plugin_info.copy()
    
    def update_plugin_config(self, plugin_name: str, new_config: Dict[str, Any]) -> bool:
        """Eklenti konfigürasyonunu güncelle"""
        try:
            with self.lock:
                if plugin_name not in self.plugins:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti bulunamadı: {plugin_name}")
                    return False
                
                plugin = self.plugins[plugin_name]
                
                if plugin.update_config(new_config):
                    self.logger.info(LogCategory.SYSTEM, f"Eklenti konfigürasyonu güncellendi: {plugin_name}")
                    return True
                else:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti konfigürasyonu güncellenemedi: {plugin_name}")
                    return False
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti konfigürasyon güncelleme hatası: {e}")
            return False
    
    def load_plugins_from_directory(self, directory: str = None) -> int:
        """Dizinden eklentileri yükle"""
        try:
            if not directory:
                directory = self.plugins_dir
            
            loaded_count = 0
            
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('_'):
                    # Yonetici dosyasini atla
                    if filename in ('plugin_manager.py',):
                        continue
                    plugin_file = os.path.join(directory, filename)
                    
                    if self.load_plugin(plugin_file):
                        loaded_count += 1
            
            self.logger.info(LogCategory.SYSTEM, f"{loaded_count} eklenti yüklendi")
            return loaded_count
            
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Dizinden eklenti yükleme hatası: {e}")
            return 0
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Eklentiyi yeniden yükle"""
        try:
            with self.lock:
                if plugin_name not in self.plugins:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti bulunamadı: {plugin_name}")
                    return False
                
                plugin_info = self.plugin_info[plugin_name]
                plugin_file = plugin_info.file_path
                config = self.plugins[plugin_name].config
                
                # Eklentiyi kaldır
                if not self.unload_plugin(plugin_name):
                    return False
                
                # Eklentiyi yeniden yükle
                if self.load_plugin(plugin_file, config):
                    self.logger.info(LogCategory.SYSTEM, f"Eklenti yeniden yüklendi: {plugin_name}")
                    return True
                else:
                    self.logger.error(LogCategory.SYSTEM, f"Eklenti yeniden yüklenemedi: {plugin_name}")
                    return False
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti yeniden yükleme hatası: {e}")
            return False
    
    def _find_plugin_class(self, module) -> Optional[Type[BasePlugin]]:
        """Eklenti sınıfını bul"""
        try:
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    return obj
            return None
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti sınıfı bulma hatası: {e}")
            return None
    
    def _extract_plugin_info(self, module, file_path: str) -> Optional[PluginInfo]:
        """Eklenti bilgilerini çıkar"""
        try:
            # Modül metadata'sını al
            name = getattr(module, '__name__', 'unknown')
            version = getattr(module, '__version__', '1.0.0')
            description = getattr(module, '__description__', 'No description')
            author = getattr(module, '__author__', 'Unknown')
            
            # Eklenti türünü belirle
            plugin_type = PluginType.CUSTOM
            if hasattr(module, '__plugin_type__'):
                plugin_type = PluginType(module.__plugin_type__)
            
            # Bağımlılıkları al
            dependencies = getattr(module, '__dependencies__', [])
            
            # Konfigürasyon şeması
            config_schema = getattr(module, '__config_schema__', {})
            
            return PluginInfo(
                name=name,
                version=version,
                description=description,
                author=author,
                plugin_type=plugin_type,
                dependencies=dependencies,
                config_schema=config_schema,
                status=PluginStatus.LOADED,
                file_path=file_path,
                loaded_at=datetime.now(),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti bilgisi çıkarma hatası: {e}")
            return None
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Eklenti istatistiklerini al"""
        try:
            with self.lock:
                total_plugins = len(self.plugins)
                active_plugins = len([p for p in self.plugins.values() if p.is_active])
                
                by_type = {}
                for plugin_type, plugin_names in self.plugin_types.items():
                    by_type[plugin_type.value] = len(plugin_names)
                
                by_status = {}
                for plugin_info in self.plugin_info.values():
                    status = plugin_info.status.value
                    by_status[status] = by_status.get(status, 0) + 1
                
                return {
                    'total_plugins': total_plugins,
                    'active_plugins': active_plugins,
                    'inactive_plugins': total_plugins - active_plugins,
                    'by_type': by_type,
                    'by_status': by_status
                }
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti istatistikleri alma hatası: {e}")
            return {}
    
    def cleanup_all_plugins(self):
        """Tüm eklentileri temizle"""
        try:
            with self.lock:
                for plugin_name in list(self.plugins.keys()):
                    self.unload_plugin(plugin_name)
                
                self.logger.info(LogCategory.SYSTEM, "Tüm eklentiler temizlendi")
                
        except Exception as e:
            self.logger.error(LogCategory.SYSTEM, f"Eklenti temizleme hatası: {e}")

# Global eklenti yöneticisi
plugin_manager = PluginManager()

# Kolay kullanım için fonksiyonlar
def load_plugin(plugin_file: str, config: Dict[str, Any] = None) -> bool:
    """Eklenti yükle"""
    return plugin_manager.load_plugin(plugin_file, config)

def unload_plugin(plugin_name: str) -> bool:
    """Eklenti kaldır"""
    return plugin_manager.unload_plugin(plugin_name)

def activate_plugin(plugin_name: str) -> bool:
    """Eklentiyi etkinleştir"""
    return plugin_manager.activate_plugin(plugin_name)

def deactivate_plugin(plugin_name: str) -> bool:
    """Eklentiyi devre dışı bırak"""
    return plugin_manager.deactivate_plugin(plugin_name)

def get_plugin(plugin_name: str) -> Optional[BasePlugin]:
    """Eklenti al"""
    return plugin_manager.get_plugin(plugin_name)

def get_plugins_by_type(plugin_type: PluginType) -> List[BasePlugin]:
    """Türe göre eklentileri al"""
    return plugin_manager.get_plugins_by_type(plugin_type)




