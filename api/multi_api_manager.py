"""
Çoklu BTCTURK API Yönetimi
Birden fazla API key ile load balancing ve failover
"""

import json
import requests
import time
import threading
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random

from api.btcturk_api import BTCTurkAPI
from api.binance_api import BinanceAPI

class APIStatus(Enum):
    """API durumu"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class APIKey:
    """API Key bilgileri"""
    name: str
    api_key: str
    secret_key: str
    status: APIStatus = APIStatus.ACTIVE
    daily_requests: int = 0
    daily_limit: int = 1000
    last_request_time: float = 0
    error_count: int = 0
    max_errors: int = 5
    created_at: datetime = None
    last_used: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()

class MultiAPIManager:
    """Çoklu API yöneticisi"""
    
    def __init__(self):
        self.api_keys: List[APIKey] = []
        self.current_api_index = 0
        self.load_balancing_strategy = "round_robin"  # round_robin, random, least_used
        self.failover_enabled = True
        self.rate_limit_buffer = 0.8  # %80 kullanımda geçiş
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        # API instance'ları
        self.api_instances: Dict[str, BTCTurkAPI] = {}
        # Public market data provider (Binance Spot) for PAPER mode
        self.market_api = BinanceAPI()
        
        # İstatistikler
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.api_switches = 0
    
    def add_api_key(self, name: str, api_key: str, secret_key: str, 
                   daily_limit: int = 1000, max_errors: int = 5) -> bool:
        """Yeni API key ekle"""
        try:
            with self.lock:
                # Aynı isimde API key var mı kontrol et
                if any(key.name == name for key in self.api_keys):
                    self.logger.warning(f"API key '{name}' zaten mevcut")
                    return False
                # Yeni kayıt oluştur
                new_key = APIKey(
                    name=name,
                    api_key=api_key,
                    secret_key=secret_key,
                    daily_limit=daily_limit,
                    max_errors=max_errors
                )
                self.api_keys.append(new_key)
                # API instance oluştur ve bağla
                api_instance = BTCTurkAPI()
                api_instance.api_key = api_key
                api_instance.secret_key = secret_key
                self.api_instances[name] = api_instance
                self.logger.info(f"API key '{name}' başarıyla eklendi")
                return True
        
        except Exception as e:
            self.logger.error(f"API key ekleme hatası: {e}")
            return False

    def get_api_key(self, name: str) -> Optional[APIKey]:
        """İsme göre APIKey döndür"""
        try:
            with self.lock:
                for key in self.api_keys:
                    if key.name == name:
                        return key
            return None
        except Exception:
            return None

    def update_api_key(self, name: str, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                       daily_limit: Optional[int] = None, max_errors: Optional[int] = None) -> bool:
        """Var olan API key kaydını güvenle güncelle. None gelen alanlar korunur."""
        try:
            with self.lock:
                target = None
                for key in self.api_keys:
                    if key.name == name:
                        target = key
                        break
                if not target:
                    self.logger.warning(f"API key '{name}' bulunamadı (update)")
                    return False

                # Alanları güncelle
                if api_key is not None and api_key != "***" and api_key != "":
                    target.api_key = api_key
                if secret_key is not None and secret_key != "***" and secret_key != "":
                    target.secret_key = secret_key
                if isinstance(daily_limit, int) and daily_limit > 0:
                    target.daily_limit = daily_limit
                if isinstance(max_errors, int) and max_errors >= 0:
                    target.max_errors = max_errors

                # API instance'ını eşitle
                inst = self.api_instances.get(name)
                if not inst:
                    inst = BTCTurkAPI()
                    self.api_instances[name] = inst
                inst.api_key = target.api_key
                inst.secret_key = target.secret_key

                self.logger.info(f"API key '{name}' güncellendi")
                return True
        except Exception as e:
            self.logger.error(f"API key güncelleme hatası: {e}")
            return False
    
    def remove_api_key(self, name: str) -> bool:
        """API key kaldır"""
        try:
            with self.lock:
                # API key'i bul ve kaldır
                key_to_remove = None
                for key in self.api_keys:
                    if key.name == name:
                        key_to_remove = key
                        break
                
                if key_to_remove:
                    self.api_keys.remove(key_to_remove)
                    
                    # API instance'ı kaldır
                    if name in self.api_instances:
                        del self.api_instances[name]
                    
                    self.logger.info(f"API key '{name}' kaldırıldı")
                    return True
                else:
                    self.logger.warning(f"API key '{name}' bulunamadı")
                    return False
                    
        except Exception as e:
            self.logger.error(f"API key kaldırma hatası: {e}")
            return False
    
    def get_active_api_key(self) -> Optional[APIKey]:
        """Aktif API key'i al"""
        with self.lock:
            if not self.api_keys:
                return None
            
            # Aktif API key'leri filtrele
            active_keys = [key for key in self.api_keys if key.status == APIStatus.ACTIVE]
            
            if not active_keys:
                self.logger.warning("Aktif API key bulunamadı")
                return None
            
            # Load balancing stratejisine göre seç
            if self.load_balancing_strategy == "round_robin":
                return self._round_robin_selection(active_keys)
            elif self.load_balancing_strategy == "random":
                return random.choice(active_keys)
            elif self.load_balancing_strategy == "least_used":
                return self._least_used_selection(active_keys)
            else:
                return active_keys[0]
    
    def _round_robin_selection(self, active_keys: List[APIKey]) -> APIKey:
        """Round-robin seçimi"""
        if self.current_api_index >= len(active_keys):
            self.current_api_index = 0
        
        selected_key = active_keys[self.current_api_index]
        self.current_api_index = (self.current_api_index + 1) % len(active_keys)
        return selected_key
    
    def _least_used_selection(self, active_keys: List[APIKey]) -> APIKey:
        """En az kullanılan API key'i seç"""
        return min(active_keys, key=lambda x: x.daily_requests)
    
    def _check_rate_limit(self, api_key: APIKey) -> bool:
        """Rate limit kontrolü"""
        usage_ratio = api_key.daily_requests / api_key.daily_limit
        return usage_ratio < self.rate_limit_buffer
    
    def _update_api_key_stats(self, api_key: APIKey, success: bool):
        """API key istatistiklerini güncelle"""
        api_key.last_used = datetime.now()
        api_key.last_request_time = time.time()
        
        if success:
            api_key.daily_requests += 1
            api_key.error_count = 0
        else:
            api_key.error_count += 1
            
            # Hata sayısı limiti aştıysa API'yi deaktif et
            if api_key.error_count >= api_key.max_errors:
                api_key.status = APIStatus.ERROR
                self.logger.warning(f"API key '{api_key.name}' hata limiti aşıldı, deaktif edildi")
    
    def _switch_to_next_api(self):
        """Sonraki API'ye geç"""
        with self.lock:
            self.api_switches += 1
            self.current_api_index = (self.current_api_index + 1) % len(self.api_keys)
            self.logger.info(f"API geçişi yapıldı. Toplam geçiş: {self.api_switches}")
    
    def make_request(self, method: str, endpoint: str, params: dict = None, data: dict = None) -> dict:
        """API isteği gönder (load balancing ile)"""
        max_retries = 3
        retry_count = 0
        ep = (endpoint or '').strip()
        # Route public market data to Binance Spot in paper mode
        if ep in ("/ticker", "/orderbook", "/trades", "/klines"):
            try:
                if method.upper() == "GET":
                    return self.market_api._make_request(method, ep, params=params)
                else:
                    return self.market_api._make_request(method, ep, data=data)
            except Exception as e:
                self.logger.error(f"Binance market request error: {e}")
                # fallback to legacy path below if needed
        
        while retry_count < max_retries:
            api_key = self.get_active_api_key()
            
            if not api_key:
                return {"error": "No active API keys available"}
            
            # Rate limit kontrolü
            if not self._check_rate_limit(api_key):
                api_key.status = APIStatus.RATE_LIMITED
                self.logger.warning(f"API key '{api_key.name}' rate limit aşıldı")
                self._switch_to_next_api()
                retry_count += 1
                continue
            
            # API instance'ı al
            api_instance = self.api_instances.get(api_key.name)
            if not api_instance:
                self.logger.error(f"API instance '{api_key.name}' bulunamadı")
                retry_count += 1
                continue
            
            # İsteği gönder
            try:
                if method.upper() == "GET":
                    response = api_instance._make_request(method, endpoint, params=params)
                else:
                    response = api_instance._make_request(method, endpoint, data=data)
                
                # Başarılı istek
                if "error" not in response:
                    self._update_api_key_stats(api_key, True)
                    self.total_requests += 1
                    self.successful_requests += 1
                    return response
                else:
                    # Hatalı istek
                    self._update_api_key_stats(api_key, False)
                    self.total_requests += 1
                    self.failed_requests += 1
                    
                    if self.failover_enabled:
                        self._switch_to_next_api()
                        retry_count += 1
                        continue
                    else:
                        return response
                        
            except Exception as e:
                self.logger.error(f"API isteği hatası: {e}")
                self._update_api_key_stats(api_key, False)
                self.total_requests += 1
                self.failed_requests += 1
                
                if self.failover_enabled:
                    self._switch_to_next_api()
                    retry_count += 1
                    continue
                else:
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def get_ticker(self, symbol: str = None) -> dict:
        """Ticker bilgilerini al (public endpoint - anahtar istatistiklerini etkilemez)"""
        try:
            # Use Binance normalized layer
            params = {"pairSymbol": symbol} if symbol else {}
            return self.market_api._make_request("GET", "/ticker", params=params)
        except Exception as e:
            self.logger.error(f"Public ticker hatası: {e}")
            return {"error": str(e)}
    
    def get_orderbook(self, symbol: str) -> dict:
        """Emir defterini al"""
        try:
            return self.market_api._make_request("GET", "/orderbook", params={"pairSymbol": symbol})
        except Exception:
            endpoint = "/orderbook"
            params = {"pairSymbol": symbol}
            return self.make_request("GET", endpoint, params=params)
    
    def get_balance(self) -> dict:
        """Hesap bakiyesini al"""
        endpoint = "/users/balances"
        return self.make_request("GET", endpoint)
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                   quantity: float, price: float = None, stop_price: float = None) -> dict:
        """Emir ver"""
        endpoint = "/order"
        data = {
            "pairSymbol": symbol,
            "orderType": order_type,
            "orderMethod": side,
            "quantity": str(quantity)
        }
        
        if price:
            data["price"] = str(price)
        if stop_price:
            data["stopPrice"] = str(stop_price)
        
        return self.make_request("POST", endpoint, data=data)
    
    def get_api_status(self) -> Dict[str, Any]:
        """API durumlarını al"""
        with self.lock:
            status = {
                "total_apis": len(self.api_keys),
                "active_apis": len([k for k in self.api_keys if k.status == APIStatus.ACTIVE]),
                "rate_limited_apis": len([k for k in self.api_keys if k.status == APIStatus.RATE_LIMITED]),
                "error_apis": len([k for k in self.api_keys if k.status == APIStatus.ERROR]),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
                "api_switches": self.api_switches,
                "load_balancing_strategy": self.load_balancing_strategy,
                "failover_enabled": self.failover_enabled
            }
            
            # Her API key'in detaylı durumu
            api_details = []
            for key in self.api_keys:
                api_details.append({
                    "name": key.name,
                    "status": key.status.value,
                    "daily_requests": key.daily_requests,
                    "daily_limit": key.daily_limit,
                    "usage_percentage": (key.daily_requests / key.daily_limit * 100) if key.daily_limit > 0 else 0,
                    "error_count": key.error_count,
                    "last_used": key.last_used.isoformat() if key.last_used else None
                })
            
            status["api_details"] = api_details
            return status
    
    def reset_daily_counters(self):
        """Günlük sayaçları sıfırla"""
        with self.lock:
            for key in self.api_keys:
                key.daily_requests = 0
                key.error_count = 0
                if key.status == APIStatus.RATE_LIMITED:
                    key.status = APIStatus.ACTIVE
            
            self.logger.info("Günlük sayaçlar sıfırlandı")
    
    def set_load_balancing_strategy(self, strategy: str):
        """Load balancing stratejisini ayarla"""
        if strategy in ["round_robin", "random", "least_used"]:
            self.load_balancing_strategy = strategy
            self.logger.info(f"Load balancing stratejisi '{strategy}' olarak ayarlandı")
        else:
            self.logger.warning(f"Geçersiz load balancing stratejisi: {strategy}")
    
    def enable_failover(self, enabled: bool):
        """Failover'ı etkinleştir/devre dışı bırak"""
        self.failover_enabled = enabled
        self.logger.info(f"Failover {'etkinleştirildi' if enabled else 'devre dışı bırakıldı'}")
    
    def save_config(self, file_path: str = "config/api_keys.json"):
        """API key'leri dosyaya kaydet"""
        try:
            config_data = {
                "load_balancing_strategy": self.load_balancing_strategy,
                "failover_enabled": self.failover_enabled,
                "rate_limit_buffer": self.rate_limit_buffer,
                "api_keys": []
            }
            
            for key in self.api_keys:
                config_data["api_keys"].append({
                    "name": key.name,
                    "api_key": key.api_key,
                    "secret_key": key.secret_key,
                    "daily_limit": key.daily_limit,
                    "max_errors": key.max_errors
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"API konfigürasyonu '{file_path}' dosyasına kaydedildi")
            
        except Exception as e:
            self.logger.error(f"API konfigürasyonu kaydetme hatası: {e}")
    
    def load_config(self, file_path: str = "config/api_keys.json"):
        """API key'leri dosyadan yükle"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Genel ayarları yükle
            self.load_balancing_strategy = config_data.get("load_balancing_strategy", "round_robin")
            self.failover_enabled = config_data.get("failover_enabled", True)
            self.rate_limit_buffer = config_data.get("rate_limit_buffer", 0.8)
            
            # API key'leri yükle
            for key_data in config_data.get("api_keys", []):
                name = key_data.get("name")
                api_key_val = key_data.get("api_key")
                secret_key_val = key_data.get("secret_key")
                daily_limit = key_data.get("daily_limit", 1000)
                max_errors = key_data.get("max_errors", 5)
                if not name:
                    continue
                # Mevcut anahtar var mi? Guncelle, yoksa ekle
                existing = None
                for k in self.api_keys:
                    if k.name == name:
                        existing = k
                        break
                if existing:
                    # Mevcut kaydi guncelle
                    existing.api_key = api_key_val
                    existing.secret_key = secret_key_val
                    existing.daily_limit = daily_limit
                    existing.max_errors = max_errors
                    # API instance'i olustur/guncelle
                    api_instance = self.api_instances.get(name)
                    if not api_instance:
                        api_instance = BTCTurkAPI()
                        self.api_instances[name] = api_instance
                    api_instance.api_key = api_key_val
                    api_instance.secret_key = secret_key_val
                    self.logger.info(f"API key '{name}' güncellendi")
                else:
                    # Yeni ekle
                    self.add_api_key(
                        name=name,
                        api_key=api_key_val,
                        secret_key=secret_key_val,
                        daily_limit=daily_limit,
                        max_errors=max_errors
                    )
            
            self.logger.info(f"API konfigürasyonu '{file_path}' dosyasından yüklendi")
            
        except FileNotFoundError:
            self.logger.warning(f"API konfigürasyon dosyası bulunamadı: {file_path}")
        except Exception as e:
            self.logger.error(f"API konfigürasyonu yükleme hatası: {e}")

# Global multi-API manager instance
multi_api_manager = MultiAPIManager()
