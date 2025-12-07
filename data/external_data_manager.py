"""
Harici Veri Kaynakları Entegrasyonu Modülü
Glassnode, CoinGecko ve diğer harici API'lerle entegrasyon
"""

import requests
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import threading
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os
from pathlib import Path
import pickle

@dataclass
class ExternalDataConfig:
    """Harici veri konfigürasyonu"""
    glassnode_api_key: str = ""
    coingecko_api_key: str = ""
    cryptocompare_api_key: str = ""
    coinmarketcap_api_key: str = ""
    update_interval: int = 300  # 5 dakika
    cache_duration: int = 3600  # 1 saat
    max_retries: int = 3
    timeout: int = 30

class DataProvider(ABC):
    """Veri sağlayıcı temel sınıfı"""
    
    @abstractmethod
    def get_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class GlassnodeProvider(DataProvider):
    """Glassnode API sağlayıcısı"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1"
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params['api_key'] = self.api_key
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Glassnode API hatası: {e}")
            return {}
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_btc_supply_metrics(self) -> Dict[str, Any]:
        """BTC arz metrikleri"""
        try:
            data = self.get_data("metrics/supply/current", {
                'a': 'BTC',
                'f': 'native'
            })
            return data
        except Exception as e:
            self.logger.error(f"BTC arz metrikleri alma hatası: {e}")
            return {}
    
    def get_btc_flow_metrics(self) -> Dict[str, Any]:
        """BTC akış metrikleri"""
        try:
            data = self.get_data("metrics/transactions/count", {
                'a': 'BTC',
                'f': 'native'
            })
            return data
        except Exception as e:
            self.logger.error(f"BTC akış metrikleri alma hatası: {e}")
            return {}

class CoinGeckoProvider(DataProvider):
    """CoinGecko API sağlayıcısı"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.api_key:
                params['api_key'] = self.api_key
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"CoinGecko API hatası: {e}")
            return {}
    
    def is_available(self) -> bool:
        return True  # CoinGecko ücretsiz API
    
    def get_market_data(self, coin_id: str = "bitcoin") -> Dict[str, Any]:
        """Piyasa verileri"""
        try:
            data = self.get_data(f"coins/{coin_id}", {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            })
            return data
        except Exception as e:
            self.logger.error(f"Piyasa verileri alma hatası: {e}")
            return {}
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Fear & Greed Index"""
        try:
            data = self.get_data("fear_greed_index", {})
            return data
        except Exception as e:
            self.logger.error(f"Fear & Greed Index alma hatası: {e}")
            return {}

class CryptoCompareProvider(DataProvider):
    """CryptoCompare API sağlayıcısı"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params['api_key'] = self.api_key
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"CryptoCompare API hatası: {e}")
            return {}
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_news_sentiment(self) -> Dict[str, Any]:
        """Haber duyarlılığı"""
        try:
            data = self.get_data("news/sentiment", {
                'feeds': 'coindesk,cointelegraph,bitcoinmagazine',
                'limit': '100'
            })
            return data
        except Exception as e:
            self.logger.error(f"Haber duyarlılığı alma hatası: {e}")
            return {}

class ExternalDataManager:
    """Harici veri yöneticisi"""
    
    def __init__(self, config: ExternalDataConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or ExternalDataConfig()
        
        # Veri sağlayıcıları
        self.providers = {
            'glassnode': GlassnodeProvider(self.config.glassnode_api_key),
            'coingecko': CoinGeckoProvider(self.config.coingecko_api_key),
            'cryptocompare': CryptoCompareProvider(self.config.cryptocompare_api_key)
        }
        
        # Veri önbelleği
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Callback'ler
        self.data_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info("Harici veri yöneticisi başlatıldı")
    
    def get_btc_onchain_metrics(self) -> Dict[str, Any]:
        """BTC on-chain metrikleri"""
        try:
            metrics = {}
            
            # Glassnode'dan BTC metrikleri
            if self.providers['glassnode'].is_available():
                supply_metrics = self.providers['glassnode'].get_btc_supply_metrics()
                flow_metrics = self.providers['glassnode'].get_btc_flow_metrics()
                
                metrics.update({
                    'supply_metrics': supply_metrics,
                    'flow_metrics': flow_metrics
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"BTC on-chain metrikleri alma hatası: {e}")
            return {}
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Piyasa duyarlılığı"""
        try:
            sentiment = {}
            
            # CoinGecko'dan Fear & Greed Index
            if self.providers['coingecko'].is_available():
                fear_greed = self.providers['coingecko'].get_fear_greed_index()
                sentiment['fear_greed_index'] = fear_greed
            
            # CryptoCompare'dan haber duyarlılığı
            if self.providers['cryptocompare'].is_available():
                news_sentiment = self.providers['cryptocompare'].get_news_sentiment()
                sentiment['news_sentiment'] = news_sentiment
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Piyasa duyarlılığı alma hatası: {e}")
            return {}
    
    def get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Kapsamlı piyasa verileri"""
        try:
            market_data = {}
            
            # BTC piyasa verileri
            if self.providers['coingecko'].is_available():
                btc_data = self.providers['coingecko'].get_market_data("bitcoin")
                market_data['btc_market_data'] = btc_data
            
            # On-chain metrikler
            onchain_metrics = self.get_btc_onchain_metrics()
            market_data['onchain_metrics'] = onchain_metrics
            
            # Piyasa duyarlılığı
            sentiment = self.get_market_sentiment()
            market_data['sentiment'] = sentiment
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Kapsamlı piyasa verileri alma hatası: {e}")
            return {}
    
    def add_data_callback(self, callback: Callable):
        """Veri callback'i ekle"""
        self.data_callbacks.append(callback)
    
    def _notify_data_callbacks(self, data: Dict[str, Any]):
        """Veri callback'lerini çağır"""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Veri callback hatası: {e}")

# Global harici veri yöneticisi
external_data_manager = ExternalDataManager()
