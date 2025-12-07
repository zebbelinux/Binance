"""
Temel Analiz Modülü
Haber akışı, makroekonomik veriler ve BTC arz-talep analizi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
import requests
import json
import re
from collections import deque
import feedparser
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import os
from pathlib import Path
import pickle

@dataclass
class NewsItem:
    """Haber öğesi"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    sentiment_score: float
    relevance_score: float
    keywords: List[str]
    impact_level: str  # 'high', 'medium', 'low'

@dataclass
class MacroEconomicData:
    """Makroekonomik veri"""
    indicator_name: str
    value: float
    previous_value: float
    forecast: float
    unit: str
    timestamp: datetime
    impact: str  # 'positive', 'negative', 'neutral'
    source: str

@dataclass
class BitcoinSupplyDemand:
    """BTC arz-talep verisi"""
    timestamp: datetime
    circulating_supply: float
    total_supply: float
    max_supply: float
    active_addresses: int
    transaction_count: int
    hash_rate: float
    difficulty: float
    mining_revenue: float
    exchange_inflows: float
    exchange_outflows: float
    whale_transactions: int
    institutional_flows: float

@dataclass
class FundamentalAnalysis:
    """Temel analiz sonucu"""
    timestamp: datetime
    news_sentiment: float
    macro_sentiment: float
    supply_demand_score: float
    overall_sentiment: float
    confidence: float
    key_factors: List[str]
    recommendations: List[str]
    risk_level: str

class NewsSource(Enum):
    """Haber kaynakları"""
    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    TWITTER = "twitter"
    REDDIT = "reddit"
    YOUTUBE = "youtube"

class FundamentalAnalyzer:
    """Temel analiz sınıfı"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Haber kaynakları
        self.news_sources = {
            NewsSource.COINDESK: {
                'rss_url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'api_url': 'https://api.coindesk.com/v1/news',
                'weight': 0.3
            },
            NewsSource.COINTELEGRAPH: {
                'rss_url': 'https://cointelegraph.com/rss',
                'api_url': 'https://api.cointelegraph.com/v1/news',
                'weight': 0.25
            },
            NewsSource.BLOOMBERG: {
                'rss_url': 'https://feeds.bloomberg.com/markets/news.rss',
                'api_url': 'https://api.bloomberg.com/v1/news',
                'weight': 0.2
            },
            NewsSource.REUTERS: {
                'rss_url': 'https://feeds.reuters.com/reuters/businessNews',
                'api_url': 'https://api.reuters.com/v1/news',
                'weight': 0.15
            }
        }
        
        # Ek RSS feed kaynakları (ücretsiz)
        self.rss_feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://cryptonews.com/news/feed",
            "https://www.investing.com/rss/news_301.rss",
            "https://cryptopotato.com/feed/",
            "https://bitcoinmagazine.com/rss",
            "https://www.coinbureau.com/feed/",
            "https://decrypt.co/feed",
            "https://coingape.com/feed/",
            "https://u.today/rss"
        ]
        
        # Makroekonomik veri kaynakları
        self.macro_sources = {
            'federal_reserve': {
                'api_url': 'https://api.stlouisfed.org/fred/series/observations',
                'series_ids': ['FEDFUNDS', 'UNRATE', 'CPIAUCSL', 'GDP'],
                'weight': 0.4
            },
            'treasury': {
                'api_url': 'https://api.fiscaldata.treasury.gov/services/api/v1/accounting/od/debt_to_penny',
                'weight': 0.3
            },
            'crypto_metrics': {
                'api_url': 'https://api.coinmetrics.io/v4/timeseries/asset-metrics',
                'weight': 0.3
            }
        }
        
        # BTC arz-talep veri kaynakları
        self.btc_sources = {
            'glassnode': {
                'api_url': 'https://api.glassnode.com/v1/metrics',
                'api_key': self.config.get('glassnode_api_key', ''),
                'weight': 0.4
            },
            'coinmetrics': {
                'api_url': 'https://api.coinmetrics.io/v4/timeseries/asset-metrics',
                'weight': 0.3
            },
            'blockchain_info': {
                'api_url': 'https://blockchain.info/stats',
                'weight': 0.3
            }
        }
        
        # Veri saklama
        self.news_history = deque(maxlen=1000)
        self.macro_history = deque(maxlen=500)
        self.btc_history = deque(maxlen=200)
        self.analysis_history = deque(maxlen=100)
        
        # Sentiment analizi için anahtar kelimeler
        self.sentiment_keywords = {
            'positive': [
                'adoption', 'institutional', 'etf', 'approval', 'bullish', 'rally',
                'breakthrough', 'partnership', 'investment', 'growth', 'surge',
                'milestone', 'record', 'high', 'gain', 'profit', 'success'
            ],
            'negative': [
                'crash', 'bearish', 'decline', 'loss', 'regulation', 'ban',
                'hack', 'security', 'risk', 'volatility', 'uncertainty',
                'correction', 'dump', 'sell-off', 'fear', 'concern'
            ],
            'neutral': [
                'analysis', 'report', 'update', 'news', 'market', 'trading',
                'price', 'volume', 'technical', 'fundamental', 'data'
            ]
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Analiz thread'i
        self.analysis_thread = None
        self.is_analyzing = False
        
        # Callback'ler
        self.news_callbacks = []
        self.macro_callbacks = []
        self.analysis_callbacks = []
        
        self.logger.info("Temel analiz modülü başlatıldı")
    
    def start_analysis(self):
        """Analizi başlat"""
        if self.is_analyzing:
            return
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("Temel analiz başlatıldı")
    
    def stop_analysis(self):
        """Analizi durdur"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.logger.info("Temel analiz durduruldu")
    
    def _analysis_loop(self):
        """Ana analiz döngüsü"""
        while self.is_analyzing:
            try:
                # Haberleri topla
                self._collect_news()
                
                # Makroekonomik verileri topla
                self._collect_macro_data()
                
                # BTC arz-talep verilerini topla
                self._collect_btc_data()
                
                # Temel analizi yap
                analysis = self._perform_fundamental_analysis()
                
                if analysis:
                    self.analysis_history.append(analysis)
                    self._notify_analysis_callbacks(analysis)
                
                # 30 dakika bekle
                time.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"Analiz döngüsü hatası: {e}")
                time.sleep(1800)
    
    def _collect_news(self):
        """Haberleri topla"""
        try:
            # Önce ücretsiz RSS feed'lerden haberleri topla
            for rss_url in self.rss_feeds:
                try:
                    news_items = self._fetch_rss_news_free(rss_url)
                    
                    for item in news_items:
                        # Sentiment analizi yap
                        sentiment_score = self._analyze_sentiment(item['title'] + ' ' + item['content'])
                        
                        # Relevance skoru hesapla
                        relevance_score = self._calculate_relevance(item['title'] + ' ' + item['content'])
                        
                        # Anahtar kelimeleri çıkar
                        keywords = self._extract_keywords(item['title'] + ' ' + item['content'])
                        
                        # Impact seviyesini belirle
                        impact_level = self._determine_impact_level(sentiment_score, relevance_score)
                        
                        news_item = NewsItem(
                            title=item['title'],
                            content=item['content'],
                            source=item['source'],
                            url=item['url'],
                            timestamp=item['timestamp'],
                            sentiment_score=sentiment_score,
                            relevance_score=relevance_score,
                            keywords=keywords,
                            impact_level=impact_level
                        )
                        
                        # Veritabanına kaydet
                        self._save_news_item(news_item)
                        
                        # Cache'e ekle
                        self.news_data.append(news_item)
                        
                except Exception as e:
                    self.logger.error(f"Ücretsiz RSS haber toplama hatası ({rss_url}): {e}")
            
            # API kaynaklarından haberleri topla
            for source, config in self.news_sources.items():
                try:
                    # RSS feed'den haberleri al
                    news_items = self._fetch_rss_news(config['rss_url'], source.value)
                    
                    for item in news_items:
                        # Sentiment analizi yap
                        sentiment_score = self._analyze_sentiment(item['title'] + ' ' + item['content'])
                        
                        # Relevance skoru hesapla
                        relevance_score = self._calculate_relevance(item['title'] + ' ' + item['content'])
                        
                        # Anahtar kelimeleri çıkar
                        keywords = self._extract_keywords(item['title'] + ' ' + item['content'])
                        
                        # Impact seviyesini belirle
                        impact_level = self._determine_impact_level(sentiment_score, relevance_score)
                        
                        news_item = NewsItem(
                            title=item['title'],
                            content=item['content'],
                            source=source.value,
                            url=item['url'],
                            timestamp=item['timestamp'],
                            sentiment_score=sentiment_score,
                            relevance_score=relevance_score,
                            keywords=keywords,
                            impact_level=impact_level
                        )
                        
                        with self.lock:
                            self.news_history.append(news_item)
                        
                        # Callback'leri çağır
                        self._notify_news_callbacks(news_item)
                        
                except Exception as e:
                    self.logger.error(f"Haber toplama hatası ({source.value}): {e}")
                    
        except Exception as e:
            self.logger.error(f"Genel haber toplama hatası: {e}")
    
    def _fetch_rss_news_free(self, rss_url: str) -> List[Dict[str, Any]]:
        """Ücretsiz RSS feed'den haberleri al"""
        try:
            feed = feedparser.parse(rss_url)
            news_items = []
            
            # Feed başlığını al
            feed_title = feed.feed.get('title', 'Unknown Source')
            
            for entry in feed.entries[:10]:  # Son 10 haber
                # İçeriği temizle
                content = self._clean_content(entry.get('summary', ''))
                
                # Timestamp'i parse et
                timestamp = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    timestamp = datetime(*entry.published_parsed[:6])
                
                news_item = {
                    'title': entry.get('title', ''),
                    'content': content,
                    'url': entry.get('link', ''),
                    'timestamp': timestamp,
                    'source': feed_title
                }
                
                news_items.append(news_item)
            
            self.logger.info(f"RSS feed'den {len(news_items)} haber alındı: {feed_title}")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Ücretsiz RSS haber alma hatası ({rss_url}): {e}")
            return []
    
    def _fetch_rss_news(self, rss_url: str, source: str) -> List[Dict[str, Any]]:
        """RSS feed'den haberleri al"""
        try:
            feed = feedparser.parse(rss_url)
            news_items = []
            
            for entry in feed.entries[:20]:  # Son 20 haber
                # İçeriği temizle
                content = self._clean_content(entry.get('summary', ''))
                
                news_item = {
                    'title': entry.get('title', ''),
                    'content': content,
                    'url': entry.get('link', ''),
                    'timestamp': datetime.now()  # RSS'de timestamp yoksa şimdiki zaman
                }
                
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"RSS haber alma hatası ({source}): {e}")
            return []
    
    def _clean_content(self, content: str) -> str:
        """İçeriği temizle"""
        try:
            # HTML etiketlerini kaldır
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            
            # Fazla boşlukları temizle
            text = re.sub(r'\s+', ' ', text)
            
            # Özel karakterleri temizle
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"İçerik temizleme hatası: {e}")
            return content
    
    def _analyze_sentiment(self, text: str) -> float:
        """Sentiment analizi yap"""
        try:
            text_lower = text.lower()
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            # Anahtar kelime sayımı
            for word in self.sentiment_keywords['positive']:
                positive_count += text_lower.count(word)
            
            for word in self.sentiment_keywords['negative']:
                negative_count += text_lower.count(word)
            
            for word in self.sentiment_keywords['neutral']:
                neutral_count += text_lower.count(word)
            
            # Sentiment skoru hesapla (-1 ile 1 arası)
            total_words = positive_count + negative_count + neutral_count
            
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            
            # Skoru normalize et
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            self.logger.error(f"Sentiment analiz hatası: {e}")
            return 0.0
    
    def _calculate_relevance(self, text: str) -> float:
        """Relevance skoru hesapla"""
        try:
            text_lower = text.lower()
            
            # Kripto ile ilgili anahtar kelimeler
            crypto_keywords = [
                'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain',
                'ethereum', 'eth', 'altcoin', 'defi', 'nft', 'mining',
                'wallet', 'exchange', 'trading', 'investment', 'hodl'
            ]
            
            relevance_count = 0
            for keyword in crypto_keywords:
                if keyword in text_lower:
                    relevance_count += 1
            
            # Relevance skoru (0 ile 1 arası)
            relevance_score = min(1.0, relevance_count / len(crypto_keywords))
            
            return relevance_score
            
        except Exception as e:
            self.logger.error(f"Relevance hesaplama hatası: {e}")
            return 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Anahtar kelimeleri çıkar"""
        try:
            text_lower = text.lower()
            keywords = []
            
            # Tüm sentiment anahtar kelimelerini kontrol et
            all_keywords = (
                self.sentiment_keywords['positive'] +
                self.sentiment_keywords['negative'] +
                self.sentiment_keywords['neutral']
            )
            
            for keyword in all_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            return list(set(keywords))  # Tekrarları kaldır
            
        except Exception as e:
            self.logger.error(f"Anahtar kelime çıkarma hatası: {e}")
            return []
    
    def _determine_impact_level(self, sentiment_score: float, relevance_score: float) -> str:
        """Impact seviyesini belirle"""
        try:
            # Kombine skor
            combined_score = abs(sentiment_score) * relevance_score
            
            if combined_score > 0.7:
                return 'high'
            elif combined_score > 0.4:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Impact seviyesi belirleme hatası: {e}")
            return 'low'
    
    def _collect_macro_data(self):
        """Makroekonomik verileri topla"""
        try:
            for source_name, config in self.macro_sources.items():
                try:
                    if source_name == 'federal_reserve':
                        macro_data = self._fetch_fed_data(config)
                    elif source_name == 'treasury':
                        macro_data = self._fetch_treasury_data(config)
                    elif source_name == 'crypto_metrics':
                        macro_data = self._fetch_crypto_metrics(config)
                    else:
                        continue
                    
                    for data in macro_data:
                        with self.lock:
                            self.macro_history.append(data)
                        
                        # Callback'leri çağır
                        self._notify_macro_callbacks(data)
                        
                except Exception as e:
                    self.logger.error(f"Makro veri toplama hatası ({source_name}): {e}")
                    
        except Exception as e:
            self.logger.error(f"Genel makro veri toplama hatası: {e}")
    
    def _fetch_fed_data(self, config: Dict[str, Any]) -> List[MacroEconomicData]:
        """Fed verilerini al"""
        try:
            macro_data = []
            
            # Simüle edilmiş Fed verileri (gerçek API entegrasyonu için FRED API key gerekli)
            fed_indicators = {
                'FEDFUNDS': {'name': 'Federal Funds Rate', 'unit': '%', 'current': 5.25, 'previous': 5.0},
                'UNRATE': {'name': 'Unemployment Rate', 'unit': '%', 'current': 3.8, 'previous': 3.9},
                'CPIAUCSL': {'name': 'Consumer Price Index', 'unit': 'Index', 'current': 307.5, 'previous': 305.0},
                'GDP': {'name': 'Gross Domestic Product', 'unit': 'Billion USD', 'current': 27500, 'previous': 27000}
            }
            
            for series_id, data in fed_indicators.items():
                # Impact belirleme
                change = data['current'] - data['previous']
                if abs(change) > data['current'] * 0.02:  # %2'den büyük değişim
                    impact = 'positive' if change > 0 else 'negative'
                else:
                    impact = 'neutral'
                
                macro_item = MacroEconomicData(
                    indicator_name=data['name'],
                    value=data['current'],
                    previous_value=data['previous'],
                    forecast=data['current'] * 1.02,  # Simüle edilmiş tahmin
                    unit=data['unit'],
                    timestamp=datetime.now(),
                    impact=impact,
                    source='federal_reserve'
                )
                
                macro_data.append(macro_item)
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Fed veri alma hatası: {e}")
            return []
    
    def _fetch_treasury_data(self, config: Dict[str, Any]) -> List[MacroEconomicData]:
        """Treasury verilerini al"""
        try:
            macro_data = []
            
            # Simüle edilmiş Treasury verileri
            treasury_indicators = {
                'debt_to_gdp': {'name': 'Debt to GDP Ratio', 'unit': '%', 'current': 120.5, 'previous': 118.0},
                'yield_10y': {'name': '10-Year Treasury Yield', 'unit': '%', 'current': 4.2, 'previous': 4.0},
                'yield_2y': {'name': '2-Year Treasury Yield', 'unit': '%', 'current': 4.8, 'previous': 4.6}
            }
            
            for indicator, data in treasury_indicators.items():
                change = data['current'] - data['previous']
                impact = 'positive' if change < 0 else 'negative' if change > 0 else 'neutral'
                
                macro_item = MacroEconomicData(
                    indicator_name=data['name'],
                    value=data['current'],
                    previous_value=data['previous'],
                    forecast=data['current'] * 1.01,
                    unit=data['unit'],
                    timestamp=datetime.now(),
                    impact=impact,
                    source='treasury'
                )
                
                macro_data.append(macro_item)
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Treasury veri alma hatası: {e}")
            return []
    
    def _fetch_crypto_metrics(self, config: Dict[str, Any]) -> List[MacroEconomicData]:
        """Kripto metriklerini al"""
        try:
            macro_data = []
            
            # Simüle edilmiş kripto metrikleri
            crypto_indicators = {
                'total_market_cap': {'name': 'Total Crypto Market Cap', 'unit': 'Trillion USD', 'current': 2.1, 'previous': 2.0},
                'defi_tvl': {'name': 'DeFi Total Value Locked', 'unit': 'Billion USD', 'current': 85, 'previous': 80},
                'stablecoin_supply': {'name': 'Stablecoin Supply', 'unit': 'Billion USD', 'current': 120, 'previous': 115}
            }
            
            for indicator, data in crypto_indicators.items():
                change = data['current'] - data['previous']
                impact = 'positive' if change > 0 else 'negative' if change < 0 else 'neutral'
                
                macro_item = MacroEconomicData(
                    indicator_name=data['name'],
                    value=data['current'],
                    previous_value=data['previous'],
                    forecast=data['current'] * 1.05,
                    unit=data['unit'],
                    timestamp=datetime.now(),
                    impact=impact,
                    source='crypto_metrics'
                )
                
                macro_data.append(macro_item)
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Kripto metrik alma hatası: {e}")
            return []
    
    def _collect_btc_data(self):
        """BTC arz-talep verilerini topla"""
        try:
            for source_name, config in self.btc_sources.items():
                try:
                    if source_name == 'glassnode':
                        btc_data = self._fetch_glassnode_data(config)
                    elif source_name == 'coinmetrics':
                        btc_data = self._fetch_coinmetrics_data(config)
                    elif source_name == 'blockchain_info':
                        btc_data = self._fetch_blockchain_info_data(config)
                    else:
                        continue
                    
                    if btc_data:
                        with self.lock:
                            self.btc_history.append(btc_data)
                        
                except Exception as e:
                    self.logger.error(f"BTC veri toplama hatası ({source_name}): {e}")
                    
        except Exception as e:
            self.logger.error(f"Genel BTC veri toplama hatası: {e}")
    
    def _fetch_glassnode_data(self, config: Dict[str, Any]) -> Optional[BitcoinSupplyDemand]:
        """Glassnode verilerini al"""
        try:
            # Simüle edilmiş Glassnode verileri (gerçek API entegrasyonu için API key gerekli)
            btc_data = BitcoinSupplyDemand(
                timestamp=datetime.now(),
                circulating_supply=19500000,  # ~19.5M BTC
                total_supply=19500000,
                max_supply=21000000,
                active_addresses=850000,  # Günlük aktif adresler
                transaction_count=280000,  # Günlük işlem sayısı
                hash_rate=450000000000000000000,  # TH/s
                difficulty=61000000000000,  # Mining difficulty
                mining_revenue=45000000,  # Günlük mining geliri (USD)
                exchange_inflows=12000,  # Günlük exchange girişi (BTC)
                exchange_outflows=11000,  # Günlük exchange çıkışı (BTC)
                whale_transactions=45,  # Günlük whale işlemleri
                institutional_flows=25000000  # Aylık kurumsal akış (USD)
            )
            
            return btc_data
            
        except Exception as e:
            self.logger.error(f"Glassnode veri alma hatası: {e}")
            return None
    
    def _fetch_coinmetrics_data(self, config: Dict[str, Any]) -> Optional[BitcoinSupplyDemand]:
        """CoinMetrics verilerini al"""
        try:
            # Simüle edilmiş CoinMetrics verileri
            btc_data = BitcoinSupplyDemand(
                timestamp=datetime.now(),
                circulating_supply=19500000,
                total_supply=19500000,
                max_supply=21000000,
                active_addresses=820000,
                transaction_count=275000,
                hash_rate=440000000000000000000,
                difficulty=60000000000000,
                mining_revenue=44000000,
                exchange_inflows=11500,
                exchange_outflows=10800,
                whale_transactions=42,
                institutional_flows=24000000
            )
            
            return btc_data
            
        except Exception as e:
            self.logger.error(f"CoinMetrics veri alma hatası: {e}")
            return None
    
    def _fetch_blockchain_info_data(self, config: Dict[str, Any]) -> Optional[BitcoinSupplyDemand]:
        """Blockchain.info verilerini al"""
        try:
            # Simüle edilmiş Blockchain.info verileri
            btc_data = BitcoinSupplyDemand(
                timestamp=datetime.now(),
                circulating_supply=19500000,
                total_supply=19500000,
                max_supply=21000000,
                active_addresses=800000,
                transaction_count=270000,
                hash_rate=430000000000000000000,
                difficulty=59000000000000,
                mining_revenue=43000000,
                exchange_inflows=11000,
                exchange_outflows=10500,
                whale_transactions=40,
                institutional_flows=23000000
            )
            
            return btc_data
            
        except Exception as e:
            self.logger.error(f"Blockchain.info veri alma hatası: {e}")
            return None
    
    def _perform_fundamental_analysis(self) -> Optional[FundamentalAnalysis]:
        """Temel analizi yap"""
        try:
            # Son haberleri analiz et
            news_sentiment = self._analyze_news_sentiment()
            
            # Makroekonomik sentiment analizi
            macro_sentiment = self._analyze_macro_sentiment()
            
            # BTC arz-talep analizi
            supply_demand_score = self._analyze_supply_demand()
            
            # Genel sentiment hesapla
            overall_sentiment = (
                news_sentiment * 0.4 +
                macro_sentiment * 0.3 +
                supply_demand_score * 0.3
            )
            
            # Güven skoru hesapla
            confidence = self._calculate_confidence()
            
            # Ana faktörleri belirle
            key_factors = self._identify_key_factors()
            
            # Önerileri oluştur
            recommendations = self._generate_recommendations(overall_sentiment, key_factors)
            
            # Risk seviyesini belirle
            risk_level = self._assess_risk_level(overall_sentiment, confidence)
            
            analysis = FundamentalAnalysis(
                timestamp=datetime.now(),
                news_sentiment=news_sentiment,
                macro_sentiment=macro_sentiment,
                supply_demand_score=supply_demand_score,
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                key_factors=key_factors,
                recommendations=recommendations,
                risk_level=risk_level
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Temel analiz hatası: {e}")
            return None
    
    def _analyze_news_sentiment(self) -> float:
        """Haber sentiment analizi"""
        try:
            if not self.news_history:
                return 0.0
            
            # Son 24 saatteki haberleri al
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_news = [news for news in self.news_history if news.timestamp > cutoff_time]
            
            if not recent_news:
                return 0.0
            
            # Ağırlıklı sentiment hesapla
            total_weighted_sentiment = 0.0
            total_weight = 0.0
            
            for news in recent_news:
                weight = news.relevance_score
                if news.impact_level == 'high':
                    weight *= 2.0
                elif news.impact_level == 'medium':
                    weight *= 1.5
                
                total_weighted_sentiment += news.sentiment_score * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            return total_weighted_sentiment / total_weight
            
        except Exception as e:
            self.logger.error(f"Haber sentiment analiz hatası: {e}")
            return 0.0
    
    def _analyze_macro_sentiment(self) -> float:
        """Makroekonomik sentiment analizi"""
        try:
            if not self.macro_history:
                return 0.0
            
            # Son makro verileri al
            recent_macro = list(self.macro_history)[-10:]  # Son 10 veri
            
            positive_count = 0
            negative_count = 0
            
            for data in recent_macro:
                if data.impact == 'positive':
                    positive_count += 1
                elif data.impact == 'negative':
                    negative_count += 1
            
            total_count = len(recent_macro)
            if total_count == 0:
                return 0.0
            
            # Sentiment skoru (-1 ile 1 arası)
            sentiment = (positive_count - negative_count) / total_count
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Makro sentiment analiz hatası: {e}")
            return 0.0
    
    def _analyze_supply_demand(self) -> float:
        """Arz-talep analizi"""
        try:
            if not self.btc_history:
                return 0.0
            
            latest_data = self.btc_history[-1]
            
            # Arz faktörleri
            supply_score = 0.0
            
            # Dolaşımdaki arz (azalma pozitif)
            max_supply_ratio = latest_data.circulating_supply / latest_data.max_supply
            supply_score += (1 - max_supply_ratio) * 0.3
            
            # Mining aktivitesi
            if latest_data.hash_rate > 400000000000000000000:  # Yüksek hash rate
                supply_score += 0.2
            
            # Talep faktörleri
            demand_score = 0.0
            
            # Exchange çıkışları (pozitif)
            net_exchange_flow = latest_data.exchange_outflows - latest_data.exchange_inflows
            if net_exchange_flow > 0:
                demand_score += 0.3
            
            # Aktif adresler
            if latest_data.active_addresses > 800000:
                demand_score += 0.2
            
            # Kurumsal akışlar
            if latest_data.institutional_flows > 20000000:
                demand_score += 0.3
            
            # Genel skor (-1 ile 1 arası)
            overall_score = demand_score - supply_score
            
            return max(-1.0, min(1.0, overall_score))
            
        except Exception as e:
            self.logger.error(f"Arz-talep analiz hatası: {e}")
            return 0.0
    
    def _calculate_confidence(self) -> float:
        """Güven skoru hesapla"""
        try:
            confidence_factors = []
            
            # Haber sayısı
            if len(self.news_history) > 50:
                confidence_factors.append(0.8)
            elif len(self.news_history) > 20:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Makro veri sayısı
            if len(self.macro_history) > 20:
                confidence_factors.append(0.7)
            elif len(self.macro_history) > 10:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.2)
            
            # BTC veri sayısı
            if len(self.btc_history) > 5:
                confidence_factors.append(0.6)
            elif len(self.btc_history) > 2:
                confidence_factors.append(0.4)
            else:
                confidence_factors.append(0.1)
            
            return np.mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Güven skoru hesaplama hatası: {e}")
            return 0.0
    
    def _identify_key_factors(self) -> List[str]:
        """Ana faktörleri belirle"""
        try:
            key_factors = []
            
            # Haber faktörleri
            if self.news_history:
                recent_news = list(self.news_history)[-10:]
                high_impact_news = [news for news in recent_news if news.impact_level == 'high']
                
                if high_impact_news:
                    key_factors.append(f"{len(high_impact_news)} yüksek etkili haber")
            
            # Makro faktörler
            if self.macro_history:
                recent_macro = list(self.macro_history)[-5:]
                positive_macro = [data for data in recent_macro if data.impact == 'positive']
                negative_macro = [data for data in recent_macro if data.impact == 'negative']
                
                if positive_macro:
                    key_factors.append(f"{len(positive_macro)} pozitif makro veri")
                if negative_macro:
                    key_factors.append(f"{len(negative_macro)} negatif makro veri")
            
            # BTC faktörleri
            if self.btc_history:
                latest_btc = self.btc_history[-1]
                
                if latest_btc.exchange_outflows > latest_btc.exchange_inflows:
                    key_factors.append("Exchange'den çıkış fazla")
                
                if latest_btc.institutional_flows > 20000000:
                    key_factors.append("Yüksek kurumsal akış")
                
                if latest_btc.active_addresses > 800000:
                    key_factors.append("Yüksek aktif adres sayısı")
            
            return key_factors
            
        except Exception as e:
            self.logger.error(f"Ana faktör belirleme hatası: {e}")
            return []
    
    def _generate_recommendations(self, sentiment: float, key_factors: List[str]) -> List[str]:
        """Öneriler oluştur"""
        try:
            recommendations = []
            
            if sentiment > 0.3:
                recommendations.append("Pozitif temel analiz - Uzun vadeli pozisyonlar değerlendirilebilir")
            elif sentiment < -0.3:
                recommendations.append("Negatif temel analiz - Risk yönetimi önemli")
            else:
                recommendations.append("Nötr temel analiz - Teknik analiz odaklı yaklaşım")
            
            # Faktörlere göre öneriler
            if "yüksek etkili haber" in str(key_factors):
                recommendations.append("Haber takibi artırılmalı")
            
            if "Exchange'den çıkış fazla" in str(key_factors):
                recommendations.append("Arz azalması pozitif sinyal")
            
            if "Yüksek kurumsal akış" in str(key_factors):
                recommendations.append("Kurumsal ilgi artışı")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Öneri oluşturma hatası: {e}")
            return []
    
    def _assess_risk_level(self, sentiment: float, confidence: float) -> str:
        """Risk seviyesini değerlendir"""
        try:
            # Sentiment ve güven seviyesine göre risk belirleme
            if confidence < 0.3:
                return 'high'
            elif abs(sentiment) > 0.5 and confidence > 0.7:
                return 'medium'
            elif abs(sentiment) < 0.2 and confidence > 0.6:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            self.logger.error(f"Risk seviyesi değerlendirme hatası: {e}")
            return 'medium'
    
    def get_latest_analysis(self) -> Optional[FundamentalAnalysis]:
        """Son analizi al"""
        try:
            with self.lock:
                if self.analysis_history:
                    return self.analysis_history[-1]
                return None
        except Exception as e:
            self.logger.error(f"Son analiz alma hatası: {e}")
            return None
    
    def get_news_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Haber özeti al"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_news = [news for news in self.news_history if news.timestamp > cutoff_time]
            
            if not recent_news:
                return {'count': 0, 'avg_sentiment': 0, 'high_impact_count': 0}
            
            avg_sentiment = np.mean([news.sentiment_score for news in recent_news])
            high_impact_count = len([news for news in recent_news if news.impact_level == 'high'])
            
            return {
                'count': len(recent_news),
                'avg_sentiment': avg_sentiment,
                'high_impact_count': high_impact_count,
                'sources': list(set([news.source for news in recent_news]))
            }
            
        except Exception as e:
            self.logger.error(f"Haber özeti alma hatası: {e}")
            return {'count': 0, 'avg_sentiment': 0, 'high_impact_count': 0}
    
    def get_macro_summary(self) -> Dict[str, Any]:
        """Makroekonomik özet al"""
        try:
            if not self.macro_history:
                return {'count': 0, 'positive_count': 0, 'negative_count': 0}
            
            recent_macro = list(self.macro_history)[-20:]  # Son 20 veri
            
            positive_count = len([data for data in recent_macro if data.impact == 'positive'])
            negative_count = len([data for data in recent_macro if data.impact == 'negative'])
            
            return {
                'count': len(recent_macro),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': len(recent_macro) - positive_count - negative_count
            }
            
        except Exception as e:
            self.logger.error(f"Makro özet alma hatası: {e}")
            return {'count': 0, 'positive_count': 0, 'negative_count': 0}
    
    def get_btc_summary(self) -> Optional[BitcoinSupplyDemand]:
        """BTC özet al"""
        try:
            with self.lock:
                if self.btc_history:
                    return self.btc_history[-1]
                return None
        except Exception as e:
            self.logger.error(f"BTC özet alma hatası: {e}")
            return None
    
    def add_news_callback(self, callback):
        """Haber callback'i ekle"""
        self.news_callbacks.append(callback)
    
    def add_macro_callback(self, callback):
        """Makro callback'i ekle"""
        self.macro_callbacks.append(callback)
    
    def add_analysis_callback(self, callback):
        """Analiz callback'i ekle"""
        self.analysis_callbacks.append(callback)
    
    def _notify_news_callbacks(self, news_item: NewsItem):
        """Haber callback'lerini çağır"""
        for callback in self.news_callbacks:
            try:
                callback(news_item)
            except Exception as e:
                self.logger.error(f"Haber callback hatası: {e}")
    
    def _notify_macro_callbacks(self, macro_data: MacroEconomicData):
        """Makro callback'lerini çağır"""
        for callback in self.macro_callbacks:
            try:
                callback(macro_data)
            except Exception as e:
                self.logger.error(f"Makro callback hatası: {e}")
    
    def _notify_analysis_callbacks(self, analysis: FundamentalAnalysis):
        """Analiz callback'lerini çağır"""
        for callback in self.analysis_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Analiz callback hatası: {e}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Konfigürasyonu güncelle"""
        try:
            self.config.update(new_config)
            
            if 'news_sources' in new_config:
                self.news_sources.update(new_config['news_sources'])
            
            if 'macro_sources' in new_config:
                self.macro_sources.update(new_config['macro_sources'])
            
            if 'btc_sources' in new_config:
                self.btc_sources.update(new_config['btc_sources'])
            
            self.logger.info("Temel analiz konfigürasyonu güncellendi")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon güncelleme hatası: {e}")

# Global temel analiz modülü
fundamental_analyzer = FundamentalAnalyzer()

# Ana sınıf alias'ı (zaten doğru)
