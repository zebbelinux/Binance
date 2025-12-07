"""
DeepSeek API Entegrasyonu
AI destekli piyasa analizi ve sinyal üretimi
"""

import requests
import json
import time
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DeepSeekAPI:
    """DeepSeek API sınıfı"""
    
    def __init__(self, api_key: Optional[str] = None):
        # API anahtarını ortam değişkeninden oku (sağlanmadıysa)
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })
        
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 saniye
        self.daily_request_count = 0
        self.daily_limit = 1000  # Günlük limit
    
    def _make_request(self, endpoint: str, data: dict) -> dict:
        """DeepSeek API isteği gönder"""
        # API anahtarı yoksa çağrıyı yapma
        if not self.api_key:
            self.logger.warning("DeepSeek API anahtarı bulunamadı. Lütfen DEEPSEEK_API_KEY ortam değişkenini ayarlayın.")
            return {"error": "Missing API key"}
        # Rate limiting kontrolü
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        self.daily_request_count += 1
        
        if self.daily_request_count > self.daily_limit:
            self.logger.warning("Günlük API limiti aşıldı")
            return {"error": "Daily limit exceeded"}
        
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek API isteği başarısız: {e}")
            return {"error": str(e)}
    
    def analyze_market_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Piyasa duyarlılığını analiz et"""
        prompt = f"""
        Aşağıdaki kripto para piyasa verilerini analiz ederek duyarlılık analizi yap:
        
        Piyasa Verileri:
        - Fiyat: {market_data.get('price', 'N/A')}
        - 24s Değişim: {market_data.get('change_24h', 'N/A')}%
        - Hacim: {market_data.get('volume_24h', 'N/A')}
        - RSI: {market_data.get('rsi', 'N/A')}
        - MACD: {market_data.get('macd', 'N/A')}
        - Bollinger Bands: {market_data.get('bb_position', 'N/A')}
        
        Lütfen şu formatta yanıt ver:
        {{
            "sentiment": "bullish/neutral/bearish",
            "confidence": 0.0-1.0,
            "reasoning": "Analiz gerekçesi",
            "recommendation": "buy/hold/sell",
            "risk_level": "low/medium/high"
        }}
        """
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen bir uzman kripto para analisti ve trading uzmanısın. Piyasa verilerini analiz ederek objektif ve güvenilir yatırım tavsiyeleri veriyorsun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = self._make_request("/chat/completions", data)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek yanıtı parse edilemedi: {e}")
            return {"error": "Response parsing failed"}
    
    def generate_trading_signals(self, technical_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Trading sinyalleri üret"""
        prompt = f"""
        Aşağıdaki teknik analiz göstergelerini kullanarak trading sinyali üret:
        
        Teknik Göstergeler:
        - RSI: {technical_indicators.get('rsi', 'N/A')}
        - MACD: {technical_indicators.get('macd', 'N/A')}
        - MACD Signal: {technical_indicators.get('macd_signal', 'N/A')}
        - MACD Histogram: {technical_indicators.get('macd_histogram', 'N/A')}
        - Moving Average 20: {technical_indicators.get('ma_20', 'N/A')}
        - Moving Average 50: {technical_indicators.get('ma_50', 'N/A')}
        - Bollinger Upper: {technical_indicators.get('bb_upper', 'N/A')}
        - Bollinger Lower: {technical_indicators.get('bb_lower', 'N/A')}
        - ATR: {technical_indicators.get('atr', 'N/A')}
        - ADX: {technical_indicators.get('adx', 'N/A')}
        - Volume: {technical_indicators.get('volume', 'N/A')}
        
        Lütfen şu formatta yanıt ver:
        {{
            "signal": "BUY/SELL/HOLD",
            "strength": 0.0-1.0,
            "entry_price": "Önerilen giriş fiyatı",
            "stop_loss": "Stop loss fiyatı",
            "take_profit": "Take profit fiyatı",
            "reasoning": "Sinyal gerekçesi",
            "risk_reward_ratio": "Risk/Ödül oranı"
        }}
        """
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen bir profesyonel trading algoritmasısın. Teknik analiz göstergelerini kullanarak güvenilir trading sinyalleri üretiyorsun. Risk yönetimini her zaman öncelikli tutuyorsun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }
        
        response = self._make_request("/chat/completions", data)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek yanıtı parse edilemedi: {e}")
            return {"error": "Response parsing failed"}
    
    def analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Haber duyarlılığını analiz et"""
        news_text = "\n".join([f"- {news.get('title', '')}: {news.get('content', '')}" for news in news_data])
        
        prompt = f"""
        Aşağıdaki kripto para haberlerini analiz ederek piyasa üzerindeki etkisini değerlendir:
        
        Haberler:
        {news_text}
        
        Lütfen şu formatta yanıt ver:
        {{
            "overall_sentiment": "positive/neutral/negative",
            "market_impact": "high/medium/low",
            "key_themes": ["Tema1", "Tema2", "Tema3"],
            "price_direction": "upward/neutral/downward",
            "confidence": 0.0-1.0,
            "summary": "Özet analiz"
        }}
        """
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen bir finansal haber analisti ve piyasa uzmanısın. Haberleri analiz ederek piyasa üzerindeki potansiyel etkileri değerlendiriyorsun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.4,
            "max_tokens": 600
        }
        
        response = self._make_request("/chat/completions", data)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek yanıtı parse edilemedi: {e}")
            return {"error": "Response parsing failed"}
    
    def optimize_strategy_parameters(self, strategy_name: str, historical_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Strateji parametrelerini optimize et"""
        prompt = f"""
        Aşağıdaki strateji performans verilerini analiz ederek parametre optimizasyonu öner:
        
        Strateji: {strategy_name}
        Performans Verileri:
        - Sharpe Ratio: {historical_performance.get('sharpe_ratio', 'N/A')}
        - Max Drawdown: {historical_performance.get('max_drawdown', 'N/A')}%
        - Win Rate: {historical_performance.get('win_rate', 'N/A')}%
        - Profit Factor: {historical_performance.get('profit_factor', 'N/A')}
        - Total Return: {historical_performance.get('total_return', 'N/A')}%
        
        Mevcut Parametreler:
        {json.dumps(historical_performance.get('current_parameters', {}), indent=2)}
        
        Lütfen şu formatta yanıt ver:
        {{
            "optimized_parameters": {{
                "param1": "yeni_değer",
                "param2": "yeni_değer"
            }},
            "expected_improvement": "Beklenen iyileştirme yüzdesi",
            "risk_assessment": "Risk değerlendirmesi",
            "recommendation": "Optimizasyon önerisi"
        }}
        """
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen bir algoritmik trading optimizasyon uzmanısın. Strateji performanslarını analiz ederek parametre optimizasyonu yapıyorsun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = self._make_request("/chat/completions", data)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek yanıtı parse edilemedi: {e}")
            return {"error": "Response parsing failed"}
    
    def predict_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Piyasa rejimini tahmin et"""
        prompt = f"""
        Aşağıdaki piyasa verilerini analiz ederek mevcut piyasa rejimini belirle:
        
        Piyasa Verileri:
        - Volatilite: {market_data.get('volatility', 'N/A')}
        - Trend Gücü: {market_data.get('trend_strength', 'N/A')}
        - Hacim: {market_data.get('volume', 'N/A')}
        - Fiyat Hareketi: {market_data.get('price_movement', 'N/A')}
        - RSI: {market_data.get('rsi', 'N/A')}
        - ADX: {market_data.get('adx', 'N/A')}
        
        Lütfen şu formatta yanıt ver:
        {{
            "regime": "trending/sideways/volatile/crash",
            "confidence": 0.0-1.0,
            "duration_estimate": "Tahmini süre",
            "recommended_strategy": "Önerilen strateji",
            "risk_level": "low/medium/high",
            "reasoning": "Analiz gerekçesi"
        }}
        """
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen bir piyasa rejimi analisti ve makroekonomik uzmanısın. Piyasa koşullarını analiz ederek en uygun trading stratejisini öneriyorsun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }
        
        response = self._make_request("/chat/completions", data)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek yanıtı parse edilemedi: {e}")
            return {"error": "Response parsing failed"}
    
    def get_daily_usage_stats(self) -> Dict[str, Any]:
        """Günlük kullanım istatistiklerini al"""
        return {
            "daily_requests": self.daily_request_count,
            "daily_limit": self.daily_limit,
            "remaining_requests": self.daily_limit - self.daily_request_count,
            "usage_percentage": (self.daily_request_count / self.daily_limit) * 100
        }
    
    def reset_daily_counter(self):
        """Günlük sayaçları sıfırla"""
        self.daily_request_count = 0

# Global DeepSeek API instance
deepseek_api = DeepSeekAPI()
