"""
AI Destekli Piyasa Analiz Modülü
DeepSeek API ile entegre analiz sistemi
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import time

from ai.deepseek_api import deepseek_api
from api.multi_api_manager import multi_api_manager

class MarketAnalyzer:
    """AI destekli piyasa analiz sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analiz sonuçları cache
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 dakika
        
        # Teknik analiz göstergeleri
        self.technical_indicators = {}
        
        # Piyasa verileri
        self.market_data = {}
        
        # Analiz thread'i
        self.analysis_thread = None
        self.is_running = False
        
        # Callback'ler
        self.analysis_callbacks = []
        self.signal_callbacks = []
    
    def start_analysis(self, symbols: List[str] = None):
        """Analiz başlat"""
        if self.is_running:
            return
        
        self.is_running = True
        # Semboller verilmediyse, Binance üzerinden USDT + min-fiyat evrenini dinamik olarak oluştur
        if symbols:
            self.symbols = symbols
        else:
            try:
                tickers = multi_api_manager.market_api.get_all_tickers_normalized()
                dyn_syms = [str(t.get('pairSymbol')).upper() for t in tickers if t.get('pairSymbol')]
                # Güvenlik: boş kalırsa fallback olarak eski sabit listeyi kullan
                self.symbols = dyn_syms or ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
            except Exception:
                self.symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
        
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("Piyasa analizi başlatıldı")
    
    def stop_analysis(self):
        """Analizi durdur"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.logger.info("Piyasa analizi durduruldu")
    
    def _analysis_loop(self):
        """Ana analiz döngüsü"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # Piyasa verilerini al
                    market_data = self._get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Teknik analiz yap
                    technical_analysis = self._perform_technical_analysis(symbol, market_data)
                    
                    # AI analizi yap
                    ai_analysis = self._perform_ai_analysis(symbol, market_data, technical_analysis)
                    
                    # Sonuçları birleştir
                    combined_analysis = self._combine_analyses(symbol, market_data, technical_analysis, ai_analysis)
                    
                    # Cache'e kaydet
                    self.analysis_cache[symbol] = {
                        'data': combined_analysis,
                        'timestamp': time.time()
                    }
                    
                    # Callback'leri çağır
                    self._notify_callbacks(combined_analysis)
                
                # 30 saniye bekle
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Analiz döngüsü hatası: {e}")
                time.sleep(60)  # Hata durumunda 1 dakika bekle
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Piyasa verilerini al"""
        try:
            # Ticker verisi
            ticker_response = multi_api_manager.get_ticker(symbol)
            # HATA FIX: ticker_response bazen list gelebilir
            ticker_data_raw: Dict[str, Any]
            if isinstance(ticker_response, list):
                if not ticker_response:
                    ticker_data_raw = {}
                else:
                    first = ticker_response[0]
                    if isinstance(first, dict) and "data" in first:
                        ticker_data_raw = first.get("data", {})
                    elif isinstance(first, dict):
                        ticker_data_raw = first
                    else:
                        ticker_data_raw = {}
            elif isinstance(ticker_response, dict):
                if ticker_response.get("error"):
                    return None
                ticker_data_raw = ticker_response.get("data", ticker_response)
            else:
                ticker_data_raw = {}

            ticker_data = ticker_data_raw
            
            # Orderbook verisi
            orderbook_response = multi_api_manager.get_orderbook(symbol)
            orderbook_data_raw: Dict[str, Any]
            if isinstance(orderbook_response, list):
                if not orderbook_response:
                    orderbook_data_raw = {}
                else:
                    first_ob = orderbook_response[0]
                    if isinstance(first_ob, dict) and "data" in first_ob:
                        orderbook_data_raw = first_ob.get("data", {})
                    elif isinstance(first_ob, dict):
                        orderbook_data_raw = first_ob
                    else:
                        orderbook_data_raw = {}
            elif isinstance(orderbook_response, dict):
                if orderbook_response.get("error"):
                    orderbook_data_raw = {}
                else:
                    orderbook_data_raw = orderbook_response.get("data", orderbook_response)
            else:
                orderbook_data_raw = {}

            orderbook_data = orderbook_data_raw
            
            # Kline verisi (1 saatlik) - Binance REST
            klines_data = []
            try:
                import requests
                url = "https://api.binance.com/api/v3/klines"
                r = requests.get(url, params={"symbol": symbol, "interval": "1h", "limit": 100}, timeout=5)
                if r.ok:
                    arr = r.json()
                    for it in arr:
                        # [openTime, open, high, low, close, volume, closeTime, ...]
                        ts = int(it[0]) // 1000
                        o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4]); v = float(it[5])
                        klines_data.append([ts, o, h, l, c, v])
            except Exception as ex:
                self.logger.error(f"Binance klines hatası ({symbol}): {ex}")
            
            return {
                'symbol': symbol,
                'price': float(ticker_data.get('last', ticker_data.get('price', 0.0))),
                'change_24h': float(ticker_data.get('change', ticker_data.get('dailyChangePercent', 0.0))),
                'volume_24h': float(ticker_data.get('volume', ticker_data.get('volume_24h', 0.0))),
                'high_24h': float(ticker_data.get('high', ticker_data.get('highPrice', 0.0))),
                'low_24h': float(ticker_data.get('low', ticker_data.get('lowPrice', 0.0))),
                'bid': float(orderbook_data.get('bids', [{}])[0].get('price', 0)) if orderbook_data.get('bids') else 0,
                'ask': float(orderbook_data.get('asks', [{}])[0].get('price', 0)) if orderbook_data.get('asks') else 0,
                'klines': klines_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Piyasa verisi alma hatası ({symbol}): {e}")
            return None
    
    def _perform_technical_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Teknik analiz yap"""
        try:
            if not market_data.get('klines'):
                return {}
            
            # Kline verilerini DataFrame'e çevir
            df = pd.DataFrame(market_data['klines'])
            if df.empty:
                return {}
            
            # Sütun isimlerini düzenle
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df.astype({
                'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
            })
            
            # Teknik göstergeleri hesapla
            indicators = {}
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(df['close'])
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # Moving Averages
            indicators['ma_20'] = df['close'].rolling(20).mean().iloc[-1]
            indicators['ma_50'] = df['close'].rolling(50).mean().iloc[-1]
            indicators['ma_200'] = df['close'].rolling(200).mean().iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_position'] = self._get_bb_position(market_data['price'], bb_upper, bb_lower)
            
            # ATR
            indicators['atr'] = self._calculate_atr(df)
            
            # ADX
            indicators['adx'] = self._calculate_adx(df)
            
            # Stochastic RSI
            indicators['stoch_rsi'] = self._calculate_stoch_rsi(df['close'])
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = market_data['volume_24h'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # Trend analysis
            indicators['trend_strength'] = self._calculate_trend_strength(df['close'])
            indicators['volatility'] = df['close'].pct_change().std() * 100
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Teknik analiz hatası ({symbol}): {e}")
            return {}
    
    def _perform_ai_analysis(self, symbol: str, market_data: Dict[str, Any], 
                           technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """AI analizi yap"""
        try:
            # Piyasa duyarlılığı analizi
            sentiment_analysis = deepseek_api.analyze_market_sentiment(market_data)
            
            # Trading sinyali üret
            signal_analysis = deepseek_api.generate_trading_signals(technical_analysis)
            
            # Piyasa rejimi tahmini
            regime_analysis = deepseek_api.predict_market_regime(market_data)
            
            return {
                'sentiment': sentiment_analysis,
                'signals': signal_analysis,
                'regime': regime_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"AI analiz hatası ({symbol}): {e}")
            return {}
    
    def _combine_analyses(self, symbol: str, market_data: Dict[str, Any], 
                         technical_analysis: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analizleri birleştir"""
        return {
            'symbol': symbol,
            'market_data': market_data,
            'technical_analysis': technical_analysis,
            'ai_analysis': ai_analysis,
            'combined_score': self._calculate_combined_score(technical_analysis, ai_analysis),
            'recommendation': self._generate_recommendation(technical_analysis, ai_analysis),
            'risk_level': self._assess_risk_level(technical_analysis, ai_analysis),
            'timestamp': datetime.now()
        }
    
    def _calculate_combined_score(self, technical: Dict[str, Any], ai: Dict[str, Any]) -> float:
        """Kombine skor hesapla"""
        try:
            score = 0.0
            weight = 0.0
            
            # Teknik analiz skoru
            if technical.get('rsi'):
                rsi_score = 1.0 - abs(technical['rsi'] - 50) / 50  # RSI 50'ye yakın = yüksek skor
                score += rsi_score * 0.2
                weight += 0.2
            
            if technical.get('macd') and technical.get('macd_signal'):
                macd_score = 1.0 if technical['macd'] > technical['macd_signal'] else 0.0
                score += macd_score * 0.15
                weight += 0.15
            
            if technical.get('bb_position'):
                bb_score = technical['bb_position']  # 0-1 arası
                score += bb_score * 0.1
                weight += 0.1
            
            # AI analiz skoru
            if ai.get('sentiment', {}).get('confidence'):
                sentiment_score = ai['sentiment']['confidence']
                if ai['sentiment'].get('sentiment') == 'bullish':
                    score += sentiment_score * 0.3
                elif ai['sentiment'].get('sentiment') == 'bearish':
                    score += (1 - sentiment_score) * 0.3
                else:
                    score += 0.5 * 0.3
                weight += 0.3
            
            if ai.get('signals', {}).get('strength'):
                signal_score = ai['signals']['strength']
                if ai['signals'].get('signal') == 'BUY':
                    score += signal_score * 0.25
                elif ai['signals'].get('signal') == 'SELL':
                    score += (1 - signal_score) * 0.25
                else:
                    score += 0.5 * 0.25
                weight += 0.25
            
            return score / weight if weight > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Kombine skor hesaplama hatası: {e}")
            return 0.5
    
    def _generate_recommendation(self, technical: Dict[str, Any], ai: Dict[str, Any]) -> str:
        """Tavsiye üret"""
        try:
            # AI sinyali öncelikli
            if ai.get('signals', {}).get('signal'):
                return ai['signals']['signal']
            
            # Teknik analiz
            if technical.get('rsi') and technical.get('macd'):
                if technical['rsi'] < 30 and technical['macd'] > technical.get('macd_signal', 0):
                    return "BUY"
                elif technical['rsi'] > 70 and technical['macd'] < technical.get('macd_signal', 0):
                    return "SELL"
            
            return "HOLD"
            
        except Exception as e:
            self.logger.error(f"Tavsiye üretme hatası: {e}")
            return "HOLD"
    
    def _assess_risk_level(self, technical: Dict[str, Any], ai: Dict[str, Any]) -> str:
        """Risk seviyesi değerlendir"""
        try:
            risk_factors = 0
            
            # Volatilite
            if technical.get('volatility', 0) > 5:
                risk_factors += 1
            
            # RSI aşırı değerler
            if technical.get('rsi', 50) < 20 or technical.get('rsi', 50) > 80:
                risk_factors += 1
            
            # AI risk değerlendirmesi
            if ai.get('sentiment', {}).get('risk_level') == 'high':
                risk_factors += 1
            
            if risk_factors >= 2:
                return "high"
            elif risk_factors == 1:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Risk değerlendirme hatası: {e}")
            return "medium"
    
    # Teknik analiz fonksiyonları
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI hesapla"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD hesapla"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            return macd.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1]
        except:
            return 0, 0, 0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands hesapla"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
        except:
            return 0, 0, 0
    
    def _get_bb_position(self, price: float, upper: float, lower: float) -> float:
        """Bollinger Bands pozisyonu (0-1 arası)"""
        try:
            if upper == lower:
                return 0.5
            return (price - lower) / (upper - lower)
        except:
            return 0.5
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR hesapla"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
            return atr.iloc[-1] if not atr.empty else 0
        except:
            return 0
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """ADX hesapla"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = self._calculate_atr(df, 1)
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            return adx.iloc[-1] if not adx.empty else 0
        except:
            return 0
    
    def _calculate_stoch_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Stochastic RSI hesapla"""
        try:
            rsi = self._calculate_rsi(prices, period)
            rsi_min = prices.rolling(period).min()
            rsi_max = prices.rolling(period).max()
            stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
            return stoch_rsi.iloc[-1] if not stoch_rsi.empty else 50
        except:
            return 50
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """Trend gücü hesapla"""
        try:
            sma_short = prices.rolling(period//2).mean()
            sma_long = prices.rolling(period).mean()
            trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1] * 100
            return trend_strength
        except:
            return 0
    
    def _notify_callbacks(self, analysis: Dict[str, Any]):
        """Callback'leri çağır"""
        for callback in self.analysis_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Callback hatası: {e}")
    
    def add_analysis_callback(self, callback):
        """Analiz callback'i ekle"""
        self.analysis_callbacks.append(callback)
    
    def get_latest_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Son analizi al"""
        cached = self.analysis_cache.get(symbol)
        if cached and time.time() - cached['timestamp'] < self.cache_duration:
            return cached['data']
        return None
    
    def get_all_analyses(self) -> Dict[str, Any]:
        """Tüm analizleri al"""
        return {symbol: data['data'] for symbol, data in self.analysis_cache.items() 
                if time.time() - data['timestamp'] < self.cache_duration}

# Global market analyzer instance
market_analyzer = MarketAnalyzer()
