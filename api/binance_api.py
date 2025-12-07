"""
Binance Spot API Integration (Paper Mode)
- Translates BTCTurk-like endpoints to Binance REST
- Normalizes responses to existing schema expected by GUI/multi_api_manager
- Only public market data endpoints are actively used in paper mode
"""

import requests
import time
import logging
import random
from typing import Dict, List, Optional

from config.config import config
from utils.metrics import metrics


class BinanceAPI:
    """Binance Spot REST API wrapper with BTCTurk-compatible interface where possible."""

    def __init__(self):
        # Base URL: live vs testnet vs paper
        app_mode = (getattr(config.api, 'app_mode', 'paper') or 'paper').lower()
        if app_mode in ('test', 'testnet'):
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        # Rate limit simple pacing
        self.last_request_time = 0.0
        self.min_request_interval = 0.1
        # Backoff
        self._backoff_base = 0.25
        self._backoff_n = 0
        self._backoff_cap = 8.0
        # Metrics
        self._m_latency = metrics.histogram('api_latency_ms')
        self._m_errors = metrics.counter('api_errors')
        # Health-check on init (non-fatal)
        try:
            self._health_check()
        except Exception:
            pass

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        # simple pacing
        now = time.time()
        dt = now - self.last_request_time
        if dt < self.min_request_interval:
            time.sleep(self.min_request_interval - dt)
        self.last_request_time = time.time()

        url = self.base_url.rstrip('/') + '/' + path.lstrip('/')
        # Ensure health before call if previously failing
        try:
            t0 = time.time()
            if method.upper() == 'GET':
                r = self.session.get(url, params=params, timeout=10)
            else:
                r = self.session.request(method.upper(), url, params=params, timeout=10)
            r.raise_for_status()
            latency_ms = (time.time() - t0) * 1000.0
            self._m_latency.observe(latency_ms)
            # Rate limit headers could be inspected here if needed
            return r.json()
        except Exception as e:
            self._m_errors.inc(1)
            self.logger.error(f"Binance request error: {e}")
            # Backoff with jitter
            self._sleep_with_backoff()
            # Health-check gate before re-activate
            try:
                self._health_check()
            except Exception as _:
                pass
            return {"error": str(e)}

    def _sleep_with_backoff(self):
        try:
            wait = min(self._backoff_cap, self._backoff_base * (2 ** self._backoff_n))
            jitter = random.uniform(0, self._backoff_base)
            time.sleep(max(0.0, wait + jitter))
            self._backoff_n = min(self._backoff_n + 1, 6)
        except Exception:
            pass

    def _health_check(self):
        try:
            url_ping = self.base_url.rstrip('/').replace('/api/v3', '') + '/api/v3/ping'
            # for testnet base '/api' needs '/api/v3/ping' already included above for live; adjust if using /api
            if 'testnet.binance.vision' in url_ping and not url_ping.endswith('/ping'):
                url_ping = self.base_url.rstrip('/') + '/v3/ping'
            t0 = time.time()
            r = self.session.get(url_ping, timeout=5)
            r.raise_for_status()
            self._m_latency.observe((time.time() - t0) * 1000.0)
            # reset backoff on success
            self._backoff_n = 0
            return True
        except Exception as e:
            self.logger.warning(f"Health-check failed: {e}")
            self._m_errors.inc(1)
            return False

    # ----- Normalized helpers -----
    def get_all_tickers_normalized(self) -> List[Dict]:
        """Return list of dicts with keys: pairSymbol, last, changePercent (if available)."""
        # Use 24hr ticker to include percent change
        data = self._request('GET', 'ticker/24hr')
        if isinstance(data, dict) and 'error' in data:
            return []
        out = []
        try:
            for item in data:
                sym = str(item.get('symbol', '')).upper()
                last = item.get('lastPrice') or item.get('weightedAvgPrice') or item.get('bidPrice')
                try:
                    last_f = float(last) if last is not None else 0.0
                except Exception:
                    last_f = 0.0
                try:
                    chg = float(item.get('priceChangePercent') or 0.0)
                except Exception:
                    chg = 0.0
                # Yalnızca USDT pariteleri ve fiyatı en az 0.0001 USDT olan coinleri al
                if sym.endswith('USDT') and last_f >= 0.0001:
                    out.append({
                        'pairSymbol': sym,
                        'last': last_f,
                        'dailyChangePercent': chg
                    })
            # Synthesize USDTTRY from TRYUSDT if available
            tryusdt = next((x for x in out if x['pairSymbol'] == 'TRYUSDT'), None)
            if tryusdt and tryusdt.get('last', 0) > 0:
                inv = 1.0 / float(tryusdt['last'])
                out.append({'pairSymbol': 'USDTTRY', 'last': inv, 'dailyChangePercent': 0.0})
        except Exception:
            pass
        return out

    def get_symbol_ticker_normalized(self, symbol: str) -> Dict:
        """Return dict with keys: pairSymbol, last, dailyChangePercent."""
        symbol = (symbol or '').upper()
        if symbol == 'USDTTRY':
            # compute from TRYUSDT
            d = self._request('GET', 'ticker/price', params={'symbol': 'TRYUSDT'})
            try:
                px = float(d.get('price', 0))
                inv = (1.0 / px) if px > 0 else 0.0
            except Exception:
                inv = 0.0
            return {'pairSymbol': 'USDTTRY', 'last': inv, 'dailyChangePercent': 0.0}
        # Use 24hr ticker for single symbol
        d = self._request('GET', 'ticker/24hr', params={'symbol': symbol})
        if isinstance(d, dict) and 'error' in d:
            return {}
        try:
            last = d.get('lastPrice') or d.get('weightedAvgPrice') or d.get('bidPrice')
            last_f = float(last) if last is not None else 0.0
            chg = float(d.get('priceChangePercent') or 0.0)
            return {'pairSymbol': symbol, 'last': last_f, 'dailyChangePercent': chg}
        except Exception:
            return {}

    def get_orderbook_normalized(self, symbol: str, limit: int = 50) -> Dict:
        symbol = (symbol or '').upper()
        d = self._request('GET', 'depth', params={'symbol': symbol, 'limit': limit})
        if isinstance(d, dict) and 'error' in d:
            return {}
        return {
            'symbol': symbol,
            'bids': d.get('bids', []),
            'asks': d.get('asks', []),
        }

    def get_trades_normalized(self, symbol: str, limit: int = 50) -> List[Dict]:
        symbol = (symbol or '').upper()
        d = self._request('GET', 'trades', params={'symbol': symbol, 'limit': limit})
        if isinstance(d, dict) and 'error' in d:
            return []
        out = []
        for it in d:
            try:
                out.append({
                    'symbol': symbol,
                    'price': float(it.get('price', 0)),
                    'qty': float(it.get('qty', 0)),
                    'time': it.get('time'),
                    'isBuyerMaker': it.get('isBuyerMaker'),
                })
            except Exception:
                pass
        return out

    def get_klines_normalized(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        symbol = (symbol or '').upper()
        d = self._request('GET', 'klines', params={'symbol': symbol, 'interval': interval, 'limit': limit})
        if isinstance(d, dict) and 'error' in d:
            return []
        return d

    # ----- Compatibility layer for BTCTurk-like calls -----
    def _make_request(self, method: str, endpoint: str, params: dict | None = None, data: dict | None = None) -> dict:
        """
        Accept BTCTurk-like endpoints and map to Binance, returning a dict possibly with 'data'.
        """
        ep = (endpoint or '').strip()
        params = params or {}
        if ep == '/ticker':
            symbol = params.get('pairSymbol') or params.get('symbol')
            if symbol:
                item = self.get_symbol_ticker_normalized(symbol)
                return {'data': [item] if item else []}
            items = self.get_all_tickers_normalized()
            return {'data': items}
        if ep == '/orderbook':
            symbol = params.get('pairSymbol') or params.get('symbol')
            if not symbol:
                return {'error': 'symbol required'}
            ob = self.get_orderbook_normalized(symbol)
            return ob
        if ep == '/trades':
            symbol = params.get('pairSymbol') or params.get('symbol')
            limit = int(params.get('last') or params.get('limit') or 50)
            lst = self.get_trades_normalized(symbol, limit)
            return {'data': lst}
        if ep == '/klines':
            symbol = params.get('pairSymbol') or params.get('symbol')
            interval = params.get('interval') or '1m'
            limit = int(params.get('limit') or 100)
            lst = self.get_klines_normalized(symbol, interval, limit)
            return {'data': lst}
        # Unsupported private endpoints in paper mode
        if ep in ('/users/balances', '/openOrders', '/allOrders', '/order'):
            return {'error': 'Private endpoints disabled in paper mode'}
        return {'error': f'Unsupported endpoint: {endpoint}'}


# Global instance (optional, most code uses MultiAPIManager)
binance_api = BinanceAPI()
