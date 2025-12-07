"""
Binance WebSocket İstemcisi
- Multi-stream tek bağlantı: wss://stream.binance.com:9443/stream?streams=<...>
- Testnet: wss://testnet.binance.vision/stream?streams=<...>
- Öncelikli kuyruk: ticker/depth yüksek öncelik, overflow'da düşük öncelikleri düşür
- Metrikler: ws_lag_ms, dropped_msgs, ws_queue_size
"""
import asyncio
import json
import logging
import ssl
import time
import random
from typing import List, Dict, Any, Optional, Callable
from collections import deque

import websockets
from config.config import config
from utils.metrics import metrics

class PriorityWSQueue:
    def __init__(self, maxlen: int = 2000):
        self.high = deque(maxlen=maxlen)
        self.low = deque(maxlen=maxlen)
    def put(self, item: Dict[str, Any], high: bool = False) -> bool:
        if high:
            if len(self.high) == self.high.maxlen:
                # Drop oldest high-priority if needed
                try:
                    self.high.popleft()
                except Exception:
                    pass
            self.high.append(item)
            return True
        else:
            if len(self.low) == self.low.maxlen:
                # drop low-priority on overflow
                return False
            self.low.append(item)
            return True
    def get(self) -> Optional[Dict[str, Any]]:
        if self.high:
            return self.high.popleft()
        if self.low:
            return self.low.popleft()
        return None
    def size(self) -> int:
        return len(self.high) + len(self.low)

class BinanceWebSocket:
    def __init__(self, symbols: List[str], channels: List[str] = None, testnet: bool = False):
        self.logger = logging.getLogger(__name__)
        self.symbols = [s.lower() for s in symbols]
        self.channels = channels or ["ticker", "depth@100ms", "trade"]
        app_mode = (getattr(config.api, 'app_mode', 'paper') or 'paper').lower()
        self.testnet = testnet or app_mode in ("test", "testnet")
        self.base_url = (
            "wss://testnet.binance.vision/stream?streams=" if self.testnet else "wss://stream.binance.com:9443/stream?streams="
        )
        self.ws = None
        self.is_running = False
        self.queue = PriorityWSQueue(maxlen=5000)
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._ssl = ssl.create_default_context()
        # Metrics
        self.m_lag = metrics.histogram('ws_lag_ms')
        self.m_drop = metrics.counter('dropped_msgs')
        self.m_q = metrics.gauge('ws_queue_size')
        # Backoff
        self._backoff_base = 0.5
        self._backoff_n = 0
        self._backoff_cap = 15.0

    def _streams_param(self) -> str:
        parts = []
        for s in self.symbols:
            for ch in self.channels:
                parts.append(f"{s}@{ch}" if '@' not in ch else f"{s}@{ch}")
        return "/".join(parts)

    def add_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self.callbacks.append(cb)

    async def _connect(self):
        url = self.base_url + self._streams_param()
        self.logger.info(f"Connecting WS: {url}")
        return await websockets.connect(url, ssl=self._ssl, ping_interval=30, ping_timeout=10)

    async def _recv_loop(self):
        while self.is_running:
            try:
                self.ws = await self._connect()
                self._backoff_n = 0
                async for msg in self.ws:
                    try:
                        t_recv = time.time()
                        data = json.loads(msg)
                        # Determine priority
                        stream = str(data.get('stream',''))
                        p_high = any(x in stream for x in ('@ticker', '@depth'))
                        ok = self.queue.put({'t': t_recv, 'data': data}, high=p_high)
                        if not ok:
                            self.m_drop.inc(1)
                        self.m_q.set(self.queue.size())
                        # Process callbacks lightweight
                        item = self.queue.get()
                        if item:
                            lag_ms = (t_recv - item['t']) * 1000.0
                            self.m_lag.observe(lag_ms)
                            for cb in self.callbacks:
                                try:
                                    cb(item['data'])
                                except Exception:
                                    pass
                    except Exception:
                        self.m_drop.inc(1)
                        continue
            except Exception as e:
                self.logger.warning(f"WS error: {e}")
            # Reconnect with jitter backoff
            await self._sleep_backoff()

    async def _sleep_backoff(self):
        try:
            wait = min(self._backoff_cap, self._backoff_base * (2 ** self._backoff_n))
            jitter = random.uniform(0, self._backoff_base)
            self._backoff_n = min(self._backoff_n + 1, 6)
            await asyncio.sleep(max(0.0, wait + jitter))
        except Exception:
            await asyncio.sleep(1.0)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        asyncio.get_event_loop().create_task(self._recv_loop())

    async def stop(self):
        self.is_running = False
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass
