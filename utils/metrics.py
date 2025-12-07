import threading
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional

class Counter:
    def __init__(self):
        self._v = 0
        self._lock = threading.Lock()
    def inc(self, value: int = 1):
        with self._lock:
            self._v += int(value)
    def get(self) -> int:
        with self._lock:
            return self._v

class Gauge:
    def __init__(self):
        self._v: float = 0.0
        self._lock = threading.Lock()
    def set(self, value: float):
        with self._lock:
            self._v = float(value)
    def add(self, delta: float):
        with self._lock:
            self._v += float(delta)
    def get(self) -> float:
        with self._lock:
            return self._v

class Histogram:
    def __init__(self, maxlen: int = 1024):
        self._values = deque(maxlen=maxlen)
        self._lock = threading.Lock()
    def observe(self, value: float):
        with self._lock:
            self._values.append(float(value))
    def summary(self) -> Dict[str, float]:
        with self._lock:
            if not self._values:
                return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            arr = list(self._values)
        n = len(arr)
        arr_sorted = sorted(arr)
        def pct(p):
            if n == 0: return 0.0
            idx = min(n-1, max(0, int(p * (n-1))))
            return arr_sorted[idx]
        return {
            "count": float(n),
            "avg": sum(arr) / n,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "max": max(arr)
        }

class MetricsRegistry:
    def __init__(self):
        self.counters: Dict[str, Counter] = defaultdict(Counter)
        self.gauges: Dict[str, Gauge] = defaultdict(Gauge)
        self.histograms: Dict[str, Histogram] = defaultdict(Histogram)
    def counter(self, name: str) -> Counter:
        return self.counters[name]
    def gauge(self, name: str) -> Gauge:
        return self.gauges[name]
    def histogram(self, name: str) -> Histogram:
        return self.histograms[name]
    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": {k: v.get() for k, v in self.counters.items()},
            "gauges": {k: v.get() for k, v in self.gauges.items()},
            "histograms": {k: v.summary() for k, v in self.histograms.items()},
        }

# Global registry
metrics = MetricsRegistry()
