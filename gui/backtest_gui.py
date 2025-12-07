import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime, timedelta
import time
from typing import List, Tuple

from api.multi_api_manager import multi_api_manager
from strategies.strategy_manager import strategy_manager

class BacktestGUI:
    def __init__(self, parent):
        self.parent = parent
        self.root = tk.Toplevel(parent)
        self.root.title("Backtest")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.symbol_var = tk.StringVar(value="BTCUSDT")
        try:
            strategies = list(strategy_manager.get_available_strategies() or [])
        except Exception:
            strategies = ["trend_following", "scalping", "mean_reversion"]
        default_strategy = strategies[0] if strategies else "trend_following"
        self.strategy_var = tk.StringVar(value=default_strategy)
        self.interval_var = tk.StringVar(value="1d")
        self.months_var = tk.StringVar(value="6")

        self._build_ui(strategies)

    def _build_ui(self, strategies: List[str]):
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        form = ttk.LabelFrame(frm, text="Ayarlar")
        form.pack(fill=tk.X)

        # Sembol
        ttk.Label(form, text="Sembol:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(form, textvariable=self.symbol_var, width=16).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Strateji
        ttk.Label(form, text="Strateji:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        cb = ttk.Combobox(form, textvariable=self.strategy_var, values=strategies, state="readonly", width=22)
        cb.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Süre
        ttk.Label(form, text="Süre (ay):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(form, textvariable=self.months_var, width=5).grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)

        # Çalıştır
        run_btn = ttk.Button(form, text="Test Et", command=self.run_backtest)
        run_btn.grid(row=0, column=6, sticky=tk.W, padx=10, pady=5)

        # Sonuçlar
        self.results = tk.Text(frm, height=25, wrap=tk.WORD)
        self.results.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        self.results.insert(tk.END, "6 aylık 1D veride basit EMA crossover ile test yapılacaktır.\n")
        self.results.config(state=tk.DISABLED)

    def _append(self, text: str):
        try:
            self.results.config(state=tk.NORMAL)
            self.results.insert(tk.END, text + "\n")
            self.results.see(tk.END)
            self.results.config(state=tk.DISABLED)
        except Exception:
            pass

    def run_backtest(self):
        try:
            months = int(self.months_var.get().strip())
            if months <= 0:
                months = 6
        except Exception:
            months = 6
        symbol = (self.symbol_var.get() or "BTCUSDT").strip().upper()
        strategy = (self.strategy_var.get() or "trend_following").strip()
        interval = (self.interval_var.get() or "1d").strip()

        threading.Thread(target=self._run_worker, args=(symbol, strategy, interval, months), daemon=True).start()

    def _run_worker(self, symbol: str, strategy: str, interval: str, months: int):
        try:
            self._append(f"Veri indiriliyor: {symbol}, {months} ay, interval={interval} ...")
            end = int(time.time() * 1000)
            start_dt = datetime.utcnow() - timedelta(days=months*30)
            start = int(start_dt.timestamp() * 1000)

            kl = self._fetch_klines(symbol, interval, start, end)
            if not kl:
                self._append("Veri alınamadı veya boş.")
                return
            closes = [float(c[4]) for c in kl if len(c) >= 5]
            times = [int(c[0]) for c in kl]
            if len(closes) < 50:
                self._append("Yeterli veri yok.")
                return

            # Basit strateji mapping: trend_following -> EMA(12/26) crossover
            # mean_reversion -> Bollinger band (20,2) ters işlem, scalping -> küçük eşik momentum
            if strategy == "mean_reversion":
                pnl, n_trades = self._backtest_bollinger(closes)
            elif strategy == "scalping":
                pnl, n_trades = self._backtest_momentum(closes)
            else:
                pnl, n_trades = self._backtest_ema(closes)

            self._append(f"Toplam İşlem: {n_trades}")
            self._append(f"Toplam PnL (yüzde): {pnl*100:.2f}%")
        except Exception as e:
            self._append(f"Hata: {e}")

    def _fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[list]:
        """Binance uyumlu klines çeker. Çok uzun aralığı parçalara böler."""
        all_rows: List[list] = []
        cur = start_ms
        # Binance limit ~1000 bar
        step = 1000
        while cur < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": 1000
            }
            resp = {}
            try:
                resp = multi_api_manager.market_api._make_request("GET", "/klines", params=params)
            except Exception:
                resp = {}
            if isinstance(resp, dict) and resp.get("error"):
                break
            if not isinstance(resp, list) or not resp:
                break
            all_rows.extend(resp)
            last_open = int(resp[-1][0])
            # ilerle
            if last_open <= cur:
                break
            cur = last_open + 1
            if len(resp) < 1000:
                break
            time.sleep(0.2)
        return all_rows

    def _ema(self, data: List[float], period: int) -> List[float]:
        k = 2 / (period + 1)
        ema = []
        for i, val in enumerate(data):
            if i == 0:
                ema.append(val)
            else:
                ema.append(val * k + ema[-1] * (1 - k))
        return ema

    def _backtest_ema(self, closes: List[float]) -> Tuple[float, int]:
        fast = self._ema(closes, 12)
        slow = self._ema(closes, 26)
        pos = 0
        entry = 0.0
        pnl = 0.0
        trades = 0
        for i in range(1, len(closes)):
            if pos == 0 and fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
                pos = 1
                entry = closes[i]
                trades += 1
            elif pos == 1 and fast[i] < slow[i] and fast[i-1] >= slow[i-1]:
                pnl += (closes[i] - entry) / entry
                pos = 0
        if pos == 1:
            pnl += (closes[-1] - entry) / entry
        return pnl, trades

    def _backtest_bollinger(self, closes: List[float]) -> Tuple[float, int]:
        import statistics as stats
        period = 20
        mult = 2.0
        pos = 0
        entry = 0.0
        pnl = 0.0
        trades = 0
        for i in range(period, len(closes)):
            window = closes[i-period:i]
            ma = sum(window) / period
            sd = stats.pstdev(window) if len(window) > 1 else 0.0
            upper = ma + mult * sd
            lower = ma - mult * sd
            price = closes[i]
            if pos == 0 and price < lower:
                pos = 1
                entry = price
                trades += 1
            elif pos == 1 and price > ma:
                pnl += (price - entry) / entry
                pos = 0
        if pos == 1:
            pnl += (closes[-1] - entry) / entry
        return pnl, trades

    def _backtest_momentum(self, closes: List[float]) -> Tuple[float, int]:
        lookback = 5
        thr = 0.01
        pos = 0
        entry = 0.0
        pnl = 0.0
        trades = 0
        for i in range(lookback, len(closes)):
            ret = (closes[i] - closes[i-lookback]) / closes[i-lookback]
            if pos == 0 and ret > thr:
                pos = 1
                entry = closes[i]
                trades += 1
            elif pos == 1 and ret < 0:
                pnl += (closes[i] - entry) / entry
                pos = 0
        if pos == 1:
            pnl += (closes[-1] - entry) / entry
        return pnl, trades
