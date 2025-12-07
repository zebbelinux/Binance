"""
Trading Dashboard
Ana trading arayüzü ve kontrol paneli
"""

from orchestrator import SonModelOrchestrator

USE_ORCHESTRATOR = True  # Binance için SonModel Orchestrator'ı aktifleştirmek için

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import threading
import time
from datetime import datetime, timedelta
import datetime as _dtmod
from typing import Dict, List, Any, Optional
import json
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from collections import deque
import random
from queue import Queue, Empty

from utils.logger import get_logger, LogCategory
from strategies.strategy_manager import strategy_manager
from trading.paper_executor import paper_executor
from trading.inventory_twap import InventoryTwapLiquidator
from api.multi_api_manager import MultiAPIManager
from ai.market_analyzer import MarketAnalyzer
from risk_management.risk_manager import risk_manager
from data.data_manager import data_manager
from gui.strategy_settings_gui import StrategySettingsGUI
from gui.manual_trading_gui import ManualTradingGUI
from gui.orderbook_gui import OrderBookGUI
from gui.advanced_strategy_selector import AdvancedStrategySelector
from indicators.orderbook_analyzer import OrderBookSnapshot as OBSnapshot, OrderBookLevel as OBLevel
from gui.api_management_gui import APIManagementGUI
from risk_management.position_sizer import position_sizer, PositionSizingMethod

class TradingDashboard:
    """Trading dashboard sınıfı"""
    
    def __init__(self, root: tk.Tk, multi_api_manager: MultiAPIManager, market_analyzer: MarketAnalyzer, basic_mode: bool = False, init_callback=None, init_steps: dict | None = None):
        self.root = root
        self.multi_api_manager = multi_api_manager
        self.market_analyzer = market_analyzer
        self.logger = get_logger("trading_dashboard")

        self.use_orchestrator: bool = bool(USE_ORCHESTRATOR)
        self.orchestrator: Optional[SonModelOrchestrator] = None

        if self.use_orchestrator:
            try:
                self.orchestrator = SonModelOrchestrator(
                    mode="paper",
                    risk_config_path="config/risk_settings.json",
                )
                self.logger.info(LogCategory.GUI, "Orchestrator modu: AÇIK (Binance / paper)")
            except Exception as e:
                self.logger.error(LogCategory.GUI, f"Orchestrator başlatma hatası, legacy moda dönülüyor: {e}")
                self.use_orchestrator = False

        self._basic_mode = bool(basic_mode)
        self._init_callback = init_callback
        self._init_steps = init_steps or {}
        
        # GUI durumu
        self.is_running = False
        self.update_thread = None
        self._price_poller_thread = None
        self._price_poller_running = False
        self._start_stop_busy = False
        # Geçici: UI donma teşhisi için güvenli mod (ağır işler kapalı)
        self._safe_mode = True
        # Mum grafik verisi (1 dakikalık)
        self._candles = []  # list of dict: {t,o,h,l,c}
        self._current_candle = None
        # Strateji motoru (tarama) durumu
        self._engine_enabled = False
        # Strateji worker durumu
        self._strategy_inflight = False
        self._last_strategy_time = 0.0
        self._strategy_min_interval = 2.0  # sn
        self._last_strategy_log_time = 0.0  # log en az 5 sn'de bir
        # Sembol tarama listesi ve kalıcılık
        self._scan_symbols = []  # USDT ve TRY pariteleri
        self._last_account_save = 0.0
        try:
            from trading.paper_executor import paper_executor
            paper_executor.load()
            # Enforce Binance paper constraints
            try:
                paper_executor.min_notional_usdt = 10.0
            except Exception:
                pass
            # Başlangıç: Yalnızca hiç bakiye ve pozisyon yoksa 4000 USDT ile başlat
            try:
                if (float(getattr(paper_executor, 'balance_usdt', 0.0) or 0.0) <= 0.0) and not (getattr(paper_executor, 'positions', None) or {}):
                    paper_executor.set_starting_usdt(4000.0)
            except Exception:
                pass
            # UI henüz kurulmadan önce log_text yok; doğrudan logger kullan
            try:
                self.logger.info(LogCategory.GUI, "Paper hesap yüklendi")
            except Exception as e:
                print(f"Paper hesap yükleme log hatası: {e}")
        except Exception as e:
            print(f"Paper hesap yükleme hatası: {e}")
            self.logger.error(LogCategory.GUI, f"Paper hesap yükleme hatası: {e}")
        
        # Veri cache ve başlangıç durumları
        self.market_data = {}
        self.positions = {}
        self.trades = []
        self.performance_data = {}
        # API bağlantı durumu cache
        self._last_api_check_time = 0
        self._last_api_ok = False
        self._api_check_inflight = False
        # Paper trading icin basit RSI sim
        self._rsi_value = 50.0
        self._last_price = 0.0
        # Gerçek RSI icin cache ve fetch durumu
        self._last_real_rsi = None
        self._last_klines_fetch = 0
        self._klines_inflight = False
        # Grafik icin kapanis verisi cache
        self._closes_cache = []
        # Klines hata loglarini kisitlamak icin zaman damgasi
        self._last_klines_error_time = 0
        # Demo sinyal frekans kontrolu (gercek RSI yokken)
        self._last_demo_trade_time = 0
        # Sembol ve fiyat cache
        self._current_symbol = 'BTCUSDT'
        self._price_cache: Dict[str, float] = {}
        self._usdt_try = 0.0
        # Strateji eşikleri (UI'dan ayarlanabilir)
        self._buy_threshold_pct = -0.5
        self._sell_threshold_pct = 0.5
        try:
            self._load_threshold_config()
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Threshold config yükleme hatası: {e}")
        # Grafik öğeleri
        self._mpl_fig = None
        self._mpl_ax = None
        self._mpl_canvas = None
        self._mpl_toolbar = None
        self._chart_after_id = None
        self._after_id = None  # Update loop zamanlayıcısı
        # Basit tarama motoru durumları
        self._prev_prices: Dict[str, float] = {}
        self._last_trade_time: Dict[str, float] = {}
        self._trade_cooldown_sec: float = 12.0
        self._price_hist: Dict[str, deque] = {}
        self._engine_ref_window_sec: float = 30.0
        self._last_engine_log_time: float = 0.0
        # Otomatik işlem yönetimi
        self.auto_enabled: bool = False
        self._auto_thread: Optional[threading.Thread] = None
        self.max_open_positions: int = 20
        # Son AI sinyali (sembol bazlı)
        self._last_ai_signal: Dict[str, str] = {}
        # 24s değişim yüzdesi cache'i
        self._change24_cache: Dict[str, float] = {}
        # Inventory TWAP likidasyon durumu
        self._twap_liquidator = InventoryTwapLiquidator(slice_interval_sec=5.0)
        self._twap_threads: Dict[str, threading.Thread] = {}

        # GUI bileşenleri: Senkron kurulum (sade ve güvenli)
        try:
            self.logger.info(LogCategory.GUI, "setup_gui SENKRON başlıyor")
            self.setup_gui()
            self.logger.info(LogCategory.GUI, "setup_gui SENKRON tamamlandı")
        except Exception as ex:
            # Güvenli geri dönüş: minimal panel
            self.logger.error(LogCategory.GUI, f"setup_gui hatası: {ex}")
            try:
                fallback = ttk.Frame(self.root)
                fallback.pack(fill=tk.BOTH, expand=True)
                ttk.Label(fallback, text=f"GUI kurulum hatası: {ex}", foreground='red').pack(pady=20)
            except Exception as e:
                self.logger.error(LogCategory.GUI, f"Fallback GUI oluşturma hatası: {e}")

        # UI kuyruğu: tüm widget güncellemeleri ana thread'de işlenir
        try:
            self._ui_queue: Queue = Queue()
            self.root.after(50, self._drain_ui_queue)
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"UI kuyruğu başlatma hatası: {e}")

        # Binance USDT evreni
        self._binance_usdt_universe: set[str] = set(["BTCUSDT"])  # en azından BTCUSDT bulunsun
        try:
            threading.Thread(target=self._load_binance_usdt_universe, daemon=True).start()
            threading.Thread(target=self._refresh_universe_periodically, daemon=True).start()
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"USDT evreni başlatma hatası: {e}")

        # Sinyal callback kaydi (paper trading ve log icin)
        try:
            strategy_manager.add_signal_callback(self.on_signals)
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Signal callback kayıt hatası: {e}")
        
        # Piyasa verisi snapshot döngüsünü hemen başlat (sinyal bekleme!)
        if not getattr(self, '_basic_mode', False):
            try:
                self._start_ticker_snapshot_loop()
            except Exception as e:
                self.logger.error(LogCategory.GUI, f"Ticker snapshot döngüsü başlatılamadı: {e}")
            try:
                self.schedule_update_loop()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Update loop başlatılamadı: {e}")
    
    def on_signals(self, signals: List[Dict[str, Any]]):
        """Sinyal callback metodu"""
        try:
            exec_signals = []
            for signal in signals:
                self.logger.info(LogCategory.GUI, f"Yeni sinyal alındı: {signal}")
                try:
                    sym = (signal.get('symbol') or '').strip().upper()
                    side = (signal.get('side') or '').lower()
                    st_name = signal.get('strategy_name') or ''
                    if sym:
                        if side == 'buy':
                            self._last_ai_signal[sym] = 'AL'
                        elif side == 'sell':
                            self._last_ai_signal[sym] = 'SAT'
                        else:
                            self._last_ai_signal[sym] = 'NÖTR'
                        if st_name:
                            if not hasattr(self, '_last_strategy_name'):
                                self._last_strategy_name = {}
                            self._last_strategy_name[sym] = st_name
                        # Oto trade: yalnız USDT pariteler
                        if sym.endswith('USDT') and side in ('buy','sell'):
                            # Fiyat: sinyal price/entry_price ya da cache
                            px = 0.0
                            try:
                                px = float(signal.get('entry_price') or signal.get('price') or 0.0)
                            except Exception:
                                px = 0.0
                            if not (px and px > 0):
                                px = float(self._get_last_price_for_symbol(sym) or 0.0)
                            if not (px and px > 0):
                                continue
                            # Miktar: size/qty/amount varsa onu kullan, yoksa usdt/notional'dan türet, o da yoksa UI toplamı, aksi halde 10 USDT/px
                            qty = None
                            for k in ('size','qty','amount'):
                                v = signal.get(k)
                                if v is not None:
                                    try:
                                        qty = float(v)
                                        break
                                    except Exception:
                                        pass
                            if qty is None:
                                usdt_notional = None
                                for k in ('usdt','notional','total','quote_amount'):
                                    v = signal.get(k)
                                    if v is not None:
                                        try:
                                            usdt_notional = float(v)
                                            break
                                        except Exception:
                                            pass
                                if usdt_notional is None:
                                    # UI toplam kutusundan dene
                                    try:
                                        ui_total = (self.paper_total_entry.get() or '').strip().replace(',','.')
                                        usdt_notional = float(ui_total) if ui_total else None
                                    except Exception:
                                        usdt_notional = None
                                if usdt_notional and usdt_notional > 0 and px > 0:
                                    qty = usdt_notional / px
                            if qty is None or qty <= 0:
                                # Position sizer kullanarak hesapla
                                try:
                                    account_balance = float(paper_executor.balance or 1000.0)
                                    signal_obj = {
                                        'symbol': sym,
                                        'side': side,
                                        'entry_price': px,
                                        'strength': 0.7,  # Orta-yüksek güç
                                        'confidence': 0.8
                                    }
                                    position_value, _ = position_sizer.calculate_position_size(
                                        method=PositionSizingMethod.SIGNAL_BASED_FULL_CAPITAL,
                                        account_balance=account_balance,
                                        signal=signal_obj,
                                        market_data={},
                                        historical_performance=None
                                    )
                                    qty = position_value / px if px > 0 else 0.0
                                    self.logger.info(LogCategory.GUI, f"Fallback position size: {position_value:.2f} USD = {qty:.6f} {sym}")
                                except Exception as e:
                                    self.logger.error(LogCategory.GUI, f"Fallback position size hatası: {e}")
                                    qty = min(250.0, account_balance) / px if px > 0 else 0.0
                            if qty and qty > 0:
                                exec_signals.append({
                                    'symbol': sym,
                                    'side': side,
                                    'size': qty,
                                    'entry_price': px,
                                    'strategy_name': st_name or 'auto'
                                })
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Sinyal işleme hatası: {e}")

        # Orchestrator modu: StrategyManager sinyallerini meta + risk + executor pipeline'ına devret
        try:
            if getattr(self, "use_orchestrator", False) and getattr(self, "orchestrator", None) and signals:
                # 1) price_map çıkar (Binance fiyat haritası)
                price_map: Dict[str, float] = {}
                try:
                    for s in signals:
                        sym = s.get("symbol")
                        last_price = s.get("last_price") or s.get("price")
                        if sym and last_price:
                            price_map[sym] = float(last_price)
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"[ORCH] price_map oluşturma hatası: {e}")

                ctx: Dict[str, Any] = {
                    "timestamp": _dtmod.datetime.utcnow(),
                    "price_map": price_map,
                }

                # İstersen StrategyManager içindeki mevcut rejim bilgisini de ekle
                try:
                    from strategies.strategy_manager import strategy_manager
                    ctx["regime_name"] = getattr(strategy_manager, "_regime_name", "unknown")
                except Exception:
                    pass

                # 2) Orchestrator'a ver
                results = self.orchestrator.process_signals(signals, ctx)

                # 3) Log
                self.logger.info(
                    LogCategory.GUI,
                    f"[ORCH][BINANCE] {len(signals)} sinyal işlendi, {len(results)} intent yürütüldü",
                )

                # Orchestrator modunda legacy Auto-Execute bloğuna girmeden çık
                return

        except Exception as e:
            self.logger.error(LogCategory.GUI, f"[ORCH] Binance orchestrator akışı genel hata: {e}")

        # Güncelleme döngüsünü başlat (main thread üzerinden after ile)
        self._after_id = None
        # Oto yürütme: exec_signals varsa paper_executor'a ilet
        # Orchestrator modu aktifken legacy auto-execute kesinlikle devre dışı
        if not getattr(self, "use_orchestrator", False):
            try:
                if exec_signals:
                    paper_executor.execute(exec_signals)
                    self.post_ui(self.update_market_data)
                    self.post_ui(self.update_performance)
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Oto yürütme hatası: {e}")
        
        try:
            self.schedule_update_loop()
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Update loop schedule hatası: {e}")

    def _is_blocked_symbol(self, sym: str) -> bool:
        try:
            s = (sym or '').upper()
            # Fan tokenleri engelle: isimde 'FAN' geçen tüm pariteleri dışla
            if 'FAN' in s:
                return True
            return False
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Symbol değişikliği hatası: {e}")
            return False

    def _on_symbol_changed(self, event=None):
        try:
            sym = (self.paper_symbol_combo.get() or self._current_symbol or 'BTCUSDT').strip().upper()
            self._current_symbol = sym
            try:
                # Regime bilgisini al ve göster
                r = getattr(strategy_manager, '_regime_name', 'unknown')
                r_disp = {
                    'trend': 'Trend',
                    'yatay': 'Yatay',
                    'volatil': 'Volatil',
                    'çöküş': 'Çöküş',
                    # Rejim bilinmiyorsa varsayılanı 'Yatay' göster
                    'unknown': 'Yatay',
                    'none': 'Yatay',
                    '': 'Yatay'
                }.get(str(r).lower(), 'Yatay')
                self.market_status_label.config(text=f"Piyasa: {sym} ({r_disp})")
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Market status label güncelleme hatası: {e}")
            # Grafikte yeni sembol için yakın geçmişi sıfırla (temiz başlangıç)
            try:
                self._closes_cache = []
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Closes cache temizleme hatası: {e}")
            # Miktar/Toplamı tazele
            self._recalc_amount_total()
            # Grafik başlığını tazele
            try:
                self.update_price_chart()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Grafik güncelleme hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Sembol değişiminde hata: {e}")
        
    # _build_ui_async kaldırıldı (senkron kurulum kullanıyoruz)
    
    def setup_gui(self):
        """GUI'yi kur"""
        try:
            self.logger.info(LogCategory.GUI, "GUI kurulum BAŞLADI")
            # Ana pencere ayarları
            self.root.title("Binance Spot (Paper) - Dashboard")
            self.root.geometry("1400x900")
            self.root.minsize(1200, 800)
            
            # Menü çubuğu oluştur
            self.create_menu_bar()
            
            # Stil ayarları
            self.logger.info(LogCategory.GUI, "setup_styles çağrılıyor")
            self.setup_styles()
            
            # Ana frame
            self.logger.info(LogCategory.GUI, "main_frame oluşturuluyor")
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Üst panel - Durum ve kontroller
            self.logger.info(LogCategory.GUI, "create_top_panel çağrılıyor")
            self.create_top_panel(main_frame)
            
            # Orta panel - Ana içerik
            self.logger.info(LogCategory.GUI, "middle_frame oluşturuluyor")
            self.middle_frame = ttk.Frame(main_frame)
            # Orta panel çok büyümesin; alt loglar görünsün
            self.middle_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
            
            # Sol panel - Piyasa verileri ve pozisyonlar
            self.logger.info(LogCategory.GUI, "create_left_panel çağrılıyor")
            self.create_left_panel(self.middle_frame)
            
            # Sağ panel - Grafikler ve analiz
            self.logger.info(LogCategory.GUI, "create_right_panel çağrılıyor")
            self.create_right_panel(self.middle_frame)
            
            # Order book analizi paneli: güvenli modda yükleme
            self.logger.info(LogCategory.GUI, "create_orderbook_analysis_panel çağrılıyor")
            if not getattr(self, '_safe_mode', False):
                self.create_orderbook_analysis_panel(self.middle_frame)
                self._orderbook_panel_enabled = True
            else:
                self.ob_placeholder = ttk.LabelFrame(self.middle_frame, text="Order Book Analizi (devre dışı)")
                self.ob_placeholder.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 0))
                ttk.Label(self.ob_placeholder, text="Güvenli mod açık: Order book analizi devre dışı.").pack(padx=10, pady=10)
                self._orderbook_panel_enabled = False
            
            self.logger.info(LogCategory.GUI, "GUI kurulum TAMAMLANDI")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"GUI kurulum hatası: {e}")
            # Fallback minimal UI
            try:
                fallback = ttk.Frame(self.root)
                fallback.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                ttk.Label(fallback, text=f"GUI kurulum hatası: {e}", foreground='red').pack()
            except Exception as e:
                self.logger.error(LogCategory.GUI, f"Fallback GUI oluşturma hatası: {e}")

    def _start_ticker_snapshot_loop(self):
        """Tek GET /ticker ile tüm pariteleri çek, TRY paritelerini alfabetik listele ve fiyat cache'i güncelle."""
        try:
            if getattr(self, '_ticker_loop_running', False):
                return
            self._ticker_loop_running = True
            def _worker():
                while self._ticker_loop_running:
                    try:
                        # Exponential backoff + jitter (429/5xx/bağlantı hataları)
                        delay = 0.5
                        max_delay = 8.0
                        attempt = 0
                        while True:
                            try:
                                resp = self.multi_api_manager.make_request('GET', '/ticker', params={})
                                break
                            except Exception as req_ex:
                                attempt += 1
                                j = random.uniform(0.0, 0.15)
                                time.sleep(min(max_delay, delay) + j)
                                delay *= 2
                                if delay > max_delay:
                                    # Birkaç denemeden sonra yine de devam; dış try yakalayacak
                                    raise req_ex
                        data = resp.get('data', resp if isinstance(resp, list) else [])
                        if isinstance(data, list):
                            local_cache = {}
                            try_syms = []
                            usdt_syms = []
                            for it in data:
                                sym = (it.get('pairSymbol') or it.get('pair') or it.get('symbol') or '').upper().replace('-', '')
                                if not sym:
                                    continue
                                p = it.get('last') or it.get('price') or it.get('bid') or it.get('a')
                                try:
                                    px = float(p) if p is not None else 0.0
                                except (ValueError, TypeError):
                                    px = 0.0
                                if px > 0:
                                    local_cache[sym] = px
                                # 24s yüzdesi (varsa) yakala
                                try:
                                    pct24 = None
                                    for k in ('dailyChangePercent','dailyPercent','changePercent','percentChange','P','pct'):
                                        if k in it:
                                            pct24 = float(it.get(k) or 0)
                                            break
                                    if pct24 is None and 'dailyChange' in it and px:
                                        pct24 = float(it.get('dailyChange') or 0) / float(px) * 100.0
                                    if pct24 is not None:
                                        self._change24_cache[sym] = pct24
                                except Exception:
                                    pass
                                if sym.endswith('TRY') and not self._is_blocked_symbol(sym) and 'TEST' not in sym:
                                    try_syms.append(sym)
                                if sym.endswith('USDT') and not self._is_blocked_symbol(sym) and 'TEST' not in sym:
                                    usdt_syms.append(sym)
                            # Cache'i uygula
                            try:
                                self._price_cache.update(local_cache)
                            except Exception as e:
                                self.logger.warning(LogCategory.GUI, f"Fiyat cache güncelleme hatası: {e}")
                            # USDTTRY FX
                            fx = local_cache.get('USDTTRY') or 0.0
                            if fx:
                                self._usdt_try = fx
                            # Binance fallback: MultiAPI /ticker bazı USDT paritelerini döndürmüyorsa, evrende olup cache'te olmayanları Binance'tan tamamla
                            try:
                                merged = set(self._binance_usdt_universe or set(["BTCUSDT"]))
                                missing = [s for s in merged if s not in local_cache]
                                if missing:
                                    add_prices = self._fetch_binance_prices()
                                    if add_prices:
                                        local_cache.update({k: v for k, v in add_prices.items() if k in merged})
                            except Exception:
                                pass
                            # Seçili sembol son fiyatı (UI thread'e dokunmadan, cache üzerinden)
                            try:
                                sym = (self._current_symbol or 'BTCUSDT').strip().upper()
                                lp = local_cache.get(sym)
                                if lp and lp > 0:
                                    self._last_price = lp
                            except Exception as e:
                                self.logger.warning(LogCategory.GUI, f"Son fiyat güncelleme hatası: {e}")
                            # Sembol listelerini alfabetik güncelle (yalnız USDT)
                            usdt_syms = sorted([s for s in set(usdt_syms) if not self._is_blocked_symbol(s)])
                            # Binance evreni ile birleştir
                            merged = sorted(set(usdt_syms) | set(self._binance_usdt_universe or set(["BTCUSDT"])))
                            self._scan_symbols = merged
                            try:
                                vals = merged
                                if hasattr(self, 'paper_symbol_combo'):
                                    self.post_ui(self._update_symbol_combo_values, vals)
                            except Exception as e:
                                self.logger.warning(LogCategory.GUI, f"Sembol combobox güncelleme hatası: {e}")
                            # API durumunu güncelle (başarılı cevap alındı)
                            try:
                                self._last_api_ok = True
                                if hasattr(self, 'api_status_label'):
                                    self.post_ui(self.api_status_label.config, text="API: Bağlı", style='Good.TLabel')
                            except Exception as e:
                                self.logger.warning(LogCategory.GUI, f"API durum etiketi güncelleme hatası: {e}")
                            # UI tazele
                            try:
                                self.post_ui(self.update_market_data)
                                self.post_ui(self.update_price_chart)
                                self.post_ui(self.update_performance)
                            except Exception as e:
                                self.logger.warning(LogCategory.GUI, f"UI tazeleme hatası: {e}")
                    except Exception as ex:
                        # Snapshot başarısız: hata logla ve tekil fallback dene
                        try:
                            self.logger.error(LogCategory.GUI, f"Ticker snapshot hatası: {ex}")
                        except Exception as le:
                            print(f"[GUI] Ticker snapshot hata loglanamadı: {le}")
                        ok_any = False
                        try:
                            # USDTTRY kuru
                            fx_resp = self.multi_api_manager.make_request('GET', '/ticker', params={'pairSymbol': 'USDTTRY'})
                            # multi_api_manager / Binance katmanı genellikle {'data': [...]} döner,
                            # ancak bazı durumlarda doğrudan liste gelebilir; her iki durumu da güvenli işle.
                            if isinstance(fx_resp, dict):
                                fx_d = fx_resp.get('data', fx_resp)
                            else:
                                fx_d = fx_resp
                            fx_item = fx_d[0] if isinstance(fx_d, list) and fx_d else fx_d
                            fx_p = None
                            if isinstance(fx_item, dict):
                                fx_p = fx_item.get('last') or fx_item.get('price') or fx_item.get('bid') or fx_item.get('a')
                            if fx_p:
                                self._usdt_try = float(fx_p)
                                ok_any = True
                        except Exception as e:
                            self.logger.warning(LogCategory.GUI, f"USDTTRY fiyatı güncelleme hatası: {e}")
                        try:
                            sym = (self._current_symbol or 'BTCUSDT').strip().upper()
                            tkr = self.multi_api_manager.make_request('GET', '/ticker', params={'pairSymbol': sym})
                            # Yukarıdaki ile aynı: hem {'data': [...]} hem de doğrudan listeyi destekle
                            if isinstance(tkr, dict):
                                d = tkr.get('data', tkr)
                            else:
                                d = tkr
                            it = d[0] if isinstance(d, list) and d else d
                            p = None
                            if isinstance(it, dict):
                                p = it.get('last') or it.get('price') or it.get('bid') or it.get('a')
                            if p is not None:
                                px = float(p)
                                if px > 0:
                                    self._price_cache[sym] = px
                                    self._last_price = px
                                    ok_any = True
                        except Exception as e:
                            self.logger.warning(LogCategory.GUI, f"Sembol fiyatı güncelleme hatası: {e}")
                        # API durum etiketini fallback sonucuna göre ayarla
                        try:
                            self._last_api_ok = bool(ok_any)
                            if hasattr(self, 'api_status_label'):
                                if ok_any:
                                    self.post_ui(self.api_status_label.config, text="API: Bağlı (fallback)", style='Good.TLabel')
                                else:
                                    self.post_ui(self.api_status_label.config, text="API: Bağlantı Yok", style='Neutral.TLabel')
                        except Exception as e:
                            self.logger.warning(LogCategory.GUI, f"API durum etiketi güncelleme hatası: {e}")
                    time.sleep(5)
            threading.Thread(target=_worker, daemon=True).start()
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Ticker snapshot loop hatası: {e}")

    def _fetch_binance_prices(self) -> dict:
        """Binance tüm semboller için son fiyatları döndürür (dict)."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            arr = r.json() or []
            out = {}
            for it in arr:
                try:
                    sym = str(it.get('symbol') or '').upper()
                    px = float(it.get('price') or 0)
                    if sym and px > 0:
                        out[sym] = px
                except Exception:
                    pass
            return out
        except Exception as e:
            try:
                self.logger.warning(LogCategory.GUI, f"Binance fiyat fallback hatası: {e}")
            except Exception:
                pass
            return {}

    def _load_binance_usdt_universe(self):
        """Binance exchangeInfo üzerinden USDT paritelerini yükle ve combobox/tarama listesine uygula."""
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            info = r.json() or {}
            syms = set()
            for s in (info.get('symbols') or []):
                try:
                    if (s.get('status') == 'TRADING' and str(s.get('quoteAsset')).upper() == 'USDT'):
                        sym = str(s.get('symbol') or '').upper()
                        if sym and not self._is_blocked_symbol(sym) and 'TEST' not in sym:
                            syms.add(sym)
                except Exception:
                    pass
            if syms:
                self._binance_usdt_universe = syms
                vals = sorted(syms | set(["BTCUSDT"]))
                if hasattr(self, 'paper_symbol_combo'):
                    self.post_ui(self._update_symbol_combo_values, vals)
                try:
                    self.logger.info(LogCategory.GUI, f"USDT evreni yüklendi: {len(vals)} sembol")
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Binance USDT evreni yükleme hatası: {e}")

    def _refresh_universe_periodically(self):
        """USDT evrenini periyodik yenile (15 dakikada bir)."""
        while True:
            try:
                time.sleep(900)
                self._load_binance_usdt_universe()
            except Exception:
                time.sleep(900)

    def _update_orderbook_labels(self):
        try:
            if not hasattr(self, 'orderbook_analyzer') or self.orderbook_analyzer is None:
                return
            sym = (self._current_symbol or 'BTCUSDT').strip().upper()
            metrics = self.orderbook_analyzer.get_orderbook_metrics(sym)
            if not metrics:
                return
            try:
                volp = float(metrics.get('price_volatility', 0.0) or 0.0) * 100.0
            except Exception:
                volp = 0.0
            try:
                self.volatility_estimate_label.config(text=f"Volatilite: {volp:.2f}%")
            except Exception:
                pass
            try:
                self.support_resistance_label.config(text=f"Destek/Direnç Seviyeleri: 0")
            except Exception:
                pass
            try:
                self.market_impact_label.config(text=f"Piyasa Etkisi: {float(metrics.get('spread_volatility',0.0) or 0.0):.3f}")
            except Exception:
                pass
        except Exception:
            pass

    def _close_all_positions(self):
        try:
            if not messagebox.askyesno("Onay", "TÜM pozisyonlar kapatılsın mı?"):
                return
            closed_any = False
            for sym, pos in list((paper_executor.positions or {}).items()):
                try:
                    qty = float(pos.get('qty') or 0.0)
                    if qty <= 0:
                        continue
                    price = float(self._get_last_price_for_symbol(sym) or 0.0)
                    if price > 0:
                        paper_executor._close_all(sym, price, reason='manual_close_all')
                        closed_any = True
                except Exception:
                    pass
            self._refresh_after_close()
            if closed_any:
                messagebox.showinfo("Başarılı", "Tüm pozisyonlar kapatıldı")
            else:
                messagebox.showinfo("Bilgi", "Kapatılacak açık pozisyon bulunamadı")
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Tümünü kapat hatası: {e}")
            except Exception:
                pass
    
    def _close_all_profitable_positions(self):
        try:
            if not messagebox.askyesno("Onay", "Kârdaki TÜM pozisyonlar kapatılsın mı?"):
                return
            closed_any = False
            for sym, pos in list((paper_executor.positions or {}).items()):
                try:
                    qty = float(pos.get('qty') or 0.0)
                    avg = float(pos.get('avg_cost') or 0.0)
                    if qty <= 0 or avg <= 0:
                        continue
                    price = float(self._get_last_price_for_symbol(sym) or 0.0)
                    if price > avg and price > 0:
                        paper_executor._close_all(sym, price, reason='manual_close_winners')
                        closed_any = True
                except Exception:
                    pass
            self._refresh_after_close()
            if closed_any:
                messagebox.showinfo("Başarılı", "Kârdaki tüm pozisyonlar kapatıldı")
            else:
                messagebox.showinfo("Bilgi", "Kapatılacak kârdaki pozisyon bulunamadı")
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Kârdaki pozisyonları kapatma hatası: {e}")
            except Exception:
                pass
    
    def _close_all_losing_positions(self):
        try:
            if not messagebox.askyesno("Onay", "Zarardaki TÜM pozisyonlar kapatılsın mı?"):
                return
            closed_any = False
            for sym, pos in list((paper_executor.positions or {}).items()):
                try:
                    qty = float(pos.get('qty') or 0.0)
                    avg = float(pos.get('avg_cost') or 0.0)
                    if qty <= 0 or avg <= 0:
                        continue
                    price = float(self._get_last_price_for_symbol(sym) or 0.0)
                    if 0 < price < avg:
                        paper_executor._close_all(sym, price, reason='manual_close_losers')
                        closed_any = True
                except Exception:
                    pass
            self._refresh_after_close()
            if closed_any:
                messagebox.showinfo("Başarılı", "Zarardaki tüm pozisyonlar kapatıldı")
            else:
                messagebox.showinfo("Bilgi", "Kapatılacak zarardaki pozisyon bulunamadı")
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Zarardaki pozisyonları kapatma hatası: {e}")
            except Exception:
                pass
    
    def _close_selected_position(self):
        try:
            if not hasattr(self, 'positions_tree') or not self.positions_tree:
                messagebox.showinfo("Bilgi", "Pozisyon tablosu bulunamadı.")
                return
            sel = self.positions_tree.selection()
            if not sel:
                messagebox.showinfo("Bilgi", "Lütfen kapatmak için bir pozisyon seçin.")
                return
            item_id = sel[0]
            vals = self.positions_tree.item(item_id, 'values')
            if not vals:
                return
            sym = str(vals[0]).strip().upper()
            price = float(self._get_last_price_for_symbol(sym) or 0.0)
            if price <= 0:
                messagebox.showerror("Hata", f"{sym} için fiyat bulunamadı. Birkaç saniye sonra tekrar deneyin.")
                return
            if not messagebox.askyesno("Onay", f"{sym} pozisyonu tamamen kapatılsın mı?"):
                return
            try:
                paper_executor._close_all(sym, price, reason='manual_close_selected')
            except Exception:
                pos = (paper_executor.positions or {}).get(sym) or {}
                qty = float(pos.get('qty') or 0.0)
                if qty > 0:
                    paper_executor.execute([{'symbol': sym, 'side': 'sell', 'size': qty, 'entry_price': price, 'strategy_name': 'manual'}])
            self._refresh_after_close()
            messagebox.showinfo("Başarılı", f"{sym} pozisyonu kapatıldı")
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Seçili pozisyon kapatma hatası: {e}")
            except Exception:
                pass
    
    def _refresh_after_close(self):
        try:
            try:
                self.update_performance()
            except Exception:
                pass
            price_map = dict(self._price_cache)
            if self._usdt_try and self._usdt_try > 0:
                price_map['USDTTRY'] = self._usdt_try
            try:
                self._refresh_positions_table(price_map)
            except Exception:
                pass
            try:
                self._refresh_trades_table()
            except Exception:
                pass
        except Exception:
            pass
    
    def _update_symbol_combo_values(self, vals: list[str]):
        try:
            cur = self.paper_symbol_combo.get() if hasattr(self, 'paper_symbol_combo') else ''
            # Yalnızca USDT paritelerini göster (fan tokenlerini yine de çıkar)
            filtered: list[str] = []
            for v in vals:
                try:
                    sym = (v or '').strip().upper()
                    if not sym:
                        continue
                    if self._is_blocked_symbol(sym):
                        continue
                    if sym.endswith('USDT'):
                        filtered.append(sym)
                except Exception:
                    continue
            # Tekrarlı sembolleri kaldır ve sırala
            filtered = sorted(list(dict.fromkeys(filtered)))
            self.paper_symbol_combo['values'] = filtered
            if not cur or cur.upper() not in filtered or self._is_blocked_symbol(cur):
                self.paper_symbol_combo.set('BTCUSDT')
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Sembol combobox güncelleme hatası: {e}")

    def post_ui(self, func, *args, **kwargs):
        """Arka plan thread'lerinden güvenli UI çağrısı kuyruğa ekle."""
        try:
            if hasattr(self, '_ui_queue') and self._ui_queue is not None:
                self._ui_queue.put((func, args, kwargs))
        except Exception:
            pass

    def _drain_ui_queue(self):
        """Ana thread'de kuyruğu tüket ve UI çağrılarını yürüt."""
        try:
            while True:
                try:
                    func, args, kwargs = self._ui_queue.get_nowait()
                except Empty:
                    break
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"UI çağrısı hatası: {e}")
        except Exception:
            pass
        # Periyodik tekrar
        try:
            self.root.after(30, self._drain_ui_queue)
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"UI kuyruğu döngü hatası: {e}")

    def setup_styles(self):
        """Stil ayarları"""
        try:
            style = ttk.Style()
            
            # Tema ayarları
            style.theme_use('clam')
            
            # Renkler
            style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
            style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
            style.configure('Status.TLabel', font=('Arial', 11, 'bold'))
            style.configure('Positive.TLabel', foreground='green')
            style.configure('Negative.TLabel', foreground='red')
            style.configure('Neutral.TLabel', foreground='blue')
            
            # Buton stilleri
            style.configure('Start.TButton', background='green')
            style.configure('Stop.TButton', background='red')
            style.configure('Warning.TButton', background='orange')
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Stil ayarları hatası: {e}")
    
    def create_top_panel(self, parent):
        """Üst panel oluştur"""
        try:
            self.logger.info(LogCategory.GUI, "Top panel başlıyor")
            top_frame = ttk.Frame(parent)
            top_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Sol taraf - Durum bilgileri
            status_frame = ttk.Frame(top_frame)
            status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Bot durumu
            self.bot_status_label = ttk.Label(status_frame, text="Bot Durumu: Durduruldu", style='Header.TLabel')
            self.bot_status_label.pack(side=tk.LEFT, padx=(0, 20))
            
            # API durumu
            self.api_status_label = ttk.Label(status_frame, text="API: Bağlantı Yok", style='Status.TLabel')
            self.api_status_label.pack(side=tk.LEFT, padx=(0, 20))
            
            # Piyasa durumu (sembol + regime)
            try:
                r = getattr(strategy_manager, '_regime_name', 'unknown')
                r_disp = {
                    'trend': 'Trend',
                    'yatay': 'Yatay',
                    'volatil': 'Volatil',
                    'çöküş': 'Çöküş',
                    'unknown': 'Bilinmiyor'
                }.get(str(r).lower(), 'Bilinmiyor')
                sym = 'BTCUSDT'
                init_text = f"Piyasa: {sym} ({r_disp})"
            except Exception:
                init_text = "Piyasa: Bilinmiyor"
            self.market_status_label = ttk.Label(status_frame, text=init_text, style='Status.TLabel')
            self.market_status_label.pack(side=tk.LEFT, padx=(0, 20))
            # Kapı durumu göstergesi gizlendi
            self.gates_status_label = None
            
            # Sağ taraf - Kontrol butonları
            control_frame = ttk.Frame(top_frame)
            control_frame.pack(side=tk.RIGHT)
            
            # Start/Stop butonu
            self.start_stop_button = ttk.Button(control_frame, text="Başlat", command=self.toggle_bot)
            self.start_stop_button.pack(side=tk.LEFT, padx=(0, 10))

            # Inventory TWAP butonu (seçili sembol için envanter boşaltma)
            twap_button = ttk.Button(control_frame, text="Inventory TWAP", command=self.start_inventory_twap_for_current_symbol)
            twap_button.pack(side=tk.LEFT, padx=(0, 10))

            # Strateji Motoru anahtarı
            self._engine_var = tk.BooleanVar(value=False)
            engine_chk = ttk.Checkbutton(control_frame, text="Strateji Motoru", variable=self._engine_var, command=self.on_engine_toggle)
            engine_chk.pack(side=tk.LEFT, padx=(0, 10))
            # Oto İşlem anahtarı
            self._auto_var = tk.BooleanVar(value=False)
            auto_chk = ttk.Checkbutton(control_frame, text="Oto İşlem", variable=self._auto_var, command=self.on_auto_toggle)
            auto_chk.pack(side=tk.LEFT, padx=(0, 10))
            # Maks açık pozisyon combobox
            self.max_pos_var = tk.StringVar(value="20")
            self.max_pos_combo = ttk.Combobox(control_frame, textvariable=self.max_pos_var, values=["0","10","20","50","sınırsız"], width=8, state="readonly")
            self.max_pos_combo.pack(side=tk.LEFT, padx=(0, 10))
            self.max_pos_combo.bind("<<ComboboxSelected>>", self.on_max_pos_changed)
            
            
            # Ayarlar butonu
            settings_button = ttk.Button(control_frame, text="Ayarlar", command=self.open_settings)
            settings_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # API yönetimi butonu
            api_button = ttk.Button(control_frame, text="API Yönetimi", command=self.open_api_management)
            api_button.pack(side=tk.LEFT)
            # Backtest butonu
            try:
                bt_button = ttk.Button(control_frame, text="Backtest", command=self.open_backtest)
                bt_button.pack(side=tk.LEFT, padx=(10, 0))
            except Exception:
                pass
            if getattr(self, '_basic_mode', False):
                try:
                    basic_ticker_btn = ttk.Button(control_frame, text="Ticker", command=self.start_ticker_loop)
                    basic_ticker_btn.pack(side=tk.LEFT, padx=(10, 0))
                except Exception:
                    pass
                try:
                    basic_update_btn = ttk.Button(control_frame, text="Update", command=self.start_update_loop)
                    basic_update_btn.pack(side=tk.LEFT, padx=(5, 0))
                except Exception:
                    pass
                if getattr(self, '_init_callback', None):
                    try:
                        basic_init_btn = ttk.Button(control_frame, text="Init", command=self._init_callback)
                        basic_init_btn.pack(side=tk.LEFT, padx=(5, 0))
                    except Exception:
                        pass
                # Step-by-step init buttons if provided
                try:
                    if callable(self._init_steps.get('setup')):
                        btn = ttk.Button(control_frame, text="Init-Setup", command=self._init_steps['setup'])
                        btn.pack(side=tk.LEFT, padx=(10, 0))
                except Exception:
                    pass
                try:
                    if callable(self._init_steps.get('modules')):
                        btn = ttk.Button(control_frame, text="Init-Modules", command=self._init_steps['modules'])
                        btn.pack(side=tk.LEFT, padx=(5, 0))
                except Exception:
                    pass
                try:
                    if callable(self._init_steps.get('services')):
                        btn = ttk.Button(control_frame, text="Init-Services", command=self._init_steps['services'])
                        btn.pack(side=tk.LEFT, padx=(5, 0))
                except Exception:
                    pass
                try:
                    if callable(self._init_steps.get('plugins')):
                        btn = ttk.Button(control_frame, text="Init-Plugins", command=self._init_steps['plugins'])
                        btn.pack(side=tk.LEFT, padx=(5, 0))
                except Exception:
                    pass
            
            self.logger.info(LogCategory.GUI, "Top panel bitti")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Üst panel oluşturma hatası: {e}")

    def start_ticker_loop(self):
        try:
            self._start_ticker_snapshot_loop()
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Ticker döngüsü başlatma hatası: {e}")

    def start_update_loop(self):
        try:
            self.schedule_update_loop()
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Update döngüsü başlatma hatası: {e}")

    def open_settings(self):
        """Ayarlar penceresini aç"""
        try:
            StrategySettingsGUI(self.root, strategy_manager)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Ayarlar penceresi hatası: {e}")

    def open_api_management(self):
        """API yönetim penceresini aç"""
        try:
            from gui.api_management_gui import APIManagementGUI
            APIManagementGUI(self.root)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"API yönetimi penceresi hatası: {e}")

    def open_backtest(self):
        """Backtest penceresini aç"""
        try:
            from gui.backtest_gui import BacktestGUI
            BacktestGUI(self.root)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Backtest penceresi hatası: {e}")

    def add_log_message(self, message: str):
        try:
            txt = getattr(self, 'log_text', None)
            if txt and hasattr(txt, 'insert'):
                ts = _dtmod.datetime.now().strftime('%H:%M:%S')
                try:
                    txt.insert('end', f"[{ts}] {message}\n")
                    txt.see('end')
                except Exception:
                    pass
            else:
                try:
                    self.logger.info(LogCategory.GUI, f"{message}")
                except Exception:
                    print(f"[LOG] {message}")
        except Exception:
            pass

    def _discover_try_symbols(self):
        try:
            # Basit ve güvenli: sabit bir liste ile combobox değerlerini güncelle
            symbols = ['BTCTRY', 'ETHTRY', 'AVAXTRY', 'ADATRY', 'SOLTRY']
            if hasattr(self, 'paper_symbol_combo') and self.paper_symbol_combo:
                try:
                    self.root.after(0, lambda v=symbols: self._update_symbol_combo_values(v))
                except Exception:
                    pass
        except Exception as e:
            try:
                self.logger.warning(LogCategory.GUI, f"TRY sembolleri keşfetme hatası: {e}")
            except Exception:
                pass

    def start_inventory_twap_for_current_symbol(self):
        """Seçili sembol için Inventory TWAP envanter boşaltma akışını başlat.

        Mevcut paper pozisyonundaki qty'yi alır ve PIEZO-TWAP mantığıyla slice slice satmaya başlar.
        """
        try:
            sym = (self._current_symbol or 'BTCUSDT').strip().upper()
            pos = (paper_executor.positions or {}).get(sym) or {}
            qty = float(pos.get('qty') or 0.0)
            if qty <= 0:
                messagebox.showinfo("Inventory TWAP", f"{sym} için açık long pozisyon bulunamadı.")
                return
            if not messagebox.askyesno("Inventory TWAP", f"{sym} için yaklaşık {qty:.6f} envanteri slice slice satmak istiyor musunuz?"):
                return
            # Daha önce bu sembol için bir TWAP thread'i varsa tekrar başlatma
            try:
                t_prev = self._twap_threads.get(sym)
                if t_prev and t_prev.is_alive():
                    messagebox.showinfo("Inventory TWAP", f"{sym} için zaten aktif bir Inventory TWAP süreci var.")
                    return
            except Exception:
                pass
            # Başlangıç state'i başlat
            try:
                self._twap_liquidator.start(sym, qty)
            except Exception:
                pass
            # Arka planda döngüyü başlat
            import threading as _th
            t = _th.Thread(target=self._run_inventory_twap_loop, args=(sym,), daemon=True)
            self._twap_threads[sym] = t
            t.start()
            try:
                self.add_log_message(f"Inventory TWAP başlatıldı: {sym} (qty={qty:.6f})")
            except Exception:
                pass
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Inventory TWAP başlatma hatası: {e}")
            except Exception:
                pass

    def _run_inventory_twap_loop(self, symbol: str):
        """Belirli bir sembol için Inventory TWAP döngüsünü çalıştır.

        Her ~5 saniyede bir InventoryTwapLiquidator'dan slice SELL sinyali üretir
        ve paper_executor.execute() ile kağıt hesapta uygular.
        """
        try:
            sym = (symbol or 'BTCUSDT').strip().upper()
            import time as _time
            while True:
                try:
                    pos = (paper_executor.positions or {}).get(sym) or {}
                    qty = float(pos.get('qty') or 0.0)
                    if qty <= 0:
                        break
                    # Son fiyat ve hacim tahmini
                    price = float(self._get_last_price_for_symbol(sym) or 0.0)
                    if price <= 0:
                        _time.sleep(5.0)
                        continue
                    # 1 dakikalık hacim ve ATR tahmini: basit fallback
                    last_1m_volume = qty  # gerçek veriye bağlanana kadar envanter büyüklüğü
                    atr_1m_pct = None
                    vwap_1m = None
                    slippage_bp_recent = None
                    best_bid = price
                    best_ask = price
                    mid_price = price
                    sig = self._twap_liquidator.generate_slice_signal(
                        symbol=sym,
                        inventory_qty=qty,
                        last_1m_volume=last_1m_volume,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mid_price=mid_price,
                        atr_1m_pct=atr_1m_pct,
                        vwap_1m=vwap_1m,
                        slippage_bp_recent=slippage_bp_recent,
                    )
                    if sig:
                        try:
                            paper_executor.execute([sig])
                        except Exception as ex:
                            try:
                                self.logger.warning(LogCategory.GUI, f"Inventory TWAP yürütme hatası: {ex}")
                            except Exception:
                                pass
                        try:
                            self.post_ui(self.update_market_data)
                            self.post_ui(self.update_performance)
                        except Exception:
                            pass
                    # TWAP state bitti mi kontrol et
                    st = None
                    try:
                        st = self._twap_liquidator.get_state(sym)
                    except Exception:
                        st = None
                    if st is not None and getattr(st, 'finished', False):
                        break
                except Exception:
                    pass
                _time.sleep(5.0)
            try:
                self.add_log_message(f"Inventory TWAP tamamlandı: {sym}")
            except Exception:
                pass
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Inventory TWAP döngü hatası: {e}")
            except Exception:
                pass

    def toggle_bot(self):
        """Botu başlat/durdur"""
        try:
            self.is_running = not self.is_running
            try:
                self.bot_status_label.config(text=f"Bot Durumu: {'Çalışıyor' if self.is_running else 'Durduruldu'}")
                self.start_stop_button.config(text=("Durdur" if self.is_running else "Başlat"))
            except Exception:
                pass
            if self.is_running:
                try:
                    strategy_manager.start_all_strategies()
                except Exception:
                    pass
            else:
                try:
                    strategy_manager.stop_all_strategies()
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Bot başlat/durdur hatası: {e}")

    def on_engine_toggle(self):
        """Strateji motorunu aç/kapat (safe_mode terslenir)."""
        try:
            self._engine_enabled = bool(self._engine_var.get())
            self._safe_mode = not self._engine_enabled
            state = "AÇIK" if self._engine_enabled else "KAPALI"
            try:
                self.add_log_message(f"Strateji Motoru: {state}")
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Log mesajı ekleme hatası: {e}")
            # Açıldıysa hemen ilk tick için zamanlayıcıları sıfırla
            if self._engine_enabled:
                self._last_strategy_time = 0.0
                self._last_strategy_log_time = 0.0
                try:
                    self.add_log_message("Strateji ilk tick planlandı (hemen)")
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"Log mesajı ekleme hatası: {e}")
                # TRY sembolleri keşfet
                try:
                    threading.Thread(target=self._discover_try_symbols, daemon=True).start()
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"TRY sembolleri keşfetme hatası: {e}")
                # Gelişmiş panelleri dinamik olarak etkinleştir
                try:
                    self._enable_dynamic_panels()
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"Gelişmiş panel etkinleştirme hatası: {e}")
            else:
                # Motor kapandı: tüm stratejileri durdur ve manuel moda geç
                try:
                    strategy_manager.stop_all_strategies()
                except Exception:
                    pass
            # Etiketleri güncelle (motor kapansa da seçili strateji adı görünür olsun)
            try:
                self._update_strategy_status()
            except Exception:
                pass
            # Kapı durum döngüsünü başlat/güncelle
            try:
                if self._engine_enabled:
                    self._start_gates_status_loop()
                else:
                    if hasattr(self, '_gates_after_id') and self._gates_after_id:
                        try:
                            self.root.after_cancel(self._gates_after_id)
                        except Exception:
                            pass
                        self._gates_after_id = None
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji anahtarı hatası: {e}")

    def _enable_dynamic_panels(self):
        """Safe mode kapandıktan sonra ağır panelleri dinamik olarak yükle."""
        try:
            # Fiyat grafiği devre dışı bırakıldı
            try:
                if hasattr(self, 'chart_placeholder') and self.chart_placeholder:
                    self.chart_placeholder.destroy()
                    self.chart_placeholder = None
            except Exception:
                pass
            self._price_chart_enabled = False
            # Order book analizi
            if not getattr(self, '_orderbook_panel_enabled', False):
                try:
                    if hasattr(self, 'ob_placeholder') and self.ob_placeholder:
                        self.ob_placeholder.destroy()
                        self.ob_placeholder = None
                except Exception:
                    pass
                try:
                    self.create_orderbook_analysis_panel(self.middle_frame)
                    self._orderbook_panel_enabled = True
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"Order book panel etkinleştirme hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Dinamik panel hatası: {e}")

    # --- Kapı durumları (Golf/Chop/Whitelist) ---
    def _start_gates_status_loop(self):
        # Kapı durumu güncellemesi devre dışı
        return

    def _update_gates_status(self):
        try:
            sym = (self._current_symbol or 'BTCUSDT').strip().upper()
            latest = None
            try:
                latest = self.market_analyzer.get_latest_analysis(sym)
            except Exception:
                latest = None
            golf_ok, golf_info = self._eval_golf_filter(latest)
            chop_on = self._is_chop_regime(latest)
            w_ok = self._is_whitelist_allowed(latest)
            # Göstergesi gizli; UI güncellemesi yapılmıyor
            pass
        except Exception:
            pass

    def _eval_golf_filter(self, latest: dict | None) -> tuple[bool, str]:
        try:
            ms = (latest or {}).get('microstructure', {}) if isinstance(latest, dict) else {}
            spread = float(ms.get('spread_pct', 0.0) or 0.0)
            ob_imb = float(ms.get('orderbook_imbalance_5lvl', 0.0) or 0.0)
            vol_z = float(ms.get('volume_zscore_30s', 0.0) or 0.0)
            tur = float(ms.get('ticks_up_ratio_30s', 0.0) or 0.0)
            ok = (spread <= 0.0008) and (ob_imb >= 0.58) and ((vol_z >= 2.0) or (tur >= 0.62))
            info = f"sp={spread*100:.2f}%, ob={ob_imb:.2f}, vz={vol_z:.2f}, tur={tur:.2f}"
            return ok, info
        except Exception:
            return False, ""

    def _is_chop_regime(self, latest: dict | None) -> bool:
        try:
            tech = (latest or {}).get('technical_analysis', {}) if isinstance(latest, dict) else {}
            adx = float(tech.get('adx', 0.0) or 0.0)
            ma20 = float(tech.get('ma_20', tech.get('ma20', 0.0)) or 0.0)
            ma50 = float(tech.get('ma_50', tech.get('ma50', 0.0)) or 0.0)
            atr = float(tech.get('atr', 0.0) or 0.0)
            return (adx < 18.0 and atr > 0.0 and abs(ma20 - ma50) < 0.1 * atr)
        except Exception:
            return False

    def _load_hour_regime_whitelist(self) -> dict:
        try:
            import json, os
            root = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(root, 'data', 'hour_regime_whitelist.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _is_whitelist_allowed(self, latest: dict | None) -> bool:
        try:
            wl = getattr(self, '_hour_regime_whitelist_cache', None)
            if wl is None:
                wl = self._load_hour_regime_whitelist()
                self._hour_regime_whitelist_cache = wl
            if not wl:
                return True
            from datetime import datetime as _dt
            hour = str(_dt.now().hour)
            ai = (latest or {}).get('ai_analysis', {}) if isinstance(latest, dict) else {}
            regime = str(ai.get('regime') or '').lower()
            row = wl.get(hour)
            if isinstance(row, dict):
                cell = row.get(regime)
                if isinstance(cell, str) and cell.upper() == 'OFF':
                    return False
                if isinstance(cell, bool) and cell is False:
                    return False
            return True
        except Exception:
            return True

    def on_auto_toggle(self):
        try:
            # Legacy auto trade tamamen devre dışı (izleyici mod)
            self._auto_var.set(False)
            self.auto_enabled = False
            try:
                self.add_log_message("Oto İşlem legacy modu devre dışı (izleyici mod)")
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Oto işlem anahtarı hatası: {e}")

    def on_max_pos_changed(self, event=None):
        try:
            raw = (self.max_pos_var.get() or "").strip().lower()
            if raw == "sınırsız" or raw == "sinirsiz":
                self.max_open_positions = None  # sınırsız
            else:
                v = int(raw)
                self.max_open_positions = v
            try:
                self.add_log_message(f"Maks açık pozisyon: {self.max_open_positions}")
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Maks açık pozisyon değişim hatası: {e}")

    def _auto_worker(self):
        """Legacy auto trade worker tamamen devre dışı (izleyici mod)."""
        try:
            self.logger.info(LogCategory.GUI, "Auto worker çağrıldı ancak izleyici modda devre dışı.")
        except Exception:
            pass
        return

    def _refresh_positions_table(self, price_map: dict):
        """paper_executor.positions -> Açık Pozisyonlar tablosunu doldur."""
        try:
            if not hasattr(self, 'positions_tree') or self.positions_tree is None:
                return
            # Temizle
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            # Doldur
            for sym, pos in (paper_executor.positions or {}).items():
                qty = float(pos.get('qty') or 0.0)
                if qty <= 0:
                    continue
                avg = float(pos.get('avg_cost') or 0.0)
                cur = float(price_map.get(sym, 0.0) or 0.0)
                # USDT dönüşüm (TRY çiftleri için)
                fx = float(self._usdt_try or 0.0)
                def _fmt_usdt(val: float) -> str:
                    try:
                        s = f"{float(val or 0.0):,.4f}"
                        # 100,000.0000 -> 100.000,0000
                        s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
                        return f"{s} USDT"
                    except Exception:
                        try:
                            s = f"{float(val or 0.0):,.4f}"
                            s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
                            return f"{s} USDT"
                        except Exception:
                            return f"{val} USDT"
                def _fmt_price_for_symbol(s: str, price: float) -> str:
                    try:
                        p = float(price or 0.0)
                        # Heuristik basamak sayısı
                        if p == 0:
                            dec = 4
                        elif p < 1:
                            dec = 6
                        elif p < 10000:
                            dec = 4
                        else:
                            dec = 2
                        fmt = f"{{:,.{dec}f}}"
                        sval = fmt.format(p)
                        sval = sval.replace(',', 'X').replace('.', ',').replace('X', '.')
                        return sval
                    except Exception:
                        try:
                            return f"{float(price):,.4f}"
                        except Exception:
                            return str(price)
                # PnL (USDT ve %)
                pnl_coin = (cur - avg) * qty if (cur > 0 and avg > 0) else 0.0
                # USDT değeri: USDT çifti direkt, TRY çifti fx ile bölünerek
                def _to_usdt(symbol: str, amount_coin: float, price_now: float) -> float:
                    try:
                        if symbol.endswith('USDT'):
                            return amount_coin * price_now
                        if symbol.endswith('TRY'):
                            return (amount_coin * price_now) / (fx if fx > 0 else 1.0)
                        return amount_coin * price_now  # default
                    except Exception:
                        return 0.0
                cur_value_usdt = _to_usdt(sym, qty, cur)
                pnl_usdt = _to_usdt(sym, qty, (cur - avg))
                pct = ((cur - avg) / avg * 100.0) if (avg > 0) else 0.0
                tag = 'pnl_pos' if pnl_usdt >= 0 else 'pnl_neg'
                # Kolonlar USDT: (Sembol, Giriş Fiyatı USDT, Miktar, Mevcut Fiyat USDT, Pozisyon Değeri USDT, P&L USDT(%) , Durum)
                self.positions_tree.insert('', 'end', values=(
                    sym,
                    _fmt_usdt(_to_usdt(sym, 1.0, avg)),
                    f"{qty:.6f}",
                    _fmt_price_for_symbol(sym, cur),
                    _fmt_usdt(cur_value_usdt),
                    f"{_fmt_usdt(pnl_usdt)} ({pct:+.2f}%)",
                    (self._last_ai_signal.get(sym, 'NÖTR'))
                ), tags=(tag,))
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Pozisyon tablosu oluşturma hatası: {e}")

    def _refresh_trades_table(self):
        """paper_executor.trades -> Son İşlemler tablosunu doldur (son 10)."""
        try:
            if not hasattr(self, 'trades_tree') or self.trades_tree is None:
                return
            # Temizle
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)
            # Son 10 işlem
            last = list(paper_executor.trades or [])[-10:]
            for t in last:
                ts = t.get('entry_time')
                try:
                    ts_str = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                except Exception:
                    ts_str = '-'
                self.trades_tree.insert('', 'end', values=(
                    ts_str,
                    t.get('symbol'),
                    (t.get('side') or '').upper(),
                    f"{float(t.get('size') or 0.0):.6f}",
                    f"{float(t.get('entry_price') or 0.0):,.4f}",
                    f"{float(t.get('net_pnl') or 0.0):,.2f}"
                ))
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"İşlem tablosu oluşturma hatası: {e}")

    def create_left_panel(self, parent):
        """Sol panel oluştur"""
        try:
            self.logger.info(LogCategory.GUI, "Sol panel başlıyor")
            left_frame = ttk.Frame(parent)
            # Sol panel aşırı büyümesin; alt loglar görünsün
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
            
            # Piyasa verileri
            self.create_market_data_section(left_frame)
            
            # Pozisyonlar
            self.create_positions_section(left_frame)
            
            # Son işlemler
            self.create_trades_section(left_frame)
            
            # Sistem logları (yukarı taşındı)
            self.create_logs_section(left_frame)
            
            self.logger.info(LogCategory.GUI, "Sol panel bitti")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Sol panel oluşturma hatası: {e}")

    def create_logs_section(self, parent):
        """Sistem Logları bölümü (sol kolonda)"""
        try:
            logs_frame = ttk.LabelFrame(parent, text="Sistem Logları")
            logs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            # Metin alanı
            self.log_text = scrolledtext.ScrolledText(logs_frame, height=5, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            # Temizle butonu
            clear_btn = ttk.Button(logs_frame, text="Logları Temizle", command=self.clear_logs)
            clear_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Log bölümü oluşturma hatası: {e}")

    def clear_logs(self):
        """Log metin alanını temizle"""
        try:
            if hasattr(self, 'log_text') and self.log_text:
                self.log_text.delete('1.0', tk.END)
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Log temizleme hatası: {e}")
    
    def create_right_panel(self, parent):
        """Sağ panel oluştur"""
        try:
            self.logger.info(LogCategory.GUI, "Sağ panel başlıyor")
            self.right_frame = ttk.Frame(parent)
            # Sağ panel de fazla büyümesin; alt loglara yer kalsın
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
            
            # Fiyat grafiği: devre dışı
            try:
                self._price_chart_enabled = False
            except Exception:
                pass
                self._price_chart_enabled = False
            
            # Acil müdahale araç çubuğu
            self.create_emergency_toolbar(self.right_frame)
            
            # Performans metrikleri
            self.create_performance_section(self.right_frame)
            
            # Risk metrikleri
            self.create_risk_section(self.right_frame)
            
            # AI analiz bölümü gizlendi
            self.logger.info(LogCategory.GUI, "Sağ panel bitti")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Sağ panel oluşturma hatası: {e}")
    
    def create_emergency_toolbar(self, parent):
        try:
            bar = ttk.Frame(parent)
            bar.pack(fill=tk.X, pady=(0, 8))
            ttk.Label(bar, text="Acil Müdahale:").pack(side=tk.LEFT, padx=(5, 8))
            try:
                style = ttk.Style()
                style.configure('Emergency.TButton', padding=(8, 4))
                style.map('Emergency.TButton', background=[('active', '#ffb74d')])
            except Exception:
                pass
            ttk.Button(bar, text="Tümünü Kapat", style='Emergency.TButton', command=self._close_all_positions).pack(side=tk.LEFT, padx=(0,6))
            ttk.Button(bar, text="Kârdakileri Kapat", style='Emergency.TButton', command=self._close_all_profitable_positions).pack(side=tk.LEFT, padx=(0,6))
            ttk.Button(bar, text="Zarardakileri Kapat", style='Emergency.TButton', command=self._close_all_losing_positions).pack(side=tk.LEFT, padx=(0,6))
            ttk.Button(bar, text="Seçili Pozisyonu Kapat", style='Emergency.TButton', command=self._close_selected_position).pack(side=tk.LEFT, padx=(0,6))
        except Exception as e:
            try:
                self.logger.warning(LogCategory.GUI, f"Acil müdahale barı hata: {e}")
            except Exception:
                pass
    
    def create_bottom_panel(self, parent):
        """Alt panel oluştur"""
        try:
            self.logger.info(LogCategory.GUI, "Alt panel başlıyor")
            bottom_frame = ttk.Frame(parent)
            # Alt paneli alta sabitle, genişlemeyi kapat
            bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(10, 0))
            
            # Log görüntüleyici
            log_frame = ttk.LabelFrame(bottom_frame, text="Sistem Logları")
            log_frame.pack(fill=tk.X, expand=False)
            # Minimum yükseklik garantisi
            try:
                log_frame.configure(height=220)
                log_frame.pack_propagate(False)
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Log frame yükseklik ayarlaması hatası: {e}")
            
            self.log_text = scrolledtext.ScrolledText(log_frame, height=11, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Log temizleme butonu
            clear_log_button = ttk.Button(log_frame, text="Logları Temizle", command=self.clear_logs)
            clear_log_button.pack(side=tk.RIGHT, padx=5, pady=5)
            
            self.logger.info(LogCategory.GUI, "Alt panel bitti")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Alt panel oluşturma hatası: {e}")
    
    def _recalc_amount_total(self, calc_from: str = 'amount'):
        """Miktar ile Toplam (USDT) alanlarını seçili sembol ve fiyata göre senkron tut."""
        try:
            sym = (self.paper_symbol_combo.get() or 'BTCUSDT').strip().upper()
            price = float(self._get_last_price_for_symbol(sym) or 0.0)
            if price <= 0:
                return
            if calc_from == 'amount':
                txt = (self.paper_size_entry.get() or '').strip().replace(',', '.')
                amt = float(txt) if txt else 0.0
                # USDT pariteleri için toplam USDT = miktar * fiyat
                # TRY pariteleri için toplam USDT = (miktar * fiyat) / USDTTRY
                total = 0.0
                if sym.endswith('USDT'):
                    total = amt * price
                elif sym.endswith('TRY'):
                    fx = float(self._usdt_try or 0.0)
                    total = (amt * price) / fx if fx > 0 else 0.0
                self.paper_total_entry.delete(0, tk.END)
                # 4 ondalık ve Türkçe biçim: X,XXXX
                try:
                    s = f"{total:,.4f}"
                    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
                except Exception:
                    s = f"{total:.4f}"
                self.paper_total_entry.insert(0, s)
            else:
                txt = (self.paper_total_entry.get() or '').strip().replace(',', '.')
                ttl = float(txt) if txt else 0.0
                amt = 0.0
                if sym.endswith('USDT'):
                    denom = price
                    amt = ttl / denom if denom > 0 else 0.0
                elif sym.endswith('TRY'):
                    fx = float(self._usdt_try or 0.0)
                    denom = price / fx if fx > 0 else 0.0
                    amt = ttl / denom if denom > 0 else 0.0
                self.paper_size_entry.delete(0, tk.END)
                self.paper_size_entry.insert(0, f"{amt:.6f}")
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Miktar/Toplam hesaplama hatası: {e}")
    
    def create_market_data_section(self, parent):
        """Piyasa verileri bölümü"""
        try:
            market_frame = ttk.LabelFrame(parent, text="Piyasa Verileri")
            market_frame.pack(fill=tk.X, pady=(0, 10))
            
            # BTC fiyatı ve manuel işlem
            price_frame = ttk.Frame(market_frame)
            price_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(price_frame, text="BTC/USDT:", style='Header.TLabel').pack(side=tk.LEFT)
            self.btc_price_label = ttk.Label(price_frame, text="0.00", style='Header.TLabel')
            self.btc_price_label.pack(side=tk.LEFT, padx=(10, 0))
            
            self.btc_change_label = ttk.Label(price_frame, text="+0.00%", style='Neutral.TLabel')
            self.btc_change_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # 24h değişim ve hacim KALDIRILDI (isteğe göre)

            # Paper manuel işlem kontrolleri
            trade_ctrl = ttk.Frame(market_frame)
            trade_ctrl.pack(fill=tk.X, padx=5, pady=5)
            # Sembol seçimi
            ttk.Label(trade_ctrl, text="Sembol:").pack(side=tk.LEFT)
            self.paper_symbol_combo = ttk.Combobox(trade_ctrl, width=12, state='readonly')
            try:
                self.paper_symbol_combo['values'] = ["BTCUSDT"]
                self.paper_symbol_combo.set("BTCUSDT")
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Sembol combobox ayarlaması hatası: {e}")
            self.paper_symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
            # Miktar
            ttk.Label(trade_ctrl, text="Miktar:").pack(side=tk.LEFT)
            self.paper_size_entry = ttk.Entry(trade_ctrl, width=10)
            self.paper_size_entry.pack(side=tk.LEFT, padx=(5, 10))
            self.paper_size_entry.insert(0, "0.01")
            # Toplam (USDT)
            ttk.Label(trade_ctrl, text="Toplam (USDT):").pack(side=tk.LEFT)
            self.paper_total_entry = ttk.Entry(trade_ctrl, width=12)
            self.paper_total_entry.pack(side=tk.LEFT, padx=(5, 10))
            buy_btn = ttk.Button(trade_ctrl, text="Market Al (paper)", command=self.on_paper_buy)
            buy_btn.pack(side=tk.LEFT, padx=(0, 5))
            sell_btn = ttk.Button(trade_ctrl, text="Market Sat (paper)", command=self.on_paper_sell)
            sell_btn.pack(side=tk.LEFT)
            # Değer değişimlerinde miktar/toplamı otomatik hesapla
            try:
                self.paper_symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_changed)
                self.paper_size_entry.bind("<KeyRelease>", lambda e: self._recalc_amount_total(calc_from='amount'))
                self.paper_total_entry.bind("<KeyRelease>", lambda e: self._recalc_amount_total(calc_from='total'))
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Miktar/Toplam hesaplama bağlama hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Piyasa verileri bölümü oluşturma hatası: {e}")

    def on_paper_buy(self):
        """Paper market al"""
        try:
            sym = (self.paper_symbol_combo.get() or 'BTCUSDT').strip().upper()
            price = float(self._get_last_price_for_symbol(sym) or 0.0)
            txt = (self.paper_size_entry.get() or '').strip().replace(',', '.')
            qty = float(txt) if txt else 0.0
            if price > 0 and qty > 0:
                # Risk yöneticisi trading'i kapattıysa manuel işlem de açma
                try:
                    from risk_management.risk_manager import risk_manager as _rm_inst  # type: ignore
                    rm = _rm_inst.get_risk_metrics() if _rm_inst is not None else {}
                    if not rm.get("trading_enabled", True):
                        messagebox.showwarning("Risk Limiti", "Trading şu anda risk yöneticisi tarafından devre dışı.")
                        return
                except Exception:
                    pass
                # Gerçek paper işlemi: yürütücüye sinyal gönder
                try:
                    paper_executor.execute([{ 'symbol': sym, 'side': 'buy', 'size': qty, 'entry_price': price, 'strategy_name': 'manual', 'source': 'manual_gui' }])
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Manual BUY yürütme hatası: {e}")
                # Log ve UI tazele
                try:
                    self.add_log_message(f"[PAPER] AL {sym} qty={qty} @ {price:,.4f}")
                except Exception:
                    pass
                try:
                    self.update_market_data(); self.update_performance(); self.update_price_chart()
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Paper buy hatası: {e}")

    def on_paper_sell(self):
        """Paper market sat"""
        try:
            sym = (self.paper_symbol_combo.get() or 'BTCUSDT').strip().upper()
            price = float(self._get_last_price_for_symbol(sym) or 0.0)
            txt = (self.paper_size_entry.get() or '').strip().replace(',', '.')
            qty = float(txt) if txt else 0.0
            if price > 0 and qty > 0:
                # Risk yöneticisi trading'i kapattıysa manuel işlem de açma
                try:
                    from risk_management.risk_manager import risk_manager as _rm_inst  # type: ignore
                    rm = _rm_inst.get_risk_metrics() if _rm_inst is not None else {}
                    if not rm.get("trading_enabled", True):
                        messagebox.showwarning("Risk Limiti", "Trading şu anda risk yöneticisi tarafından devre dışı.")
                        return
                except Exception:
                    pass
                # Gerçek paper işlemi: yürütücüye sinyal gönder
                try:
                    paper_executor.execute([{ 'symbol': sym, 'side': 'sell', 'size': qty, 'entry_price': price, 'strategy_name': 'manual', 'source': 'manual_gui' }])
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Manual SELL yürütme hatası: {e}")
                # Log ve UI tazele
                try:
                    self.add_log_message(f"[PAPER] SAT {sym} qty={qty} @ {price:,.4f}")
                except Exception:
                    pass
                try:
                    self.update_market_data(); self.update_performance(); self.update_price_chart()
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Paper sell hatası: {e}")

    def _fetch_1m_klines(self, symbol: str) -> list:
        """1 dakikalık mumları çek (basit ve güvenli)"""
        try:
            resp = self.multi_api_manager.make_request('GET', '/klines', params={'pairSymbol': symbol, 'interval': '1', 'limit': 100})
            raw = resp.get('data', resp) if isinstance(resp, dict) else resp
            data = raw if isinstance(raw, list) else []
            # Normalize to list of dicts: {t,o,h,l,c}
            norm = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        t = item.get('t') or item.get('T') or item.get('time') or item.get('open_time')
                        o = item.get('o') or item.get('O') or item.get('open')
                        h = item.get('h') or item.get('H') or item.get('high')
                        l = item.get('l') or item.get('L') or item.get('low')
                        c = item.get('c') or item.get('C') or item.get('close')
                    elif isinstance(item, (list, tuple)) and len(item) >= 5:
                        # Common array format: [open_time, open, high, low, close, ...]
                        t, o, h, l, c = item[0], item[1], item[2], item[3], item[4]
                    else:
                        continue
                    norm.append({'t': float(t), 'o': float(o), 'h': float(h), 'l': float(l), 'c': float(c)})
                except Exception:
                    continue
            return norm
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Klines fetch hatası: {e}")
            return []

    def apply_risk_settings(self):
        """Risk ayarlarını uygula"""
        try:
            if hasattr(risk_manager, 'apply_settings'):
                risk_manager.apply_settings()
            else:
                self.logger.warning(LogCategory.GUI, "Risk yöneticisi 'apply_settings' bulunamadı")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Risk ayarları uygulanamadı: {e}")
    
    def update_market_data(self):
        try:
            # BTC etiketi her zaman BTCUSDT'i göstersin
            px = float(self._price_cache.get('BTCUSDT', 0) or 0)
            if px > 0:
                self.btc_price_label.config(text=f"{px:,.2f}")
                # Yüzde değişim (tercihen 24s ticker'dan)
                pct = None
                try:
                    if 'BTCUSDT' in self._change24_cache:
                        pct = float(self._change24_cache.get('BTCUSDT') or 0.0)
                except Exception:
                    pct = None
                if pct is None:
                    prev = float(self._prev_prices.get('BTCUSDT', 0.0) or 0.0)
                    pct = 0.0
                    if prev > 0:
                        try:
                            pct = (px - prev) / prev * 100.0
                        except Exception:
                            pct = 0.0
                try:
                    self.btc_change_label.config(text=f"{pct:+.2f}%")
                    if pct > 0.0:
                        self.btc_change_label.configure(style='Positive.TLabel')
                        self.btc_price_label.configure(style='Positive.TLabel')
                    elif pct < 0.0:
                        self.btc_change_label.configure(style='Negative.TLabel')
                        self.btc_price_label.configure(style='Negative.TLabel')
                    else:
                        self.btc_change_label.configure(style='Neutral.TLabel')
                        self.btc_price_label.configure(style='Neutral.TLabel')
                except Exception:
                    pass
                # Önceki fiyatı güncelle
                try:
                    self._prev_prices['BTCUSDT'] = px
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Piyasa verileri güncelleme hatası: {e}")
    
    def create_positions_section(self, parent):
        """Pozisyonlar bölümü"""
        try:
            positions_frame = ttk.LabelFrame(parent, text="Açık Pozisyonlar")
            # Daha az yer kaplasın, log alanı görünsün
            positions_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
            
            # Pozisyon tablosu (USDT bazlı)
            columns = (
                'Sembol',
                'Giriş Fiyatı (USDT)',
                'Miktar',
                'Mevcut Fiyat (USDT)',
                'Pozisyon Değeri (USDT)',
                'P&L (USDT)',
                'Durum'
            )
            # Yüksekliği azalt
            self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show='headings', height=11)
            
            # Sütun başlıkları
            for col in columns:
                self.positions_tree.heading(col, text=col)
                self.positions_tree.column(col, width=100, anchor=tk.CENTER)
            # Renkli tag'ler
            try:
                self.positions_tree.tag_configure('pnl_pos', foreground='green')
                self.positions_tree.tag_configure('pnl_neg', foreground='red')
            except Exception:
                pass
            
            # Scrollbar
            positions_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
            self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
            
            self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Pozisyonlar bölümü oluşturma hatası: {e}")

    def _get_last_price_for_symbol(self, sym: str) -> float:
        """Sembol için son fiyatı döndür. Cache + basit TRY/USDT dönüşüm."""
        try:
            s = (sym or '').strip().upper()
            px = float(self._price_cache.get(s, 0) or 0)
            if px > 0:
                return px
            # TRY paritesi ise USDT bazlı fiyatı kur ile çevir
            if s.endswith('TRY'):
                base = s[:-3]
                usdt_sym = f"{base}USDT"
                p_usdt = float(self._price_cache.get(usdt_sym, 0) or 0)
                if p_usdt > 0 and float(self._usdt_try or 0) > 0:
                    return p_usdt * float(self._usdt_try)
            # USDT paritesi için cache yoksa 0 döndür
            return 0.0
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Sembol son fiyatı alma hatası: {e}")
            return 0.0
    
    def create_trades_section(self, parent):
        """Son işlemler bölümü"""
        try:
            trades_frame = ttk.LabelFrame(parent, text="Son İşlemler")
            trades_frame.pack(fill=tk.X, pady=(0, 10))
            
            # İşlem tablosu
            columns = ('Zaman', 'Sembol', 'Yön', 'Büyüklük', 'Fiyat', 'P&L')
            # Yüksekliği azalt
            self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=3)
            
            # Sütun başlıkları
            for col in columns:
                self.trades_tree.heading(col, text=col)
                self.trades_tree.column(col, width=80, anchor=tk.CENTER)
            
            # Scrollbar
            trades_scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
            self.trades_tree.configure(yscrollcommand=trades_scrollbar.set)
            
            self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"İşlemler bölümü oluşturma hatası: {e}")
    
    def create_performance_section(self, parent):
        """Performans bölümü"""
        try:
            performance_frame = ttk.LabelFrame(parent, text="Performans Metrikleri")
            performance_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Toplam P&L
            pnl_frame = ttk.Frame(performance_frame)
            pnl_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(pnl_frame, text="Toplam P&L:", style='Header.TLabel').pack(side=tk.LEFT)
            self.total_pnl_label = ttk.Label(pnl_frame, text="0.00", style='Header.TLabel')
            self.total_pnl_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Günlük P&L
            daily_pnl_frame = ttk.Frame(performance_frame)
            daily_pnl_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(daily_pnl_frame, text="Günlük P&L:").pack(side=tk.LEFT)
            self.daily_pnl_label = ttk.Label(daily_pnl_frame, text="0.00", style='Neutral.TLabel')
            self.daily_pnl_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Win Rate
            winrate_frame = ttk.Frame(performance_frame)
            winrate_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(winrate_frame, text="Kazanma Oranı:").pack(side=tk.LEFT)
            self.winrate_label = ttk.Label(winrate_frame, text="0.00%", style='Neutral.TLabel')
            self.winrate_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Toplam İşlem
            total_trades_frame = ttk.Frame(performance_frame)
            total_trades_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(total_trades_frame, text="Günlük İşlem:").pack(side=tk.LEFT)
            self.daily_trades_label = ttk.Label(total_trades_frame, text="0", style='Neutral.TLabel')
            self.daily_trades_label.pack(side=tk.LEFT, padx=(10, 20))
            ttk.Label(total_trades_frame, text="Toplam İşlem:").pack(side=tk.LEFT)
            self.total_trades_label = ttk.Label(total_trades_frame, text="0", style='Neutral.TLabel')
            self.total_trades_label.pack(side=tk.LEFT, padx=(10, 0))

            # Günlük sayaç sıfırla
            try:
                reset_btn = ttk.Button(performance_frame, text="Günlük Sayaç Sıfırla", command=self._reset_daily_counters)
                reset_btn.pack(anchor=tk.W, padx=5, pady=(0,5))
            except Exception:
                pass

            # Paper hesabı tam sıfırla (4000 USDT ile başlat)
            try:
                reset_paper_btn = ttk.Button(performance_frame, text="Paper Sıfırla (4000 USDT)", command=self._reset_paper_account)
                reset_paper_btn.pack(anchor=tk.W, padx=5, pady=(0,5))
            except Exception:
                pass

            # Portföy Toplamları (USDT)
            totals_frame = ttk.LabelFrame(performance_frame, text="Portföy Toplamları (USDT)")
            totals_frame.pack(fill=tk.X, padx=5, pady=5)
            row1 = ttk.Frame(totals_frame); row1.pack(fill=tk.X)
            ttk.Label(row1, text="Toplam Nakit (USDT):").pack(side=tk.LEFT)
            self.total_cash_label = ttk.Label(row1, text="0.00")
            self.total_cash_label.pack(side=tk.LEFT, padx=(10, 0))
            row2 = ttk.Frame(totals_frame); row2.pack(fill=tk.X)
            ttk.Label(row2, text="Coin Değeri (USDT):").pack(side=tk.LEFT)
            self.total_coin_label = ttk.Label(row2, text="0.00")
            self.total_coin_label.pack(side=tk.LEFT, padx=(10, 0))
            row3 = ttk.Frame(totals_frame); row3.pack(fill=tk.X)
            ttk.Label(row3, text="Portföy Toplamı (USDT):").pack(side=tk.LEFT)
            self.total_portfolio_label = ttk.Label(row3, text="0.00")
            self.total_portfolio_label.pack(side=tk.LEFT, padx=(10, 0))
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Performans bölümü oluşturma hatası: {e}")
    
    def create_price_chart_section(self, parent):
        try:
            frame = ttk.LabelFrame(parent, text="Fiyat Grafiği (1m)")
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self._mpl_fig = Figure(figsize=(5, 3), dpi=100)
            self._mpl_ax = self._mpl_fig.add_subplot(111)
            self._mpl_ax.set_title(f"1 Dakikalık Mumlar - {self._current_symbol}")
            self._mpl_ax.grid(True, alpha=0.25)
            self._mpl_canvas = FigureCanvasTkAgg(self._mpl_fig, master=frame)
            self._mpl_canvas.draw()
            self._mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            # Toolbar devre dışı: bazı sistemlerde ilk odaklanmada takılma yapabiliyor
            # İlk çizimi kısa gecikmeyle planla (UI tam ayağa kalksın)
            try:
                self.root.after(200, self.update_price_chart)
            except Exception:
                self.update_price_chart()
            self._schedule_chart_update()
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Grafik bölümü oluşturma hatası: {e}")

    def _schedule_chart_update(self, delay_ms: int = 15000):
        try:
            if self._chart_after_id:
                try:
                    self.root.after_cancel(self._chart_after_id)
                except Exception as e:
                    self.logger.warning(LogCategory.GUI, f"Grafik zamanlayıcı iptal hatası: {e}")
            self._chart_after_id = self.root.after(delay_ms, self.update_price_chart)
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Grafik zamanlayıcı hatası: {e}")
    
    def schedule_update_loop(self):
        """Geriye dönük uyum: genel update döngüsü."""
        try:
            # _tick() metodunu planla (update_gui + _engine_tick çalıştırır)
            if self._after_id:
                try:
                    self.root.after_cancel(self._after_id)
                except Exception:
                    pass
            self._after_id = self.root.after(2000, self._tick)  # 2 saniyede bir
            
            # Grafik güncellemesi ayrı (daha seyrek)
            self._schedule_chart_update()
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"schedule_update_loop hatası: {e}")
    
    def update_price_chart(self):
        try:
            if not self._mpl_ax:
                return
            try:
                raw_sym = (self.paper_symbol_combo.get() or self._current_symbol or 'BTCUSDT') if hasattr(self, 'paper_symbol_combo') else self._current_symbol
                sym = self._normalize_symbol_for_api(raw_sym)
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Sembol alma hatası: {e}")
                return
            # Non-blocking fetch: halihazırda çekim varsa bekleme
            if getattr(self, '_klines_inflight', False):
                return
            self._klines_inflight = True
            # TF seçimi
            try:
                tf = (self.chart_tf_var.get() or '1m').lower()
            except Exception:
                tf = '1m'
            def _worker(sym_local: str, tf_local: str):
                try:
                    # Son mumu eklemek için küçük limit yeterli; ilk kurulumda eksikse ana thread tamamlar
                    candles = self._fetch_klines(sym_local, tf_local, limit=120)
                except Exception:
                    candles = []
                finally:
                    try:
                        # Ana thread'de grafiği güncelle
                        self.root.after(0, lambda c=candles, s=sym_local, t=tf_local: self._update_chart_main_thread(c, s, t))
                    except Exception:
                        self._klines_inflight = False
            threading.Thread(target=_worker, args=(sym, tf), daemon=True).start()
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Grafik güncelleme hatası: {e}")
        finally:
            self._schedule_chart_update()

    def _update_chart_main_thread(self, candles: list, sym: str, tf: str):
        try:
            if hasattr(self, '_mpl_ax') and self._mpl_ax is not None:
                # Cache anahtarı
                key = f"{sym}|{tf}"
                if not hasattr(self, '_chart_data'):
                    self._chart_data = {}
                series = self._chart_data.get(key, {'xs': [], 'closes': []})
                # Gelen veriden zaman ve close dizilerini hazırla
                xs_new = []
                closes_new = []
                try:
                    xs_new = [_dtmod.datetime.fromtimestamp(c["t"]/1000.0) for c in candles if c.get("t")]
                    closes_new = [c["c"] for c in candles if c.get("c")]
                except Exception as ve:
                    self.logger.warning(LogCategory.GUI, f"Veri işleme hatası: {ve}")
                # İlk çizim: seriyi başlat
                if not series['xs'] or not series['closes']:
                    series['xs'] = xs_new[-200:] if xs_new else []
                    series['closes'] = closes_new[-200:] if closes_new else []
                else:
                    # Sadece son mumu ekle
                    if xs_new:
                        last_ts_old = series['xs'][-1] if series['xs'] else None
                        last_ts_new = xs_new[-1]
                        if (last_ts_old is None) or (last_ts_new > last_ts_old):
                            series['xs'].append(last_ts_new)
                            series['closes'].append(closes_new[-1])
                            # Eski veriyi sınırlı tut (hafıza)
                            if len(series['xs']) > 1000:
                                series['xs'] = series['xs'][-1000:]
                                series['closes'] = series['closes'][-1000:]
                # Candlestick çizim: son 200 barı yeniden çiz (stabil ve görünür çözüm)
                self._mpl_ax.clear()
                self._mpl_ax.set_title(f"{tf.upper()} Mumlar - {sym}")
                # Son çekilen 'candles' üzerinden OHLC oluştur
                try:
                    times = [_dtmod.datetime.fromtimestamp(c['t']/1000.0) for c in candles]
                    opens = [float(c['o']) for c in candles]
                    highs = [float(c['h']) for c in candles]
                    lows = [float(c['l']) for c in candles]
                    closes = [float(c['c']) for c in candles]
                except Exception:
                    times = series['xs']
                    opens = highs = lows = []
                    closes = series['closes']
                # Bar genişliği
                if times and len(times) > 1:
                    dt_sec = (times[-1] - times[-2]).total_seconds()
                else:
                    # TF varsayımı
                    dt_map = {'1m': 60, '5m': 300, '1h': 3600, '4h': 14400}
                    dt_sec = dt_map.get(tf, 60)
                width_days = (dt_sec / 86400.0) * 0.8
                up_color = 'tab:green'
                down_color = 'tab:red'
                # Sınırla
                max_n = 200
                n0 = max(0, len(times) - max_n)
                if not times or not opens or not highs or not lows or not closes:
                    # Veri yoksa mesaj göster
                    self._mpl_ax.text(0.5, 0.5, 'Veri yok', ha='center', va='center', transform=self._mpl_ax.transAxes)
                for t, o, h, l, c in zip(times[n0:], opens[n0:], highs[n0:], lows[n0:], closes[n0:]):
                    color = up_color if c >= o else down_color
                    # High-Low çizgisi
                    try:
                        xnum = mdates.date2num(t)
                        self._mpl_ax.vlines(xnum, l, h, color=color, linewidth=1, alpha=0.8)
                    except Exception:
                        pass
                    # Gövde (open-close)
                    oc_low = min(o, c)
                    oc_high = max(o, c)
                    try:
                        xnum = mdates.date2num(t)
                        rect = Rectangle((xnum - width_days/2.0, oc_low), width_days, oc_high - oc_low,
                                         facecolor=color, edgecolor=color, alpha=0.8)
                        self._mpl_ax.add_patch(rect)
                    except Exception:
                        # Yedek: basit çizgi
                        try:
                            self._mpl_ax.plot([t, t], [o, c], color=color, linewidth=4, solid_capstyle='butt')
                        except Exception:
                            pass
                # X eksenini tarih ölçeğine al
                try:
                    self._mpl_ax.xaxis_date()
                except Exception:
                    pass
                # Görünürlük için eksen limitlerini ayarla
                try:
                    if times and highs and lows:
                        xnums = [mdates.date2num(t) for t in times[n0:]]
                        self._mpl_ax.set_xlim(min(xnums), max(xnums))
                        self._mpl_ax.set_ylim(min(lows[n0:]), max(highs[n0:]))
                except Exception:
                    pass
                self._mpl_ax.grid(True, alpha=0.25)
                self._mpl_fig.autofmt_xdate()
                if hasattr(self, '_mpl_canvas') and self._mpl_canvas is not None:
                    self._mpl_canvas.draw_idle()
                # Cache'i geri yaz
                self._chart_data[key] = series
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Grafik ana thread güncelleme hatası: {e}")
        finally:
            self._klines_inflight = False

    def _fetch_klines(self, symbol: str, tf: str, limit: int = 200) -> list:
        """Genel kline fetcher. tf: '1m','5m','1h','4h'"""
        try:
            tf = (tf or '1m').lower()
            tf_map = {'1m': '1', '5m': '5', '1h': '60', '4h': '240'}
            interval = tf_map.get(tf, '1')
            def _do_fetch(sym_str: str) -> list:
                params = {'pairSymbol': self._normalize_symbol_for_api(sym_str), 'interval': interval, 'limit': int(limit or 200)}
                resp = self.multi_api_manager.make_request('GET', '/klines', params=params)
                raw = resp.get('data', resp) if isinstance(resp, dict) else resp
                return raw if isinstance(raw, list) else []

            data = _do_fetch(symbol)
            # Normalize to list of dicts: {t,o,h,l,c}
            norm = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        t = item.get('t') or item.get('T') or item.get('time') or item.get('open_time')
                        o = item.get('o') or item.get('O') or item.get('open')
                        h = item.get('h') or item.get('H') or item.get('high')
                        l = item.get('l') or item.get('L') or item.get('low')
                        c = item.get('c') or item.get('C') or item.get('close')
                    elif isinstance(item, (list, tuple)) and len(item) >= 5:
                        # Common array format: [open_time, open, high, low, close, ...]
                        t, o, h, l, c = item[0], item[1], item[2], item[3], item[4]
                    else:
                        continue
                    norm.append({'t': float(t), 'o': float(o), 'h': float(h), 'l': float(l), 'c': float(c)})
                except Exception:
                    continue
            if not norm:
                try:
                    self.logger.warning(LogCategory.GUI, f"Klines boş: sym={symbol}, tf={tf}, interval={interval}")
                except Exception:
                    pass
                # Quotesuz semboller için alternatif denemeler: SYMBOLUSDT, SYMBOLTRY
                base = self._normalize_symbol_for_api(symbol)
                if not (base.endswith('USDT') or base.endswith('TRY')):
                    for alt in (base + 'USDT', base + 'TRY'):
                        try:
                            data2 = _do_fetch(alt)
                            norm2 = []
                            for item in data2:
                                try:
                                    if isinstance(item, dict):
                                        t = item.get('t') or item.get('T') or item.get('time') or item.get('open_time')
                                        o = item.get('o') or item.get('O') or item.get('open')
                                        h = item.get('h') or item.get('H') or item.get('high')
                                        l = item.get('l') or item.get('L') or item.get('low')
                                        c = item.get('c') or item.get('C') or item.get('close')
                                    elif isinstance(item, (list, tuple)) and len(item) >= 5:
                                        t, o, h, l, c = item[0], item[1], item[2], item[3], item[4]
                                    else:
                                        continue
                                    norm2.append({'t': float(t), 'o': float(o), 'h': float(h), 'l': float(l), 'c': float(c)})
                                except Exception:
                                    continue
                            if norm2:
                                try:
                                    self.logger.info(LogCategory.GUI, f"Klines fallback başarılı: {symbol} -> {alt}")
                                except Exception:
                                    pass
                                return norm2
                        except Exception:
                            continue
            return norm
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Kline fetch hatası: {e}")
            return []

    def _normalize_symbol_for_api(self, s: str) -> str:
        try:
            if not s:
                return 'BTCUSDT'
            s = str(s).strip().upper()
            # Yaygın format düzeltmeleri: 'BTC/USDT' -> 'BTCUSDT', ' BTCTRY ' -> 'BTCTRY'
            s = s.replace('/', '').replace('-', '').replace(' ', '')
            return s
        except Exception:
            return 'BTCUSDT'

    def on_chart_tf_changed(self, event=None):
        try:
            # TF değişince çizgiyi sıfırlamadan veri cache'ini boşaltıp hızlı güncelle
            if not hasattr(self, '_chart_data'):
                self._chart_data = {}
            try:
                sym = (self.paper_symbol_combo.get() or self._current_symbol or 'BTCUSDT').strip().upper() if hasattr(self, 'paper_symbol_combo') else self._current_symbol
            except Exception:
                sym = self._current_symbol or 'BTCUSDT'
            key_prefix = f"{sym}|"
            # İlgili sembol cache'lerini temizle (farklı tf'ler)
            for k in list(self._chart_data.keys()):
                if isinstance(k, str) and k.startswith(key_prefix):
                    self._chart_data.pop(k, None)
            # Bir sonraki döngüde yeni TF ile veri çekilecek
            self.update_price_chart()
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"TF değişim hatası: {e}")

    def open_settings(self):
        """Sistem ayarları penceresini aç (yeni pencerede)."""
        try:
            w = tk.Toplevel(self.root)
            w.title("Sistem Ayarları")
            w.geometry("800x600")
            w.resizable(True, True)
            container = ttk.Frame(w)
            container.pack(fill=tk.BOTH, expand=True)
            # Scroll alanı
            canvas = tk.Canvas(container, highlightthickness=0)
            vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            inner = ttk.Frame(canvas)
            canvas.create_window((0, 0), window=inner, anchor="nw")
            def _on_config(event=None):
                try:
                    canvas.configure(scrollregion=canvas.bbox("all"))
                except Exception:
                    pass
            inner.bind("<Configure>", _on_config)
            # Basit örnek alanlar (genişlerse scroll devreye girer)
            sec1 = ttk.LabelFrame(inner, text="Ağ ve API", padding=10); sec1.pack(fill=tk.X, pady=5, padx=10)
            ttk.Label(sec1, text="API Taban URL:").pack(side=tk.LEFT)
            ttk.Entry(sec1, width=50).pack(side=tk.LEFT, padx=5)
            sec2 = ttk.LabelFrame(inner, text="Veri ve Güncellemeler", padding=10); sec2.pack(fill=tk.X, pady=5, padx=10)
            ttk.Label(sec2, text="Ticker Aralığı (ms):").pack(side=tk.LEFT)
            ttk.Entry(sec2, width=10).pack(side=tk.LEFT, padx=5)
            # Kapat butonu
            btn_frame = ttk.Frame(inner); btn_frame.pack(fill=tk.X, pady=10, padx=10)
            ttk.Button(btn_frame, text="Kapat", command=w.destroy).pack(side=tk.RIGHT)
        except Exception as e:
            try:
                messagebox.showerror("Hata", f"Ayarlar penceresi açılamadı: {e}")
            except Exception:
                pass

    def create_ai_analysis_section(self, parent):
        try:
            frame = ttk.LabelFrame(parent, text="AI Analiz")
            frame.pack(fill=tk.X, pady=(0, 10))
            # Strateji adı
            row1 = ttk.Frame(frame); row1.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(row1, text="Strateji:").pack(side=tk.LEFT)
            self.strategy_name_label = ttk.Label(row1, text="TRY Değişim Tarama", style='Neutral.TLabel')
            self.strategy_name_label.pack(side=tk.LEFT, padx=(10, 20))
            # Eşikler gösterim
            ttk.Label(row1, text="Eşikler:").pack(side=tk.LEFT)
            self.strategy_thresholds_label = ttk.Label(row1, text=f"Al <= {self._buy_threshold_pct}% | Sat >= +{self._sell_threshold_pct}%", style='Neutral.TLabel')
            self.strategy_thresholds_label.pack(side=tk.LEFT, padx=(10, 0))

            # Eşik ayarları
            row2 = ttk.Frame(frame); row2.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(row2, text="Al %:").pack(side=tk.LEFT)
            self.buy_threshold_entry = ttk.Entry(row2, width=6)
            self.buy_threshold_entry.pack(side=tk.LEFT, padx=(5, 20))
            self.buy_threshold_entry.insert(0, str(self._buy_threshold_pct))
            ttk.Label(row2, text="Sat %:").pack(side=tk.LEFT)
            self.sell_threshold_entry = ttk.Entry(row2, width=6)
            self.sell_threshold_entry.pack(side=tk.LEFT, padx=(5, 20))
            self.sell_threshold_entry.insert(0, str(self._sell_threshold_pct))
            ttk.Button(row2, text="Uygula", command=self.apply_threshold_settings).pack(side=tk.LEFT)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"AI analiz bölümü oluşturma hatası: {e}")

    def apply_threshold_settings(self):
        try:
            try:
                b = float((self.buy_threshold_entry.get() or '').replace(',', '.'))
                s = float((self.sell_threshold_entry.get() or '').replace(',', '.'))
                self._buy_threshold_pct = b
                self._sell_threshold_pct = s
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Eşik ayarı alma hatası: {e}")
            try:
                self.strategy_thresholds_label.config(text=f"Al <= {self._buy_threshold_pct}% | Sat >= +{self._sell_threshold_pct}%")
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Eşik label güncelleme hatası: {e}")
            try:
                self._save_threshold_config()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Eşik ayarı kaydetme hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Eşik ayarı uygulama hatası: {e}")

    def _threshold_config_path(self) -> str:
        try:
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"Veri dizini alma hatası: {e}")
            base = os.path.join(os.getcwd(), 'data')
        if not os.path.isdir(base):
            try:
                os.makedirs(base, exist_ok=True)
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Veri dizini oluşturma hatası: {e}")
        return os.path.join(base, 'strategy_config.json')

    def _load_threshold_config(self):
        path = self._threshold_config_path()
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self._buy_threshold_pct = float(cfg.get('buy_threshold_pct', self._buy_threshold_pct))
            self._sell_threshold_pct = float(cfg.get('sell_threshold_pct', self._sell_threshold_pct))

    def _save_threshold_config(self):
        path = self._threshold_config_path()
        cfg = {
            'buy_threshold_pct': self._buy_threshold_pct,
            'sell_threshold_pct': self._sell_threshold_pct,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    
    def create_risk_section(self, parent):
        """Risk bölümü"""
        try:
            risk_frame = ttk.LabelFrame(parent, text="Risk Metrikleri")
            risk_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Risk seviyesi
            risk_level_frame = ttk.Frame(risk_frame)
            risk_level_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(risk_level_frame, text="Risk Seviyesi:", style='Header.TLabel').pack(side=tk.LEFT)
            self.risk_level_label = ttk.Label(risk_level_frame, text="Düşük", style='Positive.TLabel')
            self.risk_level_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Max Drawdown
            drawdown_frame = ttk.Frame(risk_frame)
            drawdown_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(drawdown_frame, text="Max Drawdown:").pack(side=tk.LEFT)
            self.drawdown_label = ttk.Label(drawdown_frame, text="0.00%", style='Neutral.TLabel')
            self.drawdown_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Açık pozisyon sayısı
            open_positions_frame = ttk.Frame(risk_frame)
            open_positions_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(open_positions_frame, text="Açık Pozisyon:").pack(side=tk.LEFT)
            self.open_positions_label = ttk.Label(open_positions_frame, text="0", style='Neutral.TLabel')
            self.open_positions_label.pack(side=tk.LEFT, padx=(10, 0))

            # Risk ayarları (SL/TP/Trailing)
            settings_frame = ttk.Frame(risk_frame)
            settings_frame.pack(fill=tk.X, padx=5, pady=(10, 5))
            ttk.Label(settings_frame, text="SL %:").pack(side=tk.LEFT)
            self.sl_entry = ttk.Entry(settings_frame, width=6)
            self.sl_entry.pack(side=tk.LEFT, padx=(5, 10))
            self.sl_entry.insert(0, "1.0")
            ttk.Label(settings_frame, text="TP %:").pack(side=tk.LEFT)
            self.tp_entry = ttk.Entry(settings_frame, width=6)
            self.tp_entry.pack(side=tk.LEFT, padx=(5, 10))
            self.tp_entry.insert(0, "2.0")
            ttk.Label(settings_frame, text="Trailing %:").pack(side=tk.LEFT)
            self.trailing_entry = ttk.Entry(settings_frame, width=6)
            self.trailing_entry.pack(side=tk.LEFT, padx=(5, 10))
            self.trailing_entry.insert(0, "1.0")
            apply_btn = ttk.Button(settings_frame, text="Risk Ayarlarını Kaydet", command=self.apply_risk_settings)
            apply_btn.pack(side=tk.LEFT, padx=(10, 0))
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Risk bölümü oluşturma hatası: {e}")
    
    def create_ai_analysis_section(self, parent):
        """AI analiz bölümü"""
        try:
            ai_frame = ttk.LabelFrame(parent, text="AI Analiz")
            ai_frame.pack(fill=tk.BOTH, expand=True)

            # Piyasa rejimi
            regime_frame = ttk.Frame(ai_frame)
            regime_frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(regime_frame, text="Piyasa Rejimi:", style='Header.TLabel').pack(side=tk.LEFT)
            # Varsayılan rejim metni: Yatay (unknown yerine)
            self.market_regime_label = ttk.Label(regime_frame, text="Yatay", style='Neutral.TLabel')
            self.market_regime_label.pack(side=tk.LEFT, padx=(10, 0))

            # Seçilen strateji
            row_strategy = ttk.Frame(ai_frame)
            row_strategy.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(row_strategy, text="Seçilen Strateji:").pack(side=tk.LEFT)
            self.auto_strategy_label = ttk.Label(row_strategy, text="-", style='Neutral.TLabel')
            self.auto_strategy_label.pack(side=tk.LEFT, padx=(10, 0))

            # Strateji sinyali
            row_signal = ttk.Frame(ai_frame)
            row_signal.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(row_signal, text="Strateji Sinyali:").pack(side=tk.LEFT)
            self.strategy_signal_label = ttk.Label(row_signal, text="NÖTR", style='Neutral.TLabel')
            self.strategy_signal_label.pack(side=tk.LEFT, padx=(10, 0))

            # Eşikler ve bilgi
            info_frame = ttk.Frame(ai_frame)
            info_frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(info_frame, text="Eşikler:").pack(side=tk.LEFT)
            self.strategy_thresholds_label = ttk.Label(info_frame, text=f"AL≤{self._buy_threshold_pct}%, SAT≥{self._sell_threshold_pct}%")
            self.strategy_thresholds_label.pack(side=tk.LEFT, padx=(10, 20))
            ttk.Label(info_frame, text="Strateji Notu:").pack(side=tk.LEFT)
            self.strategy_info_label = ttk.Label(info_frame, text="Manuel mod", style='Status.TLabel')
            self.strategy_info_label.pack(side=tk.LEFT, padx=(10, 0))

            # İlk durum güncellemesi
            try:
                self._update_strategy_status()
            except Exception:
                pass

        except Exception as e:
            self.logger.error(LogCategory.GUI, f"AI analiz bölümü oluşturma hatası: {e}")

    def _tick(self):
        """Her tick'te GUI'yi güncelle ve yeniden planla"""
        try:
            self.update_gui()
            try:
                self._engine_tick()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Strateji motoru tick hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"GUI tick hatası: {e}")
        finally:
            try:
                self._after_id = None
                self.schedule_update_loop()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Zamanlayıcı planlama hatası: {e}")

    def update_gui(self):
        """Hafif GUI güncellemeleri (ana thread)"""
        try:
            self.update_market_data()
            self.update_performance()
            try:
                self._update_strategy_status()
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"update_gui hatası: {e}")

    def _engine_tick(self):
        """Hafif strateji motoru tick'i (bloklamaz) - Basit trend following tarama"""
        try:
            if not getattr(self, '_engine_enabled', False):
                return
            
            # Zamanlama kontrolü
            now = time.time()
            if now - getattr(self, '_last_strategy_time', 0.0) < self._strategy_min_interval:
                return
            
            # Zaten bir strateji çalışıyorsa, atla (çakışmayı önle)
            if getattr(self, '_strategy_inflight', False):
                return
                
            self._last_strategy_time = now
            self._strategy_inflight = True
            
            # Arka plan thread'de sinyalleri üret (UI'yı bloklamadan)
            def _strategy_worker():
                try:
                    signals = []
                    
                    # Açık pozisyon sayısını kontrol et
                    try:
                        open_cnt = sum(1 for p in (paper_executor.positions or {}).values() if (p.get('qty') or 0.0) > 0.0)
                    except Exception:
                        open_cnt = 0
                    
                    # Maksimum pozisyon kontrolü
                    allow_new = True
                    if hasattr(self, 'max_open_positions') and self.max_open_positions is not None:
                        allow_new = open_cnt < int(self.max_open_positions or 0)
                    
                    if not allow_new:
                        self._strategy_inflight = False
                        return
                    
                    # Basit trend following tarama
                    ref_sec = 30.0  # 30 saniye referans pencere
                    cooldown = 30.0  # 30 saniye cooldown
                    
                    for sym in list(getattr(self, '_scan_symbols', [])):
                        # Sadece USDT pariteleri
                        if not sym.endswith('USDT'):
                            continue
                        if self._is_blocked_symbol(sym):
                            continue
                        
                        # Fiyatı al
                        px = float(self._price_cache.get(sym, 0.0) or 0.0)
                        if px <= 0:
                            continue
                        
                        # Fiyat geçmişi
                        if not hasattr(self, '_price_hist'):
                            self._price_hist = {}
                        dq = self._price_hist.get(sym)
                        if dq is None:
                            from collections import deque
                            dq = deque(maxlen=240)
                            self._price_hist[sym] = dq
                        dq.append((now, px))
                        
                        # Referans fiyat
                        base = None
                        for t, val in dq:
                            if now - t >= ref_sec:
                                base = val
                                break
                        if not base:
                            continue
                        
                        # Fiyat değişimi
                        pct = (px - base) / base * 100.0
                        
                        # Cooldown kontrolü
                        if not hasattr(self, '_last_trade_time'):
                            self._last_trade_time = {}
                        last_t = float(self._last_trade_time.get(sym, 0.0))
                        if now - last_t < cooldown:
                            continue
                        
                        # %0.5+ artış varsa al sinyali
                        if pct >= 0.5:
                            # Bakiye al
                            account_balance = float(paper_executor.balance_usdt or 1000.0)
                            
                            # Sinyal gücü hesapla (%2 = 1.0 strength)
                            strength = min(1.0, abs(pct) / 2.0)
                            
                            # Sinyal oluştur
                            signal = {
                                'symbol': sym,
                                'side': 'buy',
                                'entry_price': px,
                                'strength': strength,
                                'confidence': 0.8
                            }
                            
                            # Position size hesapla
                            position_value, details = position_sizer.calculate_position_size(
                                method=PositionSizingMethod.SIGNAL_BASED_FULL_CAPITAL,
                                account_balance=account_balance,
                                signal=signal,
                                market_data={},
                                historical_performance=None
                            )
                            
                            # Miktar hesapla
                            qty = round(position_value / px, 6)
                            
                            if qty > 0:
                                signals.append({
                                    'symbol': sym,
                                    'side': 'buy',
                                    'entry_price': px,
                                    'size': qty,
                                    'strategy_name': 'trend_following',
                                    'strength': strength
                                })
                                self._last_trade_time[sym] = now
                                
                                # Sadece 1 sinyal yeterli
                                break
                    
                    # Sinyaller varsa işle
                    if signals and len(signals) > 0:
                        self.logger.info(LogCategory.GUI, f"{len(signals)} sinyal üretildi: {[s['symbol'] for s in signals]}")
                        self.on_signals(signals)
                        
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Sinyal üretme hatası: {e}")
                finally:
                    self._strategy_inflight = False
            
            # Thread'i başlat
            threading.Thread(target=_strategy_worker, daemon=True).start()
                
        except Exception as e:
            self.logger.warning(LogCategory.GUI, f"engine_tick hata: {e}")
            self._strategy_inflight = False
    
    def update_performance(self):
        """Performans metriklerini güncelle"""
        try:
            # Paper executor performansını göster
            # Fiyat cache'i paper executor'a vererek gerçekçi MTM yapalım
            price_map = dict(self._price_cache)
            if self._usdt_try and self._usdt_try > 0:
                price_map['USDTTRY'] = self._usdt_try
            summary = paper_executor.get_summary(price_map)
            total_pnl = summary.get('total_pnl', 0.0)
            self.total_pnl_label.config(text=f"{total_pnl:,.2f}")
            # Toplam P&L rengi
            try:
                if total_pnl > 0:
                    self.total_pnl_label.configure(style='Positive.TLabel')
                elif total_pnl < 0:
                    self.total_pnl_label.configure(style='Negative.TLabel')
                else:
                    self.total_pnl_label.configure(style='Neutral.TLabel')
            except Exception:
                pass
            # Günlük P&L: paper_executor günlük realize P&L
            daily_pnl = float(summary.get('daily_realized_pnl', 0.0) or 0.0)
            self.daily_pnl_label.config(text=f"{daily_pnl:,.2f}")
            # Günlük P&L rengi
            try:
                if daily_pnl > 0:
                    self.daily_pnl_label.configure(style='Positive.TLabel')
                elif daily_pnl < 0:
                    self.daily_pnl_label.configure(style='Negative.TLabel')
                else:
                    self.daily_pnl_label.configure(style='Neutral.TLabel')
            except Exception:
                pass
            # Genel başarı oranı (strategy_manager üzerinden) - bloklamayı önlemek için arka plan thread
            if not getattr(self, '_perf_inflight', False):
                self._perf_inflight = True
                def _perf_worker():
                    try:
                        overall = strategy_manager.get_overall_performance()
                        sr = 0.0
                        try:
                            sr = float(overall.get('success_rate', 0.0) or 0.0)
                        except Exception:
                            sr = 0.0
                        try:
                            def _safe_set_winrate(v=sr):
                                try:
                                    if hasattr(self, 'winrate_label') and self.winrate_label.winfo_exists():
                                        self.winrate_label.config(text=f"{v:.1f}%")
                                except Exception:
                                    pass
                            self.root.after(0, _safe_set_winrate)
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            self.logger.warning(LogCategory.GUI, f"Win rate güncelleme hatası: {e}")
                        except Exception:
                            pass
                    finally:
                        self._perf_inflight = False
                threading.Thread(target=_perf_worker, daemon=True).start()
            # İşlem sayaçları
            try:
                self.daily_trades_label.config(text=str(int(summary.get('daily_trade_count', 0) or 0)))
            except Exception:
                self.daily_trades_label.config(text="0")
            try:
                total_tr = int(summary.get('total_trade_count', 0) or 0)
                if total_tr <= 0:
                    total_tr = len(getattr(paper_executor, 'trades', []))
                self.total_trades_label.config(text=str(total_tr))
            except Exception:
                self.total_trades_label.config(text=str(len(getattr(paper_executor, 'trades', []))))
            # Açık pozisyon sayısı
            try:
                open_cnt = sum(1 for p in (paper_executor.positions or {}).values() if float(p.get('qty') or 0.0) > 0.0)
                if hasattr(self, 'open_positions_label'):
                    self.open_positions_label.config(text=str(open_cnt))
            except Exception:
                pass
            # Nakit/varlık toplamları (USDT)
            try:
                fx = float(self._usdt_try or 0.0)
                # Öncelik: Doğrudan USDT nakit
                cash_usdt = None
                try:
                    cash_usdt = float(summary.get('balance_usdt', 0.0) or 0.0)
                except Exception:
                    cash_usdt = None
                if cash_usdt is None:
                    cash_usdt = 0.0
                # Yedek: TRY bakiyeden türet (fx > 0 ise)
                if cash_usdt <= 0.0:
                    cash_try = float(summary.get('balance_try', 0.0) or 0.0)
                    cash_usdt = (cash_try / fx) if fx > 0 else cash_usdt
                coin_value = 0.0
                for sym, pos in (summary.get('positions') or {}).items():
                    qty = float(pos.get('qty', 0) or 0)
                    if qty <= 0:
                        continue
                    px = float(price_map.get(sym, 0) or 0)
                    if px <= 0:
                        continue
                    if sym.endswith('USDT'):
                        coin_value += qty * px
                    elif sym.endswith('TRY'):
                        coin_value += (qty * px) / (fx if fx > 0 else 1.0)
                self.total_cash_label.config(text=f"{cash_usdt:,.2f}")
                # Coin Değeri metni
                self.total_coin_label.config(text=f"{coin_value:,.2f}")
                # Renk: artış yeşil, düşüş kırmızı
                try:
                    prev_coin = getattr(self, '_prev_coin_value', None)
                    if isinstance(prev_coin, (int, float)):
                        if coin_value > prev_coin:
                            self.total_coin_label.configure(style='Positive.TLabel')
                        elif coin_value < prev_coin:
                            self.total_coin_label.configure(style='Negative.TLabel')
                        else:
                            self.total_coin_label.configure(style='Neutral.TLabel')
                except Exception:
                    pass
                self._prev_coin_value = coin_value
                # Portföy Toplamı metni ve renk
                portfolio_total = cash_usdt + coin_value
                self.total_portfolio_label.config(text=f"{portfolio_total:,.2f}")
                try:
                    prev_pt = getattr(self, '_prev_portfolio_total', None)
                    if isinstance(prev_pt, (int, float)):
                        if portfolio_total > prev_pt:
                            self.total_portfolio_label.configure(style='Positive.TLabel')
                        elif portfolio_total < prev_pt:
                            self.total_portfolio_label.configure(style='Negative.TLabel')
                        else:
                            self.total_portfolio_label.configure(style='Neutral.TLabel')
                except Exception:
                    pass
                self._prev_portfolio_total = portfolio_total
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Performans verisi hesaplama hatası: {e}")
            # Tabloları güncelle (pozisyonlar ve son işlemler)
            try:
                self._refresh_positions_table(price_map)
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Pozisyon tablosu güncelleme hatası: {e}")
            try:
                self._refresh_trades_table()
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"İşlem tablosu güncelleme hatası: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Fiyat poller başlatma hatası: {e}")

    def _reset_daily_counters(self):
        try:
            paper_executor.reset_daily_counters()
            # Etiketi hemen sıfırla
            if hasattr(self, 'daily_trades_label'):
                self.daily_trades_label.config(text="0")
            if hasattr(self, 'daily_pnl_label'):
                self.daily_pnl_label.config(text=f"{0.0:,.2f}")
            self.add_log_message("Günlük sayaçlar sıfırlandı")
        except Exception as e:
            try:
                self.logger.warning(LogCategory.GUI, f"Günlük sayaç sıfırlama hatası: {e}")
            except Exception:
                pass

    def _reset_paper_account(self):
        try:
            # 4000 USDT ile başlat: FX hazırsa TRY bakiyeyi anında ayarla, değilse USDT beklet
            fx_now = 0.0
            try:
                fx_now = float(self._usdt_try or 0.0)
            except Exception:
                fx_now = 0.0
            if fx_now > 0:
                paper_executor.reset_all(4000.0 * fx_now)
            else:
                paper_executor.reset_all(0.0)
                paper_executor.set_starting_usdt(4000.0)
            # Etiketleri ve tabloları tazele
            try:
                self.total_pnl_label.config(text=f"{0.0:,.2f}")
                self.daily_pnl_label.config(text=f"{0.0:,.2f}")
                # Nakit USDT anında yazılamayabilir; fx gelince hesaplanır. Şimdilik 4000 göster.
                self.total_cash_label.config(text=f"{4000.0:,.2f}")
                self.total_coin_label.config(text=f"{0.0:,.2f}")
                self.total_portfolio_label.config(text=f"{4000.0:,.2f}")
                if hasattr(self, 'daily_trades_label'):
                    self.daily_trades_label.config(text="0")
                if hasattr(self, 'total_trades_label'):
                    self.total_trades_label.config(text="0")
            except Exception:
                pass
            # Tabloları temizle
            try:
                price_map = dict(self._price_cache)
                if self._usdt_try and self._usdt_try > 0:
                    price_map['USDTTRY'] = self._usdt_try
                self._refresh_positions_table(price_map)
            except Exception:
                pass
            try:
                self._refresh_trades_table()
            except Exception:
                pass
            self.add_log_message("Paper hesap sıfırlandı (4000 USDT)")
        except Exception as e:
            try:
                self.logger.error(LogCategory.GUI, f"Paper reset hatası: {e}")
            except Exception:
                pass
    
    def on_closing(self):
        """Pencere kapatılırken"""
        try:
            # Bot'u durdur
            if self.is_running:
                self.stop_bot()
            
            # Güncelleme döngüsünü durdur
            self.is_running = False
            # Paper hesabı kaydet
            try:
                paper_executor.save()
            except Exception:
                pass
            
            # Uygulamayı kapat
            self.root.quit()
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Pencere kapatma hatası: {e}")
    
    def run(self):
        """Dashboard'u çalıştır"""
        try:
            # Pencere kapatma olayını bağla
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Ana döngüyü başlat
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Dashboard çalıştırma hatası: {e}")

    def create_menu_bar(self):
        """Menü çubuğu oluştur"""
        try:
            # Ana menü çubuğu
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # Strateji menüsü
            strategy_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Strateji", menu=strategy_menu)
            
            # Strateji seçimi
            strategy_menu.add_command(label="Strateji Seç", command=self.open_strategy_selector)
            strategy_menu.add_separator()
            
            # Strateji ayarları
            strategy_menu.add_command(label="Strateji Ayarları", command=self.open_strategy_settings)
            strategy_menu.add_separator()
            
            # Strateji durumu
            strategy_menu.add_command(label="Strateji Durumu", command=self.show_strategy_status)
            
            # Trading menüsü
            trading_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Trading", menu=trading_menu)
            
            # Manuel trading
            trading_menu.add_command(label="Manuel Trading", command=self.open_manual_trading)
            trading_menu.add_separator()
            
            # Order book
            trading_menu.add_command(label="Order Book", command=self.open_orderbook)
            
            # Analiz menüsü
            analysis_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Analiz", menu=analysis_menu)
            
            # Teknik analiz
            analysis_menu.add_command(label="Teknik Analiz", command=self.show_technical_analysis)
            
            # Temel analiz
            analysis_menu.add_command(label="Temel Analiz", command=self.show_fundamental_analysis)
            
            # Sistem menüsü
            system_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Sistem", menu=system_menu)
            
            # Ayarlar
            system_menu.add_command(label="Ayarlar", command=self.open_settings)
            system_menu.add_separator()
            
            # Loglar
            system_menu.add_command(label="Loglar", command=self.show_logs)
            
            # Hakkında
            system_menu.add_command(label="Hakkında", command=self.show_about)

            # Yardım menüsü
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Yardım", menu=help_menu)
            help_menu.add_command(label="Kullanım Kılavuzu", command=self.open_user_guide)

            self.logger.info(LogCategory.GUI, "Menü çubuğu oluşturuldu")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Menü çubuğu oluşturma hatası: {e}")
    
    def show_logs(self):
        """Log penceresini göster"""
        try:
            w = tk.Toplevel(self.root)
            w.title("Sistem Logları")
            w.geometry("800x400")
            frm = ttk.Frame(w)
            frm.pack(fill=tk.BOTH, expand=True)
            txt = scrolledtext.ScrolledText(frm, wrap=tk.WORD)
            txt.pack(fill=tk.BOTH, expand=True)
            try:
                content = self.log_text.get('1.0', tk.END) if hasattr(self, 'log_text') else ""
                txt.insert(tk.END, content)
                txt.config(state=tk.DISABLED)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Log penceresi hatası: {e}")

    def open_user_guide(self):
        """Kullanım kılavuzunu yeni pencerede aç"""
        try:
            w = tk.Toplevel(self.root)
            w.title("Kullanım Kılavuzu")
            w.geometry("900x700")
            w.resizable(True, True)
            frm = ttk.Frame(w); frm.pack(fill=tk.BOTH, expand=True)
            txt = scrolledtext.ScrolledText(frm, wrap=tk.WORD)
            txt.pack(fill=tk.BOTH, expand=True)
            guide = (
                "\n"
                "GENEL\n"
                "- Paper hesap kalıcıdır; açılışta yüklenir, kapanışta kaydedilir.\n"
                "- TRY pariteleri (FAN hariç) taranır. Varsayılan strateji: trend_following.\n"
                "\n"
                "OTOMATİK İŞLEM\n"
                "- Strateji Motoru: Açın (grafik ve order book etkinleşir).\n"
                "- Oto İşlem: Açın. Maks Açık Pozisyon: 5/10/15/20.\n"
                "- Eşikler: AI Analiz bölümünden Al/Sat yüzdelerini ayarlayın.\n"
                "- Emir boyutu: ~150 TL/sinyal (fiyata göre miktar). SL/TP: %1.5.\n"
                "\n"
                "MANUEL İŞLEM\n"
                "- Menü > Trading > Manuel Trading: Emir formundan market/limit/stop.\n"
                "- Emirler paper hesaba kalıcı işlenir ve tablolar/loglar güncellenir.\n"
                "\n"
                "STRATEJİLER\n"
                "- scalping: Çok kısa vadeli mikro hareketlerden küçük kar; hızlı yürütme, dar spread.\n"
                "- grid: Belirli bantta eşit aralıklı al/sat; yatay piyasada verimli.\n"
                "- trend_following: Kırılım/ortalama üstünde AL, altında SAT; trendde iyi, yatayda whipsaw riski.\n"
                "- dca: Belirli periyotlarda sabit tutarla kademeli alım; zamanlama riskini azaltır.\n"
                "- hedge: Pozisyonu karşıt pozisyonla korur; volatiliteyi düşürür, maliyet ekler.\n"
                "- pairs_trading: Korelasyonlu iki varlığın relatif sapmalarını işlem konusu yapar; piyasa-nötr.\n"
                "- cointegration: Eşbütünleşik çiftlerde mean-reversion; istatistiksel ilişki bozulma riski.\n"
                "- lead_lag: Önden hareket eden varlıktan sinyal alır; rejim değişiminde performans düşebilir.\n"
                "- dynamic_selector: Piyasa rejimine göre uygun stratejiyi otomatik seçer/ağırlıklandırır.\n"
                "\n"
                "YENİ STRATEJİLER\n"
                "- Volatility Breakout: Günlük/4s volatilite patlamalarını (ATR/Bollinger genişlemesi) yakalar; ATR tabanlı SL ve ATR bazlı dinamik pozisyon boyutu ile uygulanır; trend başlangıcını yakalama potansiyeli, yanlış breakout'ta whipsaw riski.\n"
                "- Mean Reversion (Bollinger/RSI): Fiyat Bollinger dışına taşarken ve RSI aşırı bölgede iken ters yönde pozisyon; yatay/dalgalı piyasalarda iyi çalışır; güçlü trend dönemlerinde risk artar; Grid/DCA ile uyumlu.\n"
                "- Momentum Ignition/Acceleration: Kısa sürede artan hacim ve fiyat momentumu (ör. 5 sn hacim sıçraması) tespiti; scalping-benzeri hızlı giriş/çıkış; volatility-scalping hibriti ile birleştirilebilir.\n"
                "- Reversal Confirmation: Düşüş trendinde RSI divergence + MA crossover kombinasyonu ile dönüş sinyali; 'dip al/tepe sat' yaklaşımı; volatil dönemde iyi risk-getiri; DCA'nın sinyal-doğrulamalı versiyonu.\n"
                "- Adaptive Volatility (Meta): Volatilite rejimine göre strateji değiştirir (düşük vol → grid, yüksek vol → scalping); DynamicStrategySelector ile entegre edilebilir; ATR benzeri ölçülerle sinyal tabanlı geçiş.\n"
                "- Correlation & Altseason Rotation: BTC dominansı düşerken altcoin rotasyonlarını yakalar; korelasyon/dominans verileri (CoinGecko/Glassnode) ile hedge stratejisine entegre; daha uzun zaman aralığı gerektirir.\n"
                "- AI Sentiment Fusion (Opsiyonel): On-chain/sosyal medya sentiment + teknik sinyallerin birleşimi; external_data_manager ve market_analyzer entegrasyonlarıyla; haber odaklı dönemlerde faydalı.\n"
                "\n"
                "ÖNERİLEN KOMBİNASYONLAR (Volatil BTC/Altcoin piyasaları)\n"
                "- Volatil (yönsüz): Scalping + Volatility Breakout | Not: Kısa süreli trade’ler\n"
                "- Sideways: Grid + Mean Reversion | Not: Kademeli emirlerle al/sat\n"
                "- Trend (yukarı/aşağı): Trend Following + Reversal | Not: EMA + ADX filtreli\n"
                "- Crash / Panic: DCA + Hedge | Not: Kademeli güvenli alım\n"
                "- Regime Değişimi: Dynamic Selector | Not: Otomatik strateji geçişi\n"
                "\n"
                "PANELLER\n"
                "- Açık Pozisyonlar: P&L pozitif yeşil, negatif kırmızı.\n"
                "- Son İşlemler: Son 10 işlem.\n"
                "- Performans: Toplam/Günlük P&L, Portföy (renkli).\n"
                "- Fiyat Grafiği: Klines yoksa canlı fiyat geçmişiyle çizilir.\n"
                "\n"
                "PARAMETRELER\n"
                "- Eşikler: Yükseliş ≥ +Al% -> AL; Düşüş ≤ −Sat% -> SAT.\n"
                "- Cooldown: sembol başına ~30 sn (tekrarlı işlemi önler).\n"
                "- SL/TP: %1.5 (paper_executor).\n"
                "\n"
                "SORUN GİDERME\n"
                "- Tablo boşsa 1-2 dk bekleyin; Strateji Motoru ve Oto İşlem açık olsun.\n"
                "- Çok işlem varsa eşikleri yükseltin ve cooldown’ı artırın.\n"
            )
            try:
                txt.insert(tk.END, guide)
                txt.config(state=tk.DISABLED)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Kılavuz penceresi hatası: {e}")
    
    def open_strategy_selector(self):
        """Gelişmiş strateji seçici penceresini aç"""
        try:
            # Yeni pencere oluştur
            strategy_window = tk.Toplevel(self.root)
            strategy_window.title("Gelişmiş Strateji Yönetimi")
            strategy_window.geometry("1000x700")
            strategy_window.resizable(True, True)
            
            # Ana frame
            main_frame = ttk.Frame(strategy_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Gelişmiş strateji seçiciyi oluştur
            def on_strategy_changed(strategy_name: str, action: str):
                """Strateji değişikliği callback'i"""
                self.logger.info(LogCategory.GUI, f"Strateji değişikliği: {strategy_name} - {action}")
                # Motor kapalıysa seçilen stratejiyi etkin bırakma
                if not getattr(self, '_engine_enabled', False):
                    try:
                        strategy_manager.stop_all_strategies()
                        self.add_log_message("Strateji Motoru KAPALI: Stratejiler durduruldu (Manuel mod)")
                    except Exception:
                        pass
                # Ana dashboard'u güncelle
                self._update_strategy_status()
            
            strategy_selector = AdvancedStrategySelector(main_frame, strategy_manager, on_strategy_changed)
            
            self.logger.info(LogCategory.GUI, "Gelişmiş strateji seçici açıldı")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Gelişmiş strateji seçici açma hatası: {e}")
            messagebox.showerror("Hata", f"Strateji seçici açılamadı: {e}")
    
    def activate_strategy(self, strategy_name: str):
        """Stratejiyi aktif et"""
        try:
            if not strategy_name:
                messagebox.showwarning("Uyarı", "Lütfen bir strateji seçin")
                return
            
            # Stratejiyi aktif et
            success = strategy_manager.activate_strategy(strategy_name)
            
            if success:
                messagebox.showinfo("Başarılı", f"Strateji '{strategy_name}' aktif edildi")
                self.logger.info(LogCategory.GUI, f"Strateji aktif edildi: {strategy_name}")
            else:
                messagebox.showerror("Hata", f"Strateji '{strategy_name}' aktif edilemedi")
                
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji aktif etme hatası: {e}")
            messagebox.showerror("Hata", f"Strateji aktif edilemedi: {e}")
    
    def stop_current_strategy(self):
        """Mevcut stratejiyi durdur"""
        try:
            # Aktif stratejiyi bul
            active_strategies = strategy_manager.get_active_strategies()
            
            if not active_strategies:
                messagebox.showinfo("Bilgi", "Aktif strateji yok")
                return
            
            # İlk aktif stratejiyi durdur
            strategy_name = active_strategies[0] if isinstance(active_strategies, list) else list(active_strategies.keys())[0]
            success = strategy_manager.deactivate_strategy(strategy_name)
            
            if success:
                messagebox.showinfo("Başarılı", f"Strateji '{strategy_name}' durduruldu")
                self.logger.info(LogCategory.GUI, f"Strateji durduruldu: {strategy_name}")
            else:
                messagebox.showerror("Hata", f"Strateji '{strategy_name}' durdurulamadı")
                
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji durdurma hatası: {e}")
            messagebox.showerror("Hata", f"Strateji durdurulamadı: {e}")
    
    def open_strategy_settings(self):
        """Strateji ayarları penceresini aç"""
        try:
            # Yeni pencere oluştur
            settings_window = tk.Toplevel(self.root)
            settings_window.title("Strateji Ayarları")
            settings_window.geometry("800x600")
            settings_window.resizable(True, True)
            
            # Ana frame
            main_frame = ttk.Frame(settings_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Strateji ayarları GUI'sini oluştur
            strategy_settings = StrategySettingsGUI(main_frame, strategy_manager)
            
            self.logger.info(LogCategory.GUI, "Strateji ayarları açıldı")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji ayarları açma hatası: {e}")
            messagebox.showerror("Hata", f"Strateji ayarları açılamadı: {e}")
    
    def open_manual_trading(self):
        """Manuel trading penceresini aç"""
        try:
            # Yeni pencere oluştur
            trading_window = tk.Toplevel(self.root)
            trading_window.title("Manuel Trading")
            trading_window.geometry("600x500")
            trading_window.resizable(True, True)
            
            # Ana frame
            main_frame = ttk.Frame(trading_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Manuel trading GUI'sini oluştur
            def place_order_callback(symbol, side, order_type, price, quantity):
                # Manuel emri paper hesabına uygula (kalıcı)
                try:
                    px = float(price or 0.0)
                    qty = float(quantity or 0.0)
                    if px > 0 and qty > 0:
                        signals = [{
                            'symbol': (symbol or '').upper(),
                            'side': side.lower(),
                            'entry_price': px,
                            'size': qty,
                            'strategy_name': 'manual'
                        }]
                        paper_executor.execute(signals)
                        # SL/TP kurallarını da bir kez çalıştır
                        price_map = { (symbol or '').upper(): px }
                        if self._usdt_try and self._usdt_try > 0:
                            price_map['USDTTRY'] = self._usdt_try
                        paper_executor.tick(price_map)
                        paper_executor.save()
                        try:
                            self.add_log_message(f"Manuel emir uygulandı (paper): {symbol} {side} {qty} @ {px}")
                        except Exception:
                            pass
                    else:
                        messagebox.showerror("Hata", "Geçersiz fiyat/miktar")
                        return
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Manuel emir yürütme hatası: {e}")
                # Kullanıcı bildirimi
                messagebox.showinfo("Emir", f"Emir gönderildi: {symbol} {side} {quantity} @ {price}")
            
            manual_trading = ManualTradingGUI(main_frame, place_order_callback)
            manual_trading.create_gui()
            
            self.logger.info(LogCategory.GUI, "Manuel trading açıldı")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Manuel trading açma hatası: {e}")
            messagebox.showerror("Hata", f"Manuel trading açılamadı: {e}")
    
    def open_orderbook(self):
        """Order book penceresini aç"""
        try:
            # Yeni pencere oluştur
            orderbook_window = tk.Toplevel(self.root)
            orderbook_window.title("Emir Defteri Analizi")
            orderbook_window.geometry("1200x800")
            orderbook_window.resizable(True, True)
            
            # Ana frame
            main_frame = ttk.Frame(orderbook_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # OrderBookGUI sınıfını kullan
            orderbook_gui = OrderBookGUI(main_frame)
            orderbook_gui.create_gui()
            try:
                orderbook_gui.refresh_data()
            except Exception:
                try:
                    orderbook_gui.load_sample_data()
                except Exception:
                    pass
            
            # Order book analyzer'ı başlat ve bağla
            from indicators.orderbook_analyzer import OrderBookAnalyzer
            analyzer = OrderBookAnalyzer()
            
            # Callback fonksiyonları - OrderBookGUI ile entegrasyon
            def on_imbalance_detected(imbalance_data):
                try:
                    # OrderBookGUI'deki metrikleri güncelle
                    if hasattr(orderbook_gui, 'update_imbalance_metrics'):
                        orderbook_gui.update_imbalance_metrics(imbalance_data)
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Imbalance callback hatası: {e}")
            
            def on_spoofing_detected(spoofing_data):
                try:
                    # OrderBookGUI'deki spoofing uyarısını güncelle
                    if hasattr(orderbook_gui, 'update_spoofing_alert'):
                        orderbook_gui.update_spoofing_alert(spoofing_data)
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Sahte emir callback hatası: {e}")
            
            def on_whale_detected(whale_data):
                try:
                    # OrderBookGUI'deki whale uyarısını güncelle
                    if hasattr(orderbook_gui, 'update_whale_alert'):
                        orderbook_gui.update_whale_alert(whale_data)
                except Exception as e:
                    self.logger.error(LogCategory.GUI, f"Balina callback hatası: {e}")
            
            # Callback'leri kaydet
            analyzer.add_imbalance_callback(on_imbalance_detected)
            analyzer.add_spoofing_callback(on_spoofing_detected)
            analyzer.add_whale_callback(on_whale_detected)
            
            # BTCTURK WebSocket canlı veri akışı
            try:
                from api.websocket_client import WebSocketClient, WebSocketMessage
                from indicators.orderbook_analyzer import OrderBookSnapshot, OrderBookLevel
                import json
                
                ws_client = WebSocketClient({
                    'symbols': ['BTCTRY'],
                    'url': 'wss://ws-feed.btcturk.com',
                    'reconnect_interval': 5,
                    'max_reconnect_attempts': 50,
                })
                
                def _parse_orderbook(msg_data):
                    try:
                        symbol = msg_data.get('symbol') or msg_data.get('S') or 'BTCTRY'
                        ts = _dtmod.datetime.now()
                        bids_raw = msg_data.get('bids') or msg_data.get('b') or []
                        asks_raw = msg_data.get('asks') or msg_data.get('a') or []
                        bids = []
                        asks = []
                        # Beklenen format: [[price, quantity], ...] veya list of dict
                        for it in bids_raw[:20]:
                            if isinstance(it, (list, tuple)) and len(it) >= 2:
                                price, qty = float(it[0]), float(it[1])
                            else:
                                price = float(it.get('price') or it.get('p') or 0)
                                qty = float(it.get('quantity') or it.get('q') or 0)
                            bids.append(OrderBookLevel(price, qty, ts))
                        for it in asks_raw[:20]:
                            if isinstance(it, (list, tuple)) and len(it) >= 2:
                                price, qty = float(it[0]), float(it[1])
                            else:
                                price = float(it.get('price') or it.get('p') or 0)
                                qty = float(it.get('quantity') or it.get('q') or 0)
                            asks.append(OrderBookLevel(price, qty, ts))
                        best_bid = max((l.price for l in bids), default=0.0)
                        best_ask = min((l.price for l in asks), default=0.0)
                        spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0
                        mid = (best_ask + best_bid)/2 if (best_bid and best_ask) else 0.0
                        return OrderBookSnapshot(
                            symbol=symbol,
                            timestamp=ts,
                            bids=bids,
                            asks=asks,
                            best_bid=best_bid,
                            best_ask=best_ask,
                            spread=spread,
                            mid_price=mid,
                        )
                    except Exception as e:
                        self.logger.warning(LogCategory.GUI, f"Orderbook parse hatası: {e}")
                        return None
                
                def on_ws_message(ws_msg: WebSocketMessage):
                    try:
                        if ws_msg.type.lower() not in ('orderbook', 'ob', 'depth'):
                            return
                        snapshot = _parse_orderbook(ws_msg.data)
                        if snapshot is None:
                            return
                        analyzer.add_orderbook_snapshot(snapshot)
                        if hasattr(orderbook_gui, 'update_orderbook_data'):
                            orderbook_gui.update_orderbook_data(snapshot)
                    except Exception as e:
                        self.logger.error(LogCategory.GUI, f"WS mesaj işleme hatası: {e}")
                
                ws_client.add_message_callback(on_ws_message)
                ws_client.start()
                
                # Pencere kapatılırken WS'i durdur
                def _on_close():
                    try:
                        ws_client.stop()
                    except Exception:
                        pass
                    orderbook_window.destroy()
                orderbook_window.protocol("WM_DELETE_WINDOW", _on_close)
            except Exception as e:
                self.logger.error(LogCategory.GUI, f"BTCTURK WS başlatma hatası: {e}")
            
            self.logger.info(LogCategory.GUI, "Emir defteri analizi açıldı (OrderBookGUI ile)")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Emir defteri açma hatası: {e}")
            messagebox.showerror("Hata", f"Emir defteri açılamadı: {e}")
    
    def show_strategy_status(self):
        """Strateji durumunu göster"""
        try:
            names = list(strategy_manager.get_available_strategies() or [])
            status_text = "Strateji Durumu:\n\n"
            for name in names:
                try:
                    info = strategy_manager.get_strategy_info(name) or {}
                except Exception:
                    info = {}
                is_active = info.get('is_active', False)
                stype = info.get('type', 'Bilinmiyor')
                perf = info.get('performance', {})
                status_text += f"• {name}: {'Aktif' if is_active else 'Pasif'}\n"
                status_text += f"  Tip: {stype}\n"
                status_text += f"  Performans: {perf}\n\n"
            
            messagebox.showinfo("Strateji Durumu", status_text)
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji durumu gösterme hatası: {e}")
            messagebox.showerror("Hata", f"Strateji durumu gösterilemedi: {e}")
    
    def show_technical_analysis(self):
        try:
            w = tk.Toplevel(self.root)
            w.title("Teknik Analiz")
            w.geometry("800x600")
            w.resizable(True, True)
            frm = ttk.Frame(w)
            frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            cols = ("Sembol", "Fiyat", "Sinyal Gücü")
            tree = ttk.Treeview(frm, columns=cols, show="headings")
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=120, anchor="center")
            tree.pack(fill=tk.BOTH, expand=True)
            try:
                price_map = dict(self._price_cache)
                for sym, px in price_map.items():
                    strength = 0.0
                    try:
                        strength = float(px) % 1
                    except Exception:
                        strength = 0.0
                    tree.insert("", tk.END, values=(sym, f"{float(px):,.4f}", f"{strength:.2f}"))
            except Exception as e:
                self.logger.warning(LogCategory.GUI, f"Teknik analiz verisi hazırlanamadı: {e}")
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Teknik analiz penceresi hatası: {e}")
            messagebox.showerror("Hata", f"Teknik analiz açılamadı: {e}")

    def show_fundamental_analysis(self):
        try:
            w = tk.Toplevel(self.root)
            w.title("Temel Analiz")
            w.geometry("700x500")
            w.resizable(True, True)
            frm = ttk.Frame(w)
            frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            stats = ttk.LabelFrame(frm, text="Genel Performans")
            stats.pack(fill=tk.X, pady=(0,10))
            try:
                overall = strategy_manager.get_overall_performance()
            except Exception as e:
                overall = {}
                self.logger.warning(LogCategory.GUI, f"Genel performans alınamadı: {e}")
            fields = [
                ("Toplam İşlem", int(overall.get("total_trades", 0) or 0)),
                ("Toplam PnL", f"{float(overall.get('total_profit', 0.0) or 0.0):,.2f}"),
                ("Sinyal Sayısı", int(overall.get("total_signals", 0) or 0)),
                ("Başarı Oranı", f"{float(overall.get('success_rate', 0.0) or 0.0):.2f}%"),
                ("Aktif Strateji", int(overall.get("active_strategies", 0) or 0)),
                ("Toplam Strateji", int(overall.get("total_strategies", 0) or 0)),
            ]
            grid = ttk.Frame(stats)
            grid.pack(fill=tk.X)
            for i, (k, v) in enumerate(fields):
                ttk.Label(grid, text=k+":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                ttk.Label(grid, text=str(v), font=("Arial", 10, "bold")).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            info = ttk.LabelFrame(frm, text="Notlar")
            info.pack(fill=tk.BOTH, expand=True)
            txt = tk.Text(info, wrap=tk.WORD)
            txt.pack(fill=tk.BOTH, expand=True)
            txt.insert(tk.END, "Temel analiz veri kaynağı bağlı değilse, bu panel performans ve durum özetini gösterir. ")
            txt.config(state=tk.DISABLED)
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Temel analiz penceresi hatası: {e}")
            messagebox.showerror("Hata", f"Temel analiz açılamadı: {e}")

    def show_about(self):
        """Hakkında göster"""
        about_text = """BTCTURK Trading Bot v2.0
            
    Gelişmiş kripto para trading botu
    - 5 farklı trading stratejisi
    - AI tabanlı piyasa analizi
    - Gerçek zamanlı WebSocket verisi
    - Kapsamlı risk yönetimi
    - Manuel trading desteği
    - RSS haber entegrasyonu
    - Gelişmiş strateji yönetimi

    Geliştirici: AI Assistant
    Tarih: 2024"""
        
        messagebox.showinfo("Hakkında", about_text)
    
    def _update_strategy_status(self):
        try:
            # Aktif stratejileri al ve AI panelindeki etiketleri güncelle
            names = []
            try:
                names = list(strategy_manager.get_active_strategies() or [])
            except Exception:
                names = []
            label_text = ", ".join(names) if names else "Manuel"
            if hasattr(self, 'auto_strategy_label'):
                try:
                    self.auto_strategy_label.config(text=label_text)
                except Exception:
                    pass
            # Motor kapalıysa sinyali NÖTR göster
            if hasattr(self, 'strategy_signal_label'):
                try:
                    if not getattr(self, '_engine_enabled', False):
                        self.strategy_signal_label.config(text="NÖTR", style='Neutral.TLabel')
                except Exception:
                    pass
            # Strateji notunu güncelle (bilgi amaçlı)
            if hasattr(self, 'strategy_info_label'):
                try:
                    if not getattr(self, '_engine_enabled', False):
                        self.strategy_info_label.config(text="Manuel mod")
                    else:
                        self.strategy_info_label.config(text="Oto mod")
                except Exception:
                    pass
            
            # Üst panel market_status_label ve AI panelindeki market_regime_label'ı güncelle
            try:
                r = getattr(strategy_manager, '_regime_name', 'unknown')
                r_disp = {
                    'trend': 'Trend',
                    'yatay': 'Yatay',
                    'volatil': 'Volatil',
                    'çöküş': 'Çöküş',
                    # Bilinmeyen/boş rejimler için varsayılan 'Yatay'
                    'unknown': 'Yatay',
                    'none': 'Yatay',
                    '': 'Yatay'
                }.get(str(r).lower(), 'Yatay')
                sym = (self._current_symbol or 'BTCUSDT').strip().upper()
                if hasattr(self, 'market_status_label'):
                    try:
                        self.market_status_label.config(text=f"Piyasa: {sym} ({r_disp})")
                    except Exception:
                        pass
                if hasattr(self, 'market_regime_label'):
                    try:
                        self.market_regime_label.config(text=r_disp)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Strateji durumu güncelleme hatası: {e}")
    
    def create_orderbook_analysis_panel(self, parent):
        """Order book analizi paneli oluştur"""
        try:
            # Order book analizi frame
            ob_frame = ttk.LabelFrame(parent, text="Order Book Analizi", padding=10)
            ob_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Analiz metrikleri
            metrics_frame = ttk.Frame(ob_frame)
            metrics_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Sol kolon - Temel metrikler
            left_metrics = ttk.Frame(metrics_frame)
            left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            ttk.Label(left_metrics, text="Temel Metrikler", font=("Arial", 10, "bold")).pack(anchor=tk.W)
            
            self.depth_imbalance_label = ttk.Label(left_metrics, text="Derinlik Dengesizliği: 0.00")
            self.depth_imbalance_label.pack(anchor=tk.W, pady=2)
            
            self.order_flow_label = ttk.Label(left_metrics, text="Emir Akışı: 0.00")
            self.order_flow_label.pack(anchor=tk.W, pady=2)
            
            self.liquidity_score_label = ttk.Label(left_metrics, text="Likidite Skoru: 0.00")
            self.liquidity_score_label.pack(anchor=tk.W, pady=2)
            
            self.market_impact_label = ttk.Label(left_metrics, text="Piyasa Etkisi: 0.00")
            self.market_impact_label.pack(anchor=tk.W, pady=2)
            
            # Sağ kolon - Gelişmiş metrikler
            right_metrics = ttk.Frame(metrics_frame)
            right_metrics.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            ttk.Label(right_metrics, text="Gelişmiş Metrikler", font=("Arial", 10, "bold")).pack(anchor=tk.W)
            
            self.spoofing_detected_label = ttk.Label(right_metrics, text="Sahte Emir (Spoofing): Hayır", foreground="green")
            self.spoofing_detected_label.pack(anchor=tk.W, pady=2)
            
            self.whale_activity_label = ttk.Label(right_metrics, text="Büyük Oyuncu Aktivitesi: Hayır", foreground="green")
            self.whale_activity_label.pack(anchor=tk.W, pady=2)
            
            self.support_resistance_label = ttk.Label(right_metrics, text="Destek/Direnç Seviyeleri: 0")
            self.support_resistance_label.pack(anchor=tk.W, pady=2)
            
            self.volatility_estimate_label = ttk.Label(right_metrics, text="Volatilite: 0.00%")
            self.volatility_estimate_label.pack(anchor=tk.W, pady=2)
            
            # Uyarı paneli
            warning_frame = ttk.LabelFrame(ob_frame, text="Uyarılar", padding=5)
            warning_frame.pack(fill=tk.X)
            
            self.ob_warning_text = tk.Text(warning_frame, height=3, width=60, state=tk.DISABLED)
            self.ob_warning_text.pack(fill=tk.X)
            
            # Order book analizi başlat
            self._start_orderbook_analysis()
            
            self.logger.info(LogCategory.GUI, "Order book analizi paneli oluşturuldu")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Order book analizi paneli oluşturma hatası: {e}")
    
    def _start_orderbook_analysis(self):
        """Order book analizini başlat"""
        try:
            from indicators.orderbook_analyzer import OrderBookAnalyzer
            
            # Order book analizörünü başlat
            self.orderbook_analyzer = OrderBookAnalyzer()
            
            # Callback'leri ekle
            self.orderbook_analyzer.add_imbalance_callback(self._on_imbalance_detected)
            self.orderbook_analyzer.add_spoofing_callback(self._on_spoofing_detected)
            # Poller başlat
            self._ob_poll_running = True
            def _ob_worker():
                while getattr(self, '_ob_poll_running', False):
                    try:
                        sym = (self._current_symbol or 'BTCUSDT').strip().upper()
                        bids = []
                        asks = []
                        bb = ba = mp = 0.0
                        try:
                            # Basit API denemesi
                            resp = self.multi_api_manager.make_request('GET', '/orderbook', params={'pairSymbol': sym, 'limit': 20})
                            data = resp.get('data', resp if isinstance(resp, dict) else {})
                            b = data.get('bids') or []
                            a = data.get('asks') or []
                            ts = _dtmod.datetime.now()
                            for pr, qty in (b[:20] if isinstance(b, list) else []):
                                bids.append(OBLevel(float(pr), float(qty), ts))
                            for pr, qty in (a[:20] if isinstance(a, list) else []):
                                asks.append(OBLevel(float(pr), float(qty), ts))
                            if bids and asks:
                                bb = float(bids[0].price)
                                ba = float(asks[0].price)
                                mp = (bb + ba) / 2.0
                        except Exception:
                            # Fallback: fiyat cache'ten sentetik OB üret
                            px = float(self._get_last_price_for_symbol(sym) or 0.0)
                            if px > 0:
                                ts = _dtmod.datetime.now()
                                for i in range(1, 11):
                                    asks.append(OBLevel(px + i * (px*0.0005), 1.0/(i), ts))
                                    bids.append(OBLevel(px - i * (px*0.0005), 1.0/(i), ts))
                                bb = bids[0].price
                                ba = asks[0].price
                                mp = (bb + ba) / 2.0
                        if bids or asks:
                            snap = OBSnapshot(symbol=sym, timestamp=_dtmod.datetime.now(), bids=bids, asks=asks,
                                              best_bid=bb, best_ask=ba, spread=(ba-bb if ba and bb else 0.0), mid_price=mp)
                            try:
                                self.orderbook_analyzer.add_orderbook_snapshot(snap)
                            except Exception:
                                pass
                            # Etiketleri güncelle
                            try:
                                self.root.after(0, self._update_orderbook_labels)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(5)
            threading.Thread(target=_ob_worker, daemon=True).start()
            
            self.logger.info(LogCategory.GUI, "Order book analizi başlatıldı")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Order book analizi başlatma hatası: {e}")
    
    def _on_imbalance_detected(self, analysis: Dict[str, Any]):
        """Depth imbalance tespit edildiğinde"""
        try:
            imbalance = analysis.get('imbalance_ratio', 0.0)
            try:
                if hasattr(self, 'depth_imbalance_label') and self.depth_imbalance_label.winfo_exists():
                    self.depth_imbalance_label.config(text=f"Depth Imbalance: {imbalance:.3f}")
            except Exception:
                pass
            
            # Uyarı kontrolü
            if abs(imbalance) > 0.7:
                warning_text = f"Yüksek depth imbalance tespit edildi: {imbalance:.3f}"
                self._add_orderbook_warning(warning_text, "high")
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Imbalance callback hatası: {e}")
    
    def _on_spoofing_detected(self, analysis: Dict[str, Any]):
        """Spoofing tespit edildiğinde"""
        try:
            detected = analysis.get('detected', False)
            patterns = analysis.get('patterns', [])
            
            if detected:
                try:
                    if hasattr(self, 'spoofing_detected_label') and self.spoofing_detected_label.winfo_exists():
                        self.spoofing_detected_label.config(text="Spoofing: Evet", foreground="red")
                except Exception:
                    pass
                # patterns listesinde dict'ler var; okunabilir metne çevir
                names = []
                for p in patterns:
                    try:
                        t = p.get('type') if isinstance(p, dict) else p
                        t = t.value if hasattr(t, 'value') else t
                        names.append(str(t))
                    except Exception:
                        names.append('pattern')
                warning_text = f"Spoofing tespit edildi: {', '.join(names) if names else '-'}"
                self._add_orderbook_warning(warning_text, "critical")
            else:
                try:
                    if hasattr(self, 'spoofing_detected_label') and self.spoofing_detected_label.winfo_exists():
                        self.spoofing_detected_label.config(text="Spoofing: Hayır", foreground="green")
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Spoofing callback hatası: {e}")
    
    def _add_orderbook_warning(self, message: str, level: str = "info"):
        """Order book uyarısı ekle"""
        try:
            timestamp = _dtmod.datetime.now().strftime("%H:%M:%S")
            color = "red" if level == "critical" else "orange" if level == "high" else "black"
            
            self.ob_warning_text.config(state=tk.NORMAL)
            self.ob_warning_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.ob_warning_text.config(state=tk.DISABLED)
            
            # Son 10 satırı tut
            lines = self.ob_warning_text.get("1.0", tk.END).split('\n')
            if len(lines) > 10:
                self.ob_warning_text.config(state=tk.NORMAL)
                self.ob_warning_text.delete("1.0", f"{len(lines)-10}.0")
                self.ob_warning_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(LogCategory.GUI, f"Order book uyarı ekleme hatası: {e}")




