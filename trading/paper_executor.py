import threading
import datetime as _dtmod
import json
import os
from pathlib import Path
from utils.logger import get_logger, LogCategory
from utils.metrics import metrics
from utils.trade_journal import journal

logger = get_logger("paper_executor")


class PaperTradeExecutor:
    """Basit kağıt (paper) işlem yürütücüsü. Çoklu sembol destekler, kalıcılık sağlar."""

    def __init__(self):
        self._lock = threading.Lock()
        # Başlangıç bakiyesi: 100,000 TRY
        self.balance_try = 100_000.0
        # USDT bazlı muhasebe bakiyesi (asıl referans)
        self.balance_usdt = 0.0
        # Çoklu sembol pozisyonları: { symbol: { qty, avg_cost, peak, partial_done } }
        self.positions = {}
        self.realized_pnl = 0.0
        self.total_pnl = 0.0
        self.trades = []  # list of dicts
        # Sayaçlar
        self.total_trade_count = 0
        self.daily_trade_count = 0
        self.daily_realized_pnl = 0.0
        self.last_reset_date = _dtmod.datetime.now().date()
        # Günlük trade limiti (overtrading engelleme)
        # 0 veya None ise limit devre dışı bırakılır
        self.max_daily_trades: int = 0
        # Risk parametreleri (tüm semboller için ortak)
        self.sl_pct = 0.01       # %1 stop-loss
        self.tp_pct = 0.02       # %2 take-profit (trailing kapalıysa)
        self.trailing_pct = 0.01 # %1 trailing drawdown tetikleyicisi
        # Komisyon oranları (Binance kalibrasyon)
        self.fee_rate = 0.0010  # backward-compat (taker varsay)
        self.fee_rate_taker = 0.0010
        self.fee_rate_maker = 0.0010
        # Minimum alış tutarı (TRY karşılığı)
        self.min_notional_try = 100.0
        # Minimum alış tutarı (USDT karşılığı) - Binance paper için
        # 0.001 USDT üzerindeki tüm işlemlere izin ver
        self.min_notional_usdt = 0.001
        # Çok küçük notional işlemler için otomatik büyütme (USDT)
        # notional_min_boost_usdt <= cost_usdt < notional_max_boost aralığında, hedef notional_boost_usdt'e ölçekle
        self.notional_min_boost_usdt = 0.001
        self.notional_max_boost_usdt = 1.0
        self.notional_boost_target_usdt = 150.0
        # Kalıcılık
        self._store_path = None
        # FX oranları
        self.usdt_try = 0.0  # 1 USDT kaç TRY
        # Başlangıç bakiyesini USDT ile ayarlamak için bekleyen değer (geri uyum)
        self._pending_usdt_init: float | None = None
        # Son SL olayları (stop-scaling için)
        self._recent_sl = {}  # {symbol: {'ts': datetime, 'count': int}}
        # Alternating-loss guard ve buy kilidi
        self._loss_events: dict[str, list[_dtmod.datetime]] = {}
        self._buy_lock_until: dict[str, _dtmod.datetime] = {}
        # Maker/Taker ve maliyet istatistikleri
        self._maker_count = 0
        self._taker_count = 0
        self._fee_dist = []
        self._slip_dist = []
        self._spread_dist = []
        # Son fiyat referansları (tick'den)
        self._last_prices: dict[str, float] = {}
        # Kümülatif ücret izleme (USDT)
        self.cumulative_fees_usdt: float = 0.0
        # Dedup ve cooldown korumaları
        self.dedupe_window_sec: float = 5.0
        self.symbol_cooldown_sec: float = 30.0
        self._last_order_key_time: dict[tuple, _dtmod.datetime] = {}
        self._last_buy_time: dict[str, _dtmod.datetime] = {}
        # Metrikler
        self._m_sig2ord = metrics.histogram('signal_to_order_ms')

    def _daily_reset_if_needed(self):
        today = _dtmod.datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.daily_realized_pnl = 0.0
            self.last_reset_date = today

    def reset_daily_counters(self):
        with self._lock:
            self.daily_trade_count = 0
            self.daily_realized_pnl = 0.0
            self.last_reset_date = _dtmod.datetime.now().date()

    def _record_trade(self, trade: dict):
        """Append trade (keeps only last 200 in self.trades) and increment counters."""
        self._daily_reset_if_needed()
        self.total_trade_count += 1
        try:
            if str(trade.get('side', '')).lower() == 'buy' and bool(trade.get('is_open', False)):
                # Günlük sayaç sadece yeni pozisyon açılışlarında artar
                self.daily_trade_count += 1
        except Exception:
            pass
        self.trades.append(trade)
        if len(self.trades) > 200:
            self.trades = self.trades[-200:]

    def execute(self, signals):
        if not signals:
            return
        with self._lock:
            self._daily_reset_if_needed()
            for s in signals:
                # Kaynak filtresi: yalnızca belirli origin'lerden gelen sinyalleri işle
                try:
                    src = str(s.get("source") or "").lower()
                except Exception:
                    src = ""
                allowed_sources = {"orchestrator", "orchestrator_notional", "manual_gui"}
                if allowed_sources and src not in allowed_sources:
                    try:
                        logger.info(
                            LogCategory.SYSTEM,
                            f"Sinyal atlandı (izin verilmeyen kaynak): source={src or 'none'}, symbol={s.get('symbol')}",
                        )
                        journal.log_signal_skipped(
                            strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                            symbol=str(s.get('symbol', 'unknown') or 'unknown'),
                            side=str(s.get('side', 'unknown') or 'unknown'),
                            reason_code="invalid_source",
                            details={"source": src},
                        )
                    except Exception:
                        pass
                    continue
                symbol = s.get('symbol', 'BTCTRY')
                side = s.get('side', 'buy')
                price_sig = float(s.get('entry_price', 0) or 0.0)
                size = float(s.get('size', 0) or 0.0)
                # signal_to_order_ms
                try:
                    ts_sig = s.get('ts') or s.get('timestamp')
                    if ts_sig:
                        if isinstance(ts_sig, (int, float)):
                            t0 = _dtmod.datetime.fromtimestamp(ts_sig/1000.0) if ts_sig > 1e12 else _dtmod.datetime.fromtimestamp(ts_sig)
                        elif isinstance(ts_sig, str):
                            t0 = _dtmod.datetime.fromisoformat(ts_sig)
                        else:
                            t0 = None
                        if t0 is not None:
                            dt_ms = ( _dtmod.datetime.now() - t0 ).total_seconds()*1000.0
                            self._m_sig2ord.observe(dt_ms)
                except Exception:
                    pass
                # Günlük trade limiti (yalnızca yeni ALIM pozisyonları için)
                # max_daily_trades > 0 ise aktif, 0/None ise devre dışı
                try:
                    mdt = self.max_daily_trades
                    if mdt is not None and int(mdt) > 0:
                        if str(side).lower() == 'buy' and self.daily_trade_count >= int(mdt):
                            logger.info(LogCategory.SYSTEM, f"ALIM atlandı: günlük trade limiti aşıldı ({self.daily_trade_count}/{mdt})")
                            try:
                                journal.log_signal_skipped(
                                    strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                    symbol=str(symbol),
                                    side=str(side),
                                    reason_code="daily_trade_limit",
                                    details={"daily_trade_count": self.daily_trade_count, "max_daily_trades": int(mdt)},
                                )
                            except Exception:
                                pass
                            continue
                except Exception:
                    pass

                # Deduplication for same (symbol, side)
                try:
                    key = (str(symbol).upper(), str(side).lower())
                    now = _dtmod.datetime.now()
                    last_t = self._last_order_key_time.get(key)
                    if last_t and (now - last_t).total_seconds() < float(self.dedupe_window_sec or 0.0):
                        logger.info(LogCategory.SYSTEM, f"Sinyal reddedildi (dedupe penceresi): {key}")
                        try:
                            journal.log_signal_skipped(
                                strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                symbol=str(symbol),
                                side=str(side),
                                reason_code="dedupe_window",
                                details={"window_sec": float(self.dedupe_window_sec or 0.0)},
                            )
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass
                # Yalnız USDT paritelerde işlem yap
                if not str(symbol).upper().endswith('USDT'):
                    logger.info(LogCategory.SYSTEM, f"Sinyal atlandı (yalnız USDT işlenir): {symbol}")
                    try:
                        journal.log_signal_skipped(
                            strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                            symbol=str(symbol),
                            side=str(side),
                            reason_code="non_usdt_symbol",
                            details={},
                        )
                    except Exception:
                        pass
                    continue
                # Fiyat referansı: tick'ten gelen son fiyatı kullan, yoksa sinyal fiyata düş
                px_ref = float(self._last_prices.get(symbol, 0.0) or 0.0)
                if px_ref <= 0:
                    px_ref = price_sig
                # Güvenlik: Satışta fiyat 0 ise işleme girme
                if str(side).lower() == 'sell' and px_ref <= 0:
                    try:
                        logger.warning(LogCategory.SYSTEM, f"SATIŞ atlandı: {symbol} fiyat bilgisi yok (px=0)")
                    except Exception:
                        pass
                    continue
                if px_ref <= 0 or size <= 0:
                    continue
                qty = size  # istenen base miktarı
                # Sembol bazlı cooldown (BUY)
                try:
                    if str(side).lower() == 'buy':
                        lb = self._last_buy_time.get(symbol)
                        if lb and (_dtmod.datetime.now() - lb).total_seconds() < float(self.symbol_cooldown_sec or 0.0):
                            logger.info(LogCategory.SYSTEM, f"ALIM cooldown aktif: {symbol}")
                            try:
                                remaining = float(self.symbol_cooldown_sec or 0.0) - (_dtmod.datetime.now() - lb).total_seconds()
                            except Exception:
                                remaining = None
                            try:
                                journal.log_signal_skipped(
                                    strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                    symbol=str(symbol),
                                    side=str(side),
                                    reason_code="buy_cooldown",
                                    details={"cooldown_sec": float(self.symbol_cooldown_sec or 0.0), "remaining_sec": remaining},
                                )
                            except Exception:
                                pass
                            continue
                except Exception:
                    pass
                # Alternating-loss guard: kilit kontrolü
                try:
                    until = self._buy_lock_until.get(symbol)
                    if side == 'buy' and until and _dtmod.datetime.now() < until:
                        logger.info(LogCategory.SYSTEM, f"ALIM kilitli (alternating-loss guard): {symbol}")
                        try:
                            journal.log_signal_skipped(
                                strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                symbol=str(symbol),
                                side=str(side),
                                reason_code="buy_locked_after_losses",
                                details={"lock_until": until.isoformat()},
                            )
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass
                # Maliyet ve min-notional: kote para birimine göre TRY'ye çevir
                quote = 'TRY' if symbol.endswith('TRY') else ('USDT' if symbol.endswith('USDT') else '')
                fx = float(self.usdt_try or 0.0)
                # Etkin min-notional USDT
                effective_min_usdt = float(self.min_notional_usdt or 0.0)
                # Maliyet USDT (başlangıç)
                if quote == 'TRY':
                    if fx <= 0:
                        logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} için USDTTRY (fx) bilinmiyor")
                        continue
                    cost_usdt = (qty * px_ref) / fx
                elif quote == 'USDT':
                    cost_usdt = qty * px_ref
                else:
                    # Desteklenmeyen kote: güvenlik için atla
                    logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} desteklenmeyen kote")
                    try:
                        journal.log_signal_skipped(
                            strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                            symbol=str(symbol),
                            side=str(side),
                            reason_code="unsupported_quote",
                            details={"quote": quote},
                        )
                    except Exception:
                        pass
                    continue
                pos = self.positions.get(symbol, {"qty": 0.0, "avg_cost": 0.0, "peak": 0.0, "partial_done": False, "state": "open", "opened_at": None, "t_snapshots": {}})
                # Ceza (penalty) uygula
                penalty_mult = 1.0
                rs = self._recent_sl.get(symbol)
                if rs:
                    try:
                        if (_dtmod.datetime.now() - rs['ts']).total_seconds() < 1800:
                            cnt = int(rs.get('count', 1))
                            base_pen = 0.5
                            penalty_mult = max(0.3, base_pen ** cnt)
                            mlp = float(s.get('ml_success_prob', 0.0) or 0.0)
                            if mlp >= 0.75:
                                penalty_mult = max(penalty_mult, 0.8)
                    except Exception:
                        pass
                if penalty_mult < 1.0:
                    qty = qty * penalty_mult
                    # yeniden maliyet USDT
                    if quote == 'TRY':
                        cost_usdt = (qty * px_ref) / fx
                    else:
                        cost_usdt = qty * px_ref
                # Küçük notional işlemleri (0.001-1 USDT) belirlenen hedefe otomatik büyüt
                try:
                    if str(side).lower() == 'buy':
                        nb_min = float(getattr(self, 'notional_min_boost_usdt', 0.001) or 0.0)
                        nb_max = float(getattr(self, 'notional_max_boost_usdt', 1.0) or 0.0)
                        nb_target = float(getattr(self, 'notional_boost_target_usdt', 500.0) or 0.0)
                        if nb_target > 0 and cost_usdt >= nb_min and cost_usdt < nb_max:
                            unit_cost_usdt = (px_ref / fx) if quote == 'TRY' else px_ref
                            if unit_cost_usdt > 0:
                                qty_boost = nb_target / unit_cost_usdt
                                qty_old = qty
                                qty = max(qty, qty_boost)
                                cost_usdt = qty * unit_cost_usdt
                                try:
                                    logger.info(LogCategory.SYSTEM, f"Küçük notional boost: {symbol} {qty_old:.6f} -> {qty:.6f} (hedef ~{nb_target:.3f} USDT)")
                                except Exception:
                                    pass
                except Exception:
                    pass
                # Eğer min-notional altındaysa ve ALIM ise, min_notional_usdt eşiğini karşılayacak şekilde miktarı otomatik artırmayı dene
                if side == 'buy' and cost_usdt < effective_min_usdt:
                    try:
                        unit_cost_usdt = (px_ref / fx) if quote == 'TRY' else px_ref
                        if unit_cost_usdt > 0:
                            qty_min = effective_min_usdt / unit_cost_usdt
                            qty_old = qty
                            qty = max(qty, qty_min)
                            # maliyeti güncelle
                            cost_usdt = qty * unit_cost_usdt
                            logger.info(LogCategory.SYSTEM, f"Min notional için miktar yükseltildi: {symbol} {qty_old:.6f} -> {qty:.6f}")
                    except Exception:
                        pass
                # Bakiye kırpması (USDT): toplam USDT = cost + fee
                # Maker/Taker oranı seçimi ipucu
                try:
                    is_maker = bool(s.get('is_maker', False)) or (str(s.get('liquidity','taker')).lower() == 'maker')
                except Exception:
                    is_maker = False
                fee_rate_eff = float(s.get('fee_rate', (self.fee_rate_maker if is_maker else self.fee_rate_taker)) or (self.fee_rate_maker if is_maker else self.fee_rate_taker))
                # Pozisyon boyutunu sinyalden gelen ipucu ile ayarla (varsa)
                try:
                    if str(side).lower() == 'buy':
                        size_hint = float(s.get('position_size_hint', 0.0) or 0.0)
                        if size_hint > 0:
                            # Hedef notional: bakiyenin belirli bir yüzdesi
                            # 0 < size_hint <= 1 varsayımıyla sınırla
                            size_hint_clamped = max(0.0, min(1.0, size_hint))
                            target_usdt = float(self.balance_usdt or 0.0) * size_hint_clamped
                            # Hedef notional çok küçükse mevcut cost_usdt ile devam et
                            if target_usdt >= effective_min_usdt and px_ref > 0:
                                if quote == 'TRY':
                                    unit_usdt = (px_ref / fx) if fx > 0 else 0.0
                                else:
                                    unit_usdt = px_ref
                                if unit_usdt > 0:
                                    qty = target_usdt / unit_usdt
                                    cost_usdt = target_usdt
                except Exception:
                    pass
                fee_usdt = cost_usdt * fee_rate_eff
                total_usdt = cost_usdt + fee_usdt
                # Fee-aware giriş filtresi: potansiyel kâr, fee'yi anlamlı şekilde aşmıyorsa ALIM yapma
                try:
                    if str(side).lower() == 'buy':
                        tp = float(s.get('take_profit', 0.0) or 0.0)
                        potential_usdt = None
                        if tp > 0 and tp > px_ref:
                            # USDT paritelerde tahmini potansiyel PnL
                            if quote == 'USDT':
                                potential_usdt = (tp - px_ref) * qty
                            elif quote == 'TRY' and fx > 0:
                                potential_usdt = ((tp - px_ref) * qty) / fx
                        # Eşik: min 0.5 USDT ve en az 2x fee
                        if potential_usdt is not None:
                            min_edge = max(0.5, 2.0 * float(fee_usdt or 0.0))
                            if potential_usdt < min_edge:
                                logger.info(LogCategory.SYSTEM, f"ALIM atlandı: fee-aware edge yetersiz (pot={potential_usdt:.4f} USDT < min_edge={min_edge:.4f} USDT)")
                                try:
                                    journal.log_signal_skipped(
                                        strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                        symbol=str(symbol),
                                        side=str(side),
                                        reason_code="fee_edge_too_low",
                                        details={"potential_usdt": potential_usdt, "min_edge_usdt": min_edge, "fee_usdt": float(fee_usdt or 0.0)},
                                    )
                                except Exception:
                                    pass
                                continue
                except Exception:
                    pass
                # Bakiye yetersizse ALIM yapma (miktarı küçültmek yerine sinyali atla)
                if total_usdt > self.balance_usdt and total_usdt > 0 and str(side).lower() == 'buy':
                    try:
                        logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} bakiye yetersiz (gerekli={total_usdt:.4f} USDT, mevcut={self.balance_usdt:.4f} USDT)")
                        journal.log_signal_skipped(
                            strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                            symbol=str(symbol),
                            side=str(side),
                            reason_code="balance_insufficient",
                            details={"required_usdt": float(total_usdt or 0.0), "available_usdt": float(self.balance_usdt or 0.0)},
                        )
                    except Exception:
                        pass
                    continue
                # Nihai min-notional doğrulaması (USDT tabanlı)
                if cost_usdt < effective_min_usdt:
                    # Bilgilendirici atlama logu (USDT cinsinden)
                    try:
                        usdt_notional = cost_usdt
                        logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} notional {usdt_notional:.4f} USDT < {float(self.min_notional_usdt or 0.0):.2f} USDT")
                        try:
                            journal.log_signal_skipped(
                                strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                symbol=str(symbol),
                                side=str(side),
                                reason_code="min_notional_usdt",
                                details={"notional_usdt": usdt_notional, "min_usdt": float(self.min_notional_usdt or 0.0)},
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                    continue

                # Tek sembol maruziyet tavanı (<= %25)
                try:
                    # Portföy USDT tahmini: nakit + tüm pozisyonların USDT karşılığı
                    portfolio_usdt = float(self.balance_usdt or 0.0)
                    for sy, ps in (self.positions or {}).items():
                        px_last = float(self._last_prices.get(sy, 0.0) or 0.0)
                        if ps.get('qty', 0.0) > 0 and px_last > 0:
                            if sy.endswith('USDT'):
                                portfolio_usdt += ps['qty'] * px_last
                            elif sy.endswith('TRY') and fx > 0:
                                portfolio_usdt += (ps['qty'] * px_last) / fx
                    # Bu alım sonrası sembol maruziyeti
                    sym_value_after = (pos.get('qty', 0.0) + qty) * px_ref
                    cap = 0.25 * portfolio_usdt
                    if sym_value_after > cap and px_ref > 0:
                        qty_cap = max(0.0, (cap / px_ref) - pos.get('qty', 0.0))
                        if qty_cap <= 0:
                            logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} sembol maruziyet tavanı %25")
                            try:
                                journal.log_signal_skipped(
                                    strategy_name=str(s.get('strategy_name', 'unknown') or 'unknown'),
                                    symbol=str(symbol),
                                    side=str(side),
                                    reason_code="exposure_cap_25pct",
                                    details={"portfolio_usdt": portfolio_usdt},
                                )
                            except Exception:
                                pass
                            continue
                        else:
                            logger.info(LogCategory.SYSTEM, f"Maruziyet tavanı ile miktar düşürüldü: {qty:.6f} -> {qty_cap:.6f}")
                            qty = qty_cap
                            if quote == 'TRY':
                                cost_usdt = (qty * px_ref) / fx
                            else:
                                cost_usdt = qty * px_ref
                            fee_usdt = cost_usdt * fee_rate_eff
                            total_usdt = cost_usdt + fee_usdt
                except Exception:
                    pass

                if side == 'buy' and self.balance_usdt >= total_usdt:
                    # Aynı sembolde açık pozisyon varken tekrar alım yapma (tek pozisyon kuralı)
                    if pos.get("qty", 0.0) > 0:
                        logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} için zaten açık pozisyon var")
                        continue
                    # Komisyonu kote taraftan tahsil et (qty korunur)
                    qty_filled = max(0.0, qty)
                    was_zero = (pos["qty"] <= 0)
                    new_qty = pos["qty"] + qty_filled
                    if new_qty > 0:
                        pos["avg_cost"] = (
                            pos["avg_cost"] * pos["qty"] + qty_filled * px_ref
                        ) / new_qty
                    pos["qty"] = new_qty
                    pos["partial_done"] = False if pos["qty"] > 0 else pos.get("partial_done", False)
                    # USDT bakiyeden toplam (maliyet+komisyon) düş
                    self.balance_usdt -= total_usdt
                    # Ücretleri kümülatif ve PnL'e yansıt (alış ücreti)
                    try:
                        self.cumulative_fees_usdt += float(fee_usdt or 0.0)
                        self.realized_pnl -= float(fee_usdt or 0.0)
                    except Exception:
                        pass
                    if px_ref > (pos.get("peak") or 0.0):
                        pos["peak"] = px_ref
                    # Pozisyon state init
                    try:
                        pos.setdefault('state', 'open')
                        pos['opened_at'] = _dtmod.datetime.now()
                        # ATR tabanlı SL/TP/time-stop (basit RR=1:2 kuralı)
                        atr_pct = float(s.get('atr_pct', 0.0) or 0.0)
                        strat_name = str(s.get('strategy_name', '') or '').lower()

                        # Varsayılan baz SL katsayıları
                        base_min_sl = 0.01   # en az %1
                        k_atr = 2.0          # SL ≈ 2x ATR

                        # Mean reversion: biraz daha dar, hızlı dönüş bekler
                        if 'mean_reversion' in strat_name:
                            base_min_sl = 0.008
                            k_atr = 1.5
                        # Momentum: spike'larda stop'a çok çabuk düşmemek için daha geniş
                        elif 'momentum_ignition' in strat_name:
                            base_min_sl = 0.012
                            k_atr = 2.5

                        if atr_pct > 0:
                            sl_dyn = max(base_min_sl, k_atr * atr_pct)
                        else:
                            sl_dyn = max(base_min_sl, float(self.sl_pct or 0.0))

                        # Risk/Ödül oranı 1:2 → TP yüzdesi SL'in 2 katı
                        tp_dyn = 2.0 * sl_dyn

                        # Time-stop: volatiliteye göre kaba ayrım
                        if atr_pct >= 0.01:
                            time_stop = 1800  # 30 dk
                        elif atr_pct >= 0.005:
                            time_stop = 2400  # 40 dk
                        else:
                            time_stop = 3000  # 50 dk

                        pos['sl_pct_eff'] = sl_dyn
                        pos['tp_pct_eff'] = tp_dyn
                        pos['time_stop_sec'] = time_stop
                        pos.setdefault('t_snapshots', {})
                    except Exception:
                        pass
                    self.positions[symbol] = pos
                    # Zaman damgalarını güncelle (dedupe/cooldown)
                    try:
                        self._last_order_key_time[(str(symbol).upper(), 'buy')] = _dtmod.datetime.now()
                        self._last_buy_time[symbol] = _dtmod.datetime.now()
                    except Exception:
                        pass
                    self._record_trade({
                        'entry_time': _dtmod.datetime.now(),
                        'symbol': symbol,
                        'side': side,
                        'size': qty_filled,
                        'entry_price': px_ref,
                        'net_pnl': 0.0,
                        'fee': fee_usdt,
                        'strategy': s.get('strategy_name', 'example'),
                        'is_open': True if was_zero and qty_filled > 0 else False
                    })
                # Dedup zaman damgası (sell için de kaydet)
                try:
                    self._last_order_key_time[(str(symbol).upper(), str(side).lower())] = _dtmod.datetime.now()
                except Exception:
                    pass
                    # Taker sayacı ve dağılımlar
                try:
                    self._taker_count += 1
                    mid = float(s.get('mid', 0.0) or 0.0)
                    if mid > 0:
                        slp = (px_ref - mid) / mid
                        self._slip_dist.append(slp)
                        self._spread_dist.append(float(s.get('spread_pct', 0.0) or 0.0))
                    self._fee_dist.append(fee_rate_eff)
                except Exception:
                    pass
                continue
                if side == 'buy' and self.balance_usdt < total_usdt:
                    try:
                        logger.info(LogCategory.SYSTEM, f"ALIM atlandı: {symbol} bakiye yetersiz (gerekli={total_usdt:.4f} USDT, mevcut={self.balance_usdt:.4f} USDT)")
                    except Exception:
                        pass
                    continue
                elif side == 'sell' and pos["qty"] > 0:
                    # Toz engeli: kalan değer min altına düşecekse tam kapat
                    sell_qty = min(qty, pos["qty"]) if qty > 0 else pos["qty"]
                    remain_qty = pos["qty"] - sell_qty
                    remain_value_try = (remain_qty * px_ref) if quote == 'TRY' else (remain_qty * px_ref * fx)
                    if 0 < remain_value_try < effective_min_try:
                        sell_qty = pos["qty"]
                        remain_qty = 0.0
                    realized_quote = (px_ref - pos["avg_cost"]) * sell_qty
                    # Realized P&L'yi USDT'ye çevir
                    if quote == 'USDT':
                        realized_usdt = realized_quote
                    elif quote == 'TRY':
                        realized_usdt = (realized_quote / (fx if fx > 0 else 1.0)) if fx > 0 else 0.0
                    else:
                        realized_usdt = 0.0
                    # Satış geliri USDT
                    fee_usdt = 0.0
                    if quote == 'TRY':
                        if fx > 0:
                            gross_usdt = (sell_qty * px_ref) / fx
                            fee_usdt = gross_usdt * self.fee_rate
                            self.balance_usdt += (gross_usdt - fee_usdt)
                    elif quote == 'USDT':
                        gross_usdt = sell_qty * px_ref
                        fee_usdt = gross_usdt * self.fee_rate
                        self.balance_usdt += (gross_usdt - fee_usdt)
                    # Net realized PnL ve kümülatif fee
                    try:
                        self.realized_pnl += float(realized_usdt - fee_usdt)
                        self.cumulative_fees_usdt += float(fee_usdt or 0.0)
                    except Exception:
                        pass
                    pos["qty"] -= sell_qty
                    if pos["qty"] == 0:
                        pos["avg_cost"] = 0.0
                        pos["peak"] = 0.0
                    self.positions[symbol] = pos
                else:
                    # Satış mümkün değil veya koşullar sağlanmadı
                    logger.info(LogCategory.SYSTEM, f"SATIŞ atlandı: {symbol} miktar yetersiz veya koşul sağlanmadı")
                    continue
                self._record_trade({
                    'entry_time': _dtmod.datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'size': sell_qty if side == 'sell' else qty,
                    'entry_price': px_ref,
                    'net_pnl': 0.0,
                    'fee': fee_usdt if side == 'sell' else 0.0,
                    'strategy': s.get('strategy_name', 'example'),
                    'is_open': False
                })

    def submit_order_notional(
        self,
        symbol: str,
        side: str,
        notional: float,
        price: float,
        strategy_name: str = "orchestrator",
    ) -> dict:
        """USDT notional -> miktar çevirip mevcut execute() pipeline'ını kullanan yardımcı metot."""
        try:
            if notional is None or price is None:
                return {"status": "rejected", "reason": "notional_or_price_none"}
            notional = float(notional)
            price = float(price)
            if notional <= 0 or price <= 0:
                return {"status": "rejected", "reason": "invalid_notional_or_price"}
        except Exception as e:
            return {"status": "rejected", "reason": f"cast_error: {e}"}

        qty = notional / price
        try:
            qty = float(f"{qty:.6f}")
        except Exception:
            pass
        if qty <= 0:
            return {"status": "rejected", "reason": "qty_non_positive"}

        signal = {
            "symbol": symbol,
            "side": side,
            "size": qty,
            "entry_price": price,
            "strategy_name": strategy_name,
            "source": "orchestrator_notional",
        }

        try:
            self.execute([signal])
            return {
                "status": "accepted",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "notional": notional,
                "price": price,
            }
        except Exception as e:
            try:
                logger.error(LogCategory.SYSTEM, f"submit_order_notional hata: {e}")
            except Exception:
                pass
            return {"status": "error", "reason": str(e)}

    def tick(self, current_prices):
        """Risk kurallarını uygula. current_prices: {symbol: price} veya tek float (geriye dönük uyum)."""
        with self._lock:
            # Eğer USDT ile başlangıç bakiyesi ayarlanmak istenmiş ve FX geldiyse uygula (geri uyum)
            try:
                if self._pending_usdt_init is not None:
                    self.balance_usdt = float(self._pending_usdt_init)
                    self._pending_usdt_init = None
            except Exception:
                pass
            if isinstance(current_prices, (int, float)):
                # Geriye dönük uyum: sadece BTCTRY varsay
                current_prices = {'BTCTRY': float(current_prices)}
            # FX güncelle (USDTTRY verilmişse)
            try:
                fx = float(current_prices.get('USDTTRY', 0) or 0)
                if fx > 0:
                    self.usdt_try = fx
                    # TRY bakiyeyi aynalama (sadece görüntü için)
                    self.balance_try = self.balance_usdt * fx
            except Exception as e:
                logger.warning(LogCategory.SYSTEM, f"FX güncelleme hatası: {e}")
            # Son fiyat cache'ini güncelle
            try:
                for sym, px in (current_prices or {}).items():
                    if px and px > 0:
                        self._last_prices[sym] = float(px)
            except Exception:
                pass
            for symbol, price in (current_prices or {}).items():
                if not price or price <= 0:
                    continue
                pos = self.positions.get(symbol)
                if not pos or pos["qty"] <= 0 or pos["avg_cost"] <= 0:
                    continue
                # Peak güncelle
                if price > (pos.get("peak") or 0.0):
                    pos["peak"] = price
                # Etkin SL/TP
                sl_pct_eff = float(pos.get('sl_pct_eff', self.sl_pct) or self.sl_pct)
                tp_pct_eff = float(pos.get('tp_pct_eff', self.tp_pct) or self.tp_pct)
                sl_level = pos["avg_cost"] * (1.0 - sl_pct_eff)
                if price <= sl_level:
                    self._close_all(symbol, price, reason='auto_sl')
                    continue
                # BE + trailing durum makinesi ve scale-out
                try:
                    pnl_pct = (price - pos['avg_cost']) / pos['avg_cost']
                except Exception:
                    pnl_pct = 0.0
                # State transitions
                st = pos.get('state', 'open')
                if st == 'open' and pnl_pct >= 0.01:
                    pos['state'] = 'armed'
                if pos.get('state') == 'armed':
                    # BE stop: yalnızca anlamlı kâr sonrası trailing'e geçiş ve partial çıkış
                    if pnl_pct >= 0.02:
                        pos['state'] = 'trailing'
                        # scale-out %50
                        try:
                            if pos['qty'] > 0:
                                self._sell_partial(symbol, pos['qty'] * 0.5, price, reason='auto_scaleout_be')
                        except Exception:
                            pass
                    # Eski be_stop mantığı agresif olduğu için devre dışı bırakıldı
                # Time exit: daha uzun süre ve anlamlı kâr eşiği ile realize
                try:
                    opened_at = pos.get('opened_at')
                    t_stop = float(pos.get('time_stop_sec', 0) or 0.0)
                    if isinstance(opened_at, _dtmod.datetime) and t_stop > 0:
                        elapsed = ( _dtmod.datetime.now() - opened_at).total_seconds()
                        if elapsed >= t_stop and pnl_pct >= 0.003:
                            self._close_all(symbol, price, reason='time_exit_extended')
                            continue
                except Exception:
                    pass
                # Trailing: peak’ten ≥0.8% geri çekilmede çıkış (kâr eşiği sonrası)
                tp_level = pos["avg_cost"] * (1.0 + tp_pct_eff)
                if self.trailing_pct <= 0 and price >= tp_level:
                    self._close_all(symbol, price, reason='auto_tp')
                    continue
                peak = pos.get("peak") or 0.0
                if pos.get('state') == 'trailing' and peak > 0:
                    drawdown = (peak - price) / peak
                    if drawdown >= 0.02:
                        self._close_all(symbol, price, reason='auto_trailing')
                        continue
                # T+ PnL snapshotlar (30/60/120s)
                try:
                    tsn = pos.setdefault('t_snapshots', {})
                    opened_at = pos.get('opened_at')
                    if isinstance(opened_at, _dtmod.datetime):
                        elapsed = ( _dtmod.datetime.now() - opened_at).total_seconds()
                        for tmark in (30, 60, 120):
                            key = f'T+{tmark}'
                            if elapsed >= tmark and key not in tsn:
                                tsn[key] = float(pnl_pct)
                                try:
                                    logger.info(LogCategory.SYSTEM, f"{symbol} {key} PnL: {pnl_pct*100:.3f}%")
                                except Exception:
                                    pass
                    pos['t_snapshots'] = tsn
                except Exception:
                    pass
                self.positions[symbol] = pos

    def _close_all(self, symbol: str, price: float, reason: str = 'auto'):
        """Belirli sembolde tüm pozisyonu kapat (market)."""
        pos = self.positions.get(symbol)
        if not pos or pos["qty"] <= 0:
            return
        qty = pos["qty"]
        avg_cost = float(pos.get("avg_cost", 0.0) or 0.0)
        # Kapanış geliri USDT olarak hesapla (komisyon dahil)
        fx = float(self.usdt_try or 0.0)
        if symbol.endswith('USDT'):
            gross_usdt = qty * price
        elif symbol.endswith('TRY'):
            gross_usdt = (qty * price) / fx if fx > 0 else 0.0
        else:
            gross_usdt = 0.0
        fee_usdt = gross_usdt * self.fee_rate
        net_usdt = gross_usdt - fee_usdt
        realized_quote = (price - avg_cost) * qty
        # USDT biriminde realized
        if symbol.endswith('USDT'):
            realized_usdt = realized_quote
        elif symbol.endswith('TRY'):
            realized_usdt = (realized_quote / float(self.usdt_try or 0.0)) if float(self.usdt_try or 0.0) > 0 else 0.0
        else:
            realized_usdt = 0.0
        self.realized_pnl += realized_usdt
        # USDT ana bakiyeyi artır
        self.balance_usdt += max(0.0, net_usdt)
        # Günlük realize P&L ve elde tutma süresi için bilgi hazırla
        hold_seconds = None
        try:
            opened_at = pos.get('opened_at')
            if isinstance(opened_at, _dtmod.datetime):
                hold_seconds = (_dtmod.datetime.now() - opened_at).total_seconds()
        except Exception:
            hold_seconds = None
        pos["qty"] = 0.0
        pos["avg_cost"] = 0.0
        pos["peak"] = 0.0
        pos["partial_done"] = False
        self.positions[symbol] = pos
        self._record_trade({
            'entry_time': _dtmod.datetime.now(),
            'symbol': symbol,
            'side': 'sell',
            'size': qty,
            'entry_price': price,
            'net_pnl': realized_usdt,
            'strategy': reason,
            'is_open': False
        })
        try:
            journal.log_trade_close(
                symbol=symbol,
                side='sell',
                qty=qty,
                entry_price=float(avg_cost),
                exit_price=float(price or 0.0),
                realized_pnl_usdt=float(realized_usdt or 0.0),
                hold_seconds=hold_seconds,
                close_reason=str(reason or 'auto'),
                strategy_name=None if reason is None else str(reason),
                meta={
                    'fee_usdt': float(fee_usdt or 0.0),
                },
            )
        except Exception:
            pass
        # Günlük realize P&L
        self._daily_reset_if_needed()
        self.daily_realized_pnl += realized_usdt
        # Stop-scaling: SL durumunu kaydet
        # Alternating-loss guard için loss kaydı
        if realized_usdt < 0:
            try:
                arr = self._loss_events.get(symbol, [])
                arr.append(_dtmod.datetime.now())
                # son 6 saat ile sınırlı tut
                lim = _dtmod.datetime.now() - _dtmod.timedelta(hours=6)
                arr = [x for x in arr if x >= lim]
                self._loss_events[symbol] = arr
                # Son 90 dakikada >=2 loss ise 3 saat kilit
                lim90 = _dtmod.datetime.now() - _dtmod.timedelta(minutes=90)
                recent_losses = [x for x in arr if x >= lim90]
                if len(recent_losses) >= 2:
                    self._buy_lock_until[symbol] = _dtmod.datetime.now() + _dtmod.timedelta(hours=3)
            except Exception:
                pass
        if reason == 'auto_sl':
            rs = self._recent_sl.get(symbol, {'count': 0})
            rs['count'] = int(rs.get('count', 0)) + 1
            rs['ts'] = _dtmod.datetime.now()
            self._recent_sl[symbol] = rs

    def _sell_partial(self, symbol: str, qty: float, price: float, reason: str = 'auto_partial'):
        """Pozisyonun bir kısmını kapat ve bakiyeyi güncelle (komisyon dahil)."""
        pos = self.positions.get(symbol)
        if not pos or qty <= 0 or pos["qty"] <= 0:
            return
        qty = min(qty, pos["qty"])
        # USDT bazında gelir ve komisyon
        fx = float(self.usdt_try or 0.0)
        if symbol.endswith('USDT'):
            gross_usdt = qty * price
        elif symbol.endswith('TRY'):
            gross_usdt = (qty * price) / fx if fx > 0 else 0.0
        else:
            gross_usdt = 0.0
        fee_usdt = gross_usdt * self.fee_rate
        self.balance_usdt += max(0.0, (gross_usdt - fee_usdt))
        realized_quote = (price - pos["avg_cost"]) * qty
        if symbol.endswith('USDT'):
            realized_usdt = realized_quote
        elif symbol.endswith('TRY'):
            realized_usdt = (realized_quote / float(self.usdt_try or 0.0)) if float(self.usdt_try or 0.0) > 0 else 0.0
        else:
            realized_usdt = 0.0
        self.realized_pnl += realized_usdt
        self.daily_realized_pnl += realized_usdt
        pos["qty"] -= qty
        # avg_cost aynı kalır; partial durumunu işaretle
        pos["partial_done"] = True
        self.positions[symbol] = pos
        self._record_trade({
            'entry_time': _dtmod.datetime.now(),
            'symbol': symbol,
            'side': 'sell',
            'size': qty,
            'entry_price': price,
            'net_pnl': realized_usdt,
            'strategy': reason
        })
        try:
            hold_seconds = None
            opened_at = pos.get('opened_at')
            if isinstance(opened_at, _dtmod.datetime):
                hold_seconds = (_dtmod.datetime.now() - opened_at).total_seconds()
        except Exception:
            hold_seconds = None
        try:
            journal.log_trade_close(
                symbol=symbol,
                side='sell',
                qty=qty,
                entry_price=float(pos.get('avg_cost', 0.0) or 0.0),
                exit_price=float(price or 0.0),
                realized_pnl_usdt=float(realized_usdt or 0.0),
                hold_seconds=hold_seconds,
                close_reason=str(reason or 'auto_partial'),
                strategy_name=None if reason is None else str(reason),
                meta={
                    'fee_usdt': float(fee_usdt or 0.0),
                    'partial': True,
                },
            )
        except Exception:
            pass
        self._daily_reset_if_needed()

    def get_summary(self, current_prices: dict | float | None) -> dict:
        """Toplam PnL özeti. current_prices dict verilirse mark-to-market hesaplanır."""
        with self._lock:
            unrealized = 0.0
            if isinstance(current_prices, dict):
                for sym, pos in self.positions.items():
                    px = float(current_prices.get(sym, 0) or 0)
                    if pos["qty"] > 0 and pos["avg_cost"] > 0 and px > 0:
                        delta = (px - pos["avg_cost"]) * pos["qty"]
                        if sym.endswith('USDT'):
                            unrealized += delta
                        elif sym.endswith('TRY'):
                            fx = float(self.usdt_try or 0.0)
                            unrealized += (delta / fx) if fx > 0 else 0.0
                        else:
                            # desteklenmeyen: katkı 0
                            pass
            self.total_pnl = self.realized_pnl + unrealized
            return {
                'balance_try': self.balance_try,
                'balance_usdt': self.balance_usdt,
                'positions': self.positions,
                'unrealized_pnl': unrealized,
                'realized_pnl': self.realized_pnl,
                'total_pnl': self.total_pnl,
                'total_trade_count': self.total_trade_count,
                'daily_trade_count': self.daily_trade_count,
                'daily_realized_pnl': self.daily_realized_pnl,
                'cumulative_fees_usdt': self.cumulative_fees_usdt
            }

    # Kalıcılık
    def set_store_path(self, path: str):
        self._store_path = path

    def save(self):
        try:
            path = self._ensure_store_path()
            # Pozisyonları JSON için serileştir
            try:
                positions_ser = {}
                for sym, pos in (self.positions or {}).items():
                    if not isinstance(pos, dict):
                        continue
                    p = dict(pos)
                    try:
                        if isinstance(p.get('opened_at'), _dtmod.datetime):
                            p['opened_at'] = p['opened_at'].isoformat()
                    except Exception:
                        pass
                    positions_ser[sym] = p
            except Exception:
                positions_ser = self.positions
            data = {
                'balance_try': self.balance_try,
                'balance_usdt': self.balance_usdt,
                'positions': positions_ser,
                'realized_pnl': self.realized_pnl,
                'cumulative_fees_usdt': getattr(self, 'cumulative_fees_usdt', 0.0),
                'total_trade_count': self.total_trade_count,
                'daily_trade_count': self.daily_trade_count,
                'daily_realized_pnl': self.daily_realized_pnl,
                'last_reset_date': self.last_reset_date.isoformat() if self.last_reset_date else None,
                'trades': [
                    {
                        'entry_time': t['entry_time'].isoformat() if isinstance(t.get('entry_time'), _dtmod.datetime) else t.get('entry_time'),
                        'symbol': t.get('symbol'),
                        'side': t.get('side'),
                        'size': t.get('size'),
                        'entry_price': t.get('entry_price'),
                        'net_pnl': t.get('net_pnl'),
                        'strategy': t.get('strategy')
                    } for t in self.trades
                ]
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.error(LogCategory.SYSTEM, f"Paper hesap kaydetme hatası: {e}")

    def load(self):
        try:
            path = self._ensure_store_path()
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.balance_try = float(data.get('balance_try', self.balance_try))
            # balance_usdt varsa doğrudan yükle; yoksa fx'ten türet
            try:
                if 'balance_usdt' in data:
                    self.balance_usdt = float(data.get('balance_usdt', 0.0) or 0.0)
                else:
                    fx = float(self.usdt_try or 0.0)
                    self.balance_usdt = (self.balance_try / fx) if fx > 0 else self.balance_usdt
            except Exception:
                pass
            # Pozisyonları yükle ve opened_at'i datetime'a çevir
            self.positions = data.get('positions', {})
            try:
                for sym, pos in list((self.positions or {}).items()):
                    if isinstance(pos, dict) and isinstance(pos.get('opened_at'), str):
                        try:
                            pos['opened_at'] = _dtmod.datetime.fromisoformat(pos['opened_at'])
                            self.positions[sym] = pos
                        except Exception:
                            pass
            except Exception:
                pass
            self.realized_pnl = float(data.get('realized_pnl', 0.0))
            try:
                self.cumulative_fees_usdt = float(data.get('cumulative_fees_usdt', 0.0) or 0.0)
            except Exception:
                pass
            self.total_trade_count = int(data.get('total_trade_count', 0) or 0)
            self.daily_trade_count = int(data.get('daily_trade_count', 0) or 0)
            self.daily_realized_pnl = float(data.get('daily_realized_pnl', 0.0) or 0.0)
            try:
                lrd = data.get('last_reset_date')
                self.last_reset_date = _dtmod.datetime.fromisoformat(lrd).date() if lrd else _dtmod.datetime.now().date()
            except Exception:
                self.last_reset_date = _dtmod.datetime.now().date()
            self.trades = []
            for t in data.get('trades', []):
                try:
                    ts = t.get('entry_time')
                    dt = _dtmod.datetime.fromisoformat(ts) if isinstance(ts, str) else _dtmod.datetime.now()
                except Exception as e:
                    logger.warning(LogCategory.SYSTEM, f"Trade zamanı parse hatası (ts={t.get('entry_time')}): {e}")
                    dt = _dtmod.datetime.now()
                self.trades.append({
                    'entry_time': dt,
                    'symbol': t.get('symbol'),
                    'side': t.get('side'),
                    'size': float(t.get('size', 0.0)),
                    'entry_price': float(t.get('entry_price', 0.0)),
                    'net_pnl': float(t.get('net_pnl', 0.0)),
                    'strategy': t.get('strategy')
                })
            # USDT odak: sadece USDT sembolleri tut
            try:
                self.positions = {k: v for k, v in (self.positions or {}).items() if str(k).upper().endswith('USDT')}
                self.trades = [t for t in self.trades if str(t.get('symbol','')).upper().endswith('USDT')]
            except Exception:
                pass
            # Eğer tamamen boş bir hesap yüklenmişse (USDT bakiyesi yok, pozisyon ve trade yok),
            # küçük USDT paritelerinde dahi işlem açılabilmesi için makul bir başlangıç bakiyesi ata.
            try:
                if float(self.balance_usdt or 0.0) <= 0.0 and not (self.positions or {}) and not (self.trades or []):
                    # Varsayılan: 4000 USDT ile başlat (Binance spot min-notional ve risk kuralları için yeterli)
                    self.set_starting_usdt(4000.0)
                    try:
                        logger.info(LogCategory.SYSTEM, "Paper hesap auto-init: balance_usdt 0 olduğu için 4000 USDT ile başlatıldı")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.error(LogCategory.SYSTEM, f"Paper hesap yükleme hatası: {e}")

    def reset_all(self, starting_balance_try: float = 100_000.0):
        """Tüm coin geçmişini ve metrikleri sil, bakiyeyi verilen tutarla başlat."""
        with self._lock:
            try:
                self.balance_try = float(starting_balance_try or 0.0)
                # USDT bazlı bakiye de sıfırlansın; FX varsa aynala
                try:
                    fx = float(self.usdt_try or 0.0)
                    self.balance_usdt = (self.balance_try / fx) if fx > 0 else 0.0
                except Exception:
                    self.balance_usdt = 0.0
            except Exception:
                self.balance_try = 100_000.0
            self.positions = {}
            self.realized_pnl = 0.0
            self.total_pnl = 0.0
            self.trades = []
            self.total_trade_count = 0
            self.daily_trade_count = 0
            self.daily_realized_pnl = 0.0
            self.last_reset_date = _dtmod.datetime.now().date()
            self._recent_sl = {}
            try:
                self.save()
            except Exception:
                pass

    def set_starting_usdt(self, amount_usdt: float):
        """Başlangıç bakiyesini USDT cinsinden ayarla (FX bilinir bilinmez TRY'ye çevrilir)."""
        try:
            amt = float(amount_usdt or 0.0)
            if amt <= 0:
                return
            # USDT ana bakiye olarak set edilir; TRY aynalanır (FX varsa)
            with self._lock:
                self.balance_usdt = amt
                fx = float(self.usdt_try or 0.0)
                if fx > 0:
                    self.balance_try = amt * fx
                else:
                    # FX henüz yoksa TRY aynalanamaz; sonra tick() içinde güncellenir
                    pass
        except Exception:
            pass

    def _ensure_store_path(self) -> str:
        if self._store_path:
            path = self._store_path
        else:
            root = Path(__file__).resolve().parents[1]
            data_dir = root / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            path = str(data_dir / 'paper_account.json')
            self._store_path = path
        return path


paper_executor = PaperTradeExecutor()
