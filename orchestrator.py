"""
Son Model Bot - Orchestrator

Bu dosya:
- Rejim tespiti
- Strateji orkestrasyonu
- Meta filtreler (ML / sentiment / volatility / correlation)
- Risk & execution
katmanlarını tek bir akışta birleştirir.

Ana giriş noktası: SonModelOrchestrator.run_step(market_snapshot)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import datetime as dt

from utils.logger import get_logger, LogCategory
from risk_management.risk_manager import risk_manager
from trading.paper_executor import paper_executor
from strategies.strategy_manager import strategy_manager
from ai.signal_classifier import SignalClassifier

# Rejim tespiti: varsa gelişmiş model, yoksa StrategyManager'dan gelen rejim
try:
    from ai.market_regime import detect_market_regime  # type: ignore
except Exception:  # ImportError dahil
    detect_market_regime = None  # type: ignore

# Live executor: opsiyonel, yoksa sadece paper mod desteklenir
try:
    from trading.live_executor import LiveExecutor  # type: ignore
except Exception:
    LiveExecutor = None  # type: ignore

logger = get_logger("orchestrator")


# ---------- Strateji Wrapper'ları ---------- #

# StrategyManager içindeki hazır strateji instance'larını kullanıyoruz
_tf = strategy_manager.strategies.get("trend_following")
_mi = strategy_manager.strategies.get("momentum_ignition")
_vb = strategy_manager.strategies.get("volatility_breakout")
_pd = strategy_manager.strategies.get("pump_dump")
_mr = strategy_manager.strategies.get("mean_reversion")
_micro = strategy_manager.strategies.get("micro_reversion")
_grid = strategy_manager.strategies.get("grid")
_scalp = strategy_manager.strategies.get("scalping")

_sent = strategy_manager.strategies.get("ai_sentiment_fusion")
_av = strategy_manager.strategies.get("adaptive_volatility")
_corr = strategy_manager.strategies.get("correlation_rotation")


def _safe_signals(strat, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """StrategyManager içindeki stratejiler için güvenli generate_signals wrapper'ı."""
    if strat is None:
        return []
    try:
        res = strat.generate_signals(market_snapshot)
        return res or []
    except Exception as e:  # pragma: no cover - sadece log için
        try:
            sname = getattr(strat, "name", strat.__class__.__name__)
        except Exception:
            sname = str(strat)
        logger.error(LogCategory.SYSTEM, f"{sname} generate_signals hatası: {e}")
        return []


def tf_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_tf, market_snapshot)


def mi_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_mi, market_snapshot)


def vb_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_vb, market_snapshot)


def pd_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_pd, market_snapshot)


def mr_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_mr, market_snapshot)


def micro_rev_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_micro, market_snapshot)


def grid_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_grid, market_snapshot)


def scalping_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_scalp, market_snapshot)


def sentiment_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_sent, market_snapshot)


def adaptive_vol_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_av, market_snapshot)


def corr_rotation_signals(market_snapshot, regime_dict=None):
    return _safe_signals(_corr, market_snapshot)


# ---------- Veri Yapıları ---------- #


@dataclass
class RegimeState:
    global_regime: str
    vol_regime: str
    trend_strength: float
    risk_mode: str
    timestamp: dt.datetime


@dataclass
class RawSignal:
    symbol: str
    side: str              # "buy" / "sell"
    strength: float        # 0-1 arası skor (strateji içinden gelen)
    strategy: str
    sl: Optional[float] = None
    tp: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class FinalOrderIntent:
    symbol: str
    side: str
    strength: float
    size: float
    sl: Optional[float]
    tp: Optional[float]
    strategies: List[str]
    regime: str
    meta: Dict[str, Any]


# ---------- Rejim Engine ---------- #


class RegimeEngine:
    """Global rejimi belirler."""

    def __init__(self) -> None:
        pass

    def detect(self, global_market_data: Dict[str, Any]) -> RegimeState:
        now = global_market_data.get("timestamp") or dt.datetime.utcnow()

        # 1) Eğer varsa ai.market_regime çıktısını kullan
        if detect_market_regime is not None:
            try:
                r = detect_market_regime(global_market_data)
                if isinstance(r, dict):
                    return RegimeState(
                        global_regime=str(r.get("label", "SIDEWAYS")),
                        vol_regime=str(r.get("vol_regime", "NORMAL")),
                        trend_strength=float(r.get("trend_strength", 0.0) or 0.0),
                        risk_mode=str(r.get("risk_mode", "NORMAL")),
                        timestamp=now,
                    )
            except Exception as e:  # pragma: no cover - sadece log
                logger.error(LogCategory.SYSTEM, f"detect_market_regime hata verdi: {e}")

        # 2) Fallback: StrategyManager içindeki _regime_name alanı
        try:
            rname = getattr(strategy_manager, "_regime_name", "unknown") or "unknown"
        except Exception:
            rname = "unknown"

        rlow = str(rname).lower()
        if rlow.startswith("trend"):
            g_regime = "BULL_TREND"
        elif rlow in ("sideways", "range", "sideways_high_vol"):
            g_regime = "SIDEWAYS"
        elif rlow in ("volatile", "crash"):
            g_regime = "HIGH_VOL_TREND"
        else:
            g_regime = "SIDEWAYS"

        return RegimeState(
            global_regime=g_regime,
            vol_regime="NORMAL",
            trend_strength=0.0,
            risk_mode="NORMAL",
            timestamp=now,
        )


# ---------- Strategy Pool ---------- #


class StrategyPool:
    """Hangi rejimde hangi stratejinin çağrılacağını tutar."""

    def __init__(self) -> None:
        self.registry: Dict[str, Dict[str, Any]] = {
            "trend_following": {
                "fn": tf_signals,
                "type": "trend",
                "regimes": ["BULL_TREND", "HIGH_VOL_TREND"],
                "weight": 0.5,
            },
            "momentum_ignition": {
                "fn": mi_signals,
                "type": "trend",
                "regimes": ["BULL_TREND", "HIGH_VOL_TREND"],
                "weight": 0.3,
            },
            "volatility_breakout": {
                "fn": vb_signals,
                "type": "trend",
                "regimes": ["BULL_TREND", "HIGH_VOL_TREND"],
                "weight": 0.2,
            },
            "pump_dump_momentum": {
                "fn": pd_signals,
                "type": "trend",
                "regimes": ["HIGH_VOL_TREND"],
                "weight": 0.2,
            },
            "mean_reversion": {
                "fn": mr_signals,
                "type": "reversion",
                "regimes": ["SIDEWAYS", "LOW_VOL_RANGE"],
                "weight": 0.6,
            },
            "micro_reversion": {
                "fn": micro_rev_signals,
                "type": "reversion",
                "regimes": ["SIDEWAYS"],
                "weight": 0.4,
            },
            "grid": {
                "fn": grid_signals,
                "type": "grid",
                "regimes": ["LOW_VOL_RANGE"],
                "weight": 0.7,
            },
            "scalping": {
                "fn": scalping_signals,
                "type": "scalping",
                "regimes": ["LOW_VOL_RANGE"],
                "weight": 0.3,
            },
        }

    def get_active_strategies(self, regime_state: RegimeState) -> Dict[str, Dict[str, Any]]:
        active: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self.registry.items():
            if regime_state.global_regime in cfg.get("regimes", []):
                active[name] = cfg
        return active

    def generate_signals(
        self,
        market_snapshot: Dict[str, Any],
        regime_state: RegimeState,
    ) -> List[RawSignal]:
        active_strats = self.get_active_strategies(regime_state)
        all_signals: List[RawSignal] = []

        for strat_name, cfg in active_strats.items():
            fn = cfg.get("fn")
            if fn is None:
                continue
            try:
                try:
                    strat_signals = fn(market_snapshot, regime_state.__dict__)
                except TypeError:
                    strat_signals = fn(market_snapshot)
            except Exception as e:
                logger.error(LogCategory.SYSTEM, f"{strat_name} generate_signals wrapper hatası: {e}")
                strat_signals = []

            if not strat_signals:
                continue

            for s in strat_signals:
                try:
                    sym = s.get("symbol")
                    side = s.get("side")
                    if not sym or not side:
                        continue
                    rs = RawSignal(
                        symbol=str(sym),
                        side=str(side).lower(),
                        strength=float(s.get("strength", 1.0) or 1.0),
                        strategy=strat_name,
                        sl=s.get("stop_loss") or s.get("sl"),
                        tp=s.get("take_profit") or s.get("tp"),
                        meta=s,
                    )
                    all_signals.append(rs)
                except Exception as e:
                    logger.error(LogCategory.SYSTEM, f"{strat_name} sinyal parse hatası: {e}")

        return all_signals


# ---------- Meta Filtreler ---------- #


class MetaFilterManager:
    """ML / sentiment / volatility / correlation tabanlı filtre/weight katmanı."""

    def __init__(self) -> None:
        # DEBUG: MetaFilter bypass - tüm sinyaller geçirilecek
        self.min_strength = 0.0
        try:
            self._ml_classifier: SignalClassifier | None = SignalClassifier()
        except Exception:
            self._ml_classifier = None

    def apply(
        self,
        raw_signals: List[RawSignal],
        market_snapshot: Dict[str, Any],
        regime_state: RegimeState,
    ) -> List[RawSignal]:
        if not raw_signals:
            logger.info(LogCategory.SYSTEM, "[DEBUG][META] MetaFilter bypass - raw_signals boş")
            return []

        logger.info(LogCategory.SYSTEM, f"[DEBUG][META] MetaFilter bypass aktif, gelen sinyal sayısı: {len(raw_signals)}")
        # DEBUG MODU: ML / sentiment / correlation tamamen bypass
        return raw_signals


# ---------- Allocation Engine ---------- #


class AllocationEngine:
    """Aynı sembole ait sinyalleri birleştirip final intent çıkarır."""

    def __init__(self) -> None:
        self.base_risk_per_trade = 0.005  # equity'nin %0.5'i
        self.max_strength_per_symbol = 1.5

    def aggregate_signals(
        self,
        signals: List[RawSignal],
        equity: float,
        regime_state: RegimeState,
    ) -> List[FinalOrderIntent]:
        if not signals or equity <= 0:
            return []

        agg: Dict[str, Dict[str, Any]] = {}
        for s in signals:
            key = s.symbol
            entry = agg.setdefault(
                key,
                {
                    "symbol": s.symbol,
                    "net_strength": 0.0,
                    "side_votes": {"buy": 0.0, "sell": 0.0},
                    "sl": s.sl,
                    "tp": s.tp,
                    "strategies": set(),
                    "meta_list": [],
                },
            )
            entry["strategies"].add(s.strategy)
            entry["meta_list"].append(s.meta or {})
            entry["side_votes"][s.side] = entry["side_votes"].get(s.side, 0.0) + s.strength
            entry["net_strength"] += s.strength
            if entry["sl"] is None and s.sl is not None:
                entry["sl"] = s.sl
            if entry["tp"] is None and s.tp is not None:
                entry["tp"] = s.tp

        final_intents: List[FinalOrderIntent] = []

        for sym, e in agg.items():
            buy_vote = e["side_votes"].get("buy", 0.0)
            sell_vote = e["side_votes"].get("sell", 0.0)
            if buy_vote == 0 and sell_vote == 0:
                continue
            side = "buy" if buy_vote >= sell_vote else "sell"
            raw_strength = max(buy_vote, sell_vote)
            raw_strength = min(raw_strength, self.max_strength_per_symbol)

            target_risk = self.base_risk_per_trade * raw_strength
            notional = equity * target_risk

            intent = FinalOrderIntent(
                symbol=sym,
                side=side,
                strength=raw_strength,
                size=notional,
                sl=e["sl"],
                tp=e["tp"],
                strategies=sorted(e["strategies"]),
                regime=regime_state.global_regime,
                meta={
                    "side_votes": e["side_votes"],
                    "raw_meta": e["meta_list"],
                },
            )
            final_intents.append(intent)

        return final_intents


# ---------- Orchestrator Ana Sınıf ---------- #


class SonModelOrchestrator:
    def __init__(
        self,
        mode: str = "paper",
        risk_config_path: Optional[str] = None,
        live_api_key: Optional[str] = None,
        live_api_secret: Optional[str] = None,
        live_use_sandbox: bool = False,
    ) -> None:
        self.mode = mode
        self.regime_engine = RegimeEngine()
        self.strategy_pool = StrategyPool()
        self.meta_filters = MetaFilterManager()
        self.allocation_engine = AllocationEngine()

        if mode == "paper":
            # Binance tarafında mevcut kağıt (paper) executor singleton'unu kullan
            self.executor = paper_executor
        elif mode == "live":
            # Binance tarafında live executor henüz tanımlı değil
            raise NotImplementedError("Binance live executor henüz tanımlanmadı")
        else:
            raise ValueError(f"Bilinmeyen mode: {mode}")

    def _get_equity(self) -> float:
        try:
            metrics = risk_manager.get_risk_metrics()
            eq = float(
                metrics.get("current_balance", metrics.get("available_balance", 0.0))
                or 0.0
            )
            return max(eq, 0.0)
        except Exception as e:
            logger.error(LogCategory.SYSTEM, f"Equity alınamadı: {e}")
            return 0.0

    def _submit_paper(self, intents: List[FinalOrderIntent], regime_state: RegimeState) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        for intent in intents:
            sig: Dict[str, Any] = {
                "symbol": intent.symbol,
                "side": intent.side,
                # Paper executor zaten kendi risk tahsisini yaptığı için boyutu boş bırakmak güvenli
                # istersen ileride intent.size'ı target_notional_try olarak aktarabilirsin.
                "size": 0.0,
                "strategy_name": "+".join(intent.strategies) or "orchestrator",
                "source": "orchestrator",
            }
            if intent.sl is not None:
                sig["stop_loss"] = intent.sl
            if intent.tp is not None:
                sig["take_profit"] = intent.tp
            signals.append(sig)

        if not signals:
            return []

        try:
            logger.info(
                LogCategory.SYSTEM,
                f"[ORCH][PAPER] Executor'a gönderilen sinyal sayısı: {len(signals)} (source=orchestrator)",
            )
            self.executor.execute(signals)
        except Exception as e:
            logger.error(LogCategory.SYSTEM, f"paper_executor.execute hatası: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for intent in intents:
            out.append(
                {
                    "intent": intent,
                    "order_response": None,
                    "regime_state": regime_state.__dict__,
                }
            )
        return out

    def _submit_live(self, intents: List[FinalOrderIntent], regime_state: RegimeState, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for intent in intents:
            # Basit: market_snapshot'tan son fiyatı bulmaya çalış
            price = None
            try:
                data = market_snapshot.get(intent.symbol)
                # Eğer sembol datası listeyse son elemanı kullan
                if isinstance(data, list) and data:
                    data = data[-1]
                if isinstance(data, dict):
                    price = data.get("price") or data.get("close")
            except Exception:
                price = None

            qty: float
            if price and price > 0:
                qty = float(intent.size) / float(price)
            else:
                qty = 0.0

            try:
                resp = self.executor.submit_order(
                    symbol=intent.symbol,
                    side=intent.side,
                    order_type="market",
                    quantity=qty,
                )
            except Exception as e:
                logger.error(LogCategory.SYSTEM, f"LiveExecutor.submit_order hatası: {e}")
                continue

            results.append(
                {
                    "intent": intent,
                    "order_response": resp,
                    "regime_state": regime_state.__dict__,
                }
            )
        return results

    def run_step(self, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Tek bar / tick için ana akış."""

        # 1) Rejim tespiti
        regime_state = self.regime_engine.detect(market_snapshot)

        # 2) Stratejilerden raw sinyaller
        raw_signals = self.strategy_pool.generate_signals(market_snapshot, regime_state)
        if not raw_signals:
            return []

        # 3) Meta filtreler
        filtered_signals = self.meta_filters.apply(raw_signals, market_snapshot, regime_state)
        if not filtered_signals:
            return []

        # 4) Equity
        equity = self._get_equity()
        if equity <= 0:
            logger.error(LogCategory.SYSTEM, "Equity 0 veya negatif, trade atlanıyor.")
            return []

        # 5) Allocation -> final intent listesi
        intents = self.allocation_engine.aggregate_signals(filtered_signals, equity, regime_state)
        if not intents:
            return []

        # 6) Risk: trading_enabled bayrağı kapalıysa yeni trade açma
        try:
            from risk_management.risk_manager import risk_manager as _rm_inst  # type: ignore
            rm_metrics = _rm_inst.get_risk_metrics() if _rm_inst is not None else {}
            if not rm_metrics.get("trading_enabled", True):
                logger.warning(LogCategory.SYSTEM, "[ORCH] Trading risk yöneticisi tarafından devre dışı (run_step), intents gönderilmedi")
                return []
        except Exception:
            pass

        # 7) Mode'a göre executor'a gönder
        if self.mode == "paper":
            return self._submit_paper(intents, regime_state)
        else:
            return self._submit_live(intents, regime_state, market_snapshot)

    def process_signals(
        self,
        signals: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """StrategyManager'dan gelen ham sinyalleri işler.

        - Rejim tespiti
        - Meta filtreler
        - Allocation
        - Executor (paper / live)
        """

        try:
            logger.info(LogCategory.SYSTEM, f"[DEBUG][ORCH] Gelen sinyal sayısı: {len(signals) if signals else 0}")
            if not signals:
                logger.info(LogCategory.SYSTEM, "[DEBUG][ORCH] signals boş, dönüş yok")
                return []

            # 1) Rejim tespiti (market_context üzerinden)
            ctx = market_context or {}
            if "timestamp" not in ctx:
                try:
                    ctx["timestamp"] = dt.datetime.utcnow()
                except Exception:
                    pass
            regime_state = self.regime_engine.detect(ctx)

            # 2) Ham sinyalleri RawSignal listesine çevir
            raw_signals: List[RawSignal] = []
            for s in signals:
                try:
                    sym = (s.get("symbol") or "").strip().upper()
                    side = (s.get("side") or "").strip().lower()
                    if not sym or not side:
                        continue

                    strength_val = s.get(
                        "strength",
                        s.get(
                            "confidence",
                            s.get("signal_strength", s.get("ml_success_prob", 0.0)),
                        ),
                    )
                    try:
                        strength = float(strength_val or 0.0)
                    except Exception:
                        strength = 0.0

                    strat_name = (
                        s.get("strategy_name")
                        or s.get("strategy")
                        or "unknown"
                    )

                    rs = RawSignal(
                        symbol=sym,
                        side=side,
                        strength=strength,
                        strategy=str(strat_name),
                        sl=s.get("stop_loss") or s.get("sl"),
                        tp=s.get("take_profit") or s.get("tp"),
                        meta=s,
                    )
                    raw_signals.append(rs)
                except Exception as e:
                    logger.error(LogCategory.SYSTEM, f"process_signals RawSignal parse hatası: {e}")

            logger.info(LogCategory.SYSTEM, f"[DEBUG][ORCH] RawSignal sayısı: {len(raw_signals)}")
            if not raw_signals:
                logger.info(LogCategory.SYSTEM, "[DEBUG][ORCH] raw_signals boş, dönüş yok")
                return []

            # 3) Meta filtreler
            filtered_signals = self.meta_filters.apply(raw_signals, ctx, regime_state)
            logger.info(LogCategory.SYSTEM, f"[DEBUG][ORCH] Meta filtreden geçen sinyal sayısı: {len(filtered_signals)}")
            if not filtered_signals:
                logger.info(LogCategory.SYSTEM, "[DEBUG][ORCH] filtered_signals boş, meta filtre hepsini kesti")
                return []

            # 4) Equity
            equity = self._get_equity()
            logger.info(LogCategory.SYSTEM, f"[DEBUG][ORCH] Equity: {equity}")
            if equity <= 0:
                logger.error(LogCategory.SYSTEM, "[DEBUG][ORCH] Equity <= 0, trade atlanıyor")
                return []

            # 5) Allocation -> final intent listesi
            intents = self.allocation_engine.aggregate_signals(
                filtered_signals,
                equity,
                regime_state,
            )
            logger.info(LogCategory.SYSTEM, f"[DEBUG][ORCH] Allocation sonrası intent sayısı: {len(intents)}")
            if not intents:
                logger.info(LogCategory.SYSTEM, "[DEBUG][ORCH] intents boş, allocation hiçbir şey üretmedi")
                return []

            # 6) Risk: trading_enabled bayrağı kapalıysa yeni trade açma
            try:
                from risk_management.risk_manager import risk_manager as _rm_inst  # type: ignore
                rm_metrics = _rm_inst.get_risk_metrics() if _rm_inst is not None else {}
                if not rm_metrics.get("trading_enabled", True):
                    logger.warning(LogCategory.SYSTEM, "[ORCH] Trading risk yöneticisi tarafından devre dışı (process_signals), intents gönderilmedi")
                    return []
            except Exception:
                pass

            # 7) Mode'a göre executor'a gönder
            if self.mode == "paper":
                # Binance USDT botu için: intent.size USDT notional'ı price_map üzerinden miktara çevir
                results: List[Dict[str, Any]] = []
                try:
                    price_map = ctx.get("price_map", {}) if isinstance(ctx, dict) else {}
                except Exception:
                    price_map = {}

                for intent in intents:
                    last_price = None
                    try:
                        if isinstance(price_map, dict):
                            last_price = price_map.get(intent.symbol)
                    except Exception:
                        last_price = None

                    if not last_price:
                        logger.error(LogCategory.SYSTEM, f"[ORCH] Fiyat bulunamadı, trade atlandı: {intent.symbol}")
                        continue

                    submit_fn = getattr(self.executor, "submit_order_notional", None)
                    if not callable(submit_fn):
                        # Geri uyumluluk: eski execute tabanlı akışa düş
                        return self._submit_paper(intents, regime_state)

                    try:
                        order_resp = submit_fn(
                            symbol=intent.symbol,
                            side=intent.side,
                            notional=float(intent.size),  # USDT
                            price=float(last_price),
                        )
                    except Exception as e:
                        logger.error(LogCategory.SYSTEM, f"[ORCH] Executor.submit_order_notional hatası: {e}")
                        continue

                    results.append(
                        {
                            "intent": intent,
                            "order_response": order_resp,
                            "regime_state": regime_state.__dict__,
                        }
                    )

                return results
            else:
                return self._submit_live(intents, regime_state, ctx)
        except Exception as e:
            logger.error(LogCategory.SYSTEM, f"process_signals akış hatası: {e}")
            return []
