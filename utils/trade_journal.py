import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List


class TradeJournal:
    """Kalıcı işlem/sinyal/konfigürasyon günlüğü.
    Kayıtlar JSONL + CSV formatında tutulur, paper resetlerinden etkilenmez.
    """

    def __init__(self, path: Optional[str] = None):
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = str(data_dir / "trade_journal.jsonl")
        self.path = path
        # Tüm olaylar için TEK CSV dosyası
        self.trades_csv = str(data_dir / "trade_journal_trades.csv")
        # Tek şema: tüm alanları kapsayan geniş kolon seti
        self.csv_fieldnames: List[str] = [
            "ts",
            "type",
            # Sinyal/strateji alanları
            "strategy",
            "symbol",
            "side",
            "accepted",
            "reason_code",
            "details",
            # İşlem alanları
            "qty",
            "entry_price",
            "exit_price",
            "realized_pnl_usdt",
            "hold_seconds",
            "close_reason",
            "fee_usdt",
            "partial",
            # Konfig değişiklik alanları
            "scope",
            "key",
            "old",
            "new",
            "note",
            # Strateji switch alanları
            "from",
            "to",
            "context",
        ]

    def _write_entry(self, entry_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """JSONL'e yaz ve kaydı geri döndür."""
        record = {
            "ts": datetime.utcnow().isoformat(),
            "type": entry_type,
            **payload,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Günlük tutma hiçbir zaman ana akışı bozmamalı
            pass
        return record

    def _append_csv(self, path: str, row: Dict[str, Any]) -> None:
        """Basit CSV append; yoksa tek şemalı header yazar. Sadece tanımlı alanları yazar."""
        try:
            # Alanları sıraya göre hazırla
            out_row = {}
            for fn in self.csv_fieldnames:
                val = row.get(fn)
                # Dict ise JSON string'e çevir
                if isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False, default=str)
                out_row[fn] = val

            file_exists = os.path.exists(path)
            needs_header = (not file_exists) or os.path.getsize(path) == 0
            with open(path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                if needs_header:
                    writer.writeheader()
                writer.writerow(out_row)
        except Exception:
            # CSV yazımı da ana akışı bozmasın
            pass

    # --- Sinyal kararları ---
    def log_signal_decision(
        self,
        *,
        strategy_name: str,
        symbol: Optional[str],
        side: Optional[str],
        accepted: bool,
        reason_code: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "side": side,
            "accepted": accepted,
            "reason_code": reason_code,
        }
        if details:
            payload["details"] = details
        rec = self._write_entry("signal_decision", payload)
        # CSV: tüm olaylar tek dosyada
        try:
            self._append_csv(self.trades_csv, rec)
        except Exception:
            pass

    def log_signal_skipped(
        self,
        *,
        strategy_name: str,
        symbol: Optional[str],
        side: Optional[str],
        reason_code: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_signal_decision(
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            accepted=False,
            reason_code=reason_code,
            details=details,
        )

    # --- İşlem kapanışları (kar/zarar ve elde tutma süresi) ---
    def log_trade_close(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        realized_pnl_usdt: float,
        hold_seconds: Optional[float],
        close_reason: str,
        strategy_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl_usdt": realized_pnl_usdt,
            "hold_seconds": hold_seconds,
            "close_reason": close_reason,
            "strategy": strategy_name,
        }
        if meta:
            payload["meta"] = meta
        rec = self._write_entry("trade_close", payload)
        # CSV: kapanan işlemler (meta'dan bazı alanlar düzleştirilir)
        try:
            meta_dict = rec.get("meta") or {}
            flat = dict(rec)
            # Meta içinden sık kullanılanları düz al
            flat["fee_usdt"] = meta_dict.get("fee_usdt")
            flat["partial"] = meta_dict.get("partial")
            self._append_csv(self.trades_csv, flat)
        except Exception:
            pass

    # --- Strateji / konfigürasyon değişiklikleri ---
    def log_config_change(
        self,
        *,
        scope: str,
        key: str,
        old_value: Any,
        new_value: Any,
        note: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "scope": scope,
            "key": key,
            "old": old_value,
            "new": new_value,
        }
        if note:
            payload["note"] = note
        rec = self._write_entry("config_change", payload)
        # CSV: konfig değişiklikleri
        try:
            self._append_csv(self.trades_csv, rec)
        except Exception:
            pass

    def log_strategy_switch(
        self,
        *,
        from_strategy: Optional[str],
        to_strategy: str,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "from": from_strategy,
            "to": to_strategy,
        }
        if reason:
            payload["reason"] = reason
        if context:
            payload["context"] = context
        rec = self._write_entry("strategy_switch", payload)
        # CSV: strateji değişimleri
        try:
            self._append_csv(self.trades_csv, rec)
        except Exception:
            pass


journal = TradeJournal()
