from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


TRADE_CAL_FIELDS = "cal_date,is_open,pretrade_date"
STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"
MONEYFLOW_FIELDS = "ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount"
NAMECHANGE_FIELDS = "ts_code,name,start_date,end_date"
MAIRUI_TIMEOUT = 30
DEFAULT_LOOKBACK_TRADING_DAYS = 60
DEFAULT_MAX_WORKERS = 8


@dataclass(frozen=True)
class DailyUpdateConfig:
    data_dir: Path = Path("data")
    lookback_trading_days: int = DEFAULT_LOOKBACK_TRADING_DAYS
    max_workers: int = DEFAULT_MAX_WORKERS
    request_sleep_seconds: float = 0.0
    refresh_latest: bool = True

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def daily_dir(self) -> Path:
        return self.raw_dir / "daily"

    @property
    def min5_dir(self) -> Path:
        return self.raw_dir / "5min"

    @property
    def moneyflow_dir(self) -> Path:
        return self.raw_dir / "moneyflow"

    @property
    def st_dir(self) -> Path:
        return self.raw_dir / "st"

    @property
    def calendar_path(self) -> Path:
        return self.raw_dir / "calendar.parquet"

    @property
    def stock_basic_path(self) -> Path:
        return self.raw_dir / "stock_basic.parquet"

    @property
    def meta_path(self) -> Path:
        return self.raw_dir / "update_meta.json"


class TushareClient:
    def __init__(self, token: str):
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is required")
        import tushare as ts

        ts.set_token(token)
        self.pro = ts.pro_api(token)

    def trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, fields=TRADE_CAL_FIELDS)

    def stock_basic(self) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for status in ["L", "D", "P"]:
            df = self.pro.stock_basic(exchange="", list_status=status, fields=STOCK_BASIC_FIELDS)
            if df is not None and not df.empty:
                parts.append(df)
        if not parts:
            return pd.DataFrame(columns=STOCK_BASIC_FIELDS.split(","))
        out = pd.concat(parts, ignore_index=True)
        return out.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)

    def daily(self, trade_date: str) -> pd.DataFrame:
        return self._paged_call("daily", trade_date=trade_date, fields=DAILY_FIELDS)

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        return self._paged_call("adj_factor", trade_date=trade_date, fields=ADJ_FACTOR_FIELDS)

    def moneyflow(self, trade_date: str) -> pd.DataFrame:
        return self._paged_call("moneyflow", trade_date=trade_date, fields=MONEYFLOW_FIELDS)

    def namechange(self, end_date: str) -> pd.DataFrame:
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=3650)).strftime("%Y%m%d")
        df = self.pro.namechange(ts_code="", start_date=start_date, end_date=end_date, fields=NAMECHANGE_FIELDS)
        return df if df is not None else pd.DataFrame(columns=NAMECHANGE_FIELDS.split(","))

    def _paged_call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        offset = 0
        limit = 6000
        api = getattr(self.pro, api_name)
        while True:
            df = api(offset=offset, limit=limit, **kwargs)
            if df is None or df.empty:
                break
            parts.append(df)
            if len(df) < limit:
                break
            offset += limit
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)


class MairuiClient:
    def __init__(self, licence: str, timeout: int = MAIRUI_TIMEOUT):
        if not licence:
            raise RuntimeError("MAIRUI_LICENCE is required")
        self.licence = licence
        self.timeout = int(timeout)
        self.session = requests.Session()

    def history_5min(self, ts_code: str, trade_date: str) -> list[dict[str, Any]]:
        url = f"https://api.mairuiapi.com/hsstock/history/{ts_code}/5/n/{self.licence}"
        response = self.session.get(url, params={"st": trade_date, "et": trade_date}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            if str(payload.get("code", "")) not in {"0", "200", ""}:
                raise RuntimeError(f"mairui returned error for {ts_code} {trade_date}: {json.dumps(payload, ensure_ascii=False)}")
            data = payload.get("data", [])
        elif isinstance(payload, list):
            data = payload
        else:
            raise RuntimeError(f"unexpected mairui payload type for {ts_code} {trade_date}: {type(payload)!r}")
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected mairui data field for {ts_code} {trade_date}")
        return [dict(row) for row in data if isinstance(row, dict)]


def _now_utc_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _ensure_dirs(config: DailyUpdateConfig) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.daily_dir.mkdir(parents=True, exist_ok=True)
    config.min5_dir.mkdir(parents=True, exist_ok=True)
    config.moneyflow_dir.mkdir(parents=True, exist_ok=True)
    config.st_dir.mkdir(parents=True, exist_ok=True)


def _normalize_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


def _load_trade_window(pro: TushareClient, lookback_trading_days: int) -> tuple[pd.DataFrame, list[str], str]:
    end_date = _now_utc_date().strftime("%Y%m%d")
    start_date = (_now_utc_date() - pd.Timedelta(days=max(180, lookback_trading_days * 3))).strftime("%Y%m%d")
    calendar = pro.trade_calendar(start_date=start_date, end_date=end_date)
    if calendar is None or calendar.empty:
        raise RuntimeError("trade_cal returned no rows")
    cal = calendar.copy()
    cal["cal_date"] = _normalize_date_str(cal["cal_date"])
    cal["pretrade_date"] = _normalize_date_str(cal.get("pretrade_date", pd.Series(index=cal.index, dtype=object)))
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    cal = cal.dropna(subset=["cal_date"]).drop_duplicates(subset=["cal_date"], keep="last").sort_values("cal_date").reset_index(drop=True)
    open_dates = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()
    need = max(1, int(lookback_trading_days) + 1)
    if len(open_dates) < need:
        raise RuntimeError(f"trade_cal returned only {len(open_dates)} open dates, need at least {need}")
    target_dates = open_dates[-need:]
    anchor_date = target_dates[-1]
    return cal, target_dates, anchor_date


def _write_calendar_and_stock_basic(config: DailyUpdateConfig, calendar: pd.DataFrame, stock_basic: pd.DataFrame, target_dates: list[str]) -> pd.DataFrame:
    cal = calendar.copy()
    cal = cal[cal["is_open"] == 1].copy()
    cal["in_window"] = cal["cal_date"].isin(set(target_dates))
    cal.to_parquet(config.calendar_path, index=False)

    sb = stock_basic.copy()
    if not sb.empty:
        for col in ["list_date", "delist_date"]:
            if col in sb.columns:
                sb[col] = _normalize_date_str(sb[col])
        sb["name"] = sb["name"].astype(str).str.strip()
        sb = sb.sort_values(["ts_code"]).reset_index(drop=True)
    sb.to_parquet(config.stock_basic_path, index=False)
    return sb


def _need_refresh(path: Path, refresh_latest: bool, latest_trade_date: str) -> bool:
    if not path.exists():
        return True
    if refresh_latest and path.stem == latest_trade_date:
        return True
    return False


def _merge_daily_and_adj(daily_df: pd.DataFrame, adj_df: pd.DataFrame, trade_date: str, stock_basic: pd.DataFrame) -> pd.DataFrame:
    daily = daily_df.copy() if daily_df is not None else pd.DataFrame()
    adj = adj_df.copy() if adj_df is not None else pd.DataFrame()
    if daily.empty and adj.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    for df in [daily, adj]:
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = _normalize_date_str(df["trade_date"])
    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="outer") if not daily.empty and not adj.empty else (daily if not daily.empty else adj)
    if stock_basic is not None and not stock_basic.empty:
        base_cols = [c for c in ["ts_code", "symbol", "name", "area", "industry", "market", "list_date", "delist_date", "is_hs"] if c in stock_basic.columns]
        merged = merged.merge(stock_basic[base_cols], on="ts_code", how="left")
    if "trade_date" not in merged.columns:
        merged["trade_date"] = trade_date
    merged["trade_date"] = merged["trade_date"].fillna(trade_date)
    return merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _update_daily_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    latest = target_dates[-1]
    for trade_date in target_dates:
        out_path = config.daily_dir / f"{trade_date}.parquet"
        if not _need_refresh(out_path, config.refresh_latest, latest):
            continue
        daily_df = pro.daily(trade_date)
        adj_df = pro.adj_factor(trade_date)
        merged = _merge_daily_and_adj(daily_df, adj_df, trade_date, stock_basic)
        merged.to_parquet(out_path, index=False)
        updated.append(trade_date)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return updated


def _update_moneyflow_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    latest = target_dates[-1]
    stock_cols = [c for c in ["ts_code", "name", "symbol", "industry", "market"] if c in stock_basic.columns]
    stock_view = stock_basic[stock_cols] if stock_cols else pd.DataFrame(columns=["ts_code"])
    for trade_date in target_dates:
        out_path = config.moneyflow_dir / f"{trade_date}.parquet"
        if not _need_refresh(out_path, config.refresh_latest, latest):
            continue
        df = pro.moneyflow(trade_date)
        if df is None:
            df = pd.DataFrame(columns=MONEYFLOW_FIELDS.split(","))
        if not df.empty:
            df["trade_date"] = _normalize_date_str(df["trade_date"])
            if not stock_view.empty:
                df = df.merge(stock_view, on="ts_code", how="left")
            df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        df.to_parquet(out_path, index=False)
        updated.append(trade_date)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return updated


def _normalize_mairui_rows(ts_code: str, trade_date: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("ts_code", ts_code)
        item["trade_date"] = trade_date
        item["row_no"] = idx
        normalized.append(item)
    return normalized


def _fetch_single_code_5min(mairui: MairuiClient, ts_code: str, trade_date: str, sleep_seconds: float) -> list[dict[str, Any]]:
    rows = mairui.history_5min(ts_code, trade_date)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return _normalize_mairui_rows(ts_code, trade_date, rows)


def _update_5min_files(mairui: MairuiClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    latest = target_dates[-1]
    ts_codes = sorted(stock_basic["ts_code"].dropna().astype(str).unique().tolist()) if "ts_code" in stock_basic.columns else []
    for trade_date in target_dates:
        out_path = config.min5_dir / f"{trade_date}.parquet"
        if not _need_refresh(out_path, config.refresh_latest, latest):
            continue
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
            futures = {
                executor.submit(_fetch_single_code_5min, mairui, ts_code, trade_date, config.request_sleep_seconds): ts_code
                for ts_code in ts_codes
            }
            for fut in as_completed(futures):
                ts_code = futures[fut]
                try:
                    rows.extend(fut.result())
                except Exception as exc:
                    raise RuntimeError(f"failed to fetch mairui 5min for {ts_code} on {trade_date}") from exc
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values([c for c in ["ts_code", "trade_date", "row_no"] if c in df.columns]).reset_index(drop=True)
        df.to_parquet(out_path, index=False)
        updated.append(trade_date)
    return updated


def _is_st_name(name: str) -> int:
    value = str(name or "").strip().upper()
    return int(value.startswith("ST") or value.startswith("*ST"))


def _build_st_name_history(stock_basic: pd.DataFrame, namechange: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    history: dict[str, list[tuple[pd.Timestamp, str]]] = {}

    if not namechange.empty:
        hist = namechange.copy()
        hist["ts_code"] = hist["ts_code"].astype(str)
        hist["name"] = hist["name"].astype(str).str.strip()
        hist["start_date"] = pd.to_datetime(hist["start_date"], errors="coerce")
        hist = hist.dropna(subset=["ts_code", "start_date"]).sort_values(["ts_code", "start_date"]).reset_index(drop=True)
        for ts_code, sub in hist.groupby("ts_code", sort=False):
            history[str(ts_code)] = [(pd.Timestamp(row.start_date), str(row.name).strip()) for row in sub.itertuples(index=False)]

    for row in stock_basic.itertuples(index=False):
        ts_code = str(getattr(row, "ts_code", ""))
        if not ts_code:
            continue
        current_name = str(getattr(row, "name", "")).strip()
        hist = history.get(ts_code, [])
        if hist:
            hist[-1] = (hist[-1][0], current_name)
        else:
            list_date = pd.to_datetime(getattr(row, "list_date", None), errors="coerce")
            hist = [(list_date if pd.notna(list_date) else pd.Timestamp("1900-01-01"), current_name)]
        history[ts_code] = hist
    return history


def _build_st_rows_for_date(trade_date: str, stock_basic: pd.DataFrame, name_history: dict[str, list[tuple[pd.Timestamp, str]]]) -> pd.DataFrame:
    if stock_basic.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "name", "is_st"])

    target = pd.Timestamp(trade_date)
    rows: list[dict[str, Any]] = []
    for _, base in stock_basic.iterrows():
        ts_code = str(base.get("ts_code", ""))
        if not ts_code:
            continue
        effective_name = str(base.get("name", "")).strip()
        for start_date, name in name_history.get(ts_code, []):
            if start_date <= target:
                effective_name = str(name).strip()
            else:
                break
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": ts_code,
                "name": effective_name,
                "is_st": _is_st_name(effective_name),
            }
        )
    return pd.DataFrame(rows).sort_values(["ts_code"]).reset_index(drop=True)


def _update_st_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    latest = target_dates[-1]
    namechange = pro.namechange(end_date=latest)
    name_history = _build_st_name_history(stock_basic, namechange)
    for trade_date in target_dates:
        out_path = config.st_dir / f"{trade_date}.parquet"
        if not _need_refresh(out_path, config.refresh_latest, latest):
            continue
        df = _build_st_rows_for_date(trade_date, stock_basic, name_history)
        df.to_parquet(out_path, index=False)
        updated.append(trade_date)
    return updated


def update_daily_market_data(config: DailyUpdateConfig) -> dict[str, Any]:
    _ensure_dirs(config)
    tushare = TushareClient(os.getenv("TUSHARE_TOKEN", "").strip())
    mairui = MairuiClient(os.getenv("MAIRUI_LICENCE", "").strip())

    calendar, target_dates, anchor_date = _load_trade_window(tushare, config.lookback_trading_days)
    stock_basic = _write_calendar_and_stock_basic(config, calendar, tushare.stock_basic(), target_dates)

    daily_updated = _update_daily_files(tushare, config, target_dates, stock_basic)
    moneyflow_updated = _update_moneyflow_files(tushare, config, target_dates, stock_basic)
    min5_updated = _update_5min_files(mairui, config, target_dates, stock_basic)
    st_updated = _update_st_files(tushare, config, target_dates, stock_basic)

    meta = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_trade_date": anchor_date,
        "target_trade_dates": target_dates,
        "lookback_trading_days": int(config.lookback_trading_days),
        "daily_updated": daily_updated,
        "moneyflow_updated": moneyflow_updated,
        "min5_updated": min5_updated,
        "st_updated": st_updated,
        "stock_basic_rows": int(len(stock_basic)),
    }
    config.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
