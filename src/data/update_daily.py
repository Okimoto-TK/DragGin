from __future__ import annotations

import json
import os
import shutil
import time
import zoneinfo
from concurrent.futures import ProcessPoolExecutor, as_completed
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
SUSPEND_FIELDS = "ts_code,trade_date,suspend_timing,suspend_type"
NAMECHANGE_FIELDS = "ts_code,name,start_date,end_date"
MAIRUI_TIMEOUT = 5
DEFAULT_LOOKBACK_TRADING_DAYS = 150
REAL_OPEN_LOOKBACK_DAYS = 20
DEFAULT_5MIN_PROCESS_WORKERS = 4


@dataclass(frozen=True)
class DailyUpdateConfig:
    data_dir: Path = Path("data")
    lookback_trading_days: int = DEFAULT_LOOKBACK_TRADING_DAYS
    request_sleep_seconds: float = 0.0
    refresh_latest: bool = True
    verbose: bool = False

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
    def suspend_dir(self) -> Path:
        return self.raw_dir / "suspend"

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
        for status in ["L"]:
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

    def suspend_d(self, trade_date: str) -> pd.DataFrame:
        return self._paged_call("suspend_d", trade_date=trade_date, fields=SUSPEND_FIELDS)

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

    def history_5min_range(self, ts_code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        url = f"https://api.mairuiapi.com/hsstock/history/{ts_code}/5/n/{self.licence}"
        response = self.session.get(url, params={"st": start_date, "et": end_date}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            if str(payload.get("code", "")) not in {"0", "200", ""}:
                raise RuntimeError(f"mairui returned error for {ts_code} {start_date}-{end_date}: {json.dumps(payload, ensure_ascii=False)}")
            data = payload.get("data", [])
        elif isinstance(payload, list):
            data = payload
        else:
            raise RuntimeError(f"unexpected mairui payload type for {ts_code} {start_date}-{end_date}: {type(payload)!r}")
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected mairui data field for {ts_code} {start_date}-{end_date}")
        return [dict(row) for row in data if isinstance(row, dict)]


def _now_utcp8_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai")).date())


def _now_utcp8_time() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai")))


def _log(config: DailyUpdateConfig, message: str) -> None:
    if bool(config.verbose):
        print(f"[update_daily] {message}")


def _ensure_dirs(config: DailyUpdateConfig) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.daily_dir.mkdir(parents=True, exist_ok=True)
    config.min5_dir.mkdir(parents=True, exist_ok=True)
    config.moneyflow_dir.mkdir(parents=True, exist_ok=True)
    config.st_dir.mkdir(parents=True, exist_ok=True)
    config.suspend_dir.mkdir(parents=True, exist_ok=True)


def _normalize_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


def _load_trade_window(pro: TushareClient, lookback_trading_days: int) -> tuple[pd.DataFrame, list[str], str]:
    today = datetime.today()
    if _now_utcp8_time() < pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=17, minute=0, second=0,
                                      microsecond=0, tz=zoneinfo.ZoneInfo("Asia/Shanghai")):
        end_date = (_now_utcp8_date() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    else:
        end_date = _now_utcp8_date().strftime("%Y%m%d")
    start_date = (_now_utcp8_date() - pd.Timedelta(days=max(240, lookback_trading_days * 3))).strftime("%Y%m%d")
    calendar = pro.trade_calendar(start_date=start_date, end_date=end_date)
    print(calendar)
    if calendar is None or calendar.empty:
        raise RuntimeError("trade_cal returned no rows")
    cal = calendar.copy()
    cal["cal_date"] = _normalize_date_str(cal["cal_date"])
    cal["pretrade_date"] = _normalize_date_str(cal.get("pretrade_date", pd.Series(index=cal.index, dtype=object)))
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    cal = cal.dropna(subset=["cal_date"]).drop_duplicates(subset=["cal_date"], keep="last").sort_values("cal_date").reset_index(drop=True)
    open_dates = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()
    need = max(1, int(lookback_trading_days))
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


def _select_pending_trade_dates(target_dates: list[str], out_dir: Path, refresh_latest: bool) -> list[str]:
    if not target_dates:
        return []
    latest_trade_date = target_dates[-1]
    pending: list[str] = []
    for trade_date in target_dates:
        out_path = out_dir / f"{trade_date}.parquet"
        if _need_refresh(out_path, refresh_latest, latest_trade_date):
            pending.append(trade_date)
    return pending


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


def _build_daily_snapshot_for_date(pro: TushareClient, trade_date: str, stock_basic: pd.DataFrame) -> pd.DataFrame:
    daily_df = pro.daily(trade_date)
    adj_df = pro.adj_factor(trade_date)
    return _merge_daily_and_adj(daily_df, adj_df, trade_date, stock_basic)


def _update_daily_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    for trade_date in _select_pending_trade_dates(target_dates, config.daily_dir, config.refresh_latest):
        out_path = config.daily_dir / f"{trade_date}.parquet"
        _log(config, f"updating daily snapshot: {trade_date}")
        _build_daily_snapshot_for_date(pro, trade_date, stock_basic).to_parquet(out_path, index=False)
        updated.append(trade_date)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return updated


def _build_moneyflow_snapshot_for_date(pro: TushareClient, trade_date: str, stock_basic: pd.DataFrame) -> pd.DataFrame:
    df = pro.moneyflow(trade_date)
    if df is None:
        df = pd.DataFrame(columns=MONEYFLOW_FIELDS.split(","))
    stock_cols = [c for c in ["ts_code", "name", "symbol", "industry", "market"] if c in stock_basic.columns]
    stock_view = stock_basic[stock_cols] if stock_cols else pd.DataFrame(columns=["ts_code"])
    if not df.empty:
        df["trade_date"] = _normalize_date_str(df["trade_date"])
        if not stock_view.empty:
            df = df.merge(stock_view, on="ts_code", how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _update_moneyflow_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    for trade_date in _select_pending_trade_dates(target_dates, config.moneyflow_dir, config.refresh_latest):
        out_path = config.moneyflow_dir / f"{trade_date}.parquet"
        _log(config, f"updating moneyflow snapshot: {trade_date}")
        _build_moneyflow_snapshot_for_date(pro, trade_date, stock_basic).to_parquet(out_path, index=False)
        updated.append(trade_date)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return updated


def _build_suspend_snapshot_for_date(pro: TushareClient, trade_date: str, stock_basic: pd.DataFrame) -> pd.DataFrame:
    df = pro.suspend_d(trade_date)
    if df is None:
        df = pd.DataFrame(columns=SUSPEND_FIELDS.split(","))
    if not df.empty:
        df["trade_date"] = _normalize_date_str(df["trade_date"])
        base_cols = [c for c in ["ts_code", "name", "symbol", "industry", "market"] if c in stock_basic.columns]
        if base_cols:
            df = df.merge(stock_basic[base_cols], on="ts_code", how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _update_suspend_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    for trade_date in _select_pending_trade_dates(target_dates, config.suspend_dir, config.refresh_latest):
        out_path = config.suspend_dir / f"{trade_date}.parquet"
        _log(config, f"updating suspend snapshot: {trade_date}")
        _build_suspend_snapshot_for_date(pro, trade_date, stock_basic).to_parquet(out_path, index=False)
        updated.append(trade_date)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
    return updated


def _load_suspend_map(suspend_dir: Path, target_dates: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {d: set() for d in target_dates}
    for trade_date in target_dates:
        path = suspend_dir / f"{trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "ts_code" not in df.columns:
            continue
        out[trade_date] = {str(x) for x in df["ts_code"].dropna().astype(str).tolist() if str(x)}
    return out


def _compute_code_open_dates(stock_basic: pd.DataFrame, target_dates: list[str], suspend_map: dict[str, set[str]]) -> dict[str, list[str]]:
    code_dates: dict[str, list[str]] = {}
    if not target_dates or stock_basic.empty:
        return code_dates
    all_codes = stock_basic["ts_code"].dropna().astype(str).tolist() if "ts_code" in stock_basic.columns else []
    for code in all_codes:
        open_dates = [d for d in target_dates if code not in suspend_map.get(d, set())]
        if not open_dates:
            continue
        code_dates[code] = open_dates[-REAL_OPEN_LOOKBACK_DAYS:]
    return code_dates


def _select_pending_5min_code_windows(
    config: DailyUpdateConfig,
    code_open_dates: dict[str, list[str]],
) -> dict[str, tuple[str, str]]:
    pending: dict[str, tuple[str, str]] = {}
    for code, open_dates in code_open_dates.items():
        if not open_dates:
            continue
        code_dir = config.min5_dir / code
        missing_dates: list[str] = []
        latest_trade_date = open_dates[-1]
        for trade_date in open_dates:
            out_path = code_dir / f"{trade_date}.csv"
            if _need_refresh(out_path, config.refresh_latest, latest_trade_date):
                missing_dates.append(trade_date)
        if missing_dates:
            pending[code] = (missing_dates[0], missing_dates[-1])
    return pending


def _extract_trade_date_from_5min_row(row: dict[str, Any], default_trade_date: str | None = None) -> str | None:
    for key in ["trade_date", "date", "day", "dt", "datetime", "time", "tm", "d", "t"]:
        value = row.get(key)
        if value is None:
            continue
        ts = pd.to_datetime(value, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y%m%d")
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return default_trade_date


def _normalize_mairui_rows(ts_code: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        trade_date = _extract_trade_date_from_5min_row(row)
        if not trade_date:
            continue
        item = dict(row)
        item.setdefault("ts_code", ts_code)
        item["trade_date"] = trade_date
        item["row_no"] = idx
        normalized.append(item)
    return normalized


def _write_code_5min_csvs(df: pd.DataFrame, code_dir: Path) -> list[str]:
    if df.empty or "trade_date" not in df.columns:
        return []
    code_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    for trade_date, sub in df.groupby("trade_date", sort=False):
        out_path = code_dir / f"{trade_date}.csv"
        sub.sort_values([c for c in ["trade_date", "row_no"] if c in sub.columns]).to_csv(out_path, index=False)
        written.append(str(trade_date))
    return written


def _fetch_single_code_5min_to_csv(
    ts_code: str,
    start_date: str,
    end_date: str,
    out_root: str,
    sleep_seconds: float,
    verbose: bool,
) -> dict[str, Any]:
    mairui = MairuiClient(os.getenv("MAIRUI_LICENCE", "").strip())
    rows = mairui.history_5min_range(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    df = pd.DataFrame(_normalize_mairui_rows(ts_code, rows))
    code_dir = Path(out_root) / ts_code
    written_dates = _write_code_5min_csvs(df, code_dir)
    if verbose and len(written_dates) > 0:
        print(f"[update_daily] wrote 5min csvs for {ts_code}: {len(written_dates)} days")
    return {"ts_code": ts_code, "start_date": start_date, "end_date": end_date, "written_dates": written_dates}


def _update_5min_files(config: DailyUpdateConfig, code_windows: dict[str, tuple[str, str]]) -> list[str]:
    if not code_windows:
        return []
    updated_codes: list[str] = []
    _log(config, f"updating 5min by code with {DEFAULT_5MIN_PROCESS_WORKERS} processes")
    with ProcessPoolExecutor(max_workers=DEFAULT_5MIN_PROCESS_WORKERS) as executor:
        futures = {
            executor.submit(
                _fetch_single_code_5min_to_csv,
                code,
                start_date,
                end_date,
                str(config.min5_dir),
                config.request_sleep_seconds,
                bool(config.verbose),
            ): code
            for code, (start_date, end_date) in code_windows.items()
        }
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                fut.result()
                updated_codes.append(code)
            except Exception as exc:
                raise RuntimeError(f"failed to fetch 5min csvs for {code}") from exc
    return sorted(updated_codes)


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


def _build_st_snapshot_for_date(trade_date: str, stock_basic: pd.DataFrame, name_history: dict[str, list[tuple[pd.Timestamp, str]]]) -> pd.DataFrame:
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
        rows.append({"trade_date": trade_date, "ts_code": ts_code, "name": effective_name, "is_st": _is_st_name(effective_name)})
    return pd.DataFrame(rows).sort_values(["ts_code"]).reset_index(drop=True)


def _update_st_files(pro: TushareClient, config: DailyUpdateConfig, target_dates: list[str], stock_basic: pd.DataFrame) -> list[str]:
    updated: list[str] = []
    namechange = pro.namechange(end_date=target_dates[-1])
    name_history = _build_st_name_history(stock_basic, namechange)
    for trade_date in _select_pending_trade_dates(target_dates, config.st_dir, config.refresh_latest):
        out_path = config.st_dir / f"{trade_date}.parquet"
        _log(config, f"updating st snapshot: {trade_date}")
        _build_st_snapshot_for_date(trade_date, stock_basic, name_history).to_parquet(out_path, index=False)
        updated.append(trade_date)
    return updated


def update_daily_market_data(config: DailyUpdateConfig) -> dict[str, Any]:
    _ensure_dirs(config)
    tushare = TushareClient(os.getenv("TUSHARE_TOKEN", "").strip())

    calendar, target_dates, anchor_date = _load_trade_window(tushare, config.lookback_trading_days)
    _log(config, f"anchor trade date: {anchor_date}; raw window dates: {len(target_dates)}")
    stock_basic = _write_calendar_and_stock_basic(config, calendar, tushare.stock_basic(), target_dates)

    suspend_pending = _select_pending_trade_dates(target_dates, config.suspend_dir, config.refresh_latest)
    suspend_updated = _update_suspend_files(tushare, config, target_dates, stock_basic)
    suspend_map = _load_suspend_map(config.suspend_dir, target_dates)
    code_open_dates = _compute_code_open_dates(stock_basic, target_dates, suspend_map)
    code_windows = _select_pending_5min_code_windows(config, code_open_dates)

    daily_pending = _select_pending_trade_dates(target_dates, config.daily_dir, config.refresh_latest)
    moneyflow_pending = _select_pending_trade_dates(target_dates, config.moneyflow_dir, config.refresh_latest)
    st_pending = _select_pending_trade_dates(target_dates, config.st_dir, config.refresh_latest)

    daily_updated = _update_daily_files(tushare, config, target_dates, stock_basic)
    moneyflow_updated = _update_moneyflow_files(tushare, config, target_dates, stock_basic)
    min5_updated_codes = _update_5min_files(config, code_windows)
    st_updated = _update_st_files(tushare, config, target_dates, stock_basic)

    meta = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_trade_date": anchor_date,
        "raw_trade_dates": target_dates,
        "lookback_trading_days": int(config.lookback_trading_days),
        "real_open_lookback_days": REAL_OPEN_LOOKBACK_DAYS,
        "daily_pending": daily_pending,
        "moneyflow_pending": moneyflow_pending,
        "suspend_pending": suspend_pending,
        "st_pending": st_pending,
        "daily_updated": daily_updated,
        "moneyflow_updated": moneyflow_updated,
        "suspend_updated": suspend_updated,
        "min5_updated_codes": min5_updated_codes,
        "st_updated": st_updated,
        "stock_basic_rows": int(len(stock_basic)),
        "code_windows": {k: {"L": v[0], "T": v[1]} for k, v in code_windows.items()},
        "code_open_dates": code_open_dates,
    }
    config.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
