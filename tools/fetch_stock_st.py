from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


PRIVATE_TUSHARE_URL = "http://lianghua.nanyangqiankun.top"
STOCK_ST_FIELDS = "ts_code,name,start_date,end_date,change_reason"
MAX_RETRIES = 10
PAGE_LIMIT = 6000


def _create_pro_client(token: str | None = None):
    import tushare as ts

    clean_token = (token or "").strip()
    if clean_token:
        ts.set_token(clean_token)
        pro = ts.pro_api(clean_token)
        pro._DataApi__token = clean_token
    else:
        pro = ts.pro_api()
    pro._DataApi__http_url = PRIVATE_TUSHARE_URL
    return pro


def _iter_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(
        exchange="",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date,is_open",
    )
    if cal is None or cal.empty or "cal_date" not in cal.columns:
        return []
    return cal["cal_date"].astype(str).sort_values().drop_duplicates().tolist()


def _normalize_stock_st_frame(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out = pd.DataFrame(columns=["trade_date", "ts_code", "name", "start_date", "end_date", "change_reason"])
    if "trade_date" not in out.columns:
        out.insert(0, "trade_date", trade_date)
    else:
        out["trade_date"] = out["trade_date"].astype("string").fillna(trade_date)
    desired = ["trade_date", "ts_code", "name", "start_date", "end_date", "change_reason"]
    for col in desired:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[desired]
    out = out.drop_duplicates(subset=["trade_date", "ts_code", "start_date", "end_date"], keep="last")
    return out.sort_values(["trade_date", "ts_code", "start_date", "end_date"], na_position="last").reset_index(drop=True)


def _fetch_stock_st_by_date(pro, trade_date: str, sleep_seconds: float = 0.0, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            parts: list[pd.DataFrame] = []
            offset = 0

            while True:
                df = pro.stock_st(
                    trade_date=trade_date,
                    offset=offset,
                    limit=PAGE_LIMIT,
                    fields=STOCK_ST_FIELDS,
                )
                if df is None or df.empty:
                    break
                parts.append(df)
                if len(df) < PAGE_LIMIT:
                    break
                offset += PAGE_LIMIT
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            if parts:
                return _normalize_stock_st_frame(pd.concat(parts, ignore_index=True), trade_date)

            empty = pd.DataFrame(columns=STOCK_ST_FIELDS.split(","))
            return _normalize_stock_st_frame(empty, trade_date)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch stock_st for trade_date={trade_date} after {max_retries} retries") from last_error


def _worker_main(token: str, out_dir: str, trade_date: str, sleep_seconds: float, max_retries: int, error_queue: mp.Queue) -> None:
    try:
        pro = _create_pro_client(token)
        df = _fetch_stock_st_by_date(pro=pro, trade_date=trade_date, sleep_seconds=sleep_seconds, max_retries=max_retries)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(out_dir) / f"{trade_date}_stock_st.parquet", index=False)
        error_queue.put((trade_date, None))
    except Exception as exc:  # pragma: no cover - exercised via parent process handling
        error_queue.put((trade_date, repr(exc)))


def fetch_stock_st_range(
    start_date: str,
    end_date: str,
    out_dir: Path,
    token: str | None = None,
    sleep_seconds: float = 0.0,
    max_workers: int = 4,
) -> None:
    clean_token = (token or os.getenv("TUSHARE_TOKEN", "")).strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    pro = _create_pro_client(clean_token)
    trade_dates = _iter_trade_dates(pro=pro, start_date=start_date, end_date=end_date)
    if not trade_dates:
        raise RuntimeError("No open trade dates found from trade_cal in the requested range")

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    error_queue = ctx.Queue()
    workers = max(1, min(max_workers, len(trade_dates)))
    active: dict[str, mp.Process] = {}
    pending = list(trade_dates)
    completed = 0

    with tqdm(total=len(trade_dates), desc="Fetching stock_st by date") as pbar:
        while pending or active:
            while pending and len(active) < workers:
                trade_date = pending.pop(0)
                proc = ctx.Process(
                    target=_worker_main,
                    args=(clean_token, str(out_dir), trade_date, sleep_seconds, MAX_RETRIES, error_queue),
                )
                proc.start()
                active[trade_date] = proc

            try:
                finished_trade_date, err = error_queue.get(timeout=0.2)
            except Empty:
                finished_trade_date = None
                err = None

            if finished_trade_date is None:
                continue

            proc = active.pop(finished_trade_date)
            proc.join()
            if proc.exitcode not in (0, None) or err is not None:
                for other in active.values():
                    other.terminate()
                for other in active.values():
                    other.join()
                raise RuntimeError(f"stock_st worker failed for trade_date={finished_trade_date}: {err or f'exitcode={proc.exitcode}'}")

            completed += 1
            pbar.update(1)

    if completed != len(trade_dates):
        raise RuntimeError(f"Expected {len(trade_dates)} completed trade dates, got {completed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tushare stock_st data and save one parquet per trade date")
    parser.add_argument("--st", required=True, help="Start date in YYYYmmdd")
    parser.add_argument("--et", required=True, help="End date in YYYYmmdd")
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    parser.add_argument("--token", default="", help="Optional Tushare token; falls back to TUSHARE_TOKEN")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between paginated requests")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker processes for fetching dates")
    args = parser.parse_args()

    fetch_stock_st_range(
        start_date=args.st,
        end_date=args.et,
        out_dir=Path(args.out_dir),
        token=args.token,
        sleep_seconds=max(0.0, args.sleep),
        max_workers=max(1, args.max_workers),
    )


if __name__ == "__main__":
    main()
