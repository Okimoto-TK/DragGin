from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


NAMECHANGE_FIELDS = ["ts_code", "name", "start_date", "end_date"]
MAX_RETRIES = 10


def _get_pro_client(token: str):
    import tushare as ts

    ts.set_token(token)
    return ts.pro_api(token)


def _iter_ts_codes_from_data_dir(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    codes = [p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(set(codes))


def _fetch_namechange_by_code(pro, ts_code: str, sleep_seconds: float = 0.0, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = pro.namechange(ts_code=ts_code, fields=",".join(NAMECHANGE_FIELDS))
            if df is None:
                df = pd.DataFrame(columns=NAMECHANGE_FIELDS)
            if not df.empty:
                df = df.reindex(columns=NAMECHANGE_FIELDS)
                df = df.drop_duplicates(subset=["ts_code", "start_date", "name"], keep="last")
                df = df.sort_values(["ts_code", "start_date", "end_date", "name"], na_position="last")
            return df.reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch namechange for ts_code={ts_code} after {max_retries} retries") from last_error


def _fetch_single_code(token: str, ts_code: str, sleep_seconds: float = 0.0, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    pro = _get_pro_client(token)
    return _fetch_namechange_by_code(pro=pro, ts_code=ts_code, sleep_seconds=sleep_seconds, max_retries=max_retries)


def fetch_namechange_from_data_dir(
    data_dir: Path,
    out_file: Path,
    sleep_seconds: float = 0.0,
    max_workers: int = 4,
) -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")

    ts_codes = _iter_ts_codes_from_data_dir(data_dir)
    if not ts_codes:
        raise RuntimeError(f"No stock folders found under data_dir={data_dir}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {
            executor.submit(_fetch_single_code, token, ts_code, sleep_seconds, MAX_RETRIES): ts_code
            for ts_code in ts_codes
        }

        with tqdm(total=len(future_map), desc="Fetching namechange") as pbar:
            for future in as_completed(future_map):
                pbar.update(1)
                df = future.result()
                if not df.empty:
                    parts.append(df)

    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = out.reindex(columns=NAMECHANGE_FIELDS)
        out = out.drop_duplicates(subset=["ts_code", "start_date", "name"], keep="last")
        out = out.sort_values(["ts_code", "start_date", "end_date", "name"], na_position="last").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=NAMECHANGE_FIELDS)

    out.to_parquet(out_file, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tushare namechange data for stock folders under --data-dir")
    parser.add_argument("--data-dir", required=True, help="Data directory whose child folder names are ts_code values")
    parser.add_argument("--out-file", help="Output parquet path, default is <data-dir>/namechange.parquet")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between retries")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads for fetching codes")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_file = Path(args.out_file) if args.out_file else data_dir / "namechange.parquet"
    fetch_namechange_from_data_dir(
        data_dir=data_dir,
        out_file=out_file,
        sleep_seconds=max(0.0, args.sleep),
        max_workers=max(1, args.max_workers),
    )


if __name__ == "__main__":
    main()
