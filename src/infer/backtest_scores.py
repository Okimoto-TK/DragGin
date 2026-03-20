from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_SCORE_COLUMNS = {"code", "asof_date", "yhat"}


def load_offline_scores(score_dir: Path) -> pd.DataFrame:
    """Load merged or sharded offline inference scores for backtesting."""
    score_file = score_dir / "scores.parquet"
    if score_file.exists():
        df = pd.read_parquet(score_file)
    else:
        shard_dir = score_dir / "score_shards"
        files = sorted(shard_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"cannot find offline score outputs in: {score_dir}")
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    missing = REQUIRED_SCORE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"offline score missing columns: {missing}")

    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df = df.dropna(subset=["code", "asof_date"])
    return df.sort_values(["code", "asof_date"]).reset_index(drop=True)


def build_score_by_date_with_ffill(asof_dates: list[str], codes: list[str], score_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Forward-fill score history per code, then re-index by asof_date."""
    grouped: dict[str, list[tuple[str, float]]] = {}
    for code, sub in score_df.groupby("code", sort=False):
        pairs = [(str(d), float(v)) for d, v in zip(sub["asof_date"].tolist(), sub["yhat"].tolist()) if np.isfinite(v)]
        if pairs:
            grouped[str(code)] = pairs

    score_by_date: dict[str, list[tuple[str, float]]] = {d: [] for d in asof_dates}
    for code in codes:
        hist = grouped.get(code, [])
        ptr = 0
        cur: float | None = None
        for d in asof_dates:
            while ptr < len(hist) and hist[ptr][0] <= d:
                cur = hist[ptr][1]
                ptr += 1
            if cur is not None:
                score_by_date[d].append((code, cur))
    return score_by_date


def merge_score_shards(score_dir: Path) -> pd.DataFrame:
    files = sorted(score_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no score shard parquet files found in: {score_dir}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
