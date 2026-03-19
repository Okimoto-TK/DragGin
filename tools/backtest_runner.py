from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMMISSION_RATE = 1e-4
COMMISSION_MIN = 5.0


@dataclass
class Position:
    code: str
    qty: float
    cost_per_share: float
    last_adj_factor: float

    @property
    def cost_total(self) -> float:
        return float(self.qty) * float(self.cost_per_share)


def _commission(amount: float) -> float:
    return max(COMMISSION_MIN, float(amount) * COMMISSION_RATE)


def _to_datestr(v: Any) -> str:
    ts_ = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts_):
        raise ValueError(f"invalid date value: {v}")
    return ts_.strftime("%Y-%m-%d")


def _read_calendar_dates(data_dir: Path) -> list[str]:
    cal_path = data_dir / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar file not found: {cal_path}")
    df = pd.read_parquet(cal_path)
    for col in ["trade_date", "cal_date", "date"]:
        if col in df.columns:
            values = sorted({_to_datestr(x) for x in df[col].dropna().tolist()})
            if values:
                return values
    raise ValueError(f"cannot find date column in {cal_path}")


def _resolve_codes(data_dir: Path) -> list[str]:
    codes: list[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "daily.parquet").exists() and (p / "5min.parquet").exists() and (p / "moneyflow.parquet").exists():
            codes.append(p.name)
    return codes


def _expand_paths(raw_paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_paths:
        if any(ch in raw for ch in "*?[]"):
            expanded.extend(str(p) for p in sorted(Path().glob(raw)))
        else:
            expanded.append(raw)
    return expanded


def _resolve_val_dates(calendar_dates: list[str], val_ratio: float, val_embargo_days: int, val_shards: list[str]) -> list[str]:
    if val_shards:
        out: set[str] = set()
        for path in val_shards:
            payload = np.load(path, allow_pickle=True).item()
            if "asof_dates" not in payload:
                raise KeyError(f"missing asof_dates in shard: {path}")
            out.update(str(x) for x in payload["asof_dates"].astype(str).tolist())
        return sorted(out)

    unique_dates = sorted(set(calendar_dates))
    embargo_days = max(0, int(val_embargo_days))
    ratio = float(min(max(val_ratio, 0.0), 0.99))
    usable_for_train_val = len(unique_dates) - embargo_days
    if usable_for_train_val <= 1:
        raise ValueError("insufficient calendar dates for val split")
    val_dates_count = int(round(usable_for_train_val * ratio))
    val_dates_count = max(1, min(usable_for_train_val - 1, val_dates_count))
    train_dates_count = usable_for_train_val - val_dates_count
    return sorted(unique_dates[train_dates_count + embargo_days :])


def _load_daily_map(data_dir: Path, code: str) -> dict[str, dict[str, float]]:
    p = data_dir / code / "daily.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["trade_date", "open", "high", "low", "close", "adj_factor"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "adj_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["trade_date"])] = {
            "open": float(row["open"]) if np.isfinite(row["open"]) else math.nan,
            "high": float(row["high"]) if np.isfinite(row["high"]) else math.nan,
            "low": float(row["low"]) if np.isfinite(row["low"]) else math.nan,
            "close": float(row["close"]) if np.isfinite(row["close"]) else math.nan,
            "adj_factor": float(row["adj_factor"]) if np.isfinite(row["adj_factor"]) else math.nan,
        }
    return out



def _load_st_flags(st_dir: Path | None) -> dict[str, set[str]]:
    if st_dir is None:
        return {}
    if not st_dir.exists():
        raise FileNotFoundError(f"st dir not found: {st_dir}")

    out: dict[str, set[str]] = {}
    for path in sorted(st_dir.glob("*_stock_st.parquet")):
        stem = path.stem
        date_token = stem[:8]
        if len(date_token) != 8 or not date_token.isdigit():
            continue
        df = pd.read_parquet(path, columns=["ts_code"])
        if "ts_code" not in df.columns:
            raise ValueError(f"missing ts_code column in {path}")
        codes = {str(x) for x in df["ts_code"].dropna().astype(str).tolist() if str(x)}
        out[date_token] = codes
    return out


def _is_filtered_buy_code(code: str, st_codes: set[str]) -> bool:
    prefix = str(code)[:3]
    return str(code) in st_codes or prefix in {"300", "688"}


def _round_lot_shares(max_cash: float, price: float) -> int:
    if price <= 0 or max_cash <= 0:
        return 0
    lots = int(max_cash // (price * 100.0))
    return max(0, lots * 100)


def _fmt_money(v: float) -> str:
    return f"{float(v):,.2f}"


def _print_day_summary(payload: dict[str, Any]) -> None:
    day = str(payload.get("date", ""))
    asof = str(payload.get("asof_date", ""))
    print("\n" + "=" * 88)
    print(f"回测日报 | 交易日: {day} | asof: {asof}")
    print("-" * 88)
    print(
        "期初资产  持仓: {0} | 现金: {1} | 总资产: {2}".format(
            _fmt_money(payload.get("initial_holding_amount", 0.0)),
            _fmt_money(payload.get("initial_cash", 0.0)),
            _fmt_money(payload.get("initial_total_asset", 0.0)),
        )
    )

    init_positions = payload.get("initial_positions", []) or []
    if init_positions:
        print("\n[期初持仓]")
        for row in init_positions:
            print(
                "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10} 浮盈亏(开盘):{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=f"{float(row.get('qty', 0.0)):.4f}",
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                    pnl=_fmt_money(float(row.get("float_pnl", 0.0))),
                )
            )

    sells = payload.get("sell_records", []) or []
    if sells:
        print("\n[卖出记录]")
        for row in sells:
            print(
                "  - {code:<10} 数量:{qty:>10} 成交:{price:>10} 成本:{cost:>10} 实现盈亏:{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=f"{float(row.get('qty', 0.0)):.4f}",
                    price=f"{float(row.get('price', 0.0)):.4f}",
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                    pnl=_fmt_money(float(row.get("realized_pnl", 0.0))),
                )
            )

    buys = payload.get("buy_records", []) or []
    if buys:
        print("\n[买入记录]")
        for row in buys:
            print(
                "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10}".format(
                    code=str(row.get("code", "")),
                    qty=f"{float(row.get('qty', 0.0)):.4f}",
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                )
            )

    final_positions = payload.get("final_positions", []) or []
    if final_positions:
        print("\n[期末持仓]")
        for row in final_positions:
            print(
                "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10} 浮盈亏(收盘):{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=f"{float(row.get('qty', 0.0)):.4f}",
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                    pnl=_fmt_money(float(row.get("float_pnl", 0.0))),
                )
            )

    print("\n" + "-" * 88)
    print(
        "期末资产  持仓: {0} | 现金: {1} | 总资产: {2}".format(
            _fmt_money(payload.get("final_holding_amount", 0.0)),
            _fmt_money(payload.get("final_cash", 0.0)),
            _fmt_money(payload.get("final_total_asset", 0.0)),
        )
    )
    print("=" * 88)


def _load_offline_scores(score_dir: Path) -> pd.DataFrame:
    score_file = score_dir / "scores.parquet"
    if score_file.exists():
        df = pd.read_parquet(score_file)
    else:
        shard_dir = score_dir / "score_shards"
        files = sorted(shard_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"cannot find offline score outputs in: {score_dir}")
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    need_cols = {"code", "asof_date", "yhat"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"offline score missing columns: {need_cols - set(df.columns)}")
    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df = df.dropna(subset=["code", "asof_date"])
    return df.sort_values(["code", "asof_date"]).reset_index(drop=True)


def _build_score_by_date_with_ffill(asof_dates: list[str], codes: list[str], score_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    # 对每只股票按日期 forward fill，停牌等导致当日缺失 yhat 时延续旧信号。
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


def _mark_price(day_row: dict[str, float] | None, field: str, fallback: float) -> float:
    if day_row is None:
        return fallback
    v = float(day_row.get(field, math.nan))
    return v if np.isfinite(v) and v > 0 else fallback


def _apply_adj_factor_before_open(pos: Position, day_row: dict[str, float] | None) -> None:
    if day_row is None:
        return
    adj_today = float(day_row.get("adj_factor", math.nan))
    if (not np.isfinite(adj_today)) or adj_today <= 0 or (not np.isfinite(pos.last_adj_factor)) or pos.last_adj_factor <= 0:
        return
    if abs(adj_today - pos.last_adj_factor) < 1e-12:
        return
    # 最简单稳健方案：统一做“持仓等价调整”，不单独记分红现金。
    ratio = adj_today / pos.last_adj_factor
    if (not np.isfinite(ratio)) or ratio <= 0:
        return
    pos.qty = float(pos.qty) * float(ratio)
    pos.cost_per_share = float(pos.cost_per_share) / float(ratio)
    pos.last_adj_factor = float(adj_today)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model backtest runner (offline scores)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--score-dir", required=True, help="offline score dir generated by build_backtest_batches.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--buy-gate", type=float, default=1.0)
    parser.add_argument("--sell-gate", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-embargo-days", type=int, default=30)
    parser.add_argument("--val-shards", nargs="+", default=None)
    parser.add_argument("--ts-token", default="")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--st-dir", default="", help="directory containing YYYYmmdd_stock_st.parquet files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calendar_dates = _read_calendar_dates(data_dir)
    val_shards = _expand_paths(args.val_shards or [])
    val_dates = _resolve_val_dates(calendar_dates, args.val_ratio, args.val_embargo_days, val_shards)
    if len(val_dates) < 6:
        raise ValueError("validation dates must be >= 6 to backtest until last-5 day")
    asof_dates = val_dates[:-5]

    codes = _resolve_codes(data_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found")

    score_df = _load_offline_scores(score_dir)
    score_by_date = _build_score_by_date_with_ffill(asof_dates=asof_dates, codes=codes, score_df=score_df)

    token = args.ts_token or str(os.environ.get("TUSHARE_TOKEN", ""))
    pro = ts.pro_api(token) if token else None
    limit_cache: dict[str, dict[str, tuple[float, float]]] = {}

    daily_cache = {c: _load_daily_map(data_dir, c) for c in codes}
    st_flags_by_day = _load_st_flags(Path(args.st_dir) if args.st_dir else None)

    cash = float(args.initial_cash)
    positions: dict[str, Position] = {}
    prev_final_holding_amount = 0.0
    prev_confidence = 1.0
    target_n = int(args.topk)

    for t_idx, asof in enumerate(asof_dates):
        day = val_dates[t_idx + 1]
        day_trade = day.replace("-", "")
        rank_rows = sorted(score_by_date.get(asof, []), key=lambda x: x[1], reverse=True)
        st_codes_today = st_flags_by_day.get(day_trade, set())
        score_map = {code: float(score) for code, score in rank_rows}
        if t_idx == 0:
            target_n = int(args.topk)
        else:
            target_n = int(round(target_n * prev_confidence))
            target_n = max(1, min(int(args.topk), target_n))

        if day_trade not in limit_cache:
            lm: dict[str, tuple[float, float]] = {}
            if pro is not None:
                try:
                    ldf = pro.stk_limit(trade_date=day_trade)
                    if ldf is not None and not ldf.empty:
                        for _, r in ldf.iterrows():
                            c = str(r.get("ts_code", ""))
                            up = float(r.get("up_limit", np.nan))
                            dn = float(r.get("down_limit", np.nan))
                            if c:
                                lm[c] = (up, dn)
                except Exception:
                    lm = {}
            limit_cache[day_trade] = lm

        # 开盘前先应用公司行为等价调整。
        for code, pos in positions.items():
            _apply_adj_factor_before_open(pos, daily_cache.get(code, {}).get(day))

        initial_cash = cash
        initial_positions_amount = prev_final_holding_amount
        initial_total_asset = initial_cash + initial_positions_amount

        init_pos_details: list[dict[str, Any]] = []
        sell_records: list[dict[str, Any]] = []
        buy_records: list[dict[str, Any]] = []

        sold_today: set[str] = set()

        # 统一口径：即使缺行情，也在期初明细中展示（按成本估值，浮盈亏=0）。
        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            mark_open = _mark_price(day_row, "open", pos.cost_per_share)
            init_pos_details.append(
                {
                    "code": code,
                    "cost": round(pos.cost_per_share, 6),
                    "qty": round(float(pos.qty), 6),
                    "float_pnl": round(mark_open * pos.qty - pos.cost_total, 4),
                }
            )

        # 先卖出
        for code, pos in list(positions.items()):
            day_row = daily_cache.get(code, {}).get(day)
            # 缺行情不交易
            if day_row is None:
                continue
            open_p = float(day_row.get("open", math.nan))
            if (not np.isfinite(open_p)) or open_p <= 0:
                continue

            updn = limit_cache[day_trade].get(code, (math.nan, math.nan))
            down_limit = float(updn[1])

            sell_price: float | None = None
            is_st_stock = code in st_codes_today
            score_today = score_map.get(code, math.nan)
            # ST/*ST 持仓优先在开盘尝试卖出，跌停不可卖。
            if np.isfinite(down_limit) and abs(open_p - down_limit) < 1e-8:
                sell_price = None
            elif is_st_stock:
                sell_price = open_p
            elif np.isfinite(score_today) and score_today < float(args.sell_gate):
                sell_price = open_p
            elif open_p / pos.cost_per_share < 0.925:
                sell_price = open_p

            if sell_price is None:
                continue

            proceeds = sell_price * pos.qty
            sell_fee = _commission(proceeds)
            cash += proceeds - sell_fee
            realized = (proceeds - sell_fee) - pos.cost_total
            sell_records.append(
                {
                    "code": code,
                    "cost": round(pos.cost_per_share, 6),
                    "price": round(sell_price, 6),
                    "qty": round(float(pos.qty), 6),
                    "realized_pnl": round(realized, 4),
                }
            )
            sold_today.add(code)
            del positions[code]

        # 卖出后再计算买入额度：不使用未来信息，不在买入循环中动态重算持仓价值。
        holding_value_after_sell = 0.0
        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            mark_open = _mark_price(day_row, "open", pos.cost_per_share)
            holding_value_after_sell += mark_open * pos.qty
        total_asset_for_buy = cash + holding_value_after_sell
        per_position_cap = total_asset_for_buy / max(1, int(args.topk))

        # 再买入
        for code, score in rank_rows:
            if code in positions or code in sold_today:
                continue
            if score <= float(args.buy_gate):
                break
            if _is_filtered_buy_code(code, st_codes_today):
                continue

            day_row = daily_cache.get(code, {}).get(day)
            # 缺行情不交易
            if day_row is None:
                continue
            open_p = float(day_row.get("open", math.nan))
            if (not np.isfinite(open_p)) or open_p <= 0:
                continue

            updn = limit_cache[day_trade].get(code, (math.nan, math.nan))
            up_limit = float(updn[0])
            if np.isfinite(up_limit) and abs(open_p - up_limit) < 1e-8:
                continue

            qty = _round_lot_shares(min(per_position_cap, cash), open_p)
            if qty < 100:
                continue

            gross = open_p * qty
            buy_fee = _commission(gross)
            total_needed = gross + buy_fee
            if total_needed > cash:
                qty = _round_lot_shares(max(0.0, cash - COMMISSION_MIN), open_p)
                gross = open_p * qty
                buy_fee = _commission(gross) if qty > 0 else 0.0
                total_needed = gross + buy_fee
            if qty < 100 or total_needed > cash:
                continue

            cash -= total_needed
            adj = float(day_row.get("adj_factor", math.nan))
            if (not np.isfinite(adj)) or adj <= 0:
                adj = 1.0
            pos = Position(
                code=code,
                qty=float(qty),
                cost_per_share=float((gross + buy_fee) / qty),
                last_adj_factor=float(adj),
            )
            positions[code] = pos
            buy_records.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": round(float(pos.qty), 6)})
            if len(positions) >= target_n:
                break

        # 期末统一口径：缺行情同样按成本估值，保证初末明细和资产自洽。
        final_pos_details: list[dict[str, Any]] = []
        final_holding_amount = 0.0
        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            mark_close = _mark_price(day_row, "close", pos.cost_per_share)
            market_value = mark_close * pos.qty
            final_holding_amount += market_value
            final_pos_details.append(
                {
                    "code": code,
                    "cost": round(pos.cost_per_share, 6),
                    "qty": round(float(pos.qty), 6),
                    "float_pnl": round(market_value - pos.cost_total, 4),
                }
            )

        final_total_asset = cash + final_holding_amount
        prev_final_holding_amount = final_holding_amount
        prev_holding_count = len(final_pos_details)
        if prev_holding_count > 0:
            win_count = sum(1 for row in final_pos_details if float(row.get("float_pnl", 0.0)) > 0.0)
            prev_confidence = 0.5 * pow(4, float(win_count) / float(prev_holding_count))
        else:
            prev_confidence = 1

        payload = {
            "asof_date": asof,
            "date": day,
            "initial_holding_amount": round(initial_positions_amount, 4),
            "initial_cash": round(initial_cash, 4),
            "initial_total_asset": round(initial_total_asset, 4),
            "initial_positions": init_pos_details,
            "sell_records": sell_records,
            "buy_records": buy_records,
            "final_positions": final_pos_details,
            "final_holding_amount": round(final_holding_amount, 4),
            "final_cash": round(cash, 4),
            "final_total_asset": round(final_total_asset, 4),
        }
        out_path = out_dir / f"{day_trade}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {out_path}")
        _print_day_summary(payload)


if __name__ == "__main__":
    main()
