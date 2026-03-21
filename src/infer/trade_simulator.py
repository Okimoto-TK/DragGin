from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def commission(amount: float) -> float:
    return max(COMMISSION_MIN, float(amount) * COMMISSION_RATE)


def to_datestr(v: Any) -> str:
    ts_ = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts_):
        raise ValueError(f"invalid date value: {v}")
    return ts_.strftime("%Y-%m-%d")


def read_calendar_dates(data_dir: Path) -> list[str]:
    cal_path = data_dir / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar file not found: {cal_path}")
    df = pd.read_parquet(cal_path)
    for col in ["trade_date", "cal_date", "date"]:
        if col in df.columns:
            values = sorted({to_datestr(x) for x in df[col].dropna().tolist()})
            if values:
                return values
    raise ValueError(f"cannot find date column in {cal_path}")


def resolve_codes(data_dir: Path) -> list[str]:
    codes: list[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "daily.parquet").exists() and (p / "5min.parquet").exists() and (p / "moneyflow.parquet").exists():
            codes.append(p.name)
    return codes


def load_daily_map(data_dir: Path, code: str) -> dict[str, dict[str, float]]:
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


def load_limit_map(data_dir: Path, code: str) -> dict[str, dict[str, float]]:
    p = data_dir / code / "limit.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["trade_date", "up_limit", "down_limit", "limit_pct"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["up_limit", "down_limit", "limit_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["trade_date"])] = {
            "up_limit": float(row["up_limit"]) if np.isfinite(row["up_limit"]) else math.nan,
            "down_limit": float(row["down_limit"]) if np.isfinite(row["down_limit"]) else math.nan,
            "limit_pct": float(row["limit_pct"]) if np.isfinite(row["limit_pct"]) else math.nan,
        }
    return out


def load_latest_position_state(position_dir: Path, daily_cache: dict[str, dict[str, dict[str, float]]]) -> tuple[str | None, float, dict[str, Position]]:
    if not position_dir.exists():
        return None, 0.0, {}
    records = sorted(position_dir.glob("*.json"))
    if not records:
        return None, 0.0, {}
    latest = records[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    record_date = str(payload.get("date", latest.stem))
    cash = float(payload.get("final_cash", 0.0))
    positions: dict[str, Position] = {}
    for row in payload.get("final_positions", []) or []:
        code = str(row.get("code", ""))
        if not code:
            continue
        qty = float(row.get("qty", 0.0))
        cost = float(row.get("cost", 0.0))
        if qty <= 0 or cost <= 0:
            continue
        prev_row = daily_cache.get(code, {}).get(record_date)
        adj = float(prev_row.get("adj_factor", 1.0)) if prev_row else 1.0
        positions[code] = Position(code=code, qty=qty, cost_per_share=cost, last_adj_factor=adj if adj > 0 else 1.0)
    return record_date, cash, positions


def load_st_flags(st_dir: Path | None) -> dict[str, set[str]]:
    if st_dir is None:
        return {}
    if not st_dir.exists():
        raise FileNotFoundError(f"st dir not found: {st_dir}")

    out: dict[str, set[str]] = {}
    paths = sorted({*st_dir.glob("*_stock_st.parquet"), *st_dir.glob("*.parquet")})
    for path in paths:
        stem = path.stem
        date_token = stem[:8]
        if len(date_token) != 8 or not date_token.isdigit():
            continue

        df = pd.read_parquet(path)
        code_col = "ts_code" if "ts_code" in df.columns else ("code" if "code" in df.columns else None)
        if code_col is None:
            raise ValueError(f"missing ts_code/code column in {path}")

        if "is_st" in df.columns:
            df = df[df["is_st"].fillna(0).astype(int) == 1].copy()
        elif "name" in df.columns:
            names = df["name"].astype(str).str.strip().str.upper()
            df = df[names.str.startswith("ST") | names.str.startswith("*ST")].copy()

        codes = {str(x) for x in df[code_col].dropna().astype(str).tolist() if str(x)}
        out[date_token] = codes

    return out


def is_filtered_buy_code(code: str, st_codes: set[str]) -> bool:
    code_str = str(code)
    prefix = code_str[:3]
    return code_str in st_codes or prefix in {"9", "300", "688"}


def round_lot_shares(max_cash: float, price: float) -> int:
    if price <= 0 or max_cash <= 0:
        return 0
    lots = int(max_cash // (price * 100.0))
    return max(0, lots * 100)


def fmt_money(v: float) -> str:
    return f"{float(v):,.2f}"


def print_day_summary(payload: dict[str, Any]) -> None:
    day = str(payload.get("date", ""))
    asof = str(payload.get("asof_date", ""))
    print("\n" + "=" * 88)
    print(f"回测日报 | 交易日: {day} | asof: {asof}")
    print("-" * 88)
    print(
        "期初资产  持仓: {0} | 现金: {1} | 总资产: {2}".format(
            fmt_money(payload.get("initial_holding_amount", 0.0)),
            fmt_money(payload.get("initial_cash", 0.0)),
            fmt_money(payload.get("initial_total_asset", 0.0)),
        )
    )

    for section, title, fmt in [
        (payload.get("initial_positions", []) or [], "期初持仓", lambda row: "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10} 浮盈亏(开盘):{pnl:>12}".format(code=str(row.get("code", "")), qty=f"{float(row.get('qty', 0.0)):.4f}", cost=f"{float(row.get('cost', 0.0)):.4f}", pnl=fmt_money(float(row.get("float_pnl", 0.0))))),
        (payload.get("sell_records", []) or [], "卖出记录", lambda row: "  - {code:<10} 数量:{qty:>10} 成交:{price:>10} 成本:{cost:>10} 实现盈亏:{pnl:>12}".format(code=str(row.get("code", "")), qty=f"{float(row.get('qty', 0.0)):.4f}", price=f"{float(row.get('price', 0.0)):.4f}", cost=f"{float(row.get('cost', 0.0)):.4f}", pnl=fmt_money(float(row.get("realized_pnl", 0.0))))),
        (payload.get("buy_records", []) or [], "买入记录", lambda row: "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10}".format(code=str(row.get("code", "")), qty=f"{float(row.get('qty', 0.0)):.4f}", cost=f"{float(row.get('cost', 0.0)):.4f}")),
        (payload.get("final_positions", []) or [], "期末持仓", lambda row: "  - {code:<10} 数量:{qty:>10} 成本:{cost:>10} 浮盈亏(收盘):{pnl:>12}".format(code=str(row.get("code", "")), qty=f"{float(row.get('qty', 0.0)):.4f}", cost=f"{float(row.get('cost', 0.0)):.4f}", pnl=fmt_money(float(row.get("float_pnl", 0.0))))),
    ]:
        if section:
            print(f"\n[{title}]")
            for row in section:
                print(fmt(row))

    print("\n" + "-" * 88)
    print(
        "期末资产  持仓: {0} | 现金: {1} | 总资产: {2}".format(
            fmt_money(payload.get("final_holding_amount", 0.0)),
            fmt_money(payload.get("final_cash", 0.0)),
            fmt_money(payload.get("final_total_asset", 0.0)),
        )
    )
    print("=" * 88)


def mark_price(day_row: dict[str, float] | None, field: str, fallback: float) -> float:
    if day_row is None:
        return fallback
    v = float(day_row.get(field, math.nan))
    return v if np.isfinite(v) and v > 0 else fallback


def apply_adj_factor_before_open(pos: Position, day_row: dict[str, float] | None) -> None:
    if day_row is None:
        return
    adj_today = float(day_row.get("adj_factor", math.nan))
    if (not np.isfinite(adj_today)) or adj_today <= 0 or (not np.isfinite(pos.last_adj_factor)) or pos.last_adj_factor <= 0:
        return
    if abs(adj_today - pos.last_adj_factor) < 1e-12:
        return
    ratio = adj_today / pos.last_adj_factor
    if (not np.isfinite(ratio)) or ratio <= 0:
        return
    pos.qty = float(pos.qty) * float(ratio)
    pos.cost_per_share = float(pos.cost_per_share) / float(ratio)
    pos.last_adj_factor = float(adj_today)


def run_trade_simulation(
    *,
    data_dir: Path,
    score_by_date: dict[str, list[tuple[str, float]]],
    asof_dates: list[str],
    trading_dates: list[str],
    topk: int,
    buy_gate: float,
    sell_gate: float,
    initial_cash: float,
    initial_positions: dict[str, Position] | None = None,
    pro: Any | None = None,
    st_dir: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    codes = resolve_codes(data_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found")
    if len(asof_dates) != len(trading_dates):
        raise ValueError("asof_dates and trading_dates must have identical length")

    daily_cache = {c: load_daily_map(data_dir, c) for c in codes}
    limit_cache = {c: load_limit_map(data_dir, c) for c in codes}
    st_flags_by_day = load_st_flags(st_dir)

    cash = float(initial_cash)
    positions = {str(code): Position(code=str(pos.code), qty=float(pos.qty), cost_per_share=float(pos.cost_per_share), last_adj_factor=float(pos.last_adj_factor)) for code, pos in (initial_positions or {}).items()}
    prev_final_holding_amount = 0.0
    prev_confidence = 1.0
    target_n = int(topk)
    outputs: list[dict[str, Any]] = []

    for t_idx, (asof, day) in enumerate(zip(asof_dates, trading_dates)):
        day_trade = day.replace("-", "")
        rank_rows = sorted(score_by_date.get(asof, []), key=lambda x: x[1], reverse=True)
        st_codes_today = st_flags_by_day.get(day_trade, set())
        score_map = {code: float(score) for code, score in rank_rows}
        if t_idx == 0:
            target_n = int(topk)
        else:
            target_n = int(round(target_n * prev_confidence))
            target_n = max(1, min(int(topk), target_n))

        for code, pos in positions.items():
            apply_adj_factor_before_open(pos, daily_cache.get(code, {}).get(day))

        initial_cash_day = cash
        initial_positions_amount = prev_final_holding_amount
        initial_total_asset = initial_cash_day + initial_positions_amount
        init_pos_details: list[dict[str, Any]] = []
        sell_records: list[dict[str, Any]] = []
        buy_records: list[dict[str, Any]] = []
        skipped_trades: list[dict[str, Any]] = []
        sold_today: set[str] = set()

        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            mark_open = mark_price(day_row, "open", pos.cost_per_share)
            init_pos_details.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": round(float(pos.qty), 6), "float_pnl": round(mark_open * pos.qty - pos.cost_total, 4)})

        for code, pos in list(positions.items()):
            day_row = daily_cache.get(code, {}).get(day)
            if day_row is None:
                continue
            open_p = float(day_row.get("open", math.nan))
            if (not np.isfinite(open_p)) or open_p <= 0:
                continue
            limit_row = limit_cache.get(code, {}).get(day)
            if limit_row is None:
                skipped_trades.append({"code": code, "action": "sell", "reason": "missing limit data"})
                if verbose:
                    print(f"[trade_simulator] skip sell {code} {day}: missing limit data")
                continue
            down_limit = float(limit_row.get("down_limit", math.nan))
            sell_price: float | None = None
            score_today = score_map.get(code, math.nan)
            if np.isfinite(down_limit) and abs(open_p - down_limit) < 1e-8:
                sell_price = None
            elif code in st_codes_today:
                sell_price = open_p
            elif np.isfinite(score_today) and score_today < float(sell_gate):
                sell_price = open_p
            elif open_p / pos.cost_per_share < 0.925:
                sell_price = open_p
            if sell_price is None:
                continue
            proceeds = sell_price * pos.qty
            sell_fee = commission(proceeds)
            cash += proceeds - sell_fee
            realized = (proceeds - sell_fee) - pos.cost_total
            sell_records.append({"code": code, "cost": round(pos.cost_per_share, 6), "price": round(sell_price, 6), "qty": round(float(pos.qty), 6), "realized_pnl": round(realized, 4)})
            sold_today.add(code)
            del positions[code]

        holding_value_after_sell = 0.0
        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            holding_value_after_sell += mark_price(day_row, "open", pos.cost_per_share) * pos.qty
        total_asset_for_buy = cash + holding_value_after_sell
        per_position_cap = total_asset_for_buy / max(1, int(topk))

        for code, score in rank_rows:
            if code in positions or code in sold_today:
                continue
            if score <= float(buy_gate):
                break
            if is_filtered_buy_code(code, st_codes_today):
                continue
            day_row = daily_cache.get(code, {}).get(day)
            if day_row is None:
                continue
            open_p = float(day_row.get("open", math.nan))
            if (not np.isfinite(open_p)) or open_p <= 0:
                continue
            limit_row = limit_cache.get(code, {}).get(day)
            if limit_row is None:
                skipped_trades.append({"code": code, "action": "buy", "reason": "missing limit data"})
                if verbose:
                    print(f"[trade_simulator] skip buy {code} {day}: missing limit data")
                continue
            up_limit = float(limit_row.get("up_limit", math.nan))
            if np.isfinite(up_limit) and abs(open_p - up_limit) < 1e-8:
                continue

            def ln_coe(coe: float) -> float:
                return coe if coe < 1 else 1 + math.log(coe)

            qty = round_lot_shares(min(per_position_cap * ln_coe(score), cash), open_p)
            if qty < 100:
                continue
            gross = open_p * qty
            buy_fee = commission(gross)
            total_needed = gross + buy_fee
            if total_needed > cash:
                qty = round_lot_shares(max(0.0, cash - COMMISSION_MIN), open_p)
                gross = open_p * qty
                buy_fee = commission(gross) if qty > 0 else 0.0
                total_needed = gross + buy_fee
            if qty < 100 or total_needed > cash:
                continue

            cash -= total_needed
            adj = float(day_row.get("adj_factor", math.nan))
            if (not np.isfinite(adj)) or adj <= 0:
                adj = 1.0
            pos = Position(code=code, qty=float(qty), cost_per_share=float((gross + buy_fee) / qty), last_adj_factor=float(adj))
            positions[code] = pos
            buy_records.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": round(float(pos.qty), 6)})
            if len(positions) >= target_n:
                break

        final_pos_details: list[dict[str, Any]] = []
        final_holding_amount = 0.0
        for code, pos in positions.items():
            day_row = daily_cache.get(code, {}).get(day)
            market_value = mark_price(day_row, "close", pos.cost_per_share) * pos.qty
            final_holding_amount += market_value
            final_pos_details.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": round(float(pos.qty), 6), "float_pnl": round(market_value - pos.cost_total, 4)})

        final_total_asset = cash + final_holding_amount
        prev_final_holding_amount = final_holding_amount
        prev_holding_count = len(final_pos_details)
        prev_confidence = 0.5 * pow(4, float(sum(1 for row in final_pos_details if float(row.get("float_pnl", 0.0)) > 0.0)) / float(prev_holding_count)) if prev_holding_count > 0 else 1

        payload = {
            "asof_date": asof,
            "date": day,
            "initial_holding_amount": round(initial_positions_amount, 4),
            "initial_cash": round(initial_cash_day, 4),
            "initial_total_asset": round(initial_total_asset, 4),
            "initial_positions": init_pos_details,
            "sell_records": sell_records,
            "buy_records": buy_records,
            "skipped_trades": skipped_trades,
            "final_positions": final_pos_details,
            "final_holding_amount": round(final_holding_amount, 4),
            "final_cash": round(cash, 4),
            "final_total_asset": round(final_total_asset, 4),
        }
        outputs.append(payload)
        if out_dir is not None:
            out_path = out_dir / f"{day_trade}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved {out_path}")
        if verbose:
            print_day_summary(payload)

    return outputs
