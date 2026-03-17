from __future__ import annotations

import argparse
import json
import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.train.runner import MultiScaleRegressor


COMMISSION_RATE = 1e-4
COMMISSION_MIN = 5.0
SLIPPAGE = 0.005


@dataclass
class Position:
    code: str
    qty: int
    buy_price: float
    buy_fee: float
    est_sell_fee: float

    @property
    def cost_per_share(self) -> float:
        return (self.buy_price * self.qty + self.buy_fee + self.est_sell_fee) / self.qty

    @property
    def cost_total(self) -> float:
        return self.cost_per_share * self.qty


def _commission(amount: float) -> float:
    return max(COMMISSION_MIN, float(amount) * COMMISSION_RATE)


def _to_datestr(v: Any) -> str:
    ts_ = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts_):
        raise ValueError(f"invalid date value: {v}")
    return ts_.strftime("%Y-%m-%d")


def _to_trade_datestr(v: Any) -> str:
    ts_ = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts_):
        raise ValueError(f"invalid date value: {v}")
    return ts_.strftime("%Y%m%d")


def _read_calendar_dates(data_dir: Path) -> list[str]:
    cal_path = data_dir / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar file not found: {cal_path}")
    df = pd.read_parquet(cal_path)
    for col in ["trade_date", "cal_date", "date"]:
        if col in df.columns:
            values = sorted({_to_datestr(x) for x in df[col].dropna().tolist()})
            if not values:
                break
            return values
    raise ValueError(f"cannot find date column in {cal_path}")


def _resolve_codes(data_dir: Path) -> list[str]:
    codes: list[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "daily.parquet").exists() and (p / "5min.parquet").exists() and (p / "moneyflow.parquet").exists():
            codes.append(p.name)
    return codes


def _build_flow_features(data_dir: Path, code: str, asof: str) -> tuple[np.ndarray, np.ndarray, bool]:
    d1_path = data_dir / code / "daily.parquet"
    mf_path = data_dir / code / "moneyflow.parquet"
    if not d1_path.exists() or not mf_path.exists():
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False

    d1 = pd.read_parquet(d1_path, columns=["trade_date", "volume"])
    mf = pd.read_parquet(mf_path, columns=["trade_date", "net_mf_vol", "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol"])
    d1["trade_date"] = pd.to_datetime(d1["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d1["volume"] = pd.to_numeric(d1["volume"], errors="coerce")
    d1 = d1.dropna().sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    mf = mf.dropna().sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    dates = d1["trade_date"].tolist()
    if asof not in set(dates):
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
    idx = dates.index(asof)
    if idx + 1 < 30:
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False

    tail_dates = dates[idx + 1 - 30 : idx + 1]
    w = d1[d1["trade_date"].isin(set(tail_dates))].merge(mf, on="trade_date", how="inner").sort_values("trade_date")
    if len(w) != 30:
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False

    volume_lots = w["volume"].to_numpy(dtype=np.float64) / 100.0
    if (~np.isfinite(volume_lots)).any() or (volume_lots <= 0).any():
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False

    net = w["net_mf_vol"].to_numpy(dtype=np.float64)
    lg = (w["buy_lg_vol"] - w["sell_lg_vol"]).to_numpy(dtype=np.float64)
    elg = (w["buy_elg_vol"] - w["sell_elg_vol"]).to_numpy(dtype=np.float64)
    lg_elg = (w["buy_lg_vol"] + w["buy_elg_vol"] - w["sell_lg_vol"] - w["sell_elg_vol"]).to_numpy(dtype=np.float64)
    flow_x = np.stack([net / volume_lots, lg / volume_lots, elg / volume_lots, lg_elg / volume_lots], axis=1).astype(np.float32)
    if flow_x.shape != (30, 4) or (not np.isfinite(flow_x).all()):
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
    return flow_x, np.ones((30,), dtype=np.uint8), True


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
    df = pd.read_parquet(p, columns=["trade_date", "open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates("trade_date", keep="last")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["trade_date"])] = {"open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}
    return out


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
                "  - {code:<10} 数量:{qty:>8} 成本:{cost:>10} 浮盈亏(开盘):{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=int(row.get("qty", 0)),
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                    pnl=_fmt_money(float(row.get("float_pnl", 0.0))),
                )
            )

    sells = payload.get("sell_records", []) or []
    if sells:
        print("\n[卖出记录]")
        for row in sells:
            print(
                "  - {code:<10} 数量:{qty:>8} 成交:{price:>10} 成本:{cost:>10} 实现盈亏:{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=int(row.get("qty", 0)),
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
                "  - {code:<10} 数量:{qty:>8} 成本:{cost:>10}".format(
                    code=str(row.get("code", "")),
                    qty=int(row.get("qty", 0)),
                    cost=f"{float(row.get('cost', 0.0)):.4f}",
                )
            )

    final_positions = payload.get("final_positions", []) or []
    if final_positions:
        print("\n[期末持仓]")
        for row in final_positions:
            print(
                "  - {code:<10} 数量:{qty:>8} 成本:{cost:>10} 浮盈亏(收盘):{pnl:>12}".format(
                    code=str(row.get("code", "")),
                    qty=int(row.get("qty", 0)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Model backtest runner")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-embargo-days", type=int, default=30)
    parser.add_argument("--val-shards", nargs="+", default=None)
    parser.add_argument("--ts-token", default="")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-seq-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-dynamic-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-free-branch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleRegressor(
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_seq_context=bool(args.use_seq_context),
        enable_dynamic_threshold=bool(args.enable_dynamic_threshold),
        enable_free_branch=bool(args.enable_free_branch),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()

    token = args.ts_token or str(os.environ.get("TUSHARE_TOKEN", ""))
    pro = ts.pro_api(token) if token else None
    limit_cache: dict[str, dict[str, tuple[float, float]]] = {}

    daily_cache = {c: _load_daily_map(data_dir, c) for c in codes}

    cash = float(args.initial_cash)
    positions: dict[str, Position] = {}
    prev_final_holding_amount = 0.0

    for asof in asof_dates:
        t_idx = val_dates.index(asof)
        day = val_dates[t_idx + 1]
        day_trade = day.replace("-", "")

        rank_rows: list[tuple[str, float]] = []
        for code in codes:
            dp = build_multiscale_tensors(str(data_dir), code, asof)
            flow_x, flow_mask, flow_ok = _build_flow_features(data_dir, code, asof)
            if (not dp.dp_ok) or (not flow_ok):
                continue
            batch = {
                "x_micro": torch.from_numpy(dp.X_micro.astype(np.float32)).unsqueeze(0).to(device),
                "x_mezzo": torch.from_numpy(dp.X_mezzo.astype(np.float32)).unsqueeze(0).to(device),
                "x_macro": torch.from_numpy(dp.X_macro.astype(np.float32)).unsqueeze(0).to(device),
                "mask_micro": torch.from_numpy(dp.mask_micro.astype(np.uint8)).to(torch.bool).unsqueeze(0).to(device),
                "mask_mezzo": torch.from_numpy(dp.mask_mezzo.astype(np.uint8)).to(torch.bool).unsqueeze(0).to(device),
                "mask_macro": torch.from_numpy(dp.mask_macro.astype(np.uint8)).to(torch.bool).unsqueeze(0).to(device),
                "flow_x": torch.from_numpy(flow_x).unsqueeze(0).to(device),
                "flow_mask": torch.from_numpy(flow_mask.astype(np.uint8)).to(torch.bool).unsqueeze(0).to(device),
            }
            with torch.no_grad():
                y_hat, _ = model(batch)
            rank_rows.append((code, float(y_hat.squeeze().item())))

        rank_rows.sort(key=lambda x: x[1], reverse=True)
        rank_map = {code: i + 1 for i, (code, _) in enumerate(rank_rows)}

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

        initial_cash = cash
        initial_positions_amount = prev_final_holding_amount
        initial_total_asset = initial_cash + initial_positions_amount

        init_pos_details: list[dict[str, Any]] = []
        sell_records: list[dict[str, Any]] = []
        buy_records: list[dict[str, Any]] = []

        for code, pos in list(positions.items()):
            day_row = daily_cache.get(code, {}).get(day)
            if day_row is None:
                continue
            open_p = day_row["open"]
            high_p = day_row["high"]
            low_p = day_row["low"]

            mark_proceeds = open_p * pos.qty - _commission(open_p * pos.qty)
            float_pnl = mark_proceeds - (pos.buy_price * pos.qty + pos.buy_fee)
            init_pos_details.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": pos.qty, "float_pnl": round(float_pnl, 4)})

            updn = limit_cache[day_trade].get(code, (math.nan, math.nan))
            down_limit = float(updn[1])

            sell_price: float | None = None
            if np.isfinite(down_limit) and abs(open_p - down_limit) < 1e-8:
                if (abs(high_p - down_limit) < 1e-8):
                    sell_price = None
                elif down_limit / pos.cost_per_share < 0.925:
                    sell_price = down_limit
            else:
                if low_p / pos.cost_per_share < 0.925:
                    sell_price = pos.cost_per_share * 0.92
                elif rank_map.get(code, 10**9) > 2 * int(args.topk):
                    sell_price = open_p

            if sell_price is None:
                continue

            proceeds = sell_price * pos.qty
            sell_fee = _commission(proceeds)
            cash += proceeds - sell_fee
            realized = (proceeds - sell_fee) - (pos.buy_price * pos.qty + pos.buy_fee)
            sell_records.append(
                {
                    "code": code,
                    "cost": round(pos.cost_per_share, 6),
                    "price": round(sell_price, 6),
                    "qty": pos.qty,
                    "realized_pnl": round(realized, 4),
                }
            )
            del positions[code]

        for code, score in rank_rows:
            if code in positions:
                continue
            if score < 0.5:
                break
            day_row = daily_cache.get(code, {}).get(day)
            if day_row is None:
                continue
            open_p = day_row["open"]
            updn = limit_cache[day_trade].get(code, (math.nan, math.nan))
            up_limit = float(updn[0])
            if np.isfinite(up_limit) and abs(open_p - up_limit) < 1e-8:
                continue

            holding_value = 0.0
            for c, p in positions.items():
                close_p = daily_cache.get(c, {}).get(day, {}).get("close", p.buy_price)
                holding_value += close_p * p.qty
            total_asset_now = cash + holding_value
            cap = total_asset_now / max(1, int(args.topk))
            qty = _round_lot_shares(min(cap, cash), open_p)
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
            est_sell_fee = _commission(gross)
            pos = Position(code=code, qty=qty, buy_price=open_p, buy_fee=buy_fee, est_sell_fee=est_sell_fee)
            positions[code] = pos
            buy_records.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": qty})
            if len(positions) >= int(args.topk):
                break

        final_pos_details: list[dict[str, Any]] = []
        final_holding_amount = 0.0
        for code, pos in positions.items():
            close_p = daily_cache.get(code, {}).get(day, {}).get("close", pos.buy_price)
            final_holding_amount += close_p * pos.qty
            mark_proceeds = close_p * pos.qty - _commission(close_p * pos.qty)
            float_pnl = mark_proceeds - (pos.buy_price * pos.qty + pos.buy_fee)
            final_pos_details.append({"code": code, "cost": round(pos.cost_per_share, 6), "qty": pos.qty, "float_pnl": round(float_pnl, 4)})

        final_total_asset = cash + final_holding_amount
        prev_final_holding_amount = final_holding_amount

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
