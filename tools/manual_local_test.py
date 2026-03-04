#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date

from src.feat.build_multiscale_tensor import LocalProvider, build_multiscale_tensor, build_trading_calendar


def _must(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[FAIL] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual non-pytest local test for multiscale tensor builder")
    parser.add_argument("--code", required=True)
    parser.add_argument("--asof", required=True, help="YYYY-MM-DD")
    parser.add_argument("--calendar-dir", required=True, help="Directory where daily filenames contain YYYYMMDD")
    parser.add_argument("--root-1d", required=True)
    parser.add_argument("--root-30m", required=True)
    parser.add_argument("--root-5m", required=True)
    args = parser.parse_args()

    asof_date = date.fromisoformat(args.asof)
    calendar, _ = build_trading_calendar(args.calendar_dir)
    _must(len(calendar) > 0, "trading calendar is empty")

    provider = LocalProvider(
        interval_roots={
            "1d": args.root_1d,
            "30m": args.root_30m,
            "5m": args.root_5m,
        }
    )

    out = build_multiscale_tensor(args.code, asof_date, provider, calendar)

    # contract shape checks
    _must(len(out["X_micro"]) == 48 and len(out["X_micro"][0]) == 7, "X_micro shape must be [48,7]")
    _must(len(out["X_mezzo"]) == 40 and len(out["X_mezzo"][0]) == 7, "X_mezzo shape must be [40,7]")
    _must(len(out["X_macro"]) == 30 and len(out["X_macro"][0]) == 7, "X_macro shape must be [30,7]")
    _must(len(out["mask_micro"]) == 48, "mask_micro shape must be [48]")
    _must(len(out["mask_mezzo"]) == 40, "mask_mezzo shape must be [40]")
    _must(len(out["mask_macro"]) == 30, "mask_macro shape must be [30]")

    print("[OK] shape checks passed")
    print("dp_ok:", out["dp_ok"])
    print("mask sums:", sum(out["mask_micro"]), sum(out["mask_mezzo"]), sum(out["mask_macro"]))


if __name__ == "__main__":
    main()
