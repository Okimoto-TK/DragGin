from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trader tool entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_daily = subparsers.add_parser("update-daily", help="Update raw daily market data cache under ./data")
    update_daily.add_argument("--data-dir", default="data")
    update_daily.add_argument("--lookback-trading-days", type=int, default=120)
    update_daily.add_argument("--sleep", type=float, default=0.0)
    update_daily.add_argument("--verbose", action="store_true")
    update_daily.add_argument("--no-refresh-latest", action="store_true")

    args = parser.parse_args()

    if args.command == "update-daily":
        from src.data import DailyUpdateConfig, update_daily_market_data

        meta = update_daily_market_data(
            DailyUpdateConfig(
                data_dir=Path(args.data_dir),
                lookback_trading_days=max(1, int(args.lookback_trading_days)),
                request_sleep_seconds=max(0.0, float(args.sleep)),
                refresh_latest=not bool(args.no_refresh_latest),
                verbose=bool(args.verbose),
            )
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
