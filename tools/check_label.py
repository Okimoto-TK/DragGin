from __future__ import annotations

import argparse

from src.feat.labels_risk_adj import build_label_from_data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--code", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--dp-ok", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    result = build_label_from_data_dir(
        data_dir=args.data_dir,
        code=args.code,
        asof_date=args.asof,
        dp_ok=bool(args.dp_ok),
    )

    print(f"code: {result.code}")
    print(f"asof_date: {result.asof_date}")
    print(f"label_ok: {result.label_ok}")
    print(f"loss_mask: {result.loss_mask}")
    print(f"y: {float(result.y):.8f}")
    print(f"entry_date: {result.entry_date}")
    print(f"exit_date: {result.exit_date}")
    print(f"entry_open: {result.entry_open}")
    print(f"exit_close: {result.exit_close}")
    print(f"vol30: {result.vol30}")
    print(f"ret_log: {result.ret_log}")
    print(f"fail_reason: {result.fail_reason}")


if __name__ == "__main__":
    main()
