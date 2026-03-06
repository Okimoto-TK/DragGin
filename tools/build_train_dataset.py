from __future__ import annotations

import argparse
from pathlib import Path

from src.feat.build_training_dataset import build_train_dataset, resolve_asof_dates, resolve_codes, save_train_dataset


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--codes", default="", help="comma-separated stock codes; empty means all codes")
    parser.add_argument("--asof-dates", default="", help="comma-separated asof dates; empty means all calendar dates")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--include-invalid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker processes")
    parser.add_argument("--show-progress", type=int, choices=[0, 1], default=1, help="show progress bar")
    args = parser.parse_args()

    raw_codes = _split_csv(args.codes)
    raw_asof_dates = _split_csv(args.asof_dates)
    codes = resolve_codes(args.data_dir, raw_codes or None)
    asof_dates = resolve_asof_dates(args.data_dir, raw_asof_dates or None)
    bundle = build_train_dataset(
        data_dir=args.data_dir,
        codes=codes,
        asof_dates=asof_dates,
        include_invalid=bool(args.include_invalid),
        num_workers=max(1, int(args.num_workers)),
        show_progress=bool(args.show_progress),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_train_dataset(bundle, out)

    print(f"saved: {out}")
    print(f"codes: {len(codes)}")
    print(f"asof_dates: {len(asof_dates)}")
    print(f"workers: {max(1, int(args.num_workers))}")
    print(f"samples: {len(bundle.y)}")
    print(f"X_micro: {bundle.X_micro.shape}")
    print(f"X_mezzo: {bundle.X_mezzo.shape}")
    print(f"X_macro: {bundle.X_macro.shape}")
    print(f"y: {bundle.y.shape}, loss_mask_true: {int(bundle.loss_mask.sum())}")


if __name__ == "__main__":
    main()
