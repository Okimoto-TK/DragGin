from __future__ import annotations

import argparse
from pathlib import Path

from src.feat.build_training_dataset import build_train_dataset, save_train_dataset


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--codes", required=True, help="comma-separated stock codes")
    parser.add_argument("--asof-dates", required=True, help="comma-separated asof dates, e.g. 2024-01-02")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--include-invalid", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    codes = _split_csv(args.codes)
    asof_dates = _split_csv(args.asof_dates)
    bundle = build_train_dataset(
        data_dir=args.data_dir,
        codes=codes,
        asof_dates=asof_dates,
        include_invalid=bool(args.include_invalid),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_train_dataset(bundle, out)

    print(f"saved: {out}")
    print(f"samples: {len(bundle.y)}")
    print(f"X_micro: {bundle.X_micro.shape}")
    print(f"X_mezzo: {bundle.X_mezzo.shape}")
    print(f"X_macro: {bundle.X_macro.shape}")
    print(f"y: {bundle.y.shape}, loss_mask_true: {int(bundle.loss_mask.sum())}")


if __name__ == "__main__":
    main()
