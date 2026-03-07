from __future__ import annotations

import argparse
from pathlib import Path

from src.feat.build_training_dataset import build_train_dataset_shards, resolve_asof_dates, resolve_codes


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--codes", default="", help="comma-separated stock codes; empty means all codes")
    parser.add_argument("--asof-dates", default="", help="comma-separated asof dates; empty means all calendar dates")
    parser.add_argument("--out", required=True, help="output folder path for per-code shard npy files")
    parser.add_argument("--include-invalid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker processes")
    parser.add_argument("--show-progress", type=int, choices=[0, 1], default=1, help="show progress bar")
    args = parser.parse_args()

    raw_codes = _split_csv(args.codes)
    raw_asof_dates = _split_csv(args.asof_dates)
    codes = resolve_codes(args.data_dir, raw_codes or None)
    asof_dates = resolve_asof_dates(args.data_dir, raw_asof_dates or None)
    out = Path(args.out)
    shard_infos = build_train_dataset_shards(
        data_dir=args.data_dir,
        out_dir=out,
        codes=codes,
        asof_dates=asof_dates,
        include_invalid=bool(args.include_invalid),
        num_workers=max(1, int(args.num_workers)),
        show_progress=bool(args.show_progress),
    )

    total_rows = sum(int(x.get("rows", 0)) for x in shard_infos)
    shard_paths = sorted(str(Path(x["path"]).resolve()) for x in shard_infos)

    print(f"saved: {out}")
    print(f"codes: {len(codes)}")
    print(f"asof_dates: {len(asof_dates)}")
    print(f"workers: {max(1, int(args.num_workers))}")
    print(f"samples: {total_rows}")
    print(f"shards: {len(shard_infos)}")
    for p in shard_paths:
        print(f"shard: {p}")


if __name__ == "__main__":
    main()
