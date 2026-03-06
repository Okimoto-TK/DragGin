from __future__ import annotations

import argparse
from pathlib import Path
import time

from src.feat.build_training_dataset import build_train_dataset, save_train_dataset


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]




def _print_timing_report(timings: dict[str, dict[str, float]], total_elapsed: float) -> None:
    if not timings:
        print("timing: no stats collected")
        return
    print("timing report:")
    print(f"{'function':34} {'count':>8} {'total(s)':>12} {'avg(ms)':>12} {'share':>10}")
    for name, stat in sorted(timings.items(), key=lambda item: item[1].get('total', 0.0), reverse=True):
        total = float(stat.get('total', 0.0))
        count = int(stat.get('count', 0.0))
        avg_ms = (total * 1000.0 / count) if count else 0.0
        share = (total / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
        print(f"{name:34} {count:8d} {total:12.4f} {avg_ms:12.3f} {share:9.2f}%")
    print(f"total elapsed: {total_elapsed:.4f}s")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--codes", default="", help="comma-separated stock codes; empty means all codes")
    parser.add_argument("--asof-dates", default="", help="comma-separated asof dates; empty means all calendar dates")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--include-invalid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker processes")
    parser.add_argument("--show-progress", type=int, choices=[0, 1], default=1, help="show progress bar")
    parser.add_argument("--shard-tmp-dir", default="", help="temporary directory for per-code shard files; empty uses system default")
    parser.add_argument("--benchmark", action="store_true", help="enable timing, process only one code, and print timing analysis")
    args = parser.parse_args()

    raw_codes = _split_csv(args.codes)
    raw_asof_dates = _split_csv(args.asof_dates)
    codes = raw_codes or None
    asof_dates = raw_asof_dates or None
    timings: dict[str, dict[str, float]] | None = {} if args.benchmark else None
    t0 = time.perf_counter()
    bundle = build_train_dataset(
        data_dir=args.data_dir,
        codes=codes,
        asof_dates=asof_dates,
        include_invalid=bool(args.include_invalid),
        num_workers=max(1, int(args.num_workers)),
        show_progress=bool(args.show_progress),
        shard_tmp_dir=(args.shard_tmp_dir or None),
        benchmark=bool(args.benchmark),
        timings=timings,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    save_train_dataset(bundle, out)
    if timings is not None:
        timings.setdefault('save_train_dataset', {'total': 0.0, 'count': 0.0})
        timings['save_train_dataset']['total'] += time.perf_counter() - t_save
        timings['save_train_dataset']['count'] += 1
    total_elapsed = time.perf_counter() - t0

    print(f"saved: {out}")
    print(f"codes: {len(set(bundle.codes.tolist()))}")
    print(f"asof_dates: {len(set(bundle.asof_dates.tolist()))}")
    print(f"workers: {1 if args.benchmark else max(1, int(args.num_workers))}")
    print(f"samples: {len(bundle.y)}")
    print(f"X_micro: {bundle.X_micro.shape}")
    print(f"X_mezzo: {bundle.X_mezzo.shape}")
    print(f"X_macro: {bundle.X_macro.shape}")
    print(f"y: {bundle.y.shape}, loss_mask_true: {int(bundle.loss_mask.sum())}")
    print(f"shard_tmp_dir: {args.shard_tmp_dir or '<system_temp>'}")
    if args.benchmark and timings is not None:
        _print_timing_report(timings, total_elapsed)


if __name__ == "__main__":
    main()
