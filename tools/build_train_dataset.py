from __future__ import annotations

import argparse
from pathlib import Path

from src.feat import build_multiscale_tensor, labels_risk_adj
from src.feat import build_training_dataset as training_dataset


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
    parser.add_argument("--shard-tmp-dir", default="", help="temporary directory for per-code shard files; empty uses system default")
    parser.add_argument("--benchmark", action="store_true", help="enable function-level benchmark timings and run only first code")
    args = parser.parse_args()

    raw_codes = _split_csv(args.codes)
    raw_asof_dates = _split_csv(args.asof_dates)
    codes = training_dataset.resolve_codes(args.data_dir, raw_codes or None)
    asof_dates = training_dataset.resolve_asof_dates(args.data_dir, raw_asof_dates or None)
    selected_codes = codes
    num_workers = max(1, int(args.num_workers))
    if args.benchmark:
        build_multiscale_tensor.enable_benchmark()
        labels_risk_adj.enable_benchmark()
        # Rebind function references captured by build_training_dataset at import time
        # so benchmark-wrapped functions are actually invoked.
        training_dataset.build_multiscale_tensors = build_multiscale_tensor.build_multiscale_tensors
        training_dataset.build_label_from_data_dir = labels_risk_adj.build_label_from_data_dir

        selected_codes = codes[:1]
        num_workers = 1
        print(f"benchmark mode enabled: running only first code: {selected_codes[0] if selected_codes else '<none>'}")

    bundle = training_dataset.build_train_dataset(
        data_dir=args.data_dir,
        codes=selected_codes,
        asof_dates=asof_dates,
        include_invalid=bool(args.include_invalid),
        num_workers=num_workers,
        show_progress=bool(args.show_progress),
        shard_tmp_dir=(args.shard_tmp_dir or None),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.save_train_dataset(bundle, out)

    print(f"saved: {out}")
    print(f"codes: {len(selected_codes)}")
    print(f"asof_dates: {len(asof_dates)}")
    print(f"workers: {num_workers}")
    print(f"samples: {len(bundle.y)}")
    print(f"X_micro: {bundle.X_micro.shape}")
    print(f"X_mezzo: {bundle.X_mezzo.shape}")
    print(f"X_macro: {bundle.X_macro.shape}")
    print(f"y: {bundle.y.shape}, loss_mask_true: {int(bundle.loss_mask.sum())}")
    print(f"shard_tmp_dir: {args.shard_tmp_dir or '<system_temp>'}")

    if args.benchmark:
        print("benchmark summary (build_multiscale_tensor):")
        tensor_rows = build_multiscale_tensor.get_benchmark_report()
        if not tensor_rows:
            print("  <no benchmark data collected>")
        for row in tensor_rows:
            print(
                f"  {row['function']}: count={int(row['count'])}, total={row['total_ms']:.3f}ms, "
                f"avg={row['avg_ms']:.3f}ms, min={row['min_ms']:.3f}ms, max={row['max_ms']:.3f}ms"
            )
        print("benchmark summary (labels_risk_adj):")
        label_rows = labels_risk_adj.get_benchmark_report()
        if not label_rows:
            print("  <no benchmark data collected>")
        for row in label_rows:
            print(
                f"  {row['function']}: count={int(row['count'])}, total={row['total_ms']:.3f}ms, "
                f"avg={row['avg_ms']:.3f}ms, min={row['min_ms']:.3f}ms, max={row['max_ms']:.3f}ms"
            )
        return


if __name__ == "__main__":
    main()
