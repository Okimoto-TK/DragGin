from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.runner import TrainConfig, run_training


def _expand_paths(raw_paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_paths:
        if any(ch in raw for ch in "*?[]"):
            expanded.extend(str(p) for p in sorted(Path().glob(raw)))
        else:
            expanded.append(raw)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 7 training CLI")
    parser.add_argument("--train-shards", nargs="+", required=True, help="train shard paths (supports glob)")
    parser.add_argument("--val-shards", nargs="+", default=None, help="val shard paths (supports glob)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="val split ratio if --val-shards is omitted")
    parser.add_argument("--split-mode", type=str, default="date", choices=["date", "code"], help="split mode when --val-shards is omitted")
    parser.add_argument("--val-embargo-days", type=int, default=30, help="embargo trading days between train and val in date split")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-std-target", type=float, default=0.10)
    parser.add_argument("--gate-std-reg", type=float, default=1e-2)
    parser.add_argument("--gate-mean-target", type=float, default=0.50)
    parser.add_argument("--gate-mean-reg", type=float, default=1e-2)
    parser.add_argument("--gate-entropy-reg", type=float, default=5e-3)
    parser.add_argument("--gate-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--flow-gate-force-zero-all-steps",
        action="store_true",
        help="force flow gate to 0.0 for all steps while still running flow encoder",
    )
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-seq-context", action=argparse.BooleanOptionalAction, default=True, help="enable sequence context in regression head")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--buffer", action="store_true", help="preload all shards into memory")
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker processes for shard loading")
    parser.add_argument("--pin-memory", action="store_true", help="pin CPU batch memory before H2D copy")
    parser.add_argument("--prefetch-cuda", action="store_true", help="prefetch next batch to CUDA stream")
    parser.add_argument("--log-every", type=int, default=10, help="log every N optimizer steps")
    parser.add_argument("--curve-save-every", type=int, default=100, help="save curve JSON every N optimizer steps")
    parser.add_argument("--hist-every", type=int, default=100, help="write gate histograms every N optimizer steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume training from checkpoint path")
    parser.add_argument("--save-every", type=int, default=1, help="save an epoch checkpoint every N epochs")
    parser.add_argument("--gate-lr", type=float, default=None)
    parser.add_argument("--gate-clip-grad-norm", type=float, default=0.2)
    parser.add_argument("--scheduler-name", type=str, default="plateau")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--finite-skip-max-warn", type=int, default=20)
    args = parser.parse_args()

    train_shards = _expand_paths(args.train_shards)
    val_shards = _expand_paths(args.val_shards or [])
    if len(train_shards) == 0:
        raise ValueError("No train shards found.")

    cfg = TrainConfig(
        train_shards=train_shards,
        val_shards=val_shards,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gate_temperature=args.gate_temperature,
        gate_std_target=args.gate_std_target,
        gate_std_reg=args.gate_std_reg,
        gate_mean_target=args.gate_mean_target,
        gate_mean_reg=args.gate_mean_reg,
        gate_entropy_reg=args.gate_entropy_reg,
        gate_warmup_steps=max(0, int(args.gate_warmup_steps)),
        flow_gate_force_zero_all_steps=bool(args.flow_gate_force_zero_all_steps),
        gate_lr=args.gate_lr,
        gate_clip_grad_norm=args.gate_clip_grad_norm,
        scheduler_name=str(args.scheduler_name),
        scheduler_factor=float(args.scheduler_factor),
        scheduler_patience=max(1, int(args.scheduler_patience)),
        scheduler_min_lr=float(args.scheduler_min_lr),
        finite_skip_max_warn=max(0, int(args.finite_skip_max_warn)),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_seq_context=bool(args.use_seq_context),
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        y_key=args.y_key,
        val_ratio=args.val_ratio,
        split_mode=str(args.split_mode),
        val_embargo_days=max(0, int(args.val_embargo_days)),
        buffer=args.buffer,
        num_workers=max(1, int(args.num_workers)),
        pin_memory=bool(args.pin_memory),
        prefetch_cuda=bool(args.prefetch_cuda),
        log_every=max(1, int(args.log_every)),
        curve_save_every=max(1, int(args.curve_save_every)),
        hist_every=max(1, int(args.hist_every)),
        checkpoint=args.checkpoint,
        save_every=max(1, int(args.save_every)),
    )
    output = run_training(cfg)

    print(f"status: {output['feedback']['meta']['status']}")
    print(f"epochs_completed: {output['feedback']['meta']['epochs_completed']}")
    print(f"best_val_loss: {output['feedback']['metrics']['best_val_loss']:.8f}")
    print(f"output_dir: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
