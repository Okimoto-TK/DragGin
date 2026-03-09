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
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--buffer", action="store_true", help="preload all shards into memory")
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker processes for shard loading")
    parser.add_argument("--pin-memory", action="store_true", help="pin CPU batch memory before H2D copy")
    parser.add_argument("--prefetch-cuda", action="store_true", help="prefetch next batch to CUDA stream")
    parser.add_argument("--enable-compile", action="store_true", help="enable torch.compile on CUDA")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", help="torch.compile mode")
    parser.add_argument("--log-every", type=int, default=10, help="log every N optimizer steps")
    parser.add_argument("--curve-save-every", type=int, default=100, help="save curve JSON every N optimizer steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume training from checkpoint path")
    parser.add_argument("--save-every", type=int, default=1, help="save an epoch checkpoint every N epochs")
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
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        y_key=args.y_key,
        val_ratio=args.val_ratio,
        buffer=args.buffer,
        num_workers=max(1, int(args.num_workers)),
        pin_memory=bool(args.pin_memory),
        prefetch_cuda=bool(args.prefetch_cuda),
        enable_compile=bool(args.enable_compile),
        compile_mode=str(args.compile_mode),
        log_every=max(1, int(args.log_every)),
        curve_save_every=max(1, int(args.curve_save_every)),
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
