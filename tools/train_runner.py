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
    parser.add_argument("--val-shards", nargs="+", required=True, help="val shard paths (supports glob)")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    args = parser.parse_args()

    train_shards = _expand_paths(args.train_shards)
    val_shards = _expand_paths(args.val_shards)
    if len(train_shards) == 0:
        raise ValueError("No train shards found.")
    if len(val_shards) == 0:
        raise ValueError("No val shards found.")

    cfg = TrainConfig(
        train_shards=train_shards,
        val_shards=val_shards,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        y_key=args.y_key,
    )
    output = run_training(cfg)

    print(f"status: {output['feedback']['meta']['status']}")
    print(f"epochs_completed: {output['feedback']['meta']['epochs_completed']}")
    print(f"best_val_loss: {output['feedback']['metrics']['best_val_loss']:.8f}")
    print(f"output_dir: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
