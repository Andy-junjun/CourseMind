from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import DEFAULT_MODEL_NAME


DEFAULT_PAIRS_PATH = Path("data/training/embedding_pairs.jsonl")
DEFAULT_OUTPUT_DIR = Path("models/embedding/finetuned")


def load_pairs(path: Path, max_samples: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    pairs: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            validate_pair(item)
            pairs.append(item)
            if max_samples and len(pairs) >= max_samples:
                break
    if not pairs:
        raise ValueError(f"No training pairs found in {path}")
    return pairs


def validate_pair(item: dict) -> None:
    required = {"query", "positive_text", "negative_text"}
    missing = [key for key in required if not item.get(key)]
    if missing:
        raise ValueError(f"Invalid training pair, missing: {', '.join(missing)}")


def dry_run(pairs: list[dict], output_dir: Path, base_model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "dry_run",
        "base_model": base_model,
        "pair_count": len(pairs),
        "note": (
            "This directory is a dry-run manifest, not a loadable sentence-transformers "
            "model. Run without --dry-run after installing torch and sentence-transformers."
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Dry-run embedding model output\n\n"
        "本目录只验证训练数据和脚本流程，不是可加载的 sentence-transformers 模型。\n",
        encoding="utf-8",
    )


def train(
    pairs: list[dict],
    output_dir: Path,
    *,
    base_model: str,
    epochs: int,
    batch_size: int,
    warmup_steps: int,
) -> None:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Training requires sentence-transformers and torch. "
            "Install them first, or run with --dry-run."
        ) from exc

    model = SentenceTransformer(base_model)
    examples = [
        InputExample(
            texts=[item["query"], item["positive_text"], item["negative_text"]]
        )
        for item in pairs
    ]
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    elapsed = time.time() - started_at
    manifest = {
        "mode": "trained",
        "base_model": base_model,
        "pair_count": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "elapsed_seconds": round(elapsed, 3),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence-transformers embedding model with triplet pairs."
    )
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.pairs_path, max_samples=args.max_samples)
    if args.dry_run:
        dry_run(pairs, args.output_dir, args.base_model)
        print(f"Dry-run checked {len(pairs)} pairs")
        print(f"Output manifest: {args.output_dir / 'training_manifest.json'}")
        return

    train(
        pairs,
        args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
    )
    print(f"Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
