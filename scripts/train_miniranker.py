from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_training_pairs import (  # noqa: E402
    DEFAULT_EVAL_DIR,
    DEFAULT_RAW_DIR,
    RetrievalQuery,
    file_matches,
    find_query_files,
    load_retrieval_queries,
    normalize,
    split_keywords,
)
from src.bm25_store import keyword_score  # noqa: E402
from src.graph_store import expand_with_graph  # noqa: E402
from src.miniranker import DEFAULT_MODEL_PATH, build_feature_matrix, build_torch_model  # noqa: E402
from src.retriever import retrieve  # noqa: E402
from src.schemas import Chunk  # noqa: E402
from src.vector_store import (  # noqa: E402
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_PATH,
    VectorIndex,
    build_index,
    load_chunks,
    load_index,
)


DEFAULT_MANIFEST_PATH = Path("data/training/miniranker_manifest.json")


@dataclass(frozen=True)
class TrainingExample:
    query: str
    chunk_id: str
    file_name: str
    features: list[float]
    label: float


def build_examples(
    queries: list[RetrievalQuery],
    chunks: list[Chunk],
    index: VectorIndex,
    *,
    retrieve_top_k: int,
    graph_hops: int,
    max_examples: int,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for query in queries:
        candidates = retrieve(query.query, chunks, top_k=retrieve_top_k, index=index)
        expanded = expand_with_graph(candidates, chunks, hops=graph_hops)
        if not expanded:
            continue

        features = build_feature_matrix(expanded)
        for item, row in zip(expanded, features):
            label = relevance_label(query, item.chunk)
            examples.append(
                TrainingExample(
                    query=query.query,
                    chunk_id=item.chunk.chunk_id,
                    file_name=item.chunk.file_name,
                    features=row,
                    label=label,
                )
            )
            if len(examples) >= max_examples:
                return examples
    return examples


def relevance_label(query: RetrievalQuery, chunk: Chunk) -> float:
    score = 0.0
    if query.expected_file and file_matches(query.expected_file, chunk.file_name):
        score = max(score, 1.0)

    topic = normalize(query.expected_topic)
    if topic:
        concept_text = normalize(" ".join(chunk.concepts))
        chunk_text = normalize(chunk.text)
        if topic in concept_text:
            score = max(score, 0.85)
        elif topic in chunk_text:
            score = max(score, 0.7)

    keyword_hits = 0
    normalized_text = normalize(chunk.text)
    for keyword in split_keywords(query.answer_keywords):
        if normalize(keyword) in normalized_text:
            keyword_hits += 1
    if keyword_hits:
        score = max(score, min(0.9, 0.45 + 0.15 * keyword_hits))

    if score == 0.0:
        score = min(0.35, keyword_score(query.query, chunk) * 0.6)
    return float(score)


def train_model(
    examples: list[TrainingExample],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[object, dict[str, float]]:
    if not examples:
        raise ValueError("No MiniRanker training examples were built.")

    import torch
    from torch import nn

    random.seed(seed)
    torch.manual_seed(seed)

    shuffled = list(examples)
    random.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.85))
    train_examples = shuffled[:split]
    val_examples = shuffled[split:] or shuffled[: min(20, len(shuffled))]

    model = build_torch_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.BCELoss()

    train_x = torch.tensor([item.features for item in train_examples], dtype=torch.float32)
    train_y = torch.tensor([[item.label] for item in train_examples], dtype=torch.float32)
    val_x = torch.tensor([item.features for item in val_examples], dtype=torch.float32)
    val_y = torch.tensor([[item.label] for item in val_examples], dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(train_x), train_y).item())
        val_loss = float(loss_fn(model(val_x), val_y).item())
        val_pred = model(val_x).squeeze(-1).tolist()

    metrics = {
        "train_examples": float(len(train_examples)),
        "validation_examples": float(len(val_examples)),
        "positive_labels": float(sum(1 for item in examples if item.label >= 0.5)),
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "validation_mean_prediction": float(sum(val_pred) / len(val_pred)),
    }
    return model, metrics


def save_model_and_manifest(
    model,
    examples: list[TrainingExample],
    metrics: dict[str, float],
    *,
    model_path: Path,
    manifest_path: Path,
    query_files: list[Path],
) -> None:
    import torch

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    manifest = {
        "model_path": model_path.as_posix(),
        "feature_names": [
            "dense_score",
            "bm25_score",
            "graph_score",
            "same_page_bonus",
            "same_chapter_bonus",
            "chunk_length_norm",
        ],
        "query_files": [path.as_posix() for path in query_files],
        "metrics": metrics,
        "example_count": len(examples),
        "sample_examples": [asdict(item) for item in examples[:5]],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CourseMind MiniRanker model.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--retrieve-top-k", type=int, default=12)
    parser.add_argument("--graph-hops", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_files = find_query_files(args.eval_dir, args.raw_dir)
    if not query_files:
        raise FileNotFoundError("No retrieval query files found.")

    queries = load_retrieval_queries(query_files)
    chunks = load_chunks(args.chunks_path)
    index = load_index(args.index_path) if args.index_path.exists() else build_index(chunks)
    examples = build_examples(
        queries,
        chunks,
        index,
        retrieve_top_k=args.retrieve_top_k,
        graph_hops=args.graph_hops,
        max_examples=args.max_examples,
    )
    model, metrics = train_model(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    save_model_and_manifest(
        model,
        examples,
        metrics,
        model_path=args.output_path,
        manifest_path=args.manifest_path,
        query_files=query_files,
    )

    print(f"Loaded query files: {', '.join(path.as_posix() for path in query_files)}")
    print(f"Built {len(examples)} MiniRanker examples")
    print(f"Saved model: {args.output_path}")
    print(f"Saved manifest: {args.manifest_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
