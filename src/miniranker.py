import math
import os
import re
from pathlib import Path
from typing import Protocol

from src.config import get_mode
from src.schemas import RankedChunk, RetrievedChunk


DEFAULT_MODEL_PATH = Path("data/training/miniranker.pt")


class RankerModel(Protocol):
    def predict_scores(self, features: list[list[float]]) -> list[float]:
        ...


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RankedChunk]:
    if top_k <= 0 or not candidates:
        return []

    features = build_feature_matrix(candidates)
    model = load_ranker_model()
    raw_scores = model.predict_scores(features) if model else heuristic_scores(features)
    scores = [
        calibrated_score(item, score)
        for item, score in zip(candidates, raw_scores)
    ]

    ranked = [
        RankedChunk(
            chunk=item.chunk,
            dense_score=item.dense_score,
            bm25_score=item.bm25_score,
            graph_score=item.graph_score,
            ranker_score=float(score),
        )
        for item, score in zip(candidates, scores)
    ]
    ranked.sort(
        key=lambda item: (
            item.ranker_score,
            item.bm25_score,
            item.dense_score,
            item.graph_score,
            -item.chunk.page,
            item.chunk.chunk_id,
        ),
        reverse=True,
    )
    return ranked[:top_k]


def build_feature_matrix(candidates: list[RetrievedChunk]) -> list[list[float]]:
    anchor = choose_anchor(candidates)
    return [extract_features(item, anchor) for item in candidates]


def extract_features(item: RetrievedChunk, anchor: RetrievedChunk) -> list[float]:
    return [
        clip01(item.dense_score),
        clip01(item.bm25_score),
        clip01(item.graph_score),
        same_page_bonus(item, anchor),
        same_chapter_bonus(item, anchor),
        chunk_length_norm(item),
    ]


def choose_anchor(candidates: list[RetrievedChunk]) -> RetrievedChunk:
    return max(candidates, key=lambda item: 0.6 * item.dense_score + 0.4 * item.bm25_score)


def heuristic_scores(features: list[list[float]]) -> list[float]:
    weights = [0.42, 0.32, 0.14, 0.05, 0.04, 0.03]
    return [sigmoid(sum(value * weight for value, weight in zip(row, weights))) for row in features]


def load_ranker_model() -> RankerModel | None:
    model_path = Path(os.getenv("MINIRANKER_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if get_mode() == "mock" or not model_path.exists():
        return None
    return TorchMiniRankerModel(model_path)


def build_torch_model():
    try:
        from torch import nn
    except ImportError as exc:
        if get_mode() == "real":
            raise RuntimeError("MiniRanker real mode requires PyTorch.") from exc
        raise

    return nn.Sequential(
        nn.Linear(6, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


class TorchMiniRankerModel:
    def __init__(self, model_path: Path):
        try:
            import torch
        except ImportError as exc:
            if get_mode() == "real":
                raise RuntimeError("MiniRanker real mode requires PyTorch.") from exc
            raise

        self.torch = torch
        self.model = build_torch_model()
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_scores(self, features: list[list[float]]) -> list[float]:
        with self.torch.no_grad():
            tensor = self.torch.tensor(features, dtype=self.torch.float32)
            scores = self.model(tensor).squeeze(-1)
            return [float(value) for value in scores.tolist()]


def same_page_bonus(item: RetrievedChunk, anchor: RetrievedChunk) -> float:
    return 1.0 if item.chunk.file_name == anchor.chunk.file_name and item.chunk.page == anchor.chunk.page else 0.0


def same_chapter_bonus(item: RetrievedChunk, anchor: RetrievedChunk) -> float:
    return 1.0 if item.chunk.chapter and item.chunk.chapter == anchor.chunk.chapter else 0.0


def chunk_length_norm(item: RetrievedChunk, target_length: int = 400) -> float:
    length = len(item.chunk.text)
    if length <= 0:
        return 0.0
    return min(length, target_length) / target_length


def content_quality_factor(item: RetrievedChunk) -> float:
    """Down-weight chunks that are too short to be useful as answer evidence."""
    text = item.chunk.text.strip()
    compact = re.sub(r"[\s#*`_\-–—>]+", "", text)
    if len(compact) < 20:
        return 0.35
    if text.startswith("#") and len(compact) < 60:
        return 0.55
    return 1.0


def calibrated_score(item: RetrievedChunk, model_score: float) -> float:
    blended = (
        0.45 * clip01(model_score)
        + 0.35 * clip01(item.dense_score)
        + 0.15 * clip01(item.bm25_score)
        + 0.05 * clip01(item.graph_score)
    )
    return clip01(blended * content_quality_factor(item))


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
