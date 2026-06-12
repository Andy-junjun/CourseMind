import hashlib
import math
import os
from functools import lru_cache

from src.bm25_store import tokenize
from src.chunker import DEFAULT_CONCEPT, infer_concepts
from src.config import get_mode


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_LITE_DIM = 384


def embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    mode = get_mode()
    if mode in {"real", "hybrid"}:
        try:
            return embed_text_with_provider(text)
        except ImportError:
            if mode == "real":
                raise RuntimeError(
                    "Real embedding mode requires the configured embedding provider. "
                    "Use EMBEDDING_PROVIDER=lite for a dependency-light local provider."
                )
        except Exception:
            if mode == "real":
                raise

    return mock_embed_text(text, dim=dim)


def mock_embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Deterministic Chinese-friendly fallback embedding.

    This is not a semantic model. It hashes segmented tokens into a fixed-size
    vector so similar Chinese keyword sets produce more similar vectors than a
    whole-string hash would.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")

    vector = [0.0] * dim
    tokens = tokenize(text)
    concepts = [concept for concept in infer_concepts(text) if concept != DEFAULT_CONCEPT]
    for concept in concepts:
        tokens.extend([concept] * 4)
    if not tokens:
        tokens = [text.strip()] if text.strip() else ["<empty>"]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        weight = 1.0 + min(len(token), 8) / 8.0
        vector[bucket] += weight

    return l2_normalize(vector)


def embed_texts(texts: list[str], dim: int = DEFAULT_EMBEDDING_DIM) -> list[list[float]]:
    return [embed_text(text, dim=dim) for text in texts]


def lite_embed_text(text: str, dim: int | None = None) -> list[float]:
    """Lightweight local embedding based on Chinese tokens and character n-grams.

    This provider has no Torch/ONNX dependency. It is stronger than the tiny mock
    hash because it mixes word tokens, character bigrams/trigrams, and concept
    aliases into a larger normalized vector, but it is still lexical rather than
    transformer-level semantic embedding.
    """
    target_dim = dim or int(os.getenv("EMBEDDING_DIM", str(DEFAULT_LITE_DIM)))
    if target_dim <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")

    normalized = text.strip().lower()
    vector = [0.0] * target_dim
    features: list[tuple[str, float]] = []

    for token in tokenize(normalized):
        features.append((f"tok:{token}", 1.0 + min(len(token), 8) / 8.0))

    compact = "".join(ch for ch in normalized if not ch.isspace())
    for ngram_size, weight in ((2, 0.7), (3, 0.9)):
        for index in range(max(0, len(compact) - ngram_size + 1)):
            features.append((f"char{ngram_size}:{compact[index:index + ngram_size]}", weight))

    for concept in infer_concepts(text):
        if concept != DEFAULT_CONCEPT:
            features.append((f"concept:{concept}", 3.0))

    if not features:
        features.append(("<empty>", 1.0))

    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % target_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * weight

    return l2_normalize(vector)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_text_with_model(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in vector.tolist()]


def embed_text_with_provider(text: str) -> list[float]:
    provider = os.getenv("EMBEDDING_PROVIDER", "lite").strip().lower()
    if provider in {"lite", "local_lite", "ngram"}:
        return lite_embed_text(text)
    if provider in {"sentence_transformers", "sentence-transformer", "st"}:
        return embed_text_with_model(text)
    if provider == "fastembed":
        return embed_text_with_fastembed(text)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def embed_text_with_fastembed(text: str) -> list[float]:
    model = get_fastembed_model()
    vector = next(model.embed([text]))
    return l2_normalize([float(value) for value in vector.tolist()])


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_embedding_model_path())


def get_embedding_model_path() -> str:
    return (
        os.getenv("EMBEDDING_MODEL_PATH")
        or os.getenv("EMBEDDING_MODEL_NAME")
        or DEFAULT_MODEL_NAME
    )


@lru_cache(maxsize=1)
def get_fastembed_model():
    from fastembed import TextEmbedding

    model_name = os.getenv("EMBEDDING_MODEL_NAME") or DEFAULT_MODEL_NAME
    return TextEmbedding(model_name=model_name)
