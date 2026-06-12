import hashlib
import math
import os
from functools import lru_cache

from src.bm25_store import tokenize
from src.chunker import DEFAULT_CONCEPT, infer_concepts
from src.config import get_mode


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    mode = get_mode()
    if mode in {"real", "hybrid"}:
        try:
            return embed_text_with_model(text)
        except ImportError:
            if mode == "real":
                raise RuntimeError(
                    "Real embedding mode requires sentence-transformers. "
                    "Install it with `pip install sentence-transformers`."
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


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_text_with_model(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in vector.tolist()]


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    return SentenceTransformer(model_name)
