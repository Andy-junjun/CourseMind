from src.embedder import embed_text
from src.schemas import Chunk


def build_index(chunks: list[Chunk]) -> dict[str, list[float]]:
    return {chunk.chunk_id: embed_text(chunk.text) for chunk in chunks}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb or 1.0)

