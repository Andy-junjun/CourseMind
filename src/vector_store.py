from src.embedder import embed_text
from src.schemas import Chunk


VectorIndex = dict[str, list[float]]


def build_index(chunks: list[Chunk]) -> VectorIndex:
    return {chunk.chunk_id: embed_text(chunk.text) for chunk in chunks}


def search_index(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    index: VectorIndex | None = None,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0:
        return []

    vector_index = index or build_index(chunks)
    query_vector = embed_text(query)
    scored = [
        (chunk, cosine(query_vector, vector_index.get(chunk.chunk_id, embed_text(chunk.text))))
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb or 1.0)

