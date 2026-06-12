from src.bm25_store import keyword_score
from src.schemas import Chunk, RetrievedChunk
from src.vector_store import search_index


def retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]:
    if top_k <= 0:
        return []

    dense_results = search_index(query, chunks, top_k=max(top_k, len(chunks)))
    candidates = []
    for chunk, dense_score in dense_results:
        bm25_score = keyword_score(query, chunk)
        candidates.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=dense_score,
                bm25_score=bm25_score,
                graph_score=0.0,
            )
        )

    candidates.sort(
        key=lambda item: hybrid_score(item.dense_score, item.bm25_score),
        reverse=True,
    )
    return candidates[:top_k]


def hybrid_score(dense_score: float, bm25_score: float) -> float:
    return 0.6 * dense_score + 0.4 * bm25_score

