from src.schemas import RankedChunk, RetrievedChunk


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RankedChunk]:
    ranked = [
        RankedChunk(
            chunk=item.chunk,
            dense_score=item.dense_score,
            bm25_score=item.bm25_score,
            graph_score=item.graph_score,
            ranker_score=0.45 * item.dense_score + 0.35 * item.bm25_score + 0.20 * item.graph_score,
        )
        for item in candidates
    ]
    ranked.sort(key=lambda item: item.ranker_score, reverse=True)
    return ranked[:top_k]

