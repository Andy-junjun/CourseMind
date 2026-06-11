from src.schemas import RankedChunk


def recall_at_k(expected_chunk_ids: set[str], ranked: list[RankedChunk], k: int = 5) -> float:
    if not expected_chunk_ids:
        return 0.0
    predicted = {item.chunk.chunk_id for item in ranked[:k]}
    return len(expected_chunk_ids.intersection(predicted)) / len(expected_chunk_ids)

