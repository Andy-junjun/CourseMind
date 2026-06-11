from src.schemas import Chunk, RetrievedChunk


def expand_with_graph(
    seed_chunks: list[RetrievedChunk], chunks: list[Chunk], hops: int = 1
) -> list[RetrievedChunk]:
    selected: dict[str, RetrievedChunk] = {item.chunk.chunk_id: item for item in seed_chunks}
    seed_ids = {item.chunk.chunk_id for item in seed_chunks}
    seed_concepts = {concept for item in seed_chunks for concept in item.chunk.concepts}
    seed_pages = {item.chunk.page for item in seed_chunks}

    for chunk in chunks:
        if chunk.chunk_id in selected:
            continue
        score = 0.0
        if chunk.page in seed_pages:
            score += 0.3
        shared = seed_concepts.intersection(chunk.concepts)
        if shared:
            score += 0.4 + 0.1 * len(shared)
        if score > 0:
            selected[chunk.chunk_id] = RetrievedChunk(
                chunk=chunk,
                dense_score=0.0,
                bm25_score=0.0,
                graph_score=score,
            )
    return list(selected.values())

