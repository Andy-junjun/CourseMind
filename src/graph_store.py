from src.chunker import DEFAULT_CONCEPT
from src.schemas import Chunk, RetrievedChunk


SAME_PAGE_SCORE = 0.30
SAME_CHAPTER_SCORE = 0.20
ADJACENT_CHUNK_SCORE = 0.25
SHARED_CONCEPT_BASE_SCORE = 0.40
SHARED_CONCEPT_EXTRA_SCORE = 0.10


def expand_with_graph(
    seed_chunks: list[RetrievedChunk], chunks: list[Chunk], hops: int = 1
) -> list[RetrievedChunk]:
    """Expand retrieved seed chunks with lightweight document graph relations.

    GraphRAG-lite here does not build a heavyweight external graph database.
    It treats page, chapter, chunk order, and concepts as graph relations and
    returns additional candidates with `graph_score`.
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
    if not seed_chunks:
        return []

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    seed_ids = [item.chunk.chunk_id for item in seed_chunks]
    seed_id_set = set(seed_ids)

    selected: dict[str, RetrievedChunk] = {
        item.chunk.chunk_id: item for item in seed_chunks
    }

    for chunk in chunks:
        if chunk.chunk_id in seed_id_set:
            continue

        graph_score = score_graph_relation(
            candidate=chunk,
            seed_ids=seed_ids,
            chunk_by_id=chunk_by_id,
            chunk_positions=chunk_positions,
            hops=hops,
        )
        if graph_score > 0:
            selected[chunk.chunk_id] = RetrievedChunk(
                chunk=chunk,
                dense_score=0.0,
                bm25_score=0.0,
                graph_score=graph_score,
            )

    return sort_expanded_results(selected, seed_ids)


def score_graph_relation(
    candidate: Chunk,
    seed_ids: list[str],
    chunk_by_id: dict[str, Chunk],
    chunk_positions: dict[str, int],
    hops: int,
) -> float:
    score = 0.0

    for seed_id in seed_ids:
        seed = chunk_by_id.get(seed_id)
        if seed is None:
            continue

        relation_score = 0.0
        if candidate.file_name == seed.file_name and candidate.page == seed.page:
            relation_score += SAME_PAGE_SCORE

        if same_chapter(candidate, seed):
            relation_score += SAME_CHAPTER_SCORE

        if is_adjacent(candidate, seed, chunk_positions, hops):
            relation_score += ADJACENT_CHUNK_SCORE

        shared_concepts = concept_overlap(candidate, seed)
        if shared_concepts:
            relation_score += SHARED_CONCEPT_BASE_SCORE
            relation_score += SHARED_CONCEPT_EXTRA_SCORE * len(shared_concepts)

        score = max(score, relation_score)

    return round(score, 6)


def same_chapter(candidate: Chunk, seed: Chunk) -> bool:
    return bool(
        candidate.file_name == seed.file_name
        and candidate.chapter
        and seed.chapter
        and candidate.chapter == seed.chapter
    )


def is_adjacent(
    candidate: Chunk,
    seed: Chunk,
    chunk_positions: dict[str, int],
    hops: int,
) -> bool:
    if hops == 0 or candidate.file_name != seed.file_name:
        return False
    candidate_pos = chunk_positions.get(candidate.chunk_id)
    seed_pos = chunk_positions.get(seed.chunk_id)
    if candidate_pos is None or seed_pos is None:
        return False
    return 0 < abs(candidate_pos - seed_pos) <= hops


def concept_overlap(candidate: Chunk, seed: Chunk) -> set[str]:
    candidate_concepts = {
        concept for concept in candidate.concepts if concept != DEFAULT_CONCEPT
    }
    seed_concepts = {concept for concept in seed.concepts if concept != DEFAULT_CONCEPT}
    return candidate_concepts.intersection(seed_concepts)


def sort_expanded_results(
    selected: dict[str, RetrievedChunk], seed_ids: list[str]
) -> list[RetrievedChunk]:
    seed_order = {chunk_id: index for index, chunk_id in enumerate(seed_ids)}

    return sorted(
        selected.values(),
        key=lambda item: (
            0 if item.chunk.chunk_id in seed_order else 1,
            seed_order.get(item.chunk.chunk_id, 0),
            -item.graph_score,
            item.chunk.file_name,
            item.chunk.page,
            item.chunk.chunk_id,
        ),
    )

