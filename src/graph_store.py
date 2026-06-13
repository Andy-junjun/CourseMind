from dataclasses import dataclass
from typing import Any

from src.chunker import DEFAULT_CONCEPT
from src.schemas import Chunk, RetrievedChunk


SAME_PAGE_SCORE = 0.30
SAME_CHAPTER_SCORE = 0.20
ADJACENT_CHUNK_SCORE = 0.25
SHARED_CONCEPT_BASE_SCORE = 0.40
SHARED_CONCEPT_EXTRA_SCORE = 0.10


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    score: float
    reason: str


@dataclass(frozen=True)
class GraphRelation:
    seed_chunk_id: str
    candidate_chunk_id: str
    relation: str
    score: float
    reason: str


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
        item.chunk.chunk_id: RetrievedChunk(
            chunk=item.chunk,
            dense_score=item.dense_score,
            bm25_score=item.bm25_score,
            graph_score=max(item.graph_score, 1.0),
        )
        for item in seed_chunks
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


def build_document_graph(chunks: list[Chunk]) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}
    chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    for chunk in chunks:
        file_id = f"file:{chunk.file_name}"
        page_id = f"page:{chunk.file_name}:{chunk.page}"
        chunk_id = f"chunk:{chunk.chunk_id}"
        nodes.setdefault(
            file_id,
            GraphNode(file_id, "file", chunk.file_name, {"file_name": chunk.file_name}),
        )
        nodes.setdefault(
            page_id,
            GraphNode(
                page_id,
                "page",
                f"{chunk.file_name} p.{chunk.page}",
                {"file_name": chunk.file_name, "page": chunk.page},
            ),
        )
        nodes.setdefault(
            chunk_id,
            GraphNode(
                chunk_id,
                "chunk",
                chunk.chunk_id,
                {
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.file_name,
                    "page": chunk.page,
                    "chapter": chunk.chapter,
                },
            ),
        )
        add_edge(
            edges,
            file_id,
            page_id,
            "contains_page",
            1.0,
            f"文件 {chunk.file_name} 包含第 {chunk.page} 页",
        )
        add_edge(
            edges,
            page_id,
            chunk_id,
            "contains_chunk",
            1.0,
            f"第 {chunk.page} 页包含 chunk {chunk.chunk_id}",
        )
        for concept in chunk.concepts:
            if concept == DEFAULT_CONCEPT:
                continue
            concept_id = f"concept:{concept}"
            nodes.setdefault(
                concept_id,
                GraphNode(concept_id, "concept", concept, {"concept": concept}),
            )
            add_edge(
                edges,
                chunk_id,
                concept_id,
                "mentions_concept",
                1.0,
                f"chunk {chunk.chunk_id} 提到知识点 {concept}",
            )

    for index, chunk in enumerate(chunks):
        if index == 0:
            continue
        previous = chunks[index - 1]
        if previous.file_name == chunk.file_name:
            add_edge(
                edges,
                f"chunk:{previous.chunk_id}",
                f"chunk:{chunk.chunk_id}",
                "adjacent",
                ADJACENT_CHUNK_SCORE,
                "两个 chunk 在同一文件中相邻",
            )

    return list(nodes.values()), list(edges.values())


def add_edge(
    edges: dict[tuple[str, str, str], GraphEdge],
    source_id: str,
    target_id: str,
    relation: str,
    score: float,
    reason: str,
) -> None:
    edges.setdefault(
        (source_id, target_id, relation),
        GraphEdge(source_id, target_id, relation, score, reason),
    )


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

        relation_score = sum(
            relation.score
            for relation in explain_candidate_seed_relation(
                candidate, seed, chunk_positions, hops
            )
        )
        score = max(score, relation_score)

    return round(score, 6)


def explain_graph_expansion(
    candidate: Chunk,
    seed_chunks: list[RetrievedChunk],
    chunks: list[Chunk],
    hops: int = 1,
) -> list[GraphRelation]:
    chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    relations: list[GraphRelation] = []
    for seed_item in seed_chunks:
        relations.extend(
            explain_candidate_seed_relation(
                candidate,
                seed_item.chunk,
                chunk_positions,
                hops,
            )
        )
    return sorted(
        relations,
        key=lambda item: (-item.score, item.seed_chunk_id, item.relation),
    )


def explain_candidate_seed_relation(
    candidate: Chunk,
    seed: Chunk,
    chunk_positions: dict[str, int],
    hops: int,
) -> list[GraphRelation]:
    relations: list[GraphRelation] = []
    if candidate.chunk_id == seed.chunk_id:
        relations.append(
            GraphRelation(
                seed_chunk_id=seed.chunk_id,
                candidate_chunk_id=candidate.chunk_id,
                relation="seed",
                score=1.0,
                reason="原始检索命中的种子 chunk",
            )
        )
        return relations

    if candidate.file_name == seed.file_name and candidate.page == seed.page:
        relations.append(
            GraphRelation(
                seed_chunk_id=seed.chunk_id,
                candidate_chunk_id=candidate.chunk_id,
                relation="same_page",
                score=SAME_PAGE_SCORE,
                reason=f"与种子 chunk {seed.chunk_id} 位于同一页 p.{seed.page}",
            )
        )

    if same_chapter(candidate, seed):
        relations.append(
            GraphRelation(
                seed_chunk_id=seed.chunk_id,
                candidate_chunk_id=candidate.chunk_id,
                relation="same_chapter",
                score=SAME_CHAPTER_SCORE,
                reason=f"与种子 chunk {seed.chunk_id} 同属章节 {candidate.chapter}",
            )
        )

    if is_adjacent(candidate, seed, chunk_positions, hops):
        relations.append(
            GraphRelation(
                seed_chunk_id=seed.chunk_id,
                candidate_chunk_id=candidate.chunk_id,
                relation="adjacent",
                score=ADJACENT_CHUNK_SCORE,
                reason=f"与种子 chunk {seed.chunk_id} 在同一文件中相邻",
            )
        )

    shared_concepts = concept_overlap(candidate, seed)
    if shared_concepts:
        score = SHARED_CONCEPT_BASE_SCORE + SHARED_CONCEPT_EXTRA_SCORE * len(shared_concepts)
        relations.append(
            GraphRelation(
                seed_chunk_id=seed.chunk_id,
                candidate_chunk_id=candidate.chunk_id,
                relation="shared_concept",
                score=score,
                reason="共享知识点：" + "、".join(sorted(shared_concepts)),
            )
        )

    return relations


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
