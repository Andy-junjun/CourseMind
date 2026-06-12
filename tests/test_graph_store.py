import pytest

from src.graph_store import build_document_graph, expand_with_graph, explain_graph_expansion
from src.schemas import Chunk, RetrievedChunk


def make_chunk(
    chunk_id: str,
    page: int,
    text: str,
    chapter: str | None,
    concepts: list[str],
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_name="course.pdf",
        page=page,
        text=text,
        chapter=chapter,
        concepts=concepts,
    )


def test_expand_with_graph_adds_same_page_candidate():
    chunks = [
        make_chunk("c1", 1, "RAG 检索增强生成", "一、项目目标", ["RAG"]),
        make_chunk("c2", 1, "同页补充说明", "一、项目目标", ["通用知识点"]),
        make_chunk("c3", 2, "无关内容", "二、评分标准", ["评分标准"]),
    ]
    seeds = [RetrievedChunk(chunk=chunks[0], dense_score=0.8, bm25_score=0.6)]

    expanded = expand_with_graph(seeds, chunks, hops=0)
    by_id = {item.chunk.chunk_id: item for item in expanded}

    assert "c2" in by_id
    assert by_id["c2"].graph_score >= 0.3
    assert by_id["c2"].dense_score == 0.0
    assert by_id["c1"].dense_score == 0.8


def test_expand_with_graph_adds_shared_concept_candidate():
    chunks = [
        make_chunk("c1", 1, "MiniRanker 重排序", "一、方法", ["MiniRanker"]),
        make_chunk("c2", 2, "重排序实验结果", "二、实验", ["MiniRanker"]),
        make_chunk("c3", 3, "成员贡献说明", "三、贡献", ["成员贡献"]),
    ]
    seeds = [RetrievedChunk(chunk=chunks[0], dense_score=0.7, bm25_score=0.5)]

    expanded = expand_with_graph(seeds, chunks, hops=0)
    by_id = {item.chunk.chunk_id: item for item in expanded}

    assert "c2" in by_id
    assert by_id["c2"].graph_score >= 0.5
    assert "c3" not in by_id


def test_expand_with_graph_adds_adjacent_candidate_with_hops():
    chunks = [
        make_chunk("c1", 1, "第一段", "一、项目目标", ["通用知识点"]),
        make_chunk("c2", 2, "第二段", "二、项目要求", ["通用知识点"]),
        make_chunk("c3", 3, "第三段", "三、评分标准", ["评分标准"]),
    ]
    seeds = [RetrievedChunk(chunk=chunks[1])]

    expanded_h0 = expand_with_graph(seeds, chunks, hops=0)
    expanded_h1 = expand_with_graph(seeds, chunks, hops=1)

    assert {item.chunk.chunk_id for item in expanded_h0} == {"c2"}
    assert {item.chunk.chunk_id for item in expanded_h1} == {"c1", "c2", "c3"}


def test_expand_with_graph_uses_retrieved_chunk_schema_and_stable_order():
    chunks = [
        make_chunk("c1", 1, "RAG", "一、方法", ["RAG"]),
        make_chunk("c2", 1, "RAG 同页", "一、方法", ["RAG"]),
        make_chunk("c3", 2, "评分标准", "二、评分", ["评分标准"]),
    ]
    seeds = [RetrievedChunk(chunk=chunks[0], dense_score=0.9, bm25_score=0.4)]

    first = expand_with_graph(seeds, chunks, hops=1)
    second = expand_with_graph(seeds, chunks, hops=1)

    assert [item.chunk.chunk_id for item in first] == [item.chunk.chunk_id for item in second]
    assert first[0].chunk.chunk_id == "c1"
    assert all(isinstance(item, RetrievedChunk) for item in first)


def test_expand_with_graph_rejects_negative_hops():
    chunk = make_chunk("c1", 1, "RAG", None, ["RAG"])

    with pytest.raises(ValueError, match="hops"):
        expand_with_graph([RetrievedChunk(chunk=chunk)], [chunk], hops=-1)


def test_build_document_graph_exposes_nodes_and_edges():
    chunks = [
        make_chunk("c1", 1, "Transformer 注意力", "一、注意力", ["Transformer"]),
        make_chunk("c2", 1, "多头注意力", "一、注意力", ["Transformer"]),
    ]

    nodes, edges = build_document_graph(chunks)
    node_types = {node.node_type for node in nodes}
    relations = {edge.relation for edge in edges}

    assert {"file", "page", "chunk", "concept"}.issubset(node_types)
    assert {"contains_page", "contains_chunk", "mentions_concept", "adjacent"}.issubset(
        relations
    )


def test_explain_graph_expansion_lists_relation_reasons():
    chunks = [
        make_chunk("c1", 1, "Transformer 注意力", "一、注意力", ["Transformer"]),
        make_chunk("c2", 1, "多头注意力", "一、注意力", ["Transformer"]),
        make_chunk("c3", 3, "无关内容", "二、其他", ["其他"]),
    ]
    seed = RetrievedChunk(chunk=chunks[0], dense_score=0.9, bm25_score=0.3)

    relations = explain_graph_expansion(chunks[1], [seed], chunks, hops=1)

    assert {relation.relation for relation in relations} == {
        "same_page",
        "same_chapter",
        "adjacent",
        "shared_concept",
    }
    assert any("共享知识点" in relation.reason for relation in relations)
