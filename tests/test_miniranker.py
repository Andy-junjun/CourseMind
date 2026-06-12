from src.chunker import chunk_pages
from src.document_loader import load_pdf
from src.graph_store import expand_with_graph
from src.miniranker import build_feature_matrix, rerank
from src.retriever import retrieve
from src.schemas import Chunk, RetrievedChunk


def make_chunk(chunk_id: str, text: str, page: int = 1, chapter: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_name="course.pdf",
        page=page,
        text=text,
        chapter=chapter,
        concepts=["RAG"],
    )


def test_miniranker_builds_six_features():
    candidates = [
        RetrievedChunk(
            chunk=make_chunk("c1", "RAG 检索增强生成", page=1, chapter="一、方法"),
            dense_score=0.8,
            bm25_score=0.5,
            graph_score=0.0,
        ),
        RetrievedChunk(
            chunk=make_chunk("c2", "同章补充", page=1, chapter="一、方法"),
            dense_score=0.2,
            bm25_score=0.1,
            graph_score=0.7,
        ),
    ]

    features = build_feature_matrix(candidates)

    assert len(features) == 2
    assert all(len(row) == 6 for row in features)
    assert features[0][3] == 1.0
    assert features[1][4] == 1.0


def test_rerank_works_without_miniranker_model(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    candidates = [
        RetrievedChunk(chunk=make_chunk("weak", "无关内容"), dense_score=0.1, bm25_score=0.0),
        RetrievedChunk(chunk=make_chunk("strong", "MiniRanker 重排序"), dense_score=0.8, bm25_score=0.6),
    ]

    ranked = rerank("MiniRanker 如何重排序？", candidates, top_k=2)

    assert ranked[0].chunk.chunk_id == "strong"
    assert all(isinstance(item.ranker_score, float) for item in ranked)
    assert all(0.0 <= item.ranker_score <= 1.0 for item in ranked)


def test_rerank_respects_top_k_and_empty_inputs():
    candidates = [
        RetrievedChunk(chunk=make_chunk("c1", "RAG"), dense_score=0.8, bm25_score=0.5),
        RetrievedChunk(chunk=make_chunk("c2", "BM25"), dense_score=0.7, bm25_score=0.4),
    ]

    assert len(rerank("query", candidates, top_k=1)) == 1
    assert rerank("query", candidates, top_k=0) == []
    assert rerank("query", [], top_k=5) == []


def test_miniranker_pipeline_with_real_course_pdf(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    pages = load_pdf(r"data\raw\深度学习课堂汇报说明.pdf")
    chunks = chunk_pages(pages, chunk_size=260, overlap=40)
    retrieved = retrieve("课程项目评分标准是什么？", chunks, top_k=3)
    expanded = expand_with_graph(retrieved, chunks, hops=1)

    ranked = rerank("课程项目评分标准是什么？", expanded, top_k=3)

    assert len(ranked) == 3
    assert ranked[0].ranker_score >= ranked[-1].ranker_score
    assert all(item.chunk.chunk_id for item in ranked)

