from src.chunker import chunk_pages
from src.embedder import embed_text
from src.retriever import retrieve
from src.schemas import DocumentPage
from src.vector_store import (
    build_index,
    cosine,
    load_vector_store,
    persist_vector_store,
    search_index,
)


def test_mock_embedding_is_deterministic_and_normalized(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")

    first = embed_text("检索增强生成用于中文课程问答")
    second = embed_text("检索增强生成用于中文课程问答")

    assert first == second
    assert len(first) == 128
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_faiss_index_search_ranks_related_chinese_chunk_first(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = chunk_pages(
        [
            DocumentPage(file_name="course.md", page=1, text="RAG 检索增强生成可以根据课程资料回答问题。"),
            DocumentPage(file_name="course.md", page=2, text="评分标准包括技术深度、演示效果和团队分工。"),
            DocumentPage(file_name="course.md", page=3, text="Bandit 会根据答题反馈推荐薄弱知识点。"),
        ],
        chunk_size=80,
        overlap=10,
    )

    index = build_index(chunks)
    results = search_index("什么是检索增强生成？", chunks, top_k=2, index=index)

    assert index.index.ntotal == len(chunks)
    assert index.dim == 128
    assert results[0][0].page == 1
    assert isinstance(results[0][1], float)


def test_faiss_vector_store_persists_and_reloads(monkeypatch, tmp_path):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = chunk_pages(
        [
            DocumentPage(file_name="course.md", page=1, text="向量数据库使用 FAISS 保存课程资料 embedding。"),
            DocumentPage(file_name="course.md", page=2, text="MiniRanker 负责对候选片段进行重排序。"),
        ]
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "faiss.index"

    persist_vector_store(chunks, chunks_path=chunks_path, index_path=index_path)
    loaded_chunks, loaded_index = load_vector_store(chunks_path=chunks_path, index_path=index_path)
    results = search_index("FAISS 向量数据库保存了什么？", loaded_chunks, top_k=1, index=loaded_index)

    assert chunks_path.exists()
    assert index_path.exists()
    assert (tmp_path / "faiss.meta.json").exists()
    assert len(loaded_chunks) == len(chunks)
    assert loaded_index.index.ntotal == len(chunks)
    assert results[0][0].page == 1


def test_retrieve_exposes_dense_and_bm25_scores(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = chunk_pages(
        [
            DocumentPage(file_name="course.md", page=1, text="MiniRanker 负责对候选片段进行重排序。"),
            DocumentPage(file_name="course.md", page=2, text="PPT 需要标注成员贡献。"),
        ]
    )
    index = build_index(chunks)

    results = retrieve("MiniRanker 如何进行重排序？", chunks, top_k=1, index=index)

    assert results[0].chunk.page == 1
    assert isinstance(results[0].dense_score, float)
    assert isinstance(results[0].bm25_score, float)


def test_cosine_handles_empty_vectors():
    assert cosine([], []) == 0.0
