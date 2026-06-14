from src.chunker import chunk_pages
from src.embedder import embed_text
from src.retriever import retrieve
from src.schemas import DocumentPage
from src.vector_store import (
    build_index,
    cosine,
    extend_index,
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


def test_lite_embedding_runs_in_real_mode_without_native_model(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "lite")
    monkeypatch.setenv("EMBEDDING_DIM", "384")

    vector = embed_text("这个项目如何评分？")

    assert len(vector) == 384
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6


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


def test_faiss_inner_product_matches_cosine_for_normalized_embeddings(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "lite")
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    chunks = chunk_pages(
        [
            DocumentPage(file_name="course.md", page=1, text="Transformer attention uses query key value vectors."),
            DocumentPage(file_name="course.md", page=2, text="Batch normalization stabilizes training."),
        ]
    )
    query = "query key value attention"

    index = build_index(chunks)
    results = search_index(query, chunks, top_k=1, index=index)
    expected = cosine(embed_text(query), embed_text(results[0][0].text))

    assert -1.0 <= results[0][1] <= 1.0
    assert abs(results[0][1] - expected) < 1e-5


def test_cosine_handles_empty_vectors():
    assert cosine([], []) == 0.0


def test_extend_index_merges_new_file_without_touching_base(monkeypatch):
    """Importing a file embeds only the new chunks and merges them in.

    The original index must stay searchable and must not be mutated in place
    (build_knowledge_base caches the base index and reuses it across uploads).
    """
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    base_chunks = chunk_pages(
        [DocumentPage(file_name="base.md", page=1, text="RAG 检索增强生成根据课程资料回答问题。")]
    )
    new_chunks = chunk_pages(
        [DocumentPage(file_name="upload.md", page=1, text="Transformer 使用自注意力机制处理序列。")]
    )

    base = build_index(base_chunks)
    base_total_before = base.index.ntotal
    base_ids_before = list(base.chunk_ids)

    merged = extend_index(base, new_chunks)

    # base is untouched (purity)
    assert base.index.ntotal == base_total_before
    assert base.chunk_ids == base_ids_before
    # merged has both
    assert merged.index.ntotal == len(base_chunks) + len(new_chunks)
    assert merged.chunk_ids == base.chunk_ids + [c.chunk_id for c in new_chunks]

    # uploaded content is now searchable
    all_chunks = base_chunks + new_chunks
    hits = search_index("自注意力机制是什么", all_chunks, top_k=1, index=merged)
    assert hits[0][0].file_name == "upload.md"


def test_extend_index_skips_duplicate_chunk_ids(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = chunk_pages(
        [DocumentPage(file_name="base.md", page=1, text="向量数据库使用 FAISS 保存 embedding。")]
    )
    base = build_index(chunks)

    # re-extending with the same chunks is a no-op (idempotent)
    merged = extend_index(base, chunks)
    assert merged.index.ntotal == len(chunks)
    assert merged.chunk_ids == base.chunk_ids


def test_extend_index_empty_new_chunks_returns_same(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = chunk_pages(
        [DocumentPage(file_name="base.md", page=1, text="评分标准包括技术深度与演示效果。")]
    )
    base = build_index(chunks)
    assert extend_index(base, []) is base
