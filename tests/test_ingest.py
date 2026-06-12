from scripts.ingest import build_vector_database


def test_ingest_builds_chunks_and_faiss_index(monkeypatch, tmp_path):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "course.md").write_text(
        "一、向量检索\nCourseMind 使用 FAISS 向量数据库检索中文课程资料。",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "processed" / "chunks.jsonl"
    index_path = tmp_path / "indexes" / "faiss.index"

    pages, chunks = build_vector_database(
        raw_dir=raw_dir,
        chunks_path=chunks_path,
        index_path=index_path,
    )

    assert len(pages) == 1
    assert len(chunks) == 1
    assert chunks_path.exists()
    assert index_path.exists()
    assert (tmp_path / "indexes" / "faiss.meta.json").exists()
