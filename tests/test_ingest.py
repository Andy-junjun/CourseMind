from scripts.ingest import build_vector_database, collect_documents


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


def test_collect_documents_reads_nested_raw_directories(tmp_path):
    raw_dir = tmp_path / "raw"
    member_dir = raw_dir / "xujiaze"
    member_dir.mkdir(parents=True)
    (raw_dir / "root.md").write_text("root", encoding="utf-8")
    (member_dir / "nested.md").write_text("nested", encoding="utf-8")
    (member_dir / "metadata.csv").write_text("file_name,title", encoding="utf-8")

    documents = collect_documents(raw_dir)

    assert [path.relative_to(raw_dir).as_posix() for path in documents] == [
        "root.md",
        "xujiaze/nested.md",
    ]


def test_ingest_keeps_relative_file_names_for_member_data(monkeypatch, tmp_path):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    raw_dir = tmp_path / "raw"
    member_dir = raw_dir / "member"
    member_dir.mkdir(parents=True)
    (member_dir / "course.md").write_text(
        "Transformer 注意力机制用于建模序列中不同位置之间的关系。",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "processed" / "chunks.jsonl"
    index_path = tmp_path / "indexes" / "faiss.index"

    pages, chunks = build_vector_database(
        raw_dir=raw_dir,
        chunks_path=chunks_path,
        index_path=index_path,
    )

    assert pages[0].file_name == "member/course.md"
    assert chunks[0].file_name == "member/course.md"
