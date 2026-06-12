import pytest

from src.chunker import chunk_pages, infer_concepts
from src.schemas import DocumentPage


def test_chunker_outputs_stable_required_fields():
    pages = [DocumentPage(file_name="course.pdf", page=1, text="Transformer RAG " * 50)]

    chunks = chunk_pages(pages, chunk_size=80, overlap=10)

    assert chunks
    first = chunks[0]
    assert first.chunk_id == "course_pdf_p001_c001"
    assert first.file_name == "course.pdf"
    assert first.page == 1
    assert first.text
    assert first.chapter is None
    assert "Transformer" in first.concepts


def test_chunker_keeps_chunk_id_stable_across_runs():
    pages = [
        DocumentPage(
            file_name="深度学习课堂汇报说明.pdf",
            page=2,
            text="五、评分标准\n技术深度与正确性。",
        )
    ]

    first_run = chunk_pages(pages, chunk_size=40, overlap=5)
    second_run = chunk_pages(pages, chunk_size=40, overlap=5)

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert first_run[0].chunk_id == "深度学习课堂汇报说明_pdf_p002_c001"


def test_chunker_extracts_chinese_chapter_heading():
    pages = [
        DocumentPage(
            file_name="course.md",
            page=1,
            text=(
                "一、项目目标\n"
                "本课程要求基于深度学习或强化学习实现可演示系统。\n"
                "二、评分标准\n"
                "满分100分。"
            ),
        )
    ]

    chunks = chunk_pages(pages, chunk_size=80, overlap=10)

    assert chunks[0].chapter == "一、项目目标"
    assert chunks[-1].chapter == "二、评分标准"
    assert "深度学习" in chunks[0].concepts
    assert "强化学习" in chunks[0].concepts
    assert "评分标准" in chunks[-1].concepts


def test_chunk_size_and_overlap_are_configurable():
    text = "。".join([f"这是第{i}句，包含检索增强生成和重排序" for i in range(20)])
    pages = [DocumentPage(file_name="course.txt", page=1, text=text)]

    small_chunks = chunk_pages(pages, chunk_size=60, overlap=5)
    large_chunks = chunk_pages(pages, chunk_size=180, overlap=5)

    assert len(small_chunks) > len(large_chunks)
    assert all(len(chunk.text) <= 180 for chunk in large_chunks)


def test_invalid_chunk_arguments_raise_errors():
    pages = [DocumentPage(file_name="course.txt", page=1, text="测试")]

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pages(pages, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(pages, chunk_size=20, overlap=20)


def test_infer_concepts_supports_aliases_and_defaults():
    concepts = infer_concepts("系统使用检索增强生成、神经重排序和多臂老虎机进行复习推荐。")

    assert "RAG" in concepts
    assert "MiniRanker" in concepts
    assert "Bandit" in concepts
    assert infer_concepts("普通课程内容") == ["通用知识点"]
