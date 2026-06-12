from src.bm25_store import keyword_score, tokenize
from src.chunker import chunk_pages
from src.schemas import DocumentPage


def test_chinese_tokenizer_matches_course_terms():
    tokens = tokenize("系统使用检索增强生成和重排序。")

    assert tokens
    assert "检索" in tokens or "检索增强" in tokens or "检索增强生成" in tokens


def test_chinese_keyword_score_is_positive():
    chunk = chunk_pages(
        [DocumentPage(file_name="course.md", page=1, text="RAG 检索增强生成用于中文课程问答。")]
    )[0]

    assert keyword_score("什么是检索增强生成？", chunk) > 0


def test_chinese_concepts_are_detected():
    chunks = chunk_pages(
        [DocumentPage(file_name="course.md", page=1, text="MiniRanker 负责重排序，Bandit 负责强化学习推荐。")]
    )

    assert "MiniRanker" in chunks[0].concepts
    assert "Bandit" in chunks[0].concepts
    assert "强化学习" in chunks[0].concepts
