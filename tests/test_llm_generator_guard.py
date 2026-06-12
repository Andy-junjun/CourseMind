import pytest

from src.answer_guard import refusal_message, should_refuse
from src.generator import answer_question, build_citations, summarize_document
from src.llm_client import generate_text, get_llm_config
from src.schemas import Chunk, RankedChunk


def make_ranked(
    chunk_id: str = "c1",
    text: str = "评分标准包括技术深度、演示效果和成员贡献。",
    ranker_score: float = 0.7,
    bm25_score: float = 0.3,
) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            file_name="course.pdf",
            page=2,
            text=text,
            chapter="五、评分标准",
            concepts=["评分标准"],
        ),
        dense_score=0.4,
        bm25_score=bm25_score,
        graph_score=0.2,
        ranker_score=ranker_score,
    )


def test_mock_llm_returns_stable_chinese_text(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")

    first = generate_text("请总结课程资料")
    second = generate_text("请总结课程资料")

    assert first == second
    assert "模拟总结" in first


def test_mock_llm_does_not_refuse_just_because_prompt_mentions_insufficient_evidence(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")

    answer = generate_text("如果证据不足请拒答。问题：项目评分标准是什么？证据：五、评分标准")

    assert "模拟回答" in answer
    assert "当前知识库未检索到足够相关" not in answer


def test_real_llm_without_api_key_raises(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "")

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        generate_text("测试")


def test_real_mode_can_keep_mock_llm_provider(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    answer = generate_text("项目评分标准是什么？")

    assert "模拟回答" in answer


def test_get_llm_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_BASE", "https://example.com/chat")
    monkeypatch.setenv("LLM_TIMEOUT", "12")

    config = get_llm_config()

    assert config.provider == "openai_compatible"
    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.api_base == "https://example.com/chat"
    assert config.timeout == 12


def test_answer_question_returns_answer_and_citations(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    evidence = [make_ranked()]

    result = answer_question("评分标准是什么？", evidence)

    assert "模拟回答" in result["answer"]
    assert "citations" in result
    assert result["evidence_count"] == 1
    citation = result["citations"][0]
    assert citation["chunk_id"] == "c1"
    assert citation["file_name"] == "course.pdf"
    assert citation["page"] == 2
    assert "score" in citation
    assert "text_preview" in citation


def test_build_citations_preserves_scores():
    citation = build_citations([make_ranked(ranker_score=0.66)])[0]

    assert citation["score"] == 0.66
    assert citation["dense_score"] == 0.4
    assert citation["bm25_score"] == 0.3
    assert citation["graph_score"] == 0.2


def test_summarize_document_works_in_mock_mode(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "mock")
    chunks = [
        Chunk(
            chunk_id="c1",
            file_name="course.pdf",
            page=1,
            text="课程项目要求使用深度学习。",
            concepts=["深度学习"],
        ),
        Chunk(
            chunk_id="c2",
            file_name="course.pdf",
            page=2,
            text="评分标准包括技术深度。",
            concepts=["评分标准"],
        ),
    ]

    summary = summarize_document(chunks)

    assert "模拟总结" in summary
    assert "共 2 个 chunk" in summary
    assert "深度学习" in summary
    assert "评分标准" in summary


def test_answer_guard_refuses_empty_or_low_evidence():
    assert should_refuse("YOLOv10 网络结构是什么？", []) is True
    assert should_refuse("YOLOv10 网络结构是什么？", [make_ranked(ranker_score=0.2, bm25_score=0.0)]) is True
    assert refusal_message().startswith("当前知识库")


def test_answer_guard_accepts_strong_evidence():
    assert should_refuse("评分标准是什么？", [make_ranked(ranker_score=0.7, bm25_score=0.3)]) is False
