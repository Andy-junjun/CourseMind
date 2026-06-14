import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.bandit_recommender import (
    load_state,
    recommend_concept,
    record_quiz_result,
    reset_bandit_state,
    save_state,
    get_all_status,
)
from src.schemas import QuizItem, Chunk


def test_bandit_wrong_feedback_is_recorded(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="Transformer 的核心机制？",
        options=["A. 卷积", "B. 自注意力", "C. 循环", "D. 池化"],
        answer="B",
        explanation="Transformer 使用自注意力机制",
        concept="Transformer",
        source_chunk_id="doc01_p02_c03",
    )

    record_quiz_result(quiz, is_wrong=True)

    state = load_state(Path(state_path))
    assert state["concepts"]["Transformer"]["attempts"] == 1
    assert state["concepts"]["Transformer"]["wrong"] == 1
    assert state["concepts"]["Transformer"]["last_seen"] is not None
    assert state["concepts"]["Transformer"]["source_chunk_ids"] == ["doc01_p02_c03"]

    rec = recommend_concept()
    assert "concept" in rec
    assert rec["concept"] == "Transformer"
    assert "错误率" in rec["reason"] or "从未" in rec["reason"]


def test_bandit_correct_feedback_is_recorded(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="CNN 主要用于？",
        options=["A. 文本", "B. 图像", "C. 语音", "D. 推荐"],
        answer="B",
        explanation="CNN 擅长图像",
        concept="CNN",
        source_chunk_id="doc01_p05_c01",
    )

    record_quiz_result(quiz, is_wrong=False)

    state = load_state(Path(state_path))
    assert state["concepts"]["CNN"]["attempts"] == 1
    assert state["concepts"]["CNN"]["wrong"] == 0


def test_bandit_recommends_high_wrong_rate(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz_transformer = QuizItem(
        question="Q1", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="Transformer", source_chunk_id="c1",
    )
    quiz_rnn = QuizItem(
        question="Q2", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="RNN", source_chunk_id="c2",
    )

    record_quiz_result(quiz_transformer, is_wrong=True)
    record_quiz_result(quiz_transformer, is_wrong=False)
    record_quiz_result(quiz_transformer, is_wrong=True)
    record_quiz_result(quiz_rnn, is_wrong=True)

    rec = recommend_concept()
    assert rec["concept"] in ["Transformer", "RNN"]
    assert rec["reason"] != ""
    assert "分数" not in rec["reason"]


def test_bandit_source_chunks_returned(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="Q", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="Transformer", source_chunk_id="chunk_01",
    )

    record_quiz_result(quiz, is_wrong=True)

    chunk_lookup = {
        "chunk_01": Chunk(
            chunk_id="chunk_01",
            file_name="讲义.pdf",
            page=5,
            text="Transformer 内容...",
        )
    }

    rec = recommend_concept(chunk_lookup=chunk_lookup)
    assert len(rec["source_chunks"]) == 1
    assert rec["source_chunks"][0].file_name == "讲义.pdf"
    assert rec["source_chunks"][0].page == 5


def test_bandit_persists_across_loads(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="Q", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="Bandit", source_chunk_id="c1",
    )

    record_quiz_result(quiz, is_wrong=True)
    record_quiz_result(quiz, is_wrong=True)

    state = load_state(Path(state_path))
    assert state["concepts"]["Bandit"]["attempts"] == 2
    assert state["concepts"]["Bandit"]["wrong"] == 2


def test_bandit_all_scores_returned(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="Q", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="Transformer", source_chunk_id="c1",
    )

    record_quiz_result(quiz, is_wrong=False)

    rec = recommend_concept()
    assert "all_scores" in rec
    assert len(rec["all_scores"]) > 0
    assert "Transformer" in rec["all_scores"]


def test_bandit_get_all_status(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    quiz = QuizItem(
        question="Q", options=["A", "B", "C", "D"], answer="B",
        explanation="E", concept="RNN", source_chunk_id="c1",
    )

    record_quiz_result(quiz, is_wrong=True)

    summary = get_all_status()
    assert summary["total_concepts"] > 0
    assert summary["total_attempts"] == 1
    assert summary["total_wrong"] == 1
    assert "RNN" in summary["concepts"]
    assert summary["concepts"]["RNN"]["wrong_rate"] == 1.0


def test_bandit_reset_clears_answer_history(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))
    quiz = QuizItem(
        question="Q", options=["A", "B"], answer="A",
        explanation="E", concept="Transformer", source_chunk_id="c1",
    )

    record_quiz_result(quiz, is_wrong=True)
    reset_bandit_state()

    state = load_state()
    assert state["total_attempts"] == 0
    assert state["concepts"]["Transformer"]["attempts"] == 0
    assert state["concepts"]["Transformer"]["wrong"] == 0
    assert state["concepts"]["Transformer"]["last_seen"] is None
    assert state["concepts"]["Transformer"]["source_chunk_ids"] == []


def test_recommendation_restricted_to_quizzable_concepts(tmp_path, monkeypatch):
    """Regression: the recommended concept must be one the quiz can actually use.

    Previously Bandit could recommend a concept that generate_quiz could not
    produce a question for, so the practice tab silently fell back to an
    unrelated concept and the "推荐复习" and the quiz disagreed.
    """
    from src.study_tools import quizzable_concepts, generate_quiz

    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))

    chunks = [
        Chunk(chunk_id="c1", file_name="a.md", page=1,
              text="激活函数会引入非线性，使神经网络能够学习复杂的关系。",
              concepts=["激活函数", "神经网络"]),
        Chunk(chunk_id="c2", file_name="a.md", page=1,
              text="梯度下降通过沿负梯度方向更新参数来减小损失函数。",
              concepts=["梯度下降", "损失函数"]),
        Chunk(chunk_id="c3", file_name="a.md", page=1,
              text="卷积层用于提取局部特征，池化层可以减少特征图的尺寸。",
              concepts=["卷积层", "池化层"]),
    ]
    quizzable = quizzable_concepts(chunks)
    assert quizzable, "fixture should yield at least one quizzable concept"

    lookup = {c.chunk_id: c for c in chunks}
    # Seed an out-of-scope concept that has no quizzable statement; without the
    # allowed_concepts guard UCB would happily recommend it (never seen).
    record_quiz_result(
        QuizItem(question="Q", options=["A", "B"], answer="A", explanation="E",
                 concept="深度强化学习", source_chunk_id="missing"),
        is_wrong=True,
    )

    rec = recommend_concept(chunk_lookup=lookup, allowed_concepts=quizzable)
    assert rec["concept"] in quizzable

    quiz = generate_quiz(chunks, num_questions=30, target_concept=rec["concept"])
    assert quiz, "recommended concept must be able to produce a quiz"
    assert all(item.concept == rec["concept"] for item in quiz)
