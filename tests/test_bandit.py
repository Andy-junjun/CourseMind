import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.bandit_recommender import (
    load_state,
    recommend_concept,
    record_quiz_result,
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
