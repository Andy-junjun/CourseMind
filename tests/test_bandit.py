from pathlib import Path

from src.bandit_recommender import load_state, recommend_concept, save_state, update_feedback


def test_bandit_wrong_feedback_is_recorded(tmp_path, monkeypatch):
    state_path = tmp_path / "quiz_state.json"
    monkeypatch.setenv("BANDIT_STATE_PATH", str(state_path))
    update_feedback("Transformer", correct=False)
    state = load_state(Path(state_path))
    assert state["concepts"]["Transformer"]["attempts"] == 1
    assert state["concepts"]["Transformer"]["wrong"] == 1
    rec = recommend_concept()
    assert "concept" in rec

