import json
import math
from pathlib import Path

from src.config import get_bandit_state_path


DEFAULT_CONCEPTS = ["Transformer", "RAG", "MiniRanker", "GraphRAG", "Bandit", "大语言模型"]


def load_state(path: Path | None = None) -> dict:
    path = path or get_bandit_state_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"concepts": {concept: {"attempts": 0, "wrong": 0} for concept in DEFAULT_CONCEPTS}}


def save_state(state: dict, path: Path | None = None) -> None:
    path = path or get_bandit_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def update_feedback(concept: str, correct: bool) -> None:
    state = load_state()
    stats = state["concepts"].setdefault(concept, {"attempts": 0, "wrong": 0})
    stats["attempts"] += 1
    if not correct:
        stats["wrong"] += 1
    save_state(state)


def recommend_concept() -> dict:
    state = load_state()
    total_attempts = sum(stats["attempts"] for stats in state["concepts"].values()) + 1
    best_concept = None
    best_score = -1.0
    scores = {}
    for concept, stats in state["concepts"].items():
        attempts = stats["attempts"]
        wrong_rate = stats["wrong"] / attempts if attempts else 0.0
        explore = math.sqrt(math.log(total_attempts + 1) / (attempts + 1))
        score = wrong_rate + 0.8 * explore
        scores[concept] = score
        if score > best_score:
            best_concept = concept
            best_score = score
    return {
        "concept": best_concept or "通用知识点",
        "score": best_score,
        "scores": scores,
        "reason": "分数越高，表示该知识点错题更多，或练习次数较少，需要优先复习。",
    }
