import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_bandit_state_path
from src.schemas import Chunk, QuizItem


DEFAULT_CONCEPTS = [
    "Transformer", "RAG", "MiniRanker", "GraphRAG",
    "Bandit", "大语言模型", "反向传播", "梯度下降",
    "注意力机制", "CNN", "RNN"
]


def load_state(path: Path | None = None) -> dict:
    path = path or get_bandit_state_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "concepts": {
            concept: {"attempts": 0, "wrong": 0, "last_seen": None, "source_chunk_ids": []}
            for concept in DEFAULT_CONCEPTS
        },
        "total_attempts": 0
    }


def save_state(state: dict, path: Path | None = None) -> None:
    path = path or get_bandit_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def record_quiz_result(quiz_item: QuizItem, is_wrong: bool) -> None:
    state = load_state()
    concept = quiz_item.concept
    stats = state["concepts"].setdefault(concept, {
        "attempts": 0, "wrong": 0, "last_seen": None, "source_chunk_ids": []
    })
    stats["attempts"] += 1
    if is_wrong:
        stats["wrong"] += 1
    stats["last_seen"] = datetime.now().isoformat()
    source_id = quiz_item.source_chunk_id
    if source_id and source_id not in stats["source_chunk_ids"]:
        stats["source_chunk_ids"].append(source_id)
    state["total_attempts"] += 1
    save_state(state)


def _ucb_score(stats: dict, total_attempts: int, c: float = 1.414) -> float:
    attempts = stats["attempts"]
    wrong = stats["wrong"]
    wrong_rate = wrong / attempts if attempts > 0 else 0.0
    if attempts == 0:
        # 从未练习过：给一个较高的探索分数，但不是无限大
        # 基于总尝试次数动态调整，确保已练习但错误率高的概念也能被推荐
        return 0.5 + c * math.sqrt(math.log(total_attempts + 1))
    explore = c * math.sqrt(math.log(total_attempts + 1) / attempts)
    return wrong_rate + explore


def _generate_reason(stats: dict, score: float) -> str:
    attempts = stats["attempts"]
    wrong = stats["wrong"]
    wrong_rate = wrong / attempts if attempts > 0 else 0.0
    reasons = []
    if wrong_rate >= 0.5:
        reasons.append(f"错误率 {wrong_rate:.0%}（{wrong}/{attempts}）")
    elif wrong > 0:
        reasons.append(f"有 {wrong} 次错误记录")
    if attempts == 0:
        return "从未练习过，建议优先探索"
    elif attempts <= 2:
        reasons.append(f"仅练习 {attempts} 次，样本不足")
    if not reasons:
        return "目前掌握较好，建议巩固"
    return "；".join(reasons) + " → 建议优先复习"


def recommend_concept(chunk_lookup: Optional[Dict[str, Chunk]] = None) -> dict:
    state = load_state()
    total_attempts = state.get("total_attempts", 0)
    best_concept = None
    best_score = -1.0
    all_scores = {}
    for concept, stats in state["concepts"].items():
        score = _ucb_score(stats, total_attempts)
        all_scores[concept] = round(score, 4)
        if score > best_score:
            best_concept = concept
            best_score = score
    if not best_concept:
        best_concept = "通用知识点"
        best_score = 0.0
    best_stats = state["concepts"].get(best_concept, {})
    source_chunks = []
    if chunk_lookup:
        for chunk_id in best_stats.get("source_chunk_ids", [])[-3:]:
            if chunk_id in chunk_lookup:
                source_chunks.append(chunk_lookup[chunk_id])
    return {
        "concept": best_concept,
        "score": round(best_score, 4) if best_score != float('inf') else float('inf'),
        "reason": _generate_reason(best_stats, best_score),
        "attempts": best_stats.get("attempts", 0),
        "wrong": best_stats.get("wrong", 0),
        "source_chunks": source_chunks,
        "all_scores": all_scores,
    }


def get_all_status() -> dict:
    state = load_state()
    concepts = state.get("concepts", {})
    total_attempts = sum(s["attempts"] for s in concepts.values())
    total_wrong = sum(s["wrong"] for s in concepts.values())
    return {
        "total_concepts": len(concepts),
        "total_attempts": total_attempts,
        "total_wrong": total_wrong,
        "overall_wrong_rate": round(total_wrong / total_attempts, 2) if total_attempts else 0,
        "concepts": {
            concept: {
                "attempts": s["attempts"],
                "wrong": s["wrong"],
                "wrong_rate": round(s["wrong"] / s["attempts"], 2) if s["attempts"] else 0,
                "last_seen": s.get("last_seen", "从未")
            }
            for concept, s in concepts.items()
        }
    }


def render_bandit_ui(chunk_lookup: Optional[Dict[str, Chunk]] = None):
    import streamlit as st
    st.subheader("智能复习推荐")
    summary = get_all_status()
    if summary["total_attempts"] > 0:
        cols = st.columns(4)
        cols[0].metric("知识点总数", summary["total_concepts"])
        cols[1].metric("总答题数", summary["total_attempts"])
        cols[2].metric("总错误数", summary["total_wrong"])
        cols[3].metric("整体错误率", f"{summary['overall_wrong_rate']*100:.0f}%")
    rec = recommend_concept(chunk_lookup=chunk_lookup)
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            score_display = "∞" if rec["score"] == float('inf') else f"{rec['score']:.3f}"
            st.metric(
                label="优先推荐",
                value=rec["concept"],
                delta=f"UCB: {score_display}"
            )
        with col2:
            st.write(f"**{rec['reason']}**")
            attempts = rec["attempts"]
            wrong = rec["wrong"]
            if attempts > 0:
                correct_rate = (attempts - wrong) / attempts
                st.progress(
                    correct_rate,
                    text=f"正确率 {correct_rate*100:.0f}% | 练习 {attempts} 次 | 错 {wrong} 次"
                )
            else:
                st.info("待首次练习")
            if rec["source_chunks"]:
                with st.expander("推荐复习资料"):
                    for chunk in rec["source_chunks"]:
                        st.caption(f"**{chunk.file_name}** 第{chunk.page}页")
                        st.text(chunk.text[:150] + "...")
        st.divider()
    with st.expander("查看所有知识点状态"):
        import pandas as pd
        data = []
        for concept, s in summary["concepts"].items():
            data.append({
                "知识点": concept,
                "练习次数": s["attempts"],
                "错误次数": s["wrong"],
                "错误率": f"{s['wrong_rate']*100:.0f}%" if s["attempts"] > 0 else "-",
                "最后练习": s["last_seen"][:10] if s["last_seen"] != "从未" else "从未"
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
