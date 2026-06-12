from src.schemas import RankedChunk


OUT_OF_SCOPE_HINTS = {
    "yolov10",
    "yolo",
    "股票",
    "天气",
    "世界杯",
    "电影",
}


def should_refuse(
    query: str,
    evidence: list[RankedChunk],
    threshold: float = 0.52,
    min_keyword_score: float = 0.05,
) -> bool:
    if not evidence:
        return True

    query_lower = query.lower()
    if any(hint in query_lower for hint in OUT_OF_SCOPE_HINTS):
        return True

    best = max(evidence, key=lambda item: item.ranker_score)
    if best.ranker_score >= threshold:
        return False

    if best.bm25_score >= min_keyword_score and best.ranker_score >= threshold - 0.08:
        return False

    return True


def refusal_message() -> str:
    return "当前知识库未检索到足够相关的课程资料，无法可靠回答该问题。"
