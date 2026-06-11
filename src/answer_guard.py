from src.schemas import RankedChunk


def should_refuse(query: str, evidence: list[RankedChunk], threshold: float = 0.05) -> bool:
    if not evidence:
        return True
    return max(item.ranker_score for item in evidence) < threshold

