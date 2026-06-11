import re
from collections import Counter

from src.schemas import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def keyword_score(query: str, chunk: Chunk) -> float:
    query_terms = Counter(tokenize(query))
    chunk_terms = Counter(tokenize(chunk.text))
    if not query_terms:
        return 0.0
    overlap = sum(min(count, chunk_terms[term]) for term, count in query_terms.items())
    return overlap / sum(query_terms.values())

