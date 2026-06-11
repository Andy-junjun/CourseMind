import re
from collections import Counter

from src.schemas import Chunk


CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = LATIN_PATTERN.findall(text)

    if CJK_PATTERN.search(text):
        try:
            import jieba  # type: ignore

            tokens.extend(term.strip() for term in jieba.cut(text) if term.strip())
        except ImportError:
            chinese_chars = CJK_PATTERN.findall(text)
            tokens.extend(chinese_chars)
            tokens.extend(
                "".join(chinese_chars[i : i + 2])
                for i in range(max(0, len(chinese_chars) - 1))
            )

    return tokens


def keyword_score(query: str, chunk: Chunk) -> float:
    query_terms = Counter(tokenize(query))
    chunk_terms = Counter(tokenize(chunk.text))
    if not query_terms:
        return 0.0
    overlap = sum(min(count, chunk_terms[term]) for term, count in query_terms.items())
    return overlap / sum(query_terms.values())
