from __future__ import annotations

import os
import re
from collections import Counter
from functools import lru_cache

from src.bm25_store import tokenize
from src.config import get_mode


DEFAULT_TOP_K = 6
DEFAULT_CANDIDATE_COUNT = 32
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
WORD_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_+#.-]+")
FUNCTION_WORDS = {
    "负责",
    "使用",
    "通过",
    "进行",
    "用于",
    "能够",
    "可以",
    "以及",
    "一个",
    "一种",
    "引入",
    "推荐",
    "计算",
    "实现",
    "根据",
    "包含",
    "表示",
    "作为",
    "生成",
}


def extract_keywords(text: str, top_k: int | None = None) -> list[str]:
    top_k = top_k or int(os.getenv("KEYWORD_TOP_K", str(DEFAULT_TOP_K)))
    if top_k <= 0:
        return []

    candidates = candidate_keywords(text, max_candidates=DEFAULT_CANDIDATE_COUNT)
    if not candidates:
        return []

    provider = get_keyword_provider()
    if provider in {"embedding", "keybert", "semantic"} and can_use_embedding_keywords():
        ranked = rank_keywords_by_embedding(text, candidates)
    else:
        ranked = rank_keywords_by_statistics(text, candidates)

    return [keyword for keyword, _ in ranked[:top_k]]


def get_keyword_provider() -> str:
    provider = os.getenv("KEYWORD_PROVIDER", "").strip().lower()
    if provider:
        return provider
    return "embedding" if get_mode() in {"real", "hybrid"} else "statistical"


def can_use_embedding_keywords() -> bool:
    provider = os.getenv("EMBEDDING_PROVIDER", "lite").strip().lower()
    return get_mode() in {"real", "hybrid"} and provider in {
        "sentence_transformers",
        "sentence-transformer",
        "st",
        "fastembed",
    }


def candidate_keywords(text: str, max_candidates: int = DEFAULT_CANDIDATE_COUNT) -> list[str]:
    weighted: Counter[str] = Counter()
    for keyword in jieba_keywords(text, max_candidates=max_candidates):
        weighted[normalize_keyword(keyword)] += 3
    for keyword in ordered_token_keywords(text):
        weighted[normalize_keyword(keyword)] += 2
    for token, count in Counter(tokenize(text)).items():
        weighted[normalize_keyword(token)] += count
    for phrase in latin_terms(text):
        weighted[normalize_keyword(phrase)] += 2

    candidates = [
        keyword
        for keyword, _ in weighted.most_common(max_candidates * 2)
        if is_valid_keyword(keyword) and keyword_appears_in_text(keyword, text)
    ]
    return unique_preserve_order(candidates)[:max_candidates]


def jieba_keywords(text: str, max_candidates: int) -> list[str]:
    try:
        import jieba.analyse  # type: ignore

        return [
            str(keyword)
            for keyword in jieba.analyse.extract_tags(text, topK=max_candidates, withWeight=False)
        ]
    except Exception:
        return []


def ordered_token_keywords(text: str) -> list[str]:
    terms = ordered_terms(text)
    keywords: list[str] = list(terms)
    for size, weight in ((2, 2),):
        for index in range(max(0, len(terms) - size + 1)):
            phrase_terms = terms[index : index + size]
            if not all(CJK_PATTERN.search(term) for term in phrase_terms):
                continue
            if any(term in FUNCTION_WORDS for term in phrase_terms):
                continue
            phrase = "".join(phrase_terms)
            keywords.extend([phrase] * weight)
    return keywords


def ordered_terms(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        return [
            normalize_keyword(term)
            for term in jieba.cut(text)
            if is_valid_keyword(normalize_keyword(term))
        ]
    except Exception:
        return [normalize_keyword(match.group(0)) for match in WORD_PATTERN.finditer(text)]


def latin_terms(text: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9_+#.-]{1,31}", text)
    ]


def rank_keywords_by_embedding(text: str, candidates: list[str]) -> list[tuple[str, float]]:
    vectors = embeddings_for([text, *candidates])
    text_vector = vectors[0] if vectors else ()
    if not text_vector:
        return rank_keywords_by_statistics(text, candidates)

    ranked = []
    for candidate, candidate_vector in zip(candidates, vectors[1:]):
        if not candidate_vector:
            continue
        ranked.append((candidate, cosine(text_vector, candidate_vector)))
    return sorted(ranked, key=lambda item: (item[1], len(item[0])), reverse=True)


def embeddings_for(texts: list[str]) -> list[tuple[float, ...]]:
    provider = os.getenv("EMBEDDING_PROVIDER", "lite").strip().lower()
    if provider in {"sentence_transformers", "sentence-transformer", "st"}:
        from src.embedder import get_embedding_model

        matrix = get_embedding_model().encode(texts, normalize_embeddings=True)
        return [tuple(float(value) for value in vector.tolist()) for vector in matrix]
    return [embedding_for(text) for text in texts]


def rank_keywords_by_statistics(text: str, candidates: list[str]) -> list[tuple[str, float]]:
    token_counts = Counter(normalize_keyword(token) for token in tokenize(text))
    ranked = []
    for candidate in candidates:
        count = token_counts.get(candidate, 0)
        occurrence = text.count(candidate)
        length_bonus = min(len(candidate), 8) / 8.0
        cjk_bonus = 0.2 if CJK_PATTERN.search(candidate) else 0.0
        latin_bonus = 2.0 if re.search(r"[A-Za-z]", candidate) else 0.0
        phrase_bonus = 1.1 if is_cjk_phrase(candidate) else 0.0
        ranked.append(
            (
                candidate,
                count + occurrence + length_bonus + cjk_bonus + latin_bonus + phrase_bonus,
            )
        )
    return sorted(ranked, key=lambda item: (item[1], len(item[0])), reverse=True)


@lru_cache(maxsize=8192)
def embedding_for(text: str) -> tuple[float, ...]:
    from src.embedder import embed_text

    return tuple(embed_text(text))


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b or 1.0)


def normalize_keyword(keyword: str) -> str:
    keyword = keyword.strip().strip(".,;:!?，。；：！？、（）()[]【】<>《》\"'`")
    return re.sub(r"\s+", " ", keyword)


def is_valid_keyword(keyword: str) -> bool:
    if not keyword:
        return False
    if len(keyword) < 2 or len(keyword) > 32:
        return False
    if not WORD_PATTERN.fullmatch(keyword):
        return False
    if keyword.isdigit():
        return False
    if keyword in FUNCTION_WORDS:
        return False
    if CJK_PATTERN.search(keyword) and len(keyword) == 1:
        return False
    return True


def keyword_appears_in_text(keyword: str, text: str) -> bool:
    if CJK_PATTERN.search(keyword):
        return keyword in text
    return keyword.lower() in text.lower()


def is_cjk_phrase(keyword: str) -> bool:
    return bool(CJK_PATTERN.search(keyword) and len(keyword) >= 4)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
