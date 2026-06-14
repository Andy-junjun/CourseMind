from __future__ import annotations

import re
import hashlib
from collections import Counter

from src.chunker import DEFAULT_CONCEPT
from src.schemas import Chunk, QuizItem


GENERIC_DISTRACTORS = [
    "它主要用于控制页面展示样式，不涉及模型原理。",
    "它只表示数据库中的文件编号，不对应课程知识。",
    "它是随机生成的标签，不能解释模型行为。",
]
ENGLISH_CONCEPT_ALLOWLIST = {
    "CNN",
    "RNN",
    "LSTM",
    "GRU",
    "GAN",
    "DQN",
    "BERT",
    "Transformer",
    "Attention",
    "Self-Attention",
}
SYSTEM_OR_NOISY_CONCEPTS = {
    "RAG",
    "FAISS",
    "GraphRAG",
    "MiniRanker",
    "Bandit",
    "Streamlit",
}
NOISY_CONCEPT_PREFIXES = ("多个", "一种", "一个")
NOISY_CONCEPT_SUFFIXES = ("经常", "主要", "可以", "用于")
EXPLANATION_MARKERS = (
    "是",
    "用于",
    "通过",
    "能够",
    "具有",
    "包括",
    "由",
    "使",
    "表示",
    "计算",
    "减少",
    "增强",
    "引入",
    "学习",
    "更新",
    "提取",
    "利用",
)


def generate_quiz(
    chunks: list[Chunk],
    num_questions: int = 3,
    target_concept: str | None = None,
) -> list[QuizItem]:
    """Generate concept-understanding multiple-choice questions from chunks."""
    if num_questions <= 0:
        return []

    candidates = quiz_candidates(chunks, target_concept=target_concept)
    items: list[QuizItem] = []
    used_concepts: set[str] = set()
    used_chunks: set[str] = set()

    for chunk, concept, statement in candidates:
        if (not target_concept and concept in used_concepts) or chunk.chunk_id in used_chunks:
            continue
        item = build_quiz_item(
            chunk=chunk,
            concept=concept,
            statement=statement,
            chunks=chunks,
            prefer_cloze=len(items) % 2 == 0,
        )
        if item is None:
            continue
        items.append(item)
        used_concepts.add(concept)
        used_chunks.add(chunk.chunk_id)
        if len(items) >= num_questions:
            break

    return items


def quizzable_concepts(chunks: list[Chunk]) -> list[str]:
    """Return concepts that can actually produce a quiz question from these chunks.

    This is the source of truth for "what can we quiz on". The Bandit recommender
    restricts its recommendations to this set so the recommended concept always
    matches the concept the next question is about.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for _chunk, concept, _statement in quiz_candidates(chunks):
        if concept not in seen:
            seen.add(concept)
            ordered.append(concept)
    return ordered


def build_quiz_item(
    chunk: Chunk,
    concept: str,
    statement: str,
    chunks: list[Chunk],
    *,
    prefer_cloze: bool,
) -> QuizItem | None:
    if prefer_cloze:
        cloze = build_cloze_quiz(chunk, concept, statement, chunks)
        if cloze is not None:
            return cloze
    return build_definition_quiz(chunk, concept, statement, chunks)


def build_definition_quiz(
    chunk: Chunk,
    concept: str,
    statement: str,
    chunks: list[Chunk],
) -> QuizItem | None:
    options = build_definition_options(answer=statement, chunks=chunks, concept=concept)
    if len(options) < 2:
        return None
    return QuizItem(
        question=f"关于“{concept}”，以下哪一项最符合知识库资料中的解释？",
        options=stable_shuffled_options(options, answer=statement, seed=f"definition:{chunk.chunk_id}:{concept}"),
        answer=statement,
        explanation=(
            f"正确选项概括了资料中对“{concept}”的说明。"
            f"来源：{chunk.file_name} 第 {chunk.page} 页，chunk_id={chunk.chunk_id}。"
        ),
        concept=concept,
        source_chunk_id=chunk.chunk_id,
    )


def build_cloze_quiz(
    chunk: Chunk,
    concept: str,
    statement: str,
    chunks: list[Chunk],
) -> QuizItem | None:
    keyword = choose_mask_keyword(statement, concept, chunk)
    if not keyword:
        return None
    masked_statement = mask_keyword(statement, keyword)
    if masked_statement == statement:
        return None

    options = build_keyword_options(answer=keyword, chunks=chunks)
    if len(options) < 2:
        return None
    return QuizItem(
        question=masked_statement,
        options=stable_shuffled_options(options, answer=keyword, seed=f"cloze:{chunk.chunk_id}:{keyword}"),
        answer=keyword,
        explanation=(
            f"被遮住的关键词是“{keyword}”。该句来自知识库中关于“{concept}”的说明。"
            f"来源：{chunk.file_name} 第 {chunk.page} 页，chunk_id={chunk.chunk_id}。"
        ),
        concept=concept,
        source_chunk_id=chunk.chunk_id,
    )


def quiz_candidates(
    chunks: list[Chunk],
    target_concept: str | None = None,
) -> list[tuple[Chunk, str, str]]:
    concept_counts = Counter(
        concept
        for chunk in chunks
        for concept in chunk.concepts
        if is_teachable_concept(concept)
    )
    candidates: list[tuple[float, Chunk, str, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        for concept_index, concept in enumerate(chunk.concepts):
            if not is_teachable_concept(concept):
                continue
            if target_concept and concept != target_concept:
                continue
            statement = concept_statement(chunk, concept)
            if not statement:
                continue
            frequency = concept_counts.get(concept, 1)
            specificity = min(len(concept), 12) / 12
            chinese_bonus = 1.0 if contains_chinese(concept) else 0.2
            score = chinese_bonus + specificity + 1 / frequency - 0.001 * chunk_index - 0.01 * concept_index
            candidates.append((score, chunk, concept, statement))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, concept, statement) for _, chunk, concept, statement in candidates]


def concept_statement(
    chunk: Chunk,
    concept: str,
    *,
    allow_prefix_fallback: bool = True,
) -> str:
    for sentence in split_statement_candidates(chunk.text):
        if concept in sentence and is_good_statement(sentence, concept):
            return normalize_statement(sentence)

    if not allow_prefix_fallback:
        return ""

    for sentence in split_statement_candidates(chunk.text):
        if is_good_statement(sentence, concept):
            return normalize_statement(f"{concept}：{sentence}")

    return ""


def build_definition_options(answer: str, chunks: list[Chunk], concept: str) -> list[str]:
    options = [answer]
    for chunk in chunks:
        if concept in chunk.concepts:
            continue
        for other_concept in chunk.concepts:
            if not is_teachable_concept(other_concept) or other_concept == concept:
                continue
            statement = concept_statement(
                chunk,
                other_concept,
                allow_prefix_fallback=False,
            )
            if statement and statement not in options:
                options.append(statement)
                break
        if len(options) >= 4:
            break

    for distractor in GENERIC_DISTRACTORS:
        if len(options) >= 4:
            break
        if distractor not in options:
            options.append(distractor)

    return options[:4]


def build_keyword_options(answer: str, chunks: list[Chunk]) -> list[str]:
    options = [answer]
    for chunk in chunks:
        for concept in chunk.concepts:
            if concept == answer or not is_teachable_concept(concept):
                continue
            if concept in options:
                continue
            options.append(concept)
            if len(options) >= 4:
                return options
    return options


def choose_mask_keyword(statement: str, concept: str, chunk: Chunk) -> str:
    if (
        concept in statement
        and is_teachable_concept(concept)
        and is_good_mask_keyword(statement, concept)
    ):
        return concept
    candidates = [
        candidate
        for candidate in chunk.concepts
        if candidate in statement
        and is_teachable_concept(candidate)
        and is_good_mask_keyword(statement, candidate)
    ]
    if not candidates:
        return ""
    return max(candidates, key=len)


def is_good_mask_keyword(statement: str, keyword: str) -> bool:
    index = statement.find(keyword)
    if index < 0:
        return False
    heading_end_positions = [pos for pos in (statement.find(":"), statement.find("：")) if pos >= 0]
    if heading_end_positions:
        heading_end = min(heading_end_positions)
        if heading_end <= 30 and index < heading_end:
            return False
    return True


def mask_keyword(statement: str, keyword: str) -> str:
    return statement.replace(keyword, "____", 1)


def stable_shuffled_options(options: list[str], answer: str, seed: str) -> list[str]:
    unique_options = list(dict.fromkeys(options))
    shuffled = sorted(
        unique_options,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    if len(shuffled) > 1 and shuffled[0] == answer:
        digest = hashlib.sha256(f"{seed}:answer-position".encode("utf-8")).digest()
        target_index = 1 + digest[0] % (len(shuffled) - 1)
        shuffled.pop(0)
        shuffled.insert(target_index, answer)
    return shuffled


def split_statement_candidates(text: str) -> list[str]:
    text = remove_pdf_artifacts(text)
    parts = re.split(r"[\n\r\f]+|[•]+|(?<=[。！？；;.!?])|\s+-\s+", text)
    statements = []
    for part in parts:
        statement = normalize_statement(part)
        if statement:
            statements.append(statement)
    return statements


def is_good_statement(sentence: str, concept: str) -> bool:
    compact = sentence.strip()
    if len(compact) < max(12, len(concept) + 4):
        return False
    if len(compact) > 140:
        return False
    if compact.endswith(("：", ":")):
        return False
    if "�" in compact:
        return False
    if not contains_chinese(compact):
        return False
    if digit_ratio(compact) > 0.25:
        return False
    if not any(marker in compact for marker in EXPLANATION_MARKERS):
        return False
    return True


def normalize_statement(text: str) -> str:
    text = remove_pdf_artifacts(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" #*_`-•")
    return text[:137] + "..." if len(text) > 140 else text


def is_teachable_concept(concept: str) -> bool:
    concept = concept.strip()
    if not concept or concept == DEFAULT_CONCEPT:
        return False
    if concept in SYSTEM_OR_NOISY_CONCEPTS:
        return False
    if concept.startswith(NOISY_CONCEPT_PREFIXES):
        return False
    if concept.endswith(NOISY_CONCEPT_SUFFIXES):
        return False
    if "_" in concept:
        return False
    if len(concept) < 2 or len(concept) > 24:
        return False
    if not contains_chinese(concept) and concept not in ENGLISH_CONCEPT_ALLOWLIST:
        return False
    return True


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def digit_ratio(text: str) -> float:
    compact = [char for char in text if not char.isspace()]
    if not compact:
        return 0.0
    return sum(1 for char in compact if char.isdigit()) / len(compact)


def remove_pdf_artifacts(text: str) -> str:
    text = re.sub(r"《[^》]{2,40}》\s*\d*\s*", "", text)
    text = re.sub(r"\b\d{1,3}\b", "", text)
    text = text.replace("\u200b", "")
    return text
