import re

from src.schemas import Chunk, DocumentPage


DEFAULT_CONCEPT = "通用知识点"

CONCEPT_ALIASES: dict[str, list[str]] = {
    "深度学习": ["深度学习", "神经网络", "CNN", "RNN", "LSTM", "GRU", "DRL"],
    "Transformer": ["Transformer", "注意力机制", "self-attention", "attention"],
    "文本嵌入": ["文本嵌入", "Embedding", "embedding", "向量表示"],
    "RAG": ["RAG", "检索增强生成", "Retrieval Augmented Generation"],
    "向量检索": ["向量检索", "语义检索", "dense retrieval", "FAISS"],
    "BM25": ["BM25", "关键词检索", "稀疏检索"],
    "GraphRAG": ["GraphRAG", "GraphRAG-lite", "知识图谱", "图扩展", "图检索"],
    "MiniRanker": ["MiniRanker", "重排序", "rerank", "reranker", "神经重排序"],
    "Bandit": ["Bandit", "多臂老虎机", "UCB", "epsilon-greedy", "ε-greedy"],
    "强化学习": ["强化学习", "reinforcement learning", "RL", "DRL"],
    "大语言模型": ["大语言模型", "LLM", "语言模型", "Ollama"],
    "自动出题": ["自动出题", "生成题目", "选择题", "Quiz"],
    "原文引用": ["原文引用", "引用来源", "citation", "citations"],
    "拒答机制": ["拒答机制", "拒答", "资料外问题", "证据不足"],
    "评分标准": ["评分标准", "评分项", "满分", "占比"],
    "成员贡献": ["成员贡献", "贡献清单", "分工", "团队分工"],
}

HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇][：:、.\s]?.*"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*\S+.*"),
    re.compile(r"^\d+(?:\.\d+)*[、.．\s]\s*\S+.*"),
    re.compile(r"^#{1,6}\s+\S+.*"),
]

SENTENCE_PATTERN = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")


def infer_concepts(text: str) -> list[str]:
    compact_text = re.sub(r"\s+", "", text).lower()
    lower_text = text.lower()
    concepts: list[str] = []

    for concept, aliases in CONCEPT_ALIASES.items():
        if any(_contains_alias(lower_text, compact_text, alias) for alias in aliases):
            concepts.append(concept)

    return concepts or [DEFAULT_CONCEPT]


def chunk_pages(
    pages: list[DocumentPage], chunk_size: int = 400, overlap: int = 80
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        sections = split_page_into_sections(page.text)
        chunk_index = 1
        for chapter, section_text in sections:
            for part in split_text_into_chunks(section_text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{safe_id(page.file_name)}_p{page.page:03d}_c{chunk_index:03d}",
                        file_name=page.file_name,
                        page=page.page,
                        text=part,
                        chapter=chapter,
                        concepts=infer_concepts(part),
                    )
                )
                chunk_index += 1
    return chunks


def split_page_into_sections(text: str) -> list[tuple[str | None, str]]:
    lines = [normalize_spaces(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    sections: list[tuple[str | None, list[str]]] = []
    current_chapter: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if is_heading(line):
            if current_lines:
                sections.append((current_chapter, current_lines))
            current_chapter = cleanup_heading(line)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_chapter, current_lines))

    return [(chapter, normalize_spaces(" ".join(section_lines))) for chapter, section_lines in sections]


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_text(sentence, chunk_size, overlap))
            continue

        candidate = f"{current}{sentence}" if not current else f"{current} {sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = _overlap_prefix(current, overlap, sentence)

    if current:
        chunks.append(current)

    return chunks


def split_sentences(text: str) -> list[str]:
    sentences = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    parts = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [part for part in parts if part]


def is_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def cleanup_heading(line: str) -> str:
    return line.lstrip("#").strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_id(file_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in file_name).strip("_")


def _contains_alias(lower_text: str, compact_text: str, alias: str) -> bool:
    lower_alias = alias.lower()
    compact_alias = re.sub(r"\s+", "", lower_alias)
    return lower_alias in lower_text or compact_alias in compact_text


def _overlap_prefix(previous: str, overlap: int, sentence: str) -> str:
    if overlap == 0:
        return sentence
    prefix = previous[-overlap:].strip()
    return f"{prefix} {sentence}".strip() if prefix else sentence

