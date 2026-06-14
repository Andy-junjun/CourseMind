import re

from src.keyword_extractor import extract_keywords
from src.schemas import Chunk, DocumentPage


DEFAULT_CONCEPT = "通用知识点"

HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇][：:、\s]?.*"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*\S+.*"),
    re.compile(r"^\d+(?:\.\d+)*[、.．\s]\s*\S+.*"),
    re.compile(r"^#{1,6}\s+\S+.*"),
]

SENTENCE_PATTERN = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")


def infer_concepts(text: str) -> list[str]:
    return extract_keywords(text) or [DEFAULT_CONCEPT]


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

    return [
        (chapter, normalize_spaces(" ".join(section_lines)))
        for chapter, section_lines in sections
    ]


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


def _overlap_prefix(previous: str, overlap: int, sentence: str) -> str:
    if overlap == 0:
        return sentence
    prefix = previous[-overlap:].strip()
    return f"{prefix} {sentence}".strip() if prefix else sentence
