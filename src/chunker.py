from src.schemas import Chunk, DocumentPage


CONCEPT_KEYWORDS = [
    "Transformer",
    "注意力机制",
    "文本嵌入",
    "RAG",
    "检索增强生成",
    "向量检索",
    "FAISS",
    "BM25",
    "GraphRAG",
    "知识图谱",
    "图扩展",
    "MiniRanker",
    "重排序",
    "Bandit",
    "多臂老虎机",
    "强化学习",
    "Embedding",
    "LLM",
    "大语言模型",
    "自动出题",
    "原文引用",
    "拒答机制",
]


def infer_concepts(text: str) -> list[str]:
    lower = text.lower()
    concepts = [keyword for keyword in CONCEPT_KEYWORDS if keyword.lower() in lower]
    return concepts or ["通用知识点"]


def chunk_pages(
    pages: list[DocumentPage], chunk_size: int = 400, overlap: int = 80
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        start = 0
        index = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            part = text[start:end]
            chunk_id = f"{safe_id(page.file_name)}_p{page.page:03d}_c{index:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_name=page.file_name,
                    page=page.page,
                    text=part,
                    chapter=None,
                    concepts=infer_concepts(part),
                )
            )
            if end == len(text):
                break
            start = end - overlap
            index += 1
    return chunks


def safe_id(file_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in file_name).strip("_")
