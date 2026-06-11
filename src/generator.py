from src.llm_client import generate_text
from src.schemas import Chunk, RankedChunk


def answer_question(query: str, evidence: list[RankedChunk]) -> dict:
    context = "\n".join(f"[{item.chunk.chunk_id}] {item.chunk.text}" for item in evidence)
    answer = generate_text(f"Question: {query}\nEvidence:\n{context}")
    citations = [
        {
            "chunk_id": item.chunk.chunk_id,
            "file_name": item.chunk.file_name,
            "page": item.chunk.page,
            "score": item.ranker_score,
        }
        for item in evidence
    ]
    return {"answer": answer, "citations": citations}


def summarize_document(chunks: list[Chunk]) -> str:
    if not chunks:
        return "当前没有可总结的文档内容。"
    concepts = sorted({concept for chunk in chunks for concept in chunk.concepts})
    return (
        f"当前文档共切分为 {len(chunks)} 个片段。检测到的主要知识点包括："
        f"{'、'.join(concepts)}。后续可以在 real 模式下将该总结模块替换为真实 LLM 调用。"
    )
