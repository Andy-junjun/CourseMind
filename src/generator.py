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
        return "No document content is available."
    concepts = sorted({concept for chunk in chunks for concept in chunk.concepts})
    return (
        f"This document has {len(chunks)} chunks. Main detected concepts: "
        f"{', '.join(concepts)}. The summary module can be replaced by a real LLM call later."
    )

