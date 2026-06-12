from collections import Counter

from src.llm_client import generate_text
from src.schemas import Chunk, RankedChunk


def answer_question(query: str, evidence: list[RankedChunk]) -> dict:
    context = build_evidence_context(evidence)
    prompt = (
        "请只根据下面的课程资料证据回答问题。"
        "如果证据不足，请说明无法基于当前资料可靠回答。\n\n"
        f"问题：{query}\n\n证据：\n{context}"
    )
    answer = generate_text(
        prompt,
        system_prompt="你是 CourseMind 的中文课程资料问答助手，回答必须基于证据并保留引用。",
    )
    citations = build_citations(evidence)
    return {
        "answer": answer,
        "citations": citations,
        "evidence_count": len(evidence),
    }


def summarize_document(chunks: list[Chunk]) -> str:
    if not chunks:
        return "当前没有可总结的文档内容。"

    concept_counts = Counter(concept for chunk in chunks for concept in chunk.concepts)
    files = sorted({chunk.file_name for chunk in chunks})
    pages = sorted({chunk.page for chunk in chunks})
    top_concepts = [concept for concept, _ in concept_counts.most_common(8)]

    prompt = (
        "请总结当前中文课程资料，突出项目要求、技术点、评分标准和成员贡献要求。\n"
        f"文件：{'、'.join(files)}\n"
        f"页码范围：{min(pages)}-{max(pages)}\n"
        f"片段数量：{len(chunks)}\n"
        f"知识点：{'、'.join(top_concepts)}"
    )
    generated = generate_text(prompt)
    return (
        f"{generated}\n\n"
        f"资料统计：共 {len(chunks)} 个 chunk，来源文件 {len(files)} 个，页码范围 {min(pages)}-{max(pages)}。"
        f"主要知识点：{'、'.join(top_concepts) if top_concepts else '暂无'}。"
    )


def build_evidence_context(evidence: list[RankedChunk], max_chars_per_chunk: int = 500) -> str:
    if not evidence:
        return "无可用证据。"
    return "\n".join(
        (
            f"[{item.chunk.chunk_id}] 文件={item.chunk.file_name} 页码={item.chunk.page} "
            f"ranker_score={item.ranker_score:.4f}\n"
            f"{item.chunk.text[:max_chars_per_chunk]}"
        )
        for item in evidence
    )


def build_citations(evidence: list[RankedChunk]) -> list[dict]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "file_name": item.chunk.file_name,
            "page": item.chunk.page,
            "score": item.ranker_score,
            "dense_score": item.dense_score,
            "bm25_score": item.bm25_score,
            "graph_score": item.graph_score,
            "text_preview": item.chunk.text[:120],
        }
        for item in evidence
    ]
