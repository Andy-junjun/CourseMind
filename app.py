from pathlib import Path

import streamlit as st

from src.answer_guard import refusal_message, should_refuse
from src.bandit_recommender import recommend_concept, update_feedback
from src.chunker import chunk_pages
from src.config import get_mode
from src.document_loader import load_pdf
from src.generator import answer_question, summarize_document
from src.graph_store import expand_with_graph, explain_graph_expansion
from src.miniranker import rerank
from src.retriever import retrieve
from src.schemas import DocumentPage
from src.study_tools import generate_quiz
from src.vector_store import build_index, load_vector_store, vector_store_exists


st.set_page_config(page_title="CourseMind", layout="wide")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def default_pages() -> list[DocumentPage]:
    return [
        DocumentPage(
            file_name="内置中文课程示例",
            page=1,
            text=(
                "CourseMind 是一个面向中文课程资料的智能学习助手。"
                "系统支持 PDF 解析、中文 Chunk 切分、RAG 检索、原文引用、"
                "文档总结、自动出题和复习推荐。检索阶段结合向量检索、"
                "BM25 关键词检索和 GraphRAG-lite 图扩展。MiniRanker 会综合 "
                "dense_score、bm25_score、graph_score、同页奖励、同章节奖励和文本特征，"
                "对候选片段重新排序。Bandit 模块根据学生答题反馈推荐薄弱知识点。"
            ),
        )
    ]


@st.cache_resource(show_spinner=False)
def build_knowledge_base(file_path: str | None):
    if file_path:
        pages = load_pdf(file_path)
        chunks = chunk_pages(pages)
        index = build_index(chunks)
        return pages, chunks, index, "上传文件临时索引"

    if vector_store_exists():
        chunks, index = load_vector_store()
        pages = pages_from_chunks(chunks)
        return pages, chunks, index, "data/indexes/faiss.index"

    pages = default_pages()
    chunks = chunk_pages(pages)
    index = build_index(chunks)
    return pages, chunks, index, "内置示例"


def pages_from_chunks(chunks):
    page_text: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        page_text.setdefault((chunk.file_name, chunk.page), []).append(chunk.text)
    return [
        DocumentPage(file_name=file_name, page=page, text="\n".join(parts))
        for (file_name, page), parts in sorted(page_text.items())
    ]


def graph_expansion_rows(expanded, seeds, chunks, seed_ids):
    rows = []
    for item in expanded:
        if item.chunk.chunk_id in seed_ids:
            continue
        relations = explain_graph_expansion(item.chunk, seeds, chunks, hops=1)
        for relation in relations:
            rows.append(
                {
                    "扩展chunk": relation.candidate_chunk_id,
                    "种子chunk": relation.seed_chunk_id,
                    "关系": relation.relation,
                    "分数": round(relation.score, 3),
                    "原因": relation.reason,
                    "文件": item.chunk.file_name,
                    "页码": item.chunk.page,
                }
            )
    return rows


def graph_reason(chunk_id, rows):
    reasons = [row["原因"] for row in rows if row["扩展chunk"] == chunk_id]
    return "；".join(reasons[:3])


st.sidebar.title("CourseMind")
st.sidebar.caption(f"运行模式：{get_mode()}")
uploaded = st.sidebar.file_uploader("上传课程 PDF 或文本", type=["pdf", "txt", "md"])

file_path = None
if uploaded:
    target = RAW_DIR / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    file_path = str(target)

pages, chunks, vector_index, data_source = build_knowledge_base(file_path)
st.sidebar.metric("页数", len(pages))
st.sidebar.metric("Chunks", len(chunks))
st.sidebar.caption(f"数据来源：{data_source}")

tab_qa, tab_summary, tab_quiz, tab_review, tab_status = st.tabs(
    ["问答", "总结", "出题", "复习", "状态"]
)

with tab_qa:
    query = st.text_input("问题", "神经网络中的激活函数有什么作用？")
    top_k = st.slider("Top-K 证据", 1, 10, 5)
    if st.button("提问", type="primary"):
        retrieved = retrieve(query, chunks, top_k=top_k, index=vector_index)
        expanded = expand_with_graph(retrieved, chunks, hops=1)
        ranked = rerank(query, expanded, top_k=top_k)
        seed_ids = {item.chunk.chunk_id for item in retrieved}
        graph_explanations = graph_expansion_rows(expanded, retrieved, chunks, seed_ids)

        if should_refuse(query, ranked):
            st.warning(refusal_message())
        else:
            result = answer_question(query, ranked)
            st.subheader("回答")
            st.write(result["answer"])
            st.subheader("引用来源")
            st.dataframe(result["citations"], use_container_width=True)

        st.subheader("检索与重排序结果")
        st.dataframe(
            [
                {
                    "rank": i + 1,
                    "来源": "原始检索" if item.chunk.chunk_id in seed_ids else "GraphRAG扩展",
                    "chunk_id": item.chunk.chunk_id,
                    "文件": item.chunk.file_name,
                    "页码": item.chunk.page,
                    "dense": round(item.dense_score, 3),
                    "bm25": round(item.bm25_score, 3),
                    "graph": round(item.graph_score, 3),
                    "ranker": round(item.ranker_score, 3),
                    "GraphRAG原因": graph_reason(item.chunk.chunk_id, graph_explanations),
                    "文本": item.chunk.text[:120],
                }
                for i, item in enumerate(ranked)
            ],
            use_container_width=True,
        )

        if graph_explanations:
            st.subheader("GraphRAG 扩展解释")
            st.dataframe(graph_explanations, use_container_width=True)

with tab_summary:
    if st.button("生成总结"):
        st.write(summarize_document(chunks))

with tab_quiz:
    num_questions = st.slider("题目数量", 1, 5, 3)
    quiz_items = generate_quiz(chunks, num_questions=num_questions)
    for idx, item in enumerate(quiz_items, start=1):
        st.markdown(f"**Q{idx}. {item.question}**")
        choice = st.radio("请选择", item.options, key=f"quiz_{idx}")
        correct = choice == item.answer
        col1, col2 = st.columns(2)
        with col1:
            if st.button("提交", key=f"submit_{idx}"):
                update_feedback(item.concept, correct)
                st.success("回答正确" if correct else f"回答错误。正确答案：{item.answer}")
        with col2:
            st.caption(f"知识点：{item.concept} | 来源：{item.source_chunk_id}")
        st.write(item.explanation)

with tab_review:
    rec = recommend_concept()
    st.metric("推荐复习知识点", rec["concept"])
    st.write(rec["reason"])
    st.json(rec)

with tab_status:
    st.subheader("系统状态")
    st.write(
        {
            "mode": get_mode(),
            "pages": len(pages),
            "chunks": len(chunks),
            "vector_index": data_source,
            "fallback_ready": True,
        }
    )
