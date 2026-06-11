from pathlib import Path

import streamlit as st

from src.answer_guard import should_refuse
from src.bandit_recommender import recommend_concept, update_feedback
from src.chunker import chunk_pages
from src.config import get_mode
from src.document_loader import load_pdf
from src.generator import answer_question, summarize_document
from src.graph_store import expand_with_graph
from src.miniranker import rerank
from src.retriever import retrieve
from src.study_tools import generate_quiz


st.set_page_config(page_title="CourseMind", layout="wide")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def default_pages():
    sample = RAW_DIR / "sample_course.txt"
    if not sample.exists():
        sample.write_text(
            "CourseMind uses PDF parsing, chunking, RAG retrieval, MiniRanker, "
            "GraphRAG-lite, and Bandit review recommendation. Transformer "
            "embeddings map course chunks and user questions into a semantic "
            "space. MiniRanker combines dense score, BM25 score, graph score, "
            "same-page bonus, same-chapter bonus, and chunk length features.",
            encoding="utf-8",
        )
    return load_pdf(str(sample))


@st.cache_data(show_spinner=False)
def build_knowledge_base(file_path: str | None):
    pages = load_pdf(file_path) if file_path else default_pages()
    chunks = chunk_pages(pages)
    return pages, chunks


st.sidebar.title("CourseMind")
st.sidebar.caption(f"Mode: {get_mode()}")
uploaded = st.sidebar.file_uploader("Upload course PDF or text", type=["pdf", "txt", "md"])

file_path = None
if uploaded:
    target = RAW_DIR / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    file_path = str(target)

pages, chunks = build_knowledge_base(file_path)
st.sidebar.metric("Pages", len(pages))
st.sidebar.metric("Chunks", len(chunks))

tab_qa, tab_summary, tab_quiz, tab_review, tab_status = st.tabs(
    ["Q&A", "Summary", "Quiz", "Review", "Status"]
)

with tab_qa:
    query = st.text_input("Question", "What technologies does CourseMind use?")
    top_k = st.slider("Top-K evidence", 1, 10, 5)
    if st.button("Ask", type="primary"):
        retrieved = retrieve(query, chunks, top_k=top_k)
        expanded = expand_with_graph(retrieved, chunks, hops=1)
        ranked = rerank(query, expanded, top_k=top_k)
        if should_refuse(query, ranked):
            st.warning("No sufficiently relevant evidence was found in the current knowledge base.")
        else:
            result = answer_question(query, ranked)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Citations")
            st.dataframe(result["citations"], use_container_width=True)
        st.subheader("Retrieval Visualization")
        st.dataframe(
            [
                {
                    "rank": i + 1,
                    "chunk_id": item.chunk.chunk_id,
                    "page": item.chunk.page,
                    "dense": round(item.dense_score, 3),
                    "bm25": round(item.bm25_score, 3),
                    "graph": round(item.graph_score, 3),
                    "ranker": round(item.ranker_score, 3),
                    "text": item.chunk.text[:120],
                }
                for i, item in enumerate(ranked)
            ],
            use_container_width=True,
        )

with tab_summary:
    if st.button("Generate Summary"):
        st.write(summarize_document(chunks))

with tab_quiz:
    num_questions = st.slider("Questions", 1, 5, 3)
    quiz_items = generate_quiz(chunks, num_questions=num_questions)
    for idx, item in enumerate(quiz_items, start=1):
        st.markdown(f"**Q{idx}. {item.question}**")
        choice = st.radio("Choose one", item.options, key=f"quiz_{idx}")
        correct = choice == item.answer
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit", key=f"submit_{idx}"):
                update_feedback(item.concept, correct)
                st.success("Correct" if correct else f"Incorrect. Answer: {item.answer}")
        with col2:
            st.caption(f"Concept: {item.concept} | Source: {item.source_chunk_id}")
        st.write(item.explanation)

with tab_review:
    rec = recommend_concept()
    st.metric("Recommended concept", rec["concept"])
    st.write(rec["reason"])
    st.json(rec)

with tab_status:
    st.subheader("System Status")
    st.write(
        {
            "mode": get_mode(),
            "pages": len(pages),
            "chunks": len(chunks),
            "fallback_ready": True,
        }
    )

