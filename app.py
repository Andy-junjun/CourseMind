from pathlib import Path
import html as html_lib

import streamlit as st
import streamlit.components.v1 as components

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


def graph_visualization_html(seeds, rows, max_seed_nodes=5, max_expanded_nodes=8, max_edges=14):
    if not rows:
        return "", 0

    seed_ids = [item.chunk.chunk_id for item in seeds[:max_seed_nodes]]
    seed_id_set = set(seed_ids)
    sorted_rows = sorted(rows, key=lambda row: row["分数"], reverse=True)

    expanded_ids = []
    edges = []
    for row in sorted_rows:
        if row["种子chunk"] not in seed_id_set:
            continue
        expanded_id = row["扩展chunk"]
        if expanded_id not in expanded_ids:
            if len(expanded_ids) >= max_expanded_nodes:
                continue
            expanded_ids.append(expanded_id)
        edges.append(row)
        if len(edges) >= max_edges:
            break

    if not edges:
        return "", 0

    seed_y = {chunk_id: 86 + index * 92 for index, chunk_id in enumerate(seed_ids)}
    expanded_y = {
        chunk_id: 86 + index * 92 for index, chunk_id in enumerate(expanded_ids)
    }
    height = max(360, 130 + 92 * max(len(seed_ids), len(expanded_ids)))
    width = 1120
    seed_x = 170
    expanded_x = 840
    line_start_x = 320
    line_end_x = 690

    relation_color = {
        "same_page": "#2563eb",
        "same_chapter": "#7c3aed",
        "adjacent": "#0891b2",
        "shared_concept": "#16a34a",
    }

    seed_cards = []
    seed_lookup = {item.chunk.chunk_id: item.chunk for item in seeds}
    for chunk_id in seed_ids:
        chunk = seed_lookup[chunk_id]
        seed_cards.append(
            svg_node(
                seed_x,
                seed_y[chunk_id],
                "种子",
                short_label(chunk_id),
                f"{chunk.file_name} p.{chunk.page}",
                "#e0f2fe",
                "#0369a1",
            )
        )

    expanded_cards = []
    expanded_meta = {row["扩展chunk"]: row for row in rows}
    for chunk_id in expanded_ids:
        row = expanded_meta[chunk_id]
        expanded_cards.append(
            svg_node(
                expanded_x,
                expanded_y[chunk_id],
                "扩展",
                short_label(chunk_id),
                f"{row['文件']} p.{row['页码']}",
                "#dcfce7",
                "#15803d",
            )
        )

    edge_lines = []
    for row in edges:
        source_y = seed_y[row["种子chunk"]]
        target_y = expanded_y[row["扩展chunk"]]
        color = relation_color.get(row["关系"], "#64748b")
        mid_x = (line_start_x + line_end_x) / 2
        mid_y = (source_y + target_y) / 2
        edge_lines.append(
            f"""
            <path d="M {line_start_x} {source_y} C 460 {source_y}, 550 {target_y}, {line_end_x} {target_y}"
                  fill="none" stroke="{color}" stroke-width="2.4" opacity="0.72" marker-end="url(#arrow)" />
            <rect x="{mid_x - 58}" y="{mid_y - 15}" width="116" height="30" rx="6"
                  fill="white" stroke="{color}" stroke-width="1" opacity="0.96" />
            <text x="{mid_x}" y="{mid_y + 5}" text-anchor="middle" class="edge-label" fill="{color}">
              {html_lib.escape(row["关系"])}
            </text>
            """
        )

    concept_chips = concept_chip_html(edges)
    svg = f"""
    <div class="graph-wrap">
      <div class="graph-title">GraphRAG 节点关系可视化</div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
          </marker>
        </defs>
        <text x="{seed_x}" y="34" text-anchor="middle" class="column-title">原始检索种子 chunk</text>
        <text x="{expanded_x}" y="34" text-anchor="middle" class="column-title">GraphRAG 扩展 chunk</text>
        {''.join(edge_lines)}
        {''.join(seed_cards)}
        {''.join(expanded_cards)}
      </svg>
      {concept_chips}
      <div class="legend">
        <span><b style="color:#2563eb">same_page</b> 同页</span>
        <span><b style="color:#7c3aed">same_chapter</b> 同章节</span>
        <span><b style="color:#0891b2">adjacent</b> 相邻片段</span>
        <span><b style="color:#16a34a">shared_concept</b> 共享知识点</span>
      </div>
    </div>
    <style>
      .graph-wrap {{
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        background: #f8fafc;
        padding: 12px 14px 10px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .graph-title {{
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 6px;
      }}
      .column-title {{
        font-size: 14px;
        font-weight: 700;
        fill: #334155;
      }}
      .node-title {{
        font-size: 12px;
        font-weight: 700;
      }}
      .node-label {{
        font-size: 11px;
        fill: #0f172a;
      }}
      .node-meta {{
        font-size: 10px;
        fill: #475569;
      }}
      .edge-label {{
        font-size: 11px;
        font-weight: 700;
      }}
      .concepts {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 2px 4px;
      }}
      .concept-chip {{
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
        color: #166534;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
      }}
      .legend {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        color: #475569;
        font-size: 12px;
        margin-top: 8px;
      }}
    </style>
    """
    return svg, height + 130


def svg_node(x, y, badge, title, meta, fill, stroke):
    return f"""
    <g>
      <rect x="{x - 135}" y="{y - 34}" width="270" height="68" rx="8"
            fill="{fill}" stroke="{stroke}" stroke-width="1.6" />
      <text x="{x - 118}" y="{y - 13}" class="node-title" fill="{stroke}">
        {html_lib.escape(badge)}
      </text>
      <text x="{x - 118}" y="{y + 7}" class="node-label">
        {html_lib.escape(title)}
      </text>
      <text x="{x - 118}" y="{y + 25}" class="node-meta">
        {html_lib.escape(short_label(meta, 30))}
      </text>
    </g>
    """


def concept_chip_html(edges):
    concepts = []
    for row in edges:
        reason = row["原因"]
        marker = "共享知识点："
        if marker not in reason:
            continue
        for concept in reason.split(marker, 1)[1].split("、"):
            concept = concept.strip()
            if concept and concept not in concepts:
                concepts.append(concept)
    if not concepts:
        return ""
    chips = "".join(
        f'<span class="concept-chip">concept: {html_lib.escape(short_label(concept, 18))}</span>'
        for concept in concepts[:10]
    )
    return f'<div class="concepts">{chips}</div>'


def short_label(value, max_length=24):
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


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
            graph_html, graph_height = graph_visualization_html(retrieved, graph_explanations)
            if graph_html:
                st.subheader("GraphRAG 节点关系图")
                components.html(graph_html, height=graph_height, scrolling=True)
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
